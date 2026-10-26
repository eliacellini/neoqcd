"""Standalone NE-MCMC and SNF protocol runner.

This entry point deliberately does not use parallel tempering.  A global batch
is thermalised at the prior parameter and is then evolved through a linear
parameter protocol.  The only state which is cached is that initial batch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
import numpy as np
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from neoqcd.flow import FlowPars
from neoqcd.analysis import OBSERVABLES_HEADER, analyze_observables_csv
from neoqcd.mcmc import HBOR, NEMCMC_update
from neoqcd.nn import HyperTimeScalar, time_embedding
import neoqcd.sun_utils as sun
from neoqcd.smearing import (
    CouplingLayer,
    DefectCouplingLayer,
    HYPER_CLASS_INACTIVE_RHO,
    ResidualCouplingLayer,
    ResidualNormalizingFlows,
)
from neoqcd.theory import Wilson_action
from neoqcd.utils import Defect, create_around_defect_mask, create_mask


# Kept separate from model/protocol RNG streams.  The rank term deliberately
# gives each thermalising worker a distinct stream without putting world size
# in the prior-cache compatibility key.
THERMAL_RNG_VERSION = 2
THERMAL_SEED_OFFSET = 104729


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone linear NE-MCMC/SNF protocol (no parallel tempering).",
        allow_abbrev=False,
    )
    parser.add_argument("--algorithm", choices=("nemcmc", "snf"), default="snf")
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--domain", choices=("beta", "defect"), default="beta")
    parser.add_argument("--cpu", action="store_true", help="Use CPU and the Gloo backend.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("auto", "gloo", "nccl"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)

    # Lattice and protocol.
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size.")
    parser.add_argument("--beta", type=float, default=None, help="Fixed beta for defect studies (default: 6.0).")
    parser.add_argument("--beta-initial", type=float, default=None)
    parser.add_argument("--beta-final", type=float, default=None)
    parser.add_argument("--bc-initial", type=float, default=None)
    parser.add_argument("--bc-final", type=float, default=None)
    parser.add_argument(
        "--protocol-steps", "--K", dest="protocol_steps", type=int, default=4,
        help="Number K of linear protocol transitions.",
    )
    parser.add_argument("--protocol-file", type=str, default=None, help="Text file with K target couplings; the initial coupling comes from beta-initial/bc-initial.")
    parser.add_argument("--thermal-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50, help="Training epochs (or evaluation trajectories).")
    parser.add_argument("--training-mode", choices=("full", "progressive", "blockwise"), default="blockwise")
    parser.add_argument("--epochs-per-block", type=int, default=None, help="Progressive training epochs for each protocol block (defaults to --train-steps/--steps).")
    parser.add_argument("--grad-acc", type=int, default=1, help="Prior samples accumulated per training epoch.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save resumable state every N epochs/trajectories (0: final only).")
    parser.add_argument("--resume", action="store_true", help="Resume model, optimizer, prior chain, RNG and outputs from output-dir.")
    parser.add_argument("--load-prior", "--continue-prior", dest="load_prior", action="store_true", help="Continue the prior chain and RNG from --checkpoint while starting a new evaluation output.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Load model weights from --checkpoint but start a fresh optimizer and scheduler.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--prior-mcmc-steps", type=int, default=1,
        help="MCMC updates of the persistent local prior chain before each epoch/trajectory.",
    )
    parser.add_argument("--train-steps", "--train-epochs", dest="train_steps", type=int, default=None, help="Explicit training epoch count overriding --steps.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Evaluation runs; defaults to --steps.")
    parser.add_argument(
        "--eval-samples", type=int, default=None,
        help="Requested global evaluation samples; generates ceil(N/global-batch)*global-batch trajectories.",
    )
    parser.add_argument("--updates-per-layer", type=int, default=1)
    parser.add_argument("--orsteps", type=int, default=4)
    parser.add_argument("--hierarchical-levels", "--hierarchical_levels", dest="hierarchical_levels", type=int, default=0)
    parser.add_argument("--hierarchical-updates", "--hierarchical_updates", dest="hierarchical_updates", nargs="+", type=int, default=[])
    parser.add_argument("--hierarchical-buffers", "--hierarchical_buffers", dest="hierarchical_buffers", nargs="+", type=int, default=None)

    # Defect geometry.
    parser.add_argument("--defect-size", type=int, default=2)
    parser.add_argument("--time-slice", type=int, default=2)
    parser.add_argument("--space-slice", type=int, default=2)

    # NF architecture and its existing conditional-network controls.
    parser.add_argument(
        "--nf-architecture", "--nf-layer-type", dest="nf_architecture",
        choices=("smearing", "hyper-smearing", "residual", "hyper-residual", "cnf"),
        default=None,
        help=(
            "SNF layer: residual=static trainable coefficients; "
            "hyper-residual=beta-conditioned residual MLP; smearing variants use CouplingLayer."
        ),
    )
    parser.add_argument("--cnf-checkpoint", help="Field checkpoint trained by main_cnf_rex.py")
    parser.add_argument("--cnf-integration-steps", type=int, default=4)
    parser.add_argument("--hyper-smearing-mode", choices=("shared", "per_link", "class"), default="shared")
    parser.add_argument("--hyper-time-embedding-dim", type=int, default=8)
    parser.add_argument("--hyper-hidden-dim", type=int, default=16)
    parser.add_argument("--hyper-depth", type=int, default=2)
    parser.add_argument("--hyper-activation", choices=("silu", "gelu", "tanh", "relu"), default="silu")
    parser.add_argument("--hyper-rho-init", type=float, default=1e-3)
    parser.add_argument("--hyper-rho-eps", type=float, default=0.0)
    parser.add_argument("--hyper-rho-max", type=float, default=0.0)
    parser.add_argument("--hyper-normalize-by-nstep", action="store_true", default=True)
    parser.add_argument("--hyper-no-normalize-by-nstep", dest="hyper_normalize_by_nstep", action="store_false")
    parser.add_argument("--hyper-scale-by-delta", action="store_true", default=True)
    parser.add_argument("--hyper-no-scale-by-delta", dest="hyper_scale_by_delta", action="store_false")
    parser.add_argument("--residual-include-imag", action="store_true", default=True)
    parser.add_argument("--residual-no-include-imag", dest="residual_include_imag", action="store_false")
    parser.add_argument("--residual-quadratic", action="store_true", default=True)
    parser.add_argument("--residual-no-quadratic", dest="residual_quadratic", action="store_false")
    parser.add_argument("--residual-coeff-init", type=float, default=1e-3)
    parser.add_argument("--residual-coeff-max", type=float, default=0.0)
    parser.add_argument(
        "--residual-n-pt", type=int, default=0,
        help="Gauge-covariant convolutions of frozen links per residual substep (0: original plaquettes).",
    )
    parser.add_argument(
        "--residual-mask-mod", type=int, choices=(2, 4), default=2,
        help="Residual-only coloring; MCMC keeps its checkerboard mask.",
    )
    parser.add_argument(
        "--smearing-rho-init", type=float, default=1e-3,
        help="Strictly positive initial rho for non-hyper standard smearing.",
    )
    parser.add_argument("--rho-functional", action="store_true", help="Use a shared period-2 Fourier rho(t) MLP across all flow steps.")
    parser.add_argument("--rho-functional-modes", type=int, default=9, help="Odd Fourier embedding width (1 + 2 times frequency count).")
    parser.add_argument("--rho-functional-hidden-dim", type=int, default=16)
    parser.add_argument("--rho-functional-scale", type=float, default=1e-3)
    parser.add_argument(
        "--explicit-backward", action="store_true",
        help="Use the hand-derived stout-smearing backward for SU(3) smearing layers.",
    )
    parser.add_argument("--prior-hot-start", action="store_true", help="Thermalize from exact Haar-random links instead of unit links.")
    parser.add_argument("--flow-lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument(
        "--max-consecutive-nan-grad-updates", type=int, default=5,
        help="Stop after this many consecutive SNF optimizer updates with missing or non-finite gradients.",
    )

    # Initial-configuration cache.
    parser.add_argument("--cfg-cache", choices=("auto", "read", "refresh", "off"), default="auto")
    parser.add_argument("--cfg-cache-dir", type=str, default="data/cfgs_nemcmc_snf")
    parser.add_argument("--cfg-cache-tag", type=str, default="")

    # Output and logging.
    parser.add_argument("--main-dir", type=str, default=os.getcwd())
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--transfer-rho-functional", action="store_true", help="Transfer only a trained functional rho to a different protocol length; training requires --reset-optimizer.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--int-meas", type=int, default=1, help="Record observables this many times along the protocol; must divide protocol-steps.")
    parser.add_argument(
        "--wandb", nargs="?", const="online", choices=("disabled", "offline", "online"), default=None,
        help="Enable W&B (optionally with the legacy value offline/online).",
    )
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default=None)
    parser.add_argument("--wandb-project", type=str, default="neo-nemcmc-snf")
    parser.add_argument("--wandb-entity", type=str, default="lqft-snf")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args(argv)
    if args.domain == "beta":
        args.beta_initial = 5.8 if args.beta_initial is None else args.beta_initial
        args.beta_final = 6.0 if args.beta_final is None else args.beta_final
    if args.nf_architecture is None:
        args.nf_architecture = "smearing" if args.rho_functional else ("hyper-smearing" if args.algorithm == "snf" else "smearing")
    args.wandb_mode = args.wandb_mode or args.wandb or os.environ.get("WANDB_MODE", "disabled")
    if args.wandb_mode not in {"disabled", "offline", "online"}:
        parser.error("WANDB_MODE must be disabled, offline, or online")
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.algorithm == "nemcmc" and args.checkpoint and not args.load_prior:
        raise ValueError("NEMCMC --checkpoint requires --load-prior")
    if args.algorithm == "nemcmc" and args.rho_functional:
        raise ValueError("--rho-functional requires --algorithm snf")
    if args.algorithm == "snf" and args.phase == "evaluate" and not args.checkpoint and not args.resume and not args.cnf_checkpoint:
        raise ValueError("SNF evaluation requires --checkpoint")
    if args.phase == "evaluate" and args.train_steps is not None:
        raise ValueError("--train-steps is valid only in train phase")
    if args.phase == "train" and args.eval_steps is not None:
        raise ValueError("--eval-steps is valid only in evaluate phase")
    if args.phase == "train" and args.eval_samples is not None:
        raise ValueError("--eval-samples is valid only in evaluate phase")
    if args.algorithm == "nemcmc" and args.phase == "train":
        raise ValueError("--algorithm nemcmc supports only --phase evaluate")
    if args.phase == "evaluate" and args.reset_optimizer:
        raise ValueError("--reset-optimizer is valid only in train phase")
    if args.resume and args.checkpoint:
        raise ValueError("--resume uses output-dir/snf_checkpoint.pt; omit --checkpoint")
    if args.resume and args.load_prior:
        raise ValueError("--resume and --load-prior are mutually exclusive")
    if args.load_prior and not args.checkpoint:
        raise ValueError("--load-prior requires --checkpoint")
    if args.load_prior and args.phase != "evaluate":
        raise ValueError("--load-prior is valid only for evaluation")
    if args.load_prior and args.reset_optimizer:
        raise ValueError("--load-prior cannot be combined with --reset-optimizer")
    if args.resume and args.overwrite:
        raise ValueError("--resume cannot be combined with --overwrite")
    if args.resume and args.reset_optimizer:
        raise ValueError("--reset-optimizer cannot be used with exact --resume")
    if args.resume and not args.output_dir:
        raise ValueError("--resume requires an explicit --output-dir")
    if args.reset_optimizer and not args.checkpoint:
        raise ValueError("--reset-optimizer requires --checkpoint")
    if args.transfer_rho_functional and not args.checkpoint:
        raise ValueError("--transfer-rho-functional requires --checkpoint")
    if args.transfer_rho_functional and not args.rho_functional:
        raise ValueError("--transfer-rho-functional requires --rho-functional")
    if args.transfer_rho_functional and args.phase == "train" and not args.reset_optimizer:
        raise ValueError("functional-rho transfer during training requires --reset-optimizer")
    if args.eval_samples is not None and args.eval_samples < 1:
        raise ValueError("--eval-samples must be >= 1")
    if args.eval_samples is not None and args.eval_steps is not None:
        raise ValueError("--eval-samples and --eval-steps are mutually exclusive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive and is global across ranks")
    if args.protocol_steps < 1 or args.grad_acc < 1:
        raise ValueError("protocol-steps and grad-acc must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError("--scheduler-factor must be in (0, 1)")
    if args.scheduler_patience < 0 or args.scheduler_min_lr < 0.0 or not math.isfinite(args.scheduler_min_lr):
        raise ValueError("scheduler patience and minimum learning rate must be non-negative")
    if args.int_meas < 1 or args.protocol_steps % args.int_meas:
        raise ValueError("--int-meas must be a positive divisor of --protocol-steps")
    if args.rho_functional_modes < 1 or args.rho_functional_modes % 2 != 1:
        raise ValueError("--rho-functional-modes must be a positive odd integer")
    if args.rho_functional_hidden_dim < 1:
        raise ValueError("--rho-functional-hidden-dim must be positive")
    if not math.isfinite(args.rho_functional_scale) or args.rho_functional_scale <= 0.0:
        raise ValueError("--rho-functional-scale must be finite and positive")
    if args.thermal_steps < 0 or args.steps < 1 or (args.eval_steps is not None and args.eval_steps < 1):
        raise ValueError("thermal steps must be >= 0 and run steps must be >= 1")
    if args.train_steps is not None and args.train_steps < 1:
        raise ValueError("--train-steps must be >= 1")
    if args.epochs_per_block is not None and args.epochs_per_block < 1:
        raise ValueError("--epochs-per-block must be >= 1")
    if args.epochs_per_block is not None and (args.phase != "train" or args.training_mode != "progressive"):
        raise ValueError("--epochs-per-block is valid only for progressive training")
    if args.updates_per_layer < 0 or args.orsteps < 0:
        raise ValueError("MCMC update counts must be non-negative")
    if args.hierarchical_levels < 0:
        raise ValueError("hierarchical-levels must be non-negative")
    if len(args.hierarchical_updates) != args.hierarchical_levels:
        raise ValueError("hierarchical-updates needs one count per hierarchical level")
    if any(count < 0 for count in args.hierarchical_updates):
        raise ValueError("hierarchical update counts must be non-negative")
    if args.hierarchical_levels == 0:
        if args.hierarchical_buffers is not None:
            raise ValueError("hierarchical-buffers requires hierarchical-levels > 0")
    else:
        if args.domain != "defect":
            raise ValueError("hierarchical updates require --domain defect")
        if args.hierarchical_buffers is None:
            args.hierarchical_buffers = list(range(args.hierarchical_levels + 1, 1, -1))
        if len(args.hierarchical_buffers) != args.hierarchical_levels:
            raise ValueError("hierarchical-buffers needs one positive buffer per level")
        if any(buffer <= 0 for buffer in args.hierarchical_buffers):
            raise ValueError("hierarchical buffers must be positive")
        if any(outer <= inner for outer, inner in zip(args.hierarchical_buffers, args.hierarchical_buffers[1:])):
            raise ValueError("hierarchical-buffers must decrease from largest to smallest")
        largest_buffer = args.hierarchical_buffers[0]
        if (
            args.time_slice < largest_buffer
            or args.T - args.time_slice < largest_buffer
            or args.space_slice < largest_buffer
            or args.L - args.space_slice - args.defect_size < largest_buffer
        ):
            raise ValueError("hierarchical boxes must fit inside the lattice around the defect")
    if args.prior_mcmc_steps < 0:
        raise ValueError("prior-mcmc-steps must be non-negative")
    if args.N != 3:
        raise ValueError("The existing SU(3) flow implementations require --N 3")
    if args.D < 2 or args.D > 4 or args.T < 2 or args.L < 2:
        raise ValueError("The existing lattice helpers support 2 <= D <= 4, with T,L >= 2")
    if args.defect_size < 1:
        raise ValueError("defect-size must be positive")
    if args.log_every < 1:
        raise ValueError("log-every must be >= 1")
    if args.flow_lr <= 0.0 or not math.isfinite(args.flow_lr):
        raise ValueError("flow-lr must be finite and positive")
    if args.grad_clip_norm < 0.0 or not math.isfinite(args.grad_clip_norm):
        raise ValueError("grad-clip-norm must be finite and non-negative")
    if args.max_consecutive_nan_grad_updates < 1:
        raise ValueError("max-consecutive-nan-grad-updates must be positive")
    if not math.isfinite(args.smearing_rho_init) or args.smearing_rho_init <= 0.0:
        raise ValueError("smearing-rho-init must be finite and strictly positive")
    if args.domain == "beta":
        if args.beta is not None:
            raise ValueError("--beta is valid only for --domain defect; use beta-initial/final")
        if args.bc_initial is not None or args.bc_final is not None:
            raise ValueError("bc-initial/bc-final are valid only for --domain defect")
        endpoints = (args.beta_initial, args.beta_final)
        if any(not math.isfinite(value) or value <= 0.0 for value in endpoints):
            raise ValueError("beta endpoints must be finite and strictly positive")
        if endpoints[0] == endpoints[1]:
            raise ValueError("beta-initial and beta-final must be distinct")
    else:
        if args.beta_initial is not None or args.beta_final is not None:
            raise ValueError("beta-initial/beta-final are valid only for --domain beta")
        beta = 6.0 if args.beta is None else args.beta
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("defect beta must be finite and strictly positive")
        bc_initial = 0.0 if args.bc_initial is None else args.bc_initial
        bc_final = 1.0 if args.bc_final is None else args.bc_final
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (bc_initial, bc_final)):
            raise ValueError("bc endpoints must be finite and lie in [0, 1]")
        if bc_initial == bc_final:
            raise ValueError("bc-initial and bc-final must be distinct")
    if args.domain == "defect" and args.D != 4:
        raise ValueError("The defect patch cutting/embedding adapter is defined for D=4")
    if args.domain == "defect" and args.nf_architecture != "cnf":
        if args.T < 4 or args.L < args.defect_size + 4:
            raise ValueError("defect domain requires T >= 4 and L >= defect-size + 4")
        if not 2 <= args.time_slice <= args.T - 2:
            raise ValueError("time-slice must satisfy 2 <= time-slice <= T-2")
        if not 2 <= args.space_slice <= args.L - args.defect_size - 2:
            raise ValueError("space-slice must leave a two-site buffer around the defect")
    if args.algorithm == "nemcmc" and args.nf_architecture != "smearing":
        raise ValueError("NF architecture options are valid only for --algorithm snf")
    if args.nf_architecture == "cnf":
        if not args.cnf_checkpoint or args.cnf_integration_steps < 1:
            raise ValueError("cnf requires --cnf-checkpoint and positive --cnf-integration-steps")
        if args.phase != "evaluate":
            raise ValueError("Train CNF fields with main/main_cnf_rex.py (REX Path Gradient)")
        if args.domain == "defect" and not (0 <= args.time_slice < args.T and 0 <= args.space_slice <= args.L - args.defect_size):
            raise ValueError("CNF defect must fit inside the lattice")
    elif args.cnf_checkpoint:
        raise ValueError("--cnf-checkpoint requires --nf-architecture cnf")
    if args.rho_functional and args.nf_architecture != "smearing":
        raise ValueError("--rho-functional requires the classic --nf-architecture smearing")
    if args.phase == "train" and args.training_mode != "full" and args.rho_functional:
        raise ValueError("--rho-functional training requires --training-mode full")
    if args.residual_n_pt < 0:
        raise ValueError("residual-n-pt must be non-negative")
    if args.residual_n_pt or args.residual_mask_mod != 2:
        if args.algorithm != "snf" or args.nf_architecture not in {"residual", "hyper-residual"}:
            raise ValueError("residual-n-pt and residual-mask-mod require a residual SNF architecture")
        if args.domain != "beta":
            raise ValueError("Extended residual layers require the full lattice beta domain; defect crops have a fixed halo")
        if args.T % args.residual_mask_mod or args.L % args.residual_mask_mod:
            raise ValueError("T and L must be multiples of residual-mask-mod for periodic coloring")
    if args.hyper_smearing_mode == "class" and (args.domain != "defect" or args.nf_architecture != "hyper-smearing"):
        raise ValueError("hyper-smearing-mode=class is supported only for defect hyper-smearing")
    if args.nf_architecture != "hyper-smearing" and args.hyper_smearing_mode != "shared":
        raise ValueError("hyper-smearing-mode is meaningful only with --nf-architecture hyper-smearing")
    if args.explicit_backward and (args.algorithm != "snf" or args.nf_architecture not in {"smearing", "hyper-smearing"}):
        raise ValueError("explicit-backward requires a smearing SNF architecture")
    if (
        args.algorithm == "snf"
        and args.nf_architecture in {"smearing", "hyper-smearing"}
        and args.D < 4
    ):
        raise ValueError(
            "SNF smearing and hyper-smearing use the existing D=4 plaquette-index implementation; "
            "use D=4 or choose residual for D<4"
        )
    if args.phase == "train" and args.algorithm == "snf" and args.beta_initial == args.beta_final and args.domain == "beta":
        raise ValueError("SNF training requires a non-zero beta protocol")
    if args.phase == "train" and args.training_mode == "progressive" and _nf_layer_sharing(args) == "shared":
        raise ValueError("progressive training requires per-block NF parameters")


def _protocol_endpoints(args: argparse.Namespace) -> tuple[float, float, float, str]:
    if args.domain == "beta":
        return float(args.beta_initial), float(args.beta_final), float(args.beta_initial), "beta"
    start = 0.0 if args.bc_initial is None else args.bc_initial
    end = 1.0 if args.bc_final is None else args.bc_final
    fixed_beta = 6.0 if args.beta is None else args.beta
    return float(start), float(end), float(fixed_beta), "bc"


def _protocol_values(args: argparse.Namespace, start: float, end: float) -> list[float]:
    if args.protocol_file is None:
        return [start + (end - start) * (i / args.protocol_steps) for i in range(args.protocol_steps + 1)]
    try:
        targets = np.atleast_1d(np.loadtxt(args.protocol_file, dtype=np.float64))
    except Exception as exc:
        raise ValueError(f"cannot read protocol file {args.protocol_file}: {exc}") from exc
    if targets.ndim != 1 or targets.size != args.protocol_steps:
        raise ValueError(
            f"protocol file must contain exactly {args.protocol_steps} target couplings; got shape {targets.shape}"
        )
    if not np.isfinite(targets).all():
        raise ValueError("protocol file contains non-finite couplings")
    if not math.isclose(float(targets[-1]), float(end), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"last protocol coupling {targets[-1]} must equal target {end}")
    return [float(start), *(float(value) for value in targets)]


def _choose_backend_device(args: argparse.Namespace) -> tuple[str, torch.device, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    want_cpu = bool(args.cpu or args.device == "cpu")
    if want_cpu:
        if args.backend == "nccl":
            raise ValueError("NCCL cannot be selected together with --cpu")
        return "gloo", torch.device("cpu"), local_rank
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --cpu")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK {local_rank} is outside the CUDA device range")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        if args.backend == "gloo":
            raise ValueError("GPU runs use NCCL; use --cpu for Gloo")
        return "nccl", device, local_rank
    if args.backend == "nccl":
        raise RuntimeError("NCCL requires CUDA")
    return "gloo", torch.device("cpu"), local_rank


def _init_process_group(backend: str) -> tuple[int, int, bool]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 or "RANK" in os.environ:
        dist.init_process_group(backend=backend)
    else:
        with tempfile.NamedTemporaryFile(delete=False) as init_file:
            init_method = f"file://{init_file.name}"
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    return dist.get_rank(), dist.get_world_size(), True


def setup_runtime(args: argparse.Namespace) -> tuple[int, int, torch.device, str, int, bool]:
    backend, device, local_rank = _choose_backend_device(args)
    rank, world_size, owns_process_group = _init_process_group(backend)
    if args.batch_size % world_size:
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        raise ValueError(
            f"Global --batch-size={args.batch_size} must be divisible by world size {world_size}"
        )
    seed = int(args.seed) + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return rank, world_size, device, backend, local_rank, owns_process_group


def _capture_rng_state(device: torch.device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def _thermal_seed(args: argparse.Namespace, rank: int) -> int:
    return (int(args.seed) + THERMAL_SEED_OFFSET + int(rank)) % (2**63 - 1)


def _seed_thermal_rng(device: torch.device, seed: int) -> None:
    torch.random.default_generator.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _slug(value) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def _cache_root(args: argparse.Namespace) -> Path:
    root = Path(args.cfg_cache_dir)
    return root if root.is_absolute() else Path(args.main_dir) / root


def cache_key(args: argparse.Namespace, start: float, end: float, parameter_name: str) -> dict:
    """Return the compatibility contract for the prior only.

    ``end`` is intentionally not part of this contract: changing the target
    protocol must reuse the same thermalised source ensemble.
    """
    return {
        "schema": 6,
        "D": int(args.D), "T": int(args.T), "L": int(args.L), "N": int(args.N),
        "global_batch": int(args.batch_size), "dtype": "torch.cdouble",
        "domain": str(args.domain), "parameter_name": parameter_name,
        "prior_parameter": float(start),
        "beta": 6.0 if args.beta is None and args.domain == "defect" else (None if args.beta is None else float(args.beta)),
        "defect_size": int(args.defect_size), "time_slice": int(args.time_slice),
        "space_slice": int(args.space_slice), "seed": int(args.seed),
        "orsteps": int(args.orsteps),
        "thermal_rng_version": THERMAL_RNG_VERSION,
        "thermal_seed_offset": THERMAL_SEED_OFFSET,
        "prior_hot_start": bool(args.prior_hot_start),
        "hierarchical_levels": int(args.hierarchical_levels),
        "hierarchical_updates": [int(value) for value in args.hierarchical_updates],
        "hierarchical_buffers": [int(value) for value in (args.hierarchical_buffers or [])],
        "tag": str(args.cfg_cache_tag or ""),
    }


def cache_path(args: argparse.Namespace, key: dict) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    name = (
        f"nemcmc_snf_{key['domain']}_D{key['D']}_T{key['T']}_L{key['L']}_"
        f"bs{key['global_batch']}_prior{_slug(key['prior_parameter'])}_{digest}"
    )
    if key["tag"]:
        name = f"{_slug(key['tag'])}_{name}"
    return _cache_root(args) / name


def _atomic_torch_save(payload, path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_json_save(payload: dict, path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _prepare_csv(path: Path, header: list[str], resume: bool, keep_rows: int = 0,
                 legacy_header=None, legacy_skipped_update="") -> int:
    if not resume:
        tmp_path = path.with_name(f".{path.name}.init.tmp")
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(header)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return 0
    if not path.is_file():
        raise FileNotFoundError(f"resume output is missing: {path}")
    tmp_path = path.with_name(f".{path.name}.resume.tmp")
    rows = 0
    try:
        with open(path, newline="", encoding="utf-8") as source, open(tmp_path, "w", newline="", encoding="utf-8") as target:
            reader = csv.reader(source)
            writer = csv.writer(target)
            saved_header = next(reader, None)
            legacy = legacy_header is not None and saved_header == legacy_header
            if saved_header != header and not legacy:
                raise ValueError(f"CSV header does not match this run: {path}")
            writer.writerow(header)
            for row in reader:
                if rows < keep_rows:
                    if legacy:
                        if len(row) != len(legacy_header):
                            raise ValueError(f"legacy CSV row has an invalid number of columns: {path}")
                        row = row[:9] + ["[]", "[]", "[]", legacy_skipped_update, row[10]]
                    writer.writerow(row)
                    rows += 1
            if rows < keep_rows:
                raise ValueError(f"{path} has {rows} rows but checkpoint requires {keep_rows}")
        os.replace(tmp_path, path)
        return rows
    finally:
        tmp_path.unlink(missing_ok=True)


def _validate_resume_csv(path: Path, header: list[str], keep_rows: int, legacy_header=None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"resume output is missing: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        saved_header = next(reader, None)
        if saved_header != header and (legacy_header is None or saved_header != legacy_header):
            raise ValueError(f"CSV header does not match this run: {path}")
        rows = sum(1 for _ in reader)
    if rows < keep_rows:
        raise ValueError(f"{path} has {rows} rows but checkpoint requires {keep_rows}")


METRICS_HEADER = [
    "epoch", "work_mean", "work_sem", "deltaF", "ess_fraction", "finite_count",
    "loss_epoch", "block_loss_json", "block_skipped_json", "optimizer_status_json",
    "optimizer_skipped_json", "optimizer_grad_streak_json", "skipped_update", "cache_status",
]
LEGACY_METRICS_HEADER = [
    "epoch", "work_mean", "work_sem", "deltaF", "ess_fraction", "finite_count",
    "loss_epoch", "block_loss_json", "block_skipped_json", "skipped_update", "cache_status",
]
def _write_observables(path: Path, trajectory: int, int_obs: torch.Tensor) -> int:
    # int_obs is gathered in rank-major order: (global sample, measure, observable).
    samples = int_obs.detach().cpu().numpy()
    batch, measures = samples.shape[0], samples.shape[1]
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for measure in range(measures):
            for sample in range(batch):
                work, heat, logj, action, plaq, plaq2, coupling = samples[sample, measure]
                writer.writerow([trajectory + 1, measure + 1, sample, coupling, work, heat, logj, action, plaq, plaq2])
    return batch * measures


def _checkpoint_rng_state(device: torch.device) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])),
        **_capture_rng_state(device),
    }


def _restore_checkpoint_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy(), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
    _restore_rng_state(state, device)


def _resume_semantics(args, checkpoint_semantics: dict, world_size: int) -> dict:
    semantics = {
        "phase": args.phase,
        "model": checkpoint_semantics,
        "world_size": int(world_size),
        "prior_mcmc_steps": int(args.prior_mcmc_steps),
        "prior_hot_start": bool(args.prior_hot_start),
        "int_meas": int(args.int_meas),
        "log_every": int(args.log_every),
        "training_mode": args.training_mode if args.phase == "train" else None,
        "epochs_per_block": (args.epochs_per_block or args.train_steps or args.steps) if args.phase == "train" and args.training_mode == "progressive" else None,
        "grad_acc": int(args.grad_acc) if args.phase == "train" else None,
        "flow_lr": float(args.flow_lr) if args.phase == "train" else None,
        "grad_clip_norm": float(args.grad_clip_norm) if args.phase == "train" else None,
        "scheduler_factor": float(args.scheduler_factor) if args.phase == "train" else None,
        "scheduler_patience": int(args.scheduler_patience) if args.phase == "train" else None,
        "scheduler_min_lr": float(args.scheduler_min_lr) if args.phase == "train" else None,
    }
    if args.phase == "train":
        semantics["max_consecutive_nan_grad_updates"] = int(args.max_consecutive_nan_grad_updates)
    return semantics


def _run_semantics_match(saved: dict, expected: dict) -> bool:
    if not isinstance(saved, dict):
        return False
    saved = dict(saved)
    # Older checkpoints had no guard; initialize them at the new default limit.
    if (
        expected.get("phase") == "train"
        and "max_consecutive_nan_grad_updates" in expected
        and "max_consecutive_nan_grad_updates" not in saved
    ):
        saved["max_consecutive_nan_grad_updates"] = 5
    return saved == expected


def _expected_optimizer_count(args) -> int:
    if args.phase != "train" or args.algorithm != "snf":
        return 0
    if args.training_mode == "full" or _nf_layer_sharing(args) == "shared":
        return 1
    return int(args.protocol_steps)


def _gradient_safety_state(args, optimizer_count: int, streaks=None, total_skipped_updates=0) -> dict:
    enabled = args.phase == "train" and args.algorithm == "snf"
    return {
        "schema": 1,
        "enabled": enabled,
        "policy": "global_nonfinite_or_rank_missing_gradient_v1",
        "max_consecutive_nan_grad_updates": int(args.max_consecutive_nan_grad_updates) if enabled else None,
        "consecutive_bad_grad_updates": list(streaks or []) if enabled else [],
        "total_skipped_updates": int(total_skipped_updates) if enabled else 0,
    }


def _resume_gradient_safety(checkpoint: dict, args, optimizer_count: int):
    if args.phase != "train" or args.algorithm != "snf":
        return [], 0
    run_state = checkpoint.get("run_state", {})
    if not isinstance(run_state, dict):
        raise ValueError("resume checkpoint has invalid run state")
    saved = run_state.get("gradient_safety")
    if saved is None:
        if args.max_consecutive_nan_grad_updates != 5:
            raise ValueError("legacy checkpoint only supports the default gradient safety limit of 5")
        return [0] * optimizer_count, 0
    if (
        not isinstance(saved, dict)
        or saved.get("schema") != 1
        or saved.get("enabled") is not True
        or saved.get("policy") != "global_nonfinite_or_rank_missing_gradient_v1"
        or saved.get("max_consecutive_nan_grad_updates") != args.max_consecutive_nan_grad_updates
    ):
        raise ValueError("resume checkpoint gradient safety settings do not match this run")
    streaks = saved.get("consecutive_bad_grad_updates")
    total = saved.get("total_skipped_updates")
    if (
        not isinstance(streaks, list)
        or len(streaks) != optimizer_count
        or any(type(value) is not int or value < 0 for value in streaks)
        or type(total) is not int
        or total < 0
    ):
        raise ValueError("resume checkpoint has invalid gradient safety counters")
    if "total_skipped_updates" in run_state and run_state["total_skipped_updates"] != total:
        raise ValueError("resume checkpoint skipped-update totals are inconsistent")
    return list(streaks), total


def _validate_resume_checkpoint(checkpoint: dict, args, local_batch: int, world_size: int,
                               device: torch.device, n_steps: int) -> None:
    progress = checkpoint.get("progress")
    if not isinstance(progress, int) or progress < 0 or progress > n_steps:
        raise ValueError(f"resume checkpoint progress must be between 0 and requested total {n_steps}")
    metric_rows = checkpoint.get("metric_rows")
    observable_rows = checkpoint.get("observable_rows")
    expected_metric_rows = progress // int(args.log_every)
    expected_observable_rows = progress * int(args.batch_size) * int(args.int_meas) if args.phase == "evaluate" else 0
    if metric_rows != expected_metric_rows or observable_rows != expected_observable_rows:
        raise ValueError("resume checkpoint output row counts do not match its progress")
    states = checkpoint.get("rank_states")
    if not isinstance(states, list) or len(states) != world_size:
        raise ValueError("resume checkpoint rank state does not match current world size")
    _resume_gradient_safety(checkpoint, args, _expected_optimizer_count(args))
    expected_shape = _expected_cfg_shape(args, local_batch) + (2,)
    for rank_state in states:
        prior = rank_state.get("prior_state") if isinstance(rank_state, dict) else None
        if not torch.is_tensor(prior) or prior.dtype != torch.float64 or tuple(prior.shape) != expected_shape:
            raise ValueError("resume checkpoint prior-chain shape/dtype does not match this run")
        if not torch.isfinite(prior).all().item():
            raise ValueError("resume checkpoint prior chain contains non-finite values")
        rng = rank_state.get("rng_state")
        if not isinstance(rng, dict) or not torch.is_tensor(rng.get("cpu")) or rng["cpu"].dtype != torch.uint8:
            raise ValueError("resume checkpoint is missing a valid CPU RNG state")
        numpy_state = rng.get("numpy")
        try:
            generator = torch.Generator(device="cpu")
            generator.set_state(rng["cpu"])
            if not isinstance(numpy_state, (tuple, list)) or len(numpy_state) != 5 or not torch.is_tensor(numpy_state[1]):
                raise ValueError("invalid NumPy RNG state")
            np.random.RandomState().set_state((numpy_state[0], numpy_state[1].cpu().numpy(), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
            random.Random().setstate(rng["python"])
        except Exception as exc:
            raise ValueError("resume checkpoint has an invalid RNG state") from exc
        if device.type == "cuda" and (
            not isinstance(rng.get("cuda"), list)
            or len(rng["cuda"]) != torch.cuda.device_count()
            or any(not torch.is_tensor(state) or state.dtype != torch.uint8 for state in rng["cuda"])
        ):
            raise ValueError("resume checkpoint is missing valid CUDA RNG states")
        works = rank_state.get("evaluation_work", [])
        if not isinstance(works, list) or any(
            not torch.is_tensor(work) or work.dtype != torch.float64 or tuple(work.shape) != (local_batch,)
            for work in works
        ):
            raise ValueError("resume checkpoint evaluation work does not match the local batch")
        expected_work_count = progress if args.phase == "evaluate" else 0
        if len(works) != expected_work_count:
            raise ValueError("resume checkpoint evaluation work does not match its progress")


def _validate_checkpoint_prior(checkpoint: dict, args, local_batch: int, world_size: int,
                               device: torch.device) -> None:
    states = checkpoint.get("rank_states")
    if not isinstance(states, list) or len(states) != world_size:
        raise ValueError("checkpoint prior chain does not match the current world size")
    saved_run = checkpoint.get("run_semantics")
    if not isinstance(saved_run, dict) or (
        saved_run.get("world_size") != world_size
        or saved_run.get("prior_mcmc_steps") != args.prior_mcmc_steps
    ):
        raise ValueError("checkpoint prior-chain semantics do not match this evaluation")
    expected_shape = _expected_cfg_shape(args, local_batch) + (2,)
    for rank_state in states:
        prior = rank_state.get("prior_state") if isinstance(rank_state, dict) else None
        if not torch.is_tensor(prior) or prior.dtype != torch.float64 or tuple(prior.shape) != expected_shape:
            raise ValueError("checkpoint prior-chain shape/dtype does not match this run")
        if not torch.isfinite(prior).all().item():
            raise ValueError("checkpoint prior chain contains non-finite values")
        rng = rank_state.get("rng_state")
        try:
            generator = torch.Generator(device="cpu")
            if not torch.is_tensor(rng.get("cpu")) or rng["cpu"].dtype != torch.uint8:
                raise ValueError("missing CPU RNG state")
            generator.set_state(rng["cpu"])
            numpy_state = rng["numpy"]
            if not isinstance(numpy_state, (tuple, list)) or len(numpy_state) != 5 or not torch.is_tensor(numpy_state[1]):
                raise ValueError("missing NumPy RNG state")
            np.random.RandomState().set_state((numpy_state[0], numpy_state[1].cpu().numpy(), int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4])))
            random.Random().setstate(rng["python"])
            if device.type == "cuda" and (
                not isinstance(rng.get("cuda"), list)
                or len(rng["cuda"]) != torch.cuda.device_count()
                or any(not torch.is_tensor(state) or state.dtype != torch.uint8 for state in rng["cuda"])
            ):
                raise ValueError("missing CUDA RNG state")
        except Exception as exc:
            raise ValueError("checkpoint prior RNG state is invalid") from exc


def _restore_optimizer_states(checkpoint: dict, optimizers, schedulers) -> None:
    saved_optimizers = checkpoint.get("optimizer_state_dicts")
    saved_schedulers = checkpoint.get("scheduler_state_dicts")
    if not isinstance(saved_optimizers, list) or len(saved_optimizers) != len(optimizers):
        raise ValueError("checkpoint optimizer count does not match this training mode")
    if not isinstance(saved_schedulers, list) or len(saved_schedulers) != len(schedulers):
        raise ValueError("checkpoint scheduler count does not match this training mode")
    if any(not isinstance(state, dict) for state in saved_optimizers + saved_schedulers):
        raise ValueError("checkpoint contains malformed optimizer or scheduler state")
    for optimizer, state in zip(optimizers, saved_optimizers):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(schedulers, saved_schedulers):
        scheduler.load_state_dict(state)


def _save_checkpoint(path: Path, model, optimizers, schedulers, semantics, run_semantics,
                     progress: int, metric_rows: int, observable_rows: int, rank_state: dict,
                     run_state: dict) -> None:
    rank = dist.get_rank()
    rank_path = path.with_name(f".{path.name}.rank{rank}.pt")
    local_error = ""
    try:
        _atomic_torch_save(rank_state, rank_path)
    except Exception as exc:
        local_error = f"rank {rank} checkpoint state save failed: {exc}"
    collective_device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    failed = torch.tensor([int(bool(local_error))], dtype=torch.int64, device=collective_device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        gathered_errors = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_error, gathered_errors, dst=0)
        message = next((value for value in gathered_errors if value), "rank checkpoint state save failed") if rank == 0 else "rank checkpoint state save failed"
        status = [message]
        dist.broadcast_object_list(status, src=0)
        raise RuntimeError(status[0])
    dist.barrier()
    error = ""
    if rank == 0:
        try:
            gathered = [_safe_torch_load(path.with_name(f".{path.name}.rank{i}.pt"), map_location="cpu") for i in range(dist.get_world_size())]
            model_state = None if model is None else {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            }
            _atomic_torch_save({
                "checkpoint_format": 1,
                "checkpoint_semantics": semantics,
                "run_semantics": run_semantics,
                "model_state_dict": model_state,
                "optimizer_state_dicts": [optimizer.state_dict() for optimizer in optimizers],
                "scheduler_state_dicts": [scheduler.state_dict() for scheduler in schedulers],
                "progress": int(progress),
                "metric_rows": int(metric_rows),
                "observable_rows": int(observable_rows),
                "run_state": run_state,
                "rank_states": gathered,
            }, path)
        except Exception as exc:
            error = f"checkpoint save failed: {exc}"
    _propagate_rank0_io(error)
    cleanup_error = ""
    if rank == 0:
        try:
            for i in range(dist.get_world_size()):
                path.with_name(f".{path.name}.rank{i}.pt").unlink(missing_ok=True)
        except Exception as exc:
            cleanup_error = f"checkpoint temporary-file cleanup failed: {exc}"
    _propagate_rank0_io(cleanup_error)
    dist.barrier()


def _propagate_rank0_io(error: str) -> None:
    status = [bool(not error), error]
    dist.broadcast_object_list(status, src=0)
    if not status[0]:
        raise RuntimeError(status[1])


def _propagate_rank_error(error: str, device: torch.device) -> None:
    rank = dist.get_rank()
    collective_device = device if dist.get_backend() == "nccl" else torch.device("cpu")
    failed = torch.tensor([int(bool(error))], dtype=torch.int64, device=collective_device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if not int(failed.item()):
        return
    errors = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(error, errors, dst=0)
    status = [next((value for value in errors if value), "rank validation failed") if rank == 0 else ""]
    dist.broadcast_object_list(status, src=0)
    raise RuntimeError(status[0])


def _safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        return torch.load(path, map_location=map_location)


def _validated_cache_payload(path: Path, key: dict, args: argparse.Namespace):
    """Rank-0-only complete cache validation, including the tensor payload."""
    try:
        # DONE is the atomic generation manifest.  meta.json is only an
        # inspection aid and may be overwritten by a concurrent publisher.
        with open(path / "DONE", encoding="utf-8") as handle:
            done = json.load(handle)
        payload_name = done.get("payload") if isinstance(done, dict) else None
        if (
            not isinstance(done, dict)
            or done.get("status") != "ok"
            or done.get("schema") != 6
            or done.get("key") != key
            or not isinstance(payload_name, str)
            or Path(payload_name).name != payload_name
        ):
            return False, None, "missing or incompatible DONE manifest"
        payload_path = path / payload_name
        if not payload_path.is_file():
            return False, None, "DONE payload is missing"
        payload = _safe_torch_load(payload_path, map_location="cpu")
        cfgs = payload.get("cfgs") if isinstance(payload, dict) else payload
        expected = _expected_cfg_shape(args, args.batch_size)
        cached_steps = int(done.get("thermal_steps", -1))
        if cached_steps < int(args.thermal_steps):
            return False, None, f"cached thermal_steps={cached_steps} < requested={args.thermal_steps}"
        if not torch.is_tensor(cfgs) or tuple(cfgs.shape) != expected or cfgs.dtype != torch.cdouble:
            return False, None, f"payload shape/dtype mismatch: expected {expected}, torch.cdouble"
        if not torch.isfinite(torch.view_as_real(cfgs)).all().item():
            return False, None, "payload contains non-finite values"
        return True, {"cfgs": cfgs.contiguous()}, f"ok (thermal_steps={cached_steps})"
    except Exception as exc:
        return False, None, f"cache validation error: {exc}"


def _expected_cfg_shape(args: argparse.Namespace, batch: int) -> tuple[int, ...]:
    return (batch, args.D, args.T) + (args.L,) * (args.D - 1) + (args.N, args.N)


def _initial_cfgs(args, flow_pars: FlowPars, defect_mask: Optional[torch.Tensor], prior: float, device: torch.device) -> torch.Tensor:
    local_batch = args.batch_size // dist.get_world_size()
    eye = torch.eye(args.N, dtype=torch.cdouble, device=device)
    cfgs = eye.view((1, 1, 1) + (1,) * (args.D - 1) + (args.N, args.N)).expand(
        _expected_cfg_shape(args, local_batch)
    ).clone()
    update = HBOR(
        flow_pars,
        beta=((6.0 if args.beta is None else args.beta) if args.domain == "defect" else prior),
        defect_par=(prior if args.domain == "defect" else 1.0),
    )
    with torch.no_grad():
        if args.prior_hot_start:
            cfgs = update.haar(cfgs)
        for _ in range(args.thermal_steps):
            cfgs = update(cfgs, flow_pars.mask, defect_mask)
    return cfgs


def _scatter_global_cfgs(
    global_cfgs: Optional[torch.Tensor],
    local_batch: int,
    device: torch.device,
    local_shape: tuple[int, ...],
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size == 1:
        return global_cfgs.to(device=device)
    chunks = None
    if rank == 0:
        chunks = [torch.view_as_real(chunk.contiguous()).to(device=device) for chunk in global_cfgs.split(local_batch, dim=0)]
    local_real = torch.empty(local_shape + (2,), dtype=torch.float64, device=device)
    dist.scatter(local_real, scatter_list=chunks, src=0)
    return torch.view_as_complex(local_real.contiguous())


def prepare_prior(args, flow_pars, defect_mask, prior: float, end: float, parameter_name: str, device: torch.device) -> tuple[torch.Tensor, str]:
    """Load/save only the global thermalised prior batch; never a trajectory."""
    mode = str(args.cfg_cache)
    key = cache_key(args, prior, end, parameter_name)
    path = cache_path(args, key)
    rank = dist.get_rank()
    local_batch = args.batch_size // dist.get_world_size()
    hit = False
    payload = None
    reason = "cache disabled"
    if mode in {"auto", "read"} and rank == 0:
        hit, payload, reason = _validated_cache_payload(path, key, args)
    hit_box = [bool(hit), str(reason)]
    dist.broadcast_object_list(hit_box, src=0)
    hit, reason = bool(hit_box[0]), str(hit_box[1])
    if mode == "read" and not hit:
        raise FileNotFoundError(f"No compatible prior cache found for --cfg-cache read: {path}: {reason}")

    if hit:
        global_cfgs = None
        if rank == 0:
            global_cfgs = payload["cfgs"]
        cfgs = _scatter_global_cfgs(
            global_cfgs,
            local_batch,
            device,
            _expected_cfg_shape(args, local_batch),
        )
        return cfgs, "read"

    # Cache misses use a dedicated deterministic thermal stream.  Restore the
    # caller's stream afterwards so model/protocol randomness is identical to
    # a cache hit and independent of NF architecture size.
    thermal_rng_state = _capture_rng_state(device)
    _seed_thermal_rng(device, _thermal_seed(args, rank))
    try:
        cfgs = _initial_cfgs(args, flow_pars, defect_mask, prior, device)
    finally:
        _restore_rng_state(thermal_rng_state, device)
    if mode == "off":
        return cfgs, "off"

    local_real = torch.view_as_real(cfgs.contiguous())
    gathered = None
    if rank == 0:
        gathered = [torch.empty_like(local_real) for _ in range(dist.get_world_size())]
        dist.gather(local_real, gather_list=gathered, dst=0)
    else:
        dist.gather(local_real, dst=0)
    publish_error = ""
    if rank == 0:
        try:
            path.mkdir(parents=True, exist_ok=True)
            global_cfgs = torch.view_as_complex(torch.cat(gathered, dim=0).contiguous()).detach().cpu()
            generation = uuid.uuid4().hex
            payload_name = f"prior_cfgs_{generation}.pt"
            _atomic_torch_save({"cfgs": global_cfgs}, path / payload_name)
            manifest = {
                "schema": 6,
                "status": "ok",
                "key": key,
                "payload": payload_name,
                "thermal_steps": int(args.thermal_steps),
                "thermal_rng_version": THERMAL_RNG_VERSION,
                "thermal_seed_offset": THERMAL_SEED_OFFSET,
                "thermal_seed_scheme": "seed + offset + rank",
                "thermal_seed_base": _thermal_seed(args, 0),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            _atomic_json_save(manifest, path / "DONE")
            _atomic_json_save(manifest, path / "meta.json")
        except Exception as exc:
            publish_error = f"prior cache publish failed: {exc}"
    _propagate_rank0_io(publish_error)
    dist.barrier()
    return cfgs, "refresh" if mode == "refresh" else "saved"


def _build_flow_pars(args, device: torch.device, protocol: list[float], local_batch: int):
    mask = create_mask(args.D, args.T, args.L).to(device)
    defect = None
    defect_mask = None
    kwargs = {}
    if args.domain == "defect" and args.nf_architecture == "cnf":
        defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
        defect_mask = defect.create_defect_mask().to(device)
    elif args.domain == "defect":
        buffer = 2
        if args.T < 2 * buffer or args.L < args.defect_size + 2 * buffer:
            raise ValueError("Defect SNF requires T >= 4 and L >= defect-size + 4")
        if not buffer <= args.time_slice <= args.T - buffer:
            raise ValueError("time-slice must satisfy 2 <= time-slice <= T-2")
        if not buffer <= args.space_slice <= args.L - args.defect_size - buffer:
            raise ValueError("space-slice must leave a two-site buffer around the defect")
        defect = Defect(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice)
        defect_mask = defect.create_defect_mask().to(device)
        small_t, small_l = 2 * buffer, args.defect_size + 2 * buffer
        small_defect = Defect(args.D, small_t, small_l, args.defect_size, buffer, buffer)
        kwargs.update(
            rho_shape_type=3,
            smeared_defect_mask=[create_around_defect_mask(args.D, args.defect_size, buffer).to(device)],
            small_mask=[create_mask(args.D, small_t, small_l).to(device)],
            small_defect_mask=[small_defect.create_defect_mask().to(device)],
        )
    architecture = args.nf_architecture
    use_hyper_smearing = architecture == "hyper-smearing"
    residual_parameterization = "static" if architecture == "residual" else "hyper"
    if args.domain == "beta" and use_hyper_smearing and args.hyper_smearing_mode == "per_link":
        # HyperSmearing.per_link emits one coefficient per oriented plaquette;
        # rho_shape_type=0 has no plaquette axis and is not broadcastable.
        kwargs["rho_shape_type"] = 1
    flow_pars = FlowPars(
        D=args.D, T=args.T, L=args.L, N=args.N, mask=mask, protocol=protocol,
        batch_size=local_batch, device=device, orsteps=args.orsteps,
        updates_per_layer=args.updates_per_layer, defect=defect,
        hierarchical_levels=args.hierarchical_levels,
        hierarchical_updates=args.hierarchical_updates,
        hierarchical_rectangles=args.hierarchical_buffers or [],
        residual_parameterization=residual_parameterization,
        use_hyper_smearing=use_hyper_smearing,
        hyper_smearing_mode=args.hyper_smearing_mode,
        hyper_time_embedding_dim=args.hyper_time_embedding_dim,
        hyper_hidden_dim=args.hyper_hidden_dim, hyper_depth=args.hyper_depth,
        hyper_activation=args.hyper_activation, hyper_rho_init=args.hyper_rho_init,
        hyper_rho_eps=args.hyper_rho_eps, hyper_rho_max=args.hyper_rho_max,
        hyper_normalize_by_nstep=args.hyper_normalize_by_nstep,
        hyper_scale_by_delta=args.hyper_scale_by_delta,
        residual_include_imag=args.residual_include_imag,
        residual_quadratic=args.residual_quadratic,
        residual_coeff_init=args.residual_coeff_init,
        residual_coeff_max=args.residual_coeff_max,
        residual_n_pt=args.residual_n_pt,
        residual_mask_mod=args.residual_mask_mod,
        explicit_backward=args.explicit_backward,
        use_rho_functional=args.rho_functional,
        rho_scale_fac=args.rho_functional_scale,
        F=args.rho_functional_modes,
        hidden_time_mlp=args.rho_functional_hidden_dim,
        **kwargs,
    )
    return flow_pars, defect_mask


class _DefectSNFProtocolLayer(torch.nn.Module):
    """Local, gradient-safe defect patch adapter from the OBC implementation."""

    def __init__(self, flow_pars, residual: bool, rho_init: float):
        super().__init__()
        self.flow_pars = flow_pars
        if residual:
            self.layer = ResidualNormalizingFlows(flow_pars, time=1.0)
        else:
            rho0 = torch.full(
                flow_pars.rho_shape,
                float(rho_init),
                dtype=torch.float64,
                device=flow_pars.device,
            )
            self.layer = DefectCouplingLayer(flow_pars, rho0, 1.0)

    def _embed(self, small_cfgs, cfgs, buffer=2):
        defect = self.flow_pars.defect
        t, s, dsize, D = defect.time_slice, defect.space_slice, defect.dsize, defect.D
        out = cfgs.clone()
        out[:, 0, t, s - 1:dsize + s + 1, s - 1:dsize + s + 1, s - 1:dsize + s + 1] = small_cfgs[:, 0, buffer, buffer - 1:dsize + buffer + 1, buffer - 1:dsize + buffer + 1, buffer - 1:dsize + buffer + 1]
        for mu in range(1, D):
            out[:, mu, t, s:dsize + s + 1, s:dsize + s + 1, s:dsize + s + 1] = small_cfgs[:, mu, buffer, buffer:dsize + buffer + 1, buffer:dsize + buffer + 1, buffer:dsize + buffer + 1]
            out[:, mu, t - 1, s:dsize + s + 1, s:dsize + s + 1, s:dsize + s + 1] = small_cfgs[:, mu, buffer - 1, buffer:dsize + buffer + 1, buffer:dsize + buffer + 1, buffer:dsize + buffer + 1]
        return out

    def forward(self, x, parameter, delta_parameter, rho_layer=None):
        work_dtype = x.real.dtype
        if isinstance(self.layer, ResidualNormalizingFlows):
            small = self.flow_pars.defect.cut_defect(x, buffer=2)
            small, _, logdet = self.layer(
                small, self.flow_pars.small_mask[-1], dmasking=True,
                dmask=self.flow_pars.smeared_defect_mask[-1], beta=parameter,
                delta_beta=delta_parameter,
            )
            return self._embed(small, x), logdet.to(dtype=work_dtype)
        small, _, logdet = self.layer(
            x, self.flow_pars, rho_layer=rho_layer, is_training=True,
            beta=parameter, delta_beta=delta_parameter,
        )
        return self._embed(small, x), logdet.to(dtype=work_dtype)


class _StandardSNFProtocolLayer(torch.nn.Module):
    """Adapter giving existing Flow-layer conventions a standalone interface."""

    def __init__(self, layer, flow_pars):
        super().__init__()
        self.layer = layer
        self.flow_pars = flow_pars

    def forward(self, x, parameter, delta_parameter, rho_layer=None):
        x_out, _, logdet = self.layer(
            x,
            self.flow_pars,
            rho_layer=rho_layer,
            is_training=True,
            beta=parameter,
            delta_beta=delta_parameter,
        )
        return x_out, logdet.to(dtype=x.real.dtype)


def _make_snf_layer(args, flow_pars):
    architecture = args.nf_architecture
    if architecture == "cnf":
        from neoqcd.cnf import CNFProtocolLayer
        from neoqcd.cnf_io import load_cnf_checkpoint
        expected = {key: getattr(args, key) for key in ("D", "T", "L", "domain")}
        if args.domain == "defect":
            expected.update(beta=6.0 if args.beta is None else args.beta,
                            **{key: getattr(args, key) for key in ("defect_size", "time_slice", "space_slice")})
        start, end, _, _ = _protocol_endpoints(args)
        field, _, _, _ = load_cnf_checkpoint(args.cnf_checkpoint, flow_pars.device, expected, (start, end))
        return CNFProtocolLayer(field, n_steps=args.cnf_integration_steps)
    rho_init = args.smearing_rho_init if architecture == "smearing" else 0.0
    if architecture == "hyper-residual":
        warnings.warn(
            "nf-architecture=hyper-residual selects the existing beta-conditioned residual MLP",
            RuntimeWarning,
            stacklevel=2,
        )
    if architecture in {"residual", "hyper-residual"}:
        if args.domain == "defect":
            return _DefectSNFProtocolLayer(flow_pars, residual=True, rho_init=rho_init)
        rho0 = torch.zeros(flow_pars.rho_shape, dtype=torch.float64, device=flow_pars.device)
        # residual uses static direct coefficients; hyper-residual uses the
        # existing beta-conditioned MLP through FlowPars.
        return _StandardSNFProtocolLayer(ResidualCouplingLayer(flow_pars, rho0, 1.0), flow_pars)
    if args.domain == "defect":
        return _DefectSNFProtocolLayer(flow_pars, residual=False, rho_init=rho_init)
    rho0 = torch.full(
        flow_pars.rho_shape,
        float(args.smearing_rho_init),
        dtype=torch.float64,
        device=flow_pars.device,
    )
    return _StandardSNFProtocolLayer(CouplingLayer(flow_pars, rho0, 1.0), flow_pars)


def _nf_layer_sharing(args) -> str:
    if args.rho_functional:
        return "shared"
    return "shared" if args.nf_architecture in {"hyper-smearing", "hyper-residual", "cnf"} else "per_block"


def _action(flow_pars, x, domain: str, parameter: float, defect_mask, plaquettes=None):
    action = None
    if domain == "beta":
        action = Wilson_action(flow_pars, float(parameter), 1.0)
    else:
        action = Wilson_action(flow_pars, flow_pars._standalone_beta, float(parameter))
    return action(x, defect_mask) if plaquettes is None else action.from_plaquettes(plaquettes, defect_mask)


class StandaloneProtocol(torch.nn.Module):
    """Shared-NF K-block protocol with detached stochastic transitions."""

    def __init__(self, args, flow_pars, defect_mask, start: float, end: float, protocol_values=None):
        super().__init__()
        self.args = args
        self.flow_pars = flow_pars
        self.defect_mask = defect_mask
        self.domain = args.domain
        self.start = float(start)
        self.end = float(end)
        self.protocol_values = protocol_values or [
            self.start + (self.end - self.start) * (i / args.protocol_steps)
            for i in range(args.protocol_steps + 1)
        ]
        self.nf_layer_sharing = _nf_layer_sharing(args) if args.algorithm == "snf" else None
        self.rho_functional = None
        if args.rho_functional:
            embedding = time_embedding(args.rho_functional_modes, T=2.0, device=flow_pars.device)
            self.rho_functional = HyperTimeScalar(
                embedding,
                flow_pars.rho_shape,
                args.protocol_steps,
                args.rho_functional_scale,
                hidden_dim=args.rho_functional_hidden_dim,
                device=flow_pars.device,
            )
        if args.algorithm == "snf" and self.nf_layer_sharing == "shared":
            self.nf_layer = _make_snf_layer(args, flow_pars)
            self.nf_layers = None
        elif args.algorithm == "snf":
            self.nf_layer = None
            self.nf_layers = torch.nn.ModuleList(
                [_make_snf_layer(args, flow_pars) for _ in range(args.protocol_steps)]
            )
        else:
            self.nf_layer = None
            self.nf_layers = None
        self.mcmc_updates = torch.nn.ModuleList()
        for p in self.protocol_values[1:]:
            self.mcmc_updates.append(
                NEMCMC_update(
                    flow_pars,
                    beta=((6.0 if args.beta is None else args.beta) if args.domain == "defect" else p),
                    defect_mask=defect_mask if args.domain == "defect" else None,
                    defect_par=(p if args.domain == "defect" else 1.0),
                )
            )
        self.prior_update = NEMCMC_update(
            flow_pars,
            beta=((6.0 if args.beta is None else args.beta) if args.domain == "defect" else self.start),
            defect_mask=defect_mask if args.domain == "defect" else None,
            defect_par=(self.start if args.domain == "defect" else 1.0),
        )

    def _parameter_tensor(self, value, batch, device):
        return torch.full((batch,), float(value), dtype=torch.float64, device=device)

    def _run_nf_block(self, x, step):
        if self.nf_layer_sharing == "shared":
            layer = self.nf_layer
        elif self.nf_layer_sharing == "per_block":
            layer = self.nf_layers[int(step)]
        else:
            return x, torch.zeros(x.shape[0], dtype=x.real.dtype, device=x.device)
        rho_layer = None if self.rho_functional is None else self.rho_functional((int(step) + 1) / self.args.protocol_steps)
        return layer(
            x,
            self._parameter_tensor(self.protocol_values[step], x.shape[0], x.device),
            self._parameter_tensor(
                self.protocol_values[step + 1] - self.protocol_values[step],
                x.shape[0],
                x.device,
            ),
            rho_layer=rho_layer,
        )

    def _run_stochastic_block(self, x, step, plaquettes=None, return_plaquettes=False):
        return self.mcmc_updates[step](
            x, self.flow_pars, None, plaquettes=plaquettes,
            return_plaquettes=return_plaquettes,
        )

    def trainable_parameters_for_block(self, block_index: int):
        if self.rho_functional is not None:
            return list(self.rho_functional.parameters())
        if self.nf_layer_sharing == "shared":
            return list(self.nf_layer.parameters())
        return list(self.nf_layers[int(block_index)].parameters())

    def trainable_parameters(self):
        if self.rho_functional is not None:
            return list(self.rho_functional.parameters())
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @torch.no_grad()
    def advance_prior(self, x, steps: int):
        x = x.detach()
        plaquettes = None
        for _ in range(int(steps)):
            x, _, _, plaquettes, _ = self.prior_update(
                x, self.flow_pars, None, plaquettes=plaquettes, return_plaquettes=True
            )
            x = x.detach()
        return x

    def up_to_block_nograd(self, x, s0, block_index: int):
        """Notebook-compatible prefix no-grad plus one trainable NF block."""
        x = x.detach()
        work_dtype = x.real.dtype
        q = torch.zeros(x.shape[0], dtype=work_dtype, device=x.device)
        logdet = torch.zeros_like(q)
        with torch.no_grad():
            for step in range(int(block_index)):
                x, dlogdet = self._run_nf_block(x, step)
                logdet = logdet + dlogdet
                plaquettes = sun.plaquette_fields(x, self.flow_pars.D)
                x, dq, _, _, _ = self._run_stochastic_block(
                    x.detach(), step, plaquettes=plaquettes, return_plaquettes=True
                )
                q = q + (dq if torch.is_tensor(dq) else torch.zeros_like(q))
                x = x.detach()
        x, dlogdet = self._run_nf_block(x, int(block_index))
        logdet = logdet + dlogdet
        p1 = self.protocol_values[int(block_index) + 1]
        plaquettes = sun.plaquette_fields(x, self.flow_pars.D)
        st = _action(self.flow_pars, x, self.domain, p1, self.defect_mask, plaquettes)
        work = st - s0 - q - logdet
        return x, work, st, q, logdet

    def full_training_forward(self, x):
        with torch.no_grad():
            s0 = _action(self.flow_pars, x, self.domain, self.start, self.defect_mask)
        q = x.real.new_zeros(x.shape[0])
        logdet = torch.zeros_like(q)
        plaquettes = None
        for step, update in enumerate(self.mcmc_updates):
            if self.args.algorithm == "snf":
                x, dlogdet = self._run_nf_block(x, step)
                logdet = logdet + dlogdet
                plaquettes = None
            x, dq, _, plaquettes, _ = update(
                x, self.flow_pars, None, plaquettes=plaquettes, return_plaquettes=True
            )
            q = q + (dq if torch.is_tensor(dq) else torch.zeros_like(q))
        send = _action(self.flow_pars, x, self.domain, self.end, self.defect_mask, plaquettes)
        work = send - s0 - q - logdet
        return x, work, work

    def blockwise_training_forward(self, x, s0, scale, world_size, device):
        x = x.detach()
        q = torch.zeros(x.shape[0], dtype=x.real.dtype, device=x.device)
        logdet = torch.zeros_like(q)
        block_losses, block_skips = [], []
        for step, update in enumerate(self.mcmc_updates):
            x, dlogdet = self._run_nf_block(x, step)
            plaquettes = sun.plaquette_fields(x, self.flow_pars.D)
            action = _action(
                self.flow_pars, x, self.domain, self.protocol_values[step + 1],
                self.defect_mask, plaquettes,
            )
            loss_value, skipped = _backprop_local_loss(action - dlogdet, scale, world_size, device)
            block_losses.append(loss_value)
            block_skips.append(skipped)
            logdet = logdet + dlogdet.detach()
            x = x.detach()
            with torch.no_grad():
                x, dq, _, plaquettes, _ = update(
                    x, self.flow_pars, None, plaquettes=plaquettes, return_plaquettes=True
                )
                q = q + (dq if torch.is_tensor(dq) else torch.zeros_like(q))
                x = x.detach()
        with torch.no_grad():
            final_action = _action(self.flow_pars, x, self.domain, self.end, self.defect_mask, plaquettes)
            work = final_action - s0 - q - logdet
        return x, work, block_losses, block_skips

    def forward(self, x, train: bool = False, block_index=None, s0=None, int_meas=1, return_observables=False):
        if train and block_index is not None:
            if s0 is None:
                raise ValueError("s0 is required for blockwise SNF training")
            return self.up_to_block_nograd(x, s0, int(block_index))
        if train:
            return self.full_training_forward(x)
        x = x.detach()
        if int_meas < 1 or self.args.protocol_steps % int_meas:
            raise ValueError("int_meas must be a positive divisor of protocol_steps")
        work_dtype = x.real.dtype
        initial_plaquettes = sun.plaquette_fields(x, self.flow_pars.D)
        with torch.no_grad():
            s_initial = _action(
                self.flow_pars, x, self.domain, self.start, self.defect_mask,
                initial_plaquettes,
            )
        q = torch.zeros(x.shape[0], dtype=work_dtype, device=x.device)
        total_logdet = torch.zeros_like(q)
        int_obs = torch.zeros((x.shape[0], int_meas, 7), dtype=work_dtype, device=x.device)
        measure_every = self.args.protocol_steps // int_meas
        measure_index = 0
        plaquettes = initial_plaquettes

        for i, update in enumerate(self.mcmc_updates):
            p0, p1 = self.protocol_values[i], self.protocol_values[i + 1]
            if self.args.algorithm == "snf":
                x, dlogdet = self._run_nf_block(x, i)
                total_logdet = total_logdet + dlogdet
                plaquettes = None
            with torch.no_grad():
                x, dq, _, plaquettes, _ = update(
                    x, self.flow_pars, None, plaquettes=plaquettes, return_plaquettes=True
                )
                q = q + (dq if torch.is_tensor(dq) else torch.zeros_like(q))
                x = x.detach()
            if (i + 1) % measure_every == 0 and measure_index < int_meas:
                with torch.no_grad():
                    action = _action(self.flow_pars, x, self.domain, p1, self.defect_mask, plaquettes)
                    traces = torch.stack(plaquettes, dim=1) / self.flow_pars.N
                    reduce_dims = tuple(range(1, traces.dim()))
                    int_obs[:, measure_index, 0] = action - s_initial - q - total_logdet
                    int_obs[:, measure_index, 1] = q
                    int_obs[:, measure_index, 2] = total_logdet
                    int_obs[:, measure_index, 3] = action
                    int_obs[:, measure_index, 4] = traces.mean(dim=reduce_dims)
                    int_obs[:, measure_index, 5] = traces.square().mean(dim=reduce_dims)
                    int_obs[:, measure_index, 6] = p1
                measure_index += 1

        with torch.no_grad():
            s_final = _action(self.flow_pars, x, self.domain, self.end, self.defect_mask, plaquettes)
            exact_work = s_final - s_initial - q - total_logdet.detach()
        result = (x, exact_work, exact_work)
        return (*result, int_obs) if return_observables else result


def _global_logsumexp(values: torch.Tensor, multiplier: float) -> Optional[float]:
    finite = torch.isfinite(values)
    transformed = -float(multiplier) * values
    local_max = transformed[finite].max() if bool(finite.any().item()) else torch.tensor(
        float("-inf"), dtype=values.dtype, device=values.device
    )
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    if not math.isfinite(float(local_max.item())):
        return None
    scaled = torch.where(finite, torch.exp(transformed - local_max), torch.zeros_like(values)).sum()
    dist.all_reduce(scaled, op=dist.ReduceOp.SUM)
    if not math.isfinite(float(scaled.item())) or float(scaled.item()) <= 0.0:
        return None
    result = float((local_max + torch.log(scaled)).item())
    return result if math.isfinite(result) else None


def _global_work_metrics(values: torch.Tensor) -> dict:
    """Reduce work moments and free-energy observables without NaN JSON values."""
    values = values.detach().to(dtype=torch.float64)
    finite = torch.isfinite(values)
    pair = torch.stack(
        (
            torch.where(finite, values, torch.zeros_like(values)).sum(),
            finite.to(values.dtype).sum(),
        )
    )
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    count = int(pair[1].item())
    mean = None
    sem = None
    if count > 0 and math.isfinite(float(pair[0].item())):
        mean_value = float(pair[0].item() / count)
        if math.isfinite(mean_value):
            mean = mean_value
            center = torch.tensor(mean_value, dtype=values.dtype, device=values.device)
            sq = torch.where(finite, (values - center) ** 2, torch.zeros_like(values)).sum()
            dist.all_reduce(sq, op=dist.ReduceOp.SUM)
            sem_value = math.sqrt(max(0.0, float(sq.item())) / (count * (count - 1))) if count > 1 else None
            sem = sem_value if sem_value is None or math.isfinite(sem_value) else None

    delta_f = None
    ess_fraction = None
    log_sum_w = _global_logsumexp(values, 1.0)
    log_sum_w2 = _global_logsumexp(values, 2.0)
    if count > 0 and log_sum_w is not None and log_sum_w2 is not None:
        log_count = math.log(float(count))
        delta_value = -(log_sum_w - log_count)
        ess_log = 2.0 * log_sum_w - log_count - log_sum_w2
        ess_value = math.exp(min(0.0, ess_log)) if math.isfinite(ess_log) else None
        delta_f = delta_value if math.isfinite(delta_value) else None
        ess_fraction = ess_value if ess_value is None or math.isfinite(ess_value) else None
    return {
        "work_mean": mean,
        "work_sem": sem,
        "deltaF": delta_f,
        "ess_fraction": ess_fraction,
        "finite_count": count,
    }


def _resolve_output_dir(args, rank, world_size) -> Path:
    if rank == 0:
        if args.output_dir:
            output = Path(args.output_dir)
        else:
            name = args.run_name or f"{args.algorithm}_{args.domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output = Path(args.main_dir) / "results" / "nemcmc_snf" / name
        try:
            if output.exists() and any(output.iterdir()) and not args.overwrite and not args.resume:
                decision = [False, "", f"output directory is non-empty: {output} (use --overwrite explicitly)"]
            else:
                output.mkdir(parents=True, exist_ok=True)
                decision = [True, str(output.resolve()), ""]
        except OSError as exc:
            decision = [False, "", f"cannot prepare output directory {output}: {exc}"]
    else:
        decision = [False, "", ""]
    box = decision
    dist.broadcast_object_list(box, src=0)
    if not bool(box[0]):
        raise RuntimeError(str(box[2]))
    return Path(box[1])


def _clear_managed_output_artifacts(output: Path, rank: int, input_checkpoint: Optional[Path]) -> None:
    """Remove only artifacts owned by a new run, after input checkpoint load."""
    error = ""
    if rank == 0:
        try:
            managed = ("snf_checkpoint.pt", "summary.json", "metrics.csv", "observables.csv", "gamma_analysis.csv", "run_meta.json")
            input_resolved = input_checkpoint.resolve() if input_checkpoint is not None else None
            for name in managed:
                artifact = (output / name).resolve()
                if input_resolved is not None and artifact == input_resolved:
                    raise ValueError(
                        "input checkpoint cannot be the output snf_checkpoint.pt; use a separate output directory"
                    )
                artifact.unlink(missing_ok=True)
        except Exception as exc:
            error = f"output cleanup failed: {exc}"
    status = [bool(not error), error]
    dist.broadcast_object_list(status, src=0)
    if not status[0]:
        raise RuntimeError(status[1])


def _checkpoint_semantics(args, protocol: list[float]) -> dict:
    semantics = {
        "schema": 6,
        "algorithm": args.algorithm,
        "domain": args.domain,
        "D": int(args.D), "T": int(args.T), "L": int(args.L), "N": int(args.N),
        "global_batch": int(args.batch_size),
        "updates_per_layer": int(args.updates_per_layer),
        "orsteps": int(args.orsteps),
        "prior_mcmc_steps": int(args.prior_mcmc_steps),
        "hierarchical_levels": int(args.hierarchical_levels),
        "hierarchical_updates": [int(value) for value in args.hierarchical_updates],
        "hierarchical_buffers": [int(value) for value in (args.hierarchical_buffers or [])],
        "nf_layer_sharing": _nf_layer_sharing(args),
        "protocol_steps": int(args.protocol_steps),
        "protocol": [float(value) for value in protocol],
        "fixed_beta": (6.0 if args.beta is None else float(args.beta)) if args.domain == "defect" else None,
        "nf_architecture": args.nf_architecture,
        "rho_functional": bool(args.rho_functional),
        "rho_functional_modes": int(args.rho_functional_modes),
        "rho_functional_hidden_dim": int(args.rho_functional_hidden_dim),
        "rho_functional_scale": float(args.rho_functional_scale),
        "explicit_backward": bool(args.explicit_backward),
        "residual_parameterization": "static" if args.nf_architecture == "residual" else "hyper",
        "hyper_smearing_mode": args.hyper_smearing_mode,
        "hyper_time_embedding_dim": int(args.hyper_time_embedding_dim),
        "hyper_hidden_dim": int(args.hyper_hidden_dim),
        "hyper_depth": int(args.hyper_depth),
        "hyper_activation": args.hyper_activation,
        "hyper_normalize_by_nstep": bool(args.hyper_normalize_by_nstep),
        "hyper_scale_by_delta": bool(args.hyper_scale_by_delta),
        "hyper_class_inactive_rho": float(HYPER_CLASS_INACTIVE_RHO),
        "hyper_rho_eps": float(args.hyper_rho_eps),
        "hyper_rho_max": float(args.hyper_rho_max),
        "residual_include_imag": bool(args.residual_include_imag),
        "residual_quadratic": bool(args.residual_quadratic),
        "residual_coeff_max": float(args.residual_coeff_max),
        "smearing_rho_init": float(args.smearing_rho_init),
        "smearing_steps_per_layer": 1,
        "defect_size": int(args.defect_size),
        "time_slice": int(args.time_slice), "space_slice": int(args.space_slice),
    }
    if args.algorithm == "snf" and args.nf_architecture in {"smearing", "hyper-smearing"}:
        semantics["smearing_jacobian_version"] = "su3_stout_full_haar_logdet_v1"
    if args.residual_n_pt or args.residual_mask_mod != 2:
        semantics.update(residual_n_pt=args.residual_n_pt, residual_mask_mod=args.residual_mask_mod)
    if args.nf_architecture == "cnf":
        payload = torch.load(args.cnf_checkpoint, map_location="cpu", weights_only=True)
        semantics.update(cnf_config=payload["config"], cnf_integration_steps=args.cnf_integration_steps,
                         cnf_jacobian="cf4_integrated_divergence_v1")
    return semantics


def _checkpoint_semantics_match(source: dict, expected: dict) -> bool:
    if source == expected:
        return True
    # Old residual checkpoints used the same residual map. Keep that path when
    # their saved keys and tensor shapes still match; classic smearing requires
    # the explicit full-Haar Jacobian version above.
    if expected.get("nf_architecture") not in {"residual", "hyper-residual"}:
        return False
    old_optional = {
        "rho_functional": False, "rho_functional_modes": 9,
        "rho_functional_hidden_dim": 16, "rho_functional_scale": 1e-3,
        "explicit_backward": False, "residual_parameterization": "static",
        "residual_n_pt": 0, "residual_mask_mod": 2,
        "hierarchical_levels": 0, "hierarchical_updates": [], "hierarchical_buffers": [],
    }
    source = dict(source)
    target = dict(expected)
    source.pop("schema", None)
    target.pop("schema", None)
    return all(
        source.get(key, old_optional.get(key, object())) == value
        for key, value in target.items()
    ) and all(key in target for key in source)


def _load_checkpoint_coordinated(model, path: Path, expected: dict, device: torch.device,
                                 transfer_rho_functional: bool = False,
                                 expected_run_semantics: Optional[dict] = None) -> dict:
    rank = dist.get_rank()
    checkpoint = None
    error = ""
    if rank == 0:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoint = _safe_torch_load(path, map_location="cpu")
            source_semantics = checkpoint.get("checkpoint_semantics") if isinstance(checkpoint, dict) else None
            if transfer_rho_functional:
                if not isinstance(source_semantics, dict) or not source_semantics.get("rho_functional"):
                    raise ValueError("functional-rho transfer requires a functional-rho checkpoint")
                transfer_fields = (
                    "algorithm", "domain", "D", "T", "L", "N", "nf_architecture",
                    "rho_functional", "rho_functional_modes", "rho_functional_hidden_dim",
                    "rho_functional_scale", "smearing_steps_per_layer", "defect_size",
                )
                if any(source_semantics.get(field) != expected.get(field) for field in transfer_fields):
                    raise ValueError("functional-rho checkpoint architecture/shape metadata does not match")
            elif not _checkpoint_semantics_match(source_semantics, expected):
                raise ValueError("checkpoint semantic metadata does not match this run")
            if expected_run_semantics is not None and checkpoint.get("checkpoint_format") != 1:
                raise ValueError("checkpoint is not a resumable runner checkpoint")
            if expected_run_semantics is not None and not _run_semantics_match(
                checkpoint.get("run_semantics"), expected_run_semantics
            ):
                raise ValueError("checkpoint training/evaluation semantics do not match this resume")
            state = checkpoint.get("model_state_dict")
            if model is not None and not isinstance(state, dict):
                raise ValueError("checkpoint has no model_state_dict")
            if transfer_rho_functional and state is not None:
                state = {name: value for name, value in state.items() if name.startswith("rho_functional.")}
                if not state:
                    raise ValueError("checkpoint has no functional-rho state")
                target_state = model.rho_functional.state_dict()
                source_state = {name.removeprefix("rho_functional."): value for name, value in state.items()}
                if set(source_state) != set(target_state) or any(
                    tuple(source_state[name].shape) != tuple(target_state[name].shape) for name in target_state
                ):
                    raise ValueError("functional-rho checkpoint tensor shapes do not match")
        except Exception as exc:  # propagate the same failure before any rank loads
            error = f"checkpoint preflight failed: {exc}"
    status = [error]
    dist.broadcast_object_list(status, src=0)
    if status[0]:
        raise RuntimeError(status[0])

    local_error = ""
    try:
        if rank != 0:
            checkpoint = _safe_torch_load(path, map_location="cpu")
    except Exception as exc:
        local_error = f"rank {rank} checkpoint read failed: {exc}"
    failed = torch.tensor([int(bool(local_error))], dtype=torch.int64, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        gathered = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_error, gathered, dst=0)
        message = next((value for value in gathered if value), "checkpoint read failed on at least one rank") if rank == 0 else "checkpoint read failed on at least one rank"
        status = [message]
        dist.broadcast_object_list(status, src=0)
        raise RuntimeError(status[0])
    state = checkpoint.get("model_state_dict")
    if transfer_rho_functional and isinstance(state, dict):
        state = {name: value for name, value in state.items() if name.startswith("rho_functional.")}

    local_error = ""
    try:
        if model is None or state is None:
            pass
        elif transfer_rho_functional:
            model.rho_functional.load_state_dict({name.removeprefix("rho_functional."): value for name, value in state.items()})
        else:
            model.load_state_dict(state)
    except Exception as exc:
        local_error = str(exc)
    failed = torch.tensor([1 if local_error else 0], dtype=torch.int64, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        gathered = [None for _ in range(dist.get_world_size())] if rank == 0 else None
        dist.gather_object(local_error, gathered, dst=0)
        message = next((value for value in gathered if value), "checkpoint load failed on at least one rank") if rank == 0 else "checkpoint load failed on at least one rank"
        box = [message]
        dist.broadcast_object_list(box, src=0)
        raise RuntimeError(box[0])
    return checkpoint


def _wandb_start(args, rank, world_size, output, metadata):
    global _ACTIVE_WANDB_RUN
    run = None
    error = ""
    if rank == 0 and args.wandb_mode != "disabled":
        try:
            import wandb
            name = args.wandb_run_name or args.run_name or f"{args.algorithm}_{args.domain}_r{world_size}"
            run = wandb.init(
                entity=args.wandb_entity, project=args.wandb_project, name=name,
                mode=args.wandb_mode, dir=str(output), config=metadata,
            )
        except Exception as exc:
            error = f"W&B initialization failed: {exc}"
    status = [bool(not error), error]
    dist.broadcast_object_list(status, src=0)
    if not status[0]:
        raise RuntimeError(status[1])
    _ACTIVE_WANDB_RUN = run
    return run


def _backprop_local_loss(train_work: torch.Tensor, scale: int, world_size: int, device):
    finite = torch.isfinite(train_work)
    local_count = finite.to(dtype=torch.long).sum()
    global_count = local_count.clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if int(global_count.item()) <= 0:
        return None, 1

    local_sum = torch.where(finite, train_work, torch.zeros_like(train_work)).sum()
    loss = local_sum * (world_size / float(global_count.item()))
    global_loss_sum = local_sum.detach().clone()
    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
    loss_value = float(global_loss_sum.item() / global_count.item()) if math.isfinite(float(global_loss_sum.item())) else None
    bad_loss = torch.tensor(
        [0 if torch.isfinite(loss.detach()).item() and loss_value is not None and math.isfinite(loss_value) else 1],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(bad_loss, op=dist.ReduceOp.MAX)
    no_grad = torch.tensor([0 if loss.requires_grad else 1], dtype=torch.int64, device=device)
    dist.all_reduce(no_grad, op=dist.ReduceOp.MAX)
    if int(bad_loss.item()) or int(no_grad.item()):
        return loss_value, 1

    (loss / int(scale)).backward()
    return loss_value, 0


def _broadcast_parameters(model):
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)


def _average_gradients(parameters):
    by_device_dtype = {}
    for parameter in parameters:
        by_device_dtype.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    world_size = dist.get_world_size()
    states = {}
    for parameters_group in by_device_dtype.values():
        grad_parts = [
            parameter.grad.detach().reshape(-1)
            if parameter.grad is not None else torch.zeros_like(parameter).reshape(-1)
            for parameter in parameters_group
        ]
        gradient_size = sum(part.numel() for part in grad_parts)
        flat_gradients = torch.cat(grad_parts)
        present = torch.tensor(
            [1 if parameter.grad is not None else 0 for parameter in parameters_group],
            dtype=parameters_group[0].dtype,
            device=parameters_group[0].device,
        )
        packed = torch.cat((flat_gradients, present))
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed[:gradient_size].div_(world_size)
        reduced = torch._utils._unflatten_dense_tensors(packed[:gradient_size], parameters_group)
        for parameter, average, count in zip(parameters_group, reduced, packed[gradient_size:]):
            present_count = int(round(float(count.item())))
            has_grad = present_count > 0
            if has_grad:
                if parameter.grad is None:
                    parameter.grad = average.detach().clone()
                else:
                    parameter.grad.copy_(average)
            else:
                parameter.grad = None
            states[id(parameter)] = {
                "missing_rank": 0 < present_count < world_size,
            }
    return states


def _make_optimizers(args, model):
    if args.training_mode == "full" or model.nf_layer_sharing == "shared":
        parameter_sets = [model.trainable_parameters()]
    else:
        parameter_sets = [model.trainable_parameters_for_block(block) for block in range(args.protocol_steps)]
    optimizers = [torch.optim.Adam(parameters, lr=args.flow_lr) for parameters in parameter_sets]
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.scheduler_min_lr,
        )
        for optimizer in optimizers
    ]
    return optimizers, schedulers


def _finish_optimizer_step(optimizers, schedulers, model, losses, args, active_indices=None,
                           consecutive_bad_grad_updates=None):
    if consecutive_bad_grad_updates is None:
        consecutive_bad_grad_updates = [0] * len(optimizers)
    if len(consecutive_bad_grad_updates) != len(optimizers):
        raise ValueError("gradient streak count must match optimizer count")
    if len(losses) != len(optimizers):
        raise ValueError("optimizer loss count must match optimizer count")
    streaks = list(consecutive_bad_grad_updates)
    active = set(range(len(optimizers)) if active_indices is None else active_indices)
    optimizer_parameters = [
        [parameter for group in optimizer.param_groups for parameter in group["params"]]
        for optimizer in optimizers
    ]
    parameter_ids = set()
    parameters = []
    for index in sorted(active):
        for parameter in optimizer_parameters[index]:
            if id(parameter) not in parameter_ids:
                parameter_ids.add(id(parameter))
                parameters.append(parameter)
    grad_states = _average_gradients(parameters)
    statuses = ["inactive"] * len(optimizers)
    skipped = [0] * len(optimizers)
    ready = []
    for index, (optimizer, scheduler, loss) in enumerate(zip(optimizers, schedulers, losses)):
        if index not in active:
            continue
        if loss is None or not math.isfinite(float(loss)):
            optimizer.zero_grad(set_to_none=True)
            statuses[index] = "no_loss" if loss is None else "nonfinite_loss"
            skipped[index] = 1
            continue
        params = optimizer_parameters[index]
        gradients = [p for p in params if p.grad is not None]
        missing_rank = any(grad_states.get(id(p), {}).get("missing_rank", False) for p in params)
        if not gradients or missing_rank:
            optimizer.zero_grad(set_to_none=True)
            statuses[index] = "missing_grad"
            skipped[index] = 1
            streaks[index] += 1
            continue
        if any(not torch.isfinite(p.grad).all().item() for p in gradients):
            optimizer.zero_grad(set_to_none=True)
            statuses[index] = "nonfinite_grad"
            skipped[index] = 1
            streaks[index] += 1
            continue
        statuses[index] = "ready"
        ready.append(index)

    ready_after_clip = []
    for index in ready:
        params = optimizer_parameters[index]
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in params):
            optimizers[index].zero_grad(set_to_none=True)
            statuses[index] = "nonfinite_grad"
            skipped[index] = 1
            streaks[index] += 1
        else:
            ready_after_clip.append(index)
    ready = ready_after_clip

    limit = int(args.max_consecutive_nan_grad_updates)
    failed_index = next(
        (index for index in sorted(active)
         if statuses[index] in {"missing_grad", "nonfinite_grad"} and streaks[index] >= limit),
        None,
    )
    if failed_index is not None:
        for index in ready:
            statuses[index] = "not_applied_threshold"
            skipped[index] = 1
        for index in active:
            optimizers[index].zero_grad(set_to_none=True)
        if len(optimizers) == 1:
            owner = "optimizer 0 (shared/full)"
        else:
            owner = f"optimizer {failed_index} (block {failed_index + 1})"
        return {
            "status": statuses,
            "skipped": skipped,
            "consecutive": streaks,
            "stop_error": (
                f"Consecutive invalid or missing SNF gradients reached "
                f"{streaks[failed_index]}/{limit} for {owner}; stopping before the next epoch."
            ),
        }

    for index in ready:
        optimizer, scheduler = optimizers[index], schedulers[index]
        params = optimizer_parameters[index]
        old_parameters = {id(p): p.detach().clone() for p in params}
        optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in params):
            for parameter in params:
                parameter.data.copy_(old_parameters[id(parameter)])
            raise RuntimeError("Non-finite SNF parameters after optimizer step; stopping before saving corrupted weights")
        scheduler.step(float(losses[index]))
        streaks[index] = 0
        statuses[index] = "updated"
    return {
        "status": statuses,
        "skipped": skipped,
        "consecutive": streaks,
        "stop_error": None,
    }


def _advance_persistent_prior(model, prior_state: torch.Tensor, steps: int) -> torch.Tensor:
    """Advance a local prior chain without exposing its storage to target work."""
    with torch.no_grad():
        return model.advance_prior(prior_state.detach().clone(), int(steps)).detach()


def _main_impl(args: argparse.Namespace, runtime) -> None:
    rank, world_size, device, backend, _, _owns_process_group = runtime
    start, end, fixed_beta, parameter_name = _protocol_endpoints(args)
    output = _resolve_output_dir(args, rank, world_size)
    local_batch = args.batch_size // world_size
    protocol_box = [None, ""]
    if rank == 0:
        try:
            protocol_box[0] = _protocol_values(args, start, end)
        except Exception as exc:
            protocol_box[1] = f"protocol file validation failed: {exc}"
    dist.broadcast_object_list(protocol_box, src=0)
    if protocol_box[1]:
        raise RuntimeError(protocol_box[1])
    protocol = protocol_box[0]

    # FlowPars.nstep describes NF transitions, whereas ``protocol`` also has
    # the terminal point used by the standalone driver.
    flow_pars, defect_mask = _build_flow_pars(args, device, protocol[:-1], local_batch)
    flow_pars._standalone_beta = fixed_beta
    checkpoint_semantics = _checkpoint_semantics(args, protocol)

    # Cache hit/miss must not affect model initialization or the chain RNG.
    model = StandaloneProtocol(args, flow_pars, defect_mask, start, end, protocol).to(device)
    input_checkpoint_path = None
    run_semantics = _resume_semantics(args, checkpoint_semantics, world_size)
    checkpoint = None
    if args.resume:
        checkpoint = _load_checkpoint_coordinated(
            model if args.algorithm == "snf" else None,
            output / "snf_checkpoint.pt", checkpoint_semantics, device,
            expected_run_semantics=run_semantics,
        )
        if len(checkpoint.get("rank_states", [])) != world_size:
            raise RuntimeError("resume checkpoint rank state does not match current world size")
    elif args.checkpoint:
        input_checkpoint_path = Path(args.checkpoint)
        if not input_checkpoint_path.is_absolute():
            input_checkpoint_path = Path(args.main_dir) / input_checkpoint_path
        checkpoint = _load_checkpoint_coordinated(
            model if args.algorithm == "snf" else None,
            input_checkpoint_path, checkpoint_semantics, device,
            transfer_rho_functional=args.transfer_rho_functional,
        )
    elif args.algorithm == "snf":
        _broadcast_parameters(model)

    training_epochs_per_block = args.epochs_per_block or args.train_steps or args.steps
    n_steps = (
        (training_epochs_per_block if args.training_mode == "progressive" else (args.train_steps if args.train_steps is not None else args.steps))
        * (args.protocol_steps if args.training_mode == "progressive" else 1)
        if args.phase == "train"
        else (int(math.ceil(float(args.eval_samples) / float(args.batch_size))) if args.eval_samples is not None
              else (args.eval_steps if args.eval_steps is not None else args.steps))
    )
    if args.resume:
        validation_error = ""
        try:
            _validate_resume_checkpoint(checkpoint, args, local_batch, world_size, device, n_steps)
        except Exception as exc:
            validation_error = f"resume checkpoint validation failed: {exc}"
        _propagate_rank_error(validation_error, device)
    if args.load_prior:
        validation_error = ""
        try:
            _validate_checkpoint_prior(checkpoint, args, local_batch, world_size, device)
        except Exception as exc:
            validation_error = f"checkpoint prior validation failed: {exc}"
        _propagate_rank_error(validation_error, device)
    resume_rank_state = checkpoint["rank_states"][rank] if args.resume or args.load_prior else None

    optimizers, schedulers = ([], [])
    if args.phase == "train" and args.algorithm == "snf":
        optimizers, schedulers = _make_optimizers(args, model)
        if args.resume:
            optimizer_error = ""
            try:
                _restore_optimizer_states(checkpoint, optimizers, schedulers)
            except Exception as exc:
                optimizer_error = f"resume optimizer restore failed: {exc}"
            _propagate_rank_error(optimizer_error, device)
        elif checkpoint is not None and not args.reset_optimizer and not args.transfer_rho_functional:
            optimizer_error = ""
            try:
                saved_optimizers = checkpoint.get("optimizer_state_dicts", [])
                saved_schedulers = checkpoint.get("scheduler_state_dicts", [])
                if saved_optimizers:
                    if len(saved_optimizers) != len(optimizers) or len(saved_schedulers) != len(schedulers):
                        raise RuntimeError("checkpoint optimizer count does not match this training mode")
                    for optimizer, state in zip(optimizers, saved_optimizers):
                        optimizer.load_state_dict(state)
                    for scheduler, state in zip(schedulers, saved_schedulers):
                        scheduler.load_state_dict(state)
            except Exception as exc:
                optimizer_error = f"checkpoint optimizer restore failed: {exc}"
            _propagate_rank_error(optimizer_error, device)
    if args.resume:
        optimizer_streaks, total_skipped_updates = _resume_gradient_safety(
            checkpoint, args, len(optimizers)
        )
    else:
        optimizer_streaks = [0] * len(optimizers)
        total_skipped_updates = 0

    if not args.resume:
        _clear_managed_output_artifacts(output, rank, input_checkpoint_path)

    if args.resume or args.load_prior:
        saved_prior = resume_rank_state["prior_state"]
        prior_state = torch.view_as_complex(saved_prior.contiguous()).to(device=device)
        cache_status = checkpoint.get("run_state", {}).get("cache_status", "resumed" if args.resume else "loaded_checkpoint")
    else:
        # prepare_prior owns a dedicated thermal RNG only on cache misses and
        # leaves the normal model/protocol stream untouched on both paths.
        prior_cfgs, cache_status = prepare_prior(args, flow_pars, defect_mask, start, end, parameter_name, device)
        prior_state = prior_cfgs.detach().clone()

    metadata = {
        "algorithm": args.algorithm, "phase": args.phase, "domain": args.domain,
        "parameter_name": parameter_name, "protocol": protocol, "global_batch": args.batch_size,
        "world_size": world_size, "backend": backend, "device": str(device),
        "nf_architecture": args.nf_architecture,
        "nf_architecture_alias": "residual_static_coefficients" if args.nf_architecture == "residual" else ("residual_conditional_existing" if args.nf_architecture == "hyper-residual" else None),
        "nf_layer_sharing": _nf_layer_sharing(args),
        "steps_semantics": "train_epochs" if args.phase == "train" else "evaluation_trajectories",
        "prior_mcmc_steps": int(args.prior_mcmc_steps),
        "prior_chain": "persistent_local_per_rank_from_cached_thermalized_batch",
        "thermal_rng_version": THERMAL_RNG_VERSION,
        "thermal_seed_offset": THERMAL_SEED_OFFSET,
        "work_sem_method": "pooled_iid",
        "cfg_cache_status": cache_status, "checkpoint_semantics": checkpoint_semantics,
        "run_semantics": run_semantics,
        "gradient_safety": _gradient_safety_state(
            args, len(optimizers), optimizer_streaks, total_skipped_updates
        ),
        "args": vars(args),
    }
    if args.resume:
        resume_error = ""
        if rank == 0:
            try:
                with open(output / "run_meta.json", encoding="utf-8") as handle:
                    old_metadata = json.load(handle)
                if not _run_semantics_match(old_metadata.get("run_semantics"), run_semantics):
                    raise ValueError("run metadata does not match resume semantics")
            except Exception as exc:
                resume_error = f"resume metadata validation failed: {exc}"
        _propagate_rank0_io(resume_error)

    evaluation_requested_samples = args.eval_samples if args.phase == "evaluate" else None
    evaluation_generated_samples = n_steps * args.batch_size if args.phase == "evaluate" else None
    metadata["evaluation_requested_samples"] = evaluation_requested_samples
    metadata["evaluation_generated_samples"] = evaluation_generated_samples
    metadata["evaluation_trajectory_count"] = n_steps if args.phase == "evaluate" else None
    metadata["training_epochs_per_block"] = training_epochs_per_block if args.training_mode == "progressive" and args.phase == "train" else None
    metadata["training_total_epochs"] = n_steps if args.phase == "train" else None

    csv_error = ""
    csv_rows = [0, 0]
    if rank == 0:
        try:
            if args.resume:
                _validate_resume_csv(
                    output / "metrics.csv", METRICS_HEADER, int(checkpoint.get("metric_rows", 0)),
                    legacy_header=LEGACY_METRICS_HEADER,
                )
                _validate_resume_csv(output / "observables.csv", OBSERVABLES_HEADER, int(checkpoint.get("observable_rows", 0)))
                csv_rows = [
                    _prepare_csv(
                        output / "metrics.csv", METRICS_HEADER, True, int(checkpoint.get("metric_rows", 0)),
                        legacy_header=LEGACY_METRICS_HEADER,
                        legacy_skipped_update="0" if args.phase == "evaluate" else "",
                    ),
                    _prepare_csv(output / "observables.csv", OBSERVABLES_HEADER, True, int(checkpoint.get("observable_rows", 0))),
                ]
            else:
                _prepare_csv(output / "metrics.csv", METRICS_HEADER, False)
                _prepare_csv(output / "observables.csv", OBSERVABLES_HEADER, False)
        except Exception as exc:
            csv_error = f"output CSV preparation failed: {exc}"
    status = [csv_rows, csv_error]
    dist.broadcast_object_list(status, src=0)
    if status[1]:
        raise RuntimeError(status[1])
    metric_rows, observable_rows = status[0]

    metadata_error = ""
    if rank == 0:
        try:
            _atomic_json_save(metadata, output / "run_meta.json")
        except Exception as exc:
            metadata_error = f"run metadata write failed: {exc}"
    _propagate_rank0_io(metadata_error)
    wandb_run = _wandb_start(args, rank, world_size, output, metadata)
    metrics_path = output / "metrics.csv"
    progress = int(checkpoint.get("progress", 0)) if args.resume else 0
    prior_elapsed = float(checkpoint.get("run_state", {}).get("elapsed_seconds", 0.0)) if args.resume else 0.0
    last_metrics = checkpoint.get("run_state", {}).get("last_metrics", {"work_mean": None, "work_sem": None, "deltaF": None, "ess_fraction": None, "finite_count": 0}) if args.resume else {"work_mean": None, "work_sem": None, "deltaF": None, "ess_fraction": None, "finite_count": 0}
    last_loss = checkpoint.get("run_state", {}).get("last_loss") if args.resume else None
    saw_finite_work = bool(checkpoint.get("run_state", {}).get("saw_finite_work", False)) if args.resume else False
    evaluation_work_local = checkpoint["rank_states"][rank].get("evaluation_work", []) if args.resume else []
    if args.resume or args.load_prior:
        _restore_checkpoint_rng_state(resume_rank_state["rng_state"], device)
    metric_rows = int(metric_rows)
    observable_rows = int(observable_rows)
    t0 = time.time()

    for step in range(progress, n_steps):
        work_batches = []
        block_losses = [None] * (args.protocol_steps if args.training_mode != "full" else 1)
        block_skips = [0] * len(block_losses)
        loss_accum = [0.0] * len(block_losses)
        loss_counts = [0] * len(block_losses)

        if args.phase == "train":
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            active_block = (step // training_epochs_per_block) if args.training_mode == "progressive" else None
            for _ in range(args.grad_acc):
                prior_state = _advance_persistent_prior(model, prior_state, args.prior_mcmc_steps)
                cfgs = prior_state.detach().clone()
                if args.training_mode != "full":
                    with torch.no_grad():
                        s0 = _action(flow_pars, cfgs, args.domain, start, defect_mask)
                if args.training_mode == "full":
                    _, train_work, _ = model(cfgs, train=True)
                    value, skipped = _backprop_local_loss(train_work, args.grad_acc, world_size, device)
                    if value is not None:
                        loss_accum[0] += value
                        loss_counts[0] += 1
                    block_losses[0] = (block_losses[0] or 0.0) + (value or 0.0) / args.grad_acc
                    block_skips[0] = max(block_skips[0], skipped)
                    work_batches.append(train_work.detach())
                elif args.training_mode == "progressive":
                    _, train_work, _, _, _ = model.up_to_block_nograd(cfgs, s0, active_block)
                    value, skipped = _backprop_local_loss(train_work, args.grad_acc, world_size, device)
                    if value is not None:
                        loss_accum[active_block] += value
                        loss_counts[active_block] += 1
                    block_losses[active_block] = (block_losses[active_block] or 0.0) + (value or 0.0) / args.grad_acc
                    block_skips[active_block] = max(block_skips[active_block], skipped)
                    work_batches.append(train_work.detach())
                else:
                    _, train_work, values, skips = model.blockwise_training_forward(
                        cfgs, s0, args.grad_acc, world_size, device
                    )
                    for block, (value, skipped) in enumerate(zip(values, skips)):
                        if value is not None:
                            block_losses[block] = (block_losses[block] or 0.0) + value / args.grad_acc
                            loss_counts[block] += 1
                            loss_accum[block] += value
                        block_skips[block] = max(block_skips[block], skipped)
                    work_batches.append(train_work.detach())

            for index in range(len(block_losses)):
                if loss_counts[index]:
                    block_losses[index] = loss_accum[index] / loss_counts[index]
            if args.training_mode == "progressive":
                optimizer_losses = [None] * len(optimizers)
                active_optimizer = 0 if len(optimizers) == 1 else active_block
                optimizer_losses[active_optimizer] = loss_accum[active_block] / max(1, loss_counts[active_block]) if loss_counts[active_block] else None
                active_indices = [active_optimizer]
            elif args.training_mode == "blockwise" and len(optimizers) == 1:
                values = [value for value in block_losses if value is not None]
                optimizer_losses = [sum(values) / len(values) if values else None]
                active_indices = None
            else:
                optimizer_losses = [loss_accum[i] / loss_counts[i] if loss_counts[i] else None for i in range(len(optimizers))]
                active_indices = None
            optimizer_result = _finish_optimizer_step(
                optimizers, schedulers, model, optimizer_losses, args, active_indices,
                optimizer_streaks,
            )
            optimizer_streaks = optimizer_result["consecutive"]
            optimizer_statuses = optimizer_result["status"]
            optimizer_skips = optimizer_result["skipped"]
            total_skipped_updates += sum(optimizer_skips)
            if optimizer_result["stop_error"]:
                raise FloatingPointError(optimizer_result["stop_error"])
            loss_value = sum(value for value in optimizer_losses if value is not None) / max(1, sum(value is not None for value in optimizer_losses))
            skipped_update = int(any(optimizer_skips))
            exact_work = torch.cat(work_batches, dim=0)
        else:
            prior_state = _advance_persistent_prior(model, prior_state, args.prior_mcmc_steps)
            cfgs = prior_state.detach().clone()
            with torch.no_grad():
                _, _, exact_work, int_obs = model(
                    cfgs, train=False, int_meas=args.int_meas, return_observables=True
                )
            work_batches.append(exact_work.detach())
            loss_value = None
            skipped_update = 0
            optimizer_statuses = []
            optimizer_skips = []
            evaluation_work_local.append(exact_work.detach().to(device="cpu", dtype=torch.float64))
            gathered_obs = [None] * world_size if rank == 0 else None
            dist.gather_object(int_obs.detach().cpu(), gathered_obs, dst=0)
            obs_error = ""
            if rank == 0:
                try:
                    observable_rows += _write_observables(
                        output / "observables.csv", step, torch.cat(gathered_obs, dim=0)
                    )
                except Exception as exc:
                    obs_error = f"observable output write failed: {exc}"
            _propagate_rank0_io(obs_error)

        last_metrics = _global_work_metrics(torch.cat(work_batches, dim=0))
        saw_finite_work = saw_finite_work or last_metrics["finite_count"] > 0
        last_loss = loss_value
        step_io_error = ""
        if rank == 0 and (step + 1) % args.log_every == 0:
            try:
                elapsed = prior_elapsed + time.time() - t0
                print(
                    f"step={step + 1}/{n_steps} work_mean={last_metrics['work_mean']} "
                    f"work_sem={last_metrics['work_sem']} deltaF={last_metrics['deltaF']} "
                    f"ess_fraction={last_metrics['ess_fraction']} finite={last_metrics['finite_count']} "
                    f"skipped_update={skipped_update} elapsed_s={elapsed:.2f}", flush=True,
                )
                with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow([
                        step + 1, last_metrics["work_mean"], last_metrics["work_sem"],
                        last_metrics["deltaF"], last_metrics["ess_fraction"], last_metrics["finite_count"],
                        loss_value, json.dumps(block_losses), json.dumps(block_skips),
                        json.dumps(optimizer_statuses), json.dumps(optimizer_skips),
                        json.dumps(optimizer_streaks), skipped_update, cache_status,
                    ])
                metric_rows += 1
                if wandb_run is not None:
                    values = {"epoch": step + 1, "loss/epoch_mean": loss_value, "training/skipped_update": skipped_update}
                    values.update({f"loss/block_{index}": value for index, value in enumerate(block_losses) if value is not None})
                    values.update({f"work/{key}": value for key, value in last_metrics.items() if value is not None})
                    wandb_run.log({key: value for key, value in values.items() if value is not None})
            except Exception as exc:
                step_io_error = f"step metrics/W&B write failed: {exc}"
        _propagate_rank0_io(step_io_error)

        progress = step + 1
        if args.checkpoint_every and progress % args.checkpoint_every == 0:
            local_state = {
                "prior_state": torch.view_as_real(prior_state.detach().contiguous()).cpu(),
                "rng_state": _checkpoint_rng_state(device),
                "evaluation_work": evaluation_work_local,
            }
            run_state = {
                "last_metrics": last_metrics, "last_loss": last_loss,
                "total_skipped_updates": total_skipped_updates, "saw_finite_work": saw_finite_work,
                "gradient_safety": _gradient_safety_state(
                    args, len(optimizers), optimizer_streaks, total_skipped_updates
                ),
                "cache_status": cache_status,
                "elapsed_seconds": prior_elapsed + time.time() - t0,
            }
            _save_checkpoint(output / "snf_checkpoint.pt", model if args.algorithm == "snf" else None,
                             optimizers, schedulers, checkpoint_semantics, run_semantics,
                             progress, metric_rows, observable_rows, local_state, run_state)
            metadata_error = ""
            if rank == 0:
                try:
                    metadata["checkpoint"] = str(output / "snf_checkpoint.pt")
                    metadata["checkpoint_progress"] = progress
                    metadata["gradient_safety"] = run_state["gradient_safety"]
                    _atomic_json_save(metadata, output / "run_meta.json")
                except Exception as exc:
                    metadata_error = f"run metadata update failed: {exc}"
            _propagate_rank0_io(metadata_error)

    pooled_metrics = None
    if args.phase == "evaluate" and evaluation_work_local:
        pooled_local_work = torch.cat(evaluation_work_local, dim=0).to(device=device, dtype=torch.float64)
        pooled_metrics = _global_work_metrics(pooled_local_work)

    no_finite_work = not saw_finite_work
    final_io_error = ""
    if not no_finite_work:
        local_state = {
            "prior_state": torch.view_as_real(prior_state.detach().contiguous()).cpu(),
            "rng_state": _checkpoint_rng_state(device),
            "evaluation_work": evaluation_work_local,
        }
        run_state = {
            "last_metrics": last_metrics, "last_loss": last_loss,
            "total_skipped_updates": total_skipped_updates, "saw_finite_work": saw_finite_work,
            "gradient_safety": _gradient_safety_state(
                args, len(optimizers), optimizer_streaks, total_skipped_updates
            ),
            "cache_status": cache_status,
            "elapsed_seconds": prior_elapsed + time.time() - t0,
        }
        _save_checkpoint(output / "snf_checkpoint.pt", model if args.algorithm == "snf" else None,
                         optimizers, schedulers, checkpoint_semantics, run_semantics,
                         progress, metric_rows, observable_rows, local_state, run_state)
    if rank == 0:
        try:
            if args.algorithm == "snf" and not no_finite_work:
                metadata["checkpoint"] = str(output / "snf_checkpoint.pt")
                metadata["checkpoint_progress"] = progress
            metadata["gradient_safety"] = _gradient_safety_state(
                args, len(optimizers), optimizer_streaks, total_skipped_updates
            )
            gamma_result = None
            if args.phase == "evaluate" and not no_finite_work:
                gamma_result = analyze_observables_csv(output / "observables.csv")
                metadata["gamma_analysis"] = gamma_result
            metadata["summary"] = {
                **last_metrics, "loss": last_loss, "skipped_updates": total_skipped_updates,
                "gradient_safety": metadata["gradient_safety"],
                "pooled_work_mean": pooled_metrics["work_mean"] if pooled_metrics is not None else None,
                "pooled_work_sem": pooled_metrics["work_sem"] if pooled_metrics is not None else None,
                "pooled_work_sem_method": "pooled_iid",
                "work_sem_method": "pooled_iid",
                "gamma_analysis_status": gamma_result["status"] if gamma_result is not None else None,
                "gamma_analysis_file": gamma_result["path"] if gamma_result is not None else None,
                "work_sem_gamma_final": gamma_result["work_sem_gamma_final"] if gamma_result is not None else None,
                "pooled_deltaF": pooled_metrics["deltaF"] if pooled_metrics is not None else None,
                "pooled_ess_fraction": pooled_metrics["ess_fraction"] if pooled_metrics is not None else None,
                "pooled_finite_count": pooled_metrics["finite_count"] if pooled_metrics is not None else None,
                "evaluation_requested_samples": evaluation_requested_samples,
                "evaluation_generated_samples": evaluation_generated_samples,
                "evaluation_trajectory_count": n_steps if args.phase == "evaluate" else None,
                "status": "error" if no_finite_work else "ok",
                "error": "no finite work values" if no_finite_work else None,
                "elapsed_seconds": prior_elapsed + time.time() - t0,
            }
            _atomic_json_save(metadata["summary"], output / "summary.json")
            _atomic_json_save(metadata, output / "run_meta.json")
            if wandb_run is not None:
                wandb_run.summary.update({key: value for key, value in metadata["summary"].items() if value is not None})
        except Exception as exc:
            final_io_error = f"final run artifact write failed: {exc}"
    _propagate_rank0_io(final_io_error)
    dist.barrier()
    if no_finite_work:
        raise RuntimeError("No finite work values were produced; run failed without writing a checkpoint")


_ACTIVE_WANDB_RUN = None


def main(args: argparse.Namespace) -> None:
    global _ACTIVE_WANDB_RUN
    validate_args(args)
    _ACTIVE_WANDB_RUN = None
    runtime = None
    try:
        runtime = setup_runtime(args)
        return _main_impl(args, runtime)
    finally:
        if _ACTIVE_WANDB_RUN is not None:
            try:
                _ACTIVE_WANDB_RUN.finish()
            except Exception:
                pass
            _ACTIVE_WANDB_RUN = None
        if runtime is not None and runtime[-1] and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
