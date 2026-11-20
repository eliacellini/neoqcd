"""NE-MCMC and SNF driver for N_f = 2 Wilson fermions (fermions_plan.md, phase 4 steps 3-4).

A protocol lambda_0 -> lambda_1 -> ... -> lambda_n in one parameter of the joint action
S(U, phi; lambda) = S_g(U; beta, bc) + S_pf(U, phi; kappa) (`neoqcd.fermions.action.QCDAction`):
    --protocol beta|kappa|bc --initial lambda_0 --final lambda_n --n-step n   (linear), or
    --protocol ... --initial lambda_0 --protocol-file f                       (lambda_1 .. lambda_n).
--mode nemcmc: `snf.FermionNEMCMC` (HMC blocks only, --layers none);
--mode snf:    `snf_layers.FermionSNF` (a pure-gauge layer before every HMC block, --layers
               smearing|smearing_signed|residual|hyper-residual, trained on mean(W) with Adam for
               --train-steps steps; smearing_signed --rho-init 0 starts at the identity and trains,
               --lr-warmup ramps the learning rate). Adam's first steps move every parameter by
               about --lr, so --lr should be well below --rho-init.
Residual layers (`ResidualCouplingLayer`, pure-gauge spelling of main_nemcmc_snf.py
--nf-architecture): --layers residual (static coefficients) or hyper-residual (coefficients from a
hyper-network of the conditioning value), --residual-n-pt, --residual-mask-mod,
--residual-[no-]include-imag, --residual-[no-]quadratic, --residual-coeff-init, --residual-coeff-max,
--hyper-time-embedding-dim, --hyper-hidden-dim, --hyper-depth, --hyper-activation (same names,
defaults and validation as the pure-gauge driver); --residual-condition time|parameter and
--residual-delta fraction|step choose what the layers receive as beta and delta_beta
(`snf_layers.layer_conditioning`; default time/fraction: the same scale for every protocol
parameter, see doc/fermions.md). The architecture is in run_meta.json, in the resume key and in
the flow.json sidecar: --load-flow refuses a flow of another architecture.
--fast (Wilson): `QCDAction(fast=True)`, the half-lattice operator and lean CG of
`neoqcd.fermions.fast` (--precision, --colour-product, --check-every, --inner-tol, --max-outer as
in main_hmc.py); same physics to the solver tolerance. The fast settings are in run_meta.json and
in the resume key (a run is resumed only on the path it started on); pool/flow files have the same
format on both paths.
Each HMC block runs --n-traj fermionic HMC trajectories (fresh phi at every trajectory) at lambda_i.

Starting configurations. A pool of --batch chains in equilibrium at lambda_0: from --start-config
(a checkpoint of main/main_hmc.py at the same couplings, first --batch configurations) or from a
cold start, then --therm HMC trajectories at lambda_0. Before every training step and every
evaluation batch the pool advances by --n-between trajectories at lambda_0, and the flow starts
from (a copy of) the pool: consecutive batches are correlated through the pool's autocorrelation,
as in the pure-gauge driver (main_nemcmc_snf.py).

Defect (needed for --protocol bc or --bc != 1): --defect-shape cube is `utils.Defect` (a cube of
side --defect-size in the spatial directions at t = --defect-time-slice, dsize^(D-1) time links);
--defect-shape line is --defect-size time links at t = time slice along direction 1 (other
spatial coordinates = --defect-space-slice), e.g. the 2-link defect of the tests.

Output directory (--out, default output/snf_fermions/<auto name>):
    samples.csv     one row per evaluated sample (SAMPLE_FIELDS);
    blocks.csv      one row per (evaluation batch, protocol step) (BLOCK_FIELDS);
    training.csv    one row per training step (TRAIN_FIELDS), --mode snf only; param_abs_mean is
                    the mean |parameter| the step's flow ran with (before its optimizer step);
    run_meta.json   parameters + summary: Jarzynski dF = -log <exp(-W)> = -log Z_n/Z_0 with its
                    jackknife error, ESS = <e^-W>^2/<e^-2W>, mean work/heat/logJ, acceptance per
                    block, solves and CG iterations per sample, seconds per sample;
    flow.pt / flow.json   (snf) {"state_dict": FermionSNF.state_dict(), "optimizer", "train_step"}
                    and a sidecar with geometry, protocol, couplings and layer settings;
    pool.pt / pool.json   pool links + run generator (`neoqcd.fermions.io.save_state`), with
                    extra = {"phase": "train"|"eval"|"done", "progress": steps/batches done}.
Cost columns: `solves` counts the CG solves of the flow for one batch; a batched solve serves the B
configurations, each of which needs its own, so `solves` is also the number of solves per sample.
`pool_solves` are those of the --n-between decorrelation trajectories (per sample likewise).

Resume: --resume reads pool.* (and flow.pt), checks the physics parameters, drops CSV rows past
the checkpoint and continues training or evaluation where the checkpoint left it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main_hmc import _sync, choose_device  # noqa: E402

from neoqcd.fermions._gauge import Defect, FlowPars, SUN_identity, create_mask  # noqa: E402
from neoqcd.fermions.action import QCDAction  # noqa: E402
from neoqcd.fermions.fast import resolve_colour_product  # noqa: E402
from neoqcd.fermions.hmc import HMC, INTEGRATORS  # noqa: E402
from neoqcd.fermions.io import (  # noqa: E402
    ObservableWriter,
    _atomic_write,
    atomic_json_save,
    load_state,
    read_observables,
    restore_generator,
    save_state,
)
from neoqcd.fermions.lattice import Geometry  # noqa: E402
from neoqcd.fermions.snf import (  # noqa: E402
    FermionNEMCMC,
    FieldState,
    effective_sample_size,
    jarzynski_free_energy,
)
from neoqcd.fermions.snf_layers import (  # noqa: E402
    CONDITION_MODES,
    DELTA_MODES,
    HYPER_ACTIVATIONS,
    LAYER_KINDS,
    RESIDUAL_KINDS,
    RESIDUAL_OPTIONS,
    FermionSNF,
    default_sharing,
    layer_diagnostics_message,
    make_gauge_layers,
    residual_options,
)

SAMPLE_FIELDS = (
    "batch",         # evaluation batch index
    "replica",       # entry b of the batch
    "sample",        # global sample index = batch * B + replica
    "work",          # W of the run (jumps + layers), Jarzynski work
    "heat",          # Q of the run (HMC blocks)
    "logJ",          # sum of the layer log-Jacobians (0 for nemcmc)
    "S0",            # S(x_0; lambda_0)
    "S_final",       # S(x_n; lambda_n)
    "plaquette",     # mean plaquette of the final configuration
    "solves",        # CG solves of the flow in this batch (= solves per sample, see module doc)
    "cg_iter",       # sum over those solves of the largest per-sample iteration count
    "pool_solves",   # CG solves of the decorrelation trajectories before this batch
    "wall_s",        # wall time of the flow for this batch
    "pool_wall_s",   # wall time of the decorrelation trajectories before this batch
    "ess_batch",     # ESS of the works of this batch
    "ess_cumulative",  # ESS of all works up to this batch
)

BLOCK_FIELDS = (
    "batch", "block",  # protocol step i = 1 .. n (block 0 is the first step)
    "parameter",       # lambda_i
    "acceptance",      # HMC acceptance in block i (mean over its trajectories and the batch)
    "exp_mdH",         # <exp(-dH)> in block i
    "work_mean",       # running work after block i, batch mean
    "heat_mean",       # running heat after block i, batch mean
    "plaquette_mean",  # plaquette after block i, batch mean
)

TRAIN_FIELDS = (
    "step", "loss", "work_mean", "work_std", "ess", "heat_mean", "logJ_mean", "acceptance",
    "grad_norm", "solves", "cg_iter", "wall_s",
    "lr",              # learning rate of this step (after the --lr-warmup factor)
    "param_abs_mean",  # mean |parameter| the flow of this step ran with (BEFORE its optimizer step)
    "max_abs_coeff",   # max |rho| (stout) or |coefficient| (residual) of that flow (snf_layers.layer_diagnostics)
    "injectivity_ratio",  # max |rho| / (1/(2(D-1))) (stout) or the residual contraction estimate; >= 1: no
                          # bijectivity guarantee (one RuntimeWarning per run)
)

POOL_NAME = "pool"
FLOW_NAME = "flow"


# ---- arguments ----------------------------------------------------------------------------------
def add_physics_args(p):
    p.add_argument("--D", type=int, default=4)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--beta", type=float, default=5.6)
    p.add_argument("--fermions", choices=("wilson", "staggered"), default="wilson",
                   help="wilson: N_f = 2 Wilson (--kappa); staggered: one staggered field, N_f = 2^(D/2) (--mass).")
    p.add_argument("--kappa", type=float, default=None,
                   help="Wilson hopping parameter (--fermions wilson only; default 0.153).")
    p.add_argument("--mass", type=float, default=None, help="Staggered bare mass (required with --fermions staggered).")
    p.add_argument("--time-sign", type=float, default=-1.0, choices=(-1.0, 1.0))
    p.add_argument("--solver-tol", type=float, default=1e-10, help="CG relative residual (force solves).")
    p.add_argument("--action-tol", type=float, default=None, help="CG relative residual (action solves).")
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--no-reuse-guess", dest="reuse_guess", action="store_false")
    p.add_argument("--fast", action="store_true",
                   help="Wilson only: half-lattice operator and lean CG (neoqcd.fermions.fast, "
                        "QCDAction(fast=True)); same physics. Part of the resume key.")
    p.add_argument("--precision", choices=("double", "mixed"), default="double",
                   help="With --fast: 'mixed' uses complex64 inner CG with complex128 residuals.")
    p.add_argument("--colour-product", choices=("auto", "matmul", "broadcast", "broadcast_sum"), default=None,
                   help="With --fast: SU(N) x spinor product; auto (default) = broadcast_sum on CUDA, matmul on CPU.")
    p.add_argument("--check-every", type=int, default=None,
                   help="With --fast: host convergence check of the CG every k iterations (default 10).")
    p.add_argument("--inner-tol", type=float, default=None,
                   help="With --precision mixed: relative tolerance of the complex64 inner solves (default 1e-6).")
    p.add_argument("--max-outer", type=int, default=None,
                   help="With --precision mixed: maximum outer (defect-correction) steps per solve (default 30).")
    p.add_argument("--on-unconverged", choices=("raise", "warn", "ignore"), default=None,
                   help="Wilson only: an unconverged CG solve raises (default), warns once per kind, or is "
                        "only counted (neoqcd.fermions.pseudofermion, convergence policy).")
    p.add_argument("--tau", type=float, default=1.0, help="HMC trajectory length.")
    p.add_argument("--n-steps", type=int, default=6, help="Outer integrator steps per trajectory.")
    p.add_argument("--n-gauge-substeps", type=int, default=3)
    p.add_argument("--integrator", default="omelyan", help=f"One of {INTEGRATORS}, or 'outer,inner'.")
    p.add_argument("--defect-size", type=int, default=2)
    p.add_argument("--defect-shape", choices=("cube", "line"), default="cube")
    p.add_argument("--defect-time-slice", type=int, default=0)
    p.add_argument("--defect-space-slice", type=int, default=0)
    p.add_argument("--start-config", type=str, default=None,
                   help="Checkpoint prefix of main/main_hmc.py (same couplings, batch >= this run's).")
    p.add_argument("--therm", type=int, default=0, help="HMC trajectories at the initial parameter before the run.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NE-MCMC / SNF with N_f = 2 Wilson fermions.")
    add_physics_args(p)
    p.add_argument("--bc", type=float, default=1.0, help="Defect coupling (fixed unless --protocol bc).")
    p.add_argument("--protocol", choices=("beta", "kappa", "mass", "bc"), required=True,
                   help="kappa: Wilson only; mass: staggered only.")
    p.add_argument("--initial", type=float, default=None, help="lambda_0 (default: --beta/--kappa/--mass/--bc).")
    p.add_argument("--final", type=float, default=None, help="lambda_n of the linear protocol.")
    p.add_argument("--n-step", type=int, default=None, help="Protocol steps n of the linear protocol.")
    p.add_argument("--protocol-file", type=str, default=None, help="Text file with lambda_1 .. lambda_n.")
    p.add_argument("--mode", choices=("nemcmc", "snf"), default="nemcmc")
    p.add_argument("--layers", choices=("none",) + LAYER_KINDS, default=None,
                   help="Default: none for nemcmc, smearing for snf. smearing_signed: stout with a signed "
                        "rho (snf_layers.SignedCouplingLayer), trainable from --rho-init 0 (the identity).")
    p.add_argument("--rho-init", type=float, default=1e-3,
                   help="Initial stout rho of every slot of every smearing layer (default 1e-3), the value "
                        "used by the first training step (param_abs_mean of training step 0). With --layers "
                        "smearing it must be nonzero for training: that layer uses |rho|, whose gradient at "
                        "rho = 0 is zero, so the layers would never move; use --layers smearing_signed "
                        "--rho-init 0 for a trainable identity start. Unused with --layers none or residual.")
    add_residual_args(p)
    p.add_argument("--load-flow", type=str, default=None,
                   help="flow.pt of a previous snf run (layer weights); its flow.json sidecar must record the "
                        "same layer architecture.")
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate. Adam's first steps move every parameter by about lr whatever "
                        "the gradient size, so lr should be well below --rho-init (or use --lr-warmup).")
    p.add_argument("--lr-warmup", type=int, default=0,
                   help="Linear learning-rate warm-up: lr * min(1, (step + 1) / k) at training step `step` "
                        "(0: off). Stateless in the step index, so a resume continues it exactly.")
    p.add_argument("--grad-clip", type=float, default=0.0, help="Clip the gradient norm (0: off).")
    p.add_argument("--batch", type=int, default=8, help="Configurations per flow run (training and evaluation).")
    p.add_argument("--eval-samples", type=int, default=64, help="Evaluated samples (rounded up to whole batches).")
    p.add_argument("--n-traj", type=int, default=1, help="HMC trajectories per protocol block.")
    p.add_argument("--n-between", type=int, default=1, help="Pool trajectories at lambda_0 between flow runs.")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint every k steps/batches (0: end only).")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)
    validate_args(args)
    return args


def add_residual_args(p):
    """The residual-layer options (--layers residual|hyper-residual), named as main_nemcmc_snf.py."""
    p.add_argument("--residual-include-imag", action="store_true", default=True)
    p.add_argument("--residual-no-include-imag", dest="residual_include_imag", action="store_false")
    p.add_argument("--residual-quadratic", action="store_true", default=True)
    p.add_argument("--residual-no-quadratic", dest="residual_quadratic", action="store_false")
    p.add_argument("--residual-coeff-init", type=float, default=None,
                   help="Initial linear coefficients (static) or hyper-network output bias (default 1e-3); "
                        "0 is the identity and a regular point (trainable).")
    p.add_argument("--residual-coeff-max", type=float, default=0.0, help="tanh bound of the coefficients (0: off).")
    p.add_argument("--residual-n-pt", type=int, default=0,
                   help="Gauge-covariant convolutions of frozen links per residual substep (0: original plaquettes).")
    p.add_argument("--residual-mask-mod", type=int, choices=(2, 4), default=2,
                   help="Colouring of the residual updates (T and L multiples of it if n-pt > 0 or mod 4).")
    p.add_argument("--hyper-time-embedding-dim", type=int, default=8)
    p.add_argument("--hyper-hidden-dim", type=int, default=16)
    p.add_argument("--hyper-depth", type=int, default=2)
    p.add_argument("--hyper-activation", choices=HYPER_ACTIVATIONS, default="silu")
    p.add_argument("--residual-condition", choices=CONDITION_MODES, default="time",
                   help="Conditioning value of the layers: time = normalised protocol position in (0, 1] "
                        "(default), parameter = lambda_i (pure-gauge convention).")
    p.add_argument("--layer-sharing", choices=("shared", "per_step"), default=None,
                   help="Residual layers: one layer for the whole protocol (shared) or one per step; default "
                        "shared for hyper-residual, per_step for residual (as main_nemcmc_snf.py).")
    p.add_argument("--residual-delta", choices=DELTA_MODES, default="fraction",
                   help="Step scale of the layers' generator: fraction = (lambda_i - lambda_{i-1}) / total "
                        "variation (default, 1/n for a linear protocol), step = pure-gauge forward difference.")


RESIDUAL_ARG_DEFAULTS = {"residual_include_imag": True, "residual_quadratic": True, "residual_coeff_init": None,
                         "residual_coeff_max": 0.0, "residual_n_pt": 0, "residual_mask_mod": 2,
                         "hyper_time_embedding_dim": 8, "hyper_hidden_dim": 16, "hyper_depth": 2,
                         "hyper_activation": "silu", "residual_condition": "time", "residual_delta": "fraction",
                         "layer_sharing": None}


def validate_residual_args(args):
    """args.residual: the `residual_options` dict (None unless --layers residual|hyper-residual);
    residual options with other layers are refused, as in main_nemcmc_snf.py."""
    args.residual = None
    if args.layers not in RESIDUAL_KINDS:
        changed = [k for k, v in RESIDUAL_ARG_DEFAULTS.items() if getattr(args, k) != v]
        if changed:
            raise SystemExit(f"--{changed[0].replace('_', '-')} requires --layers residual or hyper-residual")
        return
    if args.N != 3:
        raise SystemExit("the residual layers support SU(3) only (--N 3)")
    kw = {key: getattr(args, attr) for key, (attr, _) in RESIDUAL_OPTIONS.items() if hasattr(args, attr)}
    if kw.get("coeff_init") is None:
        kw.pop("coeff_init", None)
    try:
        args.residual = residual_options(args.layers, lattice=(args.T,) + (args.L,) * (args.D - 1), **kw)
    except ValueError as exc:
        raise SystemExit(f"residual options: {exc}")
    if args.layer_sharing is None:
        args.layer_sharing = default_sharing(args.layers)


def flow_architecture(args):
    """What a flow checkpoint must match to be loaded: layer kind, residual options, conditioning."""
    arch = {"layers": args.layers}
    if args.residual is not None:  # coeff_init is an initial value, not architecture
        arch["residual"] = {k: v for k, v in args.residual.items() if k != "coeff_init"}
        arch["sharing"] = args.layer_sharing
        arch["condition"] = args.residual_condition
        arch["delta"] = args.residual_delta
    return arch


def check_flow_sidecar(args, flow_pt):
    """Refuse a flow.pt whose sidecar records another architecture (or no sidecar at all)."""
    sidecar = Path(flow_pt).with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(f"--load-flow: no sidecar {sidecar} (the layer architecture cannot be checked)")
    meta = json.loads(sidecar.read_text())
    recorded = meta.get("architecture")
    if recorded is None:  # sidecars written before the residual options: layer kind only
        recorded = {"layers": meta.get("layers")}
        if meta.get("layers") in RESIDUAL_KINDS or meta.get("layers") != args.layers:
            raise SystemExit(f"--load-flow: {sidecar} records layers {meta.get('layers')!r} without the "
                             f"architecture; this run has {flow_architecture(args)}")
        return
    if recorded != clean_json(flow_architecture(args)):
        raise SystemExit(f"--load-flow: the flow of {sidecar} has architecture {recorded}, this run "
                         f"{clean_json(flow_architecture(args))}")


def learning_rate(args, step):
    """lr at training step `step` (0-based): linear warm-up over --lr-warmup steps, then --lr."""
    if args.lr_warmup <= 0:
        return float(args.lr)
    return float(args.lr) * min(1.0, (step + 1) / float(args.lr_warmup))


def parse_integrator(spec):
    levels = [s.strip().lower() for s in str(spec).split(",")]
    if len(levels) == 1:
        levels = levels * 2
    if len(levels) != 2 or any(s not in INTEGRATORS for s in levels):
        raise SystemExit(f"--integrator: one or two of {INTEGRATORS}")
    return levels


def validate_common(args):
    for name in ("D", "T", "L", "N", "n_steps", "n_gauge_substeps", "max_iter", "defect_size"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.N not in (2, 3):
        raise SystemExit("--N must be 2 or 3")
    if args.T % 2 or args.L % 2:
        raise SystemExit("even-odd pseudofermions need even T and L")
    if args.therm < 0:
        raise SystemExit("--therm must be >= 0")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite exclude each other")
    args.integrator_levels = parse_integrator(args.integrator)
    if args.fermions == "wilson":
        if args.mass is not None:
            raise SystemExit("--mass is for --fermions staggered (Wilson fermions take --kappa)")
        if args.kappa is None:
            args.kappa = 0.153
    else:
        if args.kappa is not None:
            raise SystemExit("--kappa is for --fermions wilson (staggered fermions take --mass)")
        if args.mass is None or args.mass <= 0:
            raise SystemExit("--fermions staggered needs --mass > 0")
    validate_fast(args)


def validate_fast(args):
    """The fast-path options, as main/main_hmc.py: --fast is Wilson only, its options need it,
    --inner-tol / --max-outer need --precision mixed; defaults filled in afterwards."""
    if args.fast and args.fermions != "wilson":
        raise SystemExit("--fast is for --fermions wilson")
    if args.precision != "double" and not args.fast:
        raise SystemExit("--precision mixed needs --fast")
    for name in ("colour_product", "check_every"):
        if getattr(args, name) is not None and not args.fast:
            raise SystemExit(f"--{name.replace('_', '-')} needs --fast")
    for name in ("inner_tol", "max_outer"):
        if getattr(args, name) is not None and args.precision != "mixed":
            raise SystemExit(f"--{name.replace('_', '-')} needs --fast --precision mixed")
    if args.on_unconverged is not None and args.fermions != "wilson":
        raise SystemExit("--on-unconverged is for --fermions wilson")
    args.colour_product = "auto" if args.colour_product is None else args.colour_product
    args.check_every = 10 if args.check_every is None else args.check_every
    args.inner_tol = 1e-6 if args.inner_tol is None else args.inner_tol
    args.max_outer = 30 if args.max_outer is None else args.max_outer
    if args.fermions == "wilson":  # the staggered term keeps its own behaviour (recorded as None)
        args.on_unconverged = "raise" if args.on_unconverged is None else args.on_unconverged
    if args.check_every < 1:
        raise SystemExit("--check-every must be >= 1")
    if args.max_outer < 1 or not args.inner_tol > 0:
        raise SystemExit("--max-outer must be >= 1 and --inner-tol > 0")


def fermion_parameter(args):
    """Name and value of the fermion coupling: ("kappa", kappa) or ("mass", mass)."""
    return ("kappa", args.kappa) if args.fermions == "wilson" else ("mass", args.mass)


def validate_args(args):
    validate_common(args)
    if args.protocol == "kappa" and args.fermions != "wilson":
        raise SystemExit("--protocol kappa needs --fermions wilson (use --protocol mass for staggered)")
    if args.protocol == "mass" and args.fermions != "staggered":
        raise SystemExit("--protocol mass needs --fermions staggered (use --protocol kappa for Wilson)")
    if args.initial is None:
        args.initial = getattr(args, args.protocol)
    if args.protocol_file is not None:
        if args.final is not None or args.n_step is not None:
            raise SystemExit("--protocol-file excludes --final and --n-step")
        try:
            values = np.atleast_1d(np.loadtxt(args.protocol_file, dtype=np.float64)).reshape(-1)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read --protocol-file {args.protocol_file}: {exc}")
        if values.size < 1 or not np.all(np.isfinite(values)):
            raise SystemExit("--protocol-file must hold finite values lambda_1 .. lambda_n")
        args.protocol_values = [float(v) for v in values]
    else:
        if args.final is None or args.n_step is None or args.n_step < 1:
            raise SystemExit("give --final and --n-step >= 1, or --protocol-file")
        n = int(args.n_step)
        args.protocol_values = [float(args.initial + (args.final - args.initial) * (i + 1) / n) for i in range(n)]
    args.n_step = len(args.protocol_values)
    args.final = args.protocol_values[-1]
    if args.layers is None:
        args.layers = "none" if args.mode == "nemcmc" else "smearing"
    if args.mode == "nemcmc" and args.layers != "none":
        raise SystemExit("--mode nemcmc has no layers (use --layers none or --mode snf)")
    if args.mode == "snf" and args.layers == "none":
        raise SystemExit("--mode snf needs --layers smearing, smearing_signed, residual or hyper-residual")
    if args.mode == "nemcmc" and (args.train_steps or args.load_flow):
        raise SystemExit("--train-steps / --load-flow need --mode snf")
    if args.layers == "smearing" and args.rho_init == 0.0 and args.train_steps:
        raise SystemExit("--layers smearing with --rho-init 0 cannot train (the gradient of |rho| vanishes at 0): "
                         "use --layers smearing_signed for an identity start")
    if args.lr_warmup < 0:
        raise SystemExit("--lr-warmup must be >= 0")
    for name in ("batch", "n_traj"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.train_steps < 0 or args.eval_samples < 0 or args.n_between < 0 or args.checkpoint_every < 0:
        raise SystemExit("--train-steps, --eval-samples, --n-between, --checkpoint-every must be >= 0")
    if args.eval_samples and args.eval_samples < 2:
        raise SystemExit("--eval-samples must be 0 or >= 2 (Jarzynski error)")
    uses_defect = args.protocol == "bc" or args.bc != 1.0
    args.uses_defect = uses_defect
    validate_residual_args(args)


def default_name(args):
    lat = f"{args.T}x{args.L}p{args.D - 1}"
    tag = f"{args.mode}" + ("" if args.mode == "nemcmc" else f"_{args.layers}")
    f = f"k{args.kappa:g}" if args.fermions == "wilson" else f"stag_m{args.mass:g}"
    return (f"{tag}_{args.protocol}{args.initial:g}to{args.final:g}_n{args.n_step}_b{args.beta:g}_{f}"
            f"_D{args.D}_{lat}_SU{args.N}_B{args.batch}_s{args.seed}")


def initial_couplings(args):
    """(beta_0, kappa_0 or mass_0, bc_0) at lambda_0."""
    name, value = fermion_parameter(args)
    c = {"beta": args.beta, name: value, "bc": getattr(args, "bc", 1.0)}
    c[args.protocol] = args.initial
    return float(c["beta"]), float(c[name]), float(c["bc"])


def physics_key(args):
    return {
        "geometry": {"D": args.D, "T": args.T, "L": args.L, "N": args.N},
        "batch": args.batch, "beta": float(args.beta), "kappa": None if args.kappa is None else float(args.kappa), "bc": float(args.bc),
        "time_sign": float(args.time_sign), "protocol": args.protocol, "initial": float(args.initial),
        "protocol_values": list(args.protocol_values), "mode": args.mode, "layers": args.layers,
        "defect": defect_key(args) if args.uses_defect else None,
        "n_traj": args.n_traj, "tau": float(args.tau), "n_steps": args.n_steps,
        "n_gauge_substeps": args.n_gauge_substeps, "integrator": list(args.integrator_levels),
        **staggered_key(args),
        **residual_key(args),
    }


def residual_key(args):
    """Resume-key entries of the residual layers (none for other layers: their keys unchanged)."""
    if getattr(args, "residual", None) is None:
        return {}
    arch = flow_architecture(args)
    return {"residual": dict(args.residual), "sharing": arch["sharing"], "condition": arch["condition"],
            "delta": arch["delta"]}


# ---- helpers shared with main_rex_fermions.py ---------------------------------------------------
def defect_key(args):
    return {"shape": args.defect_shape, "size": args.defect_size, "time_slice": args.defect_time_slice,
            "space_slice": args.defect_space_slice}


def build_defect_mask(args, geometry):
    """(2, D) + lattice float64 mask, layout of `utils.Defect` (module doc for the two shapes)."""
    if args.defect_shape == "cube":
        if args.defect_space_slice + args.defect_size > args.L or args.defect_time_slice >= args.T:
            raise SystemExit("defect does not fit in the lattice")
        d = Defect(args.D, args.T, args.L, args.defect_size, time_slice=args.defect_time_slice,
                   space_slice=args.defect_space_slice)
        return d.create_defect_mask().to(torch.float64)
    if args.D < 2 or args.defect_size > args.L or args.defect_time_slice >= args.T or args.defect_space_slice >= args.L:
        raise SystemExit("defect does not fit in the lattice")
    m = torch.zeros((geometry.D,) + geometry.lattice, dtype=torch.float64)
    idx = (0, args.defect_time_slice, slice(0, args.defect_size)) + (args.defect_space_slice,) * (geometry.D - 2)
    m[idx] = 1.0
    return torch.stack((1.0 - m, m), dim=0)


def staggered_key(args):
    """Extra physics-key entries for staggered runs (none for Wilson: keys of Wilson runs unchanged)
    and for --fast runs (`fast_key`)."""
    key = {"fermions": "staggered", "mass": float(args.mass)} if args.fermions == "staggered" else {}
    key.update(fast_key(args))
    return key


def fast_key(args):
    """Resume-key entries of the fast path: none without --fast (keys of reference runs unchanged,
    so their checkpoints still resume), else fast, precision, colour product, check interval (and
    the mixed-precision settings), so neither path resumes a checkpoint of the other: they agree
    only to the solver tolerance, not bitwise."""
    if not getattr(args, "fast", False):
        return {}
    key = {"fast": True, "precision": args.precision, "colour_product": args.colour_product,
           "check_every": int(args.check_every)}
    if args.precision == "mixed":
        key.update(inner_tol=float(args.inner_tol), max_outer=int(args.max_outer))
    return key


def fast_info(args, device):
    """run_meta.json "run" entries of the fermion operator (resolved colour product)."""
    if not getattr(args, "fast", False):
        return {"fermion_operator": "reference", "on_unconverged": getattr(args, "on_unconverged", None)}
    return {"fermion_operator": "half_lattice_fast", "precision": args.precision,
            "colour_product": resolve_colour_product(args.colour_product, device),
            "check_every": args.check_every, "on_unconverged": args.on_unconverged}


def action_kwargs(args):
    """Fermion type and coupling for QCDAction / FermionREX (kappa=None explicitly for staggered)."""
    if args.fermions == "wilson":
        return dict(kappa=args.kappa)
    return dict(kappa=None, fermions="staggered", mass=args.mass)


def state_couplings(args, fpar0):
    """save_state keywords: fermion type and kappa or mass."""
    if args.fermions == "wilson":
        return dict(fermions="wilson", kappa=fpar0)
    return dict(fermions="staggered", mass=fpar0)


def fermion_kwargs(args):
    """Keywords of the fermion term for QCDAction / FermionREX (the fast-path options with --fast)."""
    kw = dict(time_sign=args.time_sign, solver_tol=args.solver_tol, action_tol=args.action_tol,
              max_iter=args.max_iter, reuse_guess=args.reuse_guess)
    if args.fermions == "wilson":
        kw["on_unconverged"] = args.on_unconverged
    if getattr(args, "fast", False):
        kw.update(fast=True, precision=args.precision, colour_product=args.colour_product,
                  check_every=args.check_every, inner_tol=args.inner_tol, max_outer=args.max_outer)
    return kw


def start_links(args, geometry, device, batch, beta0, kappa0, bc0):
    """Links from --start-config (checked against the couplings) or a cold start; (U, source)."""
    if args.start_config is None:
        return SUN_identity(geometry.link_shape(batch)[:-1], dtype=torch.cdouble, device=device), "cold"
    U, meta, _ = load_state(args.start_config, device=device)
    g = meta["geometry"]
    if (g["D"], g["T"], g["L"], g["N"]) != (geometry.D, geometry.T, geometry.L, geometry.N):
        raise SystemExit(f"--start-config geometry {g} differs from the run's")
    name = fermion_parameter(args)[0]
    if meta["fermions"] != args.fermions or meta.get(name) is None:
        raise SystemExit(f"--start-config must come from a {args.fermions} run")
    if (abs(meta["beta"] - beta0) > 1e-12 or abs(meta[name] - kappa0) > 1e-12
            or meta["time_sign"] != args.time_sign or bc0 != 1.0):
        raise SystemExit(f"--start-config couplings beta={meta['beta']} {name}={meta[name]} "
                         f"time_sign={meta['time_sign']} (bc = 1) differ from the initial ones "
                         f"beta={beta0} {name}={kappa0} bc={bc0} time_sign={args.time_sign}")
    if U.shape[0] < batch:
        raise SystemExit(f"--start-config holds {U.shape[0]} configurations, {batch} needed")
    return U[:batch].clone(), f"{args.start_config} (trajectory {meta['trajectory']})"


def upgrade_csv_header(path, fields, log=print):
    """Resume of a run written before columns were added to `fields`: if the CSV at `path` has a
    header whose columns are all in `fields` (in the same order) but not all of `fields`, it is
    rewritten atomically with the new header and empty values in the new columns (read back as
    NaN), and a line is logged. Any other header is left alone (`ObservableWriter` refuses it)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header == list(fields) or [f for f in fields if f in header] != header:
            return False
        rows = [dict(zip(header, row)) for row in reader]

    def write(tmp):
        with open(tmp, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows([[row.get(f, "") for f in fields] for row in rows])
    _atomic_write(path, write)
    log(f"{path.name}: written before the columns {[f for f in fields if f not in header]} existed; "
        f"header upgraded, the new columns are empty for its {len(rows)} rows")
    return True


def clean_json(x):
    """NaN/inf -> None, numpy scalars -> python (strict JSON)."""
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def run_info(script, argv, device, args=None):
    """The "run" block of run_meta.json (with the fermion-operator entries of `fast_info` if args)."""
    info = {
        "script": script, "argv": list(sys.argv[1:] if argv is None else argv),
        "started": datetime.now().isoformat(timespec="seconds"), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
        "dtype": "complex128",
    }
    if args is not None:
        info.update(fast_info(args, device))
    return info


def _log(args, *msg):
    if not args.quiet:
        print(*msg, flush=True)


class Counter:
    """Solve / CG-iteration counter on the shared fermion term (differences of its stats)."""

    def __init__(self, fermion):
        self.fermion = fermion

    def snapshot(self):
        s = self.fermion.stats
        return s["solves"], s["cg_iterations"]

    def since(self, snap):
        s, c = self.snapshot()
        return s - snap[0], c - snap[1]


# ---- the run ------------------------------------------------------------------------------------
class PoolSampler:
    """B chains in equilibrium at lambda_0 (fermionic HMC on the shared QCDAction)."""

    def __init__(self, args, action, lam0, U):
        self.args, self.action, self.lam0, self.U = args, action, lam0, U

    @torch.no_grad()
    def advance(self, n, generator):
        acc = []
        for _ in range(int(n)):
            hmc = HMC(self.action.geometry, self.action.hamiltonian(self.lam0), self.args.tau,
                      self.args.n_steps, integrator=self.args.integrator_levels)
            self.U, accepted, _, _ = hmc.trajectory(self.U, generator)
            acc.append(float(accepted.to(torch.float64).mean()))
        return acc


def block_rows(batch, protocol, flow, int_obs):
    rows = []
    infos = flow.last_run["block_info"]
    for i, p in enumerate(protocol):
        info = infos[i]
        emdH = torch.stack([torch.as_tensor(x, dtype=torch.float64).reshape(-1).cpu() for x in info["exp_minus_dH"]])
        rows.append({
            "batch": batch, "block": i, "parameter": float(p),
            "acceptance": float(np.mean(info["acceptance"])), "exp_mdH": float(emdH.mean()),
            "work_mean": float(int_obs[:, i, 0].mean()), "heat_mean": float(int_obs[:, i, 1].mean()),
            "plaquette_mean": float(int_obs[:, i, 5].mean()),
        })
    return rows


def summarize(out, batch, protocol):
    """Summary from samples.csv / blocks.csv / training.csv (resume-safe)."""
    summary = {}
    spath, bpath, tpath = out / "samples.csv", out / "blocks.csv", out / "training.csv"
    if tpath.exists():
        t = read_observables(tpath)
        if t["step"].size:
            summary["training"] = {
                "steps": int(t["step"].size),
                "loss_first": float(t["loss"][0]), "loss_last": float(t["loss"][-1]),
                "loss_last10_mean": float(t["loss"][-10:].mean()),
                "ess_first": float(t["ess"][0]), "ess_last10_mean": float(t["ess"][-10:].mean()),
                "seconds_per_step": float(t["wall_s"].mean()),
                "solves_per_step": float(t["solves"].mean()),
            }
            if "injectivity_ratio" in t:  # rows of runs older than the column read as NaN: skipped
                ratio = t["injectivity_ratio"]
                known = ~np.isnan(ratio)
                summary["training"].update(
                    max_abs_coeff_max=float(np.nanmax(t["max_abs_coeff"])) if known.any() else None,
                    injectivity_ratio_max=float(ratio[known].max()) if known.any() else None,
                    injectivity_exceeded_steps=int((ratio[known] >= 1.0).sum()))
    if not spath.exists():
        return summary
    s = read_observables(spath)
    n = s["work"].size
    summary["n_samples"] = int(n)
    if n < 2:
        return summary
    W = torch.as_tensor(s["work"])
    dF, err = jarzynski_free_energy(W)
    first = np.unique(s["batch"], return_index=True)[1]  # one row per batch for the per-batch columns
    n_batches = first.size
    wall, pool_wall = s["wall_s"][first].sum(), s["pool_wall_s"][first].sum()
    summary.update({
        "dF": dF, "dF_error": err, "Z_ratio": math.exp(-dF),
        "ess": effective_sample_size(W),
        "work_mean": float(s["work"].mean()), "work_std": float(s["work"].std(ddof=1)),
        "heat_mean": float(s["heat"].mean()), "logJ_mean": float(s["logJ"].mean()),
        "final_plaquette_mean": float(s["plaquette"].mean()),
        "final_plaquette_error": float(s["plaquette"].std(ddof=1) / math.sqrt(n)),
        "n_batches": int(n_batches),
        "solves_per_sample": float(s["solves"][first].mean()),
        "cg_iter_per_sample": float(s["cg_iter"][first].mean()),
        "pool_solves_per_sample": float(s["pool_solves"][first].mean()),
        "seconds_per_sample": float(wall / n),
        "seconds_per_sample_with_pool": float((wall + pool_wall) / n),
        "seconds_per_batch": float(wall / n_batches),
    })
    if bpath.exists():
        b = read_observables(bpath)
        if b["block"].size:
            nb = len(protocol)
            per = lambda k: [float(b[k][b["block"] == i].mean()) for i in range(nb)]
            summary["blocks"] = {"parameter": list(protocol), "acceptance": per("acceptance"),
                                 "exp_minus_dH": per("exp_mdH"), "work_mean": per("work_mean"),
                                 "plaquette_mean": per("plaquette_mean")}
            summary["acceptance_mean"] = float(b["acceptance"].mean())
    return summary


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args.device)
    geometry = Geometry(args.D, args.T, args.L, args.N)
    out = Path(args.out) if args.out else ROOT / "output" / "snf_fermions" / default_name(args)
    out.mkdir(parents=True, exist_ok=True)
    pool_prefix, flow_path, meta_path = out / POOL_NAME, out / f"{FLOW_NAME}.pt", out / "run_meta.json"
    paths = {k: out / f"{k}.csv" for k in ("samples", "blocks", "training")}

    beta0, kappa0, bc0 = initial_couplings(args)
    lam0 = float(args.initial)
    protocol = list(args.protocol_values)
    defect_mask = build_defect_mask(args, geometry).to(device) if args.uses_defect else None
    action = QCDAction(geometry, args.protocol, beta=args.beta, bc=args.bc, defect_mask=defect_mask,
                       n_gauge_substeps=args.n_gauge_substeps, **action_kwargs(args), **fermion_kwargs(args))
    action.set_parameter(lam0)
    hmc_kw = dict(tau=args.tau, n_steps=args.n_steps, n_traj=args.n_traj, integrator=args.integrator_levels)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    # ---- flow
    flow_pars = None
    if args.mode == "snf":
        flow_pars = FlowPars(D=args.D, T=args.T, L=args.L, N=args.N,
                             mask=create_mask(args.D, args.T, args.L).to(device), protocol=protocol,
                             batch_size=args.batch, device=str(device))
        layers = make_gauge_layers(flow_pars, args.layers, rho=args.rho_init,
                                   residual=None if args.residual is None else
                                   {k: v for k, v in args.residual.items() if k != "parameterization"},
                                   sharing=getattr(args, "layer_sharing", None) if args.residual is not None else None)
        flow = FermionSNF(action, protocol, layers, flow_pars, initial_parameter=lam0,
                          condition=args.residual_condition, delta=args.residual_delta, **hmc_kw).to(device)
        params = [p for p in flow.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr) if params else None
        if args.train_steps and not params:
            raise SystemExit(f"--layers {args.layers} has no trainable parameters")
    else:
        flow = FermionNEMCMC(action, protocol, initial_parameter=lam0, **hmc_kw)
        params, optimizer = [], None
    if args.load_flow:
        check_flow_sidecar(args, args.load_flow)
        payload = torch.load(args.load_flow, map_location=device)
        flow.load_state_dict(payload["state_dict"])

    # ---- state
    key = physics_key(args)
    phase, progress = "train", 0
    if args.resume:
        if not pool_prefix.with_suffix(".json").exists():
            raise SystemExit(f"--resume: no checkpoint in {out}")
        U, meta, rng_state = load_state(pool_prefix, device=device)
        if meta["extra"].get("physics") != clean_json(key):
            raise SystemExit(f"--resume: checkpoint parameters {meta['extra'].get('physics')} differ from {key}")
        if meta["rng"]["device"].split(":")[0] != device.type:
            raise SystemExit(f"--resume: checkpoint RNG is for {meta['rng']['device']}, run on {device}")
        restore_generator(generator, rng_state)
        phase, progress = meta["extra"]["phase"], int(meta["extra"]["progress"])
        if args.mode == "snf":
            payload = torch.load(flow_path, map_location=device)
            flow.load_state_dict(payload["state_dict"])
            if optimizer is not None and payload.get("optimizer") is not None:
                optimizer.load_state_dict(payload["optimizer"])
        source = meta["extra"].get("source")
        _log(args, f"resumed {out}: phase {phase}, progress {progress}")
    else:
        previous = [p for p in list(paths.values()) + [pool_prefix.with_suffix(".json"), pool_prefix.with_suffix(".pt"),
                                                      flow_path, out / f"{FLOW_NAME}.json", meta_path] if p.exists()]
        if previous and not args.overwrite:
            raise SystemExit(f"{out} holds a previous run ({[p.name for p in previous]}): use --resume or --overwrite")
        for p in previous:
            p.unlink()
        U, source = start_links(args, geometry, device, args.batch, beta0, kappa0, bc0)
    pool = PoolSampler(args, action, lam0, U)
    counter = Counter(action.fermion)

    if args.resume:  # CSVs of a run started before columns were added
        for name, fields in (("samples", SAMPLE_FIELDS), ("blocks", BLOCK_FIELDS), ("training", TRAIN_FIELDS)):
            upgrade_csv_header(paths[name], fields, log=lambda m: _log(args, m))
    writers = {"samples": ObservableWriter(paths["samples"], SAMPLE_FIELDS, trajectory_field="batch"),
               "blocks": ObservableWriter(paths["blocks"], BLOCK_FIELDS, trajectory_field="batch")}
    if args.mode == "snf":
        writers["training"] = ObservableWriter(paths["training"], TRAIN_FIELDS, trajectory_field="step")
    if args.resume:
        train_done = progress if phase == "train" else args.train_steps
        eval_done = progress if phase == "eval" else (0 if phase == "train" else None)
        if "training" in writers:
            writers["training"].truncate(train_done)
        if eval_done is not None:
            writers["samples"].truncate(eval_done)
            writers["blocks"].truncate(eval_done)

    n_batches = -(-args.eval_samples // args.batch) if args.eval_samples else 0
    params_json = {k: v for k, v in vars(args).items()}
    info = run_info("main/main_snf_fermions.py", argv, device, args)
    info.update(lattice=list(geometry.lattice), volume=geometry.volume, start_source=source,
                fermions=args.fermions, mass=args.mass,
                initial_couplings={"beta": beta0, fermion_parameter(args)[0]: kappa0, "bc": bc0},
                defect_links=int(defect_mask[1].sum().item()) if defect_mask is not None else 0)
    if args.mode == "snf":
        info.update(flow_architecture=flow_architecture(args), layer_conditioning=[list(c) for c in flow.conditioning])

    def checkpoint(phase_, progress_, final=False):
        save_state(pool_prefix, pool.U, geometry=geometry, beta=beta0, trajectory=progress_,
                   time_sign=args.time_sign, generator=generator, **state_couplings(args, kappa0),
                   extra={"phase": phase_, "progress": int(progress_), "physics": clean_json(key),
                          "source": source, "bc": bc0, "seed": args.seed})
        if args.mode == "snf":
            torch.save({"state_dict": flow.state_dict(),
                        "optimizer": optimizer.state_dict() if optimizer is not None else None,
                        "train_step": int(progress_ if phase_ == "train" else args.train_steps)}, flow_path)
            atomic_json_save(clean_json({"physics": key, "layers": args.layers, "rho_init": args.rho_init,
                                         "architecture": flow_architecture(args),
                                         "conditioning": [list(c) for c in flow.conditioning],
                                         "protocol": protocol, "initial": lam0,
                                         "flow_pars": {"rho_shape": list(flow_pars.rho_shape)
                                                       if hasattr(flow_pars, "rho_shape") else None}}),
                             out / f"{FLOW_NAME}.json")
        summary = summarize(out, args.batch, protocol)
        atomic_json_save(clean_json({"parameters": params_json, "run": info, "phase": phase_,
                                     "progress": int(progress_), "finished": bool(final), "summary": summary}),
                         meta_path)
        return summary

    warned = set()

    def check_layers(where):
        """layer_diagnostics of the current flow; one RuntimeWarning per `where` if the bound is exceeded."""
        diag = flow.layer_diagnostics()
        if diag["exceeded"] and where not in warned:
            warned.add(where)
            msg = f"{where}: {layer_diagnostics_message(diag)}: the layers may not be bijections"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            _log(args, "WARNING: " + msg)
        return diag

    _log(args, f"{args.mode} {args.layers} protocol {args.protocol}: {lam0} -> {protocol[-1]} in {len(protocol)} "
               f"steps, lattice {geometry.lattice} SU({args.N}) beta={args.beta} {args.fermions} {'kappa=%g' % args.kappa if args.fermions == 'wilson' else 'mass=%g' % args.mass} bc={args.bc} "
               f"B={args.batch} device={device} -> {out}")

    # ---- thermalisation of the pool
    if not args.resume and args.therm:
        _sync(device)
        t0 = time.perf_counter()
        acc = pool.advance(args.therm, generator)
        _sync(device)
        info["therm_seconds"] = time.perf_counter() - t0
        info["therm_acceptance"] = float(np.mean(acc))
        _log(args, f"thermalised {args.therm} trajectories at {args.protocol} = {lam0}: acceptance "
                   f"{np.mean(acc):.3f}, {info['therm_seconds']:.1f} s")
    if not args.resume:
        checkpoint("train", 0)

    # ---- training
    if phase == "train":
        for step in range(progress, args.train_steps):
            pool.advance(args.n_between, generator)
            snap = counter.snapshot()
            _sync(device)
            t0 = time.perf_counter()
            optimizer.zero_grad()
            lr = learning_rate(args, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            # the parameters this step's flow runs with (before the optimizer step)
            param_abs_mean = float(torch.cat([p.detach().abs().reshape(-1) for p in params]).mean())
            diag = check_layers("training")
            _, W, Q, logJ, _ = flow(FieldState(pool.U.clone()), generator)
            loss = flow.loss(W)
            loss.backward()
            grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)))
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if math.isfinite(grad_norm):
                optimizer.step()
            _sync(device)
            wall = time.perf_counter() - t0
            solves, cg_iter = counter.since(snap)
            Wd = W.detach()
            acc = float(np.mean([np.mean(i["acceptance"]) for i in flow.last_run["block_info"]]))
            row = {"step": step, "loss": float(loss.detach()), "work_mean": float(Wd.mean()),
                   "work_std": float(Wd.std()) if Wd.numel() > 1 else 0.0,
                   "ess": effective_sample_size(Wd) if Wd.numel() > 1 else 1.0,
                   "heat_mean": float(Q.mean()), "logJ_mean": float(logJ.mean()), "acceptance": acc,
                   "grad_norm": grad_norm, "solves": solves, "cg_iter": cg_iter, "wall_s": wall,
                   "lr": lr, "param_abs_mean": param_abs_mean, "max_abs_coeff": diag["max_abs_coeff"],
                   "injectivity_ratio": diag["injectivity_ratio"]}
            writers["training"].write_rows([row])
            _log(args, f"train {step:5d} loss {row['loss']:+.5f} ess {row['ess']:.3f} acc {acc:.2f} "
                       f"|g| {grad_norm:.2e} solves {solves} {wall:.2f} s")
            if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0 and step + 1 < args.train_steps:
                checkpoint("train", step + 1)
        phase, progress = "eval", 0
        checkpoint("eval", 0)

    # ---- evaluation
    if phase == "eval":
        if args.mode == "snf":  # the flow evaluated (trained here, loaded with --load-flow, or resumed)
            info["layer_diagnostics"] = check_layers("evaluated flow")
            _log(args, f"evaluated flow: {layer_diagnostics_message(info['layer_diagnostics'])}")
        works = []
        if progress:
            works = list(read_observables(paths["samples"])["work"])
        for k in range(progress, n_batches):
            snap = counter.snapshot()
            _sync(device)
            t0 = time.perf_counter()
            pool.advance(args.n_between, generator)
            _sync(device)
            pool_wall = time.perf_counter() - t0
            pool_solves, _ = counter.since(snap)
            snap = counter.snapshot()
            t0 = time.perf_counter()
            with torch.no_grad():
                res = flow(FieldState(pool.U.clone()), generator)
            _sync(device)
            wall = time.perf_counter() - t0
            solves, cg_iter = counter.since(snap)
            state, W, Q = res[0], res[1], res[2]
            logJ = res[3] if args.mode == "snf" else torch.zeros_like(W)
            int_obs = res[-1].cpu()
            W, Q, logJ = W.detach().cpu(), Q.cpu(), logJ.cpu()
            works.extend(W.tolist())
            ess_b = effective_sample_size(W) if W.numel() > 1 else 1.0
            ess_c = effective_sample_size(torch.tensor(works)) if len(works) > 1 else 1.0
            S0, Sf = flow.last_run["S0"].cpu(), flow.last_run["S_final"].cpu()
            pl = int_obs[:, -1, 5]
            rows = [{"batch": k, "replica": b, "sample": k * args.batch + b, "work": float(W[b]),
                     "heat": float(Q[b]), "logJ": float(logJ[b]), "S0": float(S0[b]), "S_final": float(Sf[b]),
                     "plaquette": float(pl[b]), "solves": solves, "cg_iter": cg_iter, "pool_solves": pool_solves,
                     "wall_s": wall, "pool_wall_s": pool_wall, "ess_batch": ess_b, "ess_cumulative": ess_c}
                    for b in range(args.batch)]
            writers["samples"].write_rows(rows)
            writers["blocks"].write_rows(block_rows(k, protocol, flow, int_obs))
            _log(args, f"eval {k:5d} <W> {W.mean():+.5f} ess(batch) {ess_b:.3f} ess(all) {ess_c:.3f} "
                       f"solves {solves} {wall:.2f} s")
            if args.checkpoint_every and (k + 1) % args.checkpoint_every == 0 and k + 1 < n_batches:
                checkpoint("eval", k + 1)
        phase, progress = "done", n_batches

    summary = checkpoint("done", progress, final=True)
    if "dF" in summary:
        _log(args, f"done: {summary['n_samples']} samples, dF {summary['dF']:.6f} +- {summary['dF_error']:.6f}, "
                   f"ESS {summary['ess']:.4f}, <W> {summary['work_mean']:.5f}, "
                   f"acceptance {summary.get('acceptance_mean')}, {summary['solves_per_sample']:.1f} solves/sample, "
                   f"{summary['seconds_per_sample']:.3f} s/sample")
    return {"out": str(out), "summary": summary}


if __name__ == "__main__":
    main()
