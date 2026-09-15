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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from neoqcd.flow import FlowPars
from neoqcd.mcmc import HBOR, NEMCMC_update
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
THERMAL_RNG_VERSION = 1
THERMAL_SEED_OFFSET = 104729


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone linear NE-MCMC/SNF protocol (no parallel tempering)."
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
    parser.add_argument("--thermal-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50, help="Training epochs (or evaluation trajectories).")
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

    # Defect geometry.
    parser.add_argument("--defect-size", type=int, default=2)
    parser.add_argument("--time-slice", type=int, default=2)
    parser.add_argument("--space-slice", type=int, default=2)

    # NF architecture and its existing conditional-network controls.
    parser.add_argument(
        "--nf-architecture", "--nf-layer-type", dest="nf_architecture",
        choices=("smearing", "hyper-smearing", "residual", "hyper-residual"),
        default=None,
        help=(
            "SNF layer: residual=static trainable coefficients; "
            "hyper-residual=beta-conditioned residual MLP; smearing variants use CouplingLayer."
        ),
    )
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
        "--smearing-rho-init", type=float, default=1e-3,
        help="Strictly positive initial rho for non-hyper standard smearing.",
    )
    parser.add_argument("--flow-lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)

    # Initial-configuration cache.
    parser.add_argument("--cfg-cache", choices=("auto", "read", "refresh", "off"), default="auto")
    parser.add_argument("--cfg-cache-dir", type=str, default="data/cfgs_nemcmc_snf")
    parser.add_argument("--cfg-cache-tag", type=str, default="")

    # Output and logging.
    parser.add_argument("--main-dir", type=str, default=os.getcwd())
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    parser.add_argument("--log-every", type=int, default=1)
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
        args.nf_architecture = "hyper-smearing" if args.algorithm == "snf" else "smearing"
    args.wandb_mode = args.wandb_mode or args.wandb or os.environ.get("WANDB_MODE", "disabled")
    if args.wandb_mode not in {"disabled", "offline", "online"}:
        parser.error("WANDB_MODE must be disabled, offline, or online")
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.algorithm == "nemcmc" and args.phase == "train":
        raise ValueError("--algorithm nemcmc supports only --phase evaluate")
    if args.algorithm == "nemcmc" and args.checkpoint:
        raise ValueError("--checkpoint is an SNF option and is invalid for nemcmc")
    if args.algorithm == "snf" and args.phase == "evaluate" and not args.checkpoint:
        raise ValueError("SNF evaluation requires --checkpoint")
    if args.phase == "evaluate" and args.train_steps is not None:
        raise ValueError("--train-steps is valid only in train phase")
    if args.phase == "train" and args.eval_steps is not None:
        raise ValueError("--eval-steps is valid only in evaluate phase")
    if args.phase == "train" and args.eval_samples is not None:
        raise ValueError("--eval-samples is valid only in evaluate phase")
    if args.eval_samples is not None and args.eval_samples < 1:
        raise ValueError("--eval-samples must be >= 1")
    if args.eval_samples is not None and args.eval_steps is not None:
        raise ValueError("--eval-samples and --eval-steps are mutually exclusive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive and is global across ranks")
    if args.protocol_steps < 1:
        raise ValueError("--protocol-steps must be >= 1")
    if args.thermal_steps < 0 or args.steps < 1 or (args.eval_steps is not None and args.eval_steps < 1):
        raise ValueError("thermal steps must be >= 0 and run steps must be >= 1")
    if args.train_steps is not None and args.train_steps < 1:
        raise ValueError("--train-steps must be >= 1")
    if args.updates_per_layer < 0 or args.orsteps < 0:
        raise ValueError("MCMC update counts must be non-negative")
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
    if args.domain == "defect":
        if args.T < 4 or args.L < args.defect_size + 4:
            raise ValueError("defect domain requires T >= 4 and L >= defect-size + 4")
        if not 2 <= args.time_slice <= args.T - 2:
            raise ValueError("time-slice must satisfy 2 <= time-slice <= T-2")
        if not 2 <= args.space_slice <= args.L - args.defect_size - 2:
            raise ValueError("space-slice must leave a two-site buffer around the defect")
    if args.algorithm == "nemcmc" and args.nf_architecture != "smearing":
        raise ValueError("NF architecture options are valid only for --algorithm snf")
    if args.hyper_smearing_mode == "class" and (args.domain != "defect" or args.nf_architecture != "hyper-smearing"):
        raise ValueError("hyper-smearing-mode=class is supported only for defect hyper-smearing")
    if args.nf_architecture != "hyper-smearing" and args.hyper_smearing_mode != "shared":
        raise ValueError("hyper-smearing-mode is meaningful only with --nf-architecture hyper-smearing")
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


def _protocol_endpoints(args: argparse.Namespace) -> tuple[float, float, float, str]:
    if args.domain == "beta":
        return float(args.beta_initial), float(args.beta_final), float(args.beta_initial), "beta"
    start = 0.0 if args.bc_initial is None else args.bc_initial
    end = 1.0 if args.bc_final is None else args.bc_final
    fixed_beta = 6.0 if args.beta is None else args.beta
    return float(start), float(end), float(fixed_beta), "bc"


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
        "schema": 4,
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


def _propagate_rank0_io(error: str) -> None:
    status = [bool(not error), error]
    dist.broadcast_object_list(status, src=0)
    if not status[0]:
        raise RuntimeError(status[1])


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
            or done.get("schema") != 4
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
                "schema": 4,
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
    if args.domain == "defect":
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

    def forward(self, x, parameter, delta_parameter):
        work_dtype = x.real.dtype
        if isinstance(self.layer, ResidualNormalizingFlows):
            small = self.flow_pars.defect.cut_defect(x, buffer=2)
            small, _, logdet = self.layer(
                small, self.flow_pars.small_mask[-1], dmasking=True,
                dmask=self.flow_pars.smeared_defect_mask[-1], beta=parameter,
                delta_beta=delta_parameter,
            )
            return self._embed(small, x), logdet.to(dtype=work_dtype)
        small, _, half_logdet = self.layer(
            x, self.flow_pars, rho_layer=None, is_training=True,
            beta=parameter, delta_beta=delta_parameter,
        )
        return self._embed(small, x), (2.0 * half_logdet).to(dtype=work_dtype)


class _StandardSNFProtocolLayer(torch.nn.Module):
    """Adapter giving existing Flow-layer conventions a standalone interface."""

    def __init__(self, layer, flow_pars):
        super().__init__()
        self.layer = layer
        self.flow_pars = flow_pars

    def forward(self, x, parameter, delta_parameter):
        x_out, _, half_logdet = self.layer(
            x,
            self.flow_pars,
            rho_layer=None,
            is_training=True,
            beta=parameter,
            delta_beta=delta_parameter,
        )
        return x_out, (2.0 * half_logdet).to(dtype=x.real.dtype)


def _make_snf_layer(args, flow_pars):
    architecture = args.nf_architecture
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
    return "shared" if args.nf_architecture in {"hyper-smearing", "hyper-residual"} else "per_block"


def _action(flow_pars, x, domain: str, parameter: float, defect_mask):
    if domain == "beta":
        return Wilson_action(flow_pars, float(parameter), 1.0)(x, None)
    return Wilson_action(flow_pars, flow_pars._standalone_beta, float(parameter))(x, defect_mask)


class StandaloneProtocol(torch.nn.Module):
    """Shared-NF K-block protocol with detached stochastic transitions."""

    def __init__(self, args, flow_pars, defect_mask, start: float, end: float):
        super().__init__()
        self.args = args
        self.flow_pars = flow_pars
        self.defect_mask = defect_mask
        self.domain = args.domain
        self.start = float(start)
        self.end = float(end)
        self.protocol_values = [
            self.start + (self.end - self.start) * (i / args.protocol_steps)
            for i in range(args.protocol_steps + 1)
        ]
        self.nf_layer_sharing = _nf_layer_sharing(args) if args.algorithm == "snf" else None
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
        return layer(
            x,
            self._parameter_tensor(self.protocol_values[step], x.shape[0], x.device),
            self._parameter_tensor(
                self.protocol_values[step + 1] - self.protocol_values[step],
                x.shape[0],
                x.device,
            ),
        )

    def _run_stochastic_block(self, x, step):
        return self.mcmc_updates[step](x, self.flow_pars, None)

    def trainable_parameters_for_block(self, block_index: int):
        if self.nf_layer_sharing == "shared":
            return list(self.nf_layer.parameters())
        return list(self.nf_layers[int(block_index)].parameters())

    @torch.no_grad()
    def advance_prior(self, x, steps: int):
        x = x.detach()
        for _ in range(int(steps)):
            x, _, _ = self.prior_update(x, self.flow_pars, None)
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
                x, dq, _ = self._run_stochastic_block(x.detach(), step)
                q = q + (dq if torch.is_tensor(dq) else torch.zeros_like(q))
                x = x.detach()
        x, dlogdet = self._run_nf_block(x, int(block_index))
        logdet = logdet + dlogdet
        p1 = self.protocol_values[int(block_index) + 1]
        st = _action(self.flow_pars, x, self.domain, p1, self.defect_mask)
        work = st - s0 - q - logdet
        return x, work, st, q, logdet

    def forward(self, x, train: bool = False, block_index=None, s0=None):
        if train and block_index is not None:
            if s0 is None:
                raise ValueError("s0 is required for blockwise SNF training")
            return self.up_to_block_nograd(x, s0, int(block_index))
        if train:
            raise ValueError("SNF training must select one block at a time")
        x = x.detach()
        work_dtype = x.real.dtype
        with torch.no_grad():
            s_initial = _action(self.flow_pars, x, self.domain, self.start, self.defect_mask)
        q = torch.zeros(x.shape[0], dtype=work_dtype, device=x.device)
        total_logdet = torch.zeros_like(q)
        differentiable_work = torch.zeros_like(q)

        for i, update in enumerate(self.mcmc_updates):
            p0, p1 = self.protocol_values[i], self.protocol_values[i + 1]
            x_before = x
            if train and self.args.updates_per_layer == 0:
                # No stochastic block will detach x_before in this branch;
                # keep the action in the composition graph so the summed
                # layer work telescopes to the exact protocol work.
                s_before = _action(self.flow_pars, x_before, self.domain, p0, self.defect_mask)
            else:
                with torch.no_grad():
                    s_before = _action(self.flow_pars, x_before, self.domain, p0, self.defect_mask)
            if self.args.algorithm == "snf":
                x_layer, dlogdet = self._run_nf_block(x, i)
                with torch.no_grad():
                    s_layer = _action(self.flow_pars, x_layer, self.domain, p1, self.defect_mask)
                total_logdet = total_logdet + dlogdet
                x = x_layer
            if self.args.updates_per_layer > 0:
                with torch.no_grad():
                    x, dq, _ = update(x.detach(), self.flow_pars, None)
                    q = q + (dq if torch.is_tensor(dq) else torch.zeros_like(q))
                    x = x.detach()

        with torch.no_grad():
            s_final = _action(self.flow_pars, x, self.domain, self.end, self.defect_mask)
            exact_work = s_final - s_initial - q - total_logdet.detach()
        return x, (differentiable_work if train else exact_work), exact_work


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
            if output.exists() and any(output.iterdir()) and not args.overwrite:
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
            managed = ("snf_checkpoint.pt", "summary.json", "metrics.csv", "run_meta.json")
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
    return {
        "schema": 5,
        "algorithm": "snf",
        "domain": args.domain,
        "D": int(args.D), "T": int(args.T), "L": int(args.L), "N": int(args.N),
        "global_batch": int(args.batch_size),
        "updates_per_layer": int(args.updates_per_layer),
        "orsteps": int(args.orsteps),
        "prior_mcmc_steps": int(args.prior_mcmc_steps),
        "nf_layer_sharing": _nf_layer_sharing(args),
        "protocol_steps": int(args.protocol_steps),
        "protocol": [float(value) for value in protocol],
        "fixed_beta": (6.0 if args.beta is None else float(args.beta)) if args.domain == "defect" else None,
        "nf_architecture": args.nf_architecture,
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


def _load_checkpoint_coordinated(model, path: Path, expected: dict, device: torch.device) -> None:
    rank = dist.get_rank()
    state = None
    error = ""
    if rank == 0:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoint = _safe_torch_load(path, map_location="cpu")
            if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_semantics") != expected:
                raise ValueError("checkpoint semantic metadata does not match this run")
            state = checkpoint.get("model_state_dict")
            if not isinstance(state, dict):
                raise ValueError("checkpoint has no model_state_dict")
        except Exception as exc:  # propagate the same failure before any rank loads
            error = f"checkpoint preflight failed: {exc}"
    payload = [state, error]
    dist.broadcast_object_list(payload, src=0)
    state, error = payload
    if error:
        raise RuntimeError(error)

    local_error = ""
    try:
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


def _train_shared_block(ddp_model, optimizer, x, s0, block_index, args, world_size, device):
    optimizer.zero_grad(set_to_none=True)
    _, train_work, _, _, _ = ddp_model(x, train=True, block_index=int(block_index), s0=s0)
    active_parameters = ddp_model.module.trainable_parameters_for_block(int(block_index))
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
    if int(bad_loss.item()) or not loss.requires_grad:
        return loss_value, 1

    loss.backward()
    bad_grad = torch.tensor([0], dtype=torch.int64, device=device)
    for parameter in active_parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            bad_grad.fill_(1)
            break
    dist.all_reduce(bad_grad, op=dist.ReduceOp.MAX)
    if int(bad_grad.item()):
        optimizer.zero_grad(set_to_none=True)
        return loss_value, 1

    old_parameters = {id(parameter): parameter.detach().clone() for parameter in active_parameters}
    if args.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(active_parameters, args.grad_clip_norm)
    optimizer.step()
    bad_parameter = torch.tensor([0], dtype=torch.int64, device=device)
    for parameter in active_parameters:
        if not torch.isfinite(parameter).all().item():
            bad_parameter.fill_(1)
            break
    dist.all_reduce(bad_parameter, op=dist.ReduceOp.MAX)
    if int(bad_parameter.item()):
        for parameter in active_parameters:
            parameter.data.copy_(old_parameters[id(parameter)])
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(
            "Non-finite SNF parameters after optimizer step; "
            "terminating instead of continuing with a possibly corrupted Adam state"
        )
    return loss_value, 0


def _advance_persistent_prior(model, prior_state: torch.Tensor, steps: int) -> torch.Tensor:
    """Advance a local prior chain without exposing its storage to target work."""
    with torch.no_grad():
        return model.advance_prior(prior_state.detach().clone(), int(steps)).detach()


def _main_impl(args: argparse.Namespace, runtime) -> None:
    rank, world_size, device, backend, _, _owns_process_group = runtime
    start, end, fixed_beta, parameter_name = _protocol_endpoints(args)
    output = _resolve_output_dir(args, rank, world_size)
    local_batch = args.batch_size // world_size
    protocol = [start + (end - start) * (i / args.protocol_steps) for i in range(args.protocol_steps + 1)]

    # FlowPars.nstep describes NF transitions, whereas ``protocol`` also has
    # the terminal point used by the standalone driver.
    flow_pars, defect_mask = _build_flow_pars(args, device, protocol[:-1], local_batch)
    flow_pars._standalone_beta = fixed_beta
    checkpoint_semantics = _checkpoint_semantics(args, protocol)

    # Construct and synchronise the trainable model before thermalisation.
    # Cache hit/miss must not affect its initial random parameters.
    model = None
    ddp_model = None
    optimizer = None
    input_checkpoint_path = None
    model = StandaloneProtocol(args, flow_pars, defect_mask, start, end).to(device)
    if args.algorithm == "snf":
        if args.checkpoint:
            input_checkpoint_path = Path(args.checkpoint)
            if not input_checkpoint_path.is_absolute():
                input_checkpoint_path = Path(args.main_dir) / input_checkpoint_path
            _load_checkpoint_coordinated(model, input_checkpoint_path, checkpoint_semantics, device)
        if device.type == "cuda":
            ddp_model = DDP(
                model,
                device_ids=[device.index],
                find_unused_parameters=(model.nf_layer_sharing == "per_block"),
            )
        else:
            ddp_model = DDP(model, find_unused_parameters=(model.nf_layer_sharing == "per_block"))
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.flow_lr)

    _clear_managed_output_artifacts(output, rank, input_checkpoint_path)

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
        "cfg_cache_status": cache_status, "checkpoint_semantics": checkpoint_semantics,
        "args": vars(args),
    }
    metadata_error = ""
    if rank == 0:
        try:
            _atomic_json_save(metadata, output / "run_meta.json")
        except Exception as exc:
            metadata_error = f"run metadata write failed: {exc}"
    _propagate_rank0_io(metadata_error)

    wandb_run = _wandb_start(args, rank, world_size, output, metadata)
    metrics_path = output / "metrics.csv"
    metrics_error = ""
    if rank == 0:
        fd = None
        tmp_name = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=".metrics.", suffix=".tmp", dir=str(output))
            os.close(fd)
            fd = None
            with open(tmp_name, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    ["epoch", "work_mean", "work_sem", "deltaF", "ess_fraction", "finite_count", "loss_epoch", "block_loss_json", "block_skipped_json", "skipped_update", "cache_status"]
                )
            os.replace(tmp_name, metrics_path)
        except Exception as exc:
            metrics_error = f"metrics initialization failed: {exc}"
        finally:
            if fd is not None:
                os.close(fd)
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
    _propagate_rank0_io(metrics_error)

    if args.phase == "train":
        n_steps = args.train_steps if args.train_steps is not None else args.steps
    elif args.eval_samples is not None:
        n_steps = int(math.ceil(float(args.eval_samples) / float(args.batch_size)))
    else:
        n_steps = args.eval_steps if args.eval_steps is not None else args.steps
    evaluation_requested_samples = args.eval_samples if args.phase == "evaluate" else None
    evaluation_generated_samples = n_steps * args.batch_size if args.phase == "evaluate" else None
    evaluation_work_local = []

    metadata["evaluation_requested_samples"] = evaluation_requested_samples
    metadata["evaluation_generated_samples"] = evaluation_generated_samples
    metadata["evaluation_trajectory_count"] = n_steps if args.phase == "evaluate" else None
    t0 = time.time()
    last_metrics = {"work_mean": None, "work_sem": None, "deltaF": None, "ess_fraction": None, "finite_count": 0}
    last_loss = None
    total_skipped_updates = 0
    saw_finite_work = False
    for step in range(n_steps):
        # The cache is only the initial thermalised batch.  The local prior
        # chain persists across epochs/trajectories, while target work always
        # receives a detached clone and can never mutate that chain.
        prior_state = _advance_persistent_prior(model, prior_state, args.prior_mcmc_steps)
        cfgs = prior_state.detach().clone()
        skipped_update = 0
        block_losses = []
        block_skips = []
        if args.algorithm == "nemcmc":
            # The NE-MCMC algorithm has no differentiable layer and is always
            # evaluated; StandaloneProtocol still records real NEMCMC_update blocks.
            with torch.no_grad():
                cfgs, _, exact_work = model(cfgs, train=False)
            loss_value = None
        elif args.phase == "train":
            with torch.no_grad():
                s0 = _action(flow_pars, cfgs, args.domain, start, defect_mask)
            for block_index in range(args.protocol_steps):
                block_loss, block_skipped = _train_shared_block(
                    ddp_model, optimizer, cfgs, s0, block_index, args, world_size, device
                )
                block_losses.append(block_loss)
                block_skips.append(block_skipped)
            skipped_update = int(any(block_skips))
            loss_value = (
                sum(value for value in block_losses if value is not None) / len(block_losses)
                if any(value is not None for value in block_losses)
                else None
            )
            with torch.no_grad():
                _, _, exact_work = model(cfgs, train=False)
        else:
            with torch.no_grad():
                cfgs, _, exact_work = ddp_model.module(cfgs, train=False)
            loss_value = None

        if args.phase == "evaluate":
            evaluation_work_local.append(exact_work.detach().to(device="cpu", dtype=torch.float64))

        last_metrics = _global_work_metrics(exact_work)
        saw_finite_work = saw_finite_work or last_metrics["finite_count"] > 0
        last_loss = loss_value
        total_skipped_updates += skipped_update
        step_io_error = ""
        if rank == 0 and (step + 1) % args.log_every == 0:
            try:
                elapsed = time.time() - t0
                print(
                    f"step={step + 1} work_mean={last_metrics['work_mean']} "
                    f"work_sem={last_metrics['work_sem']} deltaF={last_metrics['deltaF']} "
                    f"ess_fraction={last_metrics['ess_fraction']} finite={last_metrics['finite_count']} "
                    f"skipped_update={skipped_update} elapsed_s={elapsed:.2f}",
                    flush=True,
                )
                with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(
                        [
                            step + 1, last_metrics["work_mean"], last_metrics["work_sem"],
                            last_metrics["deltaF"], last_metrics["ess_fraction"],
                            last_metrics["finite_count"], loss_value, json.dumps(block_losses),
                            json.dumps(block_skips), skipped_update, cache_status,
                        ]
                    )
                if wandb_run is not None:
                    values = {"epoch": step + 1, "loss/epoch_mean": loss_value, "training/skipped_update": skipped_update}
                    values.update({f"loss/block_{index}": value for index, value in enumerate(block_losses) if value is not None})
                    values.update({f"work/{key}": value for key, value in last_metrics.items() if value is not None})
                    wandb_run.log({key: value for key, value in values.items() if value is not None})
            except Exception as exc:
                step_io_error = f"step metrics/W&B write failed: {exc}"
        _propagate_rank0_io(step_io_error)

    pooled_metrics = None
    if args.phase == "evaluate":
        pooled_local_work = torch.cat(evaluation_work_local, dim=0).to(device=device, dtype=torch.float64)
        pooled_metrics = _global_work_metrics(pooled_local_work)

    no_finite_work = not saw_finite_work
    final_io_error = ""
    if rank == 0:
        try:
            if model is not None and args.algorithm == "snf" and not no_finite_work:
                checkpoint_path = output / "snf_checkpoint.pt"
                cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                _atomic_torch_save(
                    {
                        "checkpoint_semantics": checkpoint_semantics,
                        "model_state_dict": cpu_state,
                    },
                    checkpoint_path,
                )
                metadata["checkpoint"] = str(checkpoint_path)
            metadata["summary"] = {
                **last_metrics,
                "loss": last_loss,
                "skipped_updates": total_skipped_updates,
                "pooled_work_mean": pooled_metrics["work_mean"] if pooled_metrics is not None else None,
                "pooled_work_sem": pooled_metrics["work_sem"] if pooled_metrics is not None else None,
                "pooled_deltaF": pooled_metrics["deltaF"] if pooled_metrics is not None else None,
                "pooled_ess_fraction": pooled_metrics["ess_fraction"] if pooled_metrics is not None else None,
                "pooled_finite_count": pooled_metrics["finite_count"] if pooled_metrics is not None else None,
                "evaluation_requested_samples": evaluation_requested_samples,
                "evaluation_generated_samples": evaluation_generated_samples,
                "evaluation_trajectory_count": n_steps if args.phase == "evaluate" else None,
                "status": "error" if no_finite_work else "ok",
                "error": "no finite work values" if no_finite_work else None,
                "elapsed_seconds": time.time() - t0,
            }
            _atomic_json_save(metadata["summary"], output / "summary.json")
            _atomic_json_save(metadata, output / "run_meta.json")
            if wandb_run is not None:
                summary_values = {
                    key: value for key, value in metadata["summary"].items() if value is not None
                }
                wandb_run.summary.update(summary_values)
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
