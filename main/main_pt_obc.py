import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

from neoqcd.distributedPT import (
    NEOREX,
    NEOREXConfig,
    NEOREXProtocolWrapper,
    OREXPTConfig,
    PTConfig,
    OutOfEquilibriumParallelTempering,
    ParallelTempering,
)
from neoqcd.flow import FlowPars
from neoqcd.mcmc import DefectMCMCAdapter, HBOR
from neoqcd.smearing import DefectCouplingLayer, ResidualNormalizingFlows
from neoqcd.theory import Wilson_action
from neoqcd.utils import Defect, create_around_defect_mask, create_mask


@dataclass
class RunState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str


def _slug(value) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def _cfg_cache_root(args: argparse.Namespace) -> Path:
    root = Path(args.cfg_cache_dir)
    if not root.is_absolute():
        root = Path(args.main_dir) / root
    return root


def _cfg_cache_key(
    args: argparse.Namespace,
    world_size: int,
    parameter_min: float,
    parameter_max: float,
    parameter_ladder: torch.Tensor,
) -> dict:
    return {
        "schema": 1,
        "D": int(args.D),
        "T": int(args.T),
        "L": int(args.L),
        "N": int(args.N),
        "beta": float(args.beta),
        "defect_size": int(args.defect_size),
        "time_slice": int(args.time_slice),
        "space_slice": int(args.space_slice),
        "batch_size": int(args.batch_size),
        "world_size": int(world_size),
        "parameter_name": "bc",
        "parameter_min": float(parameter_min),
        "parameter_max": float(parameter_max),
        "parameter_ladder": [float(v.item()) for v in parameter_ladder.detach().cpu()],
        "orsteps": int(args.orsteps),
        "updates_per_layer": int(args.updates_per_layer),
        "dtype": "torch.cdouble",
        "tag": str(args.cfg_cache_tag or ""),
    }


def _cfg_cache_path(root: Path, key: dict) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    name = (
        f"ptobc_D{key['D']}_T{key['T']}_L{key['L']}_N{key['N']}_"
        f"beta{_slug(key['beta'])}_d{key['defect_size']}_"
        f"t{key['time_slice']}_s{key['space_slice']}_"
        f"bs{key['batch_size']}_r{key['world_size']}_"
        f"bc{_slug(key['parameter_min'])}-{_slug(key['parameter_max'])}_"
        f"or{key['orsteps']}_{digest}"
    )
    if key.get("tag"):
        name = f"{_slug(key['tag'])}_{name}"
    return root / name


def _resolve_neorex_flow_checkpoint_path(args: argparse.Namespace, spec: str):
    spec = str(spec or "").strip()
    if not spec:
        return None
    if spec.lower() == "auto":
        root = Path(args.neorex_flow_checkpoint_dir)
        if not root.is_absolute():
            root = Path(args.main_dir) / root
        return root / args.neorex_flow_checkpoint_name
    path = Path(spec)
    if not path.is_absolute():
        path = Path(args.main_dir) / path
    return path


def _neorex_flow_metadata(args: argparse.Namespace, saved_path: str) -> dict:
    return {
        "algorithm": "neorex",
        "saved_path": str(saved_path),
        "neorex_protocol_steps": int(args.neorex_protocol_steps),
        "neorex_heatbath_steps_per_step": int(args.neorex_heatbath_steps_per_step),
        "neorex_work_mode": str(args.neorex_work_mode),
        "train_steps": int(args.train_steps),
        "neorex_eval_flow": bool(args.neorex_eval_flow),
        "hyper_smearing_mode": str(args.hyper_smearing_mode),
        "hyper_time_embedding_dim": int(args.hyper_time_embedding_dim),
        "hyper_hidden_dim": int(args.hyper_hidden_dim),
        "hyper_depth": int(args.hyper_depth),
        "hyper_activation": str(args.hyper_activation),
        "hyper_rho_init": float(args.hyper_rho_init),
        "hyper_scale_by_delta": bool(args.hyper_scale_by_delta),
        "hyper_normalize_by_nstep": bool(args.hyper_normalize_by_nstep),
        "D": int(args.D),
        "T": int(args.T),
        "L": int(args.L),
        "N": int(args.N),
        "defect_size": int(args.defect_size),
        "time_slice": int(args.time_slice),
        "space_slice": int(args.space_slice),
        "batch_size": int(args.batch_size),
        "bc_min": float(args.bc_min),
        "bc_max": float(args.bc_max),
    }


def _write_neorex_flow_metadata(args: argparse.Namespace, saved_path: str) -> str:
    path = Path(saved_path)
    meta_path = Path(f"{path}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_neorex_flow_metadata(args, str(path)), f, indent=2)
    return str(meta_path)


def _rank_cfg_path(cache_dir: Path, rank: int) -> Path:
    return cache_dir / f"rank_{rank:03d}.pt"


def _cache_metadata_matches(cache_dir: Path, key: dict, world_size: int, requested_steps: int) -> tuple[bool, int]:
    meta_path = cache_dir / "meta.json"
    done_path = cache_dir / "DONE"
    if not meta_path.exists() or not done_path.exists():
        return False, 0
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, 0
    if meta.get("key") != key:
        return False, int(meta.get("thermal_steps", 0) or 0)
    cached_steps = int(meta.get("thermal_steps", 0) or 0)
    if cached_steps < int(requested_steps):
        return False, cached_steps
    for rank in range(int(world_size)):
        if not _rank_cfg_path(cache_dir, rank).exists():
            return False, cached_steps
    return True, cached_steps


def _load_cached_cfgs(cache_dir: Path, rank: int, device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    payload = torch.load(_rank_cfg_path(cache_dir, rank), map_location=device)
    cfgs = payload["cfgs"] if isinstance(payload, dict) and "cfgs" in payload else payload
    cfgs = cfgs.to(device=device)
    expected_shape = (args.batch_size, args.D, args.T) + (args.L,) * (args.D - 1) + (args.N, args.N)
    if tuple(cfgs.shape) != tuple(expected_shape):
        raise ValueError(
            f"Cached cfgs for rank {rank} have shape {tuple(cfgs.shape)}, expected {expected_shape}"
        )
    return cfgs


def _save_cached_cfgs(
    cache_dir: Path,
    rank: int,
    world_size: int,
    cfgs: torch.Tensor,
    key: dict,
    thermal_steps: int,
) -> None:
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        done_path = cache_dir / "DONE"
        if done_path.exists():
            done_path.unlink()
    dist.barrier()

    rank_path = _rank_cfg_path(cache_dir, rank)
    tmp_path = rank_path.with_suffix(f".tmp.{rank}")
    torch.save({"rank": int(rank), "cfgs": cfgs.detach().cpu()}, tmp_path)
    os.replace(tmp_path, rank_path)
    dist.barrier()

    if rank == 0:
        meta = {
            "key": key,
            "thermal_steps": int(thermal_steps),
            "world_size": int(world_size),
            "rank_files": [f"rank_{r:03d}.pt" for r in range(int(world_size))],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        meta_path = cache_dir / "meta.json"
        tmp_meta = cache_dir / "meta.json.tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta, meta_path)
        with open(cache_dir / "DONE", "w", encoding="utf-8") as f:
            f.write("ok\n")
    dist.barrier()


def _mean_sem_from_moments(sum_x: float, sum_x2: float, count: int) -> tuple[float, float]:
    n = int(count)
    if n <= 0:
        return float("nan"), float("nan")
    mean = float(sum_x) / float(n)
    if n == 1:
        return mean, float("nan")
    var_unbiased = (float(sum_x2) - (float(sum_x) * float(sum_x) / float(n))) / float(n - 1)
    var_unbiased = max(var_unbiased, 0.0)
    sem = math.sqrt(var_unbiased / float(n))
    return mean, sem


def _rank0_cuda_memory_metrics(device: torch.device, enabled: bool) -> dict[str, float]:
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    scale = float(1024 ** 3)
    return {
        "gpu_memory/allocated_gb": float(torch.cuda.memory_allocated(device)) / scale,
        "gpu_memory/reserved_gb": float(torch.cuda.memory_reserved(device)) / scale,
        "gpu_memory/max_allocated_gb": float(torch.cuda.max_memory_allocated(device)) / scale,
        "gpu_memory/max_reserved_gb": float(torch.cuda.max_memory_reserved(device)) / scale,
        "gpu_memory/free_gb": float(free_bytes) / scale,
        "gpu_memory/total_gb": float(total_bytes) / scale,
    }


class DefectSNFProtocolLayer(torch.nn.Module):
    """
    Deterministic defect-coupling NF block compatible with NEOREXProtocolWrapper.
    """

    def __init__(self, flow_pars: FlowPars):
        super().__init__()
        self.flow_pars = flow_pars
        rho0 = torch.zeros(flow_pars.rho_shape, device=flow_pars.device, dtype=torch.float64)
        self.layer = DefectCouplingLayer(flow_pars, rho0, time=1.0)

    def _normalize_parameter(self, parameter: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        p = torch.as_tensor(parameter, dtype=torch.float64, device=device)
        if p.dim() == 0:
            p = p.expand(batch_size)
        if p.dim() != 1 or int(p.shape[0]) != batch_size:
            raise ValueError(f"parameter must have shape ({batch_size},), got {tuple(p.shape)}")
        return p

    def _embed_small_cfgs(self, small_cfgs: torch.Tensor, cfgs: torch.Tensor, buffer: int = 2) -> torch.Tensor:
        """
        Gradient-safe embedding of the defect patch into a full lattice tensor.
        Avoids mutating tensors that are part of checkpointed autograd paths.
        """
        defect = self.flow_pars.defect
        t = int(defect.time_slice)
        s = int(defect.space_slice)
        dsize = int(defect.dsize)
        D = int(self.flow_pars.D)

        out = cfgs.clone()
        out[:, 0, t, (s - 1):(dsize + s + 1), (s - 1):(dsize + s + 1), (s - 1):(dsize + s + 1)] = small_cfgs[
            :, 0, buffer, (buffer - 1):(dsize + buffer + 1), (buffer - 1):(dsize + buffer + 1), (buffer - 1):(dsize + buffer + 1)
        ]
        for d in range(1, D):
            out[:, d, t, s:(dsize + s + 1), s:(dsize + s + 1), s:(dsize + s + 1)] = small_cfgs[
                :, d, buffer, buffer:(dsize + buffer + 1), buffer:(dsize + buffer + 1), buffer:(dsize + buffer + 1)
            ]
            out[:, d, t - 1, s:(dsize + s + 1), s:(dsize + s + 1), s:(dsize + s + 1)] = small_cfgs[
                :, d, buffer - 1, buffer:(dsize + buffer + 1), buffer:(dsize + buffer + 1), buffer:(dsize + buffer + 1)
            ]
        return out

    def _set_local_jac_shape(self, local_batch: int):
        """
        Smearing stores jac_shape with a fixed batch dimension from FlowPars.
        In NEO-REX we may process per-parameter sub-batches; update jac_shape
        to the local sub-batch size for this forward call.
        """
        old_shape = tuple(self.layer.smearing.jac_shape)
        if len(old_shape) < 1:
            raise RuntimeError("Invalid smearing.jac_shape")
        self.layer.smearing.jac_shape = (int(local_batch),) + tuple(old_shape[1:])
        return old_shape

    def forward(self, x: torch.Tensor, bc: torch.Tensor, delta_bc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = int(x.shape[0])
        bc = self._normalize_parameter(bc, bs, x.device)
        delta_bc = self._normalize_parameter(delta_bc, bs, x.device)
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        old_jac_shape = self._set_local_jac_shape(bs)
        try:
            x_small, _, dlogj = self.layer(
                x,
                self.flow_pars,
                rho_layer=None,
                is_training=True,
                beta=bc,
                delta_beta=delta_bc,
            )
        finally:
            self.layer.smearing.jac_shape = old_jac_shape
        x_out = self._embed_small_cfgs(x_small, x, buffer=2)
        logdet = (2.0 * dlogj).to(dtype=work_dtype)
        return x_out, logdet


class ResidualDefectSNFProtocolLayer(DefectSNFProtocolLayer):
    """
    Defect-local residual normalizing-flow block.

    ResidualNormalizingFlows already returns a real tangent-space logdet, so no
    extra factor of two is applied here.
    """

    def __init__(self, flow_pars: FlowPars):
        torch.nn.Module.__init__(self)
        self.flow_pars = flow_pars
        self.layer = ResidualNormalizingFlows(flow_pars, time=1.0)

    def forward(self, x: torch.Tensor, bc: torch.Tensor, delta_bc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = int(x.shape[0])
        bc = self._normalize_parameter(bc, bs, x.device)
        delta_bc = self._normalize_parameter(delta_bc, bs, x.device)
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype

        small_cfgs = self.flow_pars.defect.cut_defect(x, buffer=2)
        small_cfgs, _, logdet = self.layer(
            small_cfgs,
            self.flow_pars.small_mask[-1],
            dmasking=True,
            dmask=self.flow_pars.smeared_defect_mask[-1],
            beta=bc,
            delta_beta=delta_bc,
        )
        return self._embed_small_cfgs(small_cfgs, x, buffer=2), logdet.to(dtype=work_dtype)


def _build_neorex_flow_pars(
    args: argparse.Namespace,
    device: torch.device,
    mask: torch.Tensor,
    defect: Defect,
) -> FlowPars:
    buffer = 2
    required_t = 2 * buffer
    required_l = int(args.defect_size) + 2 * buffer
    if args.T < required_t or args.L < required_l:
        raise ValueError(
            f"NEO-REX defect NF requires T >= {required_t} and L >= {required_l}; "
            f"got T={args.T}, L={args.L}"
        )
    if args.time_slice < buffer or args.time_slice > (args.T - buffer):
        raise ValueError(
            f"time_slice must satisfy {buffer} <= time_slice <= {args.T - buffer} for NEO-REX"
        )
    if args.space_slice < buffer or args.space_slice > (args.L - args.defect_size - buffer):
        raise ValueError(
            f"space_slice must satisfy {buffer} <= space_slice <= {args.L - args.defect_size - buffer} "
            "for NEO-REX"
        )

    small_t = 2 * buffer
    small_l = int(args.defect_size) + 2 * buffer
    small_mask = create_mask(args.D, small_t, small_l).to(device)
    smeared_defect_mask = create_around_defect_mask(args.D, args.defect_size, buffer=buffer).to(device)
    small_defect = Defect(
        D=args.D,
        T=small_t,
        L=small_l,
        dsize=args.defect_size,
        time_slice=buffer,
        space_slice=buffer,
    )
    small_defect_mask = small_defect.create_defect_mask().to(device)

    neorex_nstep = int(args.neorex_protocol_steps)
    protocol = [float(args.beta)] * neorex_nstep
    return FlowPars(
        D=args.D,
        T=args.T,
        L=args.L,
        N=args.N,
        mask=mask,
        protocol=protocol,
        batch_size=args.batch_size,
        device=device,
        rho_shape_type=3,
        set_rho=0,
        fit_train=0,
        orsteps=args.orsteps,
        updates_per_layer=args.updates_per_layer,
        use_hyper_smearing=args.use_hyper_smearing,
        hyper_smearing_mode=args.hyper_smearing_mode,
        hyper_time_embedding_dim=args.hyper_time_embedding_dim,
        hyper_hidden_dim=args.hyper_hidden_dim,
        hyper_depth=args.hyper_depth,
        hyper_activation=args.hyper_activation,
        hyper_rho_init=args.hyper_rho_init,
        hyper_normalize_by_nstep=args.hyper_normalize_by_nstep,
        hyper_rho_eps=args.hyper_rho_eps,
        hyper_scale_by_delta=args.hyper_scale_by_delta,
        hyper_rho_max=args.hyper_rho_max,
        residual_include_imag=args.residual_include_imag,
        residual_quadratic=args.residual_quadratic,
        residual_coeff_init=args.residual_coeff_init,
        residual_coeff_max=args.residual_coeff_max,
        defect=defect,
        smeared_defect_mask=[smeared_defect_mask],
        small_mask=[small_mask],
        small_defect_mask=[small_defect_mask],
    )


def _default_output_dir(main_dir: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = run_name or f"pt_obc_{timestamp}"
    return Path(main_dir) / "results" / "pt_obc" / final_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard PT on OBC/defect coupling for SU(3)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU + gloo backend")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--algorithm", type=str, default="pt", choices=["pt", "orex", "neorex"])

    # Lattice / theory
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--beta", type=float, default=6.0)

    # Defect (boundary-coupling) geometry
    parser.add_argument("--defect-size", type=int, default=2, help="Defect cube side")
    parser.add_argument("--time-slice", type=int, default=0)
    parser.add_argument("--space-slice", type=int, default=0)

    # PT
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--parameter-min", type=float, default=None, help="Ladder min for tempering parameter")
    parser.add_argument("--parameter-max", type=float, default=None, help="Ladder max for tempering parameter")
    parser.add_argument("--bc-min", type=float, default=0.0)
    parser.add_argument("--bc-max", type=float, default=1.0)
    # Alias compatibile con vecchi launcher/template (qui indica il range del parametro PT, non beta fisica).
    parser.add_argument("--beta-min", dest="beta_min_alias", type=float, default=None)
    parser.add_argument("--beta-max", dest="beta_max_alias", type=float, default=None)
    parser.add_argument("--swap-parity", type=str, default="alternating", choices=["even", "odd", "alternating"])
    parser.add_argument("--acceptance-rule", type=str, default="exact", choices=["exact", "linear", "auto"])
    parser.add_argument("--orex-protocol-steps", type=int, default=8)
    parser.add_argument("--orex-mcmc-steps-per-step", type=int, default=1)
    parser.add_argument("--orex-schedule", type=str, default="even_odd", choices=["even_odd", "alternating", "even", "odd"])
    parser.add_argument("--neorex-protocol-steps", type=int, default=0)
    parser.add_argument("--neorex-heatbath-steps-per-step", type=int, default=0)
    parser.add_argument("--neorex-work-mode", type=str, default="logdet", choices=["logdet", "work"])
    parser.add_argument("--neorex-flow-lr", type=float, default=1e-3)
    parser.add_argument("--neorex-grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--neorex-max-consecutive-nan-grad-updates", type=int, default=5)
    parser.add_argument("--neorex-stochastic-first", action="store_true")
    parser.add_argument(
        "--neorex-eval-flow",
        action="store_true",
        help="For NEO-REX, load/use the flow in evaluation mode and skip optimizer updates.",
    )
    parser.add_argument(
        "--neorex-load-flow",
        type=str,
        default="",
        help="Load NEO-REX flow checkpoint from this path, or use 'auto' for the global default path.",
    )
    parser.add_argument(
        "--neorex-save-flow",
        type=str,
        default="",
        help="Also save final NEO-REX flow checkpoint to this path, or use 'auto' for the global default path.",
    )
    parser.add_argument("--neorex-flow-checkpoint-dir", type=str, default="data/neorex_flows")
    parser.add_argument("--neorex-flow-checkpoint-name", type=str, default="global.pt")
    parser.add_argument("--use-hyper-smearing", dest="use_hyper_smearing", action="store_true")
    parser.add_argument("--no-hyper-smearing", dest="use_hyper_smearing", action="store_false")
    parser.add_argument("--hyper-smearing-mode", type=str, default="per_link", choices=["shared", "per_link", "class"])
    parser.add_argument("--hyper-time-embedding-dim", type=int, default=8)
    parser.add_argument("--hyper-hidden-dim", type=int, default=16)
    parser.add_argument("--hyper-depth", type=int, default=2)
    parser.add_argument("--hyper-activation", type=str, default="silu", choices=["silu", "gelu", "tanh", "relu"])
    parser.add_argument("--hyper-rho-init", type=float, default=1e-3)
    parser.add_argument("--hyper-rho-eps", type=float, default=0.0)
    parser.add_argument("--hyper-rho-max", type=float, default=0.0)
    parser.add_argument("--hyper-normalize-by-nstep", dest="hyper_normalize_by_nstep", action="store_true")
    parser.add_argument("--hyper-no-normalize-by-nstep", dest="hyper_normalize_by_nstep", action="store_false")
    parser.add_argument("--hyper-scale-by-delta", dest="hyper_scale_by_delta", action="store_true")
    parser.add_argument("--hyper-no-scale-by-delta", dest="hyper_scale_by_delta", action="store_false")
    parser.add_argument("--residual-include-imag", dest="residual_include_imag", action="store_true")
    parser.add_argument("--residual-no-include-imag", dest="residual_include_imag", action="store_false")
    parser.add_argument("--residual-quadratic", dest="residual_quadratic", action="store_true")
    parser.add_argument("--residual-no-quadratic", dest="residual_quadratic", action="store_false")
    parser.add_argument("--residual-coeff-init", type=float, default=1e-3)
    parser.add_argument("--residual-coeff-max", type=float, default=0.0)

    # Run control
    parser.add_argument("--thermal-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sweeps-per-step", type=int, default=1)
    parser.add_argument("--sweep", type=int, default=None, help="Alias of --sweeps-per-step")
    parser.add_argument("--swap-every", type=int, default=1)
    parser.add_argument("--orsteps", type=int, default=4)
    parser.add_argument("--updates-per-layer", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--main-dir", type=str, default=os.getcwd(), help="Project/output root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="If set, use this directory for logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--cfg-cache-dir", type=str, default="data/cfgs")
    parser.add_argument("--cfg-cache-tag", type=str, default="")
    parser.add_argument("--use-cfg-cache", dest="use_cfg_cache", action="store_true")
    parser.add_argument("--no-cfg-cache", dest="use_cfg_cache", action="store_false")
    parser.add_argument("--overwrite-cfg-cache", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging on rank 0")
    parser.add_argument("--wandb-project", type=str, default="neo-pt")
    parser.add_argument("--wandb-entity", type=str, default="lqft-snf")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument(
        "--log-gpu-memory",
        dest="log_gpu_memory",
        action="store_true",
        help="Log rank-0 CUDA memory metrics to W&B/CSV when CUDA is available.",
    )
    parser.add_argument(
        "--no-log-gpu-memory",
        dest="log_gpu_memory",
        action="store_false",
        help="Disable rank-0 CUDA memory logging.",
    )

    # Accepted for command-template compatibility; not used in this standard PT main.
    parser.add_argument("--update", type=str, default="heatbath")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--nf-layer-type", type=str, default="", choices=["", "smearing", "residual"])
    parser.add_argument("--nf-layers-per-step", type=int, default=0)
    parser.set_defaults(
        use_hyper_smearing=True,
        hyper_normalize_by_nstep=False,
        hyper_scale_by_delta=True,
        residual_include_imag=True,
        residual_quadratic=True,
        log_gpu_memory=False,
        use_cfg_cache=True,
    )
    args = parser.parse_args()
    if args.sweep is not None:
        args.sweeps_per_step = args.sweep
    return args


def _choose_backend_and_device(force_cpu: bool) -> tuple[str, torch.device, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if force_cpu:
        return "gloo", torch.device("cpu"), local_rank

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile: usa --cpu oppure avvia su nodo GPU")

    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK fuori range: {local_rank}")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return "nccl", device, local_rank


def _init_process_group(backend: str, device: torch.device) -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    if backend == "gloo" and "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))

    if world_size_env > 1 or "RANK" in os.environ:
        # torchrun / launcher path
        if backend == "nccl":
            try:
                dist.init_process_group(backend=backend, device_id=device)
            except TypeError:
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)
    else:
        # single-process fallback
        with tempfile.NamedTemporaryFile(delete=False) as f:
            init_method = f"file://{f.name}"
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank_env,
            world_size=world_size_env,
        )

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed non inizializzato")

    return dist.get_rank(), dist.get_world_size()


def setup_runtime(args: argparse.Namespace) -> RunState:
    backend, device, local_rank = _choose_backend_and_device(args.cpu)
    rank, world_size = _init_process_group(backend, device)

    seed = int(args.seed) + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return RunState(rank=rank, world_size=world_size, local_rank=local_rank, device=device, backend=backend)


def _parameter_indices(local_parameters: torch.Tensor, parameter_ladder: torch.Tensor) -> torch.Tensor:
    # map each sample parameter to nearest ladder index
    diff = torch.abs(local_parameters.unsqueeze(-1) - parameter_ladder.view(1, -1))
    return torch.argmin(diff, dim=-1)


def local_hbor_sweep(
    cfgs: torch.Tensor,
    local_parameters: torch.Tensor,
    parameter_ladder: torch.Tensor,
    hbor_by_idx: list,
    mask: torch.Tensor,
    defect_mask: torch.Tensor,
) -> torch.Tensor:
    idx = _parameter_indices(local_parameters, parameter_ladder)
    out = cfgs
    for k in torch.unique(idx):
        k_int = int(k.item())
        sel = torch.nonzero(idx == k_int, as_tuple=False).squeeze(-1)
        batch_k = out.index_select(0, sel)
        batch_k = hbor_by_idx[k_int](batch_k, mask, defect_mask)
        out = out.clone()
        out[sel] = batch_k
    return out


def compute_action_table(
    cfgs: torch.Tensor,
    action_by_idx: list,
    defect_mask: torch.Tensor,
    world_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    table = torch.empty((cfgs.shape[0], world_size), device=cfgs.device, dtype=dtype)
    for k in range(world_size):
        table[:, k] = action_by_idx[k](cfgs, defect_mask)
    return table


def _normalize_metric_list(values, n_links: int, fill: float):
    out = [fill] * n_links
    if values is None:
        return out
    for k in range(min(n_links, len(values))):
        out[k] = values[k]
    return out


def _default_wandb_run_name(args: argparse.Namespace, world_size: int) -> str:
    if args.algorithm == "neorex":
        nsteps = int(args.neorex_protocol_steps)
    elif args.algorithm == "orex":
        nsteps = int(args.orex_protocol_steps)
    else:
        nsteps = int(args.steps)
    volume = f"T{int(args.T)}L{int(args.L)}"
    return (
        f"{args.algorithm}_"
        f"{volume}_"
        f"d{int(args.defect_size)}_"
        f"nsteps{nsteps}_"
        f"r{int(world_size)}"
    )


def main(args: argparse.Namespace) -> None:
    state = setup_runtime(args)
    rank, world_size, device = state.rank, state.world_size, state.device
    effective_acceptance_rule = (
        "exact" if args.algorithm in {"orex", "neorex"} else args.acceptance_rule
    )

    parameter_min = args.parameter_min
    parameter_max = args.parameter_max
    if parameter_min is None:
        parameter_min = args.beta_min_alias if args.beta_min_alias is not None else args.bc_min
    if parameter_max is None:
        parameter_max = args.beta_max_alias if args.beta_max_alias is not None else args.bc_max
    if parameter_min >= parameter_max:
        raise ValueError(f"parameter_min must be < parameter_max, got {parameter_min} >= {parameter_max}")

    if rank == 0:
        out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.main_dir, args.run_name)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(".")
    out_dir_box = [str(out_dir)]
    dist.broadcast_object_list(out_dir_box, src=0)
    out_dir = Path(out_dir_box[0])
    csv_path = out_dir / "pt_metrics.csv"
    meta_path = out_dir / "run_meta.json"
    wandb_mod = None
    wandb_run = None

    if args.wandb and rank == 0:
        try:
            import wandb as _wandb
        except ImportError as err:
            raise RuntimeError(
                "wandb logging requested but package is not available in the current environment."
            ) from err
        wandb_mod = _wandb
        wb_name = args.wandb_run_name.strip() if args.wandb_run_name else ""
        if not wb_name:
            wb_name = _default_wandb_run_name(args, world_size)
        wandb_run = wandb_mod.init(
            entity=args.wandb_entity,
            project=(args.wandb_project or "neo-pt"),
            name=(wb_name or None),
            dir=str(out_dir),
            config=vars(args),
        )

    if rank == 0:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "backend": state.backend,
                    "world_size": world_size,
                    "device": str(device),
                    "parameter_min": parameter_min,
                    "parameter_max": parameter_max,
                    "algorithm": args.algorithm,
                    "acceptance_rule": effective_acceptance_rule,
                    "args": vars(args),
                },
                f,
                indent=2,
            )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "swap_n_accepted",
                    "swap_n_proposed",
                    "swap_acceptance",
                    "work_mean",
                    "work_sem",
                    "work_count",
                    "flow_loss",
                    "flow_applied_updates",
                    "flow_skipped_nan_grad_updates",
                    "flow_consecutive_nan_grad_updates",
                    "train_flow",
                    "link_acceptance_json",
                    "link_work_mean_json",
                    "link_flow_loss_json",
                    "parameter_local_min",
                    "parameter_local_max",
                    "elapsed_seconds",
                    "thermalization_seconds",
                    "measurement_elapsed_seconds",
                    "total_elapsed_seconds",
                    "gpu_memory_allocated_gb",
                    "gpu_memory_reserved_gb",
                    "gpu_memory_max_allocated_gb",
                    "gpu_memory_max_reserved_gb",
                    "gpu_memory_free_gb",
                    "gpu_memory_total_gb",
                ]
            )

    if rank == 0:
        print(
            f"backend={state.backend} device={device} world_size={world_size} "
            f"algorithm={args.algorithm} parameter_range=[{parameter_min}, {parameter_max}] "
            f"acceptance_rule={effective_acceptance_rule} "
            f"output_dir={out_dir}",
            flush=True,
        )

    if args.N != 3:
        raise ValueError("Questo main e' pensato per SU(3): usa --N 3")

    neorex_flow_load_mode = "none"
    neorex_flow_load_path = None

    mask = create_mask(args.D, args.T, args.L).to(device)
    defect = Defect(
        D=args.D,
        T=args.T,
        L=args.L,
        dsize=args.defect_size,
        time_slice=args.time_slice,
        space_slice=args.space_slice,
    )
    defect_mask = defect.create_defect_mask().to(device)

    flow_pars = FlowPars(
        D=args.D,
        T=args.T,
        L=args.L,
        N=args.N,
        mask=mask,
        protocol=[args.beta],
        batch_size=args.batch_size,
        device=device,
        orsteps=args.orsteps,
        updates_per_layer=args.updates_per_layer,
        defect=defect,
    )

    if args.algorithm == "orex":
        pt_cfg = OREXPTConfig(
            batch_size=args.batch_size,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            parameter_name="bc",
            parameter_dtype=torch.float64,
            swap_parity=args.swap_parity,
            acceptance_rule="exact",
            orex_protocol_steps=args.orex_protocol_steps,
            orex_mcmc_steps_per_step=args.orex_mcmc_steps_per_step,
            orex_schedule=args.orex_schedule,
        )
        pt = OutOfEquilibriumParallelTempering(pt_cfg, rank, world_size, device)
        orex_mcmc = DefectMCMCAdapter(
            flow_pars=flow_pars,
            beta=args.beta,
            defect_mask=defect_mask,
            batch_size=args.batch_size,
            parameter_name="bc",
        )
    elif args.algorithm == "neorex":
        if int(args.neorex_protocol_steps) <= 0:
            raise ValueError("For --algorithm neorex set --neorex-protocol-steps >= 1")
        if int(args.neorex_heatbath_steps_per_step) < 0:
            raise ValueError("--neorex-heatbath-steps-per-step must be >= 0")

        neorex_flow_pars = _build_neorex_flow_pars(args, device, mask, defect)
        nf_layer_kind = str(args.nf_layer_type or "smearing").lower()
        if nf_layer_kind == "residual":
            nf_layer = ResidualDefectSNFProtocolLayer(neorex_flow_pars)
        else:
            nf_layer = DefectSNFProtocolLayer(neorex_flow_pars)
        neorex_mcmc = DefectMCMCAdapter(
            flow_pars=flow_pars,
            beta=args.beta,
            defect_mask=defect_mask,
            batch_size=args.batch_size,
            parameter_name="bc",
            default_parameter=parameter_min,
        )
        neorex_flow = NEOREXProtocolWrapper(
            nf_layer=nf_layer,
            theory=neorex_mcmc,
            n_protocol_steps=args.neorex_protocol_steps,
            work_mode=args.neorex_work_mode,
            stochastic_update=neorex_mcmc if args.neorex_heatbath_steps_per_step > 0 else None,
            stochastic_steps_per_step=args.neorex_heatbath_steps_per_step,
            nf_first=not bool(args.neorex_stochastic_first),
            parameter_name="bc",
        )
        pt_cfg = NEOREXConfig(
            batch_size=args.batch_size,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            parameter_name="bc",
            parameter_dtype=torch.float64,
            swap_parity=args.swap_parity,
            acceptance_rule="exact",
            orex_protocol_steps=args.neorex_protocol_steps,
            orex_mcmc_steps_per_step=args.neorex_heatbath_steps_per_step,
            orex_schedule=args.orex_schedule,
            flow_lr=args.neorex_flow_lr,
            flow_train=True,
            grad_clip_norm=args.neorex_grad_clip_norm,
            max_consecutive_nan_grad_updates=args.neorex_max_consecutive_nan_grad_updates,
        )
        pt = NEOREX(
            flow=neorex_flow,
            config=pt_cfg,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        load_path = _resolve_neorex_flow_checkpoint_path(args, args.neorex_load_flow)
        if load_path is not None:
            if rank == 0:
                checkpoint_exists = load_path.exists()
            else:
                checkpoint_exists = False
            exists_box = [bool(checkpoint_exists)]
            dist.broadcast_object_list(exists_box, src=0)
            checkpoint_exists = bool(exists_box[0])
            if checkpoint_exists:
                pt.load_flow(str(load_path))
                neorex_flow_load_mode = "load"
                neorex_flow_load_path = str(load_path)
                if rank == 0:
                    print(f"neorex_flow_checkpoint=load  path={load_path}", flush=True)
            elif str(args.neorex_load_flow).strip().lower() == "auto":
                neorex_flow_load_mode = "auto_missing"
                neorex_flow_load_path = str(load_path)
                if rank == 0:
                    print(f"neorex_flow_checkpoint=auto_missing  path={load_path}", flush=True)
            else:
                raise FileNotFoundError(f"NEO-REX flow checkpoint not found: {load_path}")
        orex_mcmc = neorex_mcmc
    else:
        pt_cfg = PTConfig(
            batch_size=args.batch_size,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            parameter_name="bc",
            parameter_dtype=torch.float64,
            swap_parity=args.swap_parity,
            acceptance_rule=args.acceptance_rule,
        )
        pt = ParallelTempering(pt_cfg, rank, world_size, device)
        orex_mcmc = None

    parameter_ladder = torch.linspace(
        pt_cfg.parameter_min,
        pt_cfg.parameter_max,
        world_size,
        device=device,
        dtype=pt_cfg.parameter_dtype,
    )
    n_links = max(0, world_size - 1)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "parameter_name": getattr(pt, "parameter_name", "bc"),
                "parameter_ladder": [float(v.item()) for v in parameter_ladder.detach().cpu()],
            },
            allow_val_change=True,
        )

    # Cache one HBOR/action object per ladder parameter.
    hbor_by_idx = []
    action_by_idx = []
    for lam in parameter_ladder:
        lam_f = float(lam.item())
        hbor_by_idx.append(HBOR(flow_pars, beta=args.beta, defect_par=lam_f))
        action_by_idx.append(Wilson_action(flow_pars, beta=args.beta, defect_par=lam_f))

    cache_enabled = bool(args.use_cfg_cache) and int(args.thermal_steps) > 0
    cache_mode = "disabled"
    cache_steps = 0
    cache_dir = None
    cache_key = None
    if cache_enabled:
        cache_root = _cfg_cache_root(args)
        cache_key = _cfg_cache_key(args, world_size, parameter_min, parameter_max, parameter_ladder)
        cache_dir = _cfg_cache_path(cache_root, cache_key)
        if rank == 0:
            cache_hit, cache_steps = (
                (False, 0)
                if bool(args.overwrite_cfg_cache)
                else _cache_metadata_matches(cache_dir, cache_key, world_size, int(args.thermal_steps))
            )
            cache_mode = "load" if cache_hit else "miss"
        decision = [cache_mode, int(cache_steps), str(cache_dir)]
        dist.broadcast_object_list(decision, src=0)
        cache_mode, cache_steps, cache_dir_str = decision
        cache_steps = int(cache_steps)
        cache_dir = Path(cache_dir_str)

    thermalization_t0 = time.time()
    if cache_enabled and cache_mode == "load":
        cfgs = _load_cached_cfgs(cache_dir, rank, device, args)
        thermal_steps_executed = 0
    else:
        cfgs = torch.eye(args.N, dtype=torch.cdouble, device=device)
        cfgs = cfgs.view(1, 1, 1, 1, 1, args.N, args.N).repeat(args.batch_size, args.D, args.T, args.L, args.L, args.L, 1, 1)
        thermal_steps_executed = int(args.thermal_steps)
        for _ in range(thermal_steps_executed):
            cfgs = local_hbor_sweep(
                cfgs=cfgs,
                local_parameters=pt.local_parameters,
                parameter_ladder=parameter_ladder,
                hbor_by_idx=hbor_by_idx,
                mask=mask,
                defect_mask=defect_mask,
            )
        if cache_enabled:
            _save_cached_cfgs(
                cache_dir=cache_dir,
                rank=rank,
                world_size=world_size,
                cfgs=cfgs,
                key=cache_key,
                thermal_steps=int(args.thermal_steps),
            )
            cache_mode = "saved" if cache_mode == "miss" else cache_mode
    thermalization_elapsed = time.time() - thermalization_t0
    if rank == 0:
        print(
            f"thermalization_steps={args.thermal_steps}  "
            f"thermalization_steps_executed={thermal_steps_executed}  "
            f"cfg_cache={cache_mode}  "
            f"cfg_cache_steps={cache_steps}  "
            f"thermalization_elapsed_s={thermalization_elapsed:.2f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.summary["timing/thermalization_seconds"] = float(thermalization_elapsed)
            wandb_run.summary["thermalization/steps_executed"] = int(thermal_steps_executed)
            wandb_run.summary["cfg_cache/mode"] = str(cache_mode)
            wandb_run.summary["cfg_cache/cached_thermal_steps"] = int(cache_steps)

    t0 = time.time()
    total_swap_accepted = 0
    total_swap_proposed = 0
    total_swap_attempts = 0
    total_work_sum = 0.0
    total_work_sum_sq = 0.0
    total_work_count = 0
    total_flow_loss_sum = 0.0
    total_flow_loss_count = 0
    total_flow_applied_updates = 0
    total_flow_skipped_updates = 0
    total_link_accepted = [0] * n_links
    total_link_proposed = [0] * n_links
    total_link_work_sum = [0.0] * n_links
    total_link_work_count = [0] * n_links
    last_log_elapsed = 0.0
    for step in range(args.steps):
        for _ in range(args.sweeps_per_step):
            cfgs = local_hbor_sweep(
                cfgs=cfgs,
                local_parameters=pt.local_parameters,
                parameter_ladder=parameter_ladder,
                hbor_by_idx=hbor_by_idx,
                mask=mask,
                defect_mask=defect_mask,
            )

        stats = {"n_accepted": 0, "n_proposed": 0}
        train_flow = False
        if (step + 1) % args.swap_every == 0:
            if args.algorithm == "orex":
                orex_mcmc.set_parameter(pt.local_parameters, parameter_name="bc")
                cfgs, stats = pt.step(cfgs, orex_mcmc)
            elif args.algorithm == "neorex":
                train_flow = (
                    (not bool(args.neorex_eval_flow))
                    and ((args.train_steps <= 0) or ((step + 1) <= args.train_steps))
                )
                cfgs, stats = pt.neorex_step(cfgs, train_flow=train_flow)
            else:
                local_actions = compute_action_table(
                    cfgs=cfgs,
                    action_by_idx=action_by_idx,
                    defect_mask=defect_mask,
                    world_size=world_size,
                    dtype=pt.param_dtype,
                )
                stats = pt.step(local_actions=local_actions)
            if rank == 0:
                total_swap_accepted += int(stats["n_accepted"])
                total_swap_proposed += int(stats["n_proposed"])
                total_swap_attempts += 1
                link_acc_step = _normalize_metric_list(stats.get("link_accepted"), n_links, 0)
                link_prop_step = _normalize_metric_list(stats.get("link_proposed"), n_links, 0)
                for k in range(n_links):
                    total_link_accepted[k] += int(link_acc_step[k])
                    total_link_proposed[k] += int(link_prop_step[k])
                if args.algorithm in {"orex", "neorex"}:
                    total_work_sum += float(stats.get("work_sum", 0.0))
                    total_work_sum_sq += float(stats.get("work_sum_sq", 0.0))
                    total_work_count += int(stats.get("work_count", 0))
                    link_work_mean_step = _normalize_metric_list(stats.get("link_work_mean"), n_links, float("nan"))
                    link_work_count_step = _normalize_metric_list(stats.get("link_work_count"), n_links, 0)
                    for k in range(n_links):
                        cnt = int(link_work_count_step[k])
                        val = float(link_work_mean_step[k]) if k < len(link_work_mean_step) else float("nan")
                        if cnt > 0 and math.isfinite(val):
                            total_link_work_sum[k] += val * cnt
                            total_link_work_count[k] += cnt
                if args.algorithm == "neorex":
                    flow_loss = float(stats.get("flow_loss", float("nan")))
                    flow_loss_count = int(stats.get("flow_loss_count", 0))
                    if flow_loss_count > 0 and math.isfinite(flow_loss):
                        total_flow_loss_sum += flow_loss * flow_loss_count
                        total_flow_loss_count += flow_loss_count
                    total_flow_applied_updates += int(stats.get("applied_updates", 0))
                    total_flow_skipped_updates += int(stats.get("skipped_nan_grad_updates", 0))

        link_accepted_step = _normalize_metric_list(stats.get("link_accepted"), n_links, 0)
        link_proposed_step = _normalize_metric_list(stats.get("link_proposed"), n_links, 0)
        link_acceptance_step = [
            (float(link_accepted_step[k]) / float(link_proposed_step[k]))
            if int(link_proposed_step[k]) > 0
            else float("nan")
            for k in range(n_links)
        ]
        link_work_mean_step = _normalize_metric_list(
            stats.get("link_work_mean"),
            n_links,
            float("nan"),
        )
        link_flow_loss_step = _normalize_metric_list(
            stats.get("flow_loss_link_mean"),
            n_links,
            float("nan"),
        )

        if rank == 0 and wandb_run is not None:
            swap_acc_step = (
                float(stats["n_accepted"]) / float(stats["n_proposed"])
                if int(stats.get("n_proposed", 0)) > 0
                else float("nan")
            )
            work_step = float(stats.get("work_mean", float("nan")))
            flow_loss_step = float(stats.get("flow_loss", float("nan")))
            measurement_elapsed_step = time.time() - t0
            work_cum = (
                total_work_sum / total_work_count
                if total_work_count > 0
                else float("nan")
            )
            wb = {
                "step": step + 1,
                "train_flow": int(train_flow),
                "timing/thermalization_seconds": float(thermalization_elapsed),
                "timing/measurement_elapsed_seconds": float(measurement_elapsed_step),
                "timing/total_elapsed_seconds": float(thermalization_elapsed + measurement_elapsed_step),
                "swap/n_accepted": int(stats.get("n_accepted", 0)),
                "swap/n_proposed": int(stats.get("n_proposed", 0)),
                "swap/acceptance_step": float(swap_acc_step),
                "swap/acceptance_cumulative": (
                    float(total_swap_accepted) / float(total_swap_proposed)
                    if total_swap_proposed > 0
                    else float("nan")
                ),
                "work/mean_step": float(work_step),
                "work/mean_cumulative": float(work_cum),
                "flow_loss/mean_step": float(flow_loss_step),
            }
            wb.update(_rank0_cuda_memory_metrics(device, bool(args.log_gpu_memory)))
            for k in range(n_links):
                wb[f"swap/link_{k}_acceptance_step"] = float(link_acceptance_step[k])
                wb[f"swap/link_{k}_acceptance_cumulative"] = (
                    float(total_link_accepted[k]) / float(total_link_proposed[k])
                    if total_link_proposed[k] > 0
                    else float("nan")
                )
                wb[f"work/link_{k}_mean_step"] = float(link_work_mean_step[k])
                wb[f"work/link_{k}_mean_cumulative"] = (
                    float(total_link_work_sum[k]) / float(total_link_work_count[k])
                    if total_link_work_count[k] > 0
                    else float("nan")
                )
                wb[f"flow_loss/link_{k}_mean_step"] = float(link_flow_loss_step[k])
            wandb_run.log(wb, step=step + 1)

        if rank == 0 and (step + 1) % args.log_every == 0:
            swap_acc = (
                stats["n_accepted"] / stats["n_proposed"]
                if stats["n_proposed"] > 0
                else float("nan")
            )
            step_work_mean = float("nan")
            step_work_sem = float("nan")
            step_work_count = 0
            step_flow_loss = float("nan")
            step_flow_applied_updates = 0
            step_flow_skipped_updates = 0
            step_flow_consecutive_nan = 0
            if args.algorithm in {"orex", "neorex"}:
                step_work_count = int(stats.get("work_count", 0))
                step_work_mean, step_work_sem = _mean_sem_from_moments(
                    float(stats.get("work_sum", 0.0)),
                    float(stats.get("work_sum_sq", 0.0)),
                    step_work_count,
                )
            if args.algorithm == "neorex":
                step_flow_loss = float(stats.get("flow_loss", float("nan")))
                step_flow_applied_updates = int(stats.get("applied_updates", 0))
                step_flow_skipped_updates = int(stats.get("skipped_nan_grad_updates", 0))
                step_flow_consecutive_nan = int(stats.get("consecutive_nan_grad_updates", 0))
            lmin = pt.local_parameters.min().item()
            lmax = pt.local_parameters.max().item()
            elapsed = time.time() - t0
            total_elapsed = thermalization_elapsed + elapsed
            gpu_memory = _rank0_cuda_memory_metrics(device, bool(args.log_gpu_memory))
            elapsed_since_last_log = elapsed - last_log_elapsed
            last_log_elapsed = elapsed
            memory_msg = ""
            if gpu_memory:
                memory_msg = (
                    f"  gpu_alloc_gb={gpu_memory['gpu_memory/allocated_gb']:.3f}  "
                    f"gpu_reserved_gb={gpu_memory['gpu_memory/reserved_gb']:.3f}  "
                    f"gpu_max_alloc_gb={gpu_memory['gpu_memory/max_allocated_gb']:.3f}"
                )
            print(
                f"step={step+1:5d}  swap_acc={swap_acc:.3f}  "
                f"work_mean={step_work_mean:.6e}  work_sem={step_work_sem:.3e}  "
                f"flow_loss={step_flow_loss:.6e}  train_flow={int(train_flow)}  "
                f"measurement_elapsed_s={elapsed:.2f}  total_elapsed_s={total_elapsed:.2f}  "
                f"dt_log_s={elapsed_since_last_log:.2f}  "
                f"parameter_name={pt.parameter_name}  "
                f"parameter_local=[{lmin:.6f}, {lmax:.6f}]"
                f"{memory_msg}",
                flush=True,
            )
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        step + 1,
                        int(stats["n_accepted"]),
                        int(stats["n_proposed"]),
                        float(swap_acc),
                        float(step_work_mean),
                        float(step_work_sem),
                        int(step_work_count),
                        float(step_flow_loss),
                        int(step_flow_applied_updates),
                        int(step_flow_skipped_updates),
                        int(step_flow_consecutive_nan),
                        int(train_flow),
                        json.dumps([float(v) for v in link_acceptance_step]),
                        json.dumps([float(v) for v in link_work_mean_step]),
                        json.dumps([float(v) for v in link_flow_loss_step]),
                        float(lmin),
                        float(lmax),
                        float(elapsed),
                        float(thermalization_elapsed),
                        float(elapsed),
                        float(total_elapsed),
                        float(gpu_memory.get("gpu_memory/allocated_gb", float("nan"))),
                        float(gpu_memory.get("gpu_memory/reserved_gb", float("nan"))),
                        float(gpu_memory.get("gpu_memory/max_allocated_gb", float("nan"))),
                        float(gpu_memory.get("gpu_memory/max_reserved_gb", float("nan"))),
                        float(gpu_memory.get("gpu_memory/free_gb", float("nan"))),
                        float(gpu_memory.get("gpu_memory/total_gb", float("nan"))),
                    ]
                )

    if rank == 0:
        measurement_elapsed = time.time() - t0
        total_elapsed = thermalization_elapsed + measurement_elapsed
        mean_swap_acc = (
            total_swap_accepted / total_swap_proposed
            if total_swap_proposed > 0
            else float("nan")
        )
        print(
            f"final_swap_acceptance_mean={mean_swap_acc:.6f}  "
            f"total_accepted={total_swap_accepted}  "
            f"total_proposed={total_swap_proposed}  "
            f"swap_attempts={total_swap_attempts}",
            flush=True,
        )
        print(
            f"timing_thermalization_s={thermalization_elapsed:.2f}  "
            f"timing_measurement_s={measurement_elapsed:.2f}  "
            f"timing_total_s={total_elapsed:.2f}",
            flush=True,
        )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["timing"] = {
            "thermalization_seconds": float(thermalization_elapsed),
            "measurement_seconds": float(measurement_elapsed),
            "total_seconds": float(total_elapsed),
        }
        final_gpu_memory = _rank0_cuda_memory_metrics(device, bool(args.log_gpu_memory))
        if final_gpu_memory:
            meta["gpu_memory"] = {
                key.removeprefix("gpu_memory/"): float(value)
                for key, value in final_gpu_memory.items()
            }
        if args.algorithm == "neorex":
            saved_flow_paths = []
            saved_flow_metadata_paths = []
            final_flow_path = out_dir / f"neorex_flow_final_nsteps{int(args.neorex_protocol_steps)}.pt"
            final_flow_path.parent.mkdir(parents=True, exist_ok=True)
            pt.save_flow(str(final_flow_path))
            saved_flow_paths.append(str(final_flow_path))
            saved_flow_metadata_paths.append(_write_neorex_flow_metadata(args, str(final_flow_path)))

            save_path = _resolve_neorex_flow_checkpoint_path(args, args.neorex_save_flow)
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                pt.save_flow(str(save_path))
                saved_flow_paths.append(str(save_path))
                saved_flow_metadata_paths.append(_write_neorex_flow_metadata(args, str(save_path)))
                print(f"neorex_flow_checkpoint=saved  path={save_path}", flush=True)

            meta["neorex_flow_checkpoint"] = {
                "load_mode": neorex_flow_load_mode,
                "load_path": neorex_flow_load_path,
                "saved_paths": saved_flow_paths,
                "metadata_paths": saved_flow_metadata_paths,
            }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if args.algorithm in {"orex", "neorex"}:
            run_work_mean, run_work_sem = _mean_sem_from_moments(
                total_work_sum,
                total_work_sum_sq,
                total_work_count,
            )
            print(
                f"final_work_mean={run_work_mean:.6e}  "
                f"final_work_sem={run_work_sem:.3e}  "
                f"work_count={total_work_count}",
                flush=True,
            )
        if args.algorithm == "neorex":
            run_flow_loss = (
                total_flow_loss_sum / total_flow_loss_count
                if total_flow_loss_count > 0
                else float("nan")
            )
            print(
                f"final_flow_loss_mean={run_flow_loss:.6e}  "
                f"flow_loss_count={total_flow_loss_count}  "
                f"flow_applied_updates={total_flow_applied_updates}  "
                f"flow_skipped_nan_grad_updates={total_flow_skipped_updates}",
                flush=True,
            )
        if wandb_run is not None:
            wandb_run.summary["timing/measurement_seconds"] = float(measurement_elapsed)
            wandb_run.summary["timing/total_seconds"] = float(total_elapsed)
            for key, value in final_gpu_memory.items():
                wandb_run.summary[key] = float(value)
            wandb_run.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
