import argparse
import copy
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

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
from neoqcd.mcmc import DefectMCMCAdapter, HBOR, quiesce_for_capture
import neoqcd.fused as fused
import neoqcd.fused_smearing as fused_smearing
import neoqcd.topology as topology
from neoqcd.ladder import parse_ladder, validate_ladder
from neoqcd.replica_history import rung_indices, summarize_history
from neoqcd.smearing import DefectCouplingLayer, ResidualNormalizingFlows, Smearing
from neoqcd.smearing_jacobian import LOGDET_VERSION
from neoqcd.sun_utils import plaquette_SUN
from neoqcd.theory import Wilson_action
from neoqcd.utils import (
    Defect,
    ProtocolBox,
    create_mask,
    defect_patch,
    residual_patch_buffer,
    translate_periodic_replica,
)

# Algorithms whose swap runs a flow (NEO-REX; TREX is NEO-REX without stochastic updates), and those
# whose swap runs a nonequilibrium protocol.
FLOW_ALGORITHMS = ("neorex", "trex")
NONEQ_ALGORITHMS = ("orex", "neorex", "trex")


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
    nf_layer_kind = str(args.nf_layer_type or "smearing").lower()
    return {
        "algorithm": "neorex",
        "nf_layer_type": nf_layer_kind,
        "smearing_jacobian_version": LOGDET_VERSION if nf_layer_kind == "smearing" else None,
        "explicit_backward": bool(args.explicit_backward),
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


def _write_neorex_flow_metadata(args: argparse.Namespace, saved_path: str, rank: int) -> str:
    path = Path(saved_path)
    meta_path = Path(f"{path}.meta.json")
    error = [None]
    if rank == 0:
        temporary_path = None
        try:
            fd, temporary_path = tempfile.mkstemp(
                prefix=f".{meta_path.name}.", suffix=".tmp", dir=meta_path.parent
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(_neorex_flow_metadata(args, str(path)), f, indent=2)
            os.replace(temporary_path, meta_path)
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"failed to write NEO-REX flow metadata: {error[0]}")
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


def _checkpoint_tensor_semantics(value: torch.Tensor) -> dict:
    tensor = torch.as_tensor(value).detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return {
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype),
        "sha256": digest,
    }


def _checkpoint_mlp_semantics(module: torch.nn.Sequential) -> list[dict]:
    layout = []
    for layer in module:
        entry = {"type": type(layer).__name__, "config": layer.extra_repr()}
        if isinstance(layer, torch.nn.Linear):
            entry.update(in_features=int(layer.in_features), out_features=int(layer.out_features))
        layout.append(entry)
    return layout


def _checkpoint_patch_semantics(flow_pars: FlowPars) -> dict:
    defect = flow_pars.defect
    return {
        "geometry": {
            "D": int(flow_pars.D),
            "N": int(flow_pars.N),
            "T": int(flow_pars.T),
            "L": int(flow_pars.L),
            "defect_size": int(defect.dsize),
            "crop_buffer": int(getattr(flow_pars, "patch_buffer", 2)),
        },
        "small_mask": _checkpoint_tensor_semantics(flow_pars.small_mask[-1]),
        "smeared_defect_mask": _checkpoint_tensor_semantics(
            flow_pars.smeared_defect_mask[-1]
        ),
    }


class DefectSNFProtocolLayer(torch.nn.Module):
    """
    Deterministic defect-coupling NF block compatible with NEOREXProtocolWrapper.
    """

    # CUDA graphs of the forward map and of the fixed-iteration inverse on the patch, keyed by the
    # batch shape (see _graphed). Off by default; --cuda-graphs turns it on. Class attributes, so
    # that subclasses with their own __init__ (the residual layer) inherit the eager default.
    use_cuda_graphs = False
    inverse_fixed_iterations = 10
    graph_logdet = "slogdet"  # or "lu": the tensor-only LU, if cuSOLVER cannot be captured

    def __init__(self, flow_pars: FlowPars):
        super().__init__()
        self.flow_pars = flow_pars
        self.smearing_jacobian_version = LOGDET_VERSION
        rho0 = torch.zeros(flow_pars.rho_shape, device=flow_pars.device, dtype=torch.float64)
        self.layer = DefectCouplingLayer(flow_pars, rho0, time=1.0)
        self._graphs = {}

    def set_frame(self, defect=None):
        """Where the defect sits in the configurations the layer receives: in the whole lattice
        (None, flow_pars.defect) or in a protocol box (utils.ProtocolBox.defect, --protocol-box). The
        map is the same; only the cut of the defect patch and its embedding move."""
        if defect is None:
            self.frame_pars = self.flow_pars
        else:
            self.frame_pars = copy.copy(self.flow_pars)
            self.frame_pars.defect = defect
        self._graphs = {}

    def _frame(self):
        # flow_pars with the defect coordinates of the input; subclasses without set_frame use flow_pars
        frame = self.__dict__.get("frame_pars")
        return self.flow_pars if frame is None else frame

    @property
    def checkpoint_semantics(self):
        smearing = self.layer.smearing
        if not smearing.use_hyper_smearing:
            return None
        hyper = smearing.hyper_smearing
        semantics = {
            "schema": 1,
            "family": "classic_hyper_smearing",
            **_checkpoint_patch_semantics(self.flow_pars),
            "mode": hyper.mode,
            "time": float(hyper.time),
            "time_embedding_dim": int(hyper.scalar_emb_dim),
            "embedding_frequencies": _checkpoint_tensor_semantics(hyper.freqs),
            "rho_shape_without_batch": [int(v) for v in hyper.rho_shape[1:]],
            "smearing_steps": int(hyper.smearing_steps),
            "mlp": _checkpoint_mlp_semantics(hyper.mlp),
            "normalize_by_nstep": bool(hyper.normalize_by_nstep),
            "normalization_divisor": int(hyper.nstep) if hyper.normalize_by_nstep else None,
            "rho_eps": float(hyper.rho_eps),
            "rho_max": float(hyper.rho_max),
            "scale_by_delta": bool(hyper.scale_by_delta),
            "class_masks": (
                _checkpoint_tensor_semantics(hyper.class_masks)
                if hasattr(hyper, "class_masks")
                else None
            ),
        }
        if hyper.mode in {"class", "per_site"}:
            semantics["class_unclassified_links"] = "frozen"
        return semantics

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
        defect = self._frame().defect
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

    def _run_blocks(self, kind, small, bc, delta_bc):
        """The patch map of `kind` ("forward"/"inverse") without any host-device sync, for capture."""
        smearing = self.layer.smearing
        rho = self.layer._rho(None)
        mask, dmask = self.flow_pars.small_mask[-1], self.flow_pars.smeared_defect_mask[-1]
        if kind == "forward":
            out, _, logdet = smearing._forward_blocks(small, mask, rho, dmasking=True, dmask=dmask, beta=bc, delta_beta=delta_bc)
            return out, logdet
        (out, _, logdet), _ = smearing._inverse_blocks(
            small, mask, rho, True, dmask, bc, delta_bc, 1e-12, 100, fixed_iterations=self.inverse_fixed_iterations
        )
        return out, logdet

    def _capture(self, kind, x, bc, delta_bc):
        """Capture the CUDA graph of the patch map `kind` for inputs shaped like x: the graph and its
        static tensors, or None when the capture fails (eager from then on)."""
        from neoqcd.smearing_jacobian import DeferredChecks, lu_logdet
        buffer = self.flow_pars.patch_buffer
        try:
            static_small = self._frame().defect.cut_defect(x, buffer=buffer).clone()
            static_bc, static_delta = bc.clone(), delta_bc.clone()
            old_jac_shape = self._set_local_jac_shape(int(x.shape[0]))
            try:
                # torch.linalg.slogdet captured fine on Leonardo (torch 2.0, CUDA 12.1); the
                # tensor-only LU is the alternative where cuSOLVER cannot be captured, since a
                # failed capture leaves the CUDA random generator unusable.
                with (lu_logdet() if self.graph_logdet == "lu" else contextlib.nullcontext()):
                    quiesce_for_capture(x.device)
                    stream = torch.cuda.Stream(device=x.device)
                    stream.wait_stream(torch.cuda.current_stream(device=x.device))
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            with DeferredChecks():
                                self._run_blocks(kind, static_small, static_bc, static_delta)
                    torch.cuda.current_stream(device=x.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        with DeferredChecks() as checks:
                            static_out, static_logdet = self._run_blocks(kind, static_small, static_bc, static_delta)
            finally:
                self.layer.smearing.jac_shape = old_jac_shape
        except Exception as err:  # noqa: BLE001 - any capture failure means eager from now on
            import warnings
            warnings.warn(f"CUDA graph capture of the smearing layer {kind} failed, running eagerly: "
                          f"{type(err).__name__}: {err}")
            self.use_cuda_graphs = False
            return None
        return graph, static_small, static_bc, static_delta, static_out, static_logdet, checks

    def _graphed(self, kind, x, bc, delta_bc):
        """Replay the CUDA graph of the patch map, capturing it on first use; the deferred checks
        (c1 of every block, inverse residual) are read once after the replay and a failed check
        repeats the call eagerly. Returns None when graphs are unavailable.

        The first use captures the map and its inverse together: on a rank whose chains all move up,
        the inverse is needed only steps later, and its capture (which quiesces the device) would
        then fall at a random step of the run, inside a profile window for instance."""
        if not hasattr(self, "_graphs"):
            self._graphs = {}
        shape = (tuple(x.shape), str(x.dtype))
        if (kind,) + shape not in self._graphs:
            for k in (kind, "inverse" if kind == "forward" else "forward"):
                if (k,) + shape not in self._graphs:
                    captured = self._capture(k, x, bc, delta_bc)
                    if captured is None:
                        return None
                    self._graphs[(k,) + shape] = captured
        graph, static_small, static_bc, static_delta, static_out, static_logdet, checks = self._graphs[(kind,) + shape]
        buffer = self.flow_pars.patch_buffer
        static_small.copy_(self._frame().defect.cut_defect(x, buffer=buffer))
        static_bc.copy_(bc)
        static_delta.copy_(delta_bc)
        graph.replay()
        series_valid, residual = checks.resolve()
        if not series_valid or (kind == "inverse" and not residual <= 1e-12):
            return None
        return self._embed_small_cfgs(static_out.clone(), x, buffer=buffer), static_logdet.clone()

    def forward(self, x: torch.Tensor, bc: torch.Tensor, delta_bc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = int(x.shape[0])
        bc = self._normalize_parameter(bc, bs, x.device)
        delta_bc = self._normalize_parameter(delta_bc, bs, x.device)
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        if self.use_cuda_graphs and x.is_cuda and not torch.is_grad_enabled():
            graphed = self._graphed("forward", x, bc, delta_bc)
            if graphed is not None:
                return graphed[0], graphed[1].to(dtype=work_dtype)
        old_jac_shape = self._set_local_jac_shape(bs)
        try:
            x_small, _, dlogj = self.layer(
                x,
                self._frame(),
                rho_layer=None,
                is_training=True,
                beta=bc,
                delta_beta=delta_bc,
            )
        finally:
            self.layer.smearing.jac_shape = old_jac_shape
        x_out = self._embed_small_cfgs(x_small, x, buffer=self.flow_pars.patch_buffer)
        logdet = dlogj.to(dtype=work_dtype)
        return x_out, logdet

    def _inverse_small(self, small_cfgs, bc, delta_bc, **kwargs):
        return self.layer.smearing.inverse(
            small_cfgs,
            self.flow_pars.small_mask[-1],
            self.layer._rho(None),
            dmasking=True,
            dmask=self.flow_pars.smeared_defect_mask[-1],
            beta=bc,
            delta_beta=delta_bc,
            **kwargs,
        )

    def inverse(self, x: torch.Tensor, bc: torch.Tensor, delta_bc: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of forward with the same bc and delta_bc: (x, log-Jacobian of the inverse map)."""
        bs = int(x.shape[0])
        bc = self._normalize_parameter(bc, bs, x.device)
        delta_bc = self._normalize_parameter(delta_bc, bs, x.device)
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        if self.use_cuda_graphs and x.is_cuda and not torch.is_grad_enabled() and not kwargs:
            graphed = self._graphed("inverse", x, bc, delta_bc)
            if graphed is not None:
                return graphed[0], graphed[1].to(dtype=work_dtype)
        small_cfgs = self._frame().defect.cut_defect(x, buffer=self.flow_pars.patch_buffer)
        small_cfgs, _, logdet = self._inverse_small(small_cfgs, bc, delta_bc, **kwargs)
        return self._embed_small_cfgs(small_cfgs, x, buffer=self.flow_pars.patch_buffer), logdet.to(dtype=work_dtype)


class ResidualDefectSNFProtocolLayer(DefectSNFProtocolLayer):
    """
    Defect-local residual normalizing-flow block.

    ResidualNormalizingFlows already returns a real tangent-space logdet, so no
    extra factor of two is applied here.
    """

    def __init__(self, flow_pars: FlowPars):
        torch.nn.Module.__init__(self)
        self.flow_pars = flow_pars
        self.smearing_jacobian_version = None
        self.layer = ResidualNormalizingFlows(flow_pars, time=1.0)

    @property
    def checkpoint_semantics(self):
        residual = self.layer
        semantics = {
            "schema": 1,
            "family": "residual_defect_flow",
            **_checkpoint_patch_semantics(self.flow_pars),
            "time": float(residual.time),
            "steps": int(residual.steps),
            "parameterization": residual.parameterization,
            "include_imag": bool(residual.include_imag),
            "quadratic": bool(residual.use_quadratic),
            "coeff_max": float(residual.coeff_max),
            "n_pt": int(residual.n_pt),
            "mask_mod": int(residual.mask_mod),
            "n_features": int(residual.n_features),
            "coeffs_per_step": int(residual.coeffs_per_step),
        }
        if residual.mode != "shared":  # absent from flows saved before the class mode
            semantics["mode"] = residual.mode
        if residual.parameterization == "hyper":
            semantics.update(
                time_embedding_dim=2 * int(residual.freqs.numel()),
                embedding_frequencies=_checkpoint_tensor_semantics(residual.freqs),
                mlp=_checkpoint_mlp_semantics(residual.mlp),
            )
        return semantics

    def forward(self, x: torch.Tensor, bc: torch.Tensor, delta_bc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = int(x.shape[0])
        bc = self._normalize_parameter(bc, bs, x.device)
        delta_bc = self._normalize_parameter(delta_bc, bs, x.device)
        work_dtype = x.real.dtype if torch.is_complex(x) else x.dtype

        small_cfgs = self._frame().defect.cut_defect(x, buffer=self.flow_pars.patch_buffer)
        small_cfgs, _, logdet = self.layer(
            small_cfgs,
            self.flow_pars.small_mask[-1],
            dmasking=True,
            dmask=self.flow_pars.smeared_defect_mask[-1],
            beta=bc,
            delta_beta=delta_bc,
        )
        return self._embed_small_cfgs(small_cfgs, x, buffer=self.flow_pars.patch_buffer), logdet.to(dtype=work_dtype)

    def _inverse_small(self, small_cfgs, bc, delta_bc, **kwargs):
        return self.layer.inverse(
            small_cfgs,
            self.flow_pars.small_mask[-1],
            dmasking=True,
            dmask=self.flow_pars.smeared_defect_mask[-1],
            beta=bc,
            delta_beta=delta_bc,
            **kwargs,
        )


def _protocol_box(args, defect, protocol_patch, flow_buffer, device, rank):
    """The utils.ProtocolBox of --protocol-box, or None (flag off, or a box as large as the lattice)."""
    if not bool(args.protocol_box):
        return None
    if protocol_patch is None:
        raise ValueError("--protocol-box needs --protocol-patch-halfsize")
    box = ProtocolBox(defect, args.protocol_patch_halfsize, flow_buffer=flow_buffer, device=device)
    if rank == 0:
        state = "off (the box covers the lattice)" if box.covers_lattice else "on"
        print(f"protocol_box={state}  protocol_box_shape={box.describe()}", flush=True)
    return None if box.covers_lattice else box


def _build_neorex_flow_pars(
    args: argparse.Namespace,
    device: torch.device,
    mask: torch.Tensor,
    defect: Defect,
) -> FlowPars:
    residual = str(getattr(args, "nf_layer_type", "") or "smearing").lower() == "residual"
    buffer = residual_patch_buffer(args.residual_n_pt) if residual else 2
    if args.residual_n_pt < 0:
        raise ValueError("--residual-n-pt must be non-negative")
    patch = defect_patch(args.D, args.T, args.L, args.defect_size, args.time_slice, args.space_slice,
                         buffer=buffer, device=device)

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
        explicit_backward=args.explicit_backward,
        residual_include_imag=args.residual_include_imag,
        residual_quadratic=args.residual_quadratic,
        residual_coeff_init=args.residual_coeff_init,
        residual_coeff_max=args.residual_coeff_max,
        residual_n_pt=args.residual_n_pt,
        residual_mask_mod=args.residual_mask_mod,
        residual_mode=args.residual_mode,
        defect=defect,
        **patch,
    )


# Arguments that a resumed run may change: run length and time limits, checkpoints, logging, profiling and
# the cache of the thermalization (skipped on resume); every other argument defines the run.
RESUME_FREE_ARGS = {
    "steps", "max_run_minutes", "save_resume", "resume_every_minutes", "resume", "log_every", "profile_steps",
    "profile_python", "log_gpu_memory", "replica_history_save_minutes", "wandb", "wandb_project",
    "wandb_entity", "wandb_run_name", "thermal_steps", "use_cfg_cache", "cfg_cache_dir", "cfg_cache_tag",
    "overwrite_cfg_cache", "main_dir", "resume_dir",
}


def _resume_semantics(args: argparse.Namespace, world_size: int) -> dict:
    values = {key: value for key, value in vars(args).items() if key not in RESUME_FREE_ARGS}
    values["world_size"] = int(world_size)
    return json.loads(json.dumps(values, sort_keys=True, default=str))


def _resume_root(args: argparse.Namespace, out_dir: Path) -> Path:
    """Where the resume checkpoints of the run in out_dir go (see --resume-dir)."""
    if args.resume_dir:
        return Path(args.resume_dir)
    scratch = os.environ.get("SCRATCH")
    if scratch:
        path = str(Path(out_dir).resolve())
        return Path(scratch) / "neoqcd_resume" / f"{Path(out_dir).name}_{hashlib.sha1(path.encode()).hexdigest()[:8]}"
    return Path(out_dir) / "resume"


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _truncate_csv(path: Path, last_step: int) -> None:
    """Keep the header and the rows of steps <= last_step (rows written after the resume checkpoint go)."""
    tmp = path.with_name(f".{path.name}.resume.tmp")
    with open(path, newline="", encoding="utf-8") as source, open(tmp, "w", newline="", encoding="utf-8") as target:
        reader, writer = csv.reader(source), csv.writer(target)
        writer.writerow(next(reader))
        for row in reader:
            if row and int(row[0]) <= int(last_step):
                writer.writerow(row)
    os.replace(tmp, path)


def _default_output_dir(main_dir: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_run_name = run_name or f"pt_obc_{timestamp}"
    return Path(main_dir) / "results" / "pt_obc" / final_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard PT on OBC/defect coupling for SU(3)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU + gloo backend")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--algorithm", type=str, default="pt", choices=["pt", "orex", "neorex", "trex"],
                        help="trex: NEO-REX with no stochastic updates, the swap is proposed by the flow alone "
                             "(transformed replica exchange); --neorex-* flags apply")

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
    parser.add_argument(
        "--parameter-ladder",
        default=None,
        help="Explicit couplings, comma-separated or a JSON file written by main/optimal_ladder.py: one per rank, "
        "strictly increasing; its ends set the ladder range (and must match --parameter-min/--parameter-max if "
        "given). Default: evenly spaced from min to max.",
    )
    parser.add_argument("--bc-min", type=float, default=0.0)
    parser.add_argument("--bc-max", type=float, default=1.0)
    # Alias compatibile con vecchi launcher/template (qui indica il range del parametro PT, non beta fisica).
    parser.add_argument("--beta-min", dest="beta_min_alias", type=float, default=None)
    parser.add_argument("--beta-max", dest="beta_max_alias", type=float, default=None)
    parser.add_argument(
        "--swap-parity",
        type=str,
        default="alternating",
        choices=["even", "odd", "alternating", "even_odd"],
        help="Standard PT: which pairs each step tries. even_odd = all even then all odd pairs every step (DEO, "
        "as the O-REX/NEO-REX even_odd schedule); alternating = one parity per step.",
    )
    parser.add_argument("--acceptance-rule", type=str, default="exact", choices=["exact", "linear", "auto"])
    parser.add_argument("--orex-protocol-steps", type=int, default=8)
    parser.add_argument("--orex-mcmc-steps-per-step", type=int, default=1)
    parser.add_argument("--orex-schedule", type=str, default="even_odd", choices=["even_odd", "alternating", "even", "odd"])
    parser.add_argument(
        "--protocol-patch-halfsize",
        type=int,
        default=None,
        help="Restrict the HB+OR updates along the O-REX/NEO-REX protocols to the links of a box around the "
        "defect: 0 is the smallest box holding every link of the plaquettes that depend on bc (time slices "
        "time-slice..time-slice+1, spatial range space-slice-1..space-slice+defect-size), each unit extends "
        "it by one site in every direction. Default: the whole lattice. The equilibrium sweeps between "
        "swaps always cover the whole lattice.",
    )
    parser.add_argument(
        "--protocol-box",
        action="store_true",
        help="Run the protocol of each O-REX/NEO-REX swap on a box around the defect (utils.ProtocolBox): "
        "the protocol patch with a one-site frame, the cut of the defect flow layer and the plaquettes of "
        "the action change, cut once per swap and written back into the accepted chains. The links outside "
        "the patch never change along a protocol, so this is the protocol of the whole lattice, apart from "
        "the random numbers. Needs --protocol-patch-halfsize; NEO-REX with a smearing or residual layer.",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Replay CUDA graphs of the HBOR sweeps (equilibrium and protocol) and of the defect smearing "
        "layer, captured on first use; eager execution is the fallback when a capture fails.",
    )
    parser.add_argument(
        "--fused-kernels",
        action="store_true",
        help="Run every HB+OR sweep (equilibrium and protocol) through the fused CUDA kernel of neoqcd.fused, "
        "and the stout steps of the smearing layers where no gradient is recorded (forward and fixed-iteration "
        "inverse) through those of neoqcd.fused_smearing (built with nvcc at the start of the run, cached in "
        "$NEOQCD_KERNEL_CACHE). Equal to the torch paths to rounding, so statistically, not bitwise, "
        "equivalent runs; the torch paths run where a kernel does not apply.",
    )
    parser.add_argument(
        "--graph-logdet",
        choices=("slogdet", "lu"),
        default="slogdet",
        help="How the captured smearing layer computes log|det| of the stout Jacobians: torch.linalg.slogdet "
        "or the tensor-only LU (for a CUDA/cuSOLVER that cannot be captured).",
    )
    parser.add_argument(
        "--protocol-symmetric-updates",
        action="store_true",
        help="Along the protocols apply the HB+OR sweep or its reverse (OR sweeps then HB, blocks backwards) "
        "with probability 1/2, so every update kernel is self-adjoint as the swap's detailed balance requires.",
    )
    parser.add_argument("--neorex-protocol-steps", type=int, default=0)
    parser.add_argument("--neorex-heatbath-steps-per-step", type=int, default=0)
    parser.add_argument(
        "--neorex-work-mode", type=str, default="logdet", choices=["logdet"],
        help="Built-in NF layers return logdet; work is computed as the action change minus logdet.",
    )
    parser.add_argument("--neorex-flow-lr", type=float, default=1e-3)
    parser.add_argument("--neorex-grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--neorex-max-consecutive-nan-grad-updates", type=int, default=5)
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
    parser.add_argument("--hyper-smearing-mode", type=str, default="per_link", choices=["shared", "per_link", "class", "per_site"])
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
    parser.add_argument("--residual-n-pt", type=int, default=0,
                        help="Frozen-link convolutions (rectangles and chairs); widens the defect patch buffer to 2 + n")
    parser.add_argument("--residual-mask-mod", type=int, choices=(2, 4), default=2)
    parser.add_argument("--residual-mode", choices=("shared", "class"), default="shared",
                        help="class: residual coefficients per defect class, unclassified links untransformed")

    # Run control
    parser.add_argument("--thermal-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--max-run-minutes",
        type=float,
        default=0.0,
        help="Stop the measurement before the step that would end after this many minutes from the start of "
        "main (projected with 1.5 times the mean step time) and write every output as at the end of a full run; "
        "0 = off. Set it a few minutes below the job time limit.",
    )
    parser.add_argument(
        "--save-resume",
        action="store_true",
        help="Write a resume checkpoint (<output-dir>/resume/) when --max-run-minutes stops the run, every "
        "--resume-every-minutes, and at the end: configurations, PT state, NEO-REX flow and optimizer, random "
        "generators, totals and histories, so that --resume continues the run exactly.",
    )
    parser.add_argument(
        "--resume-every-minutes",
        type=float,
        default=60.0,
        help="With --save-resume: also write a checkpoint after this many wall-clock minutes since the last "
        "(0: only at the stop and at the end).",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Directory of the resume checkpoints. Default: $SCRATCH/neoqcd_resume/<output-dir name>_<hash of its "
        "path> when $SCRATCH is set (Leonardo: no backup, files purged after ~40 days, fine between jobs), "
        "otherwise <output-dir>/resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the run in --output-dir from its last resume checkpoint, to --steps (a larger --steps "
        "extends a finished run). The arguments that define the run must equal the original ones.",
    )
    parser.add_argument("--sweeps-per-step", type=int, default=1)
    parser.add_argument("--sweep", type=int, default=None, help="Alias of --sweeps-per-step")
    parser.add_argument("--swap-every", type=int, default=1)
    parser.add_argument(
        "--periodic-translation",
        action="store_true",
        help="After every round of swaps, translate the configuration of the periodic replica (bc = 1) by one "
        "site along a random positive direction, as in PTBC: the defect explores the whole lattice.",
    )
    parser.add_argument("--orsteps", type=int, default=4)
    parser.add_argument("--updates-per-layer", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Profile this many measurement steps (from the second one) on rank 0 with torch.profiler and "
        "write the operator tables and the time of each phase to <output-dir>/step_profile.txt.",
    )
    parser.add_argument(
        "--profile-python",
        action="store_true",
        help="With --profile-steps, also run cProfile over the same steps on rank 0 and write the Python "
        "functions by own and by cumulative time to <output-dir>/step_cprofile.txt: where the host time "
        "goes outside the torch operators (cProfile slows Python calls, so the phases run slower).",
    )
    parser.add_argument(
        "--save-link-work-values",
        action="store_true",
        help="For O-REX/NEO-REX, save every proposed swap work by ladder link to link_work_values.npz.",
    )
    parser.add_argument(
        "--save-replica-history",
        dest="save_replica_history",
        action="store_true",
        default=True,
        help="Save the rung of every chain slot and its plaquette after each step to replica_history.npz "
        "(rewritten during the run, see --replica-history-save-minutes, and at the end), and report round "
        "trips and the plaquette tau_int in run_meta.json.",
    )
    parser.add_argument("--no-save-replica-history", dest="save_replica_history", action="store_false")
    parser.add_argument(
        "--topology-every",
        type=int,
        default=0,
        help="Measure the topological charge every this many steps (0: never), after --cooling-steps sweeps "
        "of Wilson cooling, with the clover definition (neoqcd.topology, as in the SNF paper); stored in "
        "replica_history.npz and summarized in run_meta.json (chi and tau_int(Q^2) of the periodic replica, "
        "the top rung). With --fused-kernels the cooling and the charge run through kernels.",
    )
    parser.add_argument("--cooling-steps", type=int, default=60, help="Wilson cooling sweeps before Q (SNF paper: 60)")
    parser.add_argument(
        "--topology-replicas",
        choices=("top", "all"),
        default="top",
        help="Measure Q on the chains of the top rung (the periodic replica, bc = parameter-max) only, or on every "
        "chain of every rung (batch-size cooled configurations per rank instead of ~batch-size/n_rep).",
    )
    parser.add_argument(
        "--replica-history-save-minutes",
        type=float,
        default=10.0,
        help="Rewrite replica_history.npz during the run at most once per this many wall-clock minutes, at a "
        "log step (0: at every log step, as before). Each rewrite compresses the whole history so far, so "
        "rewriting it every --log-every steps made the cost per step grow with the steps done (+50%% at "
        "3000 steps of 8^4). The history is written at the end of the run in any case, a stop by "
        "--max-run-minutes included; the rewrites only cover a job killed before the end.",
    )
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
    parser.add_argument("--nf-layer-type", type=str, default="", choices=["", "smearing", "residual", "cnf"])
    parser.add_argument("--cnf-checkpoint", help="Field checkpoint trained by main_cnf_rex.py")
    parser.add_argument("--cnf-integration-steps", type=int, default=4)
    parser.add_argument("--nf-layers-per-step", type=int, default=0)
    parser.add_argument("--explicit-backward", action="store_true")
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
    hbor=None,
    rand=None,
) -> torch.Tensor:
    """One HB + OR sweep of every local chain at its own bc.

    With `hbor` (one HBOR object) the whole batch is swept in one call with a per-chain defect
    coupling; without it each ladder label is swept separately with hbor_by_idx (the historical
    path: with up to one sub-batch per rung it costs up to world_size dispatch-bound sweeps).
    `rand` (whole-batch heat-bath random numbers) lets the two paths be compared exactly.
    """
    if hbor is not None:
        hbor.set_defect_parameter(local_parameters.to(dtype=torch.float64))
        return hbor(cfgs, mask, defect_mask, rand=rand)
    idx = _parameter_indices(local_parameters, parameter_ladder)
    out = cfgs
    for k in torch.unique(idx):
        k_int = int(k.item())
        sel = torch.nonzero(idx == k_int, as_tuple=False).squeeze(-1)
        batch_k = out.index_select(0, sel)
        batch_k = hbor_by_idx[k_int](batch_k, mask, defect_mask, rand=None if rand is None else rand.index_select(0, sel))
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
    if args.algorithm in FLOW_ALGORITHMS:
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
    run_t0 = time.time()
    if args.algorithm in FLOW_ALGORITHMS and args.nf_layer_type == "cnf":
        args.neorex_eval_flow = True
    state = setup_runtime(args)
    rank, world_size, device = state.rank, state.world_size, state.device
    effective_acceptance_rule = (
        "exact" if args.algorithm in NONEQ_ALGORITHMS else args.acceptance_rule
    )

    parameter_min = args.parameter_min
    parameter_max = args.parameter_max
    if parameter_min is None:
        parameter_min = args.beta_min_alias if args.beta_min_alias is not None else args.bc_min
    if parameter_max is None:
        parameter_max = args.beta_max_alias if args.beta_max_alias is not None else args.bc_max
    explicit_ladder = None
    if args.parameter_ladder is not None:
        explicit_ladder = validate_ladder(
            parse_ladder(args.parameter_ladder), world_size, args.parameter_min, args.parameter_max
        )
        parameter_min, parameter_max = explicit_ladder[0], explicit_ladder[-1]
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
    # --resume: the last complete checkpoint of <output-dir>/resume ({"step", "dir"}); run_state (rank 0) holds
    # the totals and histories, rank_state (every rank) its configurations, PT state and random generators
    resume_info, run_state, rank_state = None, None, None
    # one path for every rank, decided on rank 0
    resume_root_box = [str(_resume_root(args, out_dir)) if rank == 0 else None]
    dist.broadcast_object_list(resume_root_box, src=0)
    resume_root = Path(resume_root_box[0])
    if rank == 0 and (bool(args.save_resume) or bool(args.resume)):
        print(f"resume_dir={resume_root}", flush=True)
    if bool(args.resume):
        if not args.output_dir:
            raise ValueError("--resume needs the --output-dir of the run to continue")
        found = [None]
        if rank == 0:
            latest = resume_root / "latest.json"
            if not latest.is_file():
                found = [f"no resume checkpoint in {latest.parent} (the run needs --save-resume)"]
            else:
                info = json.loads(latest.read_text(encoding="utf-8"))
                run_state = torch.load(Path(info["dir"]) / "run.pt", map_location="cpu", weights_only=False)
                expected = _resume_semantics(args, world_size)
                if run_state["semantics"] != expected:
                    changed = sorted(k for k in set(expected) | set(run_state["semantics"])
                                     if expected.get(k) != run_state["semantics"].get(k))
                    found = [f"the run arguments differ from those of the checkpoint: {changed}"]
                elif int(info["step"]) > int(args.steps):
                    found = [f"the checkpoint is at step {info['step']}, beyond --steps {args.steps}"]
                else:
                    found = [info]
        dist.broadcast_object_list(found, src=0)
        if isinstance(found[0], str):
            raise ValueError(f"--resume: {found[0]}")
        resume_info = found[0]
        rank_state = torch.load(Path(resume_info["dir"]) / f"rank{rank}.pt", map_location="cpu", weights_only=False)
        if int(rank_state["step"]) != int(resume_info["step"]) or int(rank_state["world_size"]) != world_size:
            raise ValueError(f"--resume: inconsistent checkpoint file for rank {rank} in {resume_info['dir']}")
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

    if rank == 0 and resume_info is not None:
        # continue the outputs of the run: rows of steps after the checkpoint (if any) are dropped
        with open(meta_path, encoding="utf-8") as f:
            resumed_meta = json.load(f)
        resumed_meta.setdefault("resumes", []).append(
            {"from_step": int(resume_info["step"]), "steps": int(args.steps), "started": datetime.now().isoformat()}
        )
        resumed_meta["args"] = vars(args)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(resumed_meta, f, indent=2, default=str)
        _truncate_csv(csv_path, int(resume_info["step"]))
    elif rank == 0:
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
                    "link_accepted_json",
                    "link_proposed_json",
                    "link_acceptance_json",
                    "link_acceptance_cumulative_json",
                    "link_work_mean_json",
                    "link_work_sem_json",
                    "link_work_count_json",
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

    protocol_patch = None
    if args.protocol_patch_halfsize is not None:
        protocol_patch = defect.protocol_patch_mask(args.protocol_patch_halfsize, device=device)
        if rank == 0:
            print(
                f"protocol_patch_halfsize={int(args.protocol_patch_halfsize)}  "
                f"protocol_patch_links={int((mask * protocol_patch).sum().item())}/{int(mask.sum().item())}",
                flush=True,
            )
    protocol_updates = dict(patch_mask=protocol_patch, symmetric_updates=bool(args.protocol_symmetric_updates))

    if args.algorithm == "orex":
        pt_cfg = OREXPTConfig(
            batch_size=args.batch_size,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            parameter_name="bc",
            parameter_dtype=torch.float64,
            swap_parity=args.swap_parity,
            acceptance_rule="exact",
            parameter_ladder=explicit_ladder,
            orex_protocol_steps=args.orex_protocol_steps,
            orex_mcmc_steps_per_step=args.orex_mcmc_steps_per_step,
            orex_schedule=args.orex_schedule,
            save_work_values=bool(args.save_link_work_values),
        )
        pt = OutOfEquilibriumParallelTempering(pt_cfg, rank, world_size, device)
        orex_mcmc = DefectMCMCAdapter(
            flow_pars=flow_pars,
            beta=args.beta,
            defect_mask=defect_mask,
            batch_size=args.batch_size,
            parameter_name="bc",
            **protocol_updates,
        )
        protocol_box = _protocol_box(args, defect, protocol_patch, None, device, rank)
        protocol_mcmc = orex_mcmc if protocol_box is None else orex_mcmc.on_box(protocol_box)
        pt.protocol_box = protocol_box
    elif args.algorithm in FLOW_ALGORITHMS:
        if int(args.neorex_protocol_steps) <= 0:
            raise ValueError(f"For --algorithm {args.algorithm} set --neorex-protocol-steps >= 1")
        if int(args.neorex_heatbath_steps_per_step) < 0:
            raise ValueError("--neorex-heatbath-steps-per-step must be >= 0")
        if args.algorithm == "trex" and int(args.neorex_heatbath_steps_per_step) != 0:
            raise ValueError("--algorithm trex has no stochastic updates: drop --neorex-heatbath-steps-per-step")

        nf_layer_kind = str(args.nf_layer_type or "smearing").lower()
        neorex_flow_pars = None if nf_layer_kind == "cnf" else _build_neorex_flow_pars(args, device, mask, defect)
        if nf_layer_kind == "cnf":
            from neoqcd.cnf import CNFProtocolLayer
            from neoqcd.cnf_io import load_cnf_checkpoint
            if not args.cnf_checkpoint or args.cnf_integration_steps < 1:
                raise ValueError("cnf requires --cnf-checkpoint and positive --cnf-integration-steps")
            if args.train_steps > 0:
                raise ValueError("Train CNF fields using main/main_cnf_rex.py before NEO-REX evaluation")
            if args.neorex_work_mode != "logdet":
                raise ValueError("CNF returns logdet: use --neorex-work-mode logdet")
            expected = {key: getattr(args, key) for key in ("D", "T", "L", "beta", "defect_size", "time_slice", "space_slice")}
            expected["domain"] = "defect"
            field, _, _, _ = load_cnf_checkpoint(args.cnf_checkpoint, device, expected, (parameter_min, parameter_max))
            nf_layer = CNFProtocolLayer(field, n_steps=args.cnf_integration_steps)
        elif nf_layer_kind == "residual":
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
            **protocol_updates,
        )
        if bool(args.protocol_box) and nf_layer_kind == "cnf":
            raise ValueError("--protocol-box: the CNF layer acts on the whole lattice")
        protocol_box = _protocol_box(
            args, defect, protocol_patch, None if neorex_flow_pars is None else neorex_flow_pars.patch_buffer,
            device, rank,
        )
        protocol_mcmc = neorex_mcmc
        if protocol_box is not None:
            protocol_mcmc = neorex_mcmc.on_box(protocol_box)
            nf_layer.set_frame(protocol_box.defect)
        neorex_flow = NEOREXProtocolWrapper(
            nf_layer=nf_layer,
            theory=protocol_mcmc,
            n_protocol_steps=args.neorex_protocol_steps,
            work_mode=args.neorex_work_mode,
            stochastic_update=protocol_mcmc if args.neorex_heatbath_steps_per_step > 0 else None,
            stochastic_steps_per_step=args.neorex_heatbath_steps_per_step,
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
            parameter_ladder=explicit_ladder,
            orex_protocol_steps=args.neorex_protocol_steps,
            orex_mcmc_steps_per_step=args.neorex_heatbath_steps_per_step,
            orex_schedule=args.orex_schedule,
            save_work_values=bool(args.save_link_work_values),
            flow_lr=args.neorex_flow_lr,
            flow_train=nf_layer_kind != "cnf",
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
        pt.protocol_box = protocol_box
        orex_mcmc = neorex_mcmc  # whole lattice: the dS/dc probe of the replica history
    else:
        pt_cfg = PTConfig(
            batch_size=args.batch_size,
            parameter_min=parameter_min,
            parameter_max=parameter_max,
            parameter_name="bc",
            parameter_dtype=torch.float64,
            swap_parity=args.swap_parity,
            acceptance_rule=args.acceptance_rule,
            parameter_ladder=explicit_ladder,
        )
        pt = ParallelTempering(pt_cfg, rank, world_size, device)
        orex_mcmc = None
        protocol_mcmc = None

    if explicit_ladder is not None:
        parameter_ladder = torch.tensor(explicit_ladder, device=device, dtype=pt_cfg.parameter_dtype)
    else:
        parameter_ladder = torch.linspace(
            pt_cfg.parameter_min,
            pt_cfg.parameter_max,
            world_size,
            device=device,
            dtype=pt_cfg.parameter_dtype,
        )
    if rank == 0:
        print(
            f"parameter_ladder={'explicit' if explicit_ladder is not None else 'linear'}  "
            f"parameter_ladder_json={json.dumps([round(float(v), 10) for v in parameter_ladder.cpu()])}",
            flush=True,
        )
    n_links = max(0, world_size - 1)
    periodic_value = None
    if bool(args.periodic_translation):
        periodic_value = float(parameter_ladder[-1].item())
        if periodic_value != 1.0:
            raise ValueError(f"--periodic-translation needs the periodic coupling bc = 1 on the ladder, got max {periodic_value}")
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
    # The equilibrium sweep runs on the whole local batch at once, with the bc of each chain.
    hbor_batched = HBOR(flow_pars, beta=args.beta, defect_par=float(parameter_min))
    hbor_batched.use_cuda_graphs = bool(args.cuda_graphs)
    for adapter in (orex_mcmc, protocol_mcmc):
        if adapter is not None:
            adapter._batch_hbor.use_cuda_graphs = bool(args.cuda_graphs)
    if args.algorithm in FLOW_ALGORITHMS and type(nf_layer) is DefectSNFProtocolLayer:
        nf_layer.use_cuda_graphs = bool(args.cuda_graphs)
        nf_layer.graph_logdet = str(args.graph_logdet)
    if bool(args.fused_kernels):
        # Every HBOR and every Smearing of the run (class attributes). Rank 0 builds the libraries first,
        # the others then load them from the shared cache instead of compiling them concurrently. Loading
        # also initializes each library's CUDA runtime and loads its kernels on this device, before any
        # CUDA graph capture of the smearing layer (--cuda-graphs) launches them.
        HBOR.use_fused_kernels = True
        Smearing.use_fused_kernels = True
        fused_status = smearing_status = (False, f"device {device.type}")
        if device.type == "cuda":
            if rank == 0:
                fused_status = fused.backend_status("cuda", device)
                smearing_status = fused_smearing.backend_status("cuda", device)
            if world_size > 1:
                dist.barrier()
            if rank != 0:
                fused_status = fused.backend_status("cuda", device)
                smearing_status = fused_smearing.backend_status("cuda", device)
        if rank == 0:
            ok, where = fused_status
            smearing_ok, smearing_where = smearing_status
            print(f"fused_kernels=on  hbor={'cuda' if ok else 'torch'}  hbor_library={where}  "
                  f"smearing={'cuda' if smearing_ok else 'torch'}  smearing_library={smearing_where}", flush=True)
    if rank == 0 and bool(args.cuda_graphs):
        print("cuda_graphs=on", flush=True)
        if world_size > 1 and os.environ.get("NCCL_ASYNC_ERROR_HANDLING", "1") != "0":
            print(
                "WARNING: --cuda-graphs with NCCL: export NCCL_ASYNC_ERROR_HANDLING=0 in the job, otherwise the "
                "NCCL cleanup thread can abort the process during a graph capture (torch 2.0 has no thread-local "
                "capture mode); the captures also wait 1.5 s for that thread's queue to drain.",
                flush=True,
            )

    cache_enabled = bool(args.use_cfg_cache) and int(args.thermal_steps) > 0 and resume_info is None
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
    if resume_info is not None:
        cfgs = rank_state["cfgs"].to(device)
        thermal_steps_executed = 0
        cache_mode = "resume"
    elif cache_enabled and cache_mode == "load":
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
                hbor=hbor_batched,
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
    total_link_work_sum_sq = [0.0] * n_links
    total_link_work_count = [0] * n_links
    saved_link_work_values = [[] for _ in range(n_links)]
    history_labels = []  # rank 0: (world_size, batch) rung index of every chain slot after each step
    history_plaquette = []  # rank 0: (world_size, batch) plaquette of every chain slot after each step
    history_defect_action = []  # rank 0: (world_size, batch) dS/dc of every chain slot at its coupling
    history_topology = []  # rank 0: (world_size, batch) Q after cooling (NaN where not measured), --topology-every
    history_topology_steps = []  # rank 0: index of the step after which each Q was measured
    topology_every = int(args.topology_every)
    if topology_every < 0 or int(args.cooling_steps) < 0:
        raise ValueError("--topology-every and --cooling-steps must be non-negative")
    if topology_every > 0 and not bool(args.save_replica_history):
        raise ValueError("--topology-every stores Q in the replica history: drop --no-save-replica-history")
    history_path = out_dir / "replica_history.npz"
    # dS/dc on the bc-dependent plaquettes, for the metric of the optimal ladder (main/optimal_ladder.py).
    defect_probe = None
    if bool(args.save_replica_history):
        defect_probe = orex_mcmc if orex_mcmc is not None else DefectMCMCAdapter(
            flow_pars=flow_pars,
            beta=args.beta,
            defect_mask=defect_mask,
            batch_size=args.batch_size,
            parameter_name="bc",
        )

    def save_replica_history():
        """Rank 0 writes the history so far; a run stopped by the time limit keeps its round trips."""
        np.savez_compressed(
            history_path,
            labels=np.stack(history_labels, axis=0).astype(np.int8),
            plaquette=np.stack(history_plaquette, axis=0),
            defect_action=np.stack(history_defect_action, axis=0),
            parameter_ladder=parameter_ladder.detach().cpu().numpy().astype(np.float64),
            swap_every=np.asarray(int(args.swap_every)),
            **({"topological_charge": np.stack(history_topology, axis=0),
                "topology_steps": np.asarray(history_topology_steps, dtype=np.int64),
                "cooling_steps": np.asarray(int(args.cooling_steps))} if history_topology else {}),
        )

    def _resume_totals() -> dict:
        return {
            "total_swap_accepted": total_swap_accepted, "total_swap_proposed": total_swap_proposed,
            "total_swap_attempts": total_swap_attempts, "total_work_sum": total_work_sum,
            "total_work_sum_sq": total_work_sum_sq, "total_work_count": total_work_count,
            "total_flow_loss_sum": total_flow_loss_sum, "total_flow_loss_count": total_flow_loss_count,
            "total_flow_applied_updates": total_flow_applied_updates,
            "total_flow_skipped_updates": total_flow_skipped_updates,
            "total_link_accepted": total_link_accepted, "total_link_proposed": total_link_proposed,
            "total_link_work_sum": total_link_work_sum, "total_link_work_sum_sq": total_link_work_sum_sq,
            "total_link_work_count": total_link_work_count, "saved_link_work_values": saved_link_work_values,
            "history_labels": history_labels, "history_plaquette": history_plaquette,
            "history_defect_action": history_defect_action, "history_topology": history_topology,
            "history_topology_steps": history_topology_steps,
        }

    def write_resume(completed: int) -> None:
        """Resume checkpoint after `completed` steps: one directory per checkpoint, made valid by latest.json
        (written last, atomically), so that a job killed while writing leaves the previous one usable."""
        root = resume_root
        target = root / f"step_{int(completed):08d}"
        if rank == 0:
            target.mkdir(parents=True, exist_ok=True)
            (root / "output_dir.txt").write_text(str(Path(out_dir).resolve()) + "\n", encoding="utf-8")
        dist.barrier()
        state_of_rank = {
            "step": int(completed), "rank": rank, "world_size": world_size, "cfgs": cfgs.detach().cpu(),
            "local_parameters": pt.local_parameters.detach().cpu(), "pt_step": int(getattr(pt, "_pt_step", 0)),
            "parameter_matrix": pt.parameter_matrix.detach().cpu() if rank == 0 else None,
            "rng": {"cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
        }
        if args.algorithm in FLOW_ALGORITHMS:
            state_of_rank.update(flow=pt.flow.state_dict(), optimizer=pt.optimizer.state_dict(),
                                 consecutive_nan=int(pt.consecutive_nan_grad_updates))
        _atomic_torch_save(state_of_rank, target / f"rank{rank}.pt")
        if rank == 0:
            _atomic_torch_save({"step": int(completed), "semantics": _resume_semantics(args, world_size),
                                "totals": _resume_totals(), "thermalization_seconds": float(thermalization_elapsed),
                                "measurement_seconds": float(time.time() - t0)}, target / "run.pt")
        dist.barrier()
        if rank == 0:
            latest = root / "latest.json"
            tmp = latest.with_name(".latest.json.tmp")
            tmp.write_text(json.dumps({"step": int(completed), "dir": str(target)}), encoding="utf-8")
            os.replace(tmp, latest)
            for old in root.glob("step_*"):
                if old != target:
                    shutil.rmtree(old, ignore_errors=True)
            print(f"resume_checkpoint=saved  step={int(completed)}  dir={target}", flush=True)
        dist.barrier()

    start_step = 0
    last_log_elapsed = 0.0
    if resume_info is not None:
        # the state after resume_info["step"] steps; the random generators last, after every setup that draws
        start_step = int(resume_info["step"])
        pt.local_parameters.copy_(rank_state["local_parameters"].to(pt.local_parameters.device))
        if hasattr(pt, "_pt_step"):
            pt._pt_step = int(rank_state["pt_step"])
        if rank == 0:
            pt.parameter_matrix.copy_(rank_state["parameter_matrix"].to(pt.parameter_matrix.device))
        if args.algorithm in FLOW_ALGORITHMS:
            pt.flow.load_state_dict(rank_state["flow"])
            pt.optimizer.load_state_dict(rank_state["optimizer"])
            pt.consecutive_nan_grad_updates = int(rank_state["consecutive_nan"])
        if rank == 0:
            r = run_state["totals"]
            total_swap_accepted, total_swap_proposed = r["total_swap_accepted"], r["total_swap_proposed"]
            total_swap_attempts = r["total_swap_attempts"]
            total_work_sum, total_work_sum_sq = r["total_work_sum"], r["total_work_sum_sq"]
            total_work_count = r["total_work_count"]
            total_flow_loss_sum, total_flow_loss_count = r["total_flow_loss_sum"], r["total_flow_loss_count"]
            total_flow_applied_updates = r["total_flow_applied_updates"]
            total_flow_skipped_updates = r["total_flow_skipped_updates"]
            total_link_accepted, total_link_proposed = r["total_link_accepted"], r["total_link_proposed"]
            total_link_work_sum, total_link_work_sum_sq = r["total_link_work_sum"], r["total_link_work_sum_sq"]
            total_link_work_count = r["total_link_work_count"]
            saved_link_work_values = r["saved_link_work_values"]
            history_labels, history_plaquette = r["history_labels"], r["history_plaquette"]
            history_defect_action, history_topology = r["history_defect_action"], r["history_topology"]
            history_topology_steps = r["history_topology_steps"]
            thermalization_elapsed = float(run_state["thermalization_seconds"])
            t0 = time.time() - float(run_state["measurement_seconds"])  # elapsed times count the earlier jobs
            last_log_elapsed = float(run_state["measurement_seconds"])
        torch.set_rng_state(rank_state["rng"]["cpu"])
        if device.type == "cuda" and rank_state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(rank_state["rng"]["cuda"], device)
        rank_state = run_state = None
        if rank == 0:
            print(f"resume=1  from_step={start_step}  steps={args.steps}  dir={resume_info['dir']}", flush=True)
    history_saved_t = time.time()  # last rewrite of replica_history.npz during the run
    # Optional profile of whole REX steps on rank 0: the phases are labelled with record_function so
    # that the table attributes the time to the equilibrium sweep, the swap step and the bookkeeping.
    profiler = None
    py_profiler = None  # cProfile of the same steps, --profile-python
    # steps [start + 1, start + profile_steps]; the first step of the job warms up
    profile_first, profile_last = start_step + 1, start_step + int(args.profile_steps)
    phase_seconds = {}

    def _phase_clock():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    stop_after_s = 60.0 * float(args.max_run_minutes)
    resume_every_s = 60.0 * float(args.resume_every_minutes) if bool(args.save_resume) else 0.0
    resume_saved_t = time.time()
    loop_t0 = time.time()
    steps_completed = start_step
    for step in range(start_step, args.steps):
        if (stop_after_s > 0 or resume_every_s > 0) and step > start_step:
            # Rank 0 decides and broadcasts, so every rank leaves the loop (or writes a checkpoint) after the
            # same step.
            control = [False, False]
            if rank == 0:
                now = time.time()
                control[0] = stop_after_s > 0 and (
                    (now - run_t0) + 1.5 * (now - loop_t0) / (step - start_step) > stop_after_s
                )
                control[1] = resume_every_s > 0 and now - resume_saved_t >= resume_every_s
            dist.broadcast_object_list(control, src=0)
            stop = control[:1]
            if control[1] and not control[0]:
                write_resume(step)
                resume_saved_t = time.time()
            if stop[0]:
                if rank == 0:
                    print(f"time_limit_stop=1  steps_completed={step}  steps_requested={args.steps}  "
                          f"max_run_minutes={args.max_run_minutes}", flush=True)
                break
        steps_completed = step + 1  # this step runs to the end: a stop is decided only at the top of a step
        profiling = rank == 0 and profile_first <= step <= profile_last
        if profiling and profiler is None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities, record_shapes=False)
            profiler.__enter__()
            if bool(args.profile_python):
                import cProfile
                py_profiler = cProfile.Profile()
                py_profiler.enable()
        phase_t0 = _phase_clock() if profiling else None
        region = torch.profiler.record_function("equilibrium_sweep") if profiling else None
        if region is not None:
            region.__enter__()
        for _ in range(args.sweeps_per_step):
            cfgs = local_hbor_sweep(
                cfgs=cfgs,
                local_parameters=pt.local_parameters,
                parameter_ladder=parameter_ladder,
                hbor_by_idx=hbor_by_idx,
                mask=mask,
                defect_mask=defect_mask,
                hbor=hbor_batched,
            )
        if region is not None:
            region.__exit__(None, None, None)
            phase_seconds["equilibrium_sweep"] = phase_seconds.get("equilibrium_sweep", 0.0) + _phase_clock() - phase_t0
            phase_t0 = _phase_clock()
            region = torch.profiler.record_function("swap_step")
            region.__enter__()

        stats = {"n_accepted": 0, "n_proposed": 0}
        train_flow = False
        if (step + 1) % args.swap_every == 0:
            if args.algorithm == "orex":
                protocol_mcmc.set_parameter(pt.local_parameters, parameter_name="bc")
                cfgs, stats = pt.step(cfgs, protocol_mcmc)
            elif args.algorithm in FLOW_ALGORITHMS:
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
            if periodic_value is not None:
                cfgs = translate_periodic_replica(cfgs, pt.local_parameters, periodic_value, args.D)
            if rank == 0:
                total_swap_accepted += int(stats["n_accepted"])
                total_swap_proposed += int(stats["n_proposed"])
                total_swap_attempts += 1
                link_acc_step = _normalize_metric_list(stats.get("link_accepted"), n_links, 0)
                link_prop_step = _normalize_metric_list(stats.get("link_proposed"), n_links, 0)
                for k in range(n_links):
                    total_link_accepted[k] += int(link_acc_step[k])
                    total_link_proposed[k] += int(link_prop_step[k])
                if args.algorithm in NONEQ_ALGORITHMS:
                    total_work_sum += float(stats.get("work_sum", 0.0))
                    total_work_sum_sq += float(stats.get("work_sum_sq", 0.0))
                    total_work_count += int(stats.get("work_count", 0))
                    link_work_mean_step = _normalize_metric_list(stats.get("link_work_mean"), n_links, float("nan"))
                    link_work_sum_sq_step = _normalize_metric_list(stats.get("link_work_sum_sq"), n_links, 0.0)
                    link_work_count_step = _normalize_metric_list(stats.get("link_work_count"), n_links, 0)
                    for k in range(n_links):
                        cnt = int(link_work_count_step[k])
                        val = float(link_work_mean_step[k]) if k < len(link_work_mean_step) else float("nan")
                        if cnt > 0 and math.isfinite(val):
                            total_link_work_sum[k] += val * cnt
                            total_link_work_sum_sq[k] += float(link_work_sum_sq_step[k])
                            total_link_work_count[k] += cnt
                    if bool(args.save_link_work_values):
                        for k, values in enumerate(stats.get("link_work_values", [])):
                            if k >= n_links:
                                break
                            arr = np.asarray(values, dtype=np.float64)
                            if arr.size:
                                saved_link_work_values[k].append(arr.reshape(-1, args.batch_size))
                if args.algorithm in FLOW_ALGORITHMS:
                    flow_loss = float(stats.get("flow_loss", float("nan")))
                    flow_loss_count = int(stats.get("flow_loss_count", 0))
                    if flow_loss_count > 0 and math.isfinite(flow_loss):
                        total_flow_loss_sum += flow_loss * flow_loss_count
                        total_flow_loss_count += flow_loss_count
                    total_flow_applied_updates += int(stats.get("applied_updates", 0))
                    total_flow_skipped_updates += int(stats.get("skipped_nan_grad_updates", 0))

        if region is not None:
            region.__exit__(None, None, None)
            phase_seconds["swap_step"] = phase_seconds.get("swap_step", 0.0) + _phase_clock() - phase_t0
            phase_t0 = _phase_clock()
            region = torch.profiler.record_function("bookkeeping")
            region.__enter__()
        if bool(args.save_replica_history):
            # The plaquette of each local chain (whole lattice, Re Tr / N), gathered to rank 0 and
            # stored with the rung its slot carries now, so the carrier of any rung can be followed.
            # one kernel with --fused-kernels (sun.plaquette_SUN where it does not apply)
            local_plaquette = fused.plaquette(cfgs, args.N) if bool(args.fused_kernels) else None
            if local_plaquette is None:
                local_plaquette = plaquette_SUN(cfgs, args.D, args.N)
            local_plaquette = local_plaquette.to(dtype=torch.float64)
            gather_list = (
                [torch.zeros_like(local_plaquette) for _ in range(world_size)] if rank == 0 else None
            )
            dist.gather(local_plaquette, gather_list=gather_list, dst=0)
            local_defect_action = defect_probe.defect_action(cfgs, pt.local_parameters).to(dtype=torch.float64)
            defect_gather = (
                [torch.zeros_like(local_defect_action) for _ in range(world_size)] if rank == 0 else None
            )
            dist.gather(local_defect_action, gather_list=defect_gather, dst=0)
            if rank == 0:
                history_labels.append(rung_indices(pt.parameter_matrix, parameter_ladder).cpu().numpy())
                history_plaquette.append(torch.stack(gather_list, dim=0).cpu().numpy())
                history_defect_action.append(torch.stack(defect_gather, dim=0).cpu().numpy())
            if topology_every > 0 and (step + 1) % topology_every == 0:
                # Q of the chains of the top rung (the periodic replica), or of every chain: NaN elsewhere
                local_charge = torch.full((int(cfgs.shape[0]),), float("nan"), dtype=torch.float64, device=cfgs.device)
                if args.topology_replicas == "all":
                    chosen = torch.arange(int(cfgs.shape[0]), device=cfgs.device)
                else:
                    chosen = torch.nonzero(pt.local_parameters == parameter_ladder[-1], as_tuple=False).squeeze(-1)
                if int(chosen.numel()) > 0:
                    # cooling works on its own copy: no index_select copy when every chain is measured
                    subset = cfgs if args.topology_replicas == "all" else cfgs.index_select(0, chosen)
                    local_charge[chosen] = topology.topological_charge(
                        subset, int(args.cooling_steps), use_kernels=bool(args.fused_kernels)
                    ).to(dtype=torch.float64)
                charge_gather = [torch.zeros_like(local_charge) for _ in range(world_size)] if rank == 0 else None
                dist.gather(local_charge, gather_list=charge_gather, dst=0)
                if rank == 0:
                    history_topology.append(torch.stack(charge_gather, dim=0).cpu().numpy())
                    history_topology_steps.append(len(history_labels) - 1)

        link_accepted_step = _normalize_metric_list(stats.get("link_accepted"), n_links, 0)
        link_proposed_step = _normalize_metric_list(stats.get("link_proposed"), n_links, 0)
        link_acceptance_step = [
            (float(link_accepted_step[k]) / float(link_proposed_step[k]))
            if int(link_proposed_step[k]) > 0
            else float("nan")
            for k in range(n_links)
        ]
        link_acceptance_cumulative = [
            (float(total_link_accepted[k]) / float(total_link_proposed[k]))
            if int(total_link_proposed[k]) > 0
            else float("nan")
            for k in range(n_links)
        ]
        link_work_mean_step = _normalize_metric_list(
            stats.get("link_work_mean"),
            n_links,
            float("nan"),
        )
        link_work_sem_step = _normalize_metric_list(
            stats.get("link_work_sem"),
            n_links,
            float("nan"),
        )
        link_work_count_step = _normalize_metric_list(
            stats.get("link_work_count"),
            n_links,
            0,
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
                    float(link_acceptance_cumulative[k])
                )
                wb[f"work/link_{k}_mean_step"] = float(link_work_mean_step[k])
                wb[f"work/link_{k}_sem_step"] = float(link_work_sem_step[k])
                wb[f"work/link_{k}_count_step"] = int(link_work_count_step[k])
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
            if args.algorithm in NONEQ_ALGORITHMS:
                step_work_count = int(stats.get("work_count", 0))
                step_work_mean, step_work_sem = _mean_sem_from_moments(
                    float(stats.get("work_sum", 0.0)),
                    float(stats.get("work_sum_sq", 0.0)),
                    step_work_count,
                )
            if args.algorithm in FLOW_ALGORITHMS:
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
                        json.dumps([int(v) for v in link_accepted_step]),
                        json.dumps([int(v) for v in link_proposed_step]),
                        json.dumps([float(v) for v in link_acceptance_step]),
                        json.dumps([float(v) for v in link_acceptance_cumulative]),
                        json.dumps([float(v) for v in link_work_mean_step]),
                        json.dumps([float(v) for v in link_work_sem_step]),
                        json.dumps([int(v) for v in link_work_count_step]),
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
            if history_labels and time.time() - history_saved_t >= 60.0 * float(args.replica_history_save_minutes):
                save_replica_history()
                history_saved_t = time.time()
        if region is not None:
            region.__exit__(None, None, None)
            phase_seconds["bookkeeping"] = phase_seconds.get("bookkeeping", 0.0) + _phase_clock() - phase_t0
        if profiler is not None and step == profile_last:
            if py_profiler is not None:
                py_profiler.disable()
            profiler.__exit__(None, None, None)
            averages = profiler.key_averages()
            n_prof = profile_last - profile_first + 1
            with open(out_dir / "step_profile.txt", "w", encoding="utf-8") as handle:
                handle.write(f"# {n_prof} REX steps of {args.algorithm} at batch {args.batch_size}, device {device}\n")
                handle.write("# seconds per step by phase (device synchronized):\n")
                for name, seconds in phase_seconds.items():
                    handle.write(f"#   {name:20s} {seconds / n_prof:8.3f}\n")
                handle.write(f"# operators by self {device.type} time\n")
                handle.write(averages.table(sort_by=f"self_{device.type}_time_total", row_limit=40))
                handle.write("\n# operators by self cpu time\n")
                handle.write(averages.table(sort_by="self_cpu_time_total", row_limit=40))
                handle.write("\n# operators by call count\n")
                for event in sorted(averages, key=lambda e: -e.count)[:40]:
                    handle.write(f"{event.key:60s} {event.count:8d}\n")
            print("step_profile_phases=" + json.dumps({k: round(v / n_prof, 4) for k, v in phase_seconds.items()})
                  + f"  step_profile_path={out_dir / 'step_profile.txt'}", flush=True)
            profiler = None
            if py_profiler is not None:
                import pstats
                with open(out_dir / "step_cprofile.txt", "w", encoding="utf-8") as handle:
                    for key in ("tottime", "cumulative"):
                        handle.write(f"# Python functions by {key} time, {n_prof} REX steps on rank 0 (cProfile)\n")
                        pstats.Stats(py_profiler, stream=handle).sort_stats(key).print_stats(60)
                print(f"step_cprofile_path={out_dir / 'step_cprofile.txt'}", flush=True)
                py_profiler = None

    if bool(args.save_resume):
        write_resume(steps_completed)  # the stop of --max-run-minutes or the end: --resume continues from here

    saved_flow_paths = []
    saved_flow_metadata_paths = []
    if args.algorithm in FLOW_ALGORITHMS:
        final_flow_path = out_dir / f"neorex_flow_final_nsteps{int(args.neorex_protocol_steps)}.pt"
        pt.save_flow(str(final_flow_path))
        saved_flow_paths.append(str(final_flow_path))
        metadata_path = _write_neorex_flow_metadata(args, str(final_flow_path), rank)
        if rank == 0:
            saved_flow_metadata_paths.append(metadata_path)

        save_path = _resolve_neorex_flow_checkpoint_path(args, args.neorex_save_flow)
        if save_path is not None:
            pt.save_flow(str(save_path))
            saved_flow_paths.append(str(save_path))
            metadata_path = _write_neorex_flow_metadata(args, str(save_path), rank)
            if rank == 0:
                saved_flow_metadata_paths.append(metadata_path)
                print(f"neorex_flow_checkpoint=saved  path={save_path}", flush=True)

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
        final_link_acceptance = [
            (float(total_link_accepted[k]) / float(total_link_proposed[k]))
            if int(total_link_proposed[k]) > 0
            else float("nan")
            for k in range(n_links)
        ]
        print(
            "final_link_swap_acceptance_json="
            f"{json.dumps([float(v) for v in final_link_acceptance])}  "
            f"link_accepted_json={json.dumps([int(v) for v in total_link_accepted])}  "
            f"link_proposed_json={json.dumps([int(v) for v in total_link_proposed])}",
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
        meta["steps_requested"] = int(args.steps)
        meta["steps_completed"] = int(steps_completed)
        meta["stopped_by_time_limit"] = bool(steps_completed < int(args.steps))
        meta["swap"] = {
            "acceptance_mean": float(mean_swap_acc),
            "total_accepted": int(total_swap_accepted),
            "total_proposed": int(total_swap_proposed),
            "swap_attempts": int(total_swap_attempts),
            "parameter_name": str(pt.parameter_name),
            "link_parameter_left": [
                float(v) for v in parameter_ladder[:-1].detach().cpu().tolist()
            ],
            "link_parameter_right": [
                float(v) for v in parameter_ladder[1:].detach().cpu().tolist()
            ],
            "link_delta_parameter": [
                float(v)
                for v in (parameter_ladder[1:] - parameter_ladder[:-1]).detach().cpu().tolist()
            ],
            "link_acceptance": [float(v) for v in final_link_acceptance],
            "link_accepted": [int(v) for v in total_link_accepted],
            "link_proposed": [int(v) for v in total_link_proposed],
        }
        final_gpu_memory = _rank0_cuda_memory_metrics(device, bool(args.log_gpu_memory))
        if final_gpu_memory:
            meta["gpu_memory"] = {
                key.removeprefix("gpu_memory/"): float(value)
                for key, value in final_gpu_memory.items()
            }
        if args.algorithm in FLOW_ALGORITHMS:
            meta["neorex_flow_checkpoint"] = {
                "load_mode": neorex_flow_load_mode,
                "load_path": neorex_flow_load_path,
                "saved_paths": saved_flow_paths,
                "metadata_paths": saved_flow_metadata_paths,
            }
        if args.algorithm in NONEQ_ALGORITHMS:
            run_work_mean, run_work_sem = _mean_sem_from_moments(
                total_work_sum,
                total_work_sum_sq,
                total_work_count,
            )
            final_link_work_mean = []
            final_link_work_sem = []
            for k in range(n_links):
                mean, sem = _mean_sem_from_moments(
                    total_link_work_sum[k],
                    total_link_work_sum_sq[k],
                    total_link_work_count[k],
                )
                final_link_work_mean.append(float(mean))
                final_link_work_sem.append(float(sem))
            print(
                f"final_work_mean={run_work_mean:.6e}  "
                f"final_work_sem={run_work_sem:.3e}  "
                f"work_count={total_work_count}",
                flush=True,
            )
            print(
                "final_link_work_mean_json="
                f"{json.dumps([float(v) for v in final_link_work_mean])}  "
                f"final_link_work_sem_json={json.dumps([float(v) for v in final_link_work_sem])}  "
                f"link_work_count_json={json.dumps([int(v) for v in total_link_work_count])}",
                flush=True,
            )
            meta["work"] = {
                "mean": float(run_work_mean),
                "sem": float(run_work_sem),
                "count": int(total_work_count),
                "link_mean": [float(v) for v in final_link_work_mean],
                "link_sem": [float(v) for v in final_link_work_sem],
                "link_count": [int(v) for v in total_link_work_count],
            }
            if bool(args.save_link_work_values):
                link_work_path = out_dir / "link_work_values.npz"
                parameter_left = parameter_ladder[:-1].detach().cpu().numpy().astype(np.float64)
                parameter_right = parameter_ladder[1:].detach().cpu().numpy().astype(np.float64)
                payload = {
                    "link_index": np.arange(n_links, dtype=np.int64),
                    "parameter_name": np.asarray(str(pt.parameter_name)),
                    "parameter_left": parameter_left,
                    "parameter_right": parameter_right,
                    "parameter_delta": parameter_right - parameter_left,
                    "swap_parameter_left": parameter_left,
                    "swap_parameter_right": parameter_right,
                    "swap_delta_parameter": parameter_right - parameter_left,
                    "link_acceptance": np.asarray(final_link_acceptance, dtype=np.float64),
                    "link_accepted": np.asarray(total_link_accepted, dtype=np.int64),
                    "link_proposed": np.asarray(total_link_proposed, dtype=np.int64),
                    "link_work_mean": np.asarray(final_link_work_mean, dtype=np.float64),
                    "link_work_sem": np.asarray(final_link_work_sem, dtype=np.float64),
                    "link_work_count": np.asarray(total_link_work_count, dtype=np.int64),
                }
                for k in range(n_links):
                    if saved_link_work_values[k]:
                        payload[f"link_{k}"] = np.concatenate(saved_link_work_values[k], axis=0)
                    else:
                        payload[f"link_{k}"] = np.empty((0, args.batch_size), dtype=np.float64)
                np.savez_compressed(link_work_path, **payload)
                meta["work"]["link_work_values_path"] = str(link_work_path)
                print(f"link_work_values_path={link_work_path}", flush=True)
        if args.algorithm in FLOW_ALGORITHMS:
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
        if history_labels:
            # The history is a diagnostic: a failure here must not lose the final statistics
            # of the run, so it is recorded in run_meta.json instead of raised.
            try:
                save_replica_history()
                history = summarize_history(
                    np.stack(history_labels, axis=0),
                    np.stack(history_plaquette, axis=0),
                    world_size,
                    defect_action=np.stack(history_defect_action, axis=0),
                    topology=np.stack(history_topology, axis=0) if history_topology else None,
                    topology_steps=np.asarray(history_topology_steps, dtype=np.int64),
                    volume=int(args.T) * int(args.L) ** (int(args.D) - 1),
                )
                history["defect_action"]["parameter_ladder"] = [
                    float(v) for v in parameter_ladder.detach().cpu().tolist()
                ]
                history["path"] = str(history_path)
                top = history["plaquette_top"]
                print(
                    f"round_trips={history['round_trips']}  "
                    f"round_trip_steps={history['round_trip_steps']:.2f}+-{history['round_trip_steps_sem']:.2f}  "
                    f"round_trips_per_walker_step={history['round_trips_per_walker_step']:.3e}  "
                    f"flux_fraction_json={json.dumps([round(v, 3) for v in history['flux_fraction']])}",
                    flush=True,
                )
                print(
                    f"plaquette_top={top['value']:.6f}+-{top['error']:.1e}  "
                    f"plaquette_top_tau_int={top['tau_int']:.2f}+-{top['tau_int_error']:.2f}  "
                    f"gamma_status={top['status']}  replica_history_path={history_path}",
                    flush=True,
                )
                if "topology_top" in history:
                    topo = history["topology_top"]
                    print(
                        f"topology_top: measurements={topo['measurements']}  every={topo['every']}  "
                        f"cooling_steps={int(args.cooling_steps)}  Q={topo['Q']['value']:.4f}+-{topo['Q']['error']:.1e}  "
                        f"chi={topo['chi']:.3e}+-{topo['chi_error']:.1e}  "
                        f"tau_int_Q2_steps={topo['tau_int_Q2_steps']:.1f}+-{topo['tau_int_Q2_steps_error']:.1f}  "
                        f"gamma_status={topo['Q2']['status']}",
                        flush=True,
                    )
            except Exception as err:  # noqa: BLE001 - diagnostics must not fail the run
                history = {"path": str(history_path), "error": f"{type(err).__name__}: {err}"}
                print(f"replica_history_error={history['error']}", flush=True)
            meta["replica_history"] = history
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if wandb_run is not None:
            wandb_run.summary["timing/measurement_seconds"] = float(measurement_elapsed)
            wandb_run.summary["timing/total_seconds"] = float(total_elapsed)
            wandb_run.summary["swap/link_acceptance"] = [
                float(v) for v in final_link_acceptance
            ]
            if args.algorithm in NONEQ_ALGORITHMS:
                wandb_run.summary["work/link_mean"] = [
                    float(v) for v in final_link_work_mean
                ]
                wandb_run.summary["work/link_sem"] = [
                    float(v) for v in final_link_work_sem
                ]
            for key, value in final_gpu_memory.items():
                wandb_run.summary[key] = float(value)
            wandb_run.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
