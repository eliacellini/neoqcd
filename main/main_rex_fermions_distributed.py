"""REX on the gauge-only defect with dynamical fermions, one rung per rank (multi-GPU).

The distributed counterpart of main/main_rex_fermions.py, organised as main/main_pt_obc.py: rank r
holds --n-chains configurations (one per independent ladder) that start at rung r; the bc labels
move through `neoqcd.distributedPT.ParallelTempering` (rank 0 holds the (n_chains, world_size)
parameter matrix and does the swaps, configurations never move). One REX step =
`neoqcd.fermions.rex.distributed_rex_step`: one fermionic HMC trajectory of every local configuration
at its bc (N_f = 2 Wilson --kappa, or staggered --mass), the relative action table at the ladder from
the gauge-only `QCDAction.action_difference` (no solve), `pt.step` (rule "exact", --swap-parity),
optional translation of the bc = 1 configurations (--periodic-translation).

Thermalisation as in main_pt_obc.py: --therm (alias --thermal-steps) trajectories at fixed labels, no
swaps, no translation, not in replica_history.npz (rows flagged therm = 1 in observables.csv). (In
main_rex_fermions.py --therm steps are REX steps with swaps, kept in therm_* arrays of its history.)

Defect: `utils.Defect(D, T, L, --defect-size, --time-slice, --space-slice)` (default --defect-shape
cube), the defect of main_pt_obc.py for L_d = --defect-size; --defect-shape line is the fermion-only
test shape. Ladder: world_size rungs, --ladder (aliases --ladder-file, --parameter-ladder: a list or
the JSON file of main/optimal_ladder.py) or evenly spaced in [--bc-min, --bc-max] (the same
`distributedPT.ladder_values` as main_pt_obc.py).

--fast (Wilson) and its options (--precision, --colour-product, --check-every, --inner-tol,
--max-outer) and --on-unconverged are those of main_snf_fermions.py: every rank's `QCDAction` is
built with fast=True; the fast settings are in run_meta.json and in the resume key.

Devices: --device cpu (Gloo), or cuda / auto: rank with LOCAL_RANK = l uses cuda:l (NCCL) with an
explicit index (torch 2.0 cannot set_device("cuda") without one). A LOCAL_RANK beyond the visible
GPUs is an error, and so is --device cuda:K with more than one local rank (every rank on GPU K),
unless --allow-shared-gpu (tests and debugging: ranks then share GPUs, cuda:LOCAL_RANK mod count).
A launcher that binds one GPU per task (each rank sees device_count() == 1) must set LOCAL_RANK=0
per task or pass --allow-shared-gpu. Launch with torchrun (job/debug_rex_fermions_distributed.sh)
or any env:// launcher (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK).

Generators: on each rank a run generator on its device (pseudofermions, momenta, Metropolis,
translation). The swap uniforms of rank 0 come from `torch.rand` on the global generator of
pt.device inside `ParallelTempering`; they are drawn from a private stream (`rex.GlobalRNGStream`)
so that a resumed run repeats them. Seeds `stream_seeds(--seed, rank)` = `rex.derived_seed(seed,
rank, stream)` (SeedSequence): runs with seeds s and s + 1 share no stream on any pair of ranks
(the pure-gauge seed + rank does). The generator states are checkpointed, so resumes are unaffected.

Output directory (--out, alias --output-dir; default output/rex_fermions_distributed/<auto name>):
    observables.csv     (rank 0) one row per (step, rung), the fields of main_rex_fermions.py
                        (OBS_FIELDS): plaquette = mean over chains of the configuration carrying the
                        rung after the step, exp_mdH of the trajectories run at that rung, swaps of the
                        pair (rung, rung + 1), HMC acceptance of all replicas; solves and cg_iter are
                        summed over the ranks, wall_s is the slowest rank's step time, max_residual
                        the largest true relative CG residual of the step on any rank (NaN-sticky) and
                        unconverged_solves the sum over the ranks (`main_snf_fermions.SolveMonitor`);
    replica_history.npz (rank 0) measured steps only: labels (steps, world_size, n_chains) int8, the
                        rung of each chain slot after the step (`replica_history.rung_indices`);
                        plaquette, defect_action (S_D = dS/dc at the slot's bc, `rex.defect_action`),
                        trajectory_labels, exp_minus_dH (same shape); parameter_ladder; swap_every;
                        thermal_steps. The keys and layout of main_pt_obc.py: main/optimal_ladder.py
                        reads the directory unchanged. Rewritten every --log-every steps and at
                        every checkpoint (it may hold steps past the last committed checkpoint);
    run_meta.json       (rank 0) algorithm, backend, world_size, parameters, the "swap" and
                        "replica_history" blocks of main_pt_obc.py (round trips, flux, plaquette tau_int,
                        defect-action moments), per-rung plaquette and cost per step and per round trip;
    checkpoint_step.json          (rank 0) the committed checkpoint: {"step", "slot", "world_size"};
    rankNNN/checkpoint_G.pt/.json links and run generator of rank NNN (`fermions.io.save_state`),
                                  slot G = 0 or 1;
    rankNNN/rex_state_G.pt        step count, PT state (`pt._pt_step`, rank 0's parameter matrix,
                                  local parameters), swap-stream state, swap counters (rank 0).

Checkpoints are two-generation: every rank writes its files of step n into the slot that is not
committed (each file atomically), all ranks confirm (collective), then rank 0 saves the history and
commits by atomically rewriting checkpoint_step.json to (n, new slot). A kill at any point leaves the
marker on a complete set of files of one step on every rank; the previous generation stays intact
until the next commit overwrites it.

`ParallelTempering` exposes no state_dict (the --save-resume / --resume of main_pt_obc.py save the
same three pieces in their own checkpoints); its state is the parameter matrix on rank 0, the step
counter `_pt_step` (alternating parity) and the scattered local parameters, restored here by setting
the first two and calling `_scatter_parameters()`.

Start and resume (decided on rank 0, shared by all ranks). The run owns only the files named above
(and the temporary files of their atomic writes); anything else in --out is never read, blocks
nothing and is never deleted.
  - --resume with a committed checkpoint: every rank restores the committed slot, observables.csv
    and the history are truncated to that step, the run continues; the continuation repeats the
    uninterrupted run with the same seed;
  - a committed checkpoint without --resume: refused, unless --overwrite (then the run's own files
    are deleted and the run starts from scratch);
  - no committed checkpoint but files of this run (a job killed before its first checkpoint): they
    hold nothing resumable; they are deleted, with a log line, and the run starts from scratch
    (with or without --resume), so a resubmitted job never gets stuck;
  - nothing: fresh start (--resume logs that there was nothing to resume).
  - rankNNN/checkpoint.json of the older single-generation layout and no marker: refused unless
    --overwrite (that layout cannot be resumed; its files count as the run's own).
--max-run-minutes stops (with a checkpoint) before the step that would end past that time, as in
main_pt_obc.py.

Failures: checks and rank-0-only work (directory scan, history, CSV, commit) go through
`Collective`: a failure on one rank becomes a SystemExit on every rank, so no rank is left blocked
in a collective until the job's time limit (also under a plain `srun -n` launcher).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main_hmc import _sync  # noqa: E402
from main_rex_fermions import OBS_FIELDS, add_rex_args, resolve_ladder, swap_meta  # noqa: E402
from main_snf_fermions import (  # noqa: E402
    SolveMonitor,
    action_kwargs,
    add_physics_args,
    build_defect_mask,
    clean_json,
    convergence_from_csv,
    defect_key,
    fermion_kwargs,
    nan_max,
    run_info,
    staggered_key,
    start_links,
    state_couplings,
    upgrade_csv_header,
    validate_common,
)

from neoqcd.distributedPT import ParallelTempering, PTConfig  # noqa: E402
from neoqcd.fermions._gauge import mean_plaquette, plaquette_fields, rung_indices, summarize_history  # noqa: E402,F401
from neoqcd.fermions.action import QCDAction  # noqa: E402
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
from neoqcd.fermions.rex import (  # noqa: E402
    STREAM_GLOBAL,
    STREAM_RUN,
    STREAM_SWAP,
    FermionREXTarget,
    GlobalRNGStream,
    derived_seed,
    distributed_rex_step,
    step_observables,
)

MARKER = "checkpoint_step.json"
TOP_FILES = ("observables.csv", "replica_history.npz", "run_meta.json", MARKER)
SLOTS = (0, 1)
RANK_FILES = tuple(f"{stem}_{g}{ext}" for g in SLOTS
                   for stem, ext in (("checkpoint", ".pt"), ("checkpoint", ".json"), ("rex_state", ".pt")))
LEGACY_RANK_FILES = ("checkpoint.pt", "checkpoint.json", "rex_state.pt")  # single-generation layout (no resume)
RANK_DIR = re.compile(r"rank\d{3}")
N_PACK = 4  # per-chain values gathered each step: plaquette, defect action, exp(-dH), HMC accepted
# per-rank values gathered after them: solves, CG iterations, wall time, largest residual (-1: no solve), unconverged
N_TAIL = 5


# ---- arguments ----------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Distributed REX (one rung per rank) on the gauge-only defect "
                                            "with dynamical fermions.")
    add_physics_args(p)
    for action in p._actions:  # allow cuda:K besides auto/cpu/cuda (module doc)
        if action.dest == "device":
            action.choices = None
            action.help = "auto | cpu | cuda (cuda:LOCAL_RANK on each rank) | cuda:K"
    add_rex_args(p)
    p.add_argument("--n-chains", "--batch-size", dest="n_chains", type=int, default=1,
                   help="Configurations per rank = independent ladders (main_pt_obc.py --batch-size).")
    p.add_argument("--thermal-steps", dest="therm", type=int, default=argparse.SUPPRESS,
                   help="Alias of --therm (main_pt_obc.py name): trajectories at fixed labels, no swaps.")
    p.add_argument("--steps", type=int, default=100, help="Measured REX steps after the thermalisation.")
    p.add_argument("--swap-parity", choices=("even", "odd", "alternating", "even_odd"), default="alternating")
    p.add_argument("--periodic-translation", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Checkpoint every k steps (0: end only).")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-run-minutes", type=float, default=0.0,
                   help="Stop before the step that would end after this many minutes (1.5 x mean step "
                        "time), with a checkpoint; 0 = off.")
    p.add_argument("--out", "--output-dir", dest="out", type=str, default=None)
    p.add_argument("--allow-shared-gpu", action="store_true",
                   help="Let ranks share a GPU (LOCAL_RANK beyond the visible GPUs, or --device cuda:K with "
                        "several local ranks); tests and debugging only.")
    args = p.parse_args(argv)
    validate_common(args)
    if args.device not in ("auto", "cpu", "cuda"):
        try:
            dev = torch.device(args.device)
        except RuntimeError:
            raise SystemExit(f"--device: cannot parse {args.device!r}")
        if dev.type != "cuda" or dev.index is None:
            raise SystemExit(f"--device must be auto, cpu, cuda or cuda:K, got {args.device!r}")
    if args.n_chains < 1 or args.steps < 0 or args.checkpoint_every < 0 or args.log_every < 1:
        raise SystemExit("--n-chains >= 1, --steps >= 0, --checkpoint-every >= 0, --log-every >= 1")
    return args


def default_name(args, world_size):
    lat = f"{args.T}x{args.L}p{args.D - 1}"
    f = f"k{args.kappa:g}" if args.fermions == "wilson" else f"stag_m{args.mass:g}"
    return (f"rexd_r{world_size}_c{args.n_chains}_d{args.defect_shape}{args.defect_size}_b{args.beta:g}_{f}"
            f"_D{args.D}_{lat}_SU{args.N}_s{args.seed}")


def physics_key(args, ladder, world_size):
    return {
        "geometry": {"D": args.D, "T": args.T, "L": args.L, "N": args.N},
        "beta": float(args.beta), "kappa": None if args.kappa is None else float(args.kappa),
        "time_sign": float(args.time_sign), "ladder": [float(v) for v in ladder], "world_size": int(world_size),
        "n_chains": args.n_chains, "defect": defect_key(args), "tau": float(args.tau), "n_steps": args.n_steps,
        "n_gauge_substeps": args.n_gauge_substeps, "integrator": list(args.integrator_levels),
        "swap_parity": args.swap_parity, "periodic_translation": bool(args.periodic_translation),
        "therm": int(args.therm), **staggered_key(args),
    }


def stream_seeds(seed, rank):
    """Seeds of the run generator, the global generator and the swap stream of `rank` (module doc)."""
    return {"run": derived_seed(seed, rank, STREAM_RUN), "global": derived_seed(seed, rank, STREAM_GLOBAL),
            "swap": derived_seed(seed, rank, STREAM_SWAP)}


# ---- the run's own files ------------------------------------------------------------------------
def _own_name(name, known):
    """True for a known file name or a leftover temporary file of its atomic write (`io._atomic_write`)."""
    if name in known:
        return True
    return any(name.startswith(f".{k}.") and name.endswith(".tmp") for k in known)


def own_files(out):
    """Paths (relative to out) of the files this driver writes that exist in `out` (module doc)."""
    out = Path(out)
    if not out.is_dir():
        return []
    found = []
    for path in sorted(out.iterdir()):
        if path.is_file() and _own_name(path.name, TOP_FILES):
            found.append(path.name)
        elif path.is_dir() and RANK_DIR.fullmatch(path.name):
            found += [f"{path.name}/{q.name}" for q in sorted(path.iterdir())
                      if q.is_file() and _own_name(q.name, RANK_FILES + LEGACY_RANK_FILES)]
    return found


def remove_own_files(out, names):
    """Delete the listed own files, then the rankNNN directories left empty; nothing else."""
    out = Path(out)
    for name in names:
        (out / name).unlink()
    for name in sorted({n.split("/")[0] for n in names if "/" in n}):
        try:
            (out / name).rmdir()
        except OSError:  # not empty: holds files that are not the run's
            pass


def read_marker(out):
    """The committed checkpoint {"step", "slot", "world_size"} of `out`, or None."""
    path = Path(out) / MARKER
    if not path.exists():
        return None
    marker = json.loads(path.read_text())
    if int(marker["slot"]) not in SLOTS:
        raise ValueError(f"{path}: bad slot {marker['slot']}")
    return marker


def rank_paths(out, rank, slot):
    """(checkpoint prefix, rex_state path) of `rank` in `slot`."""
    rank_dir = Path(out) / f"rank{rank:03d}"
    return rank_dir / f"checkpoint_{slot}", rank_dir / f"rex_state_{slot}.pt"


# ---- runtime ------------------------------------------------------------------------------------
def choose_rank_device(spec, allow_shared=False):
    """(device, backend) of this rank (module doc): Gloo on CPU, NCCL on cuda:<explicit index>.
    A rank without a GPU of its own is an error unless `allow_shared`."""
    if spec == "cpu" or (spec == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu"), "gloo"
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {spec} but CUDA is not available")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    count = torch.cuda.device_count()
    if spec in ("cuda", "auto"):
        if local_rank < count:
            index = local_rank
        elif allow_shared:
            index = local_rank % count
            print(f"WARNING: LOCAL_RANK={local_rank} shares cuda:{index} ({count} GPUs visible, --allow-shared-gpu)",
                  file=sys.stderr, flush=True)
        else:
            raise SystemExit(f"LOCAL_RANK={local_rank} but only {count} CUDA device(s) are visible: one GPU per "
                             f"rank is required (--allow-shared-gpu lets ranks share GPUs, tests/debug only)")
        device = torch.device("cuda", index)
    else:
        device = torch.device(spec)
        if device.index >= count:
            raise SystemExit(f"--device {spec} but only {count} CUDA device(s) are visible")
        if local_world > 1 and not allow_shared:
            raise SystemExit(f"--device {spec} would put all {local_world} local ranks on one GPU; use --device cuda "
                             f"(cuda:LOCAL_RANK) or pass --allow-shared-gpu (tests/debug only)")
    torch.cuda.set_device(device)
    return device, "nccl"


def init_process_group(backend, device):
    """env:// group (torchrun or any launcher with RANK/WORLD_SIZE/MASTER_*), or a one-rank file group.
    Returns True if this call created the group (then main destroys it at the end)."""
    if dist.is_initialized():
        return False
    if backend == "gloo" and "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if os.uname().sysname == "Darwin" else "lo"
    if "RANK" in os.environ:
        if backend == "nccl":
            try:
                dist.init_process_group(backend=backend, device_id=device)
            except TypeError:  # torch < 2.3 has no device_id
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)
    else:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            init_method = f"file://{handle.name}"
        dist.init_process_group(backend=backend, init_method=init_method, rank=0, world_size=1)
    return True


class Collective:
    """Decisions every rank must share (errors, stop flags), on the communication device."""

    def __init__(self, device):
        self.device = device

    def all_ok(self, ok, message):
        """SystemExit on every rank if `ok` is False on any rank (no rank left waiting in a collective)."""
        flag = torch.tensor([1 if ok else 0], dtype=torch.int64, device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not ok:
            raise SystemExit(message)
        if int(flag.item()) == 0:
            raise SystemExit("stopped: another rank failed a check (see its message)")

    def from_rank0(self, value):
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def rank0(self, fn, what):
        """Run `fn()` on rank 0 only; its success is broadcast and a failure (exception or SystemExit)
        becomes a SystemExit on every rank. Returns fn's value on rank 0, None elsewhere."""
        result, ok, message = None, True, ""
        if dist.get_rank() == 0:
            try:
                result = fn()
            except SystemExit as exc:
                ok, message = False, str(exc)
            except Exception as exc:  # noqa: BLE001 - reported, then every rank exits
                traceback.print_exc()
                ok, message = False, f"{what} failed on rank 0: {exc.__class__.__name__}: {exc}"
        flag = torch.tensor([1 if ok else 0], dtype=torch.int64, device=self.device)
        dist.broadcast(flag, src=0)
        if not ok:
            raise SystemExit(message)
        if int(flag.item()) == 0:
            raise SystemExit(f"stopped: {what} failed on rank 0 (see its message)")
        return result


# ---- history and summaries (rank 0) -------------------------------------------------------------
class History:
    """Rank 0's replica history of the measured steps, (world_size, n_chains) arrays per step."""

    NAMES = ("labels", "plaquette", "defect_action", "trajectory_labels", "exp_minus_dH")

    def __init__(self, world_size, n_chains):
        self.shape = (world_size, n_chains)
        self.data = {name: [] for name in self.NAMES}

    def __len__(self):
        return len(self.data["labels"])

    def append(self, **values):
        for name in self.NAMES:
            self.data[name].append(np.asarray(values[name]))

    def arrays(self):
        out = {}
        for name in self.NAMES:
            rows = self.data[name]
            dtype = np.int8 if "labels" in name else np.float64
            out[name] = np.stack(rows, axis=0).astype(dtype) if rows else np.zeros((0,) + self.shape, dtype=dtype)
        return out

    def save(self, path, ladder, thermal_steps):
        arrays = self.arrays()

        def write(tmp):
            with open(tmp, "wb") as handle:
                np.savez_compressed(handle, **arrays, parameter_ladder=np.asarray(ladder, dtype=np.float64),
                                    swap_every=np.asarray(1), thermal_steps=np.asarray(int(thermal_steps)))
        _atomic_write(path, write)

    def load(self, path, n_steps):
        """The first n_steps measured steps of the file (a longer history is truncated)."""
        with np.load(path) as h:
            if h["labels"].shape[0] < n_steps:
                raise SystemExit(f"--resume: {path} holds {h['labels'].shape[0]} steps, the checkpoint {n_steps}")
            for name in self.NAMES:
                self.data[name] = [np.array(x) for x in h[name][:n_steps]]


def summarize(history, csv_path, ladder, path):
    """Pure-gauge "replica_history" block and the cost summary of the measured steps."""
    n_r = len(ladder)
    summary = {"steps_measured": len(history),
               "convergence": convergence_from_csv(csv_path, "step")}  # every step, thermalisation included
    if len(history) == 0:
        return summary, None
    a = history.arrays()
    try:
        hist = summarize_history(a["labels"].astype(np.int64), a["plaquette"], n_r,
                                 defect_action=a["defect_action"])
        hist["defect_action"]["parameter_ladder"] = [float(v) for v in ladder]
        hist["path"] = str(path)
    except Exception as err:  # noqa: BLE001 - diagnostics must not fail the run (as main_pt_obc.py)
        hist = {"path": str(path), "error": f"{type(err).__name__}: {err}"}
    cols = read_observables(csv_path)
    keep = cols["therm"] == 0
    cols = {k: v[keep] for k, v in cols.items()}
    per_step = cols["rung"] == 0
    summary.update({
        "plaquette_per_rung": [float(cols["plaquette"][cols["rung"] == k].mean()) for k in range(n_r)],
        "exp_minus_dH_per_rung": [float(np.nanmean(cols["exp_mdH"][cols["rung"] == k])) for k in range(n_r)],
        "hmc_acceptance": float(cols["hmc_acceptance"][per_step].mean()),
        "solves_per_step": float(cols["solves"][per_step].mean()),
        "cg_iter_per_step": float(cols["cg_iter"][per_step].mean()),
        "seconds_per_step": float(cols["wall_s"][per_step].mean()),
    })
    rt = hist.get("round_trip_steps")
    if rt is not None and np.isfinite(rt) and rt > 0:
        summary["seconds_per_round_trip_per_walker"] = float(rt * summary["seconds_per_step"])
        summary["solves_per_round_trip"] = float(rt * summary["solves_per_step"])
    return summary, hist


def _per_rung(values, labels, n_rungs):
    """Mean over chains of the value carried by each rung; values, labels (world_size, n_chains)."""
    return [float(values[labels == k].mean()) for k in range(n_rungs)]


# ---- the run ------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    device, backend = choose_rank_device(args.device, args.allow_shared_gpu)
    own_group = init_process_group(backend, device)
    try:
        return run(args, argv, device, backend)
    finally:
        if own_group and dist.is_initialized():
            dist.destroy_process_group()


@torch.no_grad()
def run(args, argv, device, backend):
    run_t0 = time.time()
    rank, world = dist.get_rank(), dist.get_world_size()
    comm = Collective(device)
    if world < 2:
        raise SystemExit("the distributed REX needs >= 2 ranks (one rung per rank)")
    C = int(args.n_chains)
    n_links = world - 1

    explicit = args.ladder is not None
    ladder = resolve_ladder(args, world, warn=rank == 0)
    cfg = PTConfig(batch_size=C, parameter_min=ladder[0], parameter_max=ladder[-1], parameter_name="bc",
                   parameter_dtype=torch.float64, swap_parity=args.swap_parity, acceptance_rule="exact",
                   parameter_ladder=ladder if explicit else None)
    pt = ParallelTempering(cfg, rank, world, device)
    ladder = [float(v) for v in pt.ladder.cpu().tolist()]  # the values the swaps use (main_pt_obc.py)

    geometry = Geometry(args.D, args.T, args.L, args.N)
    out = Path(args.out) if args.out else ROOT / "output" / "rex_fermions_distributed" / default_name(args, world)
    rank_dir = out / f"rank{rank:03d}"
    csv_path, meta_path, hist_path = out / "observables.csv", out / "run_meta.json", out / "replica_history.npz"
    marker_path = out / MARKER
    key = clean_json(physics_key(args, ladder, world))
    fpar = args.kappa if args.fermions == "wilson" else args.mass
    defect_mask = build_defect_mask(args, geometry).to(device)

    seeds = stream_seeds(args.seed, rank)
    torch.manual_seed(seeds["global"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seeds["run"])
    swap_rng = GlobalRNGStream(device, seeds["swap"])

    def log0(message):
        if rank == 0 and not args.quiet:
            print(message, flush=True)

    # ---- start or resume (decided on rank 0, shared by every rank; module doc) -------------------
    scan = comm.rank0(lambda: (read_marker(out), own_files(out)), f"scan of {out}")
    marker, own = comm.from_rank0(scan)
    resume = bool(args.resume and marker is not None)
    if marker is not None:
        comm.all_ok(int(marker.get("world_size", world)) == world,
                    f"{out}: checkpoint of {marker.get('world_size')} ranks, this run has {world}")
        if not resume:
            comm.all_ok(args.overwrite, f"{out} holds a previous run (committed checkpoint at step {marker['step']}): "
                                        f"use --resume or --overwrite")
            log0(f"--overwrite: deleting the previous run's files in {out} ({len(own)} files)")
    elif any(name.split("/")[-1] in LEGACY_RANK_FILES for name in own):
        comm.all_ok(args.overwrite, f"{out} holds a checkpoint of the older single-generation layout "
                                    f"(rankNNN/checkpoint.json), which cannot be resumed: use --overwrite or another --out")
        log0(f"--overwrite: deleting the previous run's files in {out} ({len(own)} files)")
    elif own:
        log0(f"{out}: no committed checkpoint, only the partial files of an interrupted start ({own[:6]}...): "
             f"deleted, starting from scratch")
    elif args.resume:
        log0(f"--resume: no checkpoint in {out}, starting from scratch")
    if not resume and own:
        comm.rank0(lambda: remove_own_files(out, own), f"removal of the previous files in {out}")
    rank_dir.mkdir(parents=True, exist_ok=True)

    link_accepted = [0] * n_links
    link_proposed = [0] * n_links
    swap_attempts = 0
    committed_slot = SLOTS[1]  # the first checkpoint goes to slot 0
    history = History(world, C) if rank == 0 else None
    if resume:
        committed_slot, n_done = int(marker["slot"]), int(marker["step"])
        state_prefix, rex_state_path = rank_paths(out, rank, committed_slot)
        try:
            U, meta, rng_state = load_state(state_prefix, device=device)
            rex_state = torch.load(rex_state_path, map_location="cpu")
            ok, message = True, ""
        except (OSError, KeyError, RuntimeError, ValueError) as exc:
            ok, message = False, f"--resume: rank {rank} cannot read slot {committed_slot}: {exc.__class__.__name__}: {exc}"
        comm.all_ok(ok, message)
        extra = meta["extra"]
        comm.all_ok(extra.get("physics") == key and extra.get("rank") == rank,
                    f"--resume: checkpoint of rank {extra.get('rank')} with parameters {extra.get('physics')} "
                    f"differs from rank {rank} with {key}")
        comm.all_ok(meta["rng"]["device"].split(":")[0] == device.type,
                    f"--resume: checkpoint RNG is for {meta['rng']['device']}, run on {device}")
        comm.all_ok(int(rex_state["n_steps_done"]) == n_done and int(meta["trajectory"]) == n_done,
                    f"--resume: rank {rank} slot {committed_slot} holds step {int(rex_state['n_steps_done'])} / "
                    f"{int(meta['trajectory'])}, the committed checkpoint is step {n_done}")
        restore_generator(generator, rng_state)
        swap_rng.set_state(rex_state["swap_rng"])
        pt._pt_step = int(rex_state["pt_step"])
        if rank == 0:
            pt.parameter_matrix = rex_state["parameter_matrix"].to(device=device, dtype=pt.param_dtype).contiguous()
        pt._scatter_parameters()
        comm.all_ok(torch.equal(pt.local_parameters.cpu(), rex_state["local_parameters"]),
                    f"--resume: rank {rank} parameters after the scatter differ from its checkpoint")
        source = extra.get("source")
        if rank == 0:
            link_accepted = [int(v) for v in rex_state["link_accepted"]]
            link_proposed = [int(v) for v in rex_state["link_proposed"]]
            swap_attempts = int(rex_state["swap_attempts"])
        comm.rank0(lambda: history.load(hist_path, max(n_done - args.therm, 0)), f"loading {hist_path}")
    else:
        n_done = 0
        if args.start_config is None:
            U, source = start_links(args, geometry, device, C, args.beta, fpar, 1.0)
        else:
            U, source = start_links(args, geometry, device, world * C, args.beta, fpar, 1.0)
            U = U[rank * C:(rank + 1) * C].clone()
            source = f"{source} [configurations {rank * C}..{(rank + 1) * C - 1}]"

    qcd = QCDAction(geometry, "bc", beta=args.beta, bc=pt.local_parameters.to(torch.float64).cpu(),
                    defect_mask=defect_mask, n_gauge_substeps=args.n_gauge_substeps,
                    **action_kwargs(args), **fermion_kwargs(args))
    target = FermionREXTarget(qcd, tau=args.tau, n_steps=args.n_steps, integrator=args.integrator_levels)
    fermion = qcd.fermion
    monitor = SolveMonitor(fermion)

    def open_writer():
        if resume:  # an observables.csv written before columns were added
            upgrade_csv_header(csv_path, OBS_FIELDS, log=log0)
        w = ObservableWriter(csv_path, OBS_FIELDS, trajectory_field="step")
        if resume:
            w.truncate(n_done)
        return w
    writer = comm.rank0(open_writer, f"opening {csv_path}")

    total = args.therm + args.steps
    info = run_info("main/main_rex_fermions_distributed.py", argv, device, args)
    info.update(lattice=list(geometry.lattice), volume=geometry.volume, start_source=source, backend=backend,
                world_size=world, n_chains=C, fermions=args.fermions, mass=args.mass,
                defect_links=int(defect_mask[1].sum().item()), obs_fields=list(OBS_FIELDS))
    params_json = dict(vars(args))
    stopped = False

    def write_meta(final):
        try:
            summary, hist = summarize(history, csv_path, ladder, hist_path)
        except Exception as exc:  # checkpoint files are already written; never die on statistics
            print(f"WARNING: summary failed ({exc.__class__.__name__}: {exc})", file=sys.stderr, flush=True)
            summary, hist = {"status": f"summary_failed: {exc.__class__.__name__}: {exc}"}, None
        swap = dict(swap_meta(ladder, link_accepted, link_proposed), swap_attempts=swap_attempts)
        meta = {"algorithm": "rex_fermions_distributed", "backend": backend, "world_size": world,
                "device": str(device), "parameter_min": ladder[0], "parameter_max": ladder[-1],
                "parameter_ladder": ladder, "acceptance_rule": "exact", "parameters": params_json, "run": info,
                "steps_requested": int(args.steps), "steps_completed": len(history), "steps_done": n_done,
                "thermal_steps": int(args.therm), "stopped_by_time_limit": bool(final and stopped),
                "finished": bool(final and n_done >= total), "swap": swap, "summary": summary}
        if hist is not None:
            meta["replica_history"] = hist
        atomic_json_save(clean_json(meta), meta_path)
        return meta

    def checkpoint(final=False):
        """Two-generation checkpoint (module doc): write the free slot on every rank, confirm, commit."""
        nonlocal committed_slot
        slot = SLOTS[1 - SLOTS.index(committed_slot)]
        prefix, rex_path = rank_paths(out, rank, slot)
        state = {"n_steps_done": n_done, "pt_step": int(pt._pt_step),
                 "local_parameters": pt.local_parameters.to(torch.float64).cpu().clone(),
                 "swap_rng": swap_rng.get_state()}
        if rank == 0:
            state.update(parameter_matrix=pt.parameter_matrix.to(torch.float64).cpu().clone(),
                         link_accepted=list(link_accepted), link_proposed=list(link_proposed),
                         swap_attempts=swap_attempts)
        ok, message = True, ""
        try:
            _atomic_write(rex_path, lambda tmp: torch.save(state, tmp))
            save_state(prefix, U, geometry=geometry, beta=args.beta, trajectory=n_done,
                       time_sign=args.time_sign, generator=generator, **state_couplings(args, fpar),
                       extra={"physics": key, "rank": rank, "world_size": world, "source": source,
                              "seed": args.seed, "final": bool(final), "slot": slot})
        except Exception as exc:  # noqa: BLE001 - every rank must leave together (no commit)
            traceback.print_exc()
            ok, message = False, f"checkpoint of step {n_done}: rank {rank} failed ({exc.__class__.__name__}: {exc})"
        comm.all_ok(ok, message)  # every rank holds step n_done in `slot`

        def commit():
            history.save(hist_path, ladder, args.therm)  # before the marker: the history covers the commit
            atomic_json_save({"step": n_done, "slot": slot, "world_size": world}, marker_path)
        comm.rank0(commit, f"commit of the checkpoint of step {n_done}")
        committed_slot = slot
        meta = None
        if rank == 0:
            try:
                meta = write_meta(final)
            except Exception as exc:  # noqa: BLE001 - statistics never stop the run
                print(f"WARNING: run_meta failed ({exc.__class__.__name__}: {exc})", file=sys.stderr, flush=True)
        return meta

    if rank == 0 and not args.quiet:
        fdesc = f"N_f=2 Wilson kappa={args.kappa}" if args.fermions == "wilson" else f"staggered mass={args.mass}"
        print(f"backend={backend} world_size={world} device={device} REX {fdesc} ladder {ladder} x {C} chains, "
              f"lattice {geometry.lattice} SU({args.N}) beta={args.beta}, defect {args.defect_shape} "
              f"L_d={args.defect_size} ({int(defect_mask[1].sum())} links) -> {out}"
              + (f" (resumed at step {n_done})" if resume else ""), flush=True)

    loop_t0 = time.time()
    session_start = n_done
    stop_after_s = 60.0 * float(args.max_run_minutes)
    while n_done < total:
        if stop_after_s > 0 and n_done > session_start:
            stop = False
            if rank == 0:
                now = time.time()
                stop = (now - run_t0) + 1.5 * (now - loop_t0) / (n_done - session_start) > stop_after_s
            if comm.from_rank0(stop):
                stopped = True
                if rank == 0 and not args.quiet:
                    print(f"time_limit_stop=1 steps_done={n_done} of {total}", flush=True)
                break
        step = n_done
        thermal = step < args.therm
        s0, c0 = fermion.stats["solves"], fermion.stats["cg_iterations"]
        traj_labels = rung_indices(pt.parameter_matrix, pt.ladder).cpu().numpy() if rank == 0 else None
        _sync(device)
        t0 = time.perf_counter()
        if thermal:
            target.set_parameter(pt.local_parameters.to(torch.float64).cpu())
            U, accepted, dH, hmc_stats = target.trajectory(U, generator)
            hmc_stats = dict(hmc_stats, accepted=accepted, dH=dH)
            stats = {"link_accepted": [0] * n_links, "link_proposed": [0] * n_links}
        else:
            U, stats, hmc_stats = distributed_rex_step(pt, target, U, generator,
                                                       periodic_translation=args.periodic_translation,
                                                       swap_rng=swap_rng)
        _sync(device)
        wall = time.perf_counter() - t0
        if thermal:
            plaq = mean_plaquette(plaquette_fields(U, geometry.D), geometry.N).to(torch.float64).reshape(-1)
            s_d = torch.zeros(C, dtype=torch.float64)
        else:  # one set of plaquette fields for both (rex.step_observables)
            plaq, s_d = step_observables(qcd, U, pt.local_parameters.to(torch.float64).cpu())
        max_res, unconverged = monitor.interval()
        pack = torch.cat([plaq.cpu(), s_d.cpu().reshape(-1),
                          hmc_stats["exp_minus_dH"].to(torch.float64).cpu().reshape(-1),
                          hmc_stats["accepted"].to(torch.float64).cpu().reshape(-1),
                          torch.tensor([fermion.stats["solves"] - s0, fermion.stats["cg_iterations"] - c0, wall,
                                        -1.0 if max_res is None else max_res, unconverged],
                                       dtype=torch.float64)]).to(device)
        gathered = [torch.zeros_like(pack) for _ in range(world)] if rank == 0 else None
        dist.gather(pack, gather_list=gathered, dst=0)
        n_done += 1

        def record():
            nonlocal link_accepted, link_proposed, swap_attempts
            g = torch.stack(gathered, dim=0).cpu().numpy()  # (world, N_PACK * C + N_TAIL)
            per_chain = g[:, :N_PACK * C].reshape(world, N_PACK, C)
            plaq_w, sd_w, x_w, acc_w = (per_chain[:, i] for i in range(N_PACK))
            labels = rung_indices(pt.parameter_matrix, pt.ladder).cpu().numpy()  # (world, C)
            la = [int(v) for v in stats["link_accepted"]]
            lp = [int(v) for v in stats["link_proposed"]]
            if not thermal:
                history.append(labels=labels, plaquette=plaq_w, defect_action=sd_w,
                               trajectory_labels=traj_labels, exp_minus_dH=x_w)
                link_accepted = [a + b for a, b in zip(link_accepted, la)]
                link_proposed = [a + b for a, b in zip(link_proposed, lp)]
                swap_attempts += 1
            tail = g[:, N_PACK * C:]
            solves, cg_iter, wall_max = float(tail[:, 0].sum()), float(tail[:, 1].sum()), float(tail[:, 2].max())
            max_res = None
            for r in tail[:, 3]:  # NaN-sticky over the ranks; -1: that rank did not solve
                max_res = max_res if (not np.isnan(r) and r < 0) else nan_max(max_res, float(r))
            unconverged = int(tail[:, 4].sum())
            plaq_r, x_r = _per_rung(plaq_w, labels, world), _per_rung(x_w, traj_labels, world)
            hmc_acc = float(acc_w.mean())
            writer.write_rows([{"step": step, "therm": int(thermal), "rung": k, "bc": ladder[k],
                                "plaquette": plaq_r[k], "exp_mdH": x_r[k],
                                "swap_accepted": la[k] if k < n_links else None,
                                "swap_proposed": lp[k] if k < n_links else None,
                                "hmc_acceptance": hmc_acc, "solves": int(solves), "cg_iter": int(cg_iter),
                                "wall_s": wall_max, "max_residual": max_res, "unconverged_solves": unconverged}
                               for k in range(world)])
            if unconverged and not args.quiet:
                print(f"WARNING: step {step}: {unconverged} unconverged solves (all ranks), largest true relative "
                      f"residual {max_res:.2e} (--on-unconverged {args.on_unconverged})", file=sys.stderr, flush=True)
            if (step + 1) % args.log_every == 0 or step == 0:
                cum = [a / p if p else float("nan") for a, p in zip(link_accepted, link_proposed)]
                if not args.quiet:
                    print(f"step {step:5d}{' (therm)' if thermal else ''} plaq/rung {[round(v, 5) for v in plaq_r]} "
                          f"hmc acc {hmc_acc:.2f} swaps {la}/{lp} link acc {[round(v, 3) for v in cum]} "
                          f"solves {int(solves)} res {max_res} unconverged {unconverged} {wall_max:.2f} s", flush=True)
                if len(history):
                    history.save(hist_path, ladder, args.therm)
        comm.rank0(record, f"output of step {step}")
        if args.checkpoint_every and n_done % args.checkpoint_every == 0 and n_done < total:
            checkpoint()

    meta = checkpoint(final=True)
    result = {"out": str(out), "rank": rank, "world_size": world, "steps_done": n_done,
              "local_parameters": pt.local_parameters.cpu().tolist()}
    if rank == 0:
        result["parameter_matrix"] = pt.parameter_matrix.cpu().tolist()
        result["meta"] = meta
        if not args.quiet and meta is not None:
            h = meta.get("replica_history", {})
            conv = meta["summary"].get("convergence", {})
            if conv.get("max_residual") is not None:
                print(f"solver: largest true relative residual {conv['max_residual']:.2e}, "
                      f"{conv['unconverged_solves']} unconverged solves", flush=True)
            print(f"done: {meta['steps_completed']} measured steps, round trips {h.get('round_trips')}, "
                  f"round-trip steps {h.get('round_trip_steps')}, link acceptance {meta['swap']['link_acceptance']}, "
                  f"{meta['summary'].get('seconds_per_step')} s/step", flush=True)
    return result


if __name__ == "__main__":
    main()
