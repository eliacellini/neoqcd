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
swaps, no translation, not in replica_history.npz (rows flagged therm = 1 in observables.csv).

Defect: `utils.Defect(D, T, L, --defect-size, --time-slice, --space-slice)` (default --defect-shape
cube), the defect of main_pt_obc.py for L_d = --defect-size; --defect-shape line is the fermion-only
test shape. Ladder: world_size rungs, --ladder (aliases --ladder-file, --parameter-ladder: a list or
the JSON file of main/optimal_ladder.py) or evenly spaced in [--bc-min, --bc-max] (the same
`distributedPT.ladder_values` as main_pt_obc.py).

Devices: --device cpu (Gloo), or cuda / auto: rank with LOCAL_RANK = l uses cuda:l (NCCL) with an
explicit index (torch 2.0 cannot set_device("cuda") without one); if each rank sees one GPU only
(device_count() == 1) it uses cuda:0. --device cuda:K forces index K. Launch with torchrun
(job/debug_rex_fermions_distributed.sh) or any env:// launcher (RANK, WORLD_SIZE, MASTER_ADDR,
MASTER_PORT, LOCAL_RANK).

Generators: on each rank a run generator on its device, seeded with --seed + rank (pseudofermions,
momenta, Metropolis, translation). The swap uniforms of rank 0 come from `torch.rand` on the global
generator of pt.device inside `ParallelTempering`; they are drawn from a private stream
(`rex.GlobalRNGStream`, seed --seed + SWAP_SEED_OFFSET) so that a resumed run repeats them.

Output directory (--out, alias --output-dir; default output/rex_fermions_distributed/<auto name>):
    observables.csv     (rank 0) one row per (step, rung), the fields of main_rex_fermions.py
                        (OBS_FIELDS): plaquette = mean over chains of the configuration carrying the
                        rung after the step, exp_mdH of the trajectories run at that rung, swaps of the
                        pair (rung, rung + 1), HMC acceptance of all replicas; solves and cg_iter are
                        summed over the ranks, wall_s is the slowest rank's step time;
    replica_history.npz (rank 0) measured steps only: labels (steps, world_size, n_chains) int8, the
                        rung of each chain slot after the step (`replica_history.rung_indices`);
                        plaquette, defect_action (S_D = dS/dc at the slot's bc, `rex.defect_action`),
                        trajectory_labels, exp_minus_dH (same shape); parameter_ladder; swap_every;
                        thermal_steps. The keys and layout of main_pt_obc.py: main/optimal_ladder.py
                        reads the directory unchanged. Rewritten every --log-every steps and at
                        every checkpoint;
    run_meta.json       (rank 0) algorithm, backend, world_size, parameters, the "swap" and
                        "replica_history" blocks of main_pt_obc.py (round trips, flux, plaquette tau_int,
                        defect-action moments), per-rung plaquette and cost per step and per round trip;
    rankNNN/checkpoint.pt/.json   links and run generator of rank NNN (`fermions.io.save_state`);
    rankNNN/rex_state.pt          step count, PT state (`pt._pt_step`, rank 0's parameter matrix,
                                  local parameters), swap-stream state, swap counters (rank 0).

`ParallelTempering` exposes no state_dict (main_pt_obc.py has no resume); its state is the
parameter matrix on rank 0, the step counter `_pt_step` (alternating parity) and the scattered local
parameters, restored here by setting the first two and calling `_scatter_parameters()`.

Resume: --resume restores every rank from its rankNNN files (all ranks must hold the same step),
truncates observables.csv and the history to the checkpoint and continues; the continuation repeats
the uninterrupted run with the same seed. --max-run-minutes stops (with a checkpoint) before the step
that would end past that time, as in main_pt_obc.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
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
    action_kwargs,
    add_physics_args,
    build_defect_mask,
    clean_json,
    defect_key,
    fermion_kwargs,
    run_info,
    staggered_key,
    start_links,
    state_couplings,
    validate_common,
)

from neoqcd.distributedPT import ParallelTempering, PTConfig  # noqa: E402
from neoqcd.fermions._gauge import mean_plaquette, plaquette_fields, rung_indices, summarize_history  # noqa: E402
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
from neoqcd.fermions.rex import FermionREXTarget, GlobalRNGStream, defect_action, distributed_rex_step  # noqa: E402

STATE_NAME = "checkpoint"
REX_STATE = "rex_state.pt"
SWAP_SEED_OFFSET = 1_000_003
N_PACK = 4  # per-chain values gathered each step: plaquette, defect action, exp(-dH), HMC accepted


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


# ---- runtime ------------------------------------------------------------------------------------
def choose_rank_device(spec):
    """(device, backend) of this rank (module doc): Gloo on CPU, NCCL on cuda:<explicit index>."""
    if spec == "cpu" or (spec == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu"), "gloo"
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {spec} but CUDA is not available")
    if spec in ("cuda", "auto"):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        count = torch.cuda.device_count()
        index = 0 if count == 1 else local_rank
        if not 0 <= index < count:
            raise SystemExit(f"LOCAL_RANK={local_rank} but {count} CUDA devices are visible")
        device = torch.device("cuda", index)
    else:
        device = torch.device(spec)
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
        with np.load(path) as h:
            if h["labels"].shape[0] < n_steps:
                raise SystemExit(f"--resume: {path} holds {h['labels'].shape[0]} steps, the checkpoint {n_steps}")
            for name in self.NAMES:
                self.data[name] = [np.array(x) for x in h[name][:n_steps]]


def summarize(history, csv_path, ladder, path):
    """Pure-gauge "replica_history" block and the cost summary of the measured steps."""
    n_r = len(ladder)
    summary = {"steps_measured": len(history)}
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
    device, backend = choose_rank_device(args.device)
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
    ladder = resolve_ladder(args, world)
    cfg = PTConfig(batch_size=C, parameter_min=ladder[0], parameter_max=ladder[-1], parameter_name="bc",
                   parameter_dtype=torch.float64, swap_parity=args.swap_parity, acceptance_rule="exact",
                   parameter_ladder=ladder if explicit else None)
    pt = ParallelTempering(cfg, rank, world, device)
    ladder = [float(v) for v in pt.ladder.cpu().tolist()]  # the values the swaps use (main_pt_obc.py)

    geometry = Geometry(args.D, args.T, args.L, args.N)
    out = Path(args.out) if args.out else ROOT / "output" / "rex_fermions_distributed" / default_name(args, world)
    rank_dir = out / f"rank{rank:03d}"
    csv_path, meta_path, hist_path = out / "observables.csv", out / "run_meta.json", out / "replica_history.npz"
    state_prefix, rex_state_path = rank_dir / STATE_NAME, rank_dir / REX_STATE
    key = clean_json(physics_key(args, ladder, world))
    fpar = args.kappa if args.fermions == "wilson" else args.mass
    defect_mask = build_defect_mask(args, geometry).to(device)

    torch.manual_seed(int(args.seed) + rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + rank)
    swap_rng = GlobalRNGStream(device, int(args.seed) + SWAP_SEED_OFFSET)

    # ---- start or resume (decisions shared by every rank) --------------------------------------
    previous = []
    if rank == 0 and out.exists():
        previous = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    previous = comm.from_rank0(previous)
    if args.resume:
        ok = state_prefix.with_suffix(".json").exists() and rex_state_path.exists()
        comm.all_ok(ok, f"--resume: no checkpoint in {rank_dir}")
    elif previous:
        comm.all_ok(args.overwrite, f"{out} holds a previous run ({previous[:6]}...): use --resume or --overwrite")
        dist.barrier()
        if rank == 0:
            for name in previous:
                (out / name).unlink()
        dist.barrier()
    rank_dir.mkdir(parents=True, exist_ok=True)

    link_accepted = [0] * n_links
    link_proposed = [0] * n_links
    swap_attempts = 0
    history = History(world, C) if rank == 0 else None
    if args.resume:
        U, meta, rng_state = load_state(state_prefix, device=device)
        rex_state = torch.load(rex_state_path, map_location="cpu")
        extra = meta["extra"]
        comm.all_ok(extra.get("physics") == key and extra.get("rank") == rank,
                    f"--resume: checkpoint of rank {extra.get('rank')} with parameters {extra.get('physics')} "
                    f"differs from rank {rank} with {key}")
        comm.all_ok(meta["rng"]["device"].split(":")[0] == device.type,
                    f"--resume: checkpoint RNG is for {meta['rng']['device']}, run on {device}")
        n_done = int(rex_state["n_steps_done"])
        comm.all_ok(n_done == int(meta["trajectory"]), f"--resume: {rank_dir} checkpoint.* and {REX_STATE} differ")
        steps = torch.tensor([n_done, -n_done], dtype=torch.int64, device=device)
        dist.all_reduce(steps, op=dist.ReduceOp.MAX)
        comm.all_ok(int(steps[0]) == n_done and -int(steps[1]) == n_done,
                    f"--resume: the ranks hold checkpoints of different steps (rank {rank}: {n_done})")
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
            history.load(hist_path, max(n_done - args.therm, 0))
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
    writer = None
    if rank == 0:
        writer = ObservableWriter(csv_path, OBS_FIELDS, trajectory_field="step")
        if args.resume:
            writer.truncate(n_done)

    total = args.therm + args.steps
    info = run_info("main/main_rex_fermions_distributed.py", argv, device)
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
        state = {"n_steps_done": n_done, "pt_step": int(pt._pt_step),
                 "local_parameters": pt.local_parameters.to(torch.float64).cpu().clone(),
                 "swap_rng": swap_rng.get_state()}
        if rank == 0:
            state.update(parameter_matrix=pt.parameter_matrix.to(torch.float64).cpu().clone(),
                         link_accepted=list(link_accepted), link_proposed=list(link_proposed),
                         swap_attempts=swap_attempts)
        _atomic_write(rex_state_path, lambda tmp: torch.save(state, tmp))
        save_state(state_prefix, U, geometry=geometry, beta=args.beta, trajectory=n_done,
                   time_sign=args.time_sign, generator=generator, **state_couplings(args, fpar),
                   extra={"physics": key, "rank": rank, "world_size": world, "source": source,
                          "seed": args.seed, "final": bool(final)})
        meta = None
        if rank == 0:
            history.save(hist_path, ladder, args.therm)
            try:
                meta = write_meta(final)
            except Exception as exc:  # never skip the barrier below on rank 0 alone
                print(f"WARNING: run_meta failed ({exc.__class__.__name__}: {exc})", file=sys.stderr, flush=True)
        dist.barrier()
        return meta

    if rank == 0 and not args.quiet:
        fdesc = f"N_f=2 Wilson kappa={args.kappa}" if args.fermions == "wilson" else f"staggered mass={args.mass}"
        print(f"backend={backend} world_size={world} device={device} REX {fdesc} ladder {ladder} x {C} chains, "
              f"lattice {geometry.lattice} SU({args.N}) beta={args.beta}, defect {args.defect_shape} "
              f"L_d={args.defect_size} ({int(defect_mask[1].sum())} links) -> {out}"
              + (f" (resumed at step {n_done})" if args.resume else ""), flush=True)

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
        plaq = mean_plaquette(plaquette_fields(U, geometry.D), geometry.N).to(torch.float64).reshape(-1)
        s_d = (torch.zeros(C, dtype=torch.float64) if thermal
               else defect_action(qcd, U, pt.local_parameters.to(torch.float64).cpu()))
        pack = torch.cat([plaq.cpu(), s_d.cpu().reshape(-1),
                          hmc_stats["exp_minus_dH"].to(torch.float64).cpu().reshape(-1),
                          hmc_stats["accepted"].to(torch.float64).cpu().reshape(-1),
                          torch.tensor([fermion.stats["solves"] - s0, fermion.stats["cg_iterations"] - c0, wall],
                                       dtype=torch.float64)]).to(device)
        gathered = [torch.zeros_like(pack) for _ in range(world)] if rank == 0 else None
        dist.gather(pack, gather_list=gathered, dst=0)
        n_done += 1

        if rank == 0:
            g = torch.stack(gathered, dim=0).cpu().numpy()  # (world, N_PACK * C + 3)
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
            solves, cg_iter, wall_max = float(g[:, -3].sum()), float(g[:, -2].sum()), float(g[:, -1].max())
            plaq_r, x_r = _per_rung(plaq_w, labels, world), _per_rung(x_w, traj_labels, world)
            hmc_acc = float(acc_w.mean())
            writer.write_rows([{"step": step, "therm": int(thermal), "rung": k, "bc": ladder[k],
                                "plaquette": plaq_r[k], "exp_mdH": x_r[k],
                                "swap_accepted": la[k] if k < n_links else None,
                                "swap_proposed": lp[k] if k < n_links else None,
                                "hmc_acceptance": hmc_acc, "solves": int(solves), "cg_iter": int(cg_iter),
                                "wall_s": wall_max} for k in range(world)])
            if (step + 1) % args.log_every == 0 or step == 0:
                cum = [a / p if p else float("nan") for a, p in zip(link_accepted, link_proposed)]
                if not args.quiet:
                    print(f"step {step:5d}{' (therm)' if thermal else ''} plaq/rung {[round(v, 5) for v in plaq_r]} "
                          f"hmc acc {hmc_acc:.2f} swaps {la}/{lp} link acc {[round(v, 3) for v in cum]} "
                          f"solves {int(solves)} {wall_max:.2f} s", flush=True)
                if len(history):
                    history.save(hist_path, ladder, args.therm)
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
            print(f"done: {meta['steps_completed']} measured steps, round trips {h.get('round_trips')}, "
                  f"round-trip steps {h.get('round_trip_steps')}, link acceptance {meta['swap']['link_acceptance']}, "
                  f"{meta['summary'].get('seconds_per_step')} s/step", flush=True)
    return result


if __name__ == "__main__":
    main()
