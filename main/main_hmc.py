"""HMC benchmark driver for dynamical fermions (fermions_plan.md, phase 1: Benchmark, cost).

Runs B independent HMC chains of
    --fermions wilson     N_f = 2 Wilson (TwoFlavourHamiltonian, even-odd pseudofermion),
    --fermions staggered  one staggered field, N_f = 2^(D/2) (StaggeredHamiltonian),
    --fermions none       pure Wilson gauge,
with the Wilson gauge action S_g = beta sum (1 - Re tr U_p / N). Fermion force on the outer time
scale, gauge force on the inner one with --n-gauge-substeps steps per outer position update.

Output directory (--out, default output/hmc_fermions/<auto name>):
    observables.csv   one row per (trajectory, replica); columns in OBS_FIELDS (doc below);
    observables.npz   the same columns as numpy arrays, all rows (written at the end);
    run_meta.json     parameters + summary (Gamma-method plaquette, acceptance, <exp(-dH)>,
                      seconds per trajectory and per solve, CG iterations per solve);
    checkpoint.pt / checkpoint.json   links, RNG state, trajectory count (neoqcd.fermions.io).
Trajectory numbering: `traj` = 0, 1, ... counts all trajectories, thermalisation included
(therm = 1 for the first --therm). --n-traj counts the measured ones, so the run ends at
traj = therm + n_traj. Summaries use the therm = 0 rows only.

--meas-every k: the exact log det (--exact-logdet) and the progress line are produced every k
trajectories (field empty otherwise); the cheap per-trajectory data (plaquette, dH, cost) are
always written.

--fast (Wilson only, default off) uses neoqcd.fermions.fast.FastTwoFlavourWilson: the same
action, heat bath and force through the half-lattice even-odd operator and lean_cg;
--precision mixed (with --fast) solves with complex64 inner CG and complex128 residuals to the
same tolerances; --colour-product picks the SU(N) x spinor product (auto: broadcast on CUDA,
matmul on CPU). All are recorded in run_meta.json ("parameters" and "run").

Resume: --resume reads <out>/checkpoint.*, checks that the physics parameters agree, drops CSV
rows past the checkpoint and continues with the stored generator state: the continuation is
bitwise identical to an uninterrupted run with the same seed (same device and thread count).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neoqcd.fermions._gauge import SUN_haar, SUN_identity  # noqa: E402
from neoqcd.fermions.hmc import HMC, INTEGRATORS, Hamiltonian, WilsonGaugeAction  # noqa: E402
from neoqcd.fermions.io import (  # noqa: E402
    ObservableWriter,
    atomic_json_save,
    load_state,
    n_flavours,
    read_observables,
    restore_generator,
    save_state,
    write_npz,
)
from neoqcd.fermions.fast import FastTwoFlavourHamiltonian, resolve_colour_product  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402
from neoqcd.fermions.pseudofermion import TwoFlavourHamiltonian  # noqa: E402
from neoqcd.fermions.staggered_pseudofermion import StaggeredHamiltonian  # noqa: E402

OBS_FIELDS = (
    "traj",            # trajectory index (0-based, thermalisation included)
    "replica",         # batch entry b
    "therm",           # 1 during thermalisation
    "plaquette",       # <Re tr U_p / N> of the configuration after the Metropolis step
    "dH",              # H(U', P') - H(U, P) of the proposal
    "accepted",        # 0/1
    "exp_mdH",         # exp(-dH)
    "logdet",          # exact log det of the fermion weight (Wilson: log det D^dag D; staggered: log det M_ee); empty if not measured
    "solves",          # CG solves in this trajectory (batched: one solve serves the B replicas)
    "cg_iter",         # sum over solves of the largest per-replica iteration count (cost of the batch)
    "cg_iter_replica", # CG iterations of this replica in this trajectory
    "wall_s",          # wall time of the trajectory (whole batch)
    "solve_s",         # wall time spent inside the CG solves of this trajectory
    "s_per_solve",     # solve_s / solves
)

STATE_NAME = "checkpoint"
DEFAULT_EXACT_MAX_DIM = 3072


# ---- arguments ----------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMC benchmark for N_f = 2 Wilson / staggered fermions.")
    p.add_argument("--D", type=int, default=4)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--beta", type=float, default=5.6)
    p.add_argument("--kappa", type=float, default=None, help="Wilson hopping parameter (--fermions wilson).")
    p.add_argument("--mass", type=float, default=None, help="Staggered bare mass (--fermions staggered).")
    p.add_argument("--fermions", choices=("wilson", "staggered", "none"), default="wilson")
    p.add_argument("--time-sign", type=float, default=-1.0, choices=(-1.0, 1.0),
                   help="Fermion boundary phase in time: -1 antiperiodic, +1 periodic.")
    p.add_argument("--batch", type=int, default=1, help="Independent chains B, run as one batch.")
    p.add_argument("--tau", type=float, default=1.0, help="Trajectory length.")
    p.add_argument("--n-steps", type=int, default=10, help="Outer integrator steps per trajectory.")
    p.add_argument("--n-gauge-substeps", type=int, default=4,
                   help="Gauge steps per outer position update (ignored with --fermions none).")
    p.add_argument("--integrator", default="omelyan",
                   help=f"One of {INTEGRATORS}, or one per level separated by commas (outer first).")
    p.add_argument("--solver-tol", type=float, default=1e-10, help="CG relative residual (force solves).")
    p.add_argument("--action-tol", type=float, default=None, help="CG relative residual (Metropolis solves).")
    p.add_argument("--max-iter", type=int, default=5000, help="CG iteration cap per solve.")
    p.add_argument("--no-reuse-guess", dest="reuse_guess", action="store_false",
                   help="Start every CG from zero (exactly reversible trajectories).")
    p.add_argument("--fast", action="store_true",
                   help="Wilson only: half-lattice operator and lean CG (neoqcd.fermions.fast); same physics.")
    p.add_argument("--precision", choices=("double", "mixed"), default="double",
                   help="With --fast: 'mixed' uses complex64 inner CG with complex128 residuals.")
    p.add_argument("--colour-product", choices=("auto", "matmul", "broadcast", "broadcast_sum"), default="auto",
                   help="With --fast: SU(N) x spinor product; auto = broadcast on CUDA, matmul on CPU.")
    p.add_argument("--check-every", type=int, default=10,
                   help="With --fast: host convergence check of the CG every k iterations.")
    p.add_argument("--start", choices=("cold", "hot"), default="cold")
    p.add_argument("--therm", type=int, default=0, help="Thermalisation trajectories (recorded with therm = 1).")
    p.add_argument("--n-traj", type=int, default=100, help="Measured trajectories after thermalisation.")
    p.add_argument("--meas-every", type=int, default=1, help="Exact log det and progress line every k trajectories.")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Checkpoint every k trajectories (0: end only).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--out", type=str, default=None, help="Output directory (default output/hmc_fermions/<name>).")
    p.add_argument("--resume", action="store_true", help="Continue from <out>/checkpoint.*.")
    p.add_argument("--overwrite", action="store_true", help="Start afresh in a directory that holds a previous run.")
    p.add_argument("--exact-logdet", action="store_true", help="Measure the exact fermion log det (tiny lattices only).")
    p.add_argument("--exact-max-dim", type=int, default=DEFAULT_EXACT_MAX_DIM,
                   help="Refuse --exact-logdet if the dense even-odd matrix is larger than this.")
    p.add_argument("--gamma-S", type=float, default=2.0, help="S parameter of the Gamma method.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args):
    if args.fermions == "wilson":
        if args.kappa is None:
            raise SystemExit("--fermions wilson needs --kappa")
        if args.mass is not None:
            raise SystemExit("--mass is for --fermions staggered")
    elif args.fermions == "staggered":
        if args.mass is None or args.mass <= 0:
            raise SystemExit("--fermions staggered needs --mass > 0")
        if args.kappa is not None:
            raise SystemExit("--kappa is for --fermions wilson")
    elif args.kappa is not None or args.mass is not None:
        raise SystemExit("--fermions none takes neither --kappa nor --mass")
    for name in ("D", "T", "L", "N", "batch", "n_steps", "n_gauge_substeps", "meas_every", "max_iter"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.N not in (2, 3):
        raise SystemExit("--N must be 2 or 3")
    if args.therm < 0 or args.n_traj < 0 or args.checkpoint_every < 0:
        raise SystemExit("--therm, --n-traj, --checkpoint-every must be >= 0")
    if args.fermions != "none" and (args.T % 2 or args.L % 2):
        raise SystemExit("even-odd pseudofermions need even T and L")
    if args.exact_logdet and args.fermions == "none":
        raise SystemExit("--exact-logdet needs fermions")
    if args.fast and args.fermions != "wilson":
        raise SystemExit("--fast is for --fermions wilson")
    if args.precision != "double" and not args.fast:
        raise SystemExit("--precision mixed needs --fast")
    if args.check_every < 1:
        raise SystemExit("--check-every must be >= 1")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite exclude each other")
    levels = 1 if args.fermions == "none" else 2
    integrator = [s.strip().lower() for s in args.integrator.split(",")]
    if len(integrator) == 1:
        integrator = integrator * levels
    if len(integrator) != levels or any(s not in INTEGRATORS for s in integrator):
        raise SystemExit(f"--integrator: {levels} level(s), each one of {INTEGRATORS}")
    args.integrator_levels = integrator


def default_name(args):
    if args.fermions == "wilson":
        f = f"wilson_k{args.kappa:g}"
    elif args.fermions == "staggered":
        f = f"staggered_m{args.mass:g}"
    else:
        f = "quenched"
    lat = f"{args.T}x{args.L}^{args.D - 1}".replace("^", "p")
    return f"{f}_b{args.beta:g}_D{args.D}_{lat}_SU{args.N}_B{args.batch}_s{args.seed}"


def physics_key(args):
    """Parameters that must agree between a checkpoint and a resumed run."""
    return {
        "geometry": {"D": args.D, "T": args.T, "L": args.L, "N": args.N},
        "batch": args.batch, "beta": float(args.beta), "fermions": args.fermions,
        "kappa": None if args.kappa is None else float(args.kappa),
        "mass": None if args.mass is None else float(args.mass),
        "time_sign": float(args.time_sign),
    }


# ---- runtime ------------------------------------------------------------------------------------
def choose_device(name):
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA is not available")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_hamiltonian(args, geometry):
    """Hamiltonian and its fermion term (None for --fermions none)."""
    kw = dict(time_sign=args.time_sign, solver_tol=args.solver_tol, action_tol=args.action_tol,
              max_iter=args.max_iter, reuse_guess=args.reuse_guess)
    if args.fermions == "wilson" and args.fast:
        ham = FastTwoFlavourHamiltonian(args.beta, args.kappa, geometry, n_gauge_substeps=args.n_gauge_substeps,
                                        precision=args.precision, check_every=args.check_every,
                                        colour_product=args.colour_product, **kw)
        return ham, ham.fermion
    if args.fermions == "wilson":
        ham = TwoFlavourHamiltonian(args.beta, args.kappa, geometry, n_gauge_substeps=args.n_gauge_substeps, **kw)
        return ham, ham.fermion
    if args.fermions == "staggered":
        ham = StaggeredHamiltonian(args.beta, args.mass, geometry, n_gauge_substeps=args.n_gauge_substeps, **kw)
        return ham, ham.fermion
    return Hamiltonian([([WilsonGaugeAction(args.beta, geometry)], 1)]), None


def exact_dimension(args, geometry):
    """Size of the dense even-odd matrix behind the exact log det."""
    n = geometry.volume // 2 * geometry.N
    return n * geometry.spin if args.fermions == "wilson" else n


def exact_logdet(args, geometry, U):
    """(B,) exact log det of the fermion weight (module doc of OBS_FIELDS)."""
    if args.fermions == "wilson":
        from neoqcd.fermions.exact import log_det_two_flavour
        return log_det_two_flavour(U, args.kappa, geometry, args.time_sign, method="assembled")
    from neoqcd.fermions.staggered_exact import log_det_staggered
    return log_det_staggered(U, args.mass, geometry, args.time_sign, method="assembled")


class SolveTimer:
    """Wraps the fermion term's `_solve` (instance attribute, the class is untouched) to time it."""

    def __init__(self, term, device):
        self.seconds = 0.0
        self.device = device
        original = term._solve

        def timed(op, tol):
            _sync(device)
            t0 = time.perf_counter()
            x = original(op, tol)
            _sync(device)
            self.seconds += time.perf_counter() - t0
            return x

        term._solve = timed


def initial_links(args, geometry, device, generator):
    shape = geometry.link_shape(args.batch)[:-1]
    if args.start == "hot":
        return SUN_haar(shape, device=device, dtype=torch.cdouble, generator=generator)
    return SUN_identity(shape, dtype=torch.cdouble, device=device)


# ---- analysis -----------------------------------------------------------------------------------
def gamma_summary(values, S=2.0):
    """Mean of an (n, B) array with the Gamma method (B replicas of one ensemble, neoqcd.obs.Obs).

    Falls back to the naive standard error (status "naive") with fewer than 5 rows per replica
    or if the Obs dependencies are missing.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n == 0:
        return {"value": None, "error": None, "tau_int": None, "n": 0, "status": "empty"}
    flat = values.reshape(-1)
    naive = float(flat.std(ddof=1) / math.sqrt(flat.size)) if flat.size > 1 else None
    out = {"value": float(flat.mean()), "error": naive, "error_of_error": None, "tau_int": None,
           "tau_int_error": None, "window": None, "n": int(n), "replicas": int(values.shape[1]), "status": "naive"}
    if n < 5:
        return out
    try:
        from neoqcd.obs import Obs
    except ImportError as exc:  # autograd / numdifftools missing
        out["status"] = f"naive ({exc.__class__.__name__})"
        return out
    if np.all(flat == flat[0]):  # constant data (e.g. accepted = 1 everywhere): no fluctuation
        out.update(error=0.0, status="constant")
        return out
    try:
        obs = Obs([values[:, b].copy() for b in range(values.shape[1])],
                  [f"hmc|r{b}" for b in range(values.shape[1])])
        obs.gamma_method(S=float(S))
        ens = next(iter(obs.e_tauint))
        out.update(value=float(obs.value), error=float(obs.dvalue), error_of_error=float(obs.ddvalue),
                   tau_int=float(obs.e_tauint[ens]), tau_int_error=float(obs.e_dtauint[ens]),
                   window=float(obs.e_windowsize[ens]), status="gamma")
    except Exception as exc:  # a statistics bug must never kill a run or a checkpoint
        out.update(robust_summary(values))
        out["status"] = f"gamma_failed: {exc.__class__.__name__}: {exc}"
    return out


def robust_summary(values):
    """Fallback estimate of an (n, B) array: error from the spread of the B replica means (B >= 2),
    else the naive error, plus binned errors (bins along the trajectory axis, all replicas)."""
    values = np.asarray(values, dtype=np.float64)
    n, B = values.shape
    flat = values.reshape(-1)
    res = {"value": float(flat.mean()), "error": None, "error_method": "naive", "binned_errors": {}}
    if B >= 2:
        means = values.mean(axis=0)
        res.update(error=float(means.std(ddof=1) / math.sqrt(B)), error_method="replica_means")
    elif flat.size > 1:
        res["error"] = float(flat.std(ddof=1) / math.sqrt(flat.size))
    for bs in (1, 2, 4, 8, 16):
        nb = n // bs
        if nb < 2 or nb * B < 2:
            break
        bins = values[:nb * bs].reshape(nb, bs, B).mean(axis=1).reshape(-1)
        res["binned_errors"][str(bs)] = float(bins.std(ddof=1) / math.sqrt(bins.size))
    return res


def summarize(csv_path, batch, S=2.0):
    """Summary of the therm = 0 rows of observables.csv (all derived from the file, resume-safe)."""
    cols = read_observables(csv_path)
    keep = cols["therm"] == 0
    cols = {k: v[keep] for k, v in cols.items()}
    order = np.lexsort((cols["replica"], cols["traj"]))
    cols = {k: v[order] for k, v in cols.items()}
    n_rows = cols["traj"].size
    if n_rows % batch:
        raise ValueError(f"{n_rows} measured rows are not a multiple of the batch {batch}")
    n_traj = n_rows // batch

    def per_replica(name):
        return cols[name].reshape(n_traj, batch)

    per_traj = {k: per_replica(k)[:, 0] for k in ("wall_s", "solve_s", "solves", "cg_iter")}
    total_solves = float(per_traj["solves"].sum())
    summary = {
        "n_traj_measured": int(n_traj),
        "plaquette": gamma_summary(per_replica("plaquette"), S),
        "acceptance": gamma_summary(per_replica("accepted"), S),
        "exp_minus_dH": gamma_summary(per_replica("exp_mdH"), S),
        "dH": {"mean": float(cols["dH"].mean()) if n_rows else None,
               "rms": float(np.sqrt((cols["dH"] ** 2).mean())) if n_rows else None},
        "seconds_per_traj": float(per_traj["wall_s"].mean()) if n_traj else None,
        "seconds_per_traj_excluding_first": float(per_traj["wall_s"][1:].mean()) if n_traj > 1 else None,
        "solve_seconds_per_traj": float(per_traj["solve_s"].mean()) if n_traj else None,
        "solves_per_traj": float(per_traj["solves"].mean()) if n_traj else None,
        "seconds_per_solve": float(per_traj["solve_s"].sum() / total_solves) if total_solves else None,
        "cg_iter_per_solve": float(per_traj["cg_iter"].sum() / total_solves) if total_solves else None,
        "cg_iter_per_traj": float(per_traj["cg_iter"].mean()) if n_traj else None,
        "seconds_per_cg_iter": (float(per_traj["solve_s"].sum() / per_traj["cg_iter"].sum())
                                if n_traj and per_traj["cg_iter"].sum() > 0 else None),
    }
    logdet = per_replica("logdet")
    measured = ~np.isnan(logdet[:, 0]) if n_traj else np.zeros(0, dtype=bool)
    if measured.any():
        summary["logdet"] = gamma_summary(logdet[measured], S)
    return summary, cols


# ---- main ---------------------------------------------------------------------------------------
def _log(args, *msg):
    if not args.quiet:
        print(*msg, flush=True)


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args.device)
    geometry = Geometry(args.D, args.T, args.L, args.N)
    if args.exact_logdet and exact_dimension(args, geometry) > args.exact_max_dim:
        raise SystemExit(f"--exact-logdet: dense matrix of size {exact_dimension(args, geometry)} "
                         f"> --exact-max-dim {args.exact_max_dim}")
    out = Path(args.out) if args.out else ROOT / "output" / "hmc_fermions" / default_name(args)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path, state_prefix = out / "observables.csv", out / "run_meta.json", out / STATE_NAME

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    torch.manual_seed(int(args.seed))  # no global draws are made; kept for hygiene

    started = datetime.now().isoformat(timespec="seconds")
    if args.resume:
        if not state_prefix.with_suffix(".json").exists():
            raise SystemExit(f"--resume: no checkpoint in {out}")
        U, meta, rng_state = load_state(state_prefix, device=device)
        stored = {k: meta[k] for k in ("geometry", "batch", "beta", "fermions", "kappa", "mass", "time_sign")}
        if stored != physics_key(args):
            raise SystemExit(f"--resume: checkpoint parameters {stored} differ from {physics_key(args)}")
        if meta["rng"]["device"].split(":")[0] != device.type:
            raise SystemExit(f"--resume: checkpoint RNG is for {meta['rng']['device']}, run on {device}")
        restore_generator(generator, rng_state)
        traj_done = int(meta["trajectory"])
        writer = ObservableWriter(csv_path, OBS_FIELDS)
        writer.truncate(traj_done)
        _log(args, f"resumed {out} at trajectory {traj_done}")
    else:
        previous = [p for p in (csv_path, state_prefix.with_suffix(".json"), state_prefix.with_suffix(".pt"), meta_path)
                    if p.exists()]
        if previous and not args.overwrite:
            raise SystemExit(f"{out} holds a previous run ({[p.name for p in previous]}): use --resume or --overwrite")
        for p in previous:  # --overwrite: the files of the previous run in this directory are replaced
            p.unlink()
        U = initial_links(args, geometry, device, generator)
        traj_done = 0
        writer = ObservableWriter(csv_path, OBS_FIELDS)

    ham, fermion = build_hamiltonian(args, geometry)
    integrator = args.integrator_levels
    hmc = HMC(geometry, ham, tau=args.tau, n_steps=args.n_steps, integrator=integrator)
    timer = SolveTimer(fermion, device) if fermion is not None else None

    total = args.therm + args.n_traj
    params = {k: v for k, v in vars(args).items()}
    run_info = {
        "script": "main/main_hmc.py", "argv": list(sys.argv[1:] if argv is None else argv),
        "started": started, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__, "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
        "n_flavours": n_flavours(args.fermions, args.D), "lattice": list(geometry.lattice),
        "volume": geometry.volume, "dtype": "complex128", "obs_fields": list(OBS_FIELDS),
        "fermion_operator": "half_lattice_fast" if args.fast else "reference",
        "precision": args.precision,
        "colour_product": resolve_colour_product(args.colour_product, device) if args.fast else None,
    }

    def checkpoint(final=False):
        save_state(state_prefix, U, geometry=geometry, beta=args.beta, fermions=args.fermions,
                   trajectory=traj_done, time_sign=args.time_sign, kappa=args.kappa, mass=args.mass,
                   generator=generator, extra={"seed": args.seed, "final": bool(final)})
        try:
            summary, _ = summarize(csv_path, args.batch, args.gamma_S)
        except Exception as exc:  # checkpoint files are already written; never die on statistics
            print(f"WARNING: summary failed ({exc.__class__.__name__}: {exc})", file=sys.stderr, flush=True)
            summary = {"status": f"summary_failed: {exc.__class__.__name__}: {exc}"}
        if final:
            write_npz(out / "observables.npz", read_observables(csv_path))  # all rows, therm included
        atomic_json_save({"parameters": params, "run": run_info, "trajectories_done": traj_done,
                          "finished": bool(final and traj_done >= total), "summary": summary}, meta_path)
        return summary

    _log(args, f"HMC {args.fermions} N_f={n_flavours(args.fermions, args.D)} lattice {geometry.lattice} SU({args.N}) "
               f"beta={args.beta} kappa={args.kappa} mass={args.mass} B={args.batch} device={device} -> {out}")
    while traj_done < total:
        if fermion is not None:
            s0 = dict(fermion.stats)
            per_sample0 = s0["cg_iterations_per_sample"]
            timer.seconds = 0.0
        _sync(device)
        t0 = time.perf_counter()
        U, accepted, dH, stats = hmc.trajectory(U, generator)
        _sync(device)
        wall = time.perf_counter() - t0

        if fermion is not None:
            solves = fermion.stats["solves"] - s0["solves"]
            cg_iter = fermion.stats["cg_iterations"] - s0["cg_iterations"]
            per_sample = fermion.stats["cg_iterations_per_sample"]
            if per_sample is None:
                cg_rep = torch.zeros(args.batch, dtype=torch.long)
            elif per_sample0 is None or per_sample0.shape != per_sample.shape:
                cg_rep = per_sample.cpu()
            else:
                cg_rep = (per_sample - per_sample0).cpu()
            solve_s = timer.seconds
        else:
            solves, cg_iter, solve_s = 0, 0, 0.0
            cg_rep = torch.zeros(args.batch, dtype=torch.long)

        measure = traj_done % args.meas_every == 0
        logdet = None
        if args.exact_logdet and measure:
            logdet = exact_logdet(args, geometry, U).cpu().tolist()
        plaq = stats["plaquette"].cpu().tolist()
        dHl = dH.cpu().tolist()
        acc = accepted.cpu().tolist()
        emdH = stats["exp_minus_dH"].cpu().tolist()
        rows = [{
            "traj": traj_done, "replica": b, "therm": int(traj_done < args.therm),
            "plaquette": plaq[b], "dH": dHl[b], "accepted": int(acc[b]), "exp_mdH": emdH[b],
            "logdet": None if logdet is None else logdet[b],
            "solves": solves, "cg_iter": cg_iter, "cg_iter_replica": int(cg_rep[b]),
            "wall_s": wall, "solve_s": solve_s, "s_per_solve": (solve_s / solves) if solves else None,
        } for b in range(args.batch)]
        writer.write_rows(rows)
        traj_done += 1

        if measure:
            _log(args, f"traj {traj_done - 1:6d} plaq {np.mean(plaq):.6f} dH {np.mean(dHl):+.3e} "
                       f"acc {np.mean(acc):.2f} solves {solves} cg/solve {cg_iter / max(solves, 1):.1f} "
                       f"{wall:.3f} s" + ("" if logdet is None else f" logdet {np.mean(logdet):.4f}"))
        if args.checkpoint_every and traj_done % args.checkpoint_every == 0 and traj_done < total:
            checkpoint()

    summary = checkpoint(final=True)
    if "plaquette" in summary:
        p = summary["plaquette"]
        _log(args, f"done: {summary['n_traj_measured']} trajectories, plaquette {p['value']} +- {p['error']} "
                   f"(tau_int {p['tau_int']}), acceptance {summary['acceptance']['value']}, "
                   f"<exp(-dH)> {summary['exp_minus_dH']['value']}, {summary['seconds_per_traj']} s/traj, "
                   f"{summary['seconds_per_solve']} s/solve, {summary['cg_iter_per_solve']} CG it/solve")
    else:
        _log(args, f"done: {traj_done} trajectories, summary {summary.get('status')}")
    return {"out": str(out), "summary": summary}


if __name__ == "__main__":
    main()
