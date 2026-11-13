"""REX (parallel tempering on the gauge-only defect) with N_f = 2 Wilson fermions (fermions_plan.md,
phase 5), single process: `neoqcd.fermions.rex.FermionREX`.

--n-chains independent ladders of n_rungs replicas (bc values --ladder 0,0.5,1, or --n-rep values
uniform in [--bc-min, 1]) run as one batch of B = n_rungs * n_chains configurations. One REX step =
one fermionic HMC trajectory of every replica at its bc + swap round(s) between neighbouring rungs
(--swap-parity alternating: DEO, even pairs at even steps; even_odd: both rounds every step), with
the solve-free gauge-only swap dS of `rex.swap_action_difference`; optional translation of the
bc = 1 replicas (--periodic-translation). The defect is that of main_snf_fermions.py
(--defect-shape cube|line, --defect-size, --defect-time-slice, --defect-space-slice). The default
--defect-shape cube is `utils.Defect(D, T, L, L_d, time_slice, space_slice)`, built exactly as
main/main_pt_obc.py builds it for --defect-size L_d (the pure-gauge spellings --time-slice and
--space-slice are accepted too); "line" is a fermion-only test shape.

Ladder: --ladder (aliases --ladder-file, --parameter-ladder) takes a comma-separated list or the JSON
file written by main/optimal_ladder.py, parsed and validated by `neoqcd.ladder` as in main_pt_obc.py;
or --n-rep values evenly spaced in [--bc-min, --bc-max].

Optimal ladder: replica_history.npz holds `defect_action` (S_D = dS/dc of each slot's configuration
at its bc, `rex.defect_action`) and `parameter_ladder`, and run_meta.json the "swap" block of
main_pt_obc.py, so main/optimal_ladder.py reads this directory unchanged (--metric variance or
acceptance). Unlike main_pt_obc.py, the history here includes the --therm steps (flagged by
`therm` in the file): pass optimal_ladder.py a --discard of at least therm / (therm + steps).

Output directory (--out, default output/rex_fermions/<auto name>):
    observables.csv     one row per (step, rung) (OBS_FIELDS): rung-averaged plaquette (mean over
                        chains of the configuration carrying the rung after the step), <exp(-dH)>
                        of the trajectories run at that rung, swap attempts/acceptances of the
                        pair (rung, rung + 1) in this step, cost of the step;
    replica_history.npz labels (steps, n_rungs, n_chains): rung carried by each slot after the
                        swaps; plaquette (same shape): plaquette of each slot's configuration;
                        defect_action (same shape): S_D = dS/dc at the slot's bc after the step;
                        trajectory_labels, exp_minus_dH; ladder = parameter_ladder; swap_every (1);
                        therm (number of leading thermalisation steps). Layout of
                        `neoqcd.replica_history` (world_size -> slot, B -> chain);
    run_meta.json       algorithm, parameters, "swap" and "replica_history" blocks in the layout of
                        main_pt_obc.py (steps after --therm), and the summary: `FermionREX.summarize()`
                        (round trips, mean round-trip time, flux, plaquette tau_int on the bottom
                        and top rungs), swap acceptance per pair, plaquette per rung, HMC
                        acceptance, solves / CG iterations / seconds per step;
    checkpoint.pt/.json links + run generator (`neoqcd.fermions.io.save_state`);
    rex_state.pt        labels, step count, swap counters, swap-generator state.

Two generators: the run generator (on the device: pseudofermions, momenta, Metropolis, translation)
and a CPU generator for the swap uniforms (seed + 1), passed to `FermionREX(swap_generator=...)`
(which keeps labels, swaps and history on the CPU, so the driver also runs on CUDA).

Resume: --resume restores links, labels, counters, both generators and the history (truncated to
the checkpoint) and continues; the continuation repeats an uninterrupted run with the same seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from main_hmc import _sync, choose_device  # noqa: E402
from main_snf_fermions import (  # noqa: E402
    _log,
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

from neoqcd.fermions.io import (  # noqa: E402
    ObservableWriter,
    _atomic_write,
    atomic_json_save,
    load_state,
    read_observables,
    restore_generator,
    save_state,
)
from neoqcd.fermions._gauge import parse_ladder, validate_ladder  # noqa: E402
from neoqcd.fermions.lattice import Geometry  # noqa: E402
from neoqcd.fermions.rex import FermionREX  # noqa: E402

OBS_FIELDS = (
    "step",           # REX step (0-based, thermalisation included)
    "therm",          # 1 during the first --therm steps
    "rung",           # rung k (0 = smallest bc)
    "bc",             # ladder[k]
    "plaquette",      # mean over chains of the plaquette of the configuration at rung k after the step
    "exp_mdH",        # mean over chains of exp(-dH) of the trajectories run at rung k
    "swap_accepted",  # accepted swaps of the pair (k, k+1) in this step (empty for the top rung)
    "swap_proposed",  # proposed swaps of the pair (k, k+1) in this step
    "hmc_acceptance", # HMC acceptance of the step, all replicas
    "solves",         # CG solves of the step (batched over all replicas)
    "cg_iter",        # sum over those solves of the largest per-replica iteration count
    "wall_s",         # wall time of the step
)

STATE_NAME = "checkpoint"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="REX on the gauge-only defect with N_f = 2 Wilson fermions.")
    add_physics_args(p)
    add_rex_args(p)
    p.add_argument("--n-rep", type=int, default=None, help="Uniform ladder of n values in [--bc-min, --bc-max].")
    p.add_argument("--n-chains", type=int, default=1)
    p.add_argument("--steps", type=int, default=100, help="Measured REX steps after --therm.")
    p.add_argument("--swap-parity", choices=("alternating", "even_odd"), default="alternating")
    p.add_argument("--periodic-translation", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--gamma-S", type=float, default=2.0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)
    validate_common(args)
    if (args.ladder is None) == (args.n_rep is None):
        raise SystemExit("give exactly one of --ladder (--ladder-file) and --n-rep")
    if args.ladder is None and args.n_rep < 2:
        raise SystemExit("--n-rep must be >= 2")
    args.ladder_values = resolve_ladder(args, args.n_rep)
    if args.n_chains < 1 or args.steps < 0 or args.checkpoint_every < 0:
        raise SystemExit("--n-chains >= 1, --steps >= 0, --checkpoint-every >= 0")
    return args


# ---- options and outputs shared with main_rex_fermions_distributed.py ---------------------------
def add_rex_args(p):
    """Ladder options and the pure-gauge spellings of the defect options (main/main_pt_obc.py).

    --ladder, --ladder-file and --parameter-ladder are one option: a comma-separated list or the
    JSON file of main/optimal_ladder.py (`neoqcd.ladder.parse_ladder`, as --parameter-ladder of
    main_pt_obc.py). --time-slice / --space-slice are the pure-gauge names of --defect-time-slice /
    --defect-space-slice (same destination: the last one given wins).
    """
    p.add_argument("--ladder", "--ladder-file", "--parameter-ladder", dest="ladder", type=str, default=None,
                   help="Explicit increasing bc values: comma-separated, or a JSON file written by "
                        "main/optimal_ladder.py (a list or a dict with key 'ladder'). Top rung bc = 1 for "
                        "--periodic-translation.")
    p.add_argument("--bc-min", type=float, default=0.0, help="Lowest bc of a uniform ladder.")
    p.add_argument("--bc-max", type=float, default=1.0, help="Highest bc of a uniform ladder.")
    p.add_argument("--time-slice", dest="defect_time_slice", type=int, default=argparse.SUPPRESS,
                   help="Alias of --defect-time-slice (main_pt_obc.py name).")
    p.add_argument("--space-slice", dest="defect_space_slice", type=int, default=argparse.SUPPRESS,
                   help="Alias of --defect-space-slice (main_pt_obc.py name).")


def resolve_ladder(args, n_rep):
    """The ladder as a list of floats: --ladder (parsed and validated as main_pt_obc.py does,
    `parse_ladder` + `validate_ladder` with n_rep rungs if n_rep is given), or n_rep values evenly
    spaced in [--bc-min, --bc-max] (`np.linspace`; `distributedPT.ladder_values` gives the same
    numbers up to rounding of the last bit)."""
    try:
        if args.ladder is not None:
            lad = validate_ladder(parse_ladder(args.ladder), n_rep)
        else:
            if args.bc_min >= args.bc_max:
                raise ValueError(f"--bc-min must be < --bc-max, got {args.bc_min} >= {args.bc_max}")
            lad = validate_ladder([float(v) for v in np.linspace(args.bc_min, args.bc_max, int(n_rep))])
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"--ladder: {exc}")
    if args.periodic_translation and lad[-1] != 1.0:
        raise SystemExit("--periodic-translation needs bc = 1 on the top rung")
    return lad


def swap_meta(ladder, accepted, proposed):
    """The "swap" block of main_pt_obc.py's run_meta.json (read by main/optimal_ladder.py --metric
    acceptance): per-link counts and acceptances, link k = (ladder[k], ladder[k + 1])."""
    ladder = [float(v) for v in ladder]
    accepted = [int(a) for a in accepted]
    proposed = [int(p) for p in proposed]
    acc = [a / p if p > 0 else float("nan") for a, p in zip(accepted, proposed)]
    tot_a, tot_p = sum(accepted), sum(proposed)
    return {
        "acceptance_mean": tot_a / tot_p if tot_p > 0 else float("nan"),
        "total_accepted": tot_a,
        "total_proposed": tot_p,
        "parameter_name": "bc",
        "link_parameter_left": ladder[:-1],
        "link_parameter_right": ladder[1:],
        "link_delta_parameter": [b - a for a, b in zip(ladder[:-1], ladder[1:])],
        "link_acceptance": acc,
        "link_accepted": accepted,
        "link_proposed": proposed,
    }


def default_name(args):
    lat = f"{args.T}x{args.L}p{args.D - 1}"
    return (f"rex_r{len(args.ladder_values)}_c{args.n_chains}_d{args.defect_shape}{args.defect_size}"
            f"_b{args.beta:g}_" + (f"k{args.kappa:g}" if args.fermions == "wilson" else f"stag_m{args.mass:g}") +
            f"_D{args.D}_{lat}_SU{args.N}_s{args.seed}")


def physics_key(args):
    return {
        "geometry": {"D": args.D, "T": args.T, "L": args.L, "N": args.N},
        "beta": float(args.beta), "kappa": None if args.kappa is None else float(args.kappa), "time_sign": float(args.time_sign),
        "ladder": list(args.ladder_values), "n_chains": args.n_chains, "defect": defect_key(args),
        "tau": float(args.tau), "n_steps": args.n_steps, "n_gauge_substeps": args.n_gauge_substeps,
        "integrator": list(args.integrator_levels), "swap_parity": args.swap_parity,
        "periodic_translation": bool(args.periodic_translation),
        **staggered_key(args),
    }


def save_history(path, rex, therm):
    empty = np.zeros((0,) + tuple(rex.labels.shape))
    labels, plaq, tlabels, emdH = rex.history_arrays() if rex.labels_history else (empty,) * 4
    sd = rex.defect_action_array() if rex.defect_action_history else empty

    def write(tmp):
        with open(tmp, "wb") as handle:
            np.savez(handle, labels=labels, plaquette=plaq, trajectory_labels=tlabels, exp_minus_dH=emdH,
                     defect_action=sd, ladder=rex.ladder.numpy(), parameter_ladder=rex.ladder.numpy(),
                     swap_every=np.asarray(1), therm=np.asarray(int(therm)))
    _atomic_write(path, write)


def load_history(path, rex, n_steps):
    with np.load(path) as h:
        if h["labels"].shape[0] < n_steps:
            raise SystemExit(f"--resume: {path} holds {h['labels'].shape[0]} steps, the checkpoint {n_steps}")
        names = [("labels", "labels_history"), ("plaquette", "plaquette_history"),
                 ("trajectory_labels", "trajectory_labels_history"), ("exp_minus_dH", "exp_minus_dH_history")]
        if "defect_action" in h.files:  # absent in files written before the optimal-ladder outputs
            names.append(("defect_action", "defect_action_history"))
        for name, attr in names:
            setattr(rex, attr, [torch.from_numpy(np.array(x)) for x in h[name][:n_steps]])
        if "defect_action" not in h.files:  # unknown for the old steps: NaN, so the shapes stay aligned
            rex.defect_action_history = [torch.full(tuple(rex.labels.shape), float("nan"), dtype=torch.float64)
                                         for _ in range(n_steps)]


def summarize(rex, csv_path, therm):
    """Summary of the steps after --therm (history in memory, cost columns from the CSV)."""
    summary = {"steps_total": rex.n_steps_done, "therm": int(therm)}
    n_meas = rex.n_steps_done - therm
    summary["steps_measured"] = int(max(n_meas, 0))
    if n_meas <= 0:
        return summary
    summary["history"] = rex.summarize(start=therm)
    if "defect_action" in summary["history"]:
        summary["history"]["defect_action"]["parameter_ladder"] = rex.ladder.tolist()
    cols = read_observables(csv_path)
    keep = cols["therm"] == 0
    cols = {k: v[keep] for k, v in cols.items()}
    n_r = rex.n_rungs
    acc = [float(np.nansum(cols["swap_accepted"][cols["rung"] == k]) /
                 max(np.nansum(cols["swap_proposed"][cols["rung"] == k]), 1)) for k in range(n_r - 1)]
    per_step = cols["rung"] == 0
    summary["swap"] = swap_meta(rex.ladder.tolist(),
                                [np.nansum(cols["swap_accepted"][cols["rung"] == k]) for k in range(n_r - 1)],
                                [np.nansum(cols["swap_proposed"][cols["rung"] == k]) for k in range(n_r - 1)])
    summary.update({
        "swap_acceptance": acc,
        "plaquette_per_rung": [float(cols["plaquette"][cols["rung"] == k].mean()) for k in range(n_r)],
        "exp_minus_dH_per_rung": [float(cols["exp_mdH"][cols["rung"] == k].mean()) for k in range(n_r)],
        "hmc_acceptance": float(cols["hmc_acceptance"][per_step].mean()),
        "solves_per_step": float(cols["solves"][per_step].mean()),
        "cg_iter_per_step": float(cols["cg_iter"][per_step].mean()),
        "seconds_per_step": float(cols["wall_s"][per_step].mean()),
        "configurations_per_step": rex.B,
    })
    h = summary["history"]
    rt = h.get("round_trip_steps")
    if rt is not None and np.isfinite(rt) and rt > 0:
        summary["seconds_per_round_trip_per_walker"] = float(rt * summary["seconds_per_step"])
        summary["solves_per_round_trip"] = float(rt * summary["solves_per_step"])
    return summary


def main(argv=None):
    args = parse_args(argv)
    device = choose_device(args.device)
    geometry = Geometry(args.D, args.T, args.L, args.N)
    out = Path(args.out) if args.out else ROOT / "output" / "rex_fermions" / default_name(args)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path, hist_path = out / "observables.csv", out / "run_meta.json", out / "replica_history.npz"
    state_prefix, rex_state_path = out / STATE_NAME, out / "rex_state.pt"

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    swap_generator = torch.Generator()
    swap_generator.manual_seed(int(args.seed) + 1)
    torch.manual_seed(int(args.seed))

    fpar = args.kappa if args.fermions == "wilson" else args.mass
    n_rungs = len(args.ladder_values)
    B = n_rungs * args.n_chains
    defect_mask = build_defect_mask(args, geometry).to(device)
    key = clean_json(physics_key(args))

    if args.resume:
        if not state_prefix.with_suffix(".json").exists() or not rex_state_path.exists():
            raise SystemExit(f"--resume: no checkpoint in {out}")
        U, meta, rng_state = load_state(state_prefix, device=device)
        if meta["extra"].get("physics") != key:
            raise SystemExit(f"--resume: checkpoint parameters {meta['extra'].get('physics')} differ from {key}")
        if meta["rng"]["device"].split(":")[0] != device.type:
            raise SystemExit(f"--resume: checkpoint RNG is for {meta['rng']['device']}, run on {device}")
        restore_generator(generator, rng_state)
        rex_state = torch.load(rex_state_path, map_location="cpu")
        if int(rex_state["n_steps_done"]) != int(meta["trajectory"]):
            raise SystemExit("--resume: checkpoint.* and rex_state.pt are from different steps")
        swap_generator.set_state(rex_state["swap_rng"])
        source = meta["extra"].get("source")
    else:
        previous = [p for p in (csv_path, meta_path, hist_path, rex_state_path, state_prefix.with_suffix(".json"),
                                state_prefix.with_suffix(".pt")) if p.exists()]
        if previous and not args.overwrite:
            raise SystemExit(f"{out} holds a previous run ({[p.name for p in previous]}): use --resume or --overwrite")
        for p in previous:
            p.unlink()
        U, source = start_links(args, geometry, device, B, args.beta, fpar, 1.0)

    akw = action_kwargs(args)
    rex = FermionREX(geometry, args.beta, akw.pop("kappa"), defect_mask, args.ladder_values, n_chains=args.n_chains,
                     tau=args.tau, n_steps=args.n_steps, integrator=args.integrator_levels,
                     n_gauge_substeps=args.n_gauge_substeps, swap_parity=args.swap_parity,
                     periodic_translation=args.periodic_translation, U=U, swap_generator=swap_generator,
                     **akw, **fermion_kwargs(args))
    writer = ObservableWriter(csv_path, OBS_FIELDS, trajectory_field="step")
    if args.resume:
        rex.labels = rex_state["labels"].clone()
        rex.n_steps_done = int(rex_state["n_steps_done"])
        rex.link_accepted = rex_state["link_accepted"].clone()
        rex.link_proposed = rex_state["link_proposed"].clone()
        rex.target.set_parameter(rex.parameters())
        load_history(hist_path, rex, rex.n_steps_done)
        writer.truncate(rex.n_steps_done)
        _log(args, f"resumed {out} at step {rex.n_steps_done}")

    total = args.therm + args.steps
    params_json = {k: v for k, v in vars(args).items()}
    info = run_info("main/main_rex_fermions.py", argv, device)
    info.update(lattice=list(geometry.lattice), volume=geometry.volume, start_source=source, B=B,
                fermions=args.fermions, mass=args.mass,
                n_rungs=n_rungs, defect_links=int(defect_mask[1].sum().item()), obs_fields=list(OBS_FIELDS))
    fermion = rex.qcd.fermion

    def checkpoint(final=False):
        save_history(hist_path, rex, args.therm)
        torch.save({"labels": rex.labels.clone(), "n_steps_done": rex.n_steps_done,
                    "link_accepted": rex.link_accepted.clone(), "link_proposed": rex.link_proposed.clone(),
                    "swap_rng": swap_generator.get_state()}, rex_state_path)
        save_state(state_prefix, rex.U, geometry=geometry, beta=args.beta,
                   trajectory=rex.n_steps_done, time_sign=args.time_sign, generator=generator,
                   **state_couplings(args, fpar),
                   extra={"physics": key, "source": source, "seed": args.seed, "final": bool(final)})
        summary = summarize(rex, csv_path, args.therm)
        meta = {"algorithm": "rex_fermions", "parameters": params_json, "run": info, "steps_done": rex.n_steps_done,
                "finished": bool(final and rex.n_steps_done >= total), "summary": summary}
        if "swap" in summary:  # main_pt_obc.py layout, read by main/optimal_ladder.py --metric acceptance
            meta["swap"] = summary["swap"]
        if "history" in summary:
            meta["replica_history"] = dict(summary["history"], path=str(hist_path))
        atomic_json_save(clean_json(meta), meta_path)
        return summary

    fdesc = f"N_f=2 Wilson kappa={args.kappa}" if args.fermions == "wilson" else f"staggered mass={args.mass}"
    _log(args, f"REX {fdesc} ladder {args.ladder_values} x {args.n_chains} chains, lattice {geometry.lattice} "
               f"SU({args.N}) beta={args.beta}, defect {int(defect_mask[1].sum())} links, "
               f"device={device} -> {out}")
    while rex.n_steps_done < total:
        step = rex.n_steps_done
        s0, c0 = fermion.stats["solves"], fermion.stats["cg_iterations"]
        _sync(device)
        t0 = time.perf_counter()
        st = rex.step(generator)
        _sync(device)
        wall = time.perf_counter() - t0
        solves, cg_iter = fermion.stats["solves"] - s0, fermion.stats["cg_iterations"] - c0
        rows = [{"step": step, "therm": int(step < args.therm), "rung": k, "bc": args.ladder_values[k],
                 "plaquette": st["plaquette_per_rung"][k], "exp_mdH": st["exp_minus_dH_per_rung"][k],
                 "swap_accepted": st["link_accepted"][k] if k < n_rungs - 1 else None,
                 "swap_proposed": st["link_proposed"][k] if k < n_rungs - 1 else None,
                 "hmc_acceptance": st["hmc_acceptance"], "solves": solves, "cg_iter": cg_iter, "wall_s": wall}
                for k in range(n_rungs)]
        writer.write_rows(rows)
        if step % max(1, total // 20) == 0:
            _log(args, f"step {step:5d} plaq/rung {[round(x, 5) for x in st['plaquette_per_rung']]} "
                       f"hmc acc {st['hmc_acceptance']:.2f} swaps {st['link_accepted']}/{st['link_proposed']} "
                       f"solves {solves} {wall:.2f} s")
        if args.checkpoint_every and rex.n_steps_done % args.checkpoint_every == 0 and rex.n_steps_done < total:
            checkpoint()

    summary = checkpoint(final=True)
    h = summary.get("history", {})
    _log(args, f"done: {summary['steps_measured']} measured steps, round trips {h.get('round_trips')}, "
               f"round-trip steps {h.get('round_trip_steps')}, swap acceptance {summary.get('swap_acceptance')}, "
               f"{summary.get('seconds_per_step')} s/step, {summary.get('solves_per_step')} solves/step")
    return {"out": str(out), "summary": summary}


if __name__ == "__main__":
    main()
