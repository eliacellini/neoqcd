"""REX (parallel tempering on the gauge-only defect) with N_f = 2 Wilson fermions (fermions_plan.md,
phase 5), single process: `neoqcd.fermions.rex.FermionREX`.

--n-chains independent ladders of n_rungs replicas (bc values --ladder 0,0.5,1, or --n-rep values
uniform in [--bc-min, 1]) run as one batch of B = n_rungs * n_chains configurations. One REX step =
one fermionic HMC trajectory of every replica at its bc + swap round(s) between neighbouring rungs
(--swap-parity alternating: DEO, even pairs at even steps; even_odd: both rounds every step), with
the solve-free gauge-only swap dS of `rex.swap_action_difference`; optional translation of the
bc = 1 replicas (--periodic-translation). The defect is that of main_snf_fermions.py
(--defect-shape cube|line, --defect-size, --defect-time-slice, --defect-space-slice).

Output directory (--out, default output/rex_fermions/<auto name>):
    observables.csv     one row per (step, rung) (OBS_FIELDS): rung-averaged plaquette (mean over
                        chains of the configuration carrying the rung after the step), <exp(-dH)>
                        of the trajectories run at that rung, swap attempts/acceptances of the
                        pair (rung, rung + 1) in this step, cost of the step;
    replica_history.npz labels (steps, n_rungs, n_chains): rung carried by each slot after the
                        swaps; plaquette (same shape): plaquette of each slot's configuration;
                        trajectory_labels, exp_minus_dH; ladder; therm (number of leading
                        thermalisation steps). Layout of `neoqcd.replica_history` (world_size ->
                        slot, B -> chain);
    run_meta.json       parameters + summary of the steps after --therm: `FermionREX.summarize()`
                        (round trips, mean round-trip time, flux, plaquette tau_int on the bottom
                        and top rungs), swap acceptance per pair, plaquette per rung, HMC
                        acceptance, solves / CG iterations / seconds per step;
    checkpoint.pt/.json links + run generator (`neoqcd.fermions.io.save_state`);
    rex_state.pt        labels, step count, swap counters, swap-generator state.

Two generators: the run generator (on the device: pseudofermions, momenta, Metropolis, translation)
and a CPU generator for the swap uniforms (seed + 1), so that the driver also runs on CUDA
(`FermionREX.swap_round` draws its uniforms on the CPU). The driver's `DeviceREX` subclass only
routes the swaps to that generator and keeps the per-step history on the CPU.

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
    add_physics_args,
    build_defect_mask,
    clean_json,
    defect_key,
    fermion_kwargs,
    run_info,
    start_links,
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
    p.add_argument("--ladder", type=str, default=None, help="Comma-separated increasing bc values, top = 1.")
    p.add_argument("--n-rep", type=int, default=None, help="Uniform ladder of n values in [--bc-min, 1].")
    p.add_argument("--bc-min", type=float, default=0.0)
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
        raise SystemExit("give exactly one of --ladder and --n-rep")
    if args.ladder is not None:
        try:
            args.ladder_values = [float(x) for x in args.ladder.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"--ladder: cannot parse {args.ladder!r}")
    else:
        if args.n_rep < 2:
            raise SystemExit("--n-rep must be >= 2")
        args.ladder_values = [float(v) for v in np.linspace(args.bc_min, 1.0, args.n_rep)]
    lad = args.ladder_values
    if len(lad) < 2 or any(b <= a for a, b in zip(lad, lad[1:])):
        raise SystemExit(f"the ladder must be strictly increasing with >= 2 rungs, got {lad}")
    if args.periodic_translation and lad[-1] != 1.0:
        raise SystemExit("--periodic-translation needs bc = 1 on the top rung")
    if args.n_chains < 1 or args.steps < 0 or args.checkpoint_every < 0:
        raise SystemExit("--n-chains >= 1, --steps >= 0, --checkpoint-every >= 0")
    return args


def default_name(args):
    lat = f"{args.T}x{args.L}p{args.D - 1}"
    return (f"rex_r{len(args.ladder_values)}_c{args.n_chains}_d{args.defect_shape}{args.defect_size}"
            f"_b{args.beta:g}_k{args.kappa:g}_D{args.D}_{lat}_SU{args.N}_s{args.seed}")


def physics_key(args):
    return {
        "geometry": {"D": args.D, "T": args.T, "L": args.L, "N": args.N},
        "beta": float(args.beta), "kappa": float(args.kappa), "time_sign": float(args.time_sign),
        "ladder": list(args.ladder_values), "n_chains": args.n_chains, "defect": defect_key(args),
        "tau": float(args.tau), "n_steps": args.n_steps, "n_gauge_substeps": args.n_gauge_substeps,
        "integrator": list(args.integrator_levels), "swap_parity": args.swap_parity,
        "periodic_translation": bool(args.periodic_translation),
    }


class DeviceREX(FermionREX):
    """`FermionREX` with the swap uniforms from a CPU generator and the history kept on the CPU
    (the base class is unchanged on the CPU apart from the separate swap generator)."""

    def __init__(self, *args, swap_generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.swap_generator = swap_generator

    def swap_round(self, table, parity, generator=None):
        return super().swap_round(table.cpu(), parity, self.swap_generator)

    def _per_rung(self, values, labels):
        return super()._per_rung(values.cpu(), labels)

    def step(self, generator=None):
        stats = super().step(generator)
        self.plaquette_history[-1] = self.plaquette_history[-1].cpu()
        self.exp_minus_dH_history[-1] = self.exp_minus_dH_history[-1].cpu()
        return stats


def save_history(path, rex, therm):
    labels, plaq, tlabels, emdH = rex.history_arrays() if rex.labels_history else (np.zeros((0,) + tuple(rex.labels.shape)),) * 4

    def write(tmp):
        with open(tmp, "wb") as handle:
            np.savez(handle, labels=labels, plaquette=plaq, trajectory_labels=tlabels, exp_minus_dH=emdH,
                     ladder=rex.ladder.numpy(), therm=np.asarray(int(therm)))
    _atomic_write(path, write)


def load_history(path, rex, n_steps):
    with np.load(path) as h:
        if h["labels"].shape[0] < n_steps:
            raise SystemExit(f"--resume: {path} holds {h['labels'].shape[0]} steps, the checkpoint {n_steps}")
        for name, attr in (("labels", "labels_history"), ("plaquette", "plaquette_history"),
                           ("trajectory_labels", "trajectory_labels_history"),
                           ("exp_minus_dH", "exp_minus_dH_history")):
            setattr(rex, attr, [torch.from_numpy(np.array(x)) for x in h[name][:n_steps]])


def summarize(rex, csv_path, therm):
    """Summary of the steps after --therm (history in memory, cost columns from the CSV)."""
    summary = {"steps_total": rex.n_steps_done, "therm": int(therm)}
    n_meas = rex.n_steps_done - therm
    summary["steps_measured"] = int(max(n_meas, 0))
    if n_meas <= 0:
        return summary
    summary["history"] = rex.summarize(start=therm)
    cols = read_observables(csv_path)
    keep = cols["therm"] == 0
    cols = {k: v[keep] for k, v in cols.items()}
    n_r = rex.n_rungs
    acc = [float(np.nansum(cols["swap_accepted"][cols["rung"] == k]) /
                 max(np.nansum(cols["swap_proposed"][cols["rung"] == k]), 1)) for k in range(n_r - 1)]
    per_step = cols["rung"] == 0
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
        U, source = start_links(args, geometry, device, B, args.beta, args.kappa, 1.0)

    rex = DeviceREX(geometry, args.beta, args.kappa, defect_mask, args.ladder_values, n_chains=args.n_chains,
                    tau=args.tau, n_steps=args.n_steps, integrator=args.integrator_levels,
                    n_gauge_substeps=args.n_gauge_substeps, swap_parity=args.swap_parity,
                    periodic_translation=args.periodic_translation, U=U, swap_generator=swap_generator,
                    **fermion_kwargs(args))
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
                n_rungs=n_rungs, defect_links=int(defect_mask[1].sum().item()), obs_fields=list(OBS_FIELDS))
    fermion = rex.qcd.fermion

    def checkpoint(final=False):
        save_history(hist_path, rex, args.therm)
        torch.save({"labels": rex.labels.clone(), "n_steps_done": rex.n_steps_done,
                    "link_accepted": rex.link_accepted.clone(), "link_proposed": rex.link_proposed.clone(),
                    "swap_rng": swap_generator.get_state()}, rex_state_path)
        save_state(state_prefix, rex.U, geometry=geometry, beta=args.beta, fermions="wilson",
                   trajectory=rex.n_steps_done, time_sign=args.time_sign, kappa=args.kappa, generator=generator,
                   extra={"physics": key, "source": source, "seed": args.seed, "final": bool(final)})
        summary = summarize(rex, csv_path, args.therm)
        atomic_json_save(clean_json({"parameters": params_json, "run": info, "steps_done": rex.n_steps_done,
                                     "finished": bool(final and rex.n_steps_done >= total), "summary": summary}),
                         meta_path)
        return summary

    _log(args, f"REX N_f=2 Wilson ladder {args.ladder_values} x {args.n_chains} chains, lattice {geometry.lattice} "
               f"SU({args.N}) beta={args.beta} kappa={args.kappa}, defect {int(defect_mask[1].sum())} links, "
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
