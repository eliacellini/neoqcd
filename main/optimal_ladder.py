"""Optimal replica ladder from measured runs (appendix "Optimal ladder" of the O-REX paper).

Two metrics, both turned into the thermodynamic length Lambda(c) and inverted at
Lambda(c_r) = Lambda r / (n_rep - 1) (see neoqcd.ladder):

- variance (default, the paper's): Var_c(S_D) of the defect action S_D = dS/dc on every rung, from the
  `defect_action` array of replica_history.npz (first 20% of the steps dropped, as in the paper).
  Several runs, e.g. with shifted ladders, are pooled into one Lambda(c). The ladder equalizes the
  acceptance of standard REX, erfc[Lambda / (2 (n_rep - 1))] on every link, and is reused unchanged
  for O-REX and NEO-REX, as in the paper.
- acceptance: the length 2 erfc^-1(A) of each link of one run (O-REX, NEO-REX, any K); the ladder
  equalizes the acceptance of that algorithm at that protocol length.

Usage:
  python main/optimal_ladder.py reports/neorex_eval_<id>/orex_K2 [more runs] --n-rep 8 --output ladder.json
  python main/optimal_ladder.py reports/neorex_eval_<id>/smearing_K4 --metric acceptance --output ladder.json
The JSON file goes to main_pt_obc.py / main_cnf_rex.py as --parameter-ladder ladder.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neoqcd.ladder import (  # noqa: E402
    cumulative_length,
    link_lengths_from_acceptance,
    lengths_on_ladder,
    optimal_ladder,
    predicted_acceptance,
    rex_acceptance,
    thermodynamic_length,
)
from neoqcd.replica_history import rung_moments  # noqa: E402


def _run_dir(path: Path) -> Path:
    """A run directory (with run_meta.json or replica_history.npz), or a job directory holding exactly one."""
    if (path / "run_meta.json").exists() or (path / "replica_history.npz").exists():
        return path
    runs = [p for p in sorted(path.iterdir()) if p.is_dir() and
            ((p / "run_meta.json").exists() or (p / "replica_history.npz").exists())]
    if len(runs) != 1:
        raise SystemExit(f"{path}: expected one run directory, found {[p.name for p in runs]}")
    return runs[0]


def variance_points(run: Path, discard: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(couplings, Var_c(S_D), jackknife errors) on the rungs of one run."""
    path = run / "replica_history.npz"
    if not path.exists():
        raise SystemExit(f"{run}: no replica_history.npz")
    data = np.load(path)
    if "defect_action" not in data.files:
        raise SystemExit(f"{run}: replica_history.npz has no defect_action (run made before the optimal-ladder code)")
    ladder = np.asarray(data["parameter_ladder"], dtype=np.float64)
    moments = rung_moments(data["labels"].astype(np.int64), data["defect_action"], ladder.size, discard)
    return ladder, np.asarray(moments["variance"]), np.asarray(moments["variance_error"])


def pool(points: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge the rungs of several runs; equal couplings are averaged with weights 1 / error^2."""
    c = np.concatenate([p[0] for p in points])
    var = np.concatenate([p[1] for p in points])
    err = np.concatenate([p[2] for p in points])
    keys = np.round(c, 12)
    out_c, out_var, out_err = [], [], []
    for key in np.unique(keys):
        sel = keys == key
        if np.count_nonzero(sel) == 1 or not np.all(np.isfinite(err[sel])):
            # one point, or errors unknown: plain mean
            out_c.append(float(c[sel][0]))
            out_var.append(float(np.mean(var[sel])))
            out_err.append(float(np.sqrt(np.sum(err[sel] ** 2)) / np.count_nonzero(sel)))
            continue
        w = 1.0 / np.maximum(err[sel], 1e-300) ** 2
        out_c.append(float(c[sel][0]))
        out_var.append(float(np.sum(w * var[sel]) / np.sum(w)))
        out_err.append(float(1.0 / math.sqrt(np.sum(w))))
    return np.asarray(out_c), np.asarray(out_var), np.asarray(out_err)


def length_error(c: np.ndarray, var: np.ndarray, err: np.ndarray) -> float:
    """Error of Lambda from the variance errors (rungs independent): trapezoid weights x dVar / (2 sqrt Var)."""
    weights = np.zeros_like(c)
    dc = np.diff(c)
    weights[:-1] += 0.5 * dc
    weights[1:] += 0.5 * dc
    return float(math.sqrt(np.sum((weights * err / (2.0 * np.sqrt(var))) ** 2)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories (<method>_K<K>) or neorex_eval_<job>")
    parser.add_argument("--metric", choices=("variance", "acceptance"), default="variance")
    parser.add_argument("--n-rep", type=int, default=None, help="replicas of the new ladder (default: as the input)")
    parser.add_argument("--discard", type=float, default=0.2, help="fraction of steps dropped (variance metric)")
    parser.add_argument("--output", type=Path, default=None, help="JSON file for --parameter-ladder")
    args = parser.parse_args(argv)
    runs = [_run_dir(path) for path in args.runs]

    if args.metric == "variance":
        per_run = [variance_points(run, args.discard) for run in runs]
        c, var, err = pool(per_run)
        lam = thermodynamic_length(c, var)
        total_error = length_error(c, var, err)
        n_input = per_run[0][0].size
        print(f"metric=variance  runs={len(runs)}  couplings={c.size}  discard={args.discard}")
        print(f"{'c':>10s} {'Var_c(S_D)':>14s} {'err':>10s} {'Lambda(c)/Lambda':>17s}")
        for ci, vi, ei, li in zip(c, var, err, lam):
            print(f"{ci:10.6f} {vi:14.6g} {ei:10.3g} {li / lam[-1]:17.4f}")
    else:
        if len(runs) != 1:
            raise SystemExit("--metric acceptance takes exactly one run")
        meta = json.loads((runs[0] / "run_meta.json").read_text())
        swap = meta["swap"]
        left, right = np.asarray(swap["link_parameter_left"]), np.asarray(swap["link_parameter_right"])
        if not np.allclose(left[1:], right[:-1], rtol=0, atol=1e-12):
            raise SystemExit("the links of the run do not form one ladder")
        c = np.concatenate((left, right[-1:]))
        acceptance = np.asarray(swap["link_acceptance"], dtype=np.float64)
        lam = cumulative_length(link_lengths_from_acceptance(acceptance))
        total_error = float("nan")
        n_input = c.size
        print(f"metric=acceptance  run={runs[0]}  algorithm={meta.get('algorithm')}")
        print(f"{'link':>10s} {'acceptance':>11s} {'length':>8s}")
        for k, a in enumerate(acceptance):
            print(f"{c[k]:.4f}-{c[k + 1]:.4f} {a:11.4f} {lam[k + 1] - lam[k]:8.4f}")

    n_rep = int(args.n_rep or n_input)
    ladder = optimal_ladder(c, lam, n_rep)
    total = float(lam[-1])
    target = predicted_acceptance(total, n_rep)
    print(f"Lambda={total:.4f}" + (f"+-{total_error:.4f}" if math.isfinite(total_error) else ""))
    print(f"optimal_ladder_n_rep={n_rep}  ladder_json={json.dumps([round(float(v), 10) for v in ladder])}")
    if args.metric == "variance":
        print(f"predicted_rex_acceptance_per_link={target:.4f}")
        if n_input == n_rep:
            linear = np.linspace(c[0], c[-1], n_rep)
            rex_linear = rex_acceptance(lengths_on_ladder(c, lam, linear))
            print(f"predicted_rex_acceptance_linear_ladder_json={json.dumps([round(float(a), 4) for a in rex_linear])}")
    else:
        print(f"predicted_acceptance_per_link={target:.4f} (Gaussian work; measured mean {acceptance.mean():.4f})")

    if args.output is not None:
        payload = {
            "ladder": [float(v) for v in ladder],
            "n_rep": n_rep,
            "metric": args.metric,
            "Lambda": total,
            "Lambda_error": total_error if math.isfinite(total_error) else None,
            "predicted_acceptance_per_link": target,
            "couplings": [float(v) for v in c],
            "Lambda_c": [float(v) for v in lam],
            "sources": [str(run) for run in runs],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"written {args.output}")


if __name__ == "__main__":
    main()
