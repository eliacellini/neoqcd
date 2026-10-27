"""Summarize O-REX / NEO-REX validation runs: acceptance, work, cost and the Crooks check.

Reads the run directories written by job/debug_neorex_eval.sh (reports/neorex_eval_<job>/<method>_K<K>)
and prints one row per run:

- swap acceptance (total and per ladder pair) and mean swap work;
- seconds per REX step and work x seconds, the cost-normalized figure: for the purely stochastic
  protocol the work falls as 1/K while the cost grows as K, so a smaller product beats it at
  equal cost;
- with --save-link-work-values, <exp(-W)> per pair, which detailed balance fixes to 1, and its
  largest deviation in standard errors (errors ignore the correlation of successive swaps).

Runs stopped by the time limit have no final statistics in run_meta.json: their acceptance comes
from the last row of pt_metrics.csv and is marked "partial".

    python main/summarize_neorex_validation.py reports/neorex_eval_* [--csv out.csv]
"""

import argparse
import csv
import json
import math
from pathlib import Path


def _method(args):
    if args["algorithm"] == "orex":
        return "O-REX"
    layer = args.get("nf_layer_type") or "smearing"
    if layer == "smearing":
        return f"smearing/{args.get('hyper_smearing_mode', 'per_link')}"
    if layer == "residual":
        return f"residual/{args.get('residual_mode', 'shared')}"
    return f"cnf/{args.get('cnf_integration_steps', '?')}cf4"


def _crooks(path):
    """Per pair: (<W>, <exp(-W)>, its standard error), or None without saved works."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not path.exists():
        return None
    values = np.load(path)
    pairs = sorted((key for key in values.files if key.startswith("link_") and key[5:].isdigit()),
                   key=lambda key: int(key[5:]))
    rows = []
    for key in pairs:
        w = values[key].reshape(-1)
        w = w[np.isfinite(w)]
        e = np.exp(-w)
        rows.append((float(w.mean()), float(e.mean()), float(e.std(ddof=1) / math.sqrt(len(w)))))
    return rows


def _partial(run_dir):
    """Acceptance and progress of a run without run_meta.json, from the last pt_metrics.csv row."""
    path = run_dir / "pt_metrics.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return None
    last = rows[-1]
    pairs = json.loads(last["link_acceptance_cumulative_json"])
    return {"steps": int(last["step"]), "seconds": float(last["measurement_elapsed_seconds"]), "pairs": pairs}


def summarize(run_dir):
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    args = meta.get("args", {})
    if "swap" in meta:
        steps = int(args["steps"])
        row = {
            "acceptance": float(meta["swap"]["acceptance_mean"]),
            "pairs": [float(a) for a in meta["swap"]["link_acceptance"]],
            "work": float(meta["work"]["mean"]),
            "work_sem": float(meta["work"]["sem"]),
            "s_per_step": float(meta["timing"]["measurement_seconds"]) / steps,
            "status": "done",
        }
    else:
        # Stopped by the time limit: run_meta.json has the arguments but no final statistics.
        partial = _partial(run_dir)
        if partial is None:
            return None
        steps = partial["steps"]
        row = {"acceptance": sum(partial["pairs"]) / len(partial["pairs"]), "pairs": partial["pairs"],
               "work": float("nan"), "work_sem": float("nan"),
               "s_per_step": partial["seconds"] / max(steps, 1), "status": "partial"}
    # job/debug_neorex_eval.sh names every run directory <method>_K<K>; partial runs have no arguments.
    name, K = run_dir.name.rsplit("_K", 1)
    row.update(run=str(run_dir), method=_method(args) if args else name, K=int(K), batch=args.get("batch_size"),
               steps=steps)
    row["work_x_s"] = row["work"] * row["s_per_step"]
    crooks = _crooks(run_dir / "link_work_values.npz")
    if crooks:
        row["exp_minus_w"] = [round(e, 4) for _, e, _ in crooks]
        row["crooks_max_sigma"] = max(abs(e - 1.0) / s for _, e, s in crooks if s > 0)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path, help="neorex_eval_<job> directories")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = []
    for report in args.reports:
        for run_dir in sorted(p for p in report.iterdir() if p.is_dir()):
            row = summarize(run_dir)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: (r["batch"] or 0, r["method"] != "O-REX", r["method"], r["K"]))
    header = f"{'method':20s} {'K':>3s} {'batch':>5s} {'steps':>5s} {'acc':>6s} {'work':>14s} {'s/step':>7s} " \
             f"{'work*s':>7s} {'crooks':>7s}  status  pairs"
    print(header)
    for r in rows:
        work = f"{r['work']:.3f}+-{r['work_sem']:.3f}" if math.isfinite(r["work"]) else "-"
        crooks = f"{r['crooks_max_sigma']:.1f}sig" if "crooks_max_sigma" in r else "-"
        print(f"{r['method']:20s} {r['K']:3d} {str(r['batch']):>5s} {r['steps']:5d} {r['acceptance']:6.3f} {work:>14s} "
              f"{r['s_per_step']:7.2f} {r['work_x_s']:7.3f} {crooks:>7s}  {r['status']:7s} "
              + " ".join(f"{a:.2f}" for a in r["pairs"]))
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for r in rows for key in r}))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
