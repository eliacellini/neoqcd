#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


FLOAT_RE = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


PATTERNS = {
    "backend": re.compile(
        r"backend=(?P<backend>\S+).*world_size=(?P<world_size>\d+).*"
        r"algorithm=(?P<algorithm>\S+).*output_dir=(?P<main_output_dir>\S+)"
    ),
    "output_dir": re.compile(r"\bOUTPUT_DIR=(?P<output_dir>\S+)"),
    "study_name": re.compile(r"\bSTUDY_NAME=(?P<study_name>\S+)"),
    "run_name": re.compile(r"\bRUN_NAME=(?P<run_name>\S+)"),
    "wandb_run_name": re.compile(r"\bWANDB_RUN_NAME=(?P<wandb_run_name>\S+)"),
    "nreplicas": re.compile(r"\bNREPLICAS=(?P<nreplicas>\d+)"),
    "nsteps": re.compile(r"\bOREX_PROTOCOL_STEPS=(?P<nsteps>\d+)"),
    "swap": re.compile(
        rf"final_swap_acceptance_mean=(?P<swap_acceptance>{FLOAT_RE})\s+"
        r"total_accepted=(?P<total_accepted>\d+)\s+"
        r"total_proposed=(?P<total_proposed>\d+)\s+"
        r"swap_attempts=(?P<swap_attempts>\d+)"
    ),
    "work": re.compile(
        rf"final_work_mean=(?P<work_mean>{FLOAT_RE})\s+"
        rf"final_work_sem=(?P<work_sem>{FLOAT_RE})\s+"
        r"work_count=(?P<work_count>\d+)"
    ),
    "timing": re.compile(
        rf"timing_thermalization_s=(?P<thermalization_s>{FLOAT_RE})\s+"
        rf"timing_measurement_s=(?P<measurement_s>{FLOAT_RE})\s+"
        rf"timing_total_s=(?P<total_s>{FLOAT_RE})"
    ),
}


INT_FIELDS = {
    "world_size",
    "nreplicas",
    "nsteps",
    "total_accepted",
    "total_proposed",
    "swap_attempts",
    "work_count",
}

FLOAT_FIELDS = {
    "swap_acceptance",
    "work_mean",
    "work_sem",
    "thermalization_s",
    "measurement_s",
    "total_s",
}

CSV_FIELDS = [
    "log_file",
    "study_name",
    "run_name",
    "output_dir",
    "world_size",
    "nreplicas",
    "nsteps",
    "collapse_x",
    "swap_acceptance",
    "total_accepted",
    "total_proposed",
    "swap_attempts",
    "work_mean",
    "work_sem",
    "work_count",
    "thermalization_s",
    "measurement_s",
    "total_s",
    "backend",
    "algorithm",
    "wandb_run_name",
]


def parse_float(value):
    value = str(value).strip().lower()
    if value == "nan":
        return math.nan
    if value in {"inf", "+inf"}:
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def split_grep_line(line: str):
    line = line.rstrip("\n")
    if ":" not in line:
        return "<stdin>", line
    left, right = line.split(":", 1)
    return left, right


def infer_nsteps(row):
    if row.get("nsteps") is not None:
        return
    for key in ("run_name", "output_dir", "main_output_dir", "wandb_run_name"):
        value = str(row.get(key) or "")
        match = re.search(r"nsteps(\d+)", value)
        if match:
            row["nsteps"] = int(match.group(1))
            return


def infer_study_name(row):
    if row.get("study_name"):
        return
    output_dir = str(row.get("output_dir") or row.get("main_output_dir") or "")
    parts = Path(output_dir).parts
    if len(parts) >= 2 and re.match(r"nsteps\d+$", parts[-1]):
        row["study_name"] = parts[-2]


def coerce_row(row):
    if row.get("main_output_dir") and not row.get("output_dir"):
        row["output_dir"] = row["main_output_dir"]
    infer_nsteps(row)
    infer_study_name(row)
    if row.get("nreplicas") is None and row.get("world_size") is not None:
        row["nreplicas"] = row["world_size"]
    for key in INT_FIELDS:
        if key in row and row[key] not in (None, ""):
            row[key] = int(row[key])
    for key in FLOAT_FIELDS:
        if key in row and row[key] not in (None, ""):
            row[key] = parse_float(row[key])
    return row


def parse_recovered_log(path: Path):
    rows = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            log_file, payload = split_grep_line(line)
            row = rows[log_file]
            row["log_file"] = log_file
            for pattern in PATTERNS.values():
                match = pattern.search(payload)
                if match:
                    row.update(match.groupdict())
    return [coerce_row(row) for row in rows.values()]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def add_collapse_coordinate(rows, alpha: float, defect_volume: float):
    sqrt_v = math.sqrt(defect_volume)
    for row in rows:
        nreplicas = row.get("nreplicas") or row.get("world_size")
        nsteps = row.get("nsteps")
        if nreplicas is None or nsteps is None:
            continue
        row["collapse_x"] = (
            float(nreplicas) / sqrt_v
            + float(alpha) * float(nsteps) * float(nreplicas) * float(nreplicas) / float(defect_volume)
        )


def grouped_points(rows, y_key):
    groups = defaultdict(list)
    for row in rows:
        x = row.get("nsteps")
        y = row.get(y_key)
        if x is None or not finite(y):
            continue
        label = f"r{row.get('nreplicas') or row.get('world_size') or 'unknown'}"
        groups[label].append(row)
    for label in groups:
        groups[label].sort(key=lambda r: (int(r.get("nsteps", -1)), str(r.get("log_file", ""))))
    return groups


def plot_metric(rows, y_key, ylabel, out_base: Path, yerr_key=None):
    groups = grouped_points(rows, y_key)
    if not groups:
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, items in sorted(groups.items()):
        xs = [int(r["nsteps"]) for r in items]
        ys = [float(r[y_key]) for r in items]
        if yerr_key:
            yerr = [
                float(r[yerr_key]) if finite(r.get(yerr_key)) else 0.0
                for r in items
            ]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label)
        else:
            ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("O-REX protocol steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(title="replicas")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return True


def plot_collapse(rows, y_key, ylabel, out_base: Path, alpha: float, defect_volume: float, yerr_key=None):
    groups = defaultdict(list)
    for row in rows:
        x = row.get("collapse_x")
        y = row.get(y_key)
        if not finite(x) or not finite(y):
            continue
        label = f"r{row.get('nreplicas') or row.get('world_size') or 'unknown'}"
        groups[label].append(row)
    if not groups:
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, items in sorted(groups.items()):
        items = sorted(items, key=lambda r: (float(r["collapse_x"]), int(r.get("nsteps", -1))))
        xs = [float(r["collapse_x"]) for r in items]
        ys = [float(r[y_key]) for r in items]
        if yerr_key:
            yerr = [
                float(r[yerr_key]) if finite(r.get(yerr_key)) else 0.0
                for r in items
            ]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label)
        else:
            ax.plot(xs, ys, marker="o", label=label)
        for r, x, y in zip(items, xs, ys):
            ax.annotate(
                f"{int(r['nsteps'])}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_xlabel(r"$N_r / \sqrt{V_\mathrm{def}} + \alpha n_\mathrm{steps} N_r^2 / V_\mathrm{def}$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"alpha={alpha:g}, V_def={defect_volume:g}; labels are nsteps")
    ax.grid(True, alpha=0.3)
    ax.legend(title="replicas")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return True


def plot_summary(rows, out_base: Path):
    metrics = [
        ("swap_acceptance", "swap acceptance", None),
        ("work_mean", "work mean", "work_sem"),
        ("measurement_s", "measurement time [s]", None),
        ("total_s", "total time [s]", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    any_panel = False
    for ax, (y_key, ylabel, yerr_key) in zip(axes.ravel(), metrics):
        groups = grouped_points(rows, y_key)
        if not groups:
            ax.set_visible(False)
            continue
        any_panel = True
        for label, items in sorted(groups.items()):
            xs = [int(r["nsteps"]) for r in items]
            ys = [float(r[y_key]) for r in items]
            if yerr_key:
                yerr = [
                    float(r[yerr_key]) if finite(r.get(yerr_key)) else 0.0
                    for r in items
                ]
                ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label)
            else:
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("nsteps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(title="replicas")
    if not any_panel:
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot recovered O-REX final metrics from a grep-extracted Slurm log txt."
    )
    parser.add_argument("txt", type=Path, help="Txt produced from grep over Slurm output logs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plot/orex_recovered_logs"),
        help="Directory for CSV and plots.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Collapse alpha.")
    parser.add_argument(
        "--defect-side",
        type=float,
        default=2.0,
        help="Defect side used for V_def = defect_side^3.",
    )
    args = parser.parse_args()

    rows = parse_recovered_log(args.txt)
    rows = [row for row in rows if row.get("nsteps") is not None]
    rows.sort(
        key=lambda r: (
            int(r.get("nreplicas") or r.get("world_size") or -1),
            int(r.get("nsteps") or -1),
            str(r.get("log_file", "")),
        )
    )
    defect_volume = float(args.defect_side) ** 3
    add_collapse_coordinate(rows, args.alpha, defect_volume)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "recovered_orex_metrics.csv")

    made = []
    if plot_summary(rows, args.out_dir / "summary"):
        made.append("summary")
    for y_key, ylabel, yerr_key in [
        ("swap_acceptance", "swap acceptance", None),
        ("work_mean", "work mean", "work_sem"),
        ("measurement_s", "measurement time [s]", None),
        ("total_s", "total time [s]", None),
    ]:
        if plot_metric(rows, y_key, ylabel, args.out_dir / y_key, yerr_key=yerr_key):
            made.append(y_key)
    alpha_tag = str(args.alpha).replace("-", "m").replace(".", "p")
    if plot_collapse(
        rows,
        "swap_acceptance",
        "swap acceptance",
        args.out_dir / f"collapse_acceptance_alpha{alpha_tag}",
        args.alpha,
        defect_volume,
    ):
        made.append("collapse_acceptance")
    if plot_collapse(
        rows,
        "work_mean",
        "work mean",
        args.out_dir / f"collapse_work_alpha{alpha_tag}",
        args.alpha,
        defect_volume,
        yerr_key="work_sem",
    ):
        made.append("collapse_work")

    print(f"Parsed {len(rows)} run(s)")
    print(f"Collapse alpha={args.alpha:g}, defect_volume={defect_volume:g}")
    print(f"Wrote {args.out_dir / 'recovered_orex_metrics.csv'}")
    if made:
        print("Wrote plots:", ", ".join(made))
    else:
        print("No plots written: no finite plottable metrics found.")


if __name__ == "__main__":
    main()
