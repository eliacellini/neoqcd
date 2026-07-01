#!/usr/bin/env python3
"""
Collect NEO-REX pretrained-evaluation array logs into one combined log and CSV.

Run from the project root on Leonardo, for example:

    python main/collect_neorex_eval_results.py

The combined txt keeps `path:payload` lines, so it can be pasted back here and
parsed like the previous recovered OREX logs.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


FLOAT_RE = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

KEEP_PATTERNS = [
    re.compile(r"^Running NEO-REX pretrained evaluation array task:"),
    re.compile(r"^SLURM_ARRAY_TASK_ID="),
    re.compile(r"^NREPLICAS="),
    re.compile(r"^NEOREX_PROTOCOL_STEPS="),
    re.compile(r"^NEOREX_LOAD_FLOW="),
    re.compile(r"^STUDY_NAME="),
    re.compile(r"^RUN_NAME="),
    re.compile(r"^WANDB_RUN_NAME="),
    re.compile(r"^OUTPUT_DIR="),
    re.compile(r"^Running command:"),
    re.compile(r"^ python "),
    re.compile(r"^WANDB_MODE="),
    re.compile(r"^backend="),
    re.compile(r"^neorex_flow_checkpoint="),
    re.compile(r"^thermalization_steps="),
    re.compile(r"^step="),
    re.compile(r"^final_swap_acceptance_mean="),
    re.compile(r"^timing_thermalization_s="),
    re.compile(r"^final_work_mean="),
    re.compile(r"^final_flow_loss_mean="),
]

PATTERNS = {
    "array_task_id": re.compile(r"\bSLURM_ARRAY_TASK_ID=(?P<array_task_id>\d+)"),
    "nreplicas": re.compile(r"\bNREPLICAS=(?P<nreplicas>\d+)"),
    "nsteps": re.compile(r"\bNEOREX_PROTOCOL_STEPS=(?P<nsteps>\d+)"),
    "load_flow": re.compile(r"\bNEOREX_LOAD_FLOW=(?P<load_flow>\S+)"),
    "study_name": re.compile(r"\bSTUDY_NAME=(?P<study_name>\S+)"),
    "run_name": re.compile(r"\bRUN_NAME=(?P<run_name>\S+)"),
    "wandb_run_name": re.compile(r"\bWANDB_RUN_NAME=(?P<wandb_run_name>\S+)"),
    "output_dir": re.compile(r"\bOUTPUT_DIR=(?P<output_dir>\S+)"),
    "backend": re.compile(
        r"backend=(?P<backend>\S+).*world_size=(?P<world_size>\d+).*"
        r"algorithm=(?P<algorithm>\S+).*output_dir=(?P<main_output_dir>\S+)"
    ),
    "thermalization": re.compile(
        r"thermalization_steps=(?P<thermalization_steps>\d+)\s+"
        r"thermalization_steps_executed=(?P<thermalization_steps_executed>\d+)\s+"
        r"cfg_cache=(?P<cfg_cache>\S+)\s+"
        r"cfg_cache_steps=(?P<cfg_cache_steps>\d+)\s+"
        rf"thermalization_elapsed_s=(?P<thermalization_elapsed_s>{FLOAT_RE})"
    ),
    "swap": re.compile(
        rf"final_swap_acceptance_mean=(?P<swap_acceptance>{FLOAT_RE})\s+"
        r"total_accepted=(?P<total_accepted>\d+)\s+"
        r"total_proposed=(?P<total_proposed>\d+)\s+"
        r"swap_attempts=(?P<swap_attempts>\d+)"
    ),
    "timing": re.compile(
        rf"timing_thermalization_s=(?P<thermalization_s>{FLOAT_RE})\s+"
        rf"timing_measurement_s=(?P<measurement_s>{FLOAT_RE})\s+"
        rf"timing_total_s=(?P<total_s>{FLOAT_RE})"
    ),
    "work": re.compile(
        rf"final_work_mean=(?P<work_mean>{FLOAT_RE})\s+"
        rf"final_work_sem=(?P<work_sem>{FLOAT_RE})\s+"
        r"work_count=(?P<work_count>\d+)"
    ),
    "flow": re.compile(
        rf"final_flow_loss_mean=(?P<flow_loss_mean>{FLOAT_RE})\s+"
        r"flow_loss_count=(?P<flow_loss_count>\d+)\s+"
        r"flow_applied_updates=(?P<flow_applied_updates>\d+)\s+"
        r"flow_skipped_nan_grad_updates=(?P<flow_skipped_nan_grad_updates>\d+)"
    ),
}

CSV_FIELDS = [
    "log_file",
    "array_task_id",
    "study_name",
    "run_name",
    "output_dir",
    "world_size",
    "nreplicas",
    "nsteps",
    "load_flow",
    "swap_acceptance",
    "total_accepted",
    "total_proposed",
    "swap_attempts",
    "work_mean",
    "work_sem",
    "work_count",
    "flow_loss_mean",
    "flow_loss_count",
    "flow_applied_updates",
    "flow_skipped_nan_grad_updates",
    "thermalization_steps",
    "thermalization_steps_executed",
    "cfg_cache",
    "cfg_cache_steps",
    "thermalization_elapsed_s",
    "thermalization_s",
    "measurement_s",
    "total_s",
    "backend",
    "algorithm",
    "wandb_run_name",
]

INT_FIELDS = {
    "array_task_id",
    "world_size",
    "nreplicas",
    "nsteps",
    "total_accepted",
    "total_proposed",
    "swap_attempts",
    "work_count",
    "flow_loss_count",
    "flow_applied_updates",
    "flow_skipped_nan_grad_updates",
    "thermalization_steps",
    "thermalization_steps_executed",
    "cfg_cache_steps",
}

FLOAT_FIELDS = {
    "swap_acceptance",
    "work_mean",
    "work_sem",
    "flow_loss_mean",
    "thermalization_elapsed_s",
    "thermalization_s",
    "measurement_s",
    "total_s",
}


def parse_float(value: str) -> float:
    value = str(value).strip().lower()
    if value == "nan":
        return math.nan
    if value in {"inf", "+inf"}:
        return math.inf
    if value == "-inf":
        return -math.inf
    return float(value)


def should_keep(line: str) -> bool:
    payload = line.strip()
    return any(pattern.search(payload) for pattern in KEEP_PATTERNS)


def parse_log_file(path: Path, project_dir: Path) -> dict:
    row = {"log_file": str(path.relative_to(project_dir)) if path.is_relative_to(project_dir) else str(path)}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            payload = line.strip()
            for pattern in PATTERNS.values():
                match = pattern.search(payload)
                if match:
                    row.update(match.groupdict())

    if row.get("main_output_dir") and not row.get("output_dir"):
        row["output_dir"] = row["main_output_dir"]
    if row.get("world_size") and not row.get("nreplicas"):
        row["nreplicas"] = row["world_size"]
    if not row.get("nsteps"):
        for key in ("run_name", "output_dir", "wandb_run_name"):
            match = re.search(r"nsteps(\d+)", str(row.get(key) or ""))
            if match:
                row["nsteps"] = match.group(1)
                break

    for key in INT_FIELDS:
        if row.get(key) not in (None, ""):
            row[key] = int(row[key])
    for key in FLOAT_FIELDS:
        if row.get(key) not in (None, ""):
            row[key] = parse_float(row[key])
    return row


def discover_logs(reports_dir: Path, extra_globs: list[str]) -> list[Path]:
    globs = [
        "output_neoqcd-neorex-eval-r4_*",
        "output_neoqcd-neorex-eval-r8_*",
    ]
    globs.extend(extra_globs)

    paths = []
    for pattern in globs:
        paths.extend(reports_dir.glob(pattern))
    return sorted(set(p for p in paths if p.is_file()))


def write_combined_log(paths: list[Path], out_path: Path, project_dir: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in paths:
            label = str(path.relative_to(project_dir)) if path.is_relative_to(project_dir) else str(path)
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if should_keep(line):
                        out.write(f"{label}:{line.rstrip()}\n")
                        kept += 1
    return kept


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (int(r.get("nreplicas", -1)), int(r.get("nsteps", -1)), str(r.get("log_file", "")))):
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-prefix", type=str, default="neorex_pretrained_eval")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Extra reports glob to include. Can be passed multiple times.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    reports_dir = (args.reports_dir or (project_dir / "reports")).resolve()
    out_dir = (args.out_dir or (project_dir / "results" / "pt_obc" / "collected_logs")).resolve()

    paths = discover_logs(reports_dir, args.glob)
    if not paths:
        raise SystemExit(f"No NEO-REX eval array logs found under {reports_dir}")

    rows = [parse_log_file(path, project_dir) for path in paths]
    txt_path = out_dir / f"{args.out_prefix}.txt"
    csv_path = out_dir / f"{args.out_prefix}.csv"

    kept = write_combined_log(paths, txt_path, project_dir)
    write_csv(rows, csv_path)

    print(f"Found {len(paths)} log file(s)")
    print(f"Wrote combined log: {txt_path}")
    print(f"Wrote CSV:          {csv_path}")
    print(f"Kept {kept} selected line(s) in combined log")
    print()
    print("Summary:")
    for row in sorted(rows, key=lambda r: (int(r.get("nreplicas", -1)), int(r.get("nsteps", -1)), str(r.get("log_file", "")))):
        print(
            "  "
            f"r={row.get('nreplicas', '?')} "
            f"nsteps={row.get('nsteps', '?')} "
            f"acc={row.get('swap_acceptance', '?')} "
            f"work={row.get('work_mean', '?')} "
            f"time={row.get('measurement_s', '?')} "
            f"updates={row.get('flow_applied_updates', '?')} "
            f"log={row.get('log_file', '?')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
