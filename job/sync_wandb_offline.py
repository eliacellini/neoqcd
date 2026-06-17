#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sync recent offline Weights & Biases runs produced by the SLURM jobs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results"),
        help="Root directory to scan for W&B offline-run-* folders.",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=10,
        help="Number of newest offline runs to sync. Ignored with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync every offline run found under --root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the wandb sync commands without running them.",
    )
    parser.add_argument(
        "--wandb-bin",
        default="wandb",
        help="wandb executable to use.",
    )
    return parser.parse_args()


def find_offline_runs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = [p for p in root.rglob("offline-run-*") if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    runs = find_offline_runs(root)
    if not runs:
        print(f"No offline W&B runs found under {root}")
        return 1

    selected = runs if args.all else runs[: max(args.latest, 0)]
    if not selected:
        print("No runs selected.")
        return 1

    wandb_bin = shutil.which(args.wandb_bin) or args.wandb_bin
    print(f"Found {len(runs)} offline run(s) under {root}")
    print(f"Syncing {len(selected)} run(s)")

    failed = 0
    for run_dir in selected:
        cmd = [wandb_bin, "sync", str(run_dir)]
        print("+ " + " ".join(cmd))
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
