#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-bench-fermions
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo_work/INF26_sft/ecellini/neoqcd_test

# Runs in the test worktree; never in the production checkout.
# reports/ must exist in the test worktree before sbatch (SLURM opens -o/-e before the script runs).

# Cost of the fermion kernels on one A100 (main/bench_fermions.py): hopping term (package and
# experimental variants: gather tables, spin projection, torch.compile, CUDA graphs), Wilson and
# even-odd operators, CG iterations (solver.cg vs a sync-free loop), pseudofermion force, gauge
# force, expi update, staggered M_ee, on 4^4, 8^4, 12^3x24, 16^4 x batch 1, 4, 16, 64.
# Runtime target under ~10 minutes: 0.15 s per timing trial, 3 trials, CG shortened on large
# lattices, configurations not started within 540 s are skipped (listed in run_meta.json).
# Outputs in output/bench_fermions/<job id>/ (bench.csv, run_meta.json); the table is in the log.
# Usage: sbatch job/debug_bench_fermions.sh [extra main/bench_fermions.py flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
# The environment lives only in the production checkout: activate it by absolute path, but import
# the package of the test worktree (PYTHONPATH first).
source /leonardo_work/INF26_sft/ecellini/neoqcd/.sunenv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p reports
echo "=== worktree $PWD commit $(git rev-parse HEAD)"
export OMP_NUM_THREADS=1

out="output/bench_fermions/${SLURM_JOB_ID:-manual}"
mkdir -p "$out"
echo "=== bench_fermions $(date -Is) -> $out"
srun python main/bench_fermions.py --device cuda --out "$out" \
  --target-time 0.15 --trials 3 --warmup 3 --cg-iters 50 --cg-min-iters 5 \
  --time-budget 540 "$@"
