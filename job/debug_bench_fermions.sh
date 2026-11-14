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

# Cost of the fermion kernels on one A100 (main/bench_fermions.py), reduced grid (the full one
# took 12:29): 4^4, 8^4, 16^4 x batch 4, 16, 64, complex128 configurations (the half-lattice rows
# include their complex64 copies). Rows: reference hopping / Wilson / even-odd operators, the
# half-lattice operator (broadcast_sum), lean CG and solves, the fused kernels of
# neoqcd.fermions.fused (pure-torch loop, CUDA kernel with site-major and site-fastest links; an
# unavailable backend gives a "skipped" row with the reason), pseudofermion and gauge forces, expi
# update, staggered M_ee. Dropped as settled (--skip-legacy etc.): experimental hopping variants,
# torch.compile, CUDA graphs, the breakdown, matmul / broadcast modes, solver.cg / bare / reference
# CG iterations and the reference solve.
# First the fused-kernel unit tests run on the GPU (test/test_fermions_fused.py: the CUDA kernel
# against HalfLatticeWilson); their failure does not stop the benchmark.
# The CUDA kernel is compiled at first use with nvcc (cineca-ai module) into
# $NEOQCD_KERNEL_CACHE (default ~/.cache/neoqcd/kernels), seconds, reused by later jobs.
# Runtime target under ~12 minutes: 0.15 s per timing trial, 3 trials, CG shortened on large
# lattices, configurations not started within 540 s are skipped (listed in run_meta.json).
# Outputs in output/bench_fermions/<job id>/ (bench.csv, run_meta.json); the table is in the log.
# Extra flags are appended (later flags win), e.g. --lattices 8^4 --batches 16, or
# --fused-kernels compiled to time only the CUDA kernel.
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
# one kernel cache for the tests and the benchmark (the tests otherwise build in a temporary one)
export NEOQCD_KERNEL_CACHE="${NEOQCD_KERNEL_CACHE:-$HOME/.cache/neoqcd/kernels}"
echo "=== nvcc: $(command -v nvcc || echo none) $(nvcc --version 2>/dev/null | tail -1)"

echo "=== fused-kernel unit tests $(date -Is)"
srun python -m unittest test.test_fermions_fused -v || echo "=== fused-kernel unit tests FAILED (benchmark continues)"

out="output/bench_fermions/${SLURM_JOB_ID:-manual}"
mkdir -p "$out"
echo "=== bench_fermions $(date -Is) -> $out"
srun python main/bench_fermions.py --device cuda --out "$out" \
  --lattices 4^4 8^4 16^4 --batches 4 16 64 --dtypes complex128 \
  --half-modes broadcast_sum --fused-kernels loop compiled compiled_soa --skip-legacy \
  --no-compile --no-cuda-graphs --breakdown-config "" \
  --target-time 0.15 --trials 3 --warmup 3 --cg-iters 50 --cg-min-iters 5 \
  --time-budget 540 "$@"
