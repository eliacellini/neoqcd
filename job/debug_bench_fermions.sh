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
# against HalfLatticeWilson; test/test_fermions_fused_hmc.py: the CUDA twins of the MD kernels of
# neoqcd.fermions.fused_hmc against the torch code; test/test_fermions_fused_hmc_main.py: main_hmc
# --kernel cuda --md-kernels all/each component against --kernel cuda); their failure does not
# stop the benchmark.
# Then (TRAJ=1, default) the per-component HMC trajectory breakdown with the MD kernels on and off
# (bench_fermions.py --trajectory-only): 8^4 B = 16 (beta 5.6, kappa 0.153, tau 1, 10 x 4 omelyan
# steps, the w8f integrator) for none, all and each component alone; 16^4 B = 16 for none and all
# (one timed trajectory each). Outputs in output/bench_fermions/<job id>/trajectory_8 and _16.
# GRID=0 skips the kernel grid below, TRAJ=0 the trajectory rows.
# MODE=validate runs instead the physics check of the MD kernels: the w8f point of
# job/debug_hmc_fermions.sh (8^4, beta 5.6, kappa 0.153, B = 16, tau 1, 10 x 4 omelyan, cold start)
# with --kernel cuda --md-kernels all, resumable, in output/hmc_fermions/wilson_8x8_b5.6_k0.153_fusedmd
# (100 thermalisation + 400 trajectories; repeat the job until run_meta.json says finished): the
# plaquette must agree with w8f / openQCD 0.5661(5) (doc/fermions.md 5b).
# The CUDA kernel is compiled at first use with nvcc (cineca-ai module) into
# $NEOQCD_KERNEL_CACHE (default ~/.cache/neoqcd/kernels), seconds, reused by later jobs.
# Runtime target under ~12 minutes: 0.15 s per timing trial, 3 trials, CG shortened on large
# lattices, configurations not started within 540 s are skipped (listed in run_meta.json).
# Outputs in output/bench_fermions/<job id>/ (bench.csv, run_meta.json); the table is in the log.
# Extra flags are appended (later flags win), e.g. --lattices 8^4 --batches 16, or
# --fused-kernels compiled to time only the CUDA kernel.
# Usage: sbatch job/debug_bench_fermions.sh [extra main/bench_fermions.py flags]
#        GRID=0 sbatch job/debug_bench_fermions.sh            (tests + trajectory breakdown only)
#        MODE=validate sbatch job/debug_bench_fermions.sh     (w8f physics with --md-kernels all)

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

if [[ "${MODE:-bench}" == "validate" ]]; then
  out="output/hmc_fermions/wilson_8x8_b5.6_k0.153_fusedmd"
  resume=()
  if [[ -f "${out}/checkpoint.json" ]]; then resume=(--resume); fi
  mkdir -p "$out"
  echo "=== validate --md-kernels all (w8f physics) $(date -Is) ${resume[*]:-} -> $out"
  srun python main/main_hmc.py --device cuda --seed 1234 --out "$out" "${resume[@]}" \
    --D 4 --T 8 --L 8 --N 3 --fermions wilson --beta 5.6 --kappa 0.153 --batch 16 \
    --tau 1.0 --n-steps 10 --n-gauge-substeps 4 --integrator omelyan --start cold \
    --therm 100 --n-traj 400 --meas-every 10 --checkpoint-every 25 --fast --kernel cuda --md-kernels all
  exit 0
fi

echo "=== fused-kernel unit tests $(date -Is)"
srun python -m unittest test.test_fermions_fused test.test_fermions_fused_hmc test.test_fermions_fused_hmc_main -v \
  || echo "=== fused-kernel unit tests FAILED (benchmark continues)"

out="output/bench_fermions/${SLURM_JOB_ID:-manual}"
mkdir -p "$out"
if [[ "${GRID:-1}" == 1 ]]; then
  echo "=== bench_fermions $(date -Is) -> $out"
  srun python main/bench_fermions.py --device cuda --out "$out" \
    --lattices 4^4 8^4 16^4 --batches 4 16 64 --dtypes complex128 \
    --half-modes broadcast_sum --fused-kernels loop compiled compiled_soa --skip-legacy \
    --no-compile --no-cuda-graphs --breakdown-config "" \
    --target-time 0.15 --trials 3 --warmup 3 --cg-iters 50 --cg-min-iters 5 \
    --time-budget 540 "$@"
fi
if [[ "${TRAJ:-1}" == 1 ]]; then
  echo "=== trajectory breakdown 8^4 B16 $(date -Is) -> $out/trajectory_8"
  srun python main/bench_fermions.py --device cuda --out "$out/trajectory_8" --trajectory-only \
    --trajectory-configs 8^4:16 --trajectory-md none all gauge_force update fermion_force cg \
    --trajectory-tau 1.0 --trajectory-steps 10 --trajectory-gauge-substeps 4 \
    --trajectory-warmup 1 --trajectory-repeats 2 "$@"
  echo "=== trajectory breakdown 16^4 B16 $(date -Is) -> $out/trajectory_16"
  srun python main/bench_fermions.py --device cuda --out "$out/trajectory_16" --trajectory-only \
    --trajectory-configs 16^4:16 --trajectory-md none all \
    --trajectory-tau 1.0 --trajectory-steps 10 --trajectory-gauge-substeps 4 \
    --trajectory-warmup 0 --trajectory-repeats 1 "$@"
fi
