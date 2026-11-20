#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-snf-fermions
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

# Pilots of fermions_plan.md phase 4 step 3 (main/main_snf_fermions.py), one A100, fp64,
# 4^4 SU(3), N_f = 2 Wilson, protocol kappa 0.153 -> 0.1545 at beta = 5.6 (Model C of Abbott et al.,
# on 4^4), 8 protocol steps, one HMC trajectory per block:
#   nemcmc  NE-MCMC, pool thermalised from cold (100 trajectories at kappa = 0.153), 256 samples;
#   snf     signed stout layers (--layers smearing_signed) started exactly at the identity
#           (--rho-init 0, a regular point), lr 1e-3 with a 5-step warm-up, from the pool checkpoint
#           of `nemcmc` (no new thermalisation), 20 Adam steps on mean(W), then 128 samples. (Until
#           20 November this point ran the |rho| stout layer with --rho-init 1e-3 --lr 1e-3, which
#           the driver now refuses: Adam's first step sends every rho to ~0 or ~2e-3 and flips
#           signs; its output directory was snf_smearing_..., the new one is snf_signed_...);
#   residual (optional, not in the default POINTS) static residual layers (--layers residual, one
#           per step) with two frozen-link convolutions (--residual-n-pt 2, mask mod 2), started
#           exactly at the identity (--residual-coeff-init 0, a regular point), default
#           conditioning (normalised position, fractional step d_i = 1/8), lr 1e-3 with a 5-step
#           warm-up, 20 Adam steps, then 128 samples; from the pool of `nemcmc` as `snf`.
# Both points use the fast even-odd operator (--fast, fp64, default colour product); the output
# directories end in _fast so no old non-fast checkpoint is resumed (the resume key refuses a
# fast/non-fast mix anyway).
# Outputs in output/snf_fermions/<name>/ (samples.csv, blocks.csv, training.csv, run_meta.json,
# pool.*, flow.*). A point whose checkpoint exists is resumed (--resume); remove its directory to
# restart. `snf` needs the pool of `nemcmc` (run both in one job, or nemcmc first).
# Usage: sbatch job/debug_snf_fermions.sh                 (both points)
#        POINTS="snf" sbatch job/debug_snf_fermions.sh [extra flags for main/main_snf_fermions.py]
#        POINTS="nemcmc residual" sbatch job/debug_snf_fermions.sh

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

POINTS="${POINTS:-nemcmc snf}"
SEED="${SEED:-1234}"
BASE="output/snf_fermions"
NEMCMC_OUT="${BASE}/nemcmc_kappa0.153to0.1545_n8_b5.6_4x4p3_fast"
SNF_OUT="${BASE}/snf_signed_kappa0.153to0.1545_n8_b5.6_4x4p3_fast"
RESIDUAL_OUT="${BASE}/snf_residual_npt2_kappa0.153to0.1545_n8_b5.6_4x4p3_fast"

COMMON=(--D 4 --T 4 --L 4 --N 3 --beta 5.6 --kappa 0.153
        --protocol kappa --initial 0.153 --final 0.1545 --n-step 8
        --tau 1.0 --n-steps 8 --n-gauge-substeps 4 --integrator omelyan --n-traj 1
        --batch 16 --n-between 1 --checkpoint-every 4 --fast)

run_point() {
  local out="$1"; shift
  local resume=()
  if [[ -f "${out}/pool.json" ]]; then resume=(--resume); fi
  mkdir -p "$out"
  echo "=== ${out} $(date -Is) ${resume[*]:-}"
  srun python main/main_snf_fermions.py --device cuda --seed "$SEED" --out "$out" "${resume[@]}" "$@"
}

for point in $POINTS; do
  case "$point" in
    nemcmc) run_point "$NEMCMC_OUT" "${COMMON[@]}" --mode nemcmc --therm 100 --eval-samples 256 "${@}" ;;
    snf)
      if [[ ! -f "${NEMCMC_OUT}/pool.json" ]]; then echo "snf needs ${NEMCMC_OUT}/pool.json" >&2; exit 2; fi
      run_point "$SNF_OUT" "${COMMON[@]}" --mode snf --layers smearing_signed --rho-init 0 --lr 1e-3 \
        --lr-warmup 5 --train-steps 20 --eval-samples 128 --start-config "${NEMCMC_OUT}/pool" "${@}" ;;
    residual)
      if [[ ! -f "${NEMCMC_OUT}/pool.json" ]]; then echo "residual needs ${NEMCMC_OUT}/pool.json" >&2; exit 2; fi
      run_point "$RESIDUAL_OUT" "${COMMON[@]}" --mode snf --layers residual --residual-n-pt 2 \
        --residual-mask-mod 2 --residual-coeff-init 0 --lr 1e-3 --lr-warmup 5 \
        --train-steps 20 --eval-samples 128 --start-config "${NEMCMC_OUT}/pool" "${@}" ;;
    *) echo "unknown point $point" >&2; exit 2 ;;
  esac
done
