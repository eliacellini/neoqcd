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
#   snf     stout (smearing) layers, starting from the pool checkpoint of `nemcmc` (no new
#           thermalisation), 20 Adam steps on mean(W), then 128 samples.
# Outputs in output/snf_fermions/<name>/ (samples.csv, blocks.csv, training.csv, run_meta.json,
# pool.*, flow.*). A point whose checkpoint exists is resumed (--resume); remove its directory to
# restart. `snf` needs the pool of `nemcmc` (run both in one job, or nemcmc first).
# Usage: sbatch job/debug_snf_fermions.sh                 (both points)
#        POINTS="snf" sbatch job/debug_snf_fermions.sh [extra flags for main/main_snf_fermions.py]

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
NEMCMC_OUT="${BASE}/nemcmc_kappa0.153to0.1545_n8_b5.6_4x4p3"
SNF_OUT="${BASE}/snf_smearing_kappa0.153to0.1545_n8_b5.6_4x4p3"

COMMON=(--D 4 --T 4 --L 4 --N 3 --beta 5.6 --kappa 0.153
        --protocol kappa --initial 0.153 --final 0.1545 --n-step 8
        --tau 1.0 --n-steps 8 --n-gauge-substeps 4 --integrator omelyan --n-traj 1
        --batch 16 --n-between 1 --checkpoint-every 4)

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
      run_point "$SNF_OUT" "${COMMON[@]}" --mode snf --layers smearing --rho-init 1e-3 --lr 1e-3 \
        --train-steps 20 --eval-samples 128 --start-config "${NEMCMC_OUT}/pool" "${@}" ;;
    *) echo "unknown point $point" >&2; exit 2 ;;
  esac
done
