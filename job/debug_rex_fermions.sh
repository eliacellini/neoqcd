#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-rex-fermions
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

# Pilot of fermions_plan.md phase 5 (main/main_rex_fermions.py), one A100, fp64:
# 4^4 SU(3), N_f = 2 Wilson, beta = 5.6, kappa = 0.153, gauge-only defect of 2 time links at t = 0
# (--defect-shape line --defect-size 2), ladder bc = 0, 0.5, 1, 4 independent ladders (B = 12),
# DEO swaps, periodic translation of the bc = 1 replica, 50 thermalisation + 200 measured steps.
# Start: cold, or the NE-MCMC pool of job/debug_snf_fermions.sh at kappa = 0.153 if it exists
# (equilibrium at bc = 1; the open rungs thermalise during --therm).
# Fast even-odd operator (--fast, fp64); the output directory ends in _fast so no old non-fast
# checkpoint is resumed. The start pool is that of the fast NE-MCMC pilot (links only: the format
# is the same for both paths).
# Outputs in output/rex_fermions/<name>/ (observables.csv, replica_history.npz, run_meta.json,
# checkpoint.*, rex_state.pt). An existing checkpoint is resumed (--resume); a job killed before
# its first checkpoint restarts from scratch (the driver deletes its own partial files).
# Usage: sbatch job/debug_rex_fermions.sh [extra flags for main/main_rex_fermions.py]

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

SEED="${SEED:-1234}"
OUT="output/rex_fermions/rex_r3_c4_line2_b5.6_k0.153_4x4p3_fast"
POOL="output/snf_fermions/nemcmc_kappa0.153to0.1545_n8_b5.6_4x4p3_fast/pool"

extra=()
if [[ -f "${OUT}/checkpoint.json" ]]; then
  extra=(--resume)
elif [[ -f "${POOL}.json" ]]; then
  extra=(--start-config "$POOL")
fi
mkdir -p "$OUT"
echo "=== ${OUT} $(date -Is) ${extra[*]:-}"
srun python main/main_rex_fermions.py --device cuda --seed "$SEED" --out "$OUT" "${extra[@]}" \
  --D 4 --T 4 --L 4 --N 3 --beta 5.6 --kappa 0.153 \
  --ladder 0,0.5,1 --defect-shape line --defect-size 2 --n-chains 4 \
  --tau 1.0 --n-steps 8 --n-gauge-substeps 4 --integrator omelyan \
  --therm 50 --steps 200 --swap-parity alternating --periodic-translation --checkpoint-every 25 --fast "$@"
