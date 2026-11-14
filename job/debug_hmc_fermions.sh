#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-hmc-fermions
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

# N_f = 2 Wilson HMC benchmarks of fermions_plan.md (phase 1, Benchmark and cost), one A100, fp64:
#   w4  4^4 SU(3), beta = 1.0, kappa = 0.1, with the exact log det every trajectory (1536 x 1536);
#   w8  8^4 SU(3), beta = 5.6, kappa = 0.153 (MIT point), no exact log det;
#   w8b as w8 with kappa = 0.1545 (optional, not in the default POINTS).
# All points use the fast even-odd operator (--fast, fp64, default colour product); the output
# directories end in _fast so no old non-fast checkpoint is resumed.
# Fused CUDA hop kernel (neoqcd.fermions.fused, --kernel cuda, default link layout soa; optional,
# not in the default POINTS; output directories end in _fused, so they never resume a --fast run):
#   w4f  as w4 with --kernel cuda (validate against w4: plaquette 0.06084(19), <log det> 0.9248(50);
#        openQCD 0.060765(28));
#   w8f  as w8 with --kernel cuda (validate against w8: plaquette 0.56609(29) after trajectory 120;
#        openQCD 0.5661(5));
#   w16  16^4 SU(3), beta = 5.6, kappa = 0.153, --kernel cuda --precision double, batch 16, cold
#        start, 20 thermalisation + 60 trajectories, checkpoint every 10 (n-steps 16: a first
#        guess for the 16x larger volume; adjust after the first acceptance numbers). It needs
#        more than one 30-minute debug slot: resubmit to extend (--resume is automatic).
# Outputs in output/hmc_fermions/<name>/ (observables.csv, run_meta.json, checkpoint.*).
# A point whose checkpoint exists is resumed (--resume), so resubmitting extends a run that hit
# the wall time; to restart from scratch remove its directory first.
# Usage: sbatch job/debug_hmc_fermions.sh            (both points)
#        POINTS="w8" sbatch job/debug_hmc_fermions.sh [extra flags for main/main_hmc.py]
#        POINTS="w4f w8f" sbatch job/debug_hmc_fermions.sh      (fused kernel validation)

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

POINTS="${POINTS:-w4 w8}"
SEED="${SEED:-1234}"

run_point() {
  # run_point <name> <suffix: fast|fused> <main_hmc.py arguments...>
  local name="$1" suffix="$2"; shift 2
  local out="output/hmc_fermions/${name}_${suffix}"
  local resume=()
  if [[ -f "${out}/checkpoint.json" ]]; then resume=(--resume); fi
  mkdir -p "$out"
  echo "=== ${name} $(date -Is) ${resume[*]:-}"
  srun python main/main_hmc.py --device cuda --seed "$SEED" --out "$out" "${resume[@]}" "$@"
}

for point in $POINTS; do
  case "$point" in
    w4) run_point "wilson_4x4_b1.0_k0.1" fast \
          --D 4 --T 4 --L 4 --N 3 --fermions wilson --beta 1.0 --kappa 0.1 --batch 4 \
          --tau 1.0 --n-steps 6 --n-gauge-substeps 2 --integrator omelyan --start cold \
          --therm 50 --n-traj 300 --meas-every 1 --checkpoint-every 50 --exact-logdet --fast "${@}" ;;
    w8) run_point "wilson_8x8_b5.6_k0.153" fast \
          --D 4 --T 8 --L 8 --N 3 --fermions wilson --beta 5.6 --kappa 0.153 --batch 16 \
          --tau 1.0 --n-steps 10 --n-gauge-substeps 4 --integrator omelyan --start cold \
          --therm 100 --n-traj 400 --meas-every 10 --checkpoint-every 25 --fast "${@}" ;;
    w8b) run_point "wilson_8x8_b5.6_k0.1545" fast \
          --D 4 --T 8 --L 8 --N 3 --fermions wilson --beta 5.6 --kappa 0.1545 --batch 16 \
          --tau 1.0 --n-steps 10 --n-gauge-substeps 4 --integrator omelyan --start cold \
          --therm 100 --n-traj 400 --meas-every 10 --checkpoint-every 25 --fast "${@}" ;;
    w4f) run_point "wilson_4x4_b1.0_k0.1" fused \
          --D 4 --T 4 --L 4 --N 3 --fermions wilson --beta 1.0 --kappa 0.1 --batch 4 \
          --tau 1.0 --n-steps 6 --n-gauge-substeps 2 --integrator omelyan --start cold \
          --therm 50 --n-traj 300 --meas-every 1 --checkpoint-every 50 --exact-logdet --fast --kernel cuda "${@}" ;;
    w8f) run_point "wilson_8x8_b5.6_k0.153" fused \
          --D 4 --T 8 --L 8 --N 3 --fermions wilson --beta 5.6 --kappa 0.153 --batch 16 \
          --tau 1.0 --n-steps 10 --n-gauge-substeps 4 --integrator omelyan --start cold \
          --therm 100 --n-traj 400 --meas-every 10 --checkpoint-every 25 --fast --kernel cuda "${@}" ;;
    w16) run_point "wilson_16x16_b5.6_k0.153" fused \
          --D 4 --T 16 --L 16 --N 3 --fermions wilson --beta 5.6 --kappa 0.153 --batch 16 \
          --tau 1.0 --n-steps 16 --n-gauge-substeps 4 --integrator omelyan --start cold \
          --therm 20 --n-traj 60 --meas-every 1 --checkpoint-every 10 \
          --fast --kernel cuda --precision double "${@}" ;;
    *) echo "unknown point $point" >&2; exit 2 ;;
  esac
done
