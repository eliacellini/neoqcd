#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-pg-fermions
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo_work/INF26_sft/ecellini/neoqcd_test

# Runs in the test worktree; never in the production checkout.
# reports/ must exist in the test worktree before sbatch (SLURM opens -o/-e before the script runs).

# Pilot of the PG training with dynamical fermions (main/main_pg_fermions.py): one node, 4 A100, one
# bc rung per GPU (torchrun, NCCL, rank l on cuda:l), fp64.
# 8^4 SU(3), N_f = 2 Wilson, beta = 5.6, kappa = 0.153; gauge-only defect L_d = 2 (utils.Defect, time
# slice 4, spatial corner 3: the patch of the defect field needs a 2-site buffer); 4 rungs evenly spaced
# in [0, 1] (or LADDER=<json of main/optimal_ladder.py>), 4 chains per rank, alternating swaps, no
# translation. Field: defect-hyper-smearing, class mode (the 16^4 pure-gauge field), hidden 16, depth 2.
# Epoch = 1 HMC trajectory per replica (tau = 1, 8 Omelyan steps, 4 gauge substeps), swaps, PG loss
# (pure-gauge part + pseudofermion term: 2 CG solves per chain), one Adam step (lr 1e-3).
# 20 thermalisation trajectories at fixed labels and 5 REX steps before the first epoch, then 60 epochs,
# checkpoint every 5. Cold start.
# Outputs in output/pg_fermions/<name>/ (metrics.jsonl, config.json, run_info.json, cnf_checkpoint.pt,
# cnf_checkpoint.rank<r>.slot<s>.pt). The checkpoint is resumed when present (--resume), so resubmitting
# extends a killed run; to restart from scratch pass --overwrite.
# What to look at: metrics.jsonl per rung pinn_fermion_by_rung / pinn_gauge_by_rung / pinn_cov_by_rung
# (the fermionic share of the PG noise), spf_per_component_by_rung (~1), loss_s vs sweep_s and
# fermion_loss_s, loss_cg_iter vs trajectory_cg_iter, max_residual, unconverged_solves.
# Usage: sbatch job/debug_pg_fermions.sh [extra flags for main/main_pg_fermions.py]

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
export NCCL_SHM_DISABLE=1

# One process (one rung) per GPU of the node.
NPROC="${NPROC:-${SLURM_GPUS_PER_NODE:-4}}"
NPROC="${NPROC##*:}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

SEED="${SEED:-1234}"
OUT="${OUT:-output/pg_fermions/pgf_r${NPROC}_c4_cube2_b5.6_k0.153_8x8p3_dhs_class}"

extra=()
if [[ -f "${OUT}/cnf_checkpoint.pt" ]]; then
  extra+=(--resume)
fi
if [[ -n "${LADDER:-}" ]]; then
  extra+=(--parameter-ladder "$LADDER")
fi
mkdir -p "$OUT"
echo "=== ${OUT} $(date -Is) nproc=${NPROC} ${extra[*]:-}"

srun python -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" --nproc_per_node="$NPROC" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  main/main_pg_fermions.py --device cuda --seed "$SEED" --output-dir "$OUT" "${extra[@]}" \
  --D 4 --T 8 --L 8 --N 3 --beta 5.6 --fermions wilson --kappa 0.153 \
  --defect-shape cube --defect-size 2 --time-slice 4 --space-slice 3 \
  --parameter-min 0.0 --parameter-max 1.0 --replicas "$NPROC" --batch-size 4 --swap-parity alternating \
  --tau 1.0 --n-steps 8 --n-gauge-substeps 4 --integrator omelyan \
  --architecture defect-hyper-smearing --hyper-smearing-mode class --hidden-dim 16 --depth 2 \
  --therm 20 --warmup 5 --epochs 60 --save-every 5 --lr 1e-3 --grad-clip-norm 1.0 "$@"
