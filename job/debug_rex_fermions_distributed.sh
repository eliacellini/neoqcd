#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-rex-fermions-dist
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

# Pilot of the distributed fermionic REX (main/main_rex_fermions_distributed.py): one node, 4 A100,
# one rung per GPU (torchrun, NCCL, rank l on cuda:l), fp64.
# 4^4 SU(3), N_f = 2 Wilson, beta = 5.6, kappa = 0.153; gauge-only defect of size L_d = 2 built as in
# main_pt_obc.py (utils.Defect: 2^3 time links at t = 0, spatial corner 1); 4 rungs evenly spaced in
# [0, 1] (or LADDER=<json of main/optimal_ladder.py>), 4 independent ladders per rank, alternating
# swaps, periodic translation of the bc = 1 replica; 20 thermalisation trajectories without swaps,
# then 100 REX steps. Cold start.
# Outputs in output/rex_fermions_distributed/<name>/ (observables.csv, replica_history.npz,
# run_meta.json, rankNNN/checkpoint.*, rankNNN/rex_state.pt). An existing checkpoint is resumed
# (--resume), so resubmitting extends a run stopped by --max-run-minutes; to restart from scratch
# remove the directory first. The next ladder: python main/optimal_ladder.py <out> --output ladder.json.
# Usage: sbatch job/debug_rex_fermions_distributed.sh [extra flags for main/main_rex_fermions_distributed.py]

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
OUT="${OUT:-output/rex_fermions_distributed/rexd_r${NPROC}_c4_cube2_b5.6_k0.153_4x4p3}"

extra=()
if [[ -f "${OUT}/rank000/checkpoint.json" ]]; then
  extra+=(--resume)
fi
if [[ -n "${LADDER:-}" ]]; then
  extra+=(--ladder-file "$LADDER")
else
  extra+=(--bc-min 0.0 --bc-max 1.0)
fi
mkdir -p "$OUT"
echo "=== ${OUT} $(date -Is) nproc=${NPROC} ${extra[*]:-}"

# --device cuda: each rank takes cuda:LOCAL_RANK (explicit index, required by torch 2.0's set_device).
srun python -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" --nproc_per_node="$NPROC" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  main/main_rex_fermions_distributed.py --device cuda --seed "$SEED" --out "$OUT" "${extra[@]}" \
  --D 4 --T 4 --L 4 --N 3 --beta 5.6 --fermions wilson --kappa 0.153 \
  --defect-shape cube --defect-size 2 --time-slice 0 --space-slice 1 --n-chains 4 \
  --tau 1.0 --n-steps 8 --n-gauge-substeps 4 --integrator omelyan \
  --therm 20 --steps 100 --swap-parity alternating --periodic-translation \
  --checkpoint-every 25 --log-every 10 --max-run-minutes 25 "$@"
