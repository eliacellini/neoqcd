#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-orex-array
#SBATCH -e reports/errors_%x_%A_%a
#SBATCH -o reports/output_%x_%A_%a
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/leonardo_scratch/large/userexternal/ecellini/neoqcd

set -euo pipefail

# -----------------------------
# Distributed launch
# -----------------------------
NPROC=4

# -----------------------------
# Project / environment
# -----------------------------
PROJECT_DIR="${PROJECT_DIR:-/leonardo_scratch/large/userexternal/ecellini/neoqcd}"

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate

export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# -----------------------------
# Physics / PT config
# -----------------------------
D=4
T=8
L=8
N=3
BETA=6.0
BC_MIN=0.0
BC_MAX=1.0
BS=128

DEFECT_SIZE=2
TIME_SLICE=4
SPACE_SLICE=3

THERMAL_STEPS=20
STEPS=100
SWEEP=5
SWAP_EVERY=1
SWAP_PARITY=alternating
OREX_SCHEDULE=even_odd
CFG_CACHE_TAG=production

# -----------------------------
# O-REX array config
# -----------------------------
OREX_PROTOCOL_STEPS_LIST=(0 2 4 6 8)
OREX_PROTOCOL_STEPS="${OREX_PROTOCOL_STEPS_LIST[$SLURM_ARRAY_TASK_ID]}"
OREX_MCMC_STEPS_PER_STEP=1

# -----------------------------
# Logging
# -----------------------------
LOG_EVERY=10
SEED=137
WANDB_PROJECT=neo-pt
WANDB_ENTITY=lqft-snf
RUN_NAME="${RUN_NAME:-prod_orex_obc_T${T}_L${L}_d${DEFECT_SIZE}_ts${TIME_SLICE}_ss${SPACE_SLICE}_bs${BS}_sw${SWEEP}_nsteps${OREX_PROTOCOL_STEPS}}"
OUTPUT_DIR="${PROJECT_DIR}/results/pt_obc/${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

CMD=(
  python
  -m
  torch.distributed.run
  --nnodes="$SLURM_NNODES"
  --node_rank="$SLURM_NODEID"
  --nproc_per_node="$NPROC"
  --rdzv_id="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  main/main_pt_obc.py
  --algorithm orex
  --D "$D"
  --T "$T"
  --L "$L"
  --N "$N"
  --beta "$BETA"
  --defect-size "$DEFECT_SIZE"
  --time-slice "$TIME_SLICE"
  --space-slice "$SPACE_SLICE"
  --batch-size "$BS"
  --bc-min "$BC_MIN"
  --bc-max "$BC_MAX"
  --thermal-steps "$THERMAL_STEPS"
  --steps "$STEPS"
  --sweep "$SWEEP"
  --swap-every "$SWAP_EVERY"
  --swap-parity "$SWAP_PARITY"
  --orex-schedule "$OREX_SCHEDULE"
  --orex-protocol-steps "$OREX_PROTOCOL_STEPS"
  --orex-mcmc-steps-per-step "$OREX_MCMC_STEPS_PER_STEP"
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --wandb-run-name "$RUN_NAME"
  --run-name "$RUN_NAME"
  --output-dir "$OUTPUT_DIR"
  --main-dir "$PROJECT_DIR"
  --cfg-cache-tag "$CFG_CACHE_TAG"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
)

echo "Running O-REX production array task:"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "OREX_PROTOCOL_STEPS=$OREX_PROTOCOL_STEPS"
echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
echo "WANDB_MODE=$WANDB_MODE"
echo "W&B offline sync command after the job:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"

srun "${CMD[@]}"
