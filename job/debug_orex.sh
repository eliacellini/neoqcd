#!/bin/bash
#SBATCH --job-name=neoqcd-orex-debug
#SBATCH -e job/reports/errors_%x_%j
#SBATCH -o job/reports/output_%x_%j
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00

set -euo pipefail

# -----------------------------
# Distributed launch
# -----------------------------
NPROC=4

# -----------------------------
# Project / environment
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$PROJECT_DIR"

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate

export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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
BS=8

DEFECT_SIZE=2
TIME_SLICE=4
SPACE_SLICE=3

THERMAL_STEPS=20
STEPS=200
SWEEP=1
SWAP_EVERY=1
SWAP_PARITY=alternating
OREX_SCHEDULE=even_odd

# -----------------------------
# O-REX config
# -----------------------------
OREX_PROTOCOL_STEPS=8
OREX_MCMC_STEPS_PER_STEP=1

# -----------------------------
# Logging
# -----------------------------
LOG_EVERY=10
SEED=137
WANDB_PROJECT=neo-pt
WANDB_ENTITY=lqft-snf
RUN_NAME="${RUN_NAME:-debug_orex_obc_T${T}_L${L}_r8_bs${BS}}"
OUTPUT_DIR="${PROJECT_DIR}/results/pt_obc/${RUN_NAME}_${SLURM_JOB_ID}"

CMD=(
  torchrun
  --nnodes="$SLURM_NNODES"
  --node_rank="$SLURM_NODEID"
  --nproc_per_node="$NPROC"
  --rdzv_id="$SLURM_JOB_ID"
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
  --log-every "$LOG_EVERY"
  --seed "$SEED"
)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
