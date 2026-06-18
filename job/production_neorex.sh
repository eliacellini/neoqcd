#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-neorex-production
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
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
BS=16

DEFECT_SIZE=2
TIME_SLICE=4
SPACE_SLICE=3

THERMAL_STEPS=20
STEPS=20
SWEEP=1
SWAP_EVERY=1
SWAP_PARITY=alternating
OREX_SCHEDULE=even_odd

# -----------------------------
# NEO-REX config
# -----------------------------
NEOREX_PROTOCOL_STEPS=4
NEOREX_HEATBATH_STEPS_PER_STEP=1
NEOREX_WORK_MODE=logdet
NEOREX_FLOW_LR=1e-3
NEOREX_GRAD_CLIP_NORM=0.0
TRAIN_STEPS=20

HYPER_SMEARING_MODE=class
HYPER_TIME_EMBEDDING_DIM=8
HYPER_HIDDEN_DIM=16
HYPER_RHO_INIT=1e-3

# -----------------------------
# Logging
# -----------------------------
LOG_EVERY=10
SEED=137
WANDB_PROJECT=neo-pt
WANDB_ENTITY=lqft-snf
RUN_NAME="${RUN_NAME:-production_neorex_obc_T${T}_L${L}_r8_bs${BS}}"
OUTPUT_DIR="${PROJECT_DIR}/results/pt_obc/${RUN_NAME}_${SLURM_JOB_ID}"

CMD=(
  python
  -m
  torch.distributed.run
  --nnodes="$SLURM_NNODES"
  --node_rank="$SLURM_NODEID"
  --nproc_per_node="$NPROC"
  --rdzv_id="$SLURM_JOB_ID"
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  main/main_pt_obc.py
  --algorithm neorex
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
  --neorex-protocol-steps "$NEOREX_PROTOCOL_STEPS"
  --neorex-heatbath-steps-per-step "$NEOREX_HEATBATH_STEPS_PER_STEP"
  --neorex-work-mode "$NEOREX_WORK_MODE"
  --neorex-flow-lr "$NEOREX_FLOW_LR"
  --neorex-grad-clip-norm "$NEOREX_GRAD_CLIP_NORM"
  --train-steps "$TRAIN_STEPS"
  --use-hyper-smearing
  --hyper-smearing-mode "$HYPER_SMEARING_MODE"
  --hyper-time-embedding-dim "$HYPER_TIME_EMBEDDING_DIM"
  --hyper-hidden-dim "$HYPER_HIDDEN_DIM"
  --hyper-rho-init "$HYPER_RHO_INIT"
  --hyper-rho-eps 0.0
  --hyper-rho-max 0.0
  --hyper-scale-by-delta
  --hyper-no-normalize-by-nstep
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
echo "WANDB_MODE=$WANDB_MODE"
echo "W&B offline sync command after the job:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"

srun "${CMD[@]}"
