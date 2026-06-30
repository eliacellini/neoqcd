#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-neorex-eval-r8
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
NREPLICAS=$((SLURM_NNODES * NPROC))

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
# NEO-REX pretrained evaluation array
# -----------------------------
NEOREX_PROTOCOL_STEPS_LIST=(2 4 6 8)
NEOREX_PROTOCOL_STEPS="${NEOREX_PROTOCOL_STEPS_LIST[$SLURM_ARRAY_TASK_ID]}"
NEOREX_HEATBATH_STEPS_PER_STEP=1
NEOREX_WORK_MODE=logdet
NEOREX_FLOW_LR=0.0
NEOREX_GRAD_CLIP_NORM=0.0
TRAIN_STEPS=0

NEOREX_LOAD_FLOW="${NEOREX_LOAD_FLOW:-/leonardo_scratch/large/userexternal/ecellini/neoqcd/results/pt_obc/production_neorex_obc_T8_L8_r8_bs128_47347757/neorex_flow_final.pt}"
NEOREX_FLOW_CHECKPOINT_DIR="${NEOREX_FLOW_CHECKPOINT_DIR:-data/neorex_flows}"
NEOREX_FLOW_CHECKPOINT_NAME="${NEOREX_FLOW_CHECKPOINT_NAME:-global.pt}"

HYPER_SMEARING_MODE=per_link
HYPER_TIME_EMBEDDING_DIM=8
HYPER_HIDDEN_DIM=32
HYPER_DEPTH=3
HYPER_ACTIVATION=silu
HYPER_RHO_INIT=1e-3

# -----------------------------
# Logging
# -----------------------------
LOG_EVERY=10
SEED=137
WANDB_PROJECT=neo-pt
WANDB_ENTITY=lqft-snf
PRETRAINED_TAG="${PRETRAINED_TAG:-train_nsteps6_h32_depth3_sw1_job47347757}"
STUDY_NAME="${STUDY_NAME:-neorex_pretrained_eval_T${T}L${L}_d${DEFECT_SIZE}_bs${BS}_sw${SWEEP}_r${NREPLICAS}_${PRETRAINED_TAG}}"
RUN_NAME="${RUN_NAME:-nsteps${NEOREX_PROTOCOL_STEPS}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${STUDY_NAME}_${RUN_NAME}}"
OUTPUT_DIR="${PROJECT_DIR}/results/pt_obc/${STUDY_NAME}/${RUN_NAME}"

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
  --neorex-flow-checkpoint-dir "$NEOREX_FLOW_CHECKPOINT_DIR"
  --neorex-flow-checkpoint-name "$NEOREX_FLOW_CHECKPOINT_NAME"
  --neorex-load-flow "$NEOREX_LOAD_FLOW"
  --neorex-eval-flow
  --train-steps "$TRAIN_STEPS"
  --use-hyper-smearing
  --hyper-smearing-mode "$HYPER_SMEARING_MODE"
  --hyper-time-embedding-dim "$HYPER_TIME_EMBEDDING_DIM"
  --hyper-hidden-dim "$HYPER_HIDDEN_DIM"
  --hyper-depth "$HYPER_DEPTH"
  --hyper-activation "$HYPER_ACTIVATION"
  --hyper-rho-init "$HYPER_RHO_INIT"
  --hyper-rho-eps 0.0
  --hyper-rho-max 0.0
  --hyper-scale-by-delta
  --hyper-no-normalize-by-nstep
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --wandb-run-name "$WANDB_RUN_NAME"
  --run-name "$RUN_NAME"
  --output-dir "$OUTPUT_DIR"
  --main-dir "$PROJECT_DIR"
  --cfg-cache-tag "$CFG_CACHE_TAG"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
)

echo "Running NEO-REX pretrained evaluation array task:"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "NREPLICAS=$NREPLICAS"
echo "NEOREX_PROTOCOL_STEPS=$NEOREX_PROTOCOL_STEPS"
echo "NEOREX_LOAD_FLOW=$NEOREX_LOAD_FLOW"
echo "STUDY_NAME=$STUDY_NAME"
echo "RUN_NAME=$RUN_NAME"
echo "WANDB_RUN_NAME=$WANDB_RUN_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
echo "WANDB_MODE=$WANDB_MODE"
echo "W&B offline sync command after the job:"
echo "  wandb sync ${OUTPUT_DIR}/wandb/offline-run-*"

srun "${CMD[@]}"
