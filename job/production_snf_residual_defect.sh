#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-snf-residual-defect
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
# SNF config
# -----------------------------
D=4
T=8
L=8
N=3
BS="${BS:-128}"
BETA=6.0
BC_INITIAL=0.0
BC_FINAL=1.0
DEFECT_SIZE=2
TIME_SLICE=4
SPACE_SLICE=3
PROTOCOL_STEPS=6
THERMAL_STEPS="${THERMAL_STEPS:-2000}"
PRIOR_MCMC_STEPS=1
UPDATES_PER_LAYER=1
ORSTEPS=4
TRAIN_STEPS=100
EVAL_SAMPLES=128
FLOW_LR=1e-3
GRAD_CLIP_NORM=0.0

HYPER_TIME_EMBEDDING_DIM=8
HYPER_HIDDEN_DIM=32
HYPER_DEPTH=2
HYPER_ACTIVATION=silu
RESIDUAL_COEFF_INIT=1e-3
RESIDUAL_COEFF_MAX=0.0

# -----------------------------
# Logging / output
# -----------------------------
LOG_EVERY=10
SEED=137
EVAL_SEED="${EVAL_SEED:-138}"
WANDB_PROJECT=neo-nemcmc-snf
WANDB_ENTITY=lqft-snf
BASE_RUN_NAME="${RUN_NAME:-production_snf_residual_defect_beta6_bc0to1_D${D}_T${T}_L${L}_N${N}_d${DEFECT_SIZE}_t${TIME_SLICE}_s${SPACE_SLICE}_bs${BS}_K${PROTOCOL_STEPS}_h${HYPER_HIDDEN_DIM}_depth${HYPER_DEPTH}}"
RUN_ROOT="${PROJECT_DIR}/results/nemcmc_snf/${BASE_RUN_NAME}_${SLURM_JOB_ID}"
TRAIN_DIR="${RUN_ROOT}/train"
EVAL_DIR="${RUN_ROOT}/evaluate"
CFG_CACHE_DIR="${PROJECT_DIR}/data/cfgs_nemcmc_snf"
CFG_CACHE_MODE="${CFG_CACHE_MODE:-auto}"
TRAIN_CFG_CACHE_TAG="${TRAIN_CFG_CACHE_TAG:-defect_D4_T8_L8_N3_beta6p0_bc0to1_d2_t4_s3_train}"
EVAL_CFG_CACHE_TAG="${EVAL_CFG_CACHE_TAG:-defect_D4_T8_L8_N3_beta6p0_bc0to1_d2_t4_s3_eval}"

run_phase() {
  local phase="$1"
  local output_dir="$2"
  local phase_run_name="${BASE_RUN_NAME}_${phase}"
  local phase_seed="$SEED"
  local phase_cache_tag="$TRAIN_CFG_CACHE_TAG"
  if [[ "$phase" == evaluate ]]; then
    phase_seed="$EVAL_SEED"
    phase_cache_tag="$EVAL_CFG_CACHE_TAG"
  fi

  CMD=(
    python
    -m
    torch.distributed.run
    --nnodes="$SLURM_NNODES"
    --node_rank="$SLURM_NODEID"
    --nproc_per_node="$NPROC"
    --rdzv_id="${SLURM_JOB_ID}_${phase}"
    --rdzv_backend=c10d
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
    main/main_nemcmc_snf.py
    --algorithm snf
    --domain defect
    --phase "$phase"
    --D "$D"
    --T "$T"
    --L "$L"
    --N "$N"
    --beta "$BETA"
    --bc-initial "$BC_INITIAL"
    --bc-final "$BC_FINAL"
    --defect-size "$DEFECT_SIZE"
    --time-slice "$TIME_SLICE"
    --space-slice "$SPACE_SLICE"
    --batch-size "$BS"
    --protocol-steps "$PROTOCOL_STEPS"
    --thermal-steps "$THERMAL_STEPS"
    --prior-mcmc-steps "$PRIOR_MCMC_STEPS"
    --updates-per-layer "$UPDATES_PER_LAYER"
    --orsteps "$ORSTEPS"
    --flow-lr "$FLOW_LR"
    --grad-clip-norm "$GRAD_CLIP_NORM"
    # Intentional: match the hyper-conditioned NEO-PT residual job.
    --nf-architecture hyper-residual
    --hyper-time-embedding-dim "$HYPER_TIME_EMBEDDING_DIM"
    --hyper-hidden-dim "$HYPER_HIDDEN_DIM"
    --hyper-depth "$HYPER_DEPTH"
    --hyper-activation "$HYPER_ACTIVATION"
    --hyper-scale-by-delta
    --hyper-no-normalize-by-nstep
    --residual-include-imag
    --residual-quadratic
    --residual-coeff-init "$RESIDUAL_COEFF_INIT"
    --residual-coeff-max "$RESIDUAL_COEFF_MAX"
    --cfg-cache "$CFG_CACHE_MODE"
    --cfg-cache-dir "$CFG_CACHE_DIR"
    --cfg-cache-tag "$phase_cache_tag"
    --wandb-mode "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-entity "$WANDB_ENTITY"
    --wandb-run-name "$phase_run_name"
    --run-name "$phase_run_name"
    --output-dir "$output_dir"
    --main-dir "$PROJECT_DIR"
    --log-every "$LOG_EVERY"
    --seed "$phase_seed"
    --overwrite
  )
  if [[ "$phase" == train ]]; then
    CMD+=(--train-steps "$TRAIN_STEPS")
  else
    CMD+=(--eval-samples "$EVAL_SAMPLES" --checkpoint "$TRAIN_DIR/snf_checkpoint.pt")
  fi

  echo "Running $phase command:"
  printf ' %q' "${CMD[@]}"
  echo
  echo "WANDB_MODE=$WANDB_MODE"
  echo "W&B offline sync command after the job:"
  echo "  wandb sync ${output_dir}/wandb/offline-run-*"

  srun "${CMD[@]}"
}

sync_train_wandb() {
  [[ "${WANDB_MODE:-}" == offline ]] || return 0
  if ! command -v wandb >/dev/null 2>&1; then
    echo "Skipping W&B sync: wandb command not found."
    return 0
  fi

  local runs=()
  shopt -s nullglob
  runs=("$TRAIN_DIR"/wandb/offline-run-*)
  shopt -u nullglob
  if ((${#runs[@]} == 0)); then
    echo "Skipping W&B sync: no offline training runs found in $TRAIN_DIR/wandb."
    return 0
  fi

  if ! wandb sync "${runs[@]}"; then
    echo "Warning: W&B training sync failed; continuing." >&2
  fi
}

run_phase train "$TRAIN_DIR"
sync_train_wandb
run_phase evaluate "$EVAL_DIR"
