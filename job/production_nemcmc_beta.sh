#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-nemcmc-beta
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
# NEMCMC config
# -----------------------------
D=4
T=16
L=16
N=3
BS="${BS:-16}"
BETA_INITIAL=6.02
BETA_FINAL=6.178
PROTOCOL_STEPS=6
THERMAL_STEPS="${THERMAL_STEPS:-2000}"
PRIOR_MCMC_STEPS=1
UPDATES_PER_LAYER=1
ORSTEPS=4
EVAL_SAMPLES=128

# -----------------------------
# Logging / output
# -----------------------------
LOG_EVERY=10
SEED="${SEED:-138}"
WANDB_PROJECT=neo-nemcmc-snf
WANDB_ENTITY=lqft-snf
BASE_RUN_NAME="${RUN_NAME:-production_nemcmc_beta_D${D}_T${T}_L${L}_N${N}_bs${BS}_b6p02to6p178_K${PROTOCOL_STEPS}}"
RUN_ROOT="${PROJECT_DIR}/results/nemcmc_snf/${BASE_RUN_NAME}_${SLURM_JOB_ID}"
EVAL_DIR="${RUN_ROOT}/evaluate"
CFG_CACHE_DIR="${PROJECT_DIR}/data/cfgs_nemcmc_snf"
CFG_CACHE_MODE="${CFG_CACHE_MODE:-auto}"
CFG_CACHE_TAG="${CFG_CACHE_TAG:-beta_D4_T16_L16_N3_p6p02_eval}"

# Thermalisation and cache replace training preparation for pure NEMCMC.
CMD=(
  python
  -m
  torch.distributed.run
  --nnodes="$SLURM_NNODES"
  --node_rank="$SLURM_NODEID"
  --nproc_per_node="$NPROC"
  --rdzv_id="${SLURM_JOB_ID}_evaluate"
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  main/main_nemcmc_snf.py
  --algorithm nemcmc
  --domain beta
  --phase evaluate
  --D "$D"
  --T "$T"
  --L "$L"
  --N "$N"
  --beta-initial "$BETA_INITIAL"
  --beta-final "$BETA_FINAL"
  --batch-size "$BS"
  --protocol-steps "$PROTOCOL_STEPS"
  --thermal-steps "$THERMAL_STEPS"
  --prior-mcmc-steps "$PRIOR_MCMC_STEPS"
  --updates-per-layer "$UPDATES_PER_LAYER"
  --orsteps "$ORSTEPS"
  --eval-samples "$EVAL_SAMPLES"
  --cfg-cache "$CFG_CACHE_MODE"
  --cfg-cache-dir "$CFG_CACHE_DIR"
  --cfg-cache-tag "$CFG_CACHE_TAG"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-entity "$WANDB_ENTITY"
  --wandb-run-name "${BASE_RUN_NAME}_evaluate"
  --run-name "${BASE_RUN_NAME}_evaluate"
  --output-dir "$EVAL_DIR"
  --main-dir "$PROJECT_DIR"
  --log-every "$LOG_EVERY"
  --seed "$SEED"
  --overwrite
)

echo "Running evaluate command:"
printf ' %q' "${CMD[@]}"
echo
echo "WANDB_MODE=$WANDB_MODE"

srun "${CMD[@]}"
