#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-snf-train
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --chdir=/leonardo_work/INF26_sft/ecellini/neoqcd

# KL (dissipated work) training of the defect hyper-smearing SNF, one layer at a time
# (blockwise), with rho per unit bc so that the network transfers to any K. Same lattice,
# defect and MCMC as job/debug_nemcmc_scan.sh. Chain jobs with RESUME=1 and the same RUN.
# Usage: RUN=name K=8 EPOCHS=1000 [RESUME=1] sbatch job/debug_snf_train.sh [extra flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
K="${K:-8}"
EPOCHS="${EPOCHS:-1000}"
OUT="reports/snf_train_${RUN:?set RUN}"

ARGS=(
  --algorithm snf --phase train --training-mode blockwise --domain defect --D 4 --T 8 --L 8 --N 3
  --beta 6 --bc-initial 0 --bc-final 1 --defect-size 2 --time-slice 4 --space-slice 3
  --batch-size 128 --protocol-steps "$K" --thermal-steps 1000 --prior-mcmc-steps 1
  --updates-per-layer 1 --orsteps 4 --cfg-cache auto --cfg-cache-tag nemcmc_scan
  --nf-architecture hyper-smearing --hyper-smearing-mode per_link --hyper-hidden-dim 16 --hyper-depth 2
  --hyper-scale-by-delta --hyper-no-normalize-by-nstep --flow-lr 1e-2
  --steps "$EPOCHS" --checkpoint-every 50 --log-every 10 --wandb-mode disabled
  --main-dir "$PWD" --output-dir "$OUT"
  "$@"
)
if [[ "${RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi

# SLURM_NODEID must expand on each node, not on the batch host.
srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
  --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  main/main_nemcmc_snf.py ${ARGS[*]}"
