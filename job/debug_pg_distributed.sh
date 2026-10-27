#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-pg-distributed
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

# REX + Path Gradient training with one replica per GPU: 8 replicas on 2 nodes.
# Usage: sbatch job/debug_pg_distributed.sh [extra main_cnf_rex.py flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
OUT="reports/pg_distributed_${SLURM_JOB_ID}"

ARGS=(
  --architecture hyper-residual --residual-n-pt 1 --residual-mask-mod 4
  --domain beta --D 4 --T 8 --L 8 --parameter-min 5.9 --parameter-max 6.0
  --replicas 8 --batch-size 8 --orsteps 4 --mcmc-steps 1 --warmup 5
  --epochs 20 --save-every 5 --device cuda --output-dir "$OUT"
  "$@"
)

# SLURM_NODEID must expand on each node, not on the batch host.
srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
  --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  main/main_cnf_rex.py ${ARGS[*]}"
