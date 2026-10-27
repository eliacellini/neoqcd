#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-thermalize
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

# Thermalize one chain set per rung of a defect ladder (8 rungs on 2 nodes) into the
# configuration cache data/cfgs/<run>: no swaps, rank r at the r-th rung. NEO-REX runs and
# main_cnf_rex.py --init-cfgs start from it.
# Usage: sbatch job/thermalize_defect.sh [extra main_pt_obc.py flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

ARGS=(
  --algorithm pt --D 4 --T 8 --L 8 --N 3 --beta 6 --defect-size 2 --time-slice 4 --space-slice 3
  --parameter-min 0 --parameter-max 1 --batch-size 8 --orsteps 4 --thermal-steps 1000 --steps 0
  --main-dir "$PWD" --output-dir "reports/thermalize_${SLURM_JOB_ID}"
  "$@"
)

# SLURM_NODEID must expand on each node, not on the batch host.
srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
  --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  main/main_pt_obc.py ${ARGS[*]}"
