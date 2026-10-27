#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-nemcmc-scan
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

# Pure NE-MCMC defect switching (bc 0 -> 1) for each number K of out-of-equilibrium steps,
# one evaluation after the other. The first run thermalizes the prior into the cache, the
# others load it. Usage: KS="16 32" sbatch job/debug_nemcmc_scan.sh [extra main_nemcmc_snf.py flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
KS="${KS:-16 32 64 128}"

for K in $KS; do
  MASTER_PORT=$((29500 + (SLURM_JOB_ID + K) % 1000))
  ARGS=(
    --algorithm nemcmc --domain defect --phase evaluate --D 4 --T 8 --L 8 --N 3 --beta 6
    --bc-initial 0 --bc-final 1 --defect-size 2 --time-slice 4 --space-slice 3
    --batch-size 128 --protocol-steps "$K" --thermal-steps 1000 --prior-mcmc-steps 1
    --updates-per-layer 1 --orsteps 4 --eval-samples 512
    --cfg-cache auto --cfg-cache-tag nemcmc_scan --wandb-mode disabled
    --main-dir "$PWD" --output-dir "reports/nemcmc_scan_${SLURM_JOB_ID}/K${K}" --overwrite
    "$@"
  )
  # SLURM_NODEID must expand on each node, not on the batch host.
  srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 --rdzv_id=${SLURM_JOB_ID}_K${K} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main/main_nemcmc_snf.py ${ARGS[*]}"
done
