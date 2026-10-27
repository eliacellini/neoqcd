#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-snf-eval
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

# SNF evaluation of PG-trained defect fields (bc 0 -> 1), with the settings of
# job/debug_nemcmc_scan.sh so that the pure NE-MCMC numbers are the baseline.
# Smearing and residual fields are copied into their finite layer (exact logdet);
# CloverChair is integrated as a CNF.
# Usage: FIELDS="smearing:CKPT residual:CKPT clover:CKPT" K=32 sbatch job/debug_snf_eval.sh [extra flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
K="${K:-32}"
CNF_STEPS="${CNF_STEPS:-4}"
OUT="reports/snf_eval_${SLURM_JOB_ID}"
mkdir -p "$OUT"

COMMON=(
  --algorithm snf --domain defect --D 4 --T 8 --L 8 --N 3 --beta 6 --bc-initial 0 --bc-final 1
  --defect-size 2 --time-slice 4 --space-slice 3 --batch-size 128 --protocol-steps "$K"
  --thermal-steps 1000 --prior-mcmc-steps 1 --updates-per-layer 1 --orsteps 4
  --cfg-cache auto --cfg-cache-tag nemcmc_scan
)
EVAL=(--phase evaluate --eval-samples 512 --wandb-mode disabled --main-dir "$PWD" --overwrite)

for entry in ${FIELDS:?set FIELDS}; do
  name="${entry%%:*}"; ckpt="${entry#*:}"
  case "$name" in
    smearing) NET=(--nf-architecture hyper-smearing --hyper-smearing-mode "${SMEARING_MODE:-per_link}" --hyper-hidden-dim 16
                   --hyper-depth 2 --hyper-no-normalize-by-nstep) ;;
    residual) NET=(--nf-architecture hyper-residual --hyper-hidden-dim 16 --hyper-depth 2
                   --residual-n-pt 1 --residual-mask-mod 4) ;;
    clover)   NET=(--nf-architecture cnf --cnf-checkpoint "$ckpt" --cnf-integration-steps "$CNF_STEPS") ;;
    *) echo "unknown field $name" >&2; exit 2 ;;
  esac
  FLOW=()
  if [[ "$name" != clover ]]; then
    # The copy into the finite layer is a small CPU step.
    srun -N1 -n1 python main/convert_cnf_to_smearing_flow.py snf "$ckpt" "$OUT/${name}_snf.pt" -- \
      "${COMMON[@]}" "${NET[@]}" "$@"
    FLOW=(--checkpoint "$OUT/${name}_snf.pt")
  fi
  MASTER_PORT=$((29500 + (SLURM_JOB_ID + ${#name}) % 1000))
  ARGS=("${COMMON[@]}" "${NET[@]}" "${FLOW[@]}" "${EVAL[@]}" --output-dir "$OUT/${name}_K${K}" "$@")
  # SLURM_NODEID must expand on each node, not on the batch host.
  srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 --rdzv_id=${SLURM_JOB_ID}_${name} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main/main_nemcmc_snf.py ${ARGS[*]}"
done
