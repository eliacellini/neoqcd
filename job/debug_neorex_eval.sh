#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-neorex-eval
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

# O-REX against NEO-REX with PG-trained defect fields, on the 8-rung defect ladder of the
# training (chains from the job/thermalize_defect.sh cache). Every swap runs K protocol
# steps, each with one MCMC update (1 HB + 4 OR) and, for NEO-REX, the flow layer.
# Smearing and residual fields are copied into their finite layer; CloverChair is a CNF.
# Usage: METHODS="orex smearing:CKPT residual:CKPT clover:CKPT" K=4 STEPS=300 sbatch job/debug_neorex_eval.sh

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
K="${K:-4}"
STEPS="${STEPS:-300}"
CNF_STEPS="${CNF_STEPS:-1}"
OUT="reports/neorex_eval_${SLURM_JOB_ID}"
mkdir -p "$OUT"

COMMON=(
  --D 4 --T 8 --L 8 --N 3 --beta 6 --defect-size 2 --time-slice 4 --space-slice 3
  --parameter-min 0 --parameter-max 1 --batch-size 8 --orsteps 4 --thermal-steps 1000
)

for entry in ${METHODS:?set METHODS}; do
  name="${entry%%:*}"; ckpt="${entry#*:}"
  case "$name" in
    orex)     ALG=(--algorithm orex --orex-protocol-steps "$K" --orex-mcmc-steps-per-step 1) ;;
    smearing) ALG=(--algorithm neorex --nf-layer-type smearing --use-hyper-smearing --hyper-smearing-mode "${SMEARING_MODE:-per_link}"
                   --hyper-hidden-dim 16 --hyper-depth 2 --hyper-scale-by-delta --hyper-no-normalize-by-nstep
                   --hyper-rho-eps 0 --hyper-rho-max 0) ;;
    residual) ALG=(--algorithm neorex --nf-layer-type residual --hyper-hidden-dim 16 --hyper-depth 2
                   --residual-n-pt 1 --residual-mask-mod 4 --residual-quadratic --residual-include-imag
                   --residual-coeff-max 0) ;;
    clover)   ALG=(--algorithm neorex --nf-layer-type cnf --cnf-checkpoint "$ckpt" --cnf-integration-steps "$CNF_STEPS") ;;
    *) echo "unknown method $name" >&2; exit 2 ;;
  esac
  if [[ "$name" != orex ]]; then
    ALG+=(--neorex-protocol-steps "$K" --neorex-heatbath-steps-per-step 1)
  fi
  if [[ "$name" == smearing || "$name" == residual ]]; then
    # The copy into the finite layer is a small CPU step.
    srun -N1 -n1 python main/convert_cnf_to_smearing_flow.py neorex "$ckpt" "$OUT/${name}_neorex.pt" -- \
      "${COMMON[@]}" "${ALG[@]}"
    ALG+=(--neorex-load-flow "$OUT/${name}_neorex.pt" --neorex-eval-flow)
  fi
  MASTER_PORT=$((29500 + (SLURM_JOB_ID + ${#name}) % 1000))
  ARGS=("${COMMON[@]}" "${ALG[@]}" --steps "$STEPS" --main-dir "$PWD" --output-dir "$OUT/${name}_K${K}")
  # SLURM_NODEID must expand on each node, not on the batch host.
  srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 --rdzv_id=${SLURM_JOB_ID}_${name} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    main/main_pt_obc.py ${ARGS[*]}"
done
