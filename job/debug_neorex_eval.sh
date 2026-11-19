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
# steps: the flow layer (NEO-REX) and one MCMC update (1 HB + 4 OR) at each of the K - 1
# interior points.
# Smearing and residual fields are copied into their finite layer; CloverChair is a CNF.
# Usage: METHODS="pt orex smearing:CKPT residual:CKPT clover:CKPT trex:CKPT trextrain:CKPT" K="2 4" STEPS=300 [BATCH=8] sbatch job/debug_neorex_eval.sh [extra main_pt_obc.py flags]
# K may list several values: every method runs once per value.
# "pt" is standard REX (instantaneous swaps, exact action differences); K does not enter it.
# "trex" is the smearing field with --algorithm trex: the swap is the flow alone (K layers, no MCMC);
# "trextrain" trains it further online (end-to-end in the work) for TRAIN_STEPS steps, then evaluates.
# Lattice and defect default to 8^4, L_d = 2 at time slice 4, space slice 3; set NT, NS (time and
# space extent), DSIZE, TSLICE, SSLICE, THERMAL (thermalization steps) for other geometries: the
# flow conversion below must see the same geometry as the run, so pass them here, not as flags.

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate
export OMP_NUM_THREADS=1
export NCCL_SHM_DISABLE=1
# CUDA graph captures abort if NCCL's cleanup thread polls its events meanwhile (torch 2.0):
# with --cuda-graphs that thread is not started.
case " $* " in *" --cuda-graphs "*) export NCCL_ASYNC_ERROR_HANDLING=0 ;; esac

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
# Processes (one per GPU) per node: 4, or what sbatch --gpus-per-node gives. Leonardo bills by CPU, 8 per
# GPU: with fewer GPUs per node pass --cpus-per-task 8 x GPUs too (3 GPUs and 24 CPUs cost 3/4 of a node).
NPROC="${NPROC:-${SLURM_GPUS_PER_NODE:-4}}"
NPROC="${NPROC##*:}"
K="${K:-4}"
STEPS="${STEPS:-300}"
CNF_STEPS="${CNF_STEPS:-1}"
OUT="reports/neorex_eval_${SLURM_JOB_ID}"
mkdir -p "$OUT"

COMMON=(
  --D 4 --T "${NT:-8}" --L "${NS:-8}" --N 3 --beta 6 --defect-size "${DSIZE:-2}" --time-slice "${TSLICE:-4}"
  --space-slice "${SSLICE:-3}" --parameter-min 0 --parameter-max 1 --batch-size "${BATCH:-8}" --orsteps 4
  --thermal-steps "${THERMAL:-1000}"
)

for entry in ${METHODS:?set METHODS}; do
  name="${entry%%:*}"; ckpt="${entry#*:}"
  case "$name" in
    pt)       ALG=(--algorithm pt --acceptance-rule exact) ;;
    orex)     ALG=(--algorithm orex --orex-mcmc-steps-per-step 1) ;;
    smearing) ALG=(--algorithm neorex --nf-layer-type smearing --use-hyper-smearing --hyper-smearing-mode "${SMEARING_MODE:-per_link}"
                   --hyper-hidden-dim 16 --hyper-depth 2 --hyper-scale-by-delta --hyper-no-normalize-by-nstep
                   --hyper-rho-eps 0 --hyper-rho-max 0) ;;
    residual) ALG=(--algorithm neorex --nf-layer-type residual --hyper-hidden-dim 16 --hyper-depth 2
                   --residual-n-pt 1 --residual-mask-mod 4 --residual-mode "${RESIDUAL_MODE:-shared}" --residual-quadratic --residual-include-imag
                   --residual-coeff-max 0) ;;
    clover)   ALG=(--algorithm neorex --nf-layer-type cnf --cnf-checkpoint "$ckpt" --cnf-integration-steps "$CNF_STEPS") ;;
    trex|trextrain)
              ALG=(--algorithm trex --nf-layer-type smearing --use-hyper-smearing --hyper-smearing-mode "${SMEARING_MODE:-per_link}"
                   --hyper-hidden-dim 16 --hyper-depth 2 --hyper-scale-by-delta --hyper-no-normalize-by-nstep
                   --hyper-rho-eps 0 --hyper-rho-max 0) ;;
    *) echo "unknown method $name" >&2; exit 2 ;;
  esac
  KS="$K"
  [[ "$name" == pt ]] && KS="${K%% *}"  # K does not enter standard REX: one run
  for k in $KS; do
    RUN=("${ALG[@]}")
    if [[ "$name" == trex || "$name" == trextrain ]]; then
      RUN+=(--neorex-protocol-steps "$k")
    elif [[ "$name" != orex && "$name" != pt ]]; then
      RUN+=(--neorex-protocol-steps "$k" --neorex-heatbath-steps-per-step 1)
    elif [[ "$name" == orex ]]; then
      RUN+=(--orex-protocol-steps "$k")
    fi
    if [[ "$name" == smearing || "$name" == residual || "$name" == trex || "$name" == trextrain ]]; then
      # The copy into the finite layer is a small CPU step.
      srun -N1 -n1 python main/convert_cnf_to_smearing_flow.py neorex "$ckpt" "$OUT/${name}_K${k}_neorex.pt" -- \
        "${COMMON[@]}" "${RUN[@]}"
      RUN+=(--neorex-load-flow "$OUT/${name}_K${k}_neorex.pt")
      if [[ "$name" == trextrain ]]; then
        RUN+=(--train-steps "${TRAIN_STEPS:-100}")
      else
        RUN+=(--neorex-eval-flow)
      fi
    fi
    MASTER_PORT=$((29500 + (SLURM_JOB_ID + ${#name} + 7 * k) % 1000))
    ARGS=("${COMMON[@]}" "${RUN[@]}" --steps "$STEPS" --main-dir "$PWD" --output-dir "$OUT/${name}_K${k}" "$@")
    # SLURM_NODEID must expand on each node, not on the batch host.
    srun bash -c "python -m torch.distributed.run --nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID \
      --nproc_per_node=$NPROC --rdzv_id=${SLURM_JOB_ID}_${name}_${k} --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      main/main_pt_obc.py ${ARGS[*]}"
  done
done
