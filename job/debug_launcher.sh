#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-launch-debug
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:10:00
#SBATCH --chdir=/leonardo_scratch/large/userexternal/ecellini/neoqcd

set -euo pipefail

NPROC=1
PROJECT_DIR="${PROJECT_DIR:-/leonardo_scratch/large/userexternal/ecellini/neoqcd}"

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate

export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/results/wandb_offline/launcher_${SLURM_JOB_ID}}"
mkdir -p "$WANDB_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== SLURM ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NNODES=$SLURM_NNODES"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo

echo "=== ENV ==="
echo "PATH=$PATH"
echo "which python=$(which python)"
echo "which torchrun=$(which torchrun || true)"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB__SERVICE_WAIT=$WANDB__SERVICE_WAIT"
echo "WANDB_INIT_TIMEOUT=$WANDB_INIT_TIMEOUT"
echo "W&B offline sync command after the job:"
echo "  wandb sync ${WANDB_DIR}/wandb/offline-run-* ${WANDB_DIR}/offline-run-*"
python -V
python -c "import sys, torch, neoqcd; print('python', sys.executable); print('torch', torch.__version__); print('neoqcd', neoqcd.__file__)"
echo

echo "=== WANDB SINGLE PROCESS ==="
wandb status || true
python -c "import os, socket, time, wandb; print('wandb single start', socket.gethostname(), os.environ.get('WANDB_DIR'), flush=True); t=time.time(); run=wandb.init(project='neo-pt', entity='lqft-snf', name='launcher-single-'+os.environ['SLURM_JOB_ID'], dir=os.environ['WANDB_DIR']); print('wandb single ok', run.url, 'seconds', time.time()-t, flush=True); run.finish()"
echo

echo "=== SRUN HOSTS ==="
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 hostname
echo

CMD=(
  python
  -m
  torch.distributed.run
  --nnodes="$SLURM_NNODES"
  --node_rank="$SLURM_NODEID"
  --nproc_per_node="$NPROC"
  --rdzv_id="${SLURM_JOB_ID}_launcher_debug"
  --rdzv_backend=c10d
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"
  -c
  "exec(\"\"\"\nimport os, socket, sys, time, torch, neoqcd\nrank = os.environ.get('RANK')\nprint('HELLO', 'host='+socket.gethostname(), 'rank='+str(rank), 'local_rank='+str(os.environ.get('LOCAL_RANK')), 'world='+str(os.environ.get('WORLD_SIZE')), 'python='+sys.executable, 'cuda='+str(torch.cuda.is_available()), 'device_count='+str(torch.cuda.device_count()), flush=True)\nif str(rank) == '0':\n    import wandb\n    print('wandb distributed rank0 start', os.environ.get('WANDB_DIR'), flush=True)\n    t = time.time()\n    run = wandb.init(project='neo-pt', entity='lqft-snf', name='launcher-dist-'+os.environ['SLURM_JOB_ID'], dir=os.environ['WANDB_DIR'])\n    print('wandb distributed rank0 ok', run.url, 'seconds', time.time()-t, flush=True)\n    run.finish()\nelse:\n    print('wandb distributed skip rank', rank, flush=True)\n\"\"\")"
)

echo
echo "=== TORCH DISTRIBUTED RUN ==="
printf ' %q' "${CMD[@]}"
echo
srun "${CMD[@]}"
