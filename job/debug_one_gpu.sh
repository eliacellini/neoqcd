#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-one-gpu
#SBATCH -e reports/errors_%x_%j
#SBATCH -o reports/output_%x_%j
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:15:00
#SBATCH --chdir=/leonardo_work/INF26_sft/ecellini/neoqcd

# Run one Python script on one A100 (benchmarks, profiles).
# Usage: sbatch job/debug_one_gpu.sh main/<script>.py [flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate

srun python "$@"
