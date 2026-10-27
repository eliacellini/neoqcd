#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-gpu-tests
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

# Unit tests that need a GPU (they skip elsewhere).
# Usage: sbatch job/debug_gpu_tests.sh [test modules, default: test.test_sun_mul_cuda]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate

srun python -m unittest -v "${@:-test.test_sun_mul_cuda}"
