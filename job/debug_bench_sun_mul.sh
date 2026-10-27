#!/bin/bash
#SBATCH --account=INF26_sft
#SBATCH --job-name=neoqcd-bench-sun-mul
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

# Forward + backward time of a staple per 3x3 product formulation, on one A100.
# Usage: sbatch job/debug_bench_sun_mul.sh [extra main/bench_sun_mul.py flags]

set -euo pipefail

module load profile/deeplrn
module load cineca-ai/
source .sunenv/bin/activate

srun python main/bench_sun_mul.py "$@"
