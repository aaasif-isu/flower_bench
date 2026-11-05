#!/bin/bash
#SBATCH --account=bioh-delta-gpu
#SBATCH --qos=bioh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
module load pytorch-conda/2.8
cd "${SLURM_SUBMIT_DIR}"

task_mode=${1:?Expected smoke or production}
task_group=${2:-}
task_python=${TRACK4_PYTHON:-$(command -v python)}

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [[ "$task_mode" == smoke ]]; then
    srun "$task_python" -u track4/jobs/manage.py run --smoke \
        --index "$SLURM_ARRAY_TASK_ID"
elif [[ "$task_mode" == production ]]; then
    srun "$task_python" -u track4/jobs/manage.py run \
        --group "$task_group" --index "$SLURM_ARRAY_TASK_ID"
else
    echo "Unknown mode: $task_mode" >&2
    exit 1
fi
