#!/bin/bash
#SBATCH --job-name=track4-c3
#SBATCH --account=bioh-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/nvme/bioh/psaxena5/flower_bench
#SBATCH --output=/work/nvme/bioh/psaxena5/flower_bench/track4/jobs/c3_fedavg_%j.out
#SBATCH --error=/work/nvme/bioh/psaxena5/flower_bench/track4/jobs/c3_fedavg_%j.err

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/nvme/bioh/psaxena5/flower_bench/.conda_track4

echo "Running on: $(hostname)"
nvidia-smi

python track1/cifar/run_fedavg.py \
  --partition-file track4/partitions/c3_iid_seed42.npz \
  --schedule-file track4/schedules/c3_schedule_seed42.json \
  --output track4/results/c3_fedavg_50r.csv \
  --data-dir track4/data \
  --num-clients 500 \
  --rounds 50 \
  --local-epochs 1 \
  --batch-size 64 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 0 \
  --training-seed 42
