#!/bin/bash
#SBATCH --job-name=t1_fem_shake
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-9%4
#SBATCH --output=track1/logs/slurm/fem_shake_%A_%a.out
#SBATCH --error=track1/logs/slurm/fem_shake_%A_%a.err

set -euo pipefail

ROOT=/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench
cd "$ROOT"

source .venv_fedprox/bin/activate

CONFIG=track1/slurm/natural_femnist_shakespeare.csv

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$CONFIG")
IFS=',' read -r DATASET SEED <<< "$ROW"

DATASET="${DATASET//$'\r'/}"
SEED="${SEED//$'\r'/}"

echo "================================"
echo "JOB:     $SLURM_JOB_ID"
echo "TASK:    $SLURM_ARRAY_TASK_ID"
echo "DATASET: $DATASET"
echo "SEED:    $SEED"
echo "================================"

case "$DATASET" in

  femnist)
    python -u track1/femnist/run_fedavg.py \
      --schedule-file \
      "track1/schedules/femnist/femnist_500c_20cpr_1000r_seed${SEED}.json" \
      --rounds 1000 \
      --local-epochs 1 \
      --batch-size 32 \
      --lr 0.01 \
      --momentum 0.9 \
      --weight-decay 0.0 \
      --training-seed "$SEED" \
      --eval-interval 10 \
      --output \
      "track1/results/femnist/fedavg_natural_seed${SEED}.csv"
    ;;

  shakespeare)
    python -u track1/shakespeare/run_fedavg.py \
      --schedule-file \
      "track1/schedules/shakespeare/shakespeare_100c_10cpr_1000r_seed${SEED}.json" \
      --rounds 1000 \
      --local-epochs 1 \
      --batch-size 32 \
      --lr 0.01 \
      --momentum 0.9 \
      --weight-decay 0.0 \
      --training-seed "$SEED" \
      --eval-interval 10 \
      --output \
      "track1/results/shakespeare/fedavg_natural_seed${SEED}.csv"
    ;;

  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;

esac

echo "COMPLETED $DATASET seed=$SEED"
