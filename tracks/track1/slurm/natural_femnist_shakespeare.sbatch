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
#SBATCH --output=tracks/track1/logs/slurm/fem_shake_%A_%a.out
#SBATCH --error=tracks/track1/logs/slurm/fem_shake_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

CONFIG=tracks/track1/slurm/natural_femnist_shakespeare.csv

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
    python -u -m fl_bench.run \
      --dataset femnist \
      --algorithm fedavg \
      --schedule-file \
      "tracks/track1/schedules/femnist/femnist_500c_20cpr_1000r_seed${SEED}.json" \
      --rounds 1000 \
      --local-epochs 1 \
      --batch-size 32 \
      --lr 0.01 \
      --momentum 0.9 \
      --weight-decay 0.0 \
      --training-seed "$SEED" \
      --eval-interval 10 \
      --output \
      "results/track1/legacy_csv/femnist/fedavg_natural_seed${SEED}.csv"
    ;;

  shakespeare)
    python -u -m fl_bench.run \
      --dataset shakespeare \
      --algorithm fedavg \
      --schedule-file \
      "tracks/track1/schedules/shakespeare/shakespeare_100c_10cpr_1000r_seed${SEED}.json" \
      --rounds 1000 \
      --local-epochs 1 \
      --batch-size 32 \
      --lr 0.01 \
      --momentum 0.9 \
      --weight-decay 0.0 \
      --training-seed "$SEED" \
      --eval-interval 10 \
      --output \
      "results/track1/legacy_csv/shakespeare/fedavg_natural_seed${SEED}.csv"
    ;;

  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;

esac

echo "COMPLETED $DATASET seed=$SEED"
