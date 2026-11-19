#!/bin/bash
#SBATCH --job-name=t1_cifar4
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=tracks/track1/logs/slurm/cifar4_%A_%a.out
#SBATCH --error=tracks/track1/logs/slurm/cifar4_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

CONFIG=tracks/track1/slurm/cifar4_seed42.csv

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$CONFIG")
IFS=',' read -r ALGO PARTITION_FILE LABEL <<< "$ROW"

SEED=42
SCHEDULE=tracks/track1/schedules/cifar10_100c_10cpr_500r_seed42.json

OUTDIR=results/track1/legacy_csv/cifar10/final_seed42
mkdir -p "$OUTDIR"

echo "========================================"
echo "JOB:       $SLURM_JOB_ID"
echo "TASK:      $SLURM_ARRAY_TASK_ID"
echo "ALGORITHM: $ALGO"
echo "PARTITION: $LABEL"
echo "SEED:      $SEED"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-unset}"
echo "========================================"

COMMON_ARGS="\
--dataset cifar10 \
--algorithm $ALGO \
--partition-file $PARTITION_FILE \
--schedule-file $SCHEDULE \
--rounds 500 \
--local-epochs 1 \
--batch-size 64 \
--lr 0.01 \
--momentum 0.9 \
--weight-decay 0.0 \
--training-seed $SEED"

case "$ALGO" in

  fedavg)
    python -u -m fl_bench.run \
      $COMMON_ARGS \
      --output "$OUTDIR/fedavg_${LABEL}_seed42.csv"
    ;;

  fedprox)
    python -u -m fl_bench.run \
      $COMMON_ARGS \
      --mu 0.01 \
      --output "$OUTDIR/fedprox_${LABEL}_seed42.csv"
    ;;

  fedavgm)
    python -u -m fl_bench.run \
      $COMMON_ARGS \
      --server-momentum 0.9 \
      --server-lr 1.0 \
      --output "$OUTDIR/fedavgm_${LABEL}_seed42.csv"
    ;;

  fedadam)
    python -u -m fl_bench.run \
      $COMMON_ARGS \
      --server-lr 0.001 \
      --beta1 0.9 \
      --beta2 0.99 \
      --tau 1e-9 \
      --output "$OUTDIR/fedadam_${LABEL}_seed42.csv"
    ;;

  *)
    echo "Unknown algorithm: $ALGO"
    exit 1
    ;;
esac

echo "COMPLETED: $ALGO $LABEL seed=$SEED"
