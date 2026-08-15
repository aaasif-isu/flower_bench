#!/bin/bash
#SBATCH --job-name=t1_cifar4b
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-63%4
#SBATCH --output=track1/logs/slurm/cifar4b_%A_%a.out
#SBATCH --error=track1/logs/slurm/cifar4b_%A_%a.err

set -euo pipefail

ROOT=/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench
cd "$ROOT"

source .venv_fedprox/bin/activate

CONFIG=track1/slurm/cifar4_seeds43_46.csv

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$CONFIG")
IFS=',' read -r ALGO PARTITION_FILE LABEL SEED <<< "$ROW"

ALGO="${ALGO//$'\r'/}"
PARTITION_FILE="${PARTITION_FILE//$'\r'/}"
LABEL="${LABEL//$'\r'/}"
SEED="${SEED//$'\r'/}"

SCHEDULE="track1/schedules/cifar10_100c_10cpr_500r_seed${SEED}.json"

OUTDIR="track1/results/cifar10/final_seed${SEED}"
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
    python -u track1/cifar/run_fedavg.py \
      $COMMON_ARGS \
      --output "$OUTDIR/fedavg_${LABEL}_seed${SEED}.csv"
    ;;

  fedprox)
    python -u track1/cifar/run_fedprox.py \
      $COMMON_ARGS \
      --mu 0.01 \
      --output "$OUTDIR/fedprox_${LABEL}_seed${SEED}.csv"
    ;;

  fedavgm)
    python -u track1/cifar/run_fedavgm.py \
      $COMMON_ARGS \
      --server-momentum 0.9 \
      --server-lr 1.0 \
      --output "$OUTDIR/fedavgm_${LABEL}_seed${SEED}.csv"
    ;;

  fedadam)
    python -u track1/cifar/run_fedadam.py \
      $COMMON_ARGS \
      --server-lr 0.001 \
      --beta1 0.9 \
      --beta2 0.99 \
      --tau 1e-9 \
      --output "$OUTDIR/fedadam_${LABEL}_seed${SEED}.csv"
    ;;

  *)
    echo "Unknown algorithm: $ALGO"
    exit 1
    ;;

esac

echo "COMPLETED: $ALGO $LABEL seed=$SEED"
