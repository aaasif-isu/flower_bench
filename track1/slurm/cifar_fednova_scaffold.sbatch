#!/bin/bash
#SBATCH --job-name=t1_nova_scaf
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-39%4
#SBATCH --output=track1/logs/slurm/nova_scaf_%A_%a.out
#SBATCH --error=track1/logs/slurm/nova_scaf_%A_%a.err

set -euo pipefail

ROOT=/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench
cd "$ROOT"

source .venv_fedprox/bin/activate

CONFIG=track1/slurm/cifar_fednova_scaffold.csv

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

  fednova)
    python -u track1/cifar/run_fednova.py \
      $COMMON_ARGS \
      --output "$OUTDIR/fednova_${LABEL}_seed${SEED}.csv"
    ;;

  scaffold)
    python -u track1/cifar/run_scaffold.py \
      $COMMON_ARGS \
      --output "$OUTDIR/scaffold_${LABEL}_seed${SEED}.csv"
    ;;

  *)
    echo "Unknown algorithm: $ALGO"
    exit 1
    ;;

esac

echo "COMPLETED: $ALGO $LABEL seed=$SEED"
