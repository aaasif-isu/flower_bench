#!/bin/bash
#SBATCH --job-name=t1_natural
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --array=10-49%4
#SBATCH --output=tracks/track1/logs/slurm/natural_%A_%a.out
#SBATCH --error=tracks/track1/logs/slurm/natural_%A_%a.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

CONFIG=tracks/track1/slurm/natural_remaining50.csv

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$CONFIG")
IFS=',' read -r ALGO DATASET SEED <<< "$ROW"

ALGO="${ALGO//$'\r'/}"
DATASET="${DATASET//$'\r'/}"
SEED="${SEED//$'\r'/}"

echo "========================================"
echo "JOB:       $SLURM_JOB_ID"
echo "TASK:      $SLURM_ARRAY_TASK_ID"
echo "ALGORITHM: $ALGO"
echo "DATASET:   $DATASET"
echo "SEED:      $SEED"
echo "HOST:      $(hostname)"
echo "START:     $(date)"
echo "========================================"

case "$DATASET" in

    femnist)
        SCHEDULE="tracks/track1/schedules/femnist/femnist_500c_20cpr_1000r_seed${SEED}.json"
        OUTPUT="results/track1/legacy_csv/femnist/${ALGO}_natural_seed${SEED}.csv"
        NUM_CLIENTS=500
        ;;

    shakespeare)
        SCHEDULE="tracks/track1/schedules/shakespeare/shakespeare_100c_10cpr_1000r_seed${SEED}.json"
        OUTPUT="results/track1/legacy_csv/shakespeare/${ALGO}_natural_seed${SEED}.csv"
        NUM_CLIENTS=100
        ;;

    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

COMMON_ARGS=(
    --dataset "$DATASET"
    --algorithm "$ALGO"
    --schedule-file "$SCHEDULE"
    --rounds 1000
    --local-epochs 1
    --batch-size 32
    --lr 0.01
    --momentum 0.9
    --weight-decay 0.0
    --training-seed "$SEED"
    --eval-interval 10
    --output "$OUTPUT"
)

case "$ALGO" in

    fedavgm)
        python -u -m fl_bench.run \
            "${COMMON_ARGS[@]}" \
            --server-momentum 0.9 \
            --server-lr 1.0
        ;;

    fedprox)
        python -u -m fl_bench.run \
            "${COMMON_ARGS[@]}" \
            --mu 0.01
        ;;

    fedadam)
        if [[ "$DATASET" == "femnist" ]]; then
            SERVER_LR=0.001
        else
            SERVER_LR=0.01
        fi

        python -u -m fl_bench.run \
            "${COMMON_ARGS[@]}" \
            --server-lr "$SERVER_LR" \
            --beta1 0.9 \
            --beta2 0.99 \
            --tau 1e-9
        ;;

    fednova)
        python -u -m fl_bench.run \
            "${COMMON_ARGS[@]}"
        ;;

    scaffold)
        python -u -m fl_bench.run \
            "${COMMON_ARGS[@]}" \
            --num-clients "$NUM_CLIENTS"
        ;;

    *)
        echo "Unknown algorithm: $ALGO"
        exit 1
        ;;
esac

echo "========================================"
echo "COMPLETED: $ALGO $DATASET seed=$SEED"
echo "END:       $(date)"
echo "========================================"
