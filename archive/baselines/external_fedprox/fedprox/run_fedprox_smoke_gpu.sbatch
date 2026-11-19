#!/bin/bash
#SBATCH --job-name=fedprox-smoke
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fedprox_smoke_gpu_%j.log
#SBATCH --error=logs/fedprox_smoke_gpu_%j.log

set -euo pipefail

cd /lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench
source .venv/bin/activate
cd baselines/external_fedprox/fedprox

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("CUDA build:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

rm -f \
  results/external_fedprox_mu01_smoke_gpu_3r.csv \
  results/external_fedprox_mu01_smoke_gpu_3r_weights.npz

python -u direct_train.py \
  --num-clients 10 \
  --clients-per-round 5 \
  --num-rounds 3 \
  --local-epochs 1 \
  --batch-size 64 \
  --learning-rate 0.01 \
  --mu 0.1 \
  --dirichlet-alpha 0.1 \
  --client-dropout-ratio 0.0 \
  --csv-path results/external_fedprox_mu01_smoke_gpu_3r.csv \
  --checkpoint-path results/external_fedprox_mu01_smoke_gpu_3r_weights.npz
