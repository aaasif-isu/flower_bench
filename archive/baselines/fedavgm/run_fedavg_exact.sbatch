#!/bin/bash
#SBATCH --job-name=fedavg_exact
#SBATCH --account=jannesar-lab
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=fedavg_exact_%j.out
#SBATCH --error=fedavg_exact_%j.err

cd /lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/baselines/fedavgm
source ../../.venv/bin/activate

pkill -u "$USER" -f "ray|raylet|gcs_server|python.*fedavgm|flwr" || true
rm -rf /tmp/ray /tmp/ray_* /tmp/ray/session_*

export CUDA_VISIBLE_DEVICES="-1"
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=1

python -m fedavgm.main \
  strategy=fedavg \
  model=resnet10 \
  dataset=cifar10 \
  num_rounds=50 \
  num_clients=100 \
  server.reporting_fraction=0.1 \
  client.local_epochs=5 \
  num_cpus=8 \
  num_gpus=0 \
  | tee fedavg_resnet10_cifar10_full_50rounds_5local_10clients_CPU.log
