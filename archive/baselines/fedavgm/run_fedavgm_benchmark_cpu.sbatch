#!/bin/bash
#SBATCH --job-name=fedavgm_cpu_bench
#SBATCH --partition=nova
#SBATCH --account=jannesar-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=fedavgm_cpu_bench_%j.out
#SBATCH --error=fedavgm_cpu_bench_%j.err

cd /lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/baselines/fedavgm

source .venv/bin/activate

pkill -u $USER -f "ray|flwr" 2>/dev/null || true
ray stop --force 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

export TMPDIR=/tmp/aadi_ray_${SLURM_JOB_ID}
export RAY_TMPDIR=/tmp/aadi_ray_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

echo "===== JOB INFO ====="
hostname
date
which python
python --version
python -c "import flwr; print('flwr', flwr.__version__)"
python -c "import ray; print('ray', ray.__version__)"
python -c "import pkg_resources; print('pkg_resources OK')"

echo "===== START FEDAVG BENCHMARK ====="

python -m fedavgm.main \
  strategy=fedavg \
  model=resnet10 \
  dataset=cifar10 \
  num_clients=100 \
  num_rounds=50 \
  server.reporting_fraction=0.1 \
  fraction_evaluate=0 \
  client.local_epochs=5 \
  client.batch_size=64 \
  client.lr=0.01 \
  num_cpus=24 \
  num_gpus=0

echo "===== DONE ====="
date

echo "===== LATEST CSV ====="
find outputs -name "fedavgm_round_metrics.csv" -printf "%T@ %p\n" | sort -n | tail -1
