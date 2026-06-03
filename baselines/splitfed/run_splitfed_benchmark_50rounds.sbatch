#!/bin/bash
#SBATCH --job-name=splitfed_cifar10_r50
#SBATCH --account=jannesar-lab
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=splitfed_benchmark_%j.out
#SBATCH --error=splitfed_benchmark_%j.err

echo "===== JOB INFO ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

cd /lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/baselines/splitfed
source .venv_splitfed/bin/activate

echo "===== PYTHON/FLWR INFO ====="
which python
python --version
which flwr
flwr --version

echo "===== GPU INFO ====="
nvidia-smi || true

echo "===== CLEAN OLD RAY ====="
pkill -u $USER -f ray || true
rm -rf /tmp/ray

mkdir -p results

export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

echo "===== STARTING SPLITFED BENCHMARK ====="

flwr run . local-simulation \
  --run-config "
method='SplitFed'
num-clients=100
clients-per-round=10
num-server-rounds=50
batch-size=64
local-epochs=5
train-fraction=1.0
csv-path='results/splitfed_cifar10_resnet10_benchmark_50rounds.csv'
" \
  --federation-config "options.num-supernodes=10" \
  --stream

echo "===== RUN FINISHED ====="
echo "End time: $(date)"

echo "===== CSV CHECK ====="
ls -lh results/splitfed_cifar10_resnet10_benchmark_50rounds.csv || true
wc -l results/splitfed_cifar10_resnet10_benchmark_50rounds.csv || true
tail -n 5 results/splitfed_cifar10_resnet10_benchmark_50rounds.csv || true
