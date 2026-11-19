# Archived Exploratory Baselines and Legacy Utilities

This directory preserves inactive, exploratory, and third-party reference code that is no longer part of the active `fl_bench` benchmarking framework. All original licenses, attribution files, configurations, and exploratory CSV outputs are retained intact for historical provenance.

## Contents

- **`archive/baselines/`**: Exploratory standalone and Flower-based baselines developed prior to the unified `fl_bench` engine:
  - `fedavgm/`: Flower `CustomFedAvgM` and standalone LEAF FEMNIST scripts.
  - `fedprox/`: Flower `FedAvgWithStragglerDrop` baseline.
  - `fedmeta/`: Standalone MAML / FedMeta experiments on LEAF FEMNIST and CIFAR-10.
  - `heterofl/`: Flower `HeteroFL` strategy and LEAF FEMNIST experiments.
  - `splitfed/`: Flower `ServerApp`/`ClientApp` SplitFed implementation (`SplitFedResNet10`) and `scripts/collect_all_results.py`.
  - `external_cama/`: Third-party FedZero / CAMA solar-constrained FL benchmark code.
  - `external_fedprox/`: Third-party TensorFlow/Keras FedProx reference implementation and CIFAR-10 ResNet-10 runs.
  - `external_hetrofl/`: Standalone PyTorch HeteroFL implementation (`HeteroResNet10_sBN_Scaler`).
- **`archive/fednova_reference/`**: Original Flower `FedNova` reference implementation on CIFAR-10 (moved from `track1/cifar/fednova_reference/`).
- **`archive/legacy_scripts/`**:
  - `run_with_round_timing.py`: Legacy subprocess stdout wrapper for timing `[ROUND N]` log lines.
- **`archive/legacy_results/`**:
  - `all_fed_results.csv`: Combined CSV table aggregated from `archive/baselines/` and `archive/fednova_reference/` by `collect_all_results.py`.
  - `external_hetrofl_direct_smoke_1r.csv`: 1-round smoke test CSV from `external_hetrofl`.
