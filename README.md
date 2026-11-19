# FLBench: Modular YAML-Driven Federated Learning Benchmark Framework

**FLBench** (`fl_bench`) is a modular, reproducible, YAML-driven PyTorch framework for systematically benchmarking Federated Learning (FL) methods across statistical heterogeneity (IID, balanced Dirichlet label skew, natural LEAF user splits), communication/computation efficiency, and system scale/participation regimes.

---

## Repository Structure

```text
flower_bench/
├── fl_bench/                        # Core FL benchmarking package
│   ├── __init__.py
│   ├── config.py                    # Single-run & YAML grid ExperimentConfig + validation
│   ├── run.py                       # Unified CLI entrypoint (`python -m fl_bench.run`)
│   ├── methods/                     # Pluggable FL methods & registry
│   │   ├── __init__.py              # METHOD_REGISTRY & @register_method
│   │   ├── base.py                  # BaseMethod interface & float64 FedAvg aggregation
│   │   ├── fedavg/method.py         # FedAvg (McMahan et al., 2017)
│   │   ├── fedprox/method.py        # FedProx (Li et al., 2020)
│   │   ├── fedavgm/method.py        # FedAvgM (Hsu et al., 2019)
│   │   ├── fedadam/method.py        # FedAdam (Reddi et al., 2021)
│   │   ├── fednova/method.py        # FedNova (Wang et al., 2020)
│   │   └── scaffold/method.py       # SCAFFOLD (Karimireddy et al., 2020)
│   ├── models/                      # Model architectures & registry
│   │   ├── __init__.py              # MODEL_REGISTRY & @register_model
│   │   ├── resnet18.py              # CIFAR-adapted ResNet-18 (3x3 stem, no maxpool)
│   │   ├── femnist_cnn.py           # 62-class LEAF FEMNIST 2-layer CNN
│   │   └── shakespeare_lstm.py      # 80-character 2-layer Shakespeare LSTM
│   ├── datasets/                    # Dataset loaders, partitioners, round schedules
│   │   ├── __init__.py              # DATASET_REGISTRY & @register_dataset
│   │   ├── cifar10.py               # CIFAR-10 loader with `.npz` partition support
│   │   ├── leaf.py                  # LEAF FEMNIST and Shakespeare user split loaders
│   │   ├── partition.py             # IID & balanced Dirichlet `.npz` partition CLI
│   │   └── schedule.py              # Deterministic per-round participation JSON CLI
│   ├── engine/                      # Shared synchronous training & evaluation engine
│   │   ├── __init__.py
│   │   └── trainer.py               # train_client, evaluate, run_experiment, execute_grid
│   ├── metrics/                     # Standardized CSV schemas & track-level analysis
│   │   ├── __init__.py
│   │   ├── schema.py                # Core CSV schema & communication byte accounting
│   │   └── analysis.py              # Target efficiency, quartile summaries, rankings
│   └── utils/                       # Seeding, atomic I/O, locking, and provenance
│       ├── __init__.py
│       ├── seed.py                  # Deterministic global and per-(round, client) seeding
│       ├── io.py                    # Atomic JSON/YAML/CSV I/O, RunLock, TeeLogger
│       └── provenance.py            # SHA-256 source hashes & metadata.json builder
├── configs/                         # YAML experiment & grid configurations
│   ├── track1/                      # Track 1 configs (cifar10.yaml, femnist.yaml, shakespeare.yaml)
│   ├── track2/                      # Track 2 config (efficiency.yaml)
│   └── track4/                      # Track 4 config (cifar10_scale.yaml)
├── tracks/                          # Track-specific orchestration, schedules, & Slurm scripts
│   ├── track1/                      # Track 1 manager (`manage.py`), partitions, schedules, slurm/
│   ├── track2/                      # Track 2 manager (`manage.py`) & production manifest
│   └── track4/                      # Track 4 manager (`manage.py`), partitions, schedules, status/
├── results/                         # Centralized experiment outputs & summaries
│   ├── track1/                      # `results/track1/<dataset>/<partition>/<method>/seed_<seed>/`
│   ├── track2/                      # `results/track2/<dataset>/<partition>/<method>/seed_<seed>/` + rankings
│   └── track4/                      # `results/track4/cifar10/<group_scale>/<method>/seed_<seed>/` + analysis
├── analysis/                        # Decoupled benchmark metric calculation & figure generation
│   ├── common/                      # Shared run loader, validation, statistics, & Matplotlib styling
│   └── track1/                      # Track 1 analysis (`metrics.py`, `plots.py`, `run.py`, `config.yaml`)
├── archive/                         # Archived exploratory baselines & legacy utilities
├── external/leaf/                   # LEAF benchmark metadata
├── pyproject.toml                   # Package metadata & dependencies
└── requirements.txt                 # Python requirements
```

---

## Installation & Environment Requirements

Requires Python $\ge 3.10$ and PyTorch $\ge 2.0$:

```bash
pip install -r requirements.txt
pip install -e .
```


---

## Running Experiments from YAML (`fl_bench.run`)

### 1. Previewing Experiment Grids and Asset Status (`--plan` / `--dry-run`)

Use `--plan` (or `--dry-run`) to expand any YAML configuration and inspect:
1. Which partition, client-list, and schedule assets already exist and pass content/metadata validation (`REUSE`) vs. which missing assets will be generated automatically before training (`GENERATE`).
2. All planned experiment runs, target output folders, and reproducible CLI commands—without generating files or executing training:

```bash
# Preview all 120 runs in Track 1 CIFAR-10
python -m fl_bench.run --config configs/track1/cifar10.yaml --plan

# Preview a filtered subset (multiple methods and seeds on Dirichlet alpha=0.1)
python -m fl_bench.run \
  --config configs/track1/cifar10.yaml \
  --methods fedavg,fedprox,fedadam \
  --partition alpha_0.1 \
  --seeds 42,43 \
  --plan

# Preview Track 1 FEMNIST (30 runs), Shakespeare (30 runs), or Track 4 (360 runs)
python -m fl_bench.run --config configs/track1/femnist.yaml --plan
python -m fl_bench.run --config configs/track1/shakespeare.yaml --plan
python -m fl_bench.run --config configs/track4/cifar10_scale.yaml --plan
```

### 2. Running Single or Multiple Experiments

```bash
# Run a single experiment from YAML by selecting method, partition, and seed
python -u -m fl_bench.run \
  --config configs/track1/cifar10.yaml \
  --method fedprox \
  --partition alpha_0.1 \
  --seed 42

# Run multiple methods across multiple seeds sequentially
python -u -m fl_bench.run \
  --config configs/track1/femnist.yaml \
  --methods fedavg,fedavgm,fedadam \
  --seeds 42,43

# Run a single 0-based grid index (used by Slurm job arrays)
python -u -m fl_bench.run \
  --config configs/track1/shakespeare.yaml \
  --grid-index 0
```

### 3. Concise Working YAML Examples (Track 1)

- **CIFAR-10 (`configs/track1/cifar10.yaml`)**:
  ```yaml
  track: track1
  dataset: cifar10
  model: resnet18
  data_dir: tracks/track1/data
  num_clients: 100
  clients_per_round: 10
  partitions:
    - {type: iid, label: iid}
    - {type: dirichlet, alpha: 1.0, label: alpha_1.0}
    - {type: dirichlet, alpha: 0.3, label: alpha_0.3}
    - {type: dirichlet, alpha: 0.1, label: alpha_0.1}
  methods: [fedavg, fedprox, fedavgm, fedadam, fednova, scaffold]
  seeds: [42, 43, 44, 45, 46]
  training: {rounds: 500, local_epochs: 1, batch_size: 64, lr: 0.01, momentum: 0.9, weight_decay: 0.0}
  evaluation: {eval_interval: 1, eval_batch_size: 256}
  method_params:
    fedprox: {mu: 0.01}
    fedavgm: {server_momentum: 0.9, server_lr: 1.0}
    fedadam: {server_lr: 0.001, beta1: 0.9, beta2: 0.99, tau: 1.0e-9}
  ```
- **LEAF FEMNIST (`configs/track1/femnist.yaml`)**:
  ```yaml
  track: track1
  dataset: femnist
  model: femnist_cnn
  leaf_root: external/leaf/data/femnist
  client_file: tracks/track1/femnist/eligible_clients.txt
  num_clients: 500
  clients_per_round: 20
  partition: {type: natural, label: natural}
  methods: [fedavg, fedprox, fedavgm, fedadam, fednova, scaffold]
  seeds: [42, 43, 44, 45, 46]
  training: {rounds: 1000, local_epochs: 1, batch_size: 32, lr: 0.01, momentum: 0.9, weight_decay: 0.0}
  evaluation: {eval_interval: 10, eval_batch_size: 256}
  ```
- **LEAF Shakespeare (`configs/track1/shakespeare.yaml`)**:
  ```yaml
  track: track1
  dataset: shakespeare
  model: shakespeare_lstm
  leaf_root: external/leaf/data/shakespeare
  client_file: tracks/track1/shakespeare/eligible_clients.txt
  num_clients: 100
  clients_per_round: 10
  partition: {type: natural, label: natural}
  methods: [fedavg, fedprox, fedavgm, fedadam, fednova, scaffold]
  seeds: [42, 43, 44, 45, 46]
  training: {rounds: 1000, local_epochs: 1, batch_size: 32, lr: 0.01, momentum: 0.9, weight_decay: 0.0, max_grad_norm: 10.0}
  evaluation: {eval_interval: 10, eval_batch_size: 256}
  method_params:
    fedadam: {server_lr: 0.01, beta1: 0.9, beta2: 0.99, tau: 1.0e-9}
  ```


---

## Track Orchestration & Slurm Submission

Each benchmark track provides a `manage.py` CLI under `tracks/track{1,2,4}/`:

```bash
# Track 1: Preview grid, generate CSV/JSON manifests, or render/submit Slurm array script
python tracks/track1/manage.py plan
python tracks/track1/manage.py manifest --output tracks/track1/slurm/track1_manifest.csv
python tracks/track1/manage.py sbatch --config configs/track1/cifar10.yaml

# Track 2: Preview efficiency grid, generate manifest, or compute efficiency rankings
python tracks/track2/manage.py plan
python tracks/track2/manage.py analyze

# Track 4: Preview 360-run scalability grid, verify smoke tests, or summarize completed runs
python tracks/track4/manage.py plan
python tracks/track4/manage.py plan --smoke
python tracks/track4/manage.py check-smoke
python tracks/track4/manage.py summarize
```

---

## Benchmark Analysis & Visualization (`analysis/`)

Analysis is completely decoupled from training. It reads canonical run directories under `results/`, validates configurations and evaluation checkpoints, computes standardized benchmark metrics, and generates publication-quality PDF/PNG figures:

```bash
# Run complete Track 1 analysis (metrics + summary tables + figures)
python -m analysis.track1.run --config analysis/track1/config.yaml

# Compute Track 1 metric CSVs and summary tables only
python -m analysis.track1.run --config analysis/track1/config.yaml --metrics-only

# Generate Track 1 figures only (from existing CSVs in analysis/track1/outputs/)
python -m analysis.track1.run --config analysis/track1/config.yaml --plots-only
```

---

## Centralized Results & Reproducibility

Completed runs are organized hierarchically by track, dataset, partition, method, and seed:

```text
results/
├── track1/
│   ├── cifar10/<partition>/<method>/seed_<seed>/
│   │   ├── metrics.csv      # Per-round evaluation & training metrics
│   │   ├── config.yaml      # Fully resolved ExperimentConfig
│   │   ├── metadata.json    # Run ID, seed, code hashes, completion & finite-metric flags
│   │   └── run.log          # Execution console output (when available)
│   ├── femnist/natural/<method>/seed_<seed>/
│   ├── shakespeare/natural/<method>/seed_<seed>/
│   └── legacy_csv/          # Preserved original Track 1 CSV hierarchy
├── track2/
│   ├── <dataset>/<partition>/<method>/seed_<seed>/
│   ├── raw/                 # Preserved Track 2 raw accounting CSVs
│   ├── smoke/               # Preserved 3-round smoke rankings
│   └── production_summary/  # Multi-seed target efficiency & rankings (`rankings.csv`)
└── track4/
    ├── cifar10/<group>_<N>c_<M>cpr/<method>/seed_<seed>/
    ├── production/          # Preserved Track 4 production CSVs & partial runs
    ├── smoke/               # Preserved 24 Track 4 smoke CSVs
    └── analysis/            # Aggregated summary tables (`completed_run_summary.csv`)
```

Completed runs are protected against accidental overwriting by `verify_metrics_csv()` and `RunLock`; pass `--overwrite` to `fl_bench.run` only when intentionally re-running an experiment.

---

## Extending FLBench

1. **Adding a New FL Method**:
   Create `fl_bench/methods/<my_method>/method.py`, subclass `BaseMethod`, and register it in `fl_bench/methods/__init__.py` (or decorate with `@register_method("my_method")`):
   ```python
   from typing import Dict, Sequence, Set, Tuple
   from fl_bench.methods import BaseMethod, ClientResult, StateDict, fedavg_aggregate, register_method

   @register_method("my_method")
   class MyMethod(BaseMethod):
       extra_csv_fields = ("custom_metric",)

       def aggregate(
           self,
           round_id: int,
           global_state: StateDict,
           client_results: Sequence[ClientResult],
           trainable_keys: Set[str],
       ) -> Tuple[StateDict, Dict[str, float]]:
           return fedavg_aggregate(client_results), {"custom_metric": 0.0}
   ```
2. **Adding a New Dataset**:
   Implement a loader returning `FederatedData` in `fl_bench/datasets/`, decorate with `@register_dataset("my_dataset")`, and add default settings to `DATASET_DEFAULTS` in `fl_bench/config.py`.
3. **Adding a New Model**:
   Implement a constructor in `fl_bench/models/`, decorate with `@register_model("my_model")`, and import it in `fl_bench/models/__init__.py`.

---

## Known Limitations & Preserved Historical Results

1. **Corrected SCAFFOLD (`momentum=0.0`, Option II) vs. Historical Legacy Runs (`momentum=0.9`)**:
   Per Karimireddy et al. (ICML 2020, Algorithm 1), SCAFFOLD's Option II control-variate update $\Delta c_i = -c + \frac{1}{K \eta_l}(x - y_i)$ requires vanilla local SGD ($\text{momentum}=0.0$). All YAML configs and `ExperimentConfig.resolve_defaults()` now set `momentum: 0.0` and `version: v2_option2_mom0` specifically for `scaffold` while keeping `momentum: 0.9` for all other methods. Historical `momentum=0.9` SCAFFOLD results are preserved under `results/<track>/<dataset>/<partition>/scaffold_legacy_mom0.9/` (with `"method_version": "legacy_option2_mom0.9"` in `metadata.json`) as well as in `results/track1/legacy_csv/` and `results/track4/production/scaffold_*.csv`. New corrected SCAFFOLD runs use versioned run IDs (`scaffold_v2_...`) and write to `results/<track>/<dataset>/<partition>/scaffold/seed_<seed>/` without overwriting historical outputs.
2. **Balanced CIFAR-10 Partitions**:
   `fl_bench/datasets/partition.py` generates strictly balanced client sample counts ($50,000 / N$), so every CIFAR-10 client performs the same number of local steps $\tau_i$, making `fednova` mathematically equivalent to `fedavg` on CIFAR-10 (whereas `fednova` differs on unbalanced natural LEAF `femnist` and `shakespeare` splits).
3. **Raw Dataset Files**:
   Raw CIFAR-10 (`tracks/track1/data/`) and LEAF (`external/leaf/data/{femnist,shakespeare}/data/`) binaries/JSONs are excluded via `.gitignore` and must be downloaded/placed at those paths before running new training jobs.
4. **Synchronous Single-Process Execution**:
   `fl_bench` executes selected clients sequentially within a synchronous round loop on a single GPU/CPU. Asynchronous FL and mid-round model checkpoint resumption are not currently supported.

---

## License

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
