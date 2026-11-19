"""Unified YAML-driven configuration system for FL benchmark experiments and grids."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "cifar10": {
        "model": "resnet18",
        "data_dir": "tracks/track1/data",
        "partition_type": "iid",
        "num_clients": 100,
        "clients_per_round": 10,
        "rounds": 500,
        "local_epochs": 1,
        "batch_size": 64,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "eval_interval": 1,
        "max_grad_norm": None,
        "server_lr_fedadam": 0.001,
        "include_num_fit_examples": False,
        "include_perplexity": False,
    },
    "femnist": {
        "model": "femnist_cnn",
        "leaf_root": "external/leaf/data/femnist",
        "client_file": "tracks/track1/femnist/eligible_clients.txt",
        "partition_type": "natural",
        "num_clients": 500,
        "clients_per_round": 20,
        "rounds": 1000,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "eval_interval": 10,
        "max_grad_norm": None,
        "server_lr_fedadam": 0.001,
        "include_num_fit_examples": True,
        "include_perplexity": False,
    },
    "shakespeare": {
        "model": "shakespeare_lstm",
        "leaf_root": "external/leaf/data/shakespeare",
        "client_file": "tracks/track1/shakespeare/eligible_clients.txt",
        "partition_type": "natural",
        "num_clients": 100,
        "clients_per_round": 10,
        "rounds": 1000,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "eval_interval": 10,
        "max_grad_norm": 10.0,
        "server_lr_fedadam": 0.01,
        "include_num_fit_examples": True,
        "include_perplexity": True,
    },
}


# Canonical paper/reference default hyperparameters per algorithm.
# Dataset-specific overrides (e.g., LEAF sequence/CNN local learning rates) are stored under `by_dataset`.
METHOD_PAPER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "fedavg": {
        "reference": "McMahan et al., AISTATS 2017",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "server_lr": 1.0,
    },
    "fedprox": {
        "reference": "Li et al., MLSys 2020",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "mu": 0.01,
        "server_lr": 1.0,
    },
    "fedavgm": {
        "reference": "Hsu et al., arXiv:1909.06335 (2019)",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "server_momentum": 0.9,
        "server_lr": 1.0,
    },
    "fedadam": {
        "reference": "Reddi et al., ICLR 2021",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "server_lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.99,
        "tau": 1e-9,
        "by_dataset": {
            "shakespeare": {"server_lr": 0.01},
        },
    },
    "fednova": {
        "reference": "Wang et al., NeurIPS 2020",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "server_lr": 1.0,
    },
    "scaffold": {
        "reference": "Karimireddy et al., ICML 2020 (Option II)",
        "version": "v2_option2_mom0",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "server_lr": 1.0,
    },
    "fedspeed": {
        "reference": "Sun et al., ICLR 2023 (woodenchild95/FL-Simulator)",
        "lr": 0.1,
        "lr_decay": 0.998,
        "momentum": 0.0,
        "weight_decay": 0.001,
        "max_grad_norm": 10.0,
        "server_lr": 1.0,
        "rho": 0.1,
        "lamb": 0.1,
        "fedspeed_alpha": 1.0,
        "fedspeed_prox_mode": "paper_symmetric",
        "fedspeed_comm_mode": "reference_simulator",
        "fedspeed_freeze_bn_on_perturb": True,
        "use_ri": False,
        "ri_beta": 0.1,
    },
    "fedlesam": {
        "reference": "Fan et al., ICML 2024 (MediaBrain-SJTU/FedLESAM)",
        "lr": 0.1,
        "lr_decay": 0.998,
        "momentum": 0.0,
        "weight_decay": 0.001,
        "max_grad_norm": 10.0,
        "server_lr": 1.0,
        "rho": 0.1,
        "lamb": 0.1,
        "fedlesam_variant": "base",
        "fedlesam_norm_mode": "reference_sum_l2",
        "fedlesam_perturbation_target": "server_global_update",
        "fedlesam_prox_mode": "reference_simulator",
    },
    "fedgps": {
        "reference": "Yang et al., NeurIPS 2025 (visitworld123/FedGPS)",
        "lr": 0.01,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 1.0e-5,
        "max_grad_norm": 10.0,
        "server_lr": 1.0,
        "rho": 0.5,
        "fedgps_use_ema": False,
        "fedgps_ema_decay": 0.9,
        "fedgps_use_minus": True,
        "fedgps_require_prev_delta": True,
        "fedgps_use_minus_lesam": False,
        "fedgps_weighted_sum": False,
        "fedgps_label_shift": True,
        "fedgps_surrogate_mode": "gaussian",
        "fedgps_surrogate_alpha": 1.0,
        "fedgps_align_loss_weight": 0.1,
        "fedgps_proto_align_method": "mse",
        "fedgps_surrogate_num": 10,
        "by_dataset": {
            "femnist": {"fedgps_surrogate_num": 62},
            "shakespeare": {
                "fedgps_label_shift": False,
                "fedgps_surrogate_mode": "none",
                "fedgps_surrogate_num": 80,
            },
        },
    },
    "fedwmsam": {
        "reference": "Li et al., NeurIPS 2025 (Huang-Yongzhi/NeurlPS_FedWMSAM)",
        "lr": 0.1,
        "lr_decay": 0.998,
        "momentum": 0.0,
        "weight_decay": 0.001,
        "max_grad_norm": 10.0,
        "server_lr": 1.0,
        "rho": 0.01,
        "wmsam_alpha": 0.1,
        "wmsam_norm_mode": "reference_sum_l2",
    },
    "heterofl": {
        "reference": "Diao et al., ICLR 2021 (diaoenmao/HeteroFL)",
        "lr": 0.1,
        "lr_decay": 1.0,
        "momentum": 0.9,
        "weight_decay": 5.0e-4,
        "max_grad_norm": 1.0,
        "server_lr": 1.0,
        "heterofl_split_mode": "fix",
        "heterofl_model_mode": "a2-b2-c2-d2-e2",
        "heterofl_scaler": True,
        "heterofl_mask": True,
        "heterofl_sbn": True,
        "by_dataset": {
            "femnist": {"lr": 0.01},
        },
    },
}


def get_method_paper_defaults(method: str, dataset: str = "cifar10") -> Dict[str, Any]:
    """Return a copy of the paper/reference default hyperparameters for `(method, dataset)`."""
    m_key = str(method).strip().lower()
    spec = dict(METHOD_PAPER_DEFAULTS.get(m_key, {}))
    by_ds = spec.pop("by_dataset", {})
    if isinstance(by_ds, dict) and dataset in by_ds and isinstance(by_ds[dataset], dict):
        spec.update(by_ds[dataset])
    return spec


def format_alpha_label(alpha: float) -> str:
    """Return canonical directory label for Dirichlet alpha (e.g. `alpha_0.1`, `alpha_1.0`)."""
    val = float(alpha)
    if val.is_integer():
        return f"alpha_{val:.1f}"
    return f"alpha_{val:g}"


def _legacy_cifar_alpha_tag(alpha: float) -> str:
    """Return legacy Track 1 CIFAR-10 filename tag (`alpha1`, `alpha03`, `alpha01`)."""
    if math.isclose(alpha, 1.0):
        return "alpha1"
    if math.isclose(alpha, 0.3):
        return "alpha03"
    if math.isclose(alpha, 0.1):
        return "alpha01"
    if math.isclose(alpha, 0.5):
        return "dirichlet05"
    return f"alpha{str(alpha).replace('.', '')}"


@dataclass
class ExperimentConfig:
    """Complete configuration for a single Federated Learning experiment run."""

    track: str = "track1"
    experiment_name: str = ""
    run_id: str = ""
    grid_cell: str = ""

    method: str = "fedavg"
    method_version: Optional[str] = None
    dataset: str = "cifar10"
    model: Optional[str] = None

    # Partitioning
    partition_type: Optional[str] = None
    partition_label: Optional[str] = None
    alpha: Optional[float] = None
    partition_seed: Optional[int] = None
    schedule_seed: Optional[int] = None

    # Paths
    schedule_file: str = ""
    partition_file: Optional[str] = None
    output_dir: str = "results"
    run_dir: Optional[str] = None
    output: str = ""
    data_dir: Optional[str] = None
    leaf_root: Optional[str] = None
    client_file: Optional[str] = None

    # FL system parameters
    num_clients: Optional[int] = None
    clients_per_round: Optional[int] = None
    participation_ratio: Optional[float] = None
    rounds: Optional[int] = None
    eval_interval: Optional[int] = None
    eval_batch_size: int = 256
    training_seed: int = 42

    # Local client optimizer parameters
    local_epochs: Optional[int] = None
    local_steps: Optional[int] = None
    local_budget_label: Optional[str] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    lr_decay: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    max_grad_norm: Optional[float] = None
    hyperparameter_source: str = "paper_default"
    hyperparameter_overrides: List[str] = field(default_factory=list)

    # Method-specific hyperparameters
    mu: float = 0.01
    server_momentum: float = 0.9
    server_lr: Optional[float] = None
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-9
    rho: float = 0.1
    lamb: float = 0.1
    fedspeed_alpha: float = 1.0
    fedspeed_prox_mode: str = "paper_symmetric"
    fedspeed_comm_mode: str = "reference_simulator"
    fedspeed_freeze_bn_on_perturb: bool = True
    use_ri: bool = False
    ri_beta: float = 0.1
    fedlesam_variant: str = "base"
    fedlesam_norm_mode: str = "reference_sum_l2"
    fedlesam_perturbation_target: str = "server_global_update"
    fedlesam_prox_mode: str = "reference_simulator"
    wmsam_alpha: float = 0.5
    wmsam_norm_mode: str = "reference_sum_l2"
    fedgps_use_ema: bool = False
    fedgps_ema_decay: float = 0.9
    fedgps_use_minus: bool = True
    fedgps_require_prev_delta: bool = True
    fedgps_use_minus_lesam: bool = False
    fedgps_weighted_sum: bool = False
    fedgps_label_shift: bool = True
    fedgps_surrogate_mode: str = "gaussian"
    fedgps_surrogate_path: Optional[str] = None
    fedgps_surrogate_batch_size: Optional[int] = None
    fedgps_surrogate_alpha: float = 1.0
    fedgps_align_loss_weight: float = 0.1
    fedgps_proto_align_method: str = "mse"
    fedgps_surrogate_num: int = 10
    heterofl_split_mode: str = "fix"
    heterofl_model_mode: str = "a2-b2-c2-d2-e2"
    heterofl_scaler: bool = True
    heterofl_mask: bool = True
    heterofl_sbn: bool = True
    heterofl_custom_rates: Optional[Dict[str, float]] = None

    # Metric & execution flags
    include_num_fit_examples: Optional[bool] = None
    include_perplexity: Optional[bool] = None
    include_comm_accounting: bool = False
    overwrite: bool = False
    slurm: Dict[str, Any] = field(default_factory=dict)

    @property
    def algorithm(self) -> str:
        return self.method

    @algorithm.setter
    def algorithm(self, value: str) -> None:
        self.method = value

    def resolve_defaults(self) -> "ExperimentConfig":
        """Populate dataset- and method-specific defaults and validate settings."""
        from fl_bench.datasets.assets import (
            resolve_default_partition_file,
            resolve_default_schedule_file,
        )

        ds_defaults = DATASET_DEFAULTS.get(self.dataset, {})
        m_paper = get_method_paper_defaults(self.method, self.dataset)

        if self.partition_type is None and self.partition_file:
            pf_lower = Path(self.partition_file).stem.lower()
            if "dirichlet" in pf_lower or "alpha" in pf_lower:
                self.partition_type = "dirichlet"
                if self.alpha is None:
                    if "alpha1" in pf_lower:
                        self.alpha = 1.0
                    elif "alpha03" in pf_lower:
                        self.alpha = 0.3
                    elif "alpha01" in pf_lower:
                        self.alpha = 0.1
                    elif "dirichlet05" in pf_lower or "dirichlet_" in pf_lower:
                        self.alpha = 0.5
            else:
                self.partition_type = "iid"

        # 1. Populate shared experimental condition defaults from dataset specification
        for key in (
            "model",
            "data_dir",
            "leaf_root",
            "client_file",
            "partition_type",
            "num_clients",
            "rounds",
            "local_epochs",
            "batch_size",
            "eval_interval",
            "include_num_fit_examples",
            "include_perplexity",
        ):
            if getattr(self, key) is None and key in ds_defaults:
                setattr(self, key, ds_defaults[key])

        # 2. Populate optimizer & algorithm defaults from official paper defaults first,
        #    falling back to dataset defaults only when unset in METHOD_PAPER_DEFAULTS.
        for key in (
            "lr",
            "lr_decay",
            "momentum",
            "weight_decay",
            "max_grad_norm",
            "server_lr",
        ):
            if getattr(self, key) is None:
                if key in m_paper and m_paper[key] is not None:
                    setattr(self, key, m_paper[key])
                elif key in ds_defaults:
                    setattr(self, key, ds_defaults[key])

        if self.lr_decay is None:
            self.lr_decay = 1.0

        if self.method_version is None and "version" in m_paper:
            self.method_version = str(m_paper["version"])

        # Track any deviations from the canonical paper defaults for provenance
        overrides_list: List[str] = []
        for k, paper_val in m_paper.items():
            if k in ("reference", "version") or paper_val is None:
                continue
            cur_val = getattr(self, k, None)
            if cur_val is None:
                continue
            if isinstance(paper_val, float) and isinstance(cur_val, (int, float)):
                if not math.isclose(float(cur_val), float(paper_val), rel_tol=1e-6, abs_tol=1e-9):
                    overrides_list.append(f"{k}={cur_val} (paper={paper_val})")
            elif cur_val != paper_val:
                overrides_list.append(f"{k}={cur_val} (paper={paper_val})")
        self.hyperparameter_overrides = overrides_list
        if overrides_list and self.hyperparameter_source == "paper_default":
            self.hyperparameter_source = "benchmark_configured"

        if self.partition_seed is None:
            self.partition_seed = int(self.training_seed)
        if self.schedule_seed is None:
            self.schedule_seed = int(self.training_seed)

        if self.num_clients is not None:
            if (
                self.clients_per_round is not None
                and self.participation_ratio is not None
            ):
                expected_cpr = int(
                    round(self.num_clients * self.participation_ratio)
                )
                if expected_cpr != self.clients_per_round:
                    raise ValueError(
                        f"Incompatible clients_per_round={self.clients_per_round} and "
                        f"participation_ratio={self.participation_ratio} for num_clients={self.num_clients}."
                    )
            elif (
                self.clients_per_round is None
                and self.participation_ratio is not None
            ):
                self.clients_per_round = max(
                    1, int(round(self.num_clients * self.participation_ratio))
                )
            elif (
                self.clients_per_round is None
                and "clients_per_round" in ds_defaults
            ):
                self.clients_per_round = int(ds_defaults["clients_per_round"])

            if self.clients_per_round is not None:
                self.participation_ratio = (
                    self.clients_per_round / self.num_clients
                )

        if self.partition_label is None:
            if self.partition_type == "dirichlet" and self.alpha is not None:
                self.partition_label = format_alpha_label(self.alpha)
            elif self.partition_type:
                self.partition_label = self.partition_type
            else:
                self.partition_label = "default"

        if self.dataset == "cifar10" and not self.partition_file:
            if self.partition_type == "dirichlet" and self.alpha is None:
                raise ValueError(
                    "Dirichlet partition requires `alpha` to be specified."
                )
            assert self.num_clients is not None
            assert self.partition_type is not None
            self.partition_file = resolve_default_partition_file(
                track=self.track,
                dataset=self.dataset,
                partition_type=self.partition_type,
                alpha=self.alpha,
                num_clients=self.num_clients,
                partition_seed=self.partition_seed,
            )

        if not self.schedule_file:
            assert self.num_clients is not None
            assert self.clients_per_round is not None
            assert self.rounds is not None
            self.schedule_file = resolve_default_schedule_file(
                track=self.track,
                dataset=self.dataset,
                num_clients=self.num_clients,
                clients_per_round=self.clients_per_round,
                rounds=self.rounds,
                schedule_seed=self.schedule_seed,
            )

        if self.server_lr is None:
            if self.method == "fedavgm":
                self.server_lr = 1.0
            elif self.method == "fedadam":
                self.server_lr = (
                    0.01
                    if self.track == "track4"
                    else ds_defaults.get("server_lr_fedadam", 0.001)
                )
            else:
                self.server_lr = 1.0

        if self.local_budget_label is None:
            if self.local_steps is not None:
                self.local_budget_label = f"steps{int(self.local_steps)}"
            else:
                self.local_budget_label = f"epochs{int(self.local_epochs or 1)}"

        if not self.run_id:
            m_tag = (
                f"{self.method}_v2"
                if self.method == "scaffold"
                else self.method
            )
            if self.track == "track4":
                pct = int(round((self.participation_ratio or 0.1) * 100))
                grp = f"{'niid' if self.partition_type == 'dirichlet' else 'iid'}{pct}"
                self.run_id = f"{m_tag}_{grp}_{self.num_clients}c_{self.clients_per_round}cpr_seed{self.training_seed}"
            elif self.local_steps is not None or (
                self.local_epochs is not None and int(self.local_epochs) != 1
            ):
                self.run_id = f"{m_tag}_{self.partition_label}_{self.local_budget_label}_seed{self.training_seed}"
            else:
                self.run_id = f"{m_tag}_{self.partition_label}_seed{self.training_seed}"

        run_dir_path = self.resolve_run_dir()
        if self.run_dir is None:
            self.run_dir = str(run_dir_path)
        if not self.output:
            self.output = str(run_dir_path / "metrics.csv")

        self.validate()
        return self

    def resolve_run_dir(self) -> Path:
        """
        Return the canonical directory for this run:
            `results/<track>/<dataset>/<partition_label>[/<local_budget_label>]/<method>/seed_<seed>/`
        """
        if self.run_dir:
            return Path(self.run_dir)
        part_folder = self.partition_label or self.partition_type or "default"
        if self.track == "track4" and self.num_clients and self.clients_per_round:
            pct = int(round((self.participation_ratio or (self.clients_per_round / self.num_clients)) * 100))
            grp = f"{'niid' if self.partition_type == 'dirichlet' else 'iid'}{pct}"
            part_folder = f"{grp}_{self.num_clients}c_{self.clients_per_round}cpr"
        base_path = Path(self.output_dir) / self.track / self.dataset / part_folder
        if self.local_steps is not None or (
            self.local_epochs is not None and int(self.local_epochs) != 1
        ):
            budget_folder = self.local_budget_label or (
                f"steps{int(self.local_steps)}"
                if self.local_steps is not None
                else f"epochs{int(self.local_epochs or 1)}"
            )
            base_path = base_path / budget_folder
        return base_path / self.method / f"seed_{self.training_seed}"
    # __PART3__
    def validate(self) -> None:
        """Validate configuration fields and raise `ValueError` on invalid combinations."""
        if self.dataset not in DATASET_DEFAULTS:
            raise ValueError(
                f"Unsupported dataset '{self.dataset}'. Supported: {sorted(DATASET_DEFAULTS)}"
            )
        if self.partition_type not in ("iid", "dirichlet", "natural"):
            raise ValueError(
                f"Invalid partition_type '{self.partition_type}'. Expected 'iid', 'dirichlet', or 'natural'."
            )
        if self.dataset in ("femnist", "shakespeare") and self.partition_type != "natural":
            raise ValueError(
                f"Dataset '{self.dataset}' only supports natural user partitioning (`partition_type='natural'`)."
            )
        if self.dataset == "cifar10" and self.partition_type == "natural":
            raise ValueError(
                "Dataset 'cifar10' requires 'iid' or 'dirichlet' partitioning."
            )
        if self.partition_type == "dirichlet" and (
            self.alpha is None or float(self.alpha) <= 0.0
        ):
            raise ValueError(
                "Dirichlet partitioning requires a positive `alpha` value."
            )
        if not self.schedule_file:
            raise ValueError("`schedule_file` (--schedule-file) is required.")
        if self.dataset == "cifar10" and not self.partition_file:
            raise ValueError(
                "`partition_file` (--partition-file) is required for cifar10."
            )
        if self.num_clients is None or self.num_clients <= 0:
            raise ValueError("`num_clients` must be a positive integer.")
        if (
            self.clients_per_round is None
            or self.clients_per_round <= 0
            or self.clients_per_round > self.num_clients
        ):
            raise ValueError(
                f"`clients_per_round` ({self.clients_per_round}) must be in 1..{self.num_clients}."
            )
        if self.dataset == "cifar10" and 50_000 % self.num_clients != 0:
            raise ValueError(
                f"CIFAR-10 balanced partition requires `num_clients` dividing 50,000, got {self.num_clients}."
            )
        if self.rounds is None or self.rounds <= 0:
            raise ValueError("`rounds` must be a positive integer.")
        if self.local_epochs is None or self.local_epochs <= 0:
            raise ValueError("`local_epochs` must be a positive integer.")
        if self.local_steps is not None and int(self.local_steps) <= 0:
            raise ValueError("`local_steps` must be a positive integer when set.")
        if self.batch_size is None or self.batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        if self.lr is None or self.lr <= 0.0:
            raise ValueError("`lr` must be a positive float.")

        # Validate capability-based algorithm compatibility (model, data modality, auxiliary data)
        from fl_bench.methods import METHOD_REGISTRY

        if self.method not in METHOD_REGISTRY:
            raise ValueError(
                f"Unknown method '{self.method}'. Supported: {sorted(METHOD_REGISTRY)}"
            )
        METHOD_REGISTRY[self.method].validate_compatibility(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_cli_args(self, executable: str = "python -u -m fl_bench.run") -> List[str]:
        """Construct the explicit CLI argument list to reproduce this single run."""
        parts = executable.split() + [
            "--track", str(self.track),
            "--dataset", str(self.dataset),
            "--model", str(self.model),
            "--method", str(self.method),
            "--schedule-file", str(self.schedule_file),
            "--output", str(self.output),
            "--num-clients", str(self.num_clients),
            "--clients-per-round", str(self.clients_per_round),
            "--rounds", str(self.rounds),
            "--local-epochs", str(self.local_epochs),
            "--batch-size", str(self.batch_size),
            "--lr", str(self.lr),
            "--momentum", str(self.momentum),
            "--weight-decay", str(self.weight_decay),
            "--eval-interval", str(self.eval_interval),
            "--training-seed", str(self.training_seed),
        ]
        if self.partition_seed is not None and self.partition_seed != self.training_seed:
            parts.extend(["--partition-seed", str(self.partition_seed)])
        if self.schedule_seed is not None and self.schedule_seed != self.training_seed:
            parts.extend(["--schedule-seed", str(self.schedule_seed)])
        if self.local_steps is not None:
            parts.extend(["--local-steps", str(self.local_steps)])
        if self.partition_file:
            parts.extend(["--partition-file", str(self.partition_file)])
        if self.partition_type:
            parts.extend(["--partition", str(self.partition_type)])
        if self.alpha is not None:
            parts.extend(["--alpha", str(self.alpha)])
        if self.data_dir and self.dataset == "cifar10":
            parts.extend(["--data-dir", str(self.data_dir)])
        if self.leaf_root and self.dataset in ("femnist", "shakespeare"):
            parts.extend(["--leaf-root", str(self.leaf_root)])
        if self.client_file and self.dataset in ("femnist", "shakespeare"):
            parts.extend(["--client-file", str(self.client_file)])
        if self.max_grad_norm is not None:
            parts.extend(["--max-grad-norm", str(self.max_grad_norm)])
        if self.lr_decay is not None and not math.isclose(float(self.lr_decay), 1.0):
            parts.extend(["--lr-decay", str(self.lr_decay)])
        if self.method == "fedprox":
            parts.extend(["--mu", str(self.mu)])
        elif self.method == "fedavgm":
            parts.extend([
                "--server-momentum", str(self.server_momentum),
                "--server-lr", str(self.server_lr),
            ])
        elif self.method == "fedadam":
            parts.extend([
                "--server-lr", str(self.server_lr),
                "--beta1", str(self.beta1),
                "--beta2", str(self.beta2),
                "--tau", str(self.tau),
            ])
        elif self.method == "fedspeed":
            parts.extend([
                "--rho", str(self.rho),
                "--lamb", str(self.lamb),
                "--fedspeed-alpha", str(self.fedspeed_alpha),
                "--fedspeed-prox-mode", str(self.fedspeed_prox_mode),
                "--fedspeed-comm-mode", str(self.fedspeed_comm_mode),
            ])
        elif self.method == "fedlesam":
            parts.extend([
                "--server-lr", str(self.server_lr),
                "--rho", str(self.rho),
                "--lamb", str(self.lamb),
                "--fedlesam-variant", str(self.fedlesam_variant),
                "--fedlesam-norm-mode", str(self.fedlesam_norm_mode),
                "--fedlesam-perturbation-target", str(self.fedlesam_perturbation_target),
            ])
        elif self.method == "fedwmsam":
            parts.extend([
                "--server-lr", str(self.server_lr),
                "--rho", str(self.rho),
                "--wmsam-alpha", str(self.wmsam_alpha),
                "--wmsam-norm-mode", str(self.wmsam_norm_mode),
            ])
        elif self.method == "fedgps":
            parts.extend([
                "--server-lr", str(self.server_lr),
                "--rho", str(self.rho),
                "--fedgps-surrogate-mode", str(self.fedgps_surrogate_mode),
                "--fedgps-surrogate-alpha", str(self.fedgps_surrogate_alpha),
                "--fedgps-align-loss-weight", str(self.fedgps_align_loss_weight),
                "--fedgps-proto-align-method", str(self.fedgps_proto_align_method),
                "--fedgps-surrogate-num", str(self.fedgps_surrogate_num),
            ])
            if self.fedgps_surrogate_path:
                parts.extend(["--fedgps-surrogate-path", str(self.fedgps_surrogate_path)])
        elif self.method == "heterofl":
            parts.extend([
                "--heterofl-split-mode", str(self.heterofl_split_mode),
                "--heterofl-model-mode", str(self.heterofl_model_mode),
            ])
        if self.include_comm_accounting:
            parts.append("--include-comm-accounting")
        return parts
def load_config_file(path: str | Path) -> Dict[str, Any]:
    """Load experiment or grid configuration from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text()

    if path.suffix.lower() == ".json":
        return json.loads(text)

    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}

    raise ValueError(f"Unsupported config file extension: {path.suffix}")


def _normalize_partition_spec(spec: Any) -> Dict[str, Any]:
    """Normalize a partition specification string or dict into `{type, alpha, label, file}`."""
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("iid", "natural"):
            return {"type": s, "alpha": None, "label": s}
        if s.startswith("alpha_") or s.startswith("alpha"):
            raw = s.replace("alpha_", "").replace("alpha", "")
            alpha_map = {"1": 1.0, "1.0": 1.0, "03": 0.3, "0.3": 0.3, "01": 0.1, "0.1": 0.1, "05": 0.5, "0.5": 0.5}
            alpha_val = alpha_map.get(raw, float(raw))
            return {
                "type": "dirichlet",
                "alpha": alpha_val,
                "label": format_alpha_label(alpha_val),
            }
        return {"type": s, "alpha": None, "label": s}

    if isinstance(spec, dict):
        raw_t = spec.get("type") or spec.get("partition_type")
        ptype = str(raw_t).lower() if raw_t is not None else None
        alpha = spec.get("alpha")
        alpha_val = float(alpha) if alpha is not None else None
        label = spec.get("label") or spec.get("partition_label")
        if not label and ptype is not None:
            label = (
                format_alpha_label(alpha_val)
                if (ptype == "dirichlet" and alpha_val is not None)
                else ptype
            )
        return {
            "type": ptype,
            "alpha": alpha_val,
            "label": str(label) if label else None,
            "file": spec.get("file") or spec.get("partition_file"),
            "seed": int(spec["seed"]) if spec.get("seed") is not None else None,
        }

    raise ValueError(f"Invalid partition specification: {spec}")


def expand_experiment_grid(
    raw_cfg: Dict[str, Any],
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> List[ExperimentConfig]:
    """
    Expand a single-run or grid YAML configuration dictionary (plus optional
    CLI overrides/filters) into a deterministic list of `ExperimentConfig` objects.
    """
    cfg = dict(raw_cfg)
    overrides = {
        k: v
        for k, v in (cli_overrides or {}).items()
        if v is not None and k not in ("config", "plan", "dry_run", "grid_index")
    }

    training_block = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
    eval_block = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else {}
    slurm_block = cfg.get("slurm") if isinstance(cfg.get("slurm"), dict) else {}
    method_params_map = (
        cfg.get("method_params") if isinstance(cfg.get("method_params"), dict) else {}
    )

    base: Dict[str, Any] = {
        "track": overrides.get("track") or cfg.get("track") or "track1",
        "experiment_name": overrides.get("experiment_name") or cfg.get("experiment_name") or "",
        "dataset": overrides.get("dataset") or cfg.get("dataset") or "cifar10",
        "model": overrides.get("model") or cfg.get("model"),
        "output_dir": overrides.get("output_dir") or cfg.get("output_dir") or "results",
        "data_dir": overrides.get("data_dir") or cfg.get("data_dir"),
        "leaf_root": overrides.get("leaf_root") or cfg.get("leaf_root"),
        "client_file": overrides.get("client_file") or cfg.get("client_file"),
        "rounds": overrides.get("rounds") or training_block.get("rounds") or cfg.get("rounds"),
        "local_epochs": overrides.get("local_epochs") or training_block.get("local_epochs") or cfg.get("local_epochs"),
        "local_steps": (
            int(overrides["local_steps"])
            if overrides.get("local_steps") is not None
            else (
                int(training_block["local_steps"])
                if training_block.get("local_steps") is not None
                else (int(cfg["local_steps"]) if cfg.get("local_steps") is not None else None)
            )
        ),
        "batch_size": overrides.get("batch_size") or training_block.get("batch_size") or cfg.get("batch_size"),
        "lr": overrides.get("lr") or training_block.get("lr") or cfg.get("lr"),
        "momentum": overrides.get("momentum")
        if overrides.get("momentum") is not None
        else (
            training_block.get("momentum")
            if training_block.get("momentum") is not None
            else cfg.get("momentum")
        ),
        "weight_decay": overrides.get("weight_decay")
        if overrides.get("weight_decay") is not None
        else (
            training_block.get("weight_decay")
            if training_block.get("weight_decay") is not None
            else cfg.get("weight_decay")
        ),
        "max_grad_norm": overrides.get("max_grad_norm")
        if overrides.get("max_grad_norm") is not None
        else (
            training_block.get("max_grad_norm")
            if "max_grad_norm" in training_block
            else cfg.get("max_grad_norm")
        ),
        "eval_interval": overrides.get("eval_interval") or eval_block.get("eval_interval") or cfg.get("eval_interval"),
        "eval_batch_size": overrides.get("eval_batch_size") or eval_block.get("eval_batch_size") or cfg.get("eval_batch_size", 256),
        "include_num_fit_examples": overrides.get("include_num_fit_examples")
        if overrides.get("include_num_fit_examples") is not None
        else eval_block.get("include_num_fit_examples", cfg.get("include_num_fit_examples")),
        "include_perplexity": overrides.get("include_perplexity")
        if overrides.get("include_perplexity") is not None
        else eval_block.get("include_perplexity", cfg.get("include_perplexity")),
        "include_comm_accounting": bool(
            overrides.get("include_comm_accounting")
            if overrides.get("include_comm_accounting") is not None
            else eval_block.get("include_comm_accounting", cfg.get("include_comm_accounting", False))
        ),
        "overwrite": bool(overrides.get("overwrite", cfg.get("overwrite", False))),
        "slurm": dict(slurm_block),
    }
    # Methods list
    if overrides.get("methods"):
        methods = [m.strip() for m in str(overrides["methods"]).split(",") if m.strip()]
    elif overrides.get("method") or overrides.get("algorithm"):
        raw_m = str(overrides.get("method") or overrides.get("algorithm"))
        methods = [m.strip() for m in raw_m.split(",") if m.strip()]
    elif isinstance(cfg.get("methods"), list) and cfg["methods"]:
        methods = [str(m) for m in cfg["methods"]]
    elif isinstance(cfg.get("algorithms"), list) and cfg["algorithms"]:
        methods = [str(m) for m in cfg["algorithms"]]
    else:
        methods = [str(cfg.get("method") or cfg.get("algorithm") or "fedavg")]

    # Partitions list
    if overrides.get("partition") or overrides.get("alpha") is not None:
        p_type = overrides.get("partition") or cfg.get("partition_type") or (
            "dirichlet" if overrides.get("alpha") is not None else "iid"
        )
        part_spec = _normalize_partition_spec(p_type)
        if overrides.get("alpha") is not None:
            part_spec["type"] = "dirichlet"
            part_spec["alpha"] = float(overrides["alpha"])
            part_spec["label"] = format_alpha_label(float(overrides["alpha"]))
        partitions = [part_spec]
    elif isinstance(cfg.get("partitions"), list) and cfg["partitions"]:
        partitions = [_normalize_partition_spec(p) for p in cfg["partitions"]]
    elif cfg.get("partition") is not None:
        partitions = [_normalize_partition_spec(cfg["partition"])]
    else:
        has_explicit_pfile = bool(overrides.get("partition_file") or cfg.get("partition_file"))
        default_pt = (
            None
            if has_explicit_pfile and cfg.get("partition_type") is None
            else DATASET_DEFAULTS.get(base["dataset"], {}).get("partition_type", "iid")
        )
        partitions = [_normalize_partition_spec({
            "type": cfg.get("partition_type", default_pt),
            "alpha": cfg.get("alpha"),
            "label": cfg.get("partition_label"),
        })]

    # Scales list (num_clients, clients_per_round, participation_ratio)
    if (
        overrides.get("num_clients") is not None
        or overrides.get("clients_per_round") is not None
        or overrides.get("participation_ratio") is not None
    ):
        scales = [{
            "num_clients": overrides.get("num_clients") or cfg.get("num_clients"),
            "clients_per_round": overrides.get("clients_per_round") or cfg.get("clients_per_round"),
            "participation_ratio": overrides.get("participation_ratio") or cfg.get("participation_ratio"),
        }]
    elif isinstance(cfg.get("scales"), list) and cfg["scales"]:
        scales = list(cfg["scales"])
    else:
        scales = [{
            "num_clients": cfg.get("num_clients"),
            "clients_per_round": cfg.get("clients_per_round"),
            "participation_ratio": cfg.get("participation_ratio"),
        }]
    # Seeds list
    if overrides.get("seeds"):
        seeds = [int(s.strip()) for s in str(overrides["seeds"]).split(",") if s.strip()]
    elif overrides.get("training_seed") is not None or overrides.get("seed") is not None:
        s_val = overrides.get("training_seed") if overrides.get("training_seed") is not None else overrides.get("seed")
        seeds = [int(s_val)]
    elif isinstance(cfg.get("seeds"), list) and cfg["seeds"]:
        seeds = [int(s) for s in cfg["seeds"]]
    else:
        seeds = [int(cfg.get("training_seed") or cfg.get("seed") or 42)]

    # Local optimizer budgets list (local_epochs or exact local_steps)
    if overrides.get("local_steps") is not None:
        local_budgets = [{
            "local_steps": int(overrides["local_steps"]),
            "local_epochs": base.get("local_epochs"),
            "label": f"steps{int(overrides['local_steps'])}",
        }]
    elif overrides.get("local_epochs") is not None:
        local_budgets = [{
            "local_epochs": int(overrides["local_epochs"]),
            "local_steps": None,
            "label": f"epochs{int(overrides['local_epochs'])}",
        }]
    elif isinstance(cfg.get("local_budgets"), list) and cfg["local_budgets"]:
        local_budgets = list(cfg["local_budgets"])
    elif isinstance(training_block.get("local_budgets"), list) and training_block["local_budgets"]:
        local_budgets = list(training_block["local_budgets"])
    else:
        local_budgets = [{
            "local_epochs": base.get("local_epochs"),
            "local_steps": base.get("local_steps"),
            "label": cfg.get("local_budget_label"),
        }]

    expanded: List[ExperimentConfig] = []
    cell_counter = 0

    for part in partitions:
        for scale in scales:
            for budget in local_budgets:
                cell_counter += 1
                grid_cell = scale.get("grid_cell") or f"C{cell_counter}"
                b_steps = (
                    int(budget["local_steps"])
                    if budget.get("local_steps") is not None
                    else None
                )
                b_epochs = (
                    int(budget["local_epochs"])
                    if budget.get("local_epochs") is not None
                    else base.get("local_epochs")
                )
                b_label = budget.get("label")
                for seed in seeds:
                    for method in methods:
                        m_defaults = method_params_map.get(method, {})
                        if not isinstance(m_defaults, dict):
                            m_defaults = {}
                        sched_block = cfg.get("schedule") if isinstance(cfg.get("schedule"), dict) else {}
                        p_seed = (
                            overrides.get("partition_seed")
                            if overrides.get("partition_seed") is not None
                            else (part.get("seed") if part.get("seed") is not None else cfg.get("partition_seed"))
                        )
                        s_seed = (
                            overrides.get("schedule_seed")
                            if overrides.get("schedule_seed") is not None
                            else (sched_block.get("seed") if sched_block.get("seed") is not None else cfg.get("schedule_seed"))
                        )

                        run_kwargs = dict(base)
                        m_paper = get_method_paper_defaults(method, str(base["dataset"]))

                        def _resolve_opt(key: str) -> Any:
                            if overrides.get(key) is not None:
                                return overrides[key]
                            if key in m_defaults and m_defaults[key] is not None:
                                return m_defaults[key]
                            if key in m_paper and m_paper[key] is not None:
                                return m_paper[key]
                            return base.get(key)

                        method_lr = _resolve_opt("lr")
                        method_lr_decay = _resolve_opt("lr_decay")
                        method_momentum = _resolve_opt("momentum")
                        method_wd = _resolve_opt("weight_decay")
                        method_grad_norm = _resolve_opt("max_grad_norm")
                        hp_source = (
                            "cli_override"
                            if any(
                                overrides.get(k) is not None
                                for k in ("lr", "lr_decay", "momentum", "weight_decay", "max_grad_norm", "rho", "lamb", "mu", "server_lr", "wmsam_alpha")
                            )
                            else str(m_defaults.get("hyperparameter_source", "paper_default"))
                        )
                        default_rho = float(m_paper.get("rho", 0.1))
                        default_wmsam_alpha = float(m_paper.get("wmsam_alpha", 0.1))
                        run_kwargs.update({
                            "hyperparameter_source": hp_source,
                            "lr_decay": float(method_lr_decay) if method_lr_decay is not None else 1.0,
                            "weight_decay": float(method_wd) if method_wd is not None else 0.0,
                            "max_grad_norm": float(method_grad_norm) if method_grad_norm is not None else None,
                            "rho": float(overrides.get("rho") if overrides.get("rho") is not None else m_defaults.get("rho", cfg.get("rho", default_rho))),
                            "lamb": float(overrides.get("lamb") if overrides.get("lamb") is not None else m_defaults.get("lamb", cfg.get("lamb", float(m_paper.get("lamb", 0.1))))),
                            "fedspeed_alpha": float(overrides.get("fedspeed_alpha") if overrides.get("fedspeed_alpha") is not None else m_defaults.get("fedspeed_alpha", cfg.get("fedspeed_alpha", 1.0))),
                            "fedspeed_prox_mode": str(overrides.get("fedspeed_prox_mode") or m_defaults.get("fedspeed_prox_mode") or cfg.get("fedspeed_prox_mode") or "paper_symmetric"),
                            "fedspeed_comm_mode": str(overrides.get("fedspeed_comm_mode") or m_defaults.get("fedspeed_comm_mode") or cfg.get("fedspeed_comm_mode") or "reference_simulator"),
                            "fedspeed_freeze_bn_on_perturb": bool(overrides.get("fedspeed_freeze_bn_on_perturb") if overrides.get("fedspeed_freeze_bn_on_perturb") is not None else m_defaults.get("fedspeed_freeze_bn_on_perturb", cfg.get("fedspeed_freeze_bn_on_perturb", True))),
                            "use_ri": bool(overrides.get("use_ri") if overrides.get("use_ri") is not None else m_defaults.get("use_ri", cfg.get("use_ri", False))),
                            "ri_beta": float(overrides.get("ri_beta") if overrides.get("ri_beta") is not None else m_defaults.get("ri_beta", cfg.get("ri_beta", 0.1))),
                            "fedlesam_variant": str(overrides.get("fedlesam_variant") or m_defaults.get("fedlesam_variant") or cfg.get("fedlesam_variant") or "base"),
                            "fedlesam_norm_mode": str(overrides.get("fedlesam_norm_mode") or m_defaults.get("fedlesam_norm_mode") or cfg.get("fedlesam_norm_mode") or "reference_sum_l2"),
                            "fedlesam_perturbation_target": str(overrides.get("fedlesam_perturbation_target") or m_defaults.get("fedlesam_perturbation_target") or cfg.get("fedlesam_perturbation_target") or "server_global_update"),
                            "fedlesam_prox_mode": str(overrides.get("fedlesam_prox_mode") or m_defaults.get("fedlesam_prox_mode") or cfg.get("fedlesam_prox_mode") or "reference_simulator"),
                            "wmsam_alpha": float(overrides.get("wmsam_alpha") if overrides.get("wmsam_alpha") is not None else m_defaults.get("wmsam_alpha", cfg.get("wmsam_alpha", default_wmsam_alpha))),
                            "wmsam_norm_mode": str(overrides.get("wmsam_norm_mode") or m_defaults.get("wmsam_norm_mode") or cfg.get("wmsam_norm_mode") or "reference_sum_l2"),
                        })
                        run_kwargs.update({
                            "fedgps_use_ema": bool(overrides.get("fedgps_use_ema") if overrides.get("fedgps_use_ema") is not None else m_defaults.get("fedgps_use_ema", cfg.get("fedgps_use_ema", False))),
                            "fedgps_ema_decay": float(overrides.get("fedgps_ema_decay") if overrides.get("fedgps_ema_decay") is not None else m_defaults.get("fedgps_ema_decay", cfg.get("fedgps_ema_decay", 0.9))),
                            "fedgps_use_minus": bool(overrides.get("fedgps_use_minus") if overrides.get("fedgps_use_minus") is not None else m_defaults.get("fedgps_use_minus", cfg.get("fedgps_use_minus", True))),
                            "fedgps_require_prev_delta": bool(overrides.get("fedgps_require_prev_delta") if overrides.get("fedgps_require_prev_delta") is not None else m_defaults.get("fedgps_require_prev_delta", cfg.get("fedgps_require_prev_delta", True))),
                            "fedgps_use_minus_lesam": bool(overrides.get("fedgps_use_minus_lesam") if overrides.get("fedgps_use_minus_lesam") is not None else m_defaults.get("fedgps_use_minus_lesam", cfg.get("fedgps_use_minus_lesam", False))),
                            "fedgps_weighted_sum": bool(overrides.get("fedgps_weighted_sum") if overrides.get("fedgps_weighted_sum") is not None else m_defaults.get("fedgps_weighted_sum", cfg.get("fedgps_weighted_sum", False))),
                            "fedgps_label_shift": bool(overrides.get("fedgps_label_shift") if overrides.get("fedgps_label_shift") is not None else m_defaults.get("fedgps_label_shift", cfg.get("fedgps_label_shift", True))),
                            "fedgps_surrogate_mode": str(overrides.get("fedgps_surrogate_mode") or m_defaults.get("fedgps_surrogate_mode") or cfg.get("fedgps_surrogate_mode") or "gaussian"),
                            "fedgps_surrogate_path": overrides.get("fedgps_surrogate_path") or m_defaults.get("fedgps_surrogate_path") or cfg.get("fedgps_surrogate_path"),
                            "fedgps_surrogate_batch_size": overrides.get("fedgps_surrogate_batch_size") or m_defaults.get("fedgps_surrogate_batch_size") or cfg.get("fedgps_surrogate_batch_size"),
                            "fedgps_surrogate_alpha": float(overrides.get("fedgps_surrogate_alpha") if overrides.get("fedgps_surrogate_alpha") is not None else m_defaults.get("fedgps_surrogate_alpha", cfg.get("fedgps_surrogate_alpha", 1.0))),
                            "fedgps_align_loss_weight": float(overrides.get("fedgps_align_loss_weight") if overrides.get("fedgps_align_loss_weight") is not None else m_defaults.get("fedgps_align_loss_weight", cfg.get("fedgps_align_loss_weight", 0.1))),
                            "fedgps_proto_align_method": str(overrides.get("fedgps_proto_align_method") or m_defaults.get("fedgps_proto_align_method") or cfg.get("fedgps_proto_align_method") or "mse"),
                            "fedgps_surrogate_num": int(overrides.get("fedgps_surrogate_num") if overrides.get("fedgps_surrogate_num") is not None else m_defaults.get("fedgps_surrogate_num", cfg.get("fedgps_surrogate_num", int(m_paper.get("fedgps_surrogate_num", 10))))),
                            "heterofl_split_mode": str(overrides.get("heterofl_split_mode") or m_defaults.get("heterofl_split_mode") or cfg.get("heterofl_split_mode") or "fix"),
                            "heterofl_model_mode": str(overrides.get("heterofl_model_mode") or m_defaults.get("heterofl_model_mode") or cfg.get("heterofl_model_mode") or "a2-b2-c2-d2-e2"),
                            "heterofl_scaler": bool(overrides.get("heterofl_scaler") if overrides.get("heterofl_scaler") is not None else m_defaults.get("heterofl_scaler", cfg.get("heterofl_scaler", True))),
                            "heterofl_mask": bool(overrides.get("heterofl_mask") if overrides.get("heterofl_mask") is not None else m_defaults.get("heterofl_mask", cfg.get("heterofl_mask", True))),
                            "heterofl_sbn": bool(overrides.get("heterofl_sbn") if overrides.get("heterofl_sbn") is not None else m_defaults.get("heterofl_sbn", cfg.get("heterofl_sbn", True))),
                            "heterofl_custom_rates": m_defaults.get("heterofl_custom_rates") or cfg.get("heterofl_custom_rates"),
                        })
                        run_kwargs.update({
                            "grid_cell": grid_cell,
                            "method": method,
                            "method_version": m_defaults.get("version") or m_paper.get("version"),
                            "local_epochs": b_epochs,
                            "local_steps": b_steps,
                            "local_budget_label": b_label,
                            "lr": method_lr,
                            "momentum": method_momentum,
                            "partition_type": part["type"],
                            "partition_label": part.get("label"),
                            "alpha": part.get("alpha"),
                            "partition_seed": int(p_seed) if p_seed is not None else None,
                            "schedule_seed": int(s_seed) if s_seed is not None else None,
                            "partition_file": overrides.get("partition_file") or part.get("file") or cfg.get("partition_file"),
                            "schedule_file": overrides.get("schedule_file") or sched_block.get("file") or cfg.get("schedule_file") or "",
                            "num_clients": scale.get("num_clients"),
                            "clients_per_round": scale.get("clients_per_round"),
                            "participation_ratio": scale.get("participation_ratio"),
                            "training_seed": seed,
                            "mu": float(overrides.get("mu") if overrides.get("mu") is not None else m_defaults.get("mu", cfg.get("mu", 0.01))),
                            "server_momentum": float(
                                overrides.get("server_momentum")
                                if overrides.get("server_momentum") is not None
                                else m_defaults.get("server_momentum", cfg.get("server_momentum", 0.9))
                            ),
                            "server_lr": (
                                float(overrides["server_lr"])
                                if overrides.get("server_lr") is not None
                                else (
                                    float(m_defaults["server_lr"])
                                    if m_defaults.get("server_lr") is not None
                                    else (
                                        float(cfg["server_lr"])
                                        if cfg.get("server_lr") is not None and not method_params_map
                                        else None
                                    )
                                )
                            ),
                            "beta1": float(overrides.get("beta1") if overrides.get("beta1") is not None else m_defaults.get("beta1", cfg.get("beta1", 0.9))),
                            "beta2": float(overrides.get("beta2") if overrides.get("beta2") is not None else m_defaults.get("beta2", cfg.get("beta2", 0.99))),
                            "tau": float(overrides.get("tau") if overrides.get("tau") is not None else m_defaults.get("tau", cfg.get("tau", 1e-9))),
                        })

                        if (
                            len(partitions) == 1
                            and len(scales) == 1
                            and len(local_budgets) == 1
                            and len(seeds) == 1
                            and len(methods) == 1
                        ):
                            if overrides.get("output") or cfg.get("output"):
                                run_kwargs["output"] = str(overrides.get("output") or cfg.get("output"))
                            if overrides.get("run_dir") or cfg.get("run_dir"):
                                run_kwargs["run_dir"] = str(overrides.get("run_dir") or cfg.get("run_dir"))

                        exp_cfg = ExperimentConfig(**run_kwargs).resolve_defaults()
                        expanded.append(exp_cfg)

    return expanded
