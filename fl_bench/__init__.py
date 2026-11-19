"""Modular, YAML-driven Federated Learning Benchmarking Framework (FLBench)."""

from fl_bench.config import (
    DATASET_DEFAULTS,
    ExperimentConfig,
    expand_experiment_grid,
    load_config_file,
)
from fl_bench.datasets import DATASET_REGISTRY, FederatedData, load_federated_dataset
from fl_bench.engine import execute_grid, run_experiment
from fl_bench.methods import BaseMethod, METHOD_REGISTRY, create_method
from fl_bench.models import MODEL_REGISTRY, create_model

__version__ = "0.2.0"

__all__ = [
    "BaseMethod",
    "DATASET_DEFAULTS",
    "DATASET_REGISTRY",
    "ExperimentConfig",
    "FederatedData",
    "METHOD_REGISTRY",
    "MODEL_REGISTRY",
    "__version__",
    "create_method",
    "create_model",
    "execute_grid",
    "expand_experiment_grid",
    "load_config_file",
    "load_federated_dataset",
    "run_experiment",
]

