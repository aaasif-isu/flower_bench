"""Dataset registry and unified federated data container."""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

from torch.utils.data import Dataset

from fl_bench.config import ExperimentConfig
from fl_bench.datasets.assets import (
    AssetStatus,
    ensure_config_assets,
    inspect_config_assets,
    resolve_default_partition_file,
    resolve_default_schedule_file,
)
from fl_bench.datasets.cifar10 import (
    get_cifar10_datasets,
    load_cifar10_federated,
)
from fl_bench.datasets.leaf import (
    FEMNISTDataset,
    ShakespeareDataset,
    load_femnist,
    load_femnist_federated,
    load_shakespeare,
    load_shakespeare_federated,
    validate_natural_client_file,
)
from fl_bench.datasets.partition import (
    balanced_dirichlet_partition,
    iid_partition,
    load_cifar10_labels,
    load_partition,
    save_partition,
    validate_partition_file,
)
from fl_bench.datasets.schedule import (
    generate_schedule,
    load_schedule,
    save_schedule,
    validate_schedule_file,
)


@dataclass
class FederatedData:
    """Standardized container for federated client splits and global test data."""

    name: str
    train_clients: Sequence[Dataset]
    test_dataset: Dataset
    num_clients: int
    include_num_fit_examples: bool = False
    include_perplexity: bool = False


DATASET_REGISTRY: Dict[str, Callable[[ExperimentConfig], FederatedData]] = {
    "cifar10": load_cifar10_federated,
    "femnist": load_femnist_federated,
    "shakespeare": load_shakespeare_federated,
}


def register_dataset(name: str):
    """Decorator to register a new federated dataset loader."""

    def decorator(
        fn: Callable[[ExperimentConfig], FederatedData],
    ) -> Callable[[ExperimentConfig], FederatedData]:
        DATASET_REGISTRY[name] = fn
        return fn

    return decorator


def load_federated_dataset(config: ExperimentConfig) -> FederatedData:
    """Load a registered federated dataset according to `config.dataset`."""
    if config.dataset not in DATASET_REGISTRY:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset '{config.dataset}'. Supported: {supported}"
        )
    return DATASET_REGISTRY[config.dataset](config)


__all__ = [
    "AssetStatus",
    "DATASET_REGISTRY",
    "FEMNISTDataset",
    "FederatedData",
    "ShakespeareDataset",
    "balanced_dirichlet_partition",
    "ensure_config_assets",
    "generate_schedule",
    "get_cifar10_datasets",
    "iid_partition",
    "inspect_config_assets",
    "load_cifar10_federated",
    "load_cifar10_labels",
    "load_federated_dataset",
    "load_femnist",
    "load_femnist_federated",
    "load_partition",
    "load_schedule",
    "load_shakespeare",
    "load_shakespeare_federated",
    "register_dataset",
    "resolve_default_partition_file",
    "resolve_default_schedule_file",
    "save_partition",
    "save_schedule",
    "validate_natural_client_file",
    "validate_partition_file",
    "validate_schedule_file",
]
