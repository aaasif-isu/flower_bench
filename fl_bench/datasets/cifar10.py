"""CIFAR-10 dataset and federated partition loader."""

from typing import Tuple

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from fl_bench.config import ExperimentConfig
from fl_bench.datasets.partition import load_partition
from fl_bench.utils.io import resolve_repo_path


def get_cifar10_datasets(data_dir: str) -> Tuple[Dataset, Dataset]:
    """Return CIFAR-10 `(trainset, testset)` with standard transforms."""
    resolved_dir = resolve_repo_path(data_dir)
    extracted = resolved_dir / "cifar-10-batches-py"
    need_download = not extracted.exists()

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    try:
        trainset = datasets.CIFAR10(
            root=str(resolved_dir),
            train=True,
            download=need_download,
            transform=train_transform,
        )
        testset = datasets.CIFAR10(
            root=str(resolved_dir),
            train=False,
            download=False,
            transform=test_transform,
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Raw CIFAR-10 dataset is unavailable at '{resolved_dir}' and automatic download "
            f"could not be completed ({exc}). Place 'cifar-10-batches-py' inside '{resolved_dir}'."
        ) from exc

    return trainset, testset


def load_cifar10_federated(config: ExperimentConfig):
    """Build client `Subset` datasets and centralized test set for CIFAR-10."""
    from fl_bench.datasets import FederatedData

    assert config.data_dir is not None
    assert config.partition_file is not None
    assert config.num_clients is not None

    trainset, testset = get_cifar10_datasets(config.data_dir)
    part_path = resolve_repo_path(config.partition_file)
    partitions = load_partition(
        part_path,
        num_clients=config.num_clients,
        partition_type=config.partition_type,
        alpha=config.alpha,
        seed=config.partition_seed,
    )
    train_clients = [Subset(trainset, indices) for indices in partitions]

    return FederatedData(
        name="cifar10",
        train_clients=train_clients,
        test_dataset=testset,
        num_clients=config.num_clients,
        include_num_fit_examples=bool(config.include_num_fit_examples),
        include_perplexity=bool(config.include_perplexity),
    )
