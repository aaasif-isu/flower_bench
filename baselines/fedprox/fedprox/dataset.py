"""fedprox: Dataset loading for MNIST, FEMNIST, and CIFAR-10."""

import numpy as np
from datasets import DatasetDict, load_dataset
from easydict import EasyDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DistributionPartitioner
from flwr_datasets.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

FDS = None  # Cache FederatedDataset

MNIST_TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

CIFAR10_TRANSFORMS = Compose(
    [
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)


class FEMNISTFilter(Preprocessor):
    """Filter FEMNIST to labels 0-9."""

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        allowed_labels = list(range(10))
        return dataset.filter(lambda example: example["character"] in allowed_labels)


def apply_transforms(batch):
    """Apply transforms to MNIST/FEMNIST/CIFAR-10 batches."""

    # CIFAR-10 from Hugging Face uses "img" + "label".
    # Important: return ONLY tensor image + label.
    # Do not return raw "img", because DataLoader cannot collate PIL images.
    if "img" in batch:
        return {
            "image": [CIFAR10_TRANSFORMS(img.convert("RGB")) for img in batch["img"]],
            "label": batch["label"],
        }

    # MNIST uses "image" + "label".
    if "label" in batch and "image" in batch:
        return {
            "image": [MNIST_TRANSFORMS(img) for img in batch["image"]],
            "label": batch["label"],
        }

    # FEMNIST uses "image" + "character".
    if "character" in batch and "image" in batch:
        return {
            "image": [MNIST_TRANSFORMS(img) for img in batch["image"]],
            "character": batch["character"],
        }

    raise KeyError(f"No valid image/label columns found. Batch keys: {batch.keys()}")


def process_femnist(dataset):
    """Process FEMNIST when setting up centralised test data."""
    return dataset.filter(lambda example: example["character"] in list(range(10)))


def load_data(
    dataset_config: EasyDict,
    partition_id: int,
    num_partitions: int,
):
    """Load and partition data."""
    global FDS  # pylint: disable=global-statement

    if FDS is None:
        rng = np.random.default_rng(dataset_config.seed)
        distribution_array = rng.lognormal(
            dataset_config.mu,
            dataset_config.sigma,
            (num_partitions * dataset_config.num_unique_labels_per_partition),
        )
        distribution_array = distribution_array.reshape(
            (dataset_config.num_unique_labels, -1)
        )

        label_key = "character" if "femnist" in dataset_config.path else "label"

        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=num_partitions,
            num_unique_labels_per_partition=dataset_config.num_unique_labels_per_partition,
            partition_by=label_key,
            preassigned_num_samples_per_label=dataset_config.preassigned_num_samples_per_label,
        )

        if "femnist" in dataset_config.path:
            FDS = FederatedDataset(
                dataset=dataset_config.path,
                partitioners={"train": partitioner},
                preprocessor=FEMNISTFilter(),
            )
        else:
            FDS = FederatedDataset(
                dataset=dataset_config.path,
                partitioners={"train": partitioner},
            )

    partition = FDS.load_partition(partition_id)

    partition_train_test = partition.train_test_split(
        test_size=dataset_config.val_ratio,
        seed=dataset_config.seed,
    )

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    return (
        DataLoader(
            partition_train_test["train"],
            batch_size=dataset_config.batch_size,
            shuffle=True,
        ),
        DataLoader(
            partition_train_test["test"],
            batch_size=dataset_config.batch_size,
        ),
    )


def prepare_test_loader(dataset_config: EasyDict):
    """Generate centralized test dataloader."""
    if "femnist" in dataset_config.path:
        dataset = load_dataset(path=dataset_config.path)["train"]
        split_dataset = dataset.train_test_split(
            test_size=dataset_config.val_ratio,
            seed=dataset_config.seed,
        )
        test_dataset = process_femnist(split_dataset["test"])
        test_dataset = test_dataset.with_transform(apply_transforms)
    else:
        test_dataset = load_dataset(path=dataset_config.path)["test"]
        test_dataset = test_dataset.with_transform(apply_transforms)

    return DataLoader(test_dataset, batch_size=dataset_config.batch_size)
