"""LEAF dataset loaders for FEMNIST and Shakespeare."""

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from fl_bench.config import ExperimentConfig
from fl_bench.models.shakespeare_lstm import ALL_LETTERS, NUM_LETTERS
from fl_bench.utils.io import resolve_repo_path

SEQ_LEN = 80
CHAR_TO_IDX = {c: i for i, c in enumerate(ALL_LETTERS)}


class FEMNISTDataset(Dataset):
    """Per-client LEAF FEMNIST dataset (`1x28x28` float32 images, 62 classes)."""

    def __init__(self, x, y) -> None:
        self.x = np.asarray(x, dtype=np.float32).reshape(-1, 1, 28, 28)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class ShakespeareDataset(Dataset):
    """Per-client LEAF Shakespeare dataset (80-char sequences)."""

    def __init__(self, x, y) -> None:
        encoded_x = []

        for seq in x:
            if len(seq) != SEQ_LEN:
                raise ValueError(
                    f"Expected sequence length {SEQ_LEN}, got {len(seq)}"
                )
            try:
                encoded = [CHAR_TO_IDX[c] for c in seq]
            except KeyError as e:
                raise ValueError(f"Unknown Shakespeare character: {e}") from e
            encoded_x.append(encoded)

        try:
            encoded_y = [CHAR_TO_IDX[c] for c in y]
        except KeyError as e:
            raise ValueError(f"Unknown Shakespeare target: {e}") from e

        self.x = np.asarray(encoded_x, dtype=np.int64)
        self.y = np.asarray(encoded_y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def read_client_list(path: str | Path) -> List[str]:
    """Read ordered list of eligible LEAF user IDs."""
    resolved = resolve_repo_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"LEAF client list file not found: {path}")
    return [
        line.strip()
        for line in resolved.read_text().splitlines()
        if line.strip()
    ]


def validate_natural_client_file(
    path: str | Path,
    expected_num_clients: Optional[int] = None,
) -> List[str]:
    """
    Validate a LEAF `eligible_clients.txt` file and verify that its client count
    matches `expected_num_clients`.
    """
    users = read_client_list(path)
    if not users:
        raise ValueError(f"LEAF client list file '{path}' is empty.")
    if len(set(users)) != len(users):
        raise ValueError(
            f"LEAF client list file '{path}' contains duplicate user IDs."
        )
    if expected_num_clients is not None and len(users) != int(expected_num_clients):
        raise ValueError(
            f"Incompatible LEAF client list '{path}': contains {len(users)} natural clients, "
            f"but num_clients={expected_num_clients} was requested. Synthetic partitioning "
            "is not supported for LEAF natural splits."
        )
    return users


def load_leaf_split(
    leaf_root: str | Path,
    split: str,
    selected_users: List[str],
    dataset_cls: Callable[[list, list], Dataset],
    early_stop: bool = True,
) -> List[Dataset]:
    """Load per-user LEAF JSON split in the exact order of `selected_users`."""
    data_dir = resolve_repo_path(leaf_root) / "data" / split
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Raw LEAF dataset directory not found: '{data_dir}'. "
            f"Preprocess or place LEAF '{split}' JSON split files under '{resolve_repo_path(leaf_root) / 'data'}'."
        )
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No LEAF '{split}' JSON files found in '{data_dir}'."
        )
    selected_set = set(selected_users)
    found = {}

    for path in json_files:
        with path.open() as f:
            obj = json.load(f)

        for user in obj["users"]:
            if user not in selected_set:
                continue
            ud = obj["user_data"][user]
            found[user] = dataset_cls(ud["x"], ud["y"])

        if early_stop and len(found) == len(selected_users):
            break

    missing = [user for user in selected_users if user not in found]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} {split} clients. "
            f"First missing: {missing[:5]}"
        )

    return [found[user] for user in selected_users]


def load_femnist(
    leaf_root: str | Path,
    client_file: str | Path,
) -> Tuple[List[str], List[Dataset], List[Dataset]]:
    users = read_client_list(client_file)
    train_clients = load_leaf_split(
        leaf_root, "train", users, FEMNISTDataset, early_stop=True
    )
    test_clients = load_leaf_split(
        leaf_root, "test", users, FEMNISTDataset, early_stop=True
    )
    return users, train_clients, test_clients


def load_shakespeare(
    leaf_root: str | Path,
    client_file: str | Path,
) -> Tuple[List[str], List[Dataset], List[Dataset]]:
    users = read_client_list(client_file)
    train_clients = load_leaf_split(
        leaf_root, "train", users, ShakespeareDataset, early_stop=False
    )
    test_clients = load_leaf_split(
        leaf_root, "test", users, ShakespeareDataset, early_stop=False
    )
    return users, train_clients, test_clients


def load_femnist_federated(config: ExperimentConfig):
    from fl_bench.datasets import FederatedData

    assert config.leaf_root is not None
    assert config.client_file is not None

    users = validate_natural_client_file(config.client_file, config.num_clients)
    train_clients = load_leaf_split(
        config.leaf_root, "train", users, FEMNISTDataset, early_stop=True
    )
    test_clients = load_leaf_split(
        config.leaf_root, "test", users, FEMNISTDataset, early_stop=True
    )
    global_test = ConcatDataset(test_clients)

    print(
        f"Loaded FEMNIST: {len(users)} clients, "
        f"{len(global_test)} total test samples"
    )

    return FederatedData(
        name="femnist",
        train_clients=train_clients,
        test_dataset=global_test,
        num_clients=len(users),
        include_num_fit_examples=True,
        include_perplexity=False,
    )


def load_shakespeare_federated(config: ExperimentConfig):
    from fl_bench.datasets import FederatedData

    assert config.leaf_root is not None
    assert config.client_file is not None

    users = validate_natural_client_file(config.client_file, config.num_clients)
    train_clients = load_leaf_split(
        config.leaf_root, "train", users, ShakespeareDataset, early_stop=False
    )
    test_clients = load_leaf_split(
        config.leaf_root, "test", users, ShakespeareDataset, early_stop=False
    )
    global_test = ConcatDataset(test_clients)

    print(
        f"Loaded Shakespeare: {len(users)} clients, "
        f"{len(global_test)} total test samples"
    )

    return FederatedData(
        name="shakespeare",
        train_clients=train_clients,
        test_dataset=global_test,
        num_clients=len(users),
        include_num_fit_examples=True,
        include_perplexity=True,
    )


__all__ = [
    "ALL_LETTERS",
    "CHAR_TO_IDX",
    "FEMNISTDataset",
    "NUM_LETTERS",
    "SEQ_LEN",
    "ShakespeareDataset",
    "load_femnist",
    "load_femnist_federated",
    "load_shakespeare",
    "load_shakespeare_federated",
    "read_client_list",
    "validate_natural_client_file",
]
