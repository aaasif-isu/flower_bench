import json
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class LeafFemnistDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        image = torch.tensor(x, dtype=torch.float32).view(1, 28, 28)
        label = torch.tensor(y, dtype=torch.long)
        return image, label


def _load_leaf_json_dir_limited(
    data_dir: str,
    max_clients: Optional[int] = None,
) -> Dict[str, List[Tuple[list, int]]]:
    """
    Memory-safe LEAF loader.

    For non-IID, we only load the first max_clients clients instead of loading
    the entire FEMNIST dataset into RAM.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"LEAF FEMNIST directory not found: {data_dir}")

    clients = {}

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(data_dir, filename)

        with open(path, "r") as f:
            data = json.load(f)

        for user in data["users"]:
            if max_clients is not None and len(clients) >= max_clients:
                return clients

            x_values = data["user_data"][user]["x"]
            y_values = data["user_data"][user]["y"]
            clients[user] = list(zip(x_values, y_values))

    if not clients:
        raise RuntimeError(f"No LEAF FEMNIST clients found in {data_dir}")

    return clients


def _load_leaf_iid_streaming(
    data_dir: str,
    num_clients: int,
    seed: int,
    max_source_clients: Optional[int] = None,
) -> Dict[str, List[Tuple[list, int]]]:
    """
    Streaming-ish IID builder.

    It does not first store all examples in one giant list. It reads LEAF users
    and randomly assigns each sample to one of num_clients synthetic IID clients.
    For smoke tests, max_source_clients can keep this small.
    """
    rng = random.Random(seed)
    iid_clients = {str(i): [] for i in range(num_clients)}

    seen_source_clients = 0

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(data_dir, filename)

        with open(path, "r") as f:
            data = json.load(f)

        for user in data["users"]:
            if max_source_clients is not None and seen_source_clients >= max_source_clients:
                return iid_clients

            x_values = data["user_data"][user]["x"]
            y_values = data["user_data"][user]["y"]

            for x, y in zip(x_values, y_values):
                cid = str(rng.randrange(num_clients))
                iid_clients[cid].append((x, y))

            seen_source_clients += 1

    return iid_clients


def _make_loaders(client_examples, batch_size, shuffle):
    loaders = {}

    for cid, examples in client_examples.items():
        if len(examples) == 0:
            continue

        loaders[str(cid)] = DataLoader(
            LeafFemnistDataset(examples),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    return loaders


def load_leaf_femnist(
    leaf_root: str,
    partition: str,
    num_clients: int,
    batch_size: int,
    seed: int = 0,
    max_iid_source_clients: Optional[int] = None,
):
    train_dir = os.path.join(leaf_root, "data", "train")
    test_dir = os.path.join(leaf_root, "data", "test")

    partition = partition.lower()

    if partition in ["niid", "non-iid", "noniid"]:
        train_clients = _load_leaf_json_dir_limited(
            train_dir,
            max_clients=num_clients,
        )

        test_clients = _load_leaf_json_dir_limited(
            test_dir,
            max_clients=num_clients,
        )

        selected_leaf_ids = sorted(train_clients.keys())[:num_clients]

        selected_train = {
            str(i): train_clients[leaf_id]
            for i, leaf_id in enumerate(selected_leaf_ids)
        }

        selected_test = {}
        for i, leaf_id in enumerate(selected_leaf_ids):
            if leaf_id in test_clients:
                selected_test[str(i)] = test_clients[leaf_id]

    elif partition == "iid":
        selected_train = _load_leaf_iid_streaming(
            train_dir,
            num_clients=num_clients,
            seed=seed,
            max_source_clients=max_iid_source_clients,
        )

        selected_test = _load_leaf_iid_streaming(
            test_dir,
            num_clients=num_clients,
            seed=seed + 1,
            max_source_clients=max_iid_source_clients,
        )

    else:
        raise ValueError(f"Unknown partition '{partition}'. Use 'iid' or 'niid'.")

    client_ids = sorted(selected_train.keys(), key=int)

    trainloaders = _make_loaders(selected_train, batch_size, shuffle=True)
    testloaders = _make_loaders(selected_test, batch_size, shuffle=False)

    return client_ids, trainloaders, testloaders


def get_leaf_femnist_info():
    return {
        "input_channels": 1,
        "num_classes": 62,
        "image_size": 28,
    }
