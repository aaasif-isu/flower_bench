import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FEMNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=np.float32).reshape(-1, 1, 28, 28)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def read_client_list(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def load_split(leaf_root, split, selected_users):
    data_dir = Path(leaf_root) / "data" / split

    selected_set = set(selected_users)
    found = {}

    for path in sorted(data_dir.glob("*.json")):
        with path.open() as f:
            obj = json.load(f)

        for user in obj["users"]:
            if user not in selected_set:
                continue

            ud = obj["user_data"][user]

            found[user] = FEMNISTDataset(
                ud["x"],
                ud["y"],
            )

        if len(found) == len(selected_users):
            break

    missing = [
        user for user in selected_users
        if user not in found
    ]

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} {split} clients. "
            f"First missing: {missing[:5]}"
        )

    # Numeric client ID 0..N-1 maps exactly to the saved
    # eligible_clients.txt order.
    return [
        found[user]
        for user in selected_users
    ]


def load_femnist(
    leaf_root,
    client_file,
):
    users = read_client_list(client_file)

    train_clients = load_split(
        leaf_root,
        "train",
        users,
    )

    test_clients = load_split(
        leaf_root,
        "test",
        users,
    )

    return users, train_clients, test_clients
