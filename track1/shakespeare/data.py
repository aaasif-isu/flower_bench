import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[]"
    "abcdefghijklmnopqrstuvwxyz}"
)

NUM_LETTERS = len(ALL_LETTERS)
SEQ_LEN = 80

CHAR_TO_IDX = {
    c: i
    for i, c in enumerate(ALL_LETTERS)
}


class ShakespeareDataset(Dataset):
    def __init__(self, x, y):
        encoded_x = []

        for seq in x:
            if len(seq) != SEQ_LEN:
                raise ValueError(
                    f"Expected sequence length {SEQ_LEN}, "
                    f"got {len(seq)}"
                )

            try:
                encoded = [
                    CHAR_TO_IDX[c]
                    for c in seq
                ]
            except KeyError as e:
                raise ValueError(
                    f"Unknown Shakespeare character: {e}"
                )

            encoded_x.append(encoded)

        try:
            encoded_y = [
                CHAR_TO_IDX[c]
                for c in y
            ]
        except KeyError as e:
            raise ValueError(
                f"Unknown Shakespeare target: {e}"
            )

        self.x = np.asarray(
            encoded_x,
            dtype=np.int64,
        )

        self.y = np.asarray(
            encoded_y,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(
                self.y[idx],
                dtype=torch.long,
            ),
        )


def read_client_list(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def load_split(
    leaf_root,
    split,
    selected_users,
):
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

            found[user] = ShakespeareDataset(
                ud["x"],
                ud["y"],
            )

    missing = [
        user for user in selected_users
        if user not in found
    ]

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} {split} clients. "
            f"First missing: {missing[:5]}"
        )

    return [
        found[user]
        for user in selected_users
    ]


def load_shakespeare(
    leaf_root,
    client_file,
):
    users = read_client_list(
        client_file
    )

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

    return (
        users,
        train_clients,
        test_clients,
    )
