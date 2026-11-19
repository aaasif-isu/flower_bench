"""Dataset utilities for federated learning."""

import numpy as np
from tensorflow import keras

from fedavgm.common import create_lda_partitions


def cifar10(num_classes, input_shape):
    """Prepare the CIFAR-10.

    This method considers CIFAR-10 for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading CIFAR-10. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def fmnist(num_classes, input_shape):
    """Prepare the FMNIST.

    This method considers FMNIST for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading FMNIST. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes



def _load_leaf_json_dir(json_dir):
    """Load LEAF FEMNIST json files from one train/test directory."""
    import json
    from pathlib import Path

    json_dir = Path(json_dir)
    files = sorted(json_dir.glob("all_data_*.json"))
    if not files:
        files = sorted(json_dir.glob("*.json"))

    users = []
    data_by_user = {}

    for fp in files:
        with fp.open("r") as f:
            obj = json.load(f)

        for user in obj.get("users", []):
            user_data = obj["user_data"][user]
            x = np.asarray(user_data["x"], dtype=np.float32)
            y = np.asarray(user_data["y"], dtype=np.int64)

            # FEMNIST is usually flattened 28*28
            if x.ndim == 2 and x.shape[1] == 784:
                x = x.reshape((-1, 28, 28, 1))
            elif x.ndim == 3:
                x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

            # Normalize if needed
            if x.size > 0 and x.max() > 1.0:
                x = x / 255.0

            users.append(user)
            data_by_user[user] = (x, y)

    return users, data_by_user


def _find_leaf_train_test_dirs(root):
    """Find train/test dirs under LEAF FEMNIST root."""
    from pathlib import Path

    root = Path(root)

    candidates = [
        (root / "data" / "train", root / "data" / "test"),
        (root / "train", root / "test"),
        (root / "data" / "all_data" / "train", root / "data" / "all_data" / "test"),
    ]

    for train_dir, test_dir in candidates:
        if train_dir.exists() and test_dir.exists():
            return train_dir, test_dir

    raise FileNotFoundError(
        f"Could not find LEAF FEMNIST train/test dirs under root={root}"
    )


def leaf_femnist(
    num_classes=62,
    input_shape=(28, 28, 1),
    root="/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/external/leaf/data/femnist",
    num_clients=20,
    partition_name="niid",
):
    """Load LEAF FEMNIST as true client partitions.

    Returns:
        x_train: list of per-client (x, y) tuples
        y_train: None marker so partition() can return true LEAF partitions
        x_test/y_test: centralized test set from selected clients
    """
    print(f">>> [Dataset] Loading LEAF FEMNIST from {root}")
    print(f">>> [Dataset] partition={partition_name}, num_clients={num_clients}")

    train_dir, test_dir = _find_leaf_train_test_dirs(root)

    train_users, train_data = _load_leaf_json_dir(train_dir)
    test_users, test_data = _load_leaf_json_dir(test_dir)

    selected_users = [u for u in train_users if u in test_data][: int(num_clients)]
    if len(selected_users) < int(num_clients):
        selected_users = train_users[: int(num_clients)]

    train_partitions = []
    test_x_parts = []
    test_y_parts = []

    for user in selected_users:
        x_tr, y_tr = train_data[user]
        train_partitions.append((x_tr, y_tr))

        if user in test_data:
            x_te, y_te = test_data[user]
            test_x_parts.append(x_te)
            test_y_parts.append(y_te)

    if not train_partitions:
        raise RuntimeError("No LEAF FEMNIST client partitions loaded")

    if test_x_parts:
        x_test = np.concatenate(test_x_parts, axis=0)
        y_test = np.concatenate(test_y_parts, axis=0)
    else:
        # Fallback: use 10% of train as centralized eval if no test user match
        x_test = np.concatenate([xy[0] for xy in train_partitions], axis=0)
        y_test = np.concatenate([xy[1] for xy in train_partitions], axis=0)

    print(f">>> [Dataset] selected users: {len(selected_users)}")
    print(f">>> [Dataset] train partitions: {len(train_partitions)}")
    print(f">>> [Dataset] centralized test shape: {x_test.shape}, {y_test.shape}")
    print(f">>> [Dataset] num_classes={num_classes}, input_shape={input_shape}")

    return train_partitions, None, x_test, y_test, tuple(input_shape), int(num_classes)


def partition(x_train, y_train, num_clients, concentration):
    """Create non-iid partitions or pass through true LEAF client partitions."""

    # LEAF path: x_train is already a list of client partitions
    if isinstance(x_train, list) and y_train is None:
        print(f">>> [Dataset] Using true LEAF client partitions: {num_clients} clients")
        return x_train[: int(num_clients)]

    print(
        f">>> [Dataset] {num_clients} clients, non-iid concentration {concentration}..."
    )
    dataset = [x_train, y_train]
    partitions, _ = create_lda_partitions(
        dataset,
        num_partitions=num_clients,
        concentration=concentration,
        seed=1234,
    )
    return partitions
