from pathlib import Path
import argparse

import numpy as np
from torchvision.datasets import CIFAR10


def iid_partition(labels, num_clients, seed):
    rng = np.random.default_rng(seed)

    indices = np.arange(len(labels))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)

    return [np.asarray(x, dtype=np.int64) for x in splits]


def balanced_dirichlet_partition(labels, num_clients, alpha, seed):
    """
    Generate label-skewed Dirichlet partitions while keeping every client's
    sample count identical.

    CIFAR-10:
        50,000 examples / 100 clients = 500 examples per client.

    Each client receives a Dirichlet-distributed preference over classes.
    Samples are then allocated without replacement while enforcing the
    fixed per-client sample budget.
    """

    labels = np.asarray(labels, dtype=np.int64)

    num_samples = len(labels)
    num_classes = int(labels.max()) + 1

    if num_samples % num_clients != 0:
        raise ValueError(
            "Balanced partition requires dataset size divisible by num_clients"
        )

    samples_per_client = num_samples // num_clients
    rng = np.random.default_rng(seed)

    # Actual sample indices remaining for each class.
    class_pools = []

    for cls in range(num_classes):
        pool = np.where(labels == cls)[0].copy()
        rng.shuffle(pool)
        class_pools.append(pool.tolist())

    # Draw one class-preference vector per client.
    client_probs = rng.dirichlet(
        np.full(num_classes, float(alpha)),
        size=num_clients,
    )

    # Randomize allocation order so the same client ID is not always
    # advantaged/disadvantaged by being filled first.
    client_order = rng.permutation(num_clients)

    partitions = [[] for _ in range(num_clients)]

    for position, client_id in enumerate(client_order):

        # Last client simply receives all remaining samples.
        if position == num_clients - 1:
            leftovers = []

            for cls in range(num_classes):
                leftovers.extend(class_pools[cls])
                class_pools[cls] = []

            if len(leftovers) != samples_per_client:
                raise RuntimeError(
                    f"Expected {samples_per_client} leftovers, "
                    f"found {len(leftovers)}"
                )

            rng.shuffle(leftovers)
            partitions[client_id] = leftovers
            continue

        probs = client_probs[client_id].copy()

        for _ in range(samples_per_client):

            available = np.asarray(
                [len(pool) > 0 for pool in class_pools],
                dtype=np.float64,
            )

            weighted = probs * available

            # If all preferred classes are exhausted, fall back to the
            # global remaining class proportions.
            if weighted.sum() <= 0:
                weighted = np.asarray(
                    [len(pool) for pool in class_pools],
                    dtype=np.float64,
                )

            weighted /= weighted.sum()

            cls = int(rng.choice(num_classes, p=weighted))

            sample_index = class_pools[cls].pop()
            partitions[client_id].append(sample_index)

    partitions = [
        np.asarray(indices, dtype=np.int64)
        for indices in partitions
    ]

    # Safety checks.
    sizes = np.asarray([len(x) for x in partitions])

    if not np.all(sizes == samples_per_client):
        raise RuntimeError(
            f"Unbalanced partition generated: {sizes.min()}-{sizes.max()}"
        )

    combined = np.concatenate(partitions)

    if len(np.unique(combined)) != num_samples:
        raise RuntimeError("Duplicate or missing training samples detected")

    return partitions


def save_partition(
    output_path,
    partitions,
    partition_type,
    alpha,
    seed,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        f"client_{i}": indices
        for i, indices in enumerate(partitions)
    }

    payload["partition_type"] = np.asarray(partition_type)
    payload["seed"] = np.asarray(seed)

    if alpha is not None:
        payload["alpha"] = np.asarray(alpha)

    np.savez_compressed(output_path, **payload)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="track1/data",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--partition",
        choices=["iid", "dirichlet"],
        required=True,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
    )

    labels = np.asarray(dataset.targets)

    if args.partition == "iid":
        partitions = iid_partition(
            labels,
            args.num_clients,
            args.seed,
        )

    else:
        if args.alpha is None:
            raise ValueError(
                "--alpha is required for Dirichlet partitions"
            )

        partitions = balanced_dirichlet_partition(
            labels,
            args.num_clients,
            args.alpha,
            args.seed,
        )

    save_partition(
        args.output,
        partitions,
        args.partition,
        args.alpha,
        args.seed,
    )

    sizes = np.asarray([len(x) for x in partitions])

    print("Saved:", args.output)
    print("Clients:", len(partitions))
    print("Total samples:", sizes.sum())
    print("Min client samples:", sizes.min())
    print("Max client samples:", sizes.max())
    print("Mean client samples:", sizes.mean())
    print("Std client samples:", sizes.std())


if __name__ == "__main__":
    main()
