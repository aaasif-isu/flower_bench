"""Dataset partitioning utilities (IID and balanced Dirichlet) with content validation."""

import argparse
import math
from pathlib import Path
from typing import List, Optional
import zipfile

import numpy as np
from torchvision.datasets import CIFAR10

from fl_bench.utils.io import resolve_repo_path

CIFAR10_NUM_TRAIN_SAMPLES = 50_000


def iid_partition(labels, num_clients: int, seed: int) -> List[np.ndarray]:
    """Uniformly shuffle and split sample indices across `num_clients`."""
    labels = np.asarray(labels, dtype=np.int64)
    num_samples = len(labels)
    if num_clients <= 0 or num_samples % num_clients != 0:
        raise ValueError(
            f"IID partition requires positive num_clients dividing dataset size ({num_samples}), got {num_clients}."
        )
    rng = np.random.default_rng(seed)

    indices = np.arange(num_samples)
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    return [np.asarray(x, dtype=np.int64) for x in splits]



def balanced_dirichlet_partition(
    labels,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[np.ndarray]:
    """
    Generate label-skewed Dirichlet partitions while keeping every client's
    sample count identical.
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

    class_pools = []
    for cls in range(num_classes):
        pool = np.where(labels == cls)[0].copy()
        rng.shuffle(pool)
        class_pools.append(pool.tolist())

    client_probs = rng.dirichlet(
        np.full(num_classes, float(alpha)),
        size=num_clients,
    )

    client_order = rng.permutation(num_clients)
    partitions: List[ list[int] ] = [[] for _ in range(num_clients)]

    for position, client_id in enumerate(client_order):
        if position == num_clients - 1:
            leftovers: list[int] = []
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

            if weighted.sum() <= 0:
                weighted = np.asarray(
                    [len(pool) for pool in class_pools],
                    dtype=np.float64,
                )

            weighted /= weighted.sum()
            cls = int(rng.choice(num_classes, p=weighted))
            sample_index = class_pools[cls].pop()
            partitions[client_id].append(sample_index)

    arrays = [np.asarray(indices, dtype=np.int64) for indices in partitions]

    sizes = np.asarray([len(x) for x in arrays])
    if not np.all(sizes == samples_per_client):
        raise RuntimeError(
            f"Unbalanced partition generated: {sizes.min()}-{sizes.max()}"
        )

    combined = np.concatenate(arrays)
    if len(np.unique(combined)) != num_samples:
        raise RuntimeError("Duplicate or missing training samples detected")

    return arrays


def save_partition(
    output_path: str | Path,
    partitions: List[np.ndarray],
    partition_type: str,
    alpha: Optional[float],
    seed: int,
    dataset: str = "cifar10",
    overwrite: bool = False,
) -> Path:
    """Atomically save client index arrays and metadata into a compressed `.npz` file."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing partition file: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {f"client_{i}": np.asarray(indices, dtype=np.int64) for i, indices in enumerate(partitions)}
    payload["partition_type"] = np.asarray(str(partition_type))
    payload["seed"] = np.asarray(int(seed))
    payload["num_clients"] = np.asarray(len(partitions))
    payload["dataset"] = np.asarray(str(dataset))

    if alpha is not None:
        payload["alpha"] = np.asarray(float(alpha))

    tmp_path = output_path.with_name(output_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(output_path)
    return output_path


def validate_partition_file(
    path: str | Path,
    num_clients: int,
    partition_type: Optional[str] = None,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
    dataset: Optional[str] = None,
    expected_total_samples: Optional[int] = CIFAR10_NUM_TRAIN_SAMPLES,
) -> None:
    """
    Validate `.npz` partition file contents, client counts, sample coverage,
    and embedded metadata. Raises `ValueError` if corrupted or incompatible.
    """
    resolved = resolve_repo_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Partition file not found: {path}")

    try:
        with np.load(resolved, allow_pickle=False) as data:
            files = set(data.files)
            client_keys = {k for k in files if k.startswith("client_")}
            expected_keys = {f"client_{i}" for i in range(num_clients)}
            if client_keys != expected_keys:
                raise ValueError(
                    f"Incompatible partition file '{path}': contains {len(client_keys)} clients, "
                    f"expected num_clients={num_clients}."
                )

            if "num_clients" in files:
                file_n = int(data["num_clients"].item())
                if file_n != int(num_clients):
                    raise ValueError(
                        f"Incompatible partition file '{path}': metadata num_clients={file_n}, "
                        f"expected {num_clients}."
                    )

            if "dataset" in files and dataset is not None:
                file_ds = str(data["dataset"].item())
                if file_ds != str(dataset):
                    raise ValueError(
                        f"Incompatible partition file '{path}': dataset='{file_ds}', expected '{dataset}'."
                    )
            if "partition_type" not in files:
                raise ValueError(
                    f"Corrupted partition file '{path}': missing 'partition_type' metadata."
                )
            file_ptype = str(data["partition_type"].item())
            if partition_type is not None and file_ptype != str(partition_type):
                raise ValueError(
                    f"Incompatible partition file '{path}': partition_type='{file_ptype}', "
                    f"expected '{partition_type}'."
                )

            if "seed" not in files:
                raise ValueError(
                    f"Corrupted partition file '{path}': missing 'seed' metadata."
                )
            file_seed = int(data["seed"].item())
            if seed is not None and file_seed != int(seed):
                raise ValueError(
                    f"Incompatible partition file '{path}': seed={file_seed}, expected seed={seed}."
                )

            if file_ptype == "dirichlet":
                if "alpha" not in files:
                    raise ValueError(
                        f"Corrupted Dirichlet partition file '{path}': missing 'alpha' metadata."
                    )
                file_alpha = float(data["alpha"].item())
                if alpha is not None and not math.isclose(
                    file_alpha, float(alpha), rel_tol=1e-6, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"Incompatible partition file '{path}': alpha={file_alpha:g}, "
                        f"expected alpha={float(alpha):g}."
                    )
            elif "alpha" in files:
                raise ValueError(
                    f"Incompatible IID partition file '{path}': unexpected 'alpha' metadata present."
                )

            parts = [
                np.asarray(data[f"client_{cid}"], dtype=np.int64)
                for cid in range(num_clients)
            ]
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and (
            "Incompatible" in str(exc) or "Corrupted" in str(exc)
        ):
            raise
        raise ValueError(
            f"Corrupted or unreadable partition file '{path}': {exc}"
        ) from exc

    if any(p.ndim != 1 or len(p) == 0 for p in parts):
        raise ValueError(
            f"Corrupted partition file '{path}': client index arrays must be non-empty 1D arrays."
        )

    combined = np.concatenate(parts)
    total = (
        int(expected_total_samples)
        if expected_total_samples is not None
        else len(combined)
    )
    if total % num_clients != 0:
        raise ValueError(
            f"Dataset size {total} is not divisible by num_clients={num_clients}."
        )
    expected_per_client = total // num_clients
    if any(len(p) != expected_per_client for p in parts):
        raise ValueError(
            f"Incompatible partition file '{path}': client sample counts do not equal {expected_per_client}."
        )
    if len(combined) != total or not np.array_equal(
        np.sort(combined), np.arange(total, dtype=np.int64)
    ):
        raise ValueError(
            f"Corrupted partition file '{path}': indices do not form a valid partition of 0..{total - 1}."
        )


def load_partition(
    path: str | Path,
    num_clients: int,
    partition_type: Optional[str] = None,
    alpha: Optional[float] = None,
    seed: Optional[int] = None,
    expected_total_samples: Optional[int] = CIFAR10_NUM_TRAIN_SAMPLES,
) -> List[np.ndarray]:
    """Validate and load per-client index arrays `client_0 .. client_{num_clients-1}`."""
    resolved = resolve_repo_path(path)
    validate_partition_file(
        resolved,
        num_clients=num_clients,
        partition_type=partition_type,
        alpha=alpha,
        seed=seed,
        expected_total_samples=expected_total_samples,
    )
    with np.load(resolved, allow_pickle=False) as data:
        return [
            np.asarray(data[f"client_{cid}"], dtype=np.int64)
            for cid in range(num_clients)
        ]


def load_cifar10_labels(data_dir: str | Path) -> np.ndarray:
    """Load CIFAR-10 training labels for partition generation with clear error reporting."""
    resolved_dir = resolve_repo_path(data_dir)
    extracted = resolved_dir / "cifar-10-batches-py"
    try:
        dataset = CIFAR10(
            root=str(resolved_dir),
            train=True,
            download=not extracted.exists(),
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Raw CIFAR-10 dataset is unavailable at '{resolved_dir}' and automatic download "
            f"could not be completed ({exc}). Place 'cifar-10-batches-py' inside '{resolved_dir}'."
        ) from exc
    return np.asarray(dataset.targets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IID or balanced Dirichlet CIFAR-10 partitions."
    )
    parser.add_argument("--data-dir", default="track1/data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-clients", type=int, default=100)
    parser.add_argument(
        "--partition",
        choices=["iid", "dirichlet"],
        required=True,
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
    )
    labels = np.asarray(dataset.targets)

    if args.partition == "iid":
        partitions = iid_partition(labels, args.num_clients, args.seed)
    else:
        if args.alpha is None:
            raise ValueError("--alpha is required for Dirichlet partitions")
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
