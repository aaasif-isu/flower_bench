#!/usr/bin/env python3
"""Prepare, inspect, and run the Track 4 CIFAR-10 grid."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ("fedavg", "fedprox", "fedavgm", "fedadam", "fednova", "scaffold")
GROUPS = (("iid10", "iid", 10), ("iid20", "iid", 20),
          ("niid10", "dirichlet", 10), ("niid20", "dirichlet", 20))
EXTRAS = {
    "fedavg": [],
    "fedprox": ["--mu", "0.01"],
    "fedavgm": ["--server-momentum", "0.9", "--server-lr", "1.0"],
    "fedadam": ["--server-lr", "0.01", "--beta1", "0.9",
                "--beta2", "0.99", "--tau", "1e-9"],
    "fednova": [],
    "scaffold": [],
}


def grid(smoke=False):
    rows = []
    for group_index, (group, partition, participation) in enumerate(GROUPS):
        for n_index, n in enumerate((50,) if smoke else (50, 100, 500)):
            m = n * participation // 100
            for seed in ((42,) if smoke else range(42, 47)):
                for algorithm in ALGORITHMS:
                    name = f"{algorithm}_{group}_{n}c_{m}cpr_seed{seed}"
                    rows.append({
                        "grid": f"C{group_index * 3 + n_index + 1}",
                        "group": group,
                        "algorithm": algorithm,
                        "partition": partition,
                        "N": n,
                        "M": m,
                        "seed": seed,
                        "rounds": 2 if smoke else 500,
                        "partition_file": f"track4/partitions/{partition}_{n}c_seed{seed}.npz",
                        "schedule_file": f"track4/schedules/{n}c_{m}cpr_seed{seed}.json",
                        "output": f"track4/results/{'smoke' if smoke else 'production'}/{name}.csv",
                    })
    return rows


def command(row, output=None):
    return [
        sys.executable, "-u",
        f"track1/cifar/run_{row['algorithm']}.py",
        "--partition-file", row["partition_file"],
        "--schedule-file", row["schedule_file"],
        "--output", output or row["output"],
        "--data-dir", "track1/data",
        "--num-clients", str(row["N"]),
        "--rounds", str(row["rounds"]),
        "--local-epochs", "1",
        "--batch-size", "64",
        "--lr", "0.01",
        "--momentum", "0.9",
        "--weight-decay", "0.0",
        "--training-seed", str(row["seed"]),
    ] + EXTRAS[row["algorithm"]]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def source_hashes():
    paths = sorted((ROOT / "track1/cifar").glob("*.py"))
    paths.append(Path(__file__).resolve())
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def verify_runners():
    for algorithm in ALGORITHMS:
        path = ROOT / f"track1/cifar/run_{algorithm}.py"
        source = path.read_text()
        tree = ast.parse(source)
        supported = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                supported.update(
                    arg.value for arg in node.args
                    if isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                )
        row = next(r for r in grid(True) if r["algorithm"] == algorithm)
        required = {arg for arg in command(row) if arg.startswith("--")}
        if required - supported:
            raise RuntimeError(
                f"Unsupported arguments in {path}: {required - supported}"
            )
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_resnet18"
            for node in ast.walk(tree)
        ):
            raise RuntimeError(f"{path} does not call create_resnet18")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    verify_runners()
    import numpy as np
    from torchvision.datasets import CIFAR10

    for directory in (
        "jobs", "logs", "partitions", "schedules", "status",
        "results/smoke", "results/production",
    ):
        (ROOT / "track4" / directory).mkdir(parents=True, exist_ok=True)

    metadata_path = ROOT / "track4/jobs/prepared.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["source_hashes"] != source_hashes():
            raise RuntimeError(
                "Sources changed since preparation. "
                "Use a new Track 4 directory/version."
            )

    train = CIFAR10(
        root=str(ROOT / "track1/data"), train=True, download=True
    )
    CIFAR10(root=str(ROOT / "track1/data"), train=False, download=True)
    labels = np.asarray(train.targets)

    partitioner = load_module(
        "track4_partition", ROOT / "track1/cifar/partition.py"
    )
    scheduler = load_module(
        "track4_schedule", ROOT / "track1/cifar/schedule.py"
    )

    for kind in ("iid", "dirichlet"):
        for n in (50, 100, 500):
            for seed in range(42, 47):
                path = ROOT / f"track4/partitions/{kind}_{n}c_seed{seed}.npz"
                if not path.exists():
                    print(f"Generating {path.relative_to(ROOT)}", flush=True)
                    if kind == "iid":
                        parts = partitioner.iid_partition(labels, n, seed)
                    else:
                        parts = partitioner.balanced_dirichlet_partition(
                            labels, n, 0.5, seed
                        )
                    partitioner.save_partition(
                        path, parts, kind,
                        None if kind == "iid" else 0.5, seed,
                    )

                with np.load(path) as data:
                    parts = [data[f"client_{i}"] for i in range(n)]
                    combined = np.concatenate(parts)
                    if (
                        any(len(p) != 50000 // n for p in parts)
                        or not np.array_equal(
                            np.sort(combined), np.arange(50000)
                        )
                        or data["partition_type"].item() != kind
                        or data["seed"].item() != seed
                        or (
                            kind == "dirichlet"
                            and data["alpha"].item() != 0.5
                        )
                    ):
                        raise RuntimeError(
                            f"Partition validation failed: {path}"
                        )

    for n in (50, 100, 500):
        for participation in (10, 20):
            m = n * participation // 100
            for seed in range(42, 47):
                path = ROOT / f"track4/schedules/{n}c_{m}cpr_seed{seed}.json"
                payload = {
                    "num_clients": n,
                    "clients_per_round": m,
                    "num_rounds": 500,
                    "seed": seed,
                    "schedule": scheduler.generate_schedule(n, m, 500, seed),
                }
                if path.exists() and json.loads(path.read_text()) != payload:
                    raise RuntimeError(f"Existing schedule differs: {path}")
                write_json(path, payload)

    write_json(ROOT / "track4/jobs/manifest.json", grid())
    write_json(ROOT / "track4/jobs/smoke.json", grid(True))
    write_json(metadata_path, {
        "source_hashes": source_hashes(),
        "python": sys.executable,
        "created": time.time(),
        "algorithm_options": EXTRAS,
    })
    print(
        "Ready: 30 partitions, 30 schedules, "
        "360 production runs, 24 smoke runs."
    )


def complete(path, rounds):
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        numbers = [int(r["round"]) for r in rows]
        expected = list(range(1, rounds + 1))
        if numbers not in (expected, [0] + expected):
            return False
        for row in rows:
            accuracy = float(row["test_accuracy"])
            loss = float(row["test_loss"])
            if not (
                math.isfinite(loss)
                and math.isfinite(accuracy)
                and 0 <= accuracy <= 1
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires a Slurm compute allocation.")

    metadata = json.loads(
        (ROOT / "track4/jobs/prepared.json").read_text()
    )
    if source_hashes() != metadata["source_hashes"]:
        raise RuntimeError(
            "Sources changed since preparation; refusing mixed-code results."
        )

    manifest = "smoke" if args.smoke else "manifest"
    rows = json.loads(
        (ROOT / f"track4/jobs/{manifest}.json").read_text()
    )
    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    row = rows[args.index]

    expected_row = next(
        r for r in grid(args.smoke) if r["output"] == row["output"]
    )
    if row != expected_row:
        raise RuntimeError("Manifest differs from the Track 4 configuration.")

    output = ROOT / row["output"]
    lock = output.with_suffix(".lock")
    try:
        with lock.open("x") as handle:
            handle.write(f"Slurm job {os.environ['SLURM_JOB_ID']}\n")
    except FileExistsError:
        raise RuntimeError(
            f"Run locked: {lock}. Check squeue before removing a stale lock."
        )

    try:
        if output.exists():
            if complete(output, row["rounds"]):
                print(f"Already complete: {output}", flush=True)
                return
            raise RuntimeError(
                f"Incomplete result: {output}; move it aside before restarting."
            )

        partial = output.with_suffix(".partial.csv")
        if partial.exists():
            raise RuntimeError(
                f"Previous partial run: {partial}; "
                "move it aside before restarting."
            )

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU training.")

        print(
            f"GPU: {torch.cuda.get_device_name(0)}; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        cmd = command(row, str(partial))
        print(shlex.join(cmd), flush=True)
        started = time.time()
        subprocess.run(cmd, cwd=ROOT, check=True)

        if not complete(partial, row["rounds"]):
            raise RuntimeError(f"Incomplete/invalid CSV: {partial}")
        partial.replace(output)

        suffix = ".smoke" if args.smoke else ""
        write_json(
            ROOT / "track4/status" / (output.stem + suffix + ".json"),
            {
                "job_id": os.environ["SLURM_JOB_ID"],
                "seconds": time.time() - started,
                "command": cmd,
                "run": row,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        )
        print(f"Completed: {output}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def check_smoke():
    missing = [
        r["output"] for r in grid(True)
        if not complete(ROOT / r["output"], 2)
    ]
    if missing:
        raise RuntimeError(
            f"Smoke tests incomplete/invalid ({len(missing)}/24). "
            f"First: {missing[0]}"
        )
    print("All 24 smoke CSVs passed completeness and finite-metric checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("prepare")
    commands.add_parser("check-smoke")
    plan = commands.add_parser("plan")
    plan.add_argument("--smoke", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--smoke", action="store_true")
    execute.add_argument("--group", choices=[g[0] for g in GROUPS])
    execute.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    if args.action == "prepare":
        prepare()
    elif args.action == "check-smoke":
        check_smoke()
    elif args.action == "plan":
        for row in grid(args.smoke):
            print(row["grid"], row["group"], shlex.join(command(row)))
        print(f"Total: {len(grid(args.smoke))}")
    else:
        run(args)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
