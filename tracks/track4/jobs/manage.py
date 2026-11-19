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

ROOT = Path(__file__).resolve().parents[3]
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
                    algo_tag = "scaffold_v2" if (algorithm == "scaffold" and not smoke) else algorithm
                    name = f"{algo_tag}_{group}_{n}c_{m}cpr_seed{seed}"
                    rows.append({
                        "grid": f"C{group_index * 3 + n_index + 1}",
                        "group": group,
                        "algorithm": algorithm,
                        "partition": partition,
                        "N": n,
                        "M": m,
                        "seed": seed,
                        "rounds": 2 if smoke else 500,
                        "partition_file": f"tracks/track4/partitions/{partition}_{n}c_seed{seed}.npz",
                        "schedule_file": (
                            f"tracks/track4/schedules/{n}c_{m}cpr_2r_seed{seed}.json"
                            if smoke
                            else f"tracks/track4/schedules/{n}c_{m}cpr_seed{seed}.json"
                        ),
                        "output": f"results/track4/{'smoke' if smoke else 'production'}/{name}.csv",
                    })
    return rows


def command(row, output=None):
    mom = "0.0" if row["algorithm"] == "scaffold" else "0.9"
    return [
        sys.executable, "-u",
        "-m", "fl_bench.run",
        "--dataset", "cifar10",
        "--model", "resnet18",
        "--algorithm", row["algorithm"],
        "--partition-file", row["partition_file"],
        "--schedule-file", row["schedule_file"],
        "--output", output or row["output"],
        "--data-dir", "tracks/track1/data",
        "--num-clients", str(row["N"]),
        "--rounds", str(row["rounds"]),
        "--local-epochs", "1",
        "--batch-size", "64",
        "--lr", "0.01",
        "--momentum", mom,
        "--weight-decay", "0.0",
        "--training-seed", str(row["seed"]),
    ] + EXTRAS[row["algorithm"]]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def source_hashes():
    paths = sorted((ROOT / "fl_bench").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def verify_runners():
    runner_path = ROOT / "fl_bench/run.py"
    if not runner_path.exists():
        raise RuntimeError(f"Unified runner missing: {runner_path}")

    tree = ast.parse(runner_path.read_text())
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

    resnet_path = ROOT / "fl_bench/models/resnet18.py"
    resnet_tree = ast.parse(resnet_path.read_text())
    if not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "create_resnet18"
        for node in ast.walk(resnet_tree)
    ):
        raise RuntimeError(f"{resnet_path} does not define create_resnet18")

    for algorithm in ALGORITHMS:
        algo_path = ROOT / f"fl_bench/methods/{algorithm}/method.py"
        if not algo_path.exists():
            raise RuntimeError(f"Missing method module: {algo_path}")
        row = next(r for r in grid(True) if r["algorithm"] == algorithm)
        required = {arg for arg in command(row) if arg.startswith("--")}
        if required - supported:
            raise RuntimeError(
                f"Unsupported arguments in {runner_path}: "
                f"{required - supported}"
            )


def prepare():
    verify_runners()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fl_bench.config import expand_experiment_grid, load_config_file
    from fl_bench.datasets.assets import ensure_config_assets

    for directory in ("jobs", "logs", "partitions", "schedules", "status"):
        (ROOT / "tracks/track4" / directory).mkdir(parents=True, exist_ok=True)
    for res_dir in ("smoke", "production"):
        (ROOT / "results/track4" / res_dir).mkdir(parents=True, exist_ok=True)

    metadata_path = ROOT / "tracks/track4/jobs/prepared.json"
    legacy_path = ROOT / "tracks/track4/jobs/prepared_legacy.json"
    existing_metadata = {}
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text())
        if not legacy_path.exists():
            write_json(legacy_path, existing_metadata)

    raw_cfg = load_config_file(ROOT / "configs/track4/cifar10_scale.yaml")
    for cfg in expand_experiment_grid(raw_cfg, cli_overrides={"methods": "fedavg"}):
        ensure_config_assets(cfg)

    write_json(ROOT / "tracks/track4/jobs/manifest.json", grid())
    write_json(ROOT / "tracks/track4/jobs/smoke.json", grid(True))
    updated_metadata = dict(existing_metadata)
    updated_metadata.update({
        "source_hashes": source_hashes(),
        "python": existing_metadata.get("python", sys.executable),
        "created": existing_metadata.get("created", time.time()),
        "algorithm_options": EXTRAS,
    })
    write_json(metadata_path, updated_metadata)
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
        (ROOT / "tracks/track4/jobs/prepared.json").read_text()
    )
    if source_hashes() != metadata["source_hashes"]:
        raise RuntimeError(
            "Sources changed since preparation; refusing mixed-code results."
        )

    manifest = "smoke" if args.smoke else "manifest"
    rows = json.loads(
        (ROOT / f"tracks/track4/jobs/{manifest}.json").read_text()
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
            ROOT / "tracks/track4/status" / (output.stem + suffix + ".json"),
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


def summarize():
    out_path = ROOT / "results/track4/analysis/completed_run_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "meta_run_id", "meta_grid", "meta_algorithm", "meta_group",
        "meta_N", "meta_M", "meta_seed", "partition", "alpha", "rounds",
        "final_test_accuracy", "final_test_loss", "best_test_accuracy",
        "best_round", "mean_last10_test_accuracy", "elapsed_seconds",
        "gpu", "torch_version", "cuda_version", "job_id", "source_csv",
        "partition_file", "schedule_file", "command_json",
    ]
    rows_out = []
    for row in grid(False):
        csv_path = ROOT / row["output"]
        algo_label = row["algorithm"]
        if not csv_path.exists() and row["algorithm"] == "scaffold":
            legacy_csv = ROOT / row["output"].replace("/scaffold_v2_", "/scaffold_")
            if legacy_csv.exists():
                csv_path = legacy_csv
                algo_label = "scaffold_legacy_mom0.9"
        if not complete(csv_path, row["rounds"]):
            continue
        with csv_path.open(newline="") as handle:
            r_list = list(csv.DictReader(handle))
        best_r = max(r_list, key=lambda x: float(x["test_accuracy"]))
        last10 = [float(x["test_accuracy"]) for x in r_list[-10:]]
        st_path = ROOT / "tracks/track4/status" / (csv_path.stem + ".json")
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        rows_out.append({
            "meta_run_id": csv_path.stem,
            "meta_grid": row["grid"],
            "meta_algorithm": algo_label,
            "meta_group": row["group"],
            "meta_N": row["N"],
            "meta_M": row["M"],
            "meta_seed": row["seed"],
            "partition": row["partition"],
            "alpha": 0.5 if row["partition"] == "dirichlet" else "",
            "rounds": len(r_list),
            "final_test_accuracy": float(r_list[-1]["test_accuracy"]),
            "final_test_loss": float(r_list[-1]["test_loss"]),
            "best_test_accuracy": float(best_r["test_accuracy"]),
            "best_round": int(best_r["round"]),
            "mean_last10_test_accuracy": sum(last10) / len(last10),
            "elapsed_seconds": st.get("seconds", ""),
            "gpu": st.get("gpu", ""),
            "torch_version": st.get("torch", ""),
            "cuda_version": st.get("cuda", ""),
            "job_id": st.get("job_id", ""),
            "source_csv": row["output"],
            "partition_file": row["partition_file"],
            "schedule_file": row["schedule_file"],
            "command_json": json.dumps(st.get("command", command(row))),
        })
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} completed runs to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("prepare")
    commands.add_parser("check-smoke")
    commands.add_parser("summarize")
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
    elif args.action == "summarize":
        summarize()
    elif args.action == "plan":
        for row in grid(args.smoke):
            print(row["grid"], row["group"], shlex.join(command(row)))
        print(f"Total: {len(grid(args.smoke))}")
    else:
        run(args)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
