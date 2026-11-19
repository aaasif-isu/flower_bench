#!/usr/bin/env python3
"""YAML-driven manifest generator, planner, and Slurm orchestrator for Track 1."""

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_bench.config import ExperimentConfig, expand_experiment_grid, load_config_file
from fl_bench.engine import run_experiment
from fl_bench.run import print_experiment_plan

DEFAULT_CONFIGS = (
    ROOT / "configs/track1/cifar10.yaml",
    ROOT / "configs/track1/femnist.yaml",
    ROOT / "configs/track1/shakespeare.yaml",
)


def load_track1_configs(config_paths: List[str] | None = None) -> List[ExperimentConfig]:
    paths = [Path(p) for p in config_paths] if config_paths else list(DEFAULT_CONFIGS)
    all_configs: List[ExperimentConfig] = []
    for path in paths:
        raw = load_config_file(path)
        all_configs.extend(expand_experiment_grid(raw))
    return all_configs


def write_manifest(configs: List[ExperimentConfig], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        payload = [c.to_dict() for c in configs]
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    else:
        fieldnames = [
            "index",
            "run_id",
            "dataset",
            "model",
            "method",
            "partition_type",
            "partition_label",
            "alpha",
            "num_clients",
            "clients_per_round",
            "rounds",
            "training_seed",
            "partition_file",
            "schedule_file",
            "run_dir",
            "output",
        ]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, c in enumerate(configs):
                writer.writerow({
                    "index": idx,
                    "run_id": c.run_id,
                    "dataset": c.dataset,
                    "model": c.model,
                    "method": c.method,
                    "partition_type": c.partition_type,
                    "partition_label": c.partition_label,
                    "alpha": c.alpha if c.alpha is not None else "",
                    "num_clients": c.num_clients,
                    "clients_per_round": c.clients_per_round,
                    "rounds": c.rounds,
                    "training_seed": c.training_seed,
                    "partition_file": c.partition_file or "",
                    "schedule_file": c.schedule_file,
                    "run_dir": c.run_dir,
                    "output": c.output,
                })
    return output_path
def render_sbatch(config_path: Path, configs: List[ExperimentConfig]) -> str:
    slurm = configs[0].slurm if configs else {}
    job_name = slurm.get("job_name", "flbench_t1")
    account = slurm.get("account", "jannesar-lab")
    partition = slurm.get("partition", "nova")
    nodes = slurm.get("nodes", 1)
    ntasks = slurm.get("ntasks", 1)
    cpus = slurm.get("cpus_per_task", 4)
    gpus = slurm.get("gpus_per_task", 1)
    mem = slurm.get("mem", "24G")
    time_limit = slurm.get("time", "24:00:00")
    max_conc = slurm.get("max_concurrent", 4)
    log_dir = slurm.get("log_dir", "tracks/track1/logs/slurm")
    n_tasks = len(configs)

    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --array=0-{n_tasks - 1}%{max_conc}",
        f"#SBATCH --output={log_dir}/{job_name}_%A_%a.out",
        f"#SBATCH --error={log_dir}/{job_name}_%A_%a.err",
        "",
        "set -euo pipefail",
        'cd "${SLURM_SUBMIT_DIR}"',
        "python -u -m fl_bench.run \\",
        f"  --config {shlex.quote(str(config_path))} \\",
        '  --grid-index "${SLURM_ARRAY_TASK_ID}"',
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_plan = sub.add_parser("plan", help="Preview expanded Track 1 experiment grid.")
    p_plan.add_argument("--config", action="append", default=None)

    p_man = sub.add_parser("manifest", help="Generate CSV or JSON manifest from YAML.")
    p_man.add_argument("--config", action="append", default=None)
    p_man.add_argument(
        "--output",
        default=str(ROOT / "tracks/track1/slurm/track1_manifest.csv"),
    )

    p_sbatch = sub.add_parser("sbatch", help="Print or write Slurm array script for a YAML config.")
    p_sbatch.add_argument("--config", required=True)
    p_sbatch.add_argument("--output", default=None)
    p_sbatch.add_argument("--submit", action="store_true")

    p_run = sub.add_parser("run", help="Run a single grid index from Track 1 YAML configs.")
    p_run.add_argument("--config", action="append", default=None)
    p_run.add_argument("--index", type=int, required=True)

    args = parser.parse_args()

    if args.action == "plan":
        configs = load_track1_configs(args.config)
        print_experiment_plan(configs)
    elif args.action == "manifest":
        configs = load_track1_configs(args.config)
        out = write_manifest(configs, Path(args.output))
        print(f"Wrote {len(configs)} runs to {out}")
    elif args.action == "sbatch":
        cfg_path = Path(args.config)
        configs = load_track1_configs([str(cfg_path)])
        script = render_sbatch(cfg_path, configs)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(script)
            print(f"Wrote Slurm script ({len(configs)} tasks) to {out_path}")
            if args.submit:
                subprocess.run(["sbatch", str(out_path)], cwd=ROOT, check=True)
        else:
            print(script)
    elif args.action == "run":
        configs = load_track1_configs(args.config)
        run_experiment(configs[args.index])


if __name__ == "__main__":
    main()