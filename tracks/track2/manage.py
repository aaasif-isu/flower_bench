#!/usr/bin/env python3
"""YAML-driven manifest generator and efficiency/ranking analyzer for Track 2."""

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fl_bench.config import expand_experiment_grid, load_config_file
from fl_bench.metrics.analysis import (
    aggregate_target_runs,
    compute_method_rankings,
    evaluate_target_efficiency,
)
from fl_bench.run import print_experiment_plan

DEFAULT_CONFIG = ROOT / "configs/track2/efficiency.yaml"
DATASETS = ("cifar10", "femnist", "shakespeare")
METHODS = ("fedavg", "fedprox", "fedavgm", "fedadam", "fednova", "scaffold")
SEEDS = (42, 43, 44, 45, 46)
DATASET_MODELS = {
    "cifar10": "resnet18",
    "femnist": "femnist_cnn",
    "shakespeare": "shakespeare_lstm",
}


def generate_manifest(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "algorithm", "seed"])
        writer.writeheader()
        for ds in DATASETS:
            for seed in SEEDS:
                for method in METHODS:
                    writer.writerow({
                        "dataset": ds,
                        "algorithm": method,
                        "seed": seed,
                    })
    return output_path


def analyze_efficiency(
    raw_dir: Path = ROOT / "results/track2/raw",
    out_dir: Path = ROOT / "results/track2/production_summary",
    config_path: Path = DEFAULT_CONFIG,
) -> Path:
    """
    Evaluate target efficiency and multi-seed rankings across all 75 Track 2 raw
    CSVs (`cifar10`, `femnist`, `shakespeare`) using targets from `efficiency.yaml`.
    """
    raw_cfg = load_config_file(config_path)
    eval_cfg = raw_cfg.get("evaluation", {})
    sustained_w = int(eval_cfg.get("sustained_eval_checkpoints", 3))
    targets_by_ds: Dict[str, List[float]] = eval_cfg.get(
        "targets_by_dataset",
        {
            "cifar10": [0.70, 0.80, 0.85],
            "femnist": [0.80, 0.85, 0.90],
            "shakespeare": [0.45, 0.50, 0.53],
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = out_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    all_rankings: List[Dict[str, Any]] = []
    per_method_agg: Dict[str, List[Dict[str, Any]]] = {m: [] for m in METHODS}
    per_method_runs: Dict[str, List[Dict[str, Any]]] = {m: [] for m in METHODS}

    for ds in DATASETS:
        ds_dir = raw_dir / ds
        if not ds_dir.exists():
            continue
        partition_name = "iid" if ds == "cifar10" else "natural"
        model_name = DATASET_MODELS.get(ds, "")
        targets = [float(t) for t in targets_by_ds.get(ds, [])]

        for target in targets:
            target_agg_rows: List[Dict[str, Any]] = []
            for method in METHODS:
                method_run_rows: List[Dict[str, Any]] = []
                for seed in SEEDS:
                    csv_file = ds_dir / f"{method}_{partition_name}_seed{seed}.csv"
                    if not csv_file.exists():
                        continue
                    rel_src = str(csv_file.relative_to(ROOT))
                    rows = evaluate_target_efficiency(
                        csv_file,
                        [target],
                        source_label=rel_src,
                        sustained_checkpoints=sustained_w,
                        method=method,
                        model_name=model_name,
                    )
                    method_run_rows.extend(rows)
                    per_method_runs[method].extend(rows)

                agg_row = aggregate_target_runs(
                    method_run_rows,
                    method=method,
                    dataset=ds.upper(),
                    partition=partition_name,
                    local_setting="epochs1",
                    target=target,
                    expected_seeds=len(SEEDS),
                )
                per_method_agg[method].append(agg_row)
                target_agg_rows.append(agg_row)

            all_rankings.extend(compute_method_rankings(target_agg_rows))

    for method, rows in per_method_runs.items():
        if rows:
            p_out = out_dir / f"{method}.csv"
            with p_out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    for method, rows in per_method_agg.items():
        if rows:
            p_out = agg_dir / f"{method}.csv"
            with p_out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    rankings_path = out_dir / "rankings.csv"
    if all_rankings:
        with rankings_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rankings[0].keys()))
            w.writeheader()
            w.writerows(all_rankings)

    return rankings_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_plan = sub.add_parser("plan", help="Preview Track 2 experiment grid.")
    p_plan.add_argument("--config", default=str(DEFAULT_CONFIG))

    p_man = sub.add_parser("manifest", help="Generate Track 2 production CSV manifest.")
    p_man.add_argument(
        "--output",
        default=str(ROOT / "tracks/track2/jobs/track2_production.csv"),
    )

    p_ana = sub.add_parser("analyze", help="Compute Track 2 efficiency summaries and rankings.")
    p_ana.add_argument("--raw-dir", default=str(ROOT / "results/track2/raw"))
    p_ana.add_argument("--out-dir", default=str(ROOT / "results/track2/production_summary"))

    args = parser.parse_args()

    if args.action == "plan":
        raw = load_config_file(args.config)
        configs = expand_experiment_grid(raw)
        print_experiment_plan(configs)
    elif args.action == "manifest":
        out = generate_manifest(Path(args.output))
        print(f"Generated manifest: {out}")
    elif args.action == "analyze":
        out = analyze_efficiency(Path(args.raw_dir), Path(args.out_dir))
        print(f"Wrote Track 2 rankings to {out}")


if __name__ == "__main__":
    main()
