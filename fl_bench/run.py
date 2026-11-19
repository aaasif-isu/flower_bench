"""Unified command-line entrypoint for running FL benchmark experiments and grids."""

import argparse
import shlex
from typing import Any, Dict, List, Optional, Sequence

from fl_bench.config import (
    ExperimentConfig,
    expand_experiment_grid,
    load_config_file,
)
from fl_bench.datasets import DATASET_REGISTRY, inspect_config_assets
from fl_bench.engine import execute_grid, run_experiment
from fl_bench.methods import METHOD_REGISTRY
from fl_bench.models import MODEL_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or plan Federated Learning benchmark experiments."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML/JSON configuration file path (single run or experiment grid).",
    )
    parser.add_argument("--track", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--method",
        "--algorithm",
        dest="method",
        default=None,
        help=f"FL method(s) to run (comma-separated or single; choices: {', '.join(sorted(METHOD_REGISTRY))}).",
    )
    parser.add_argument("--methods", default=None)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_REGISTRY),
        default=None,
        help="Dataset to run (default: cifar10).",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        default=None,
        help="Model architecture (defaults to dataset's canonical model).",
    )

    # Partitioning & Paths
    parser.add_argument("--partition", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--partition-file", default=None)
    parser.add_argument("--schedule-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--leaf-root", default=None)
    parser.add_argument("--client-file", default=None)

    # FL system settings
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--participation-ratio", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--training-seed",
        "--seed",
        dest="training_seed",
        type=int,
        default=None,
    )
    parser.add_argument("--partition-seed", type=int, default=None)
    parser.add_argument("--schedule-seed", type=int, default=None)
    parser.add_argument("--seeds", default=None)

    # Client optimizer hyperparameters
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument(
        "--local-steps",
        type=int,
        default=None,
        help="Exact number of local mini-batch optimizer steps per client round (overrides --local-epochs when set).",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-decay", type=float, default=None, help="Multiplicative per-round client learning rate decay (default 1.0; 0.998 for FedSpeed/FedLESAM/FedWMSAM).")
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    # Method-specific hyperparameters
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--server-momentum", type=float, default=None)
    parser.add_argument("--server-lr", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None, help="SAM perturbation radius (FedSpeed, FedLESAM, FedGPS, FedWMSAM).")
    parser.add_argument("--lamb", type=float, default=None, help="Proximal/dual regularization coefficient (FedSpeed, FedLESAM-S/D).")
    parser.add_argument("--fedspeed-alpha", type=float, default=None, help="FedSpeed quasi-gradient mixing coefficient (default 1.0).")
    parser.add_argument("--fedspeed-prox-mode", choices=["paper_symmetric", "reference_simulator"], default=None)
    parser.add_argument("--fedspeed-comm-mode", choices=["reference_simulator", "paper_stateful_client"], default=None)
    parser.add_argument("--fedlesam-variant", choices=["base", "s", "d"], default=None, help="FedLESAM variant: base, s (SCAFFOLD), or d (FedDyn).")
    parser.add_argument("--fedlesam-norm-mode", choices=["reference_sum_l2", "paper_global_l2"], default=None)
    parser.add_argument("--fedlesam-perturbation-target", choices=["server_global_update", "client_prev_global"], default=None)
    parser.add_argument("--wmsam-alpha", type=float, default=None, help="Initial FedWMSAM mixing coefficient alpha_0 in [0.1, 0.9].")
    parser.add_argument("--wmsam-norm-mode", choices=["reference_sum_l2", "paper_global_l2"], default=None)
    parser.add_argument("--fedgps-surrogate-mode", choices=["gaussian", "dataset", "none"], default=None)
    parser.add_argument("--fedgps-surrogate-path", type=str, default=None)
    parser.add_argument("--fedgps-surrogate-alpha", type=float, default=None)
    parser.add_argument("--fedgps-align-loss-weight", type=float, default=None)
    parser.add_argument("--fedgps-proto-align-method", choices=["mse", "triplet"], default=None)
    parser.add_argument("--fedgps-surrogate-num", type=int, default=None)
    parser.add_argument("--heterofl-split-mode", choices=["fix", "dynamic"], default=None, help="HeteroFL client capacity assignment mode ('fix' or 'dynamic').")
    parser.add_argument("--heterofl-model-mode", type=str, default=None, help="HeteroFL capacity tier distribution (e.g. 'a2-b2-c2-d2-e2', 'a1-b1-c1').")

    # Execution & dry-run / planning options
    parser.add_argument(
        "--include-comm-accounting",
        action="store_true",
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument(
        "--plan",
        "--dry-run",
        dest="plan",
        action="store_true",
        default=False,
        help="Display the expanded experiment grid and CLI commands without executing training.",
    )
    parser.add_argument("--grid-index", type=int, default=None)

    return parser


def parse_configs(
    argv: Optional[List[str]] = None,
) -> tuple[List[ExperimentConfig], argparse.Namespace]:
    """Parse CLI arguments and YAML config into an expanded list of `ExperimentConfig` objects."""
    parser = build_parser()
    args = parser.parse_args(argv)

    raw_cfg: Dict[str, Any] = {}
    if args.config is not None:
        raw_cfg = load_config_file(args.config)

    cli_dict = vars(args)
    configs = expand_experiment_grid(raw_cfg, cli_overrides=cli_dict)

    for cfg in configs:
        if cfg.method not in METHOD_REGISTRY:
            supported = ", ".join(sorted(METHOD_REGISTRY))
            raise ValueError(
                f"Unknown method '{cfg.method}'. Supported: {supported}"
            )

    if args.grid_index is not None:
        if args.grid_index < 0 or args.grid_index >= len(configs):
            raise IndexError(
                f"--grid-index {args.grid_index} out of range (0..{len(configs) - 1})."
            )
        configs = [configs[args.grid_index]]

    return configs, args


def parse_config(argv: Optional[List[str]] = None) -> ExperimentConfig:
    """Parse a single `ExperimentConfig` (first element if a grid is provided)."""
    configs, _ = parse_configs(argv)
    return configs[0]


def print_experiment_plan(configs: Sequence[ExperimentConfig]) -> None:
    """
    Print a structured dry-run summary of required assets (reused vs. to-be-generated),
    the expanded experiment grid, and reproducible CLI commands without writing files.
    """
    asset_statuses = [inspect_config_assets(cfg) for cfg in configs]

    unique_assets = {}
    invalid_errors: List[str] = []
    for part_st, sched_st in asset_statuses:
        for st in (part_st, sched_st):
            key = (st.kind, st.path)
            if key not in unique_assets:
                unique_assets[key] = st
                if st.action == "INVALID" and st.error:
                    invalid_errors.append(st.error)

    print(f"Experiment Assets ({len(unique_assets)} unique):")
    print("-" * 120)
    print(
        f"{'Action':<10s} | {'Type':<11s} | {'Parameters':<48s} | {'Asset Path'}"
    )
    print("-" * 120)
    for st in unique_assets.values():
        print(
            f"{st.action:<10s} | {st.kind:<11s} | {st.identity:<48s} | {st.path}"
        )
    print("-" * 120)

    print(f"\nPlanned Experiments: {len(configs)}")
    print("-" * 130)
    header = (
        f"{'Idx':>4s} | {'Track':<6s} | {'Dataset':<11s} | {'Partition':<16s} | "
        f"{'Method':<8s} | {'N':>4s} | {'M':>4s} | {'R':>4s} | {'Seed':>4s} | "
        f"{'Part':<8s} | {'Sched':<8s} | {'Run Directory'}"
    )
    print(header)
    print("-" * 130)
    for idx, (cfg, (part_st, sched_st)) in enumerate(zip(configs, asset_statuses)):
        part_str = cfg.partition_label or cfg.partition_type or ""
        print(
            f"{idx:4d} | {cfg.track:<6s} | {cfg.dataset:<11s} | {part_str:<16s} | "
            f"{cfg.method:<8s} | {int(cfg.num_clients or 0):4d} | "
            f"{int(cfg.clients_per_round or 0):4d} | {int(cfg.rounds or 0):4d} | "
            f"{cfg.training_seed:4d} | {part_st.action:<8s} | {sched_st.action:<8s} | "
            f"{cfg.resolve_run_dir()}"
        )
    print("-" * 130)
    for idx, cfg in enumerate(configs):
        print(f"[{idx}] {shlex.join(cfg.to_cli_args())}")

    if invalid_errors:
        raise ValueError(
            "One or more existing assets failed validation:\n  - "
            + "\n  - ".join(invalid_errors)
        )


def main(argv: Optional[List[str]] = None) -> None:
    configs, args = parse_configs(argv)
    if args.plan:
        print_experiment_plan(configs)
        return
    execute_grid(configs)


if __name__ == "__main__":
    main()
