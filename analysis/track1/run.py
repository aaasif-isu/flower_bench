#!/usr/bin/env python3
"""Unified CLI entry point for Track 1 benchmark analysis and visualization."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.track1.metrics import run_track1_metrics
from analysis.track1.plots import run_track1_plots

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_analysis_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Analysis configuration file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text())
    return data if isinstance(data, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Track 1 benchmark metric calculation and figure generation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to Track 1 analysis YAML configuration.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--mode",
        choices=("all", "metrics", "plots"),
        default="all",
        help="Execution mode: 'metrics' (CSVs/tables only), 'plots' (figures from existing CSVs), or 'all' (default).",
    )
    group.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics and summary tables only without generating figures.",
    )
    group.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate figures only from previously computed analysis CSVs.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    mode = args.mode
    if args.metrics_only:
        mode = "metrics"
    elif args.plots_only:
        mode = "plots"

    cfg = load_analysis_config(args.config)

    if mode in ("all", "metrics"):
        written = run_track1_metrics(cfg, REPO_ROOT)
        print(f"[Track 1 Analysis] Wrote {len(written)} metric/table CSV files:")
        for name, path in written.items():
            print(f"  - {name}: {path.relative_to(REPO_ROOT)}")

    if mode in ("all", "plots"):
        figures = run_track1_plots(cfg, REPO_ROOT)
        print(f"[Track 1 Analysis] Generated {len(figures)} figure files:")
        for path in figures:
            print(f"  - {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
