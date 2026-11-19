#!/usr/bin/env python3
"""
CLI entrypoint for the FLBench Track 2 (Efficiency at Matched Model Quality)
analysis and figure generation pipeline.

Usage:
    python -m analysis.track2.run --config analysis/track2/config.yaml
    python -m analysis.track2.run --config analysis/track2/config.yaml --metrics-only
    python -m analysis.track2.run --config analysis/track2/config.yaml --plots-only
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.track2.metrics import run_track2_metrics
from analysis.track2.plots import run_track2_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FLBench Track 2 efficiency analysis, rankings, and figure generation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "analysis/track2/config.yaml"),
        help="Path to Track 2 analysis YAML configuration.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute efficiency CSVs and ranking tables only (skip figure generation).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate figures from existing Track 2 analysis CSVs only.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r") as f:
        analysis_cfg = yaml.safe_load(f) or {}

    written_csvs: Dict[str, Path] = {}
    generated_figs: List[Path] = []

    if not args.plots_only:
        written_csvs = run_track2_metrics(analysis_cfg, REPO_ROOT)
        print(f"Track 2 metrics & tables written ({len(written_csvs)} files):")
        for name, path in written_csvs.items():
            print(f"  - [{name}] {path.relative_to(REPO_ROOT)}")

    if not args.metrics_only:
        generated_figs = run_track2_plots(analysis_cfg, REPO_ROOT)
        print(f"Track 2 figures generated ({len(generated_figs)} files):")
        for path in generated_figs:
            print(f"  - {path.relative_to(REPO_ROOT)}")

    return {"csvs": written_csvs, "figures": generated_figs}


if __name__ == "__main__":
    main()
