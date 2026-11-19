"""
Track 2 (Efficiency at Matched Model Quality) metric computation, resource
accounting aggregation, cost-basis ranking, and pairwise rank-reversal pipeline.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.common.loader import resolve_analysis_methods
from analysis.common.stats import mean_std_sem
from fl_bench.metrics.analysis import (
    aggregate_target_runs,
    compute_method_rankings,
    compute_pairwise_rank_reversals,
    enrich_trajectory_accounting,
    evaluate_target_efficiency,
)
from fl_bench.metrics.schema import describe_comm_accounting


def _fmt(val: Any, digits: int = 6) -> Any:
    if val is None or val == "":
        return ""
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            return ""
        if val.is_integer() and abs(val) >= 1.0:
            return int(val)
        return f"{val:.{digits}f}"
    return val


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k, "")) for k in fieldnames})
    return path


def resolve_track2_run_csv(
    *,
    results_root: Path,
    raw_fallback_dir: Path,
    dataset: str,
    partition: str,
    local_budget: str,
    method: str,
    seed: int,
) -> Optional[Path]:
    """
    Locate a Track 2 `metrics.csv` in canonical directory layout or raw fallback layout.
    """
    candidates: List[Path] = [
        results_root / dataset / partition / local_budget / method / f"seed_{seed}" / "metrics.csv",
    ]
    if local_budget == "epochs1":
        candidates.append(
            results_root / dataset / partition / method / f"seed_{seed}" / "metrics.csv"
        )
    candidates.append(
        raw_fallback_dir / dataset / f"{method}_{partition}_{local_budget}_seed{seed}.csv"
    )
    if local_budget == "epochs1":
        candidates.append(
            raw_fallback_dir / dataset / f"{method}_{partition}_seed{seed}.csv"
        )

    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 0:
            return cand
    return None
def run_track2_metrics(analysis_cfg: Dict[str, Any], repo_root: Path) -> Dict[str, Path]:
    """
    Execute the Track 2 resource-efficiency and ranking pipeline and write all
    metric CSVs and summary tables.
    """
    results_root = repo_root / str(analysis_cfg.get("results_root", "results/track2"))
    raw_fallback_dir = repo_root / str(analysis_cfg.get("raw_fallback_dir", "results/track2/raw"))
    out_dirs = analysis_cfg.get("output_dirs", {})
    metrics_dir = repo_root / str(out_dirs.get("metrics", "analysis/track2/outputs/metrics"))
    tables_dir = repo_root / str(out_dirs.get("tables", "analysis/track2/outputs/tables"))

    methods: List[str] = resolve_analysis_methods(
        analysis_cfg,
        results_root,
        track=str(analysis_cfg.get("track", "track2")),
    )
    expected_seeds: List[int] = [int(s) for s in analysis_cfg.get("seeds", [42, 43, 44, 45, 46])]
    acct_cfg = analysis_cfg.get("accounting", {})
    sustained_w = int(acct_cfg.get("sustained_eval_checkpoints", 3))
    recompute_bytes = bool(acct_cfg.get("recompute_analytical_bytes", True))
    strict_full_success = bool(acct_cfg.get("strict_full_success_ranking", False))
    datasets_cfg: Dict[str, Dict[str, Any]] = analysis_cfg.get("datasets", {})

    run_level_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    learning_curve_rows: List[Dict[str, Any]] = []

    for ds_name, ds_spec in datasets_cfg.items():
        if not bool(ds_spec.get("enabled", True)):
            continue
        model_name = str(ds_spec.get("model", ""))
        cpr = int(ds_spec.get("clients_per_round", 10))
        partitions = [str(p) for p in ds_spec.get("partitions", [])]
        local_budgets = [str(b) for b in ds_spec.get("local_budgets", ["epochs1"])]
        targets = [float(t) for t in ds_spec.get("targets", [])]

        for part in partitions:
            for budget in local_budgets:
                method_run_evals: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
                for method in methods:
                    discovered_seeds: List[int] = []
                    missing_seeds: List[int] = []
                    trajectories: List[List[Dict[str, Any]]] = []

                    comm_info = describe_comm_accounting(method, model_name)
                    for seed in expected_seeds:
                        csv_path = resolve_track2_run_csv(
                            results_root=results_root,
                            raw_fallback_dir=raw_fallback_dir,
                            dataset=ds_name,
                            partition=part,
                            local_budget=budget,
                            method=method,
                            seed=seed,
                        )
                        if csv_path is None:
                            missing_seeds.append(seed)
                            continue
                        discovered_seeds.append(seed)
                        rel_src = (
                            str(csv_path.relative_to(repo_root))
                            if csv_path.is_relative_to(repo_root)
                            else str(csv_path)
                        )
                        evals = evaluate_target_efficiency(
                            csv_path,
                            targets,
                            source_label=rel_src,
                            sustained_checkpoints=sustained_w,
                            method=method,
                            model_name=model_name,
                            default_clients_per_round=cpr,
                            recompute_analytical_bytes=recompute_bytes,
                        )
                        for ev in evals:
                            ev_row = {
                                "dataset": ds_name,
                                "partition": part,
                                "local_setting": budget,
                                "algorithm": method,
                                "seed": seed,
                            }
                            ev_row.update(ev)
                            run_level_rows.append(ev_row)
                            method_run_evals[method].append(ev_row)

                        with csv_path.open(newline="") as fh:
                            raw_r = list(csv.DictReader(fh))
                        if raw_r:
                            trajectories.append(
                                enrich_trajectory_accounting(
                                    raw_r,
                                    method=method,
                                    model_name=model_name,
                                    default_clients_per_round=cpr,
                                    recompute_analytical_bytes=recompute_bytes,
                                )
                            )
                    coverage_rows.append(
                        {
                            "dataset": ds_name,
                            "model": model_name,
                            "partition": part,
                            "local_setting": budget,
                            "algorithm": method,
                            "n_seeds_expected": len(expected_seeds),
                            "n_seeds_discovered": len(discovered_seeds),
                            "missing_seeds": ";".join(str(s) for s in missing_seeds),
                            "coverage_complete": len(discovered_seeds) == len(expected_seeds),
                            "per_client_upload_bytes": comm_info["per_client_upload_bytes"],
                            "per_client_download_bytes": comm_info["per_client_download_bytes"],
                            "per_client_total_bytes": comm_info["per_client_total_bytes"],
                            "byte_accounting_type": comm_info["accounting_type"],
                        }
                    )

                    if trajectories:
                        by_round: Dict[int, List[Dict[str, Any]]] = {}
                        for traj in trajectories:
                            for r in traj:
                                try:
                                    acc = float(r.get("test_accuracy", "nan"))
                                    if math.isfinite(acc):
                                        rnd = int(r["round"])
                                        by_round.setdefault(rnd, []).append(r)
                                except (ValueError, TypeError, KeyError):
                                    continue
                        for rnd in sorted(by_round):
                            r_list = by_round[rnd]
                            accs_pct = [100.0 * float(x["test_accuracy"]) for x in r_list]
                            a_mean, a_std, a_sem = mean_std_sem(accs_pct)
                            r0 = r_list[0]
                            learning_curve_rows.append(
                                {
                                    "dataset": ds_name,
                                    "partition": part,
                                    "local_setting": budget,
                                    "algorithm": method,
                                    "round": rnd,
                                    "n_seeds": len(r_list),
                                    "test_accuracy_mean_percent": a_mean,
                                    "test_accuracy_std_pp": a_std,
                                    "test_accuracy_sem_pp": a_sem,
                                    "cumulative_fit_clients": int(float(r0.get("cumulative_fit_clients", 0))),
                                    "cumulative_upload_bytes": int(float(r0.get("cumulative_upload_bytes", 0))),
                                    "cumulative_download_bytes": int(float(r0.get("cumulative_download_bytes", 0))),
                                    "cumulative_total_bytes": int(float(r0.get("cumulative_total_bytes", 0))),
                                    "cumulative_local_steps": int(float(r0.get("cumulative_local_steps", 0))),
                                }
                            )

                for target in targets:
                    cond_agg_rows: List[Dict[str, Any]] = []
                    for method in methods:
                        agg = aggregate_target_runs(
                            method_run_evals[method],
                            method=method,
                            dataset=ds_name,
                            partition=part,
                            local_setting=budget,
                            target=target,
                            expected_seeds=len(expected_seeds),
                        )
                        summary_rows.append(agg)
                        cond_agg_rows.append(agg)
                    ranking_rows.extend(
                        compute_method_rankings(
                            cond_agg_rows,
                            strict_full_success_only=strict_full_success,
                        )
                    )
    reversal_rows = compute_pairwise_rank_reversals(ranking_rows)

    written: Dict[str, Path] = {}
    if run_level_rows:
        written["run_level_efficiency"] = _write_csv(
            metrics_dir / "run_level_efficiency.csv",
            list(run_level_rows[0].keys()),
            run_level_rows,
        )
    if summary_rows:
        written["efficiency_summary"] = _write_csv(
            metrics_dir / "efficiency_summary.csv",
            list(summary_rows[0].keys()),
            summary_rows,
        )
    if learning_curve_rows:
        written["learning_curves"] = _write_csv(
            metrics_dir / "learning_curves.csv",
            list(learning_curve_rows[0].keys()),
            learning_curve_rows,
        )
    reversals_fields = [
        "dataset", "partition", "local_setting", "target",
        "basis_a", "basis_b", "method_1", "method_2",
        "method_1_success_rate", "method_2_success_rate",
        "method_1_rank_a", "method_2_rank_a", "method_1_rank_b", "method_2_rank_b",
        "method_1_cost_a_median", "method_2_cost_a_median",
        "method_1_cost_b_median", "method_2_cost_b_median",
    ]
    written["rank_reversals"] = _write_csv(
        metrics_dir / "rank_reversals.csv",
        reversals_fields,
        reversal_rows,
    )
    if ranking_rows:
        written["track2_rankings"] = _write_csv(
            tables_dir / "track2_rankings.csv",
            list(ranking_rows[0].keys()),
            ranking_rows,
        )
    if coverage_rows:
        written["coverage_report"] = _write_csv(
            tables_dir / "coverage_report.csv",
            list(coverage_rows[0].keys()),
            coverage_rows,
        )
    return written
