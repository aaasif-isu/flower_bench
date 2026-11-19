"""Track 1 metric calculations and summary table generation."""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.common.loader import LoadedRun, load_track_runs
from analysis.common.scoring import compute_hierarchical_t1us_scores
from analysis.common.stats import compute_sample_summary, trapezoidal_learning_auc


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def _fmt_float(val: Optional[float], digits: int = 6) -> str:
    if val is None or not math.isfinite(float(val)):
        return ""
    return f"{float(val):.{digits}f}"


def compute_convergence_metrics(
    runs: Sequence[LoadedRun],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, int], float]]:
    """
    Compute per-run Convergence Learning AUC:
        C_{m,h,s} = (1 / (R - r_start)) * int_{r_start}^R A_{m,h,s}(r) dr
    over actual observed evaluation checkpoints.
    """
    rows: List[Dict[str, Any]] = []
    auc_lookup: Dict[Tuple[str, str, str, int], float] = {}

    for run in runs:
        if run.is_valid:
            auc_info = trapezoidal_learning_auc(
                run.eval_checkpoints,
                expected_final_round=run.expected_rounds,
            )
            c_val = float(auc_info["auc_fraction"])
            auc_lookup[(run.dataset, run.partition, run.method, run.seed)] = c_val
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": _fmt_float(run.alpha, 2) if run.alpha is not None else "",
                "method": run.method,
                "seed": run.seed,
                "status": run.status,
                "learning_auc_fraction": _fmt_float(c_val, 6),
                "learning_auc_percent": _fmt_float(c_val * 100.0, 4),
                "learning_auc_nominal_r_fraction": _fmt_float(
                    float(auc_info["auc_nominal_r_fraction"]), 6
                ),
                "units": "accuracy_fraction",
                "eval_start_round": auc_info["eval_start_round"],
                "eval_end_round": auc_info["eval_end_round"],
                "observed_span_rounds": auc_info["observed_span_rounds"],
                "num_eval_checkpoints": auc_info["num_eval_checkpoints"],
                "round_0_in_csv": auc_info["round_0_in_csv"],
                "integration_note": auc_info["integration_note"],
                "source_csv": run.source_csv or "",
                "run_dir": str(run.run_dir),
            })
        else:
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": _fmt_float(run.alpha, 2) if run.alpha is not None else "",
                "method": run.method,
                "seed": run.seed,
                "status": run.status,
                "learning_auc_fraction": "",
                "learning_auc_percent": "",
                "learning_auc_nominal_r_fraction": "",
                "units": "accuracy_fraction",
                "eval_start_round": "",
                "eval_end_round": "",
                "observed_span_rounds": "",
                "num_eval_checkpoints": len(run.eval_checkpoints),
                "round_0_in_csv": False,
                "integration_note": run.status_reason,
                "source_csv": run.source_csv or "",
                "run_dir": str(run.run_dir),
            })

    return rows, auc_lookup
def compute_heterogeneity_metrics(
    runs: Sequence[LoadedRun],
    auc_lookup: Dict[Tuple[str, str, str, int], float],
    analysis_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, int], float]]:
    """
    Compute per-run paired Heterogeneity-Induced Degradation (HID):
        HID_{m,h,s} = ((C_{m,IID,s} - C_{m,h,s}) / C_{m,IID,s}) * 100%
    """
    datasets_cfg = analysis_cfg.get("datasets", {})
    run_lookup = {(r.dataset, r.partition, r.method, r.seed): r for r in runs}

    rows: List[Dict[str, Any]] = []
    hid_lookup: Dict[Tuple[str, str, str, int], float] = {}

    for run in runs:
        ds_spec = datasets_cfg.get(run.dataset, {})
        iid_part = ds_spec.get("iid_baseline_partition")
        if run.partition == iid_part:
            continue

        if not iid_part:
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": "",
                "baseline_partition": "",
                "method": run.method,
                "seed": run.seed,
                "status": "UNAVAILABLE",
                "c_iid_fraction": "",
                "c_non_iid_fraction": _fmt_float(
                    auc_lookup.get((run.dataset, run.partition, run.method, run.seed)), 6
                ),
                "auc_drop_fraction": "",
                "hid_fraction": "",
                "hid_percent": "",
                "units": "percent",
                "reason": f"No comparable IID baseline partition exists for natural dataset '{run.dataset}'",
                "iid_source_csv": "",
                "non_iid_source_csv": run.source_csv or "",
            })
            continue

        iid_run = run_lookup.get((run.dataset, str(iid_part), run.method, run.seed))
        c_iid = auc_lookup.get((run.dataset, str(iid_part), run.method, run.seed))
        c_non_iid = auc_lookup.get((run.dataset, run.partition, run.method, run.seed))

        if (
            iid_run is not None
            and iid_run.is_valid
            and run.is_valid
            and c_iid is not None
            and c_non_iid is not None
            and c_iid > 0.0
            and iid_run.expected_rounds == run.expected_rounds
            and iid_run.eval_interval == run.eval_interval
            and iid_run.lr == run.lr
            and iid_run.batch_size == run.batch_size
            and iid_run.local_epochs == run.local_epochs
        ):
            drop = c_iid - c_non_iid
            hid_frac = drop / c_iid
            hid_pct = hid_frac * 100.0
            hid_lookup[(run.dataset, run.partition, run.method, run.seed)] = hid_pct
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": _fmt_float(run.alpha, 2) if run.alpha is not None else "",
                "baseline_partition": str(iid_part),
                "method": run.method,
                "seed": run.seed,
                "status": "VALID",
                "c_iid_fraction": _fmt_float(c_iid, 6),
                "c_non_iid_fraction": _fmt_float(c_non_iid, 6),
                "auc_drop_fraction": _fmt_float(drop, 6),
                "hid_fraction": _fmt_float(hid_frac, 6),
                "hid_percent": _fmt_float(hid_pct, 4),
                "units": "percent",
                "reason": "Matched by dataset, model, method, seed, and training budget",
                "iid_source_csv": iid_run.source_csv or "",
                "non_iid_source_csv": run.source_csv or "",
            })
        else:
            missing_reason = (
                run.status_reason
                if not run.is_valid
                else (
                    iid_run.status_reason
                    if (iid_run is not None and not iid_run.is_valid)
                    else "Comparable IID baseline run missing or incompatible"
                )
            )
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": _fmt_float(run.alpha, 2) if run.alpha is not None else "",
                "baseline_partition": str(iid_part),
                "method": run.method,
                "seed": run.seed,
                "status": "UNAVAILABLE",
                "c_iid_fraction": _fmt_float(c_iid, 6),
                "c_non_iid_fraction": _fmt_float(c_non_iid, 6),
                "auc_drop_fraction": "",
                "hid_fraction": "",
                "hid_percent": "",
                "units": "percent",
                "reason": missing_reason,
                "iid_source_csv": iid_run.source_csv if iid_run else "",
                "non_iid_source_csv": run.source_csv or "",
            })

    return rows, hid_lookup
def compute_final_utility_metrics(
    runs: Sequence[LoadedRun],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, int], float]]:
    """
    Compute per-run Final Test Utility:
        FTU_{m,h,s} = A_{m,h,s}(R)
    plus final test loss, best accuracy, and (for Shakespeare) test perplexity.
    """
    rows: List[Dict[str, Any]] = []
    ftu_lookup: Dict[Tuple[str, str, str, int], float] = {}

    for run in runs:
        if run.is_valid and run.eval_checkpoints:
            final_rnd, final_acc = run.eval_checkpoints[-1]
            _, final_loss = run.eval_losses[-1]
            best_rnd, best_acc = max(run.eval_checkpoints, key=lambda x: x[1])
            final_ppl = run.eval_perplexities[-1][1] if run.eval_perplexities else None
            best_ppl = (
                min(x[1] for x in run.eval_perplexities)
                if run.eval_perplexities
                else None
            )
            ftu_lookup[(run.dataset, run.partition, run.method, run.seed)] = final_acc
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": _fmt_float(run.alpha, 2) if run.alpha is not None else "",
                "method": run.method,
                "seed": run.seed,
                "final_round": final_rnd,
                "status": run.status,
                "ftu_fraction": _fmt_float(final_acc, 6),
                "ftu_percent": _fmt_float(final_acc * 100.0, 4),
                "final_test_loss": _fmt_float(final_loss, 6),
                "final_test_perplexity": _fmt_float(final_ppl, 4),
                "best_test_accuracy_fraction": _fmt_float(best_acc, 6),
                "best_test_accuracy_percent": _fmt_float(best_acc * 100.0, 4),
                "best_round": best_rnd,
                "best_test_perplexity": _fmt_float(best_ppl, 4),
                "units": "accuracy_fraction",
                "source_csv": run.source_csv or "",
                "run_dir": str(run.run_dir),
            })
        else:
            rows.append({
                "dataset": run.dataset,
                "model": run.model,
                "partition": run.partition,
                "alpha": _fmt_float(run.alpha, 2) if run.alpha is not None else "",
                "method": run.method,
                "seed": run.seed,
                "final_round": run.expected_rounds,
                "status": run.status,
                "ftu_fraction": "",
                "ftu_percent": "",
                "final_test_loss": "",
                "final_test_perplexity": "",
                "best_test_accuracy_fraction": "",
                "best_test_accuracy_percent": "",
                "best_round": "",
                "best_test_perplexity": "",
                "units": "accuracy_fraction",
                "source_csv": run.source_csv or "",
                "run_dir": str(run.run_dir),
            })

    return rows, ftu_lookup
def compute_stability_metrics(
    runs: Sequence[LoadedRun],
    ftu_lookup: Dict[Tuple[str, str, str, int], float],
    expected_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    """
    Compute Cross-Seed Variability (Stability) per `(dataset, model, partition, method)`:
        SV_{m,h} = sqrt( (1 / (K - 1)) * sum_{s=1}^K (FTU_{m,h,s} - mean(FTU_{m,h}))^2 )
    Requires K >= 2 valid seeds.
    """
    groups: Dict[Tuple[str, str, str, str], List[LoadedRun]] = {}
    for run in runs:
        key = (run.dataset, run.model, run.partition, run.method)
        groups.setdefault(key, []).append(run)

    rows: List[Dict[str, Any]] = []
    for (dataset, model, partition, method), grp_runs in groups.items():
        valid_seeds = [r.seed for r in grp_runs if r.is_valid]
        invalid_seeds = [
            f"{r.seed}({r.status})" for r in grp_runs if not r.is_valid
        ]
        ftu_vals = [
            ftu_lookup[(dataset, partition, method, s)]
            for s in valid_seeds
            if (dataset, partition, method, s) in ftu_lookup
        ]
        summary = compute_sample_summary(ftu_vals)
        alpha_val = next((r.alpha for r in grp_runs if r.alpha is not None), None)

        if summary.count == len(expected_seeds):
            cov_status = "COMPLETE"
            note = f"All {summary.count}/{len(expected_seeds)} seeds valid"
        elif summary.count >= 2:
            cov_status = "PARTIAL"
            note = f"Partial seed coverage ({summary.count}/{len(expected_seeds)} valid)"
        else:
            cov_status = "INSUFFICIENT_SEEDS"
            reasons = sorted({r.status_reason for r in grp_runs if not r.is_valid})
            note = (
                f"Requires >= 2 valid seeds (found {summary.count}/{len(expected_seeds)}): "
                + "; ".join(reasons)
            )

        cv_pct = (
            (summary.sample_std / summary.mean) * 100.0
            if (summary.count >= 2 and summary.mean > 0.0)
            else float("nan")
        )

        rows.append({
            "dataset": dataset,
            "model": model,
            "partition": partition,
            "alpha": _fmt_float(alpha_val, 2) if alpha_val is not None else "",
            "method": method,
            "valid_seed_count": summary.count,
            "expected_seed_count": len(expected_seeds),
            "missing_or_excluded_seeds": ";".join(invalid_seeds),
            "coverage_status": cov_status,
            "sv_fraction": _fmt_float(summary.sample_std, 6),
            "sv_pp": _fmt_float(summary.sample_std * 100.0, 4),
            "ftu_mean_fraction": _fmt_float(summary.mean, 6),
            "ftu_mean_percent": _fmt_float(summary.mean * 100.0, 4),
            "ftu_min_percent": _fmt_float(summary.minimum * 100.0, 4),
            "ftu_max_percent": _fmt_float(summary.maximum * 100.0, 4),
            "cv_percent": _fmt_float(cv_pct, 4),
            "units": "percentage_points",
            "notes": note,
        })

    return rows


def compute_learning_curves_table(runs: Sequence[LoadedRun]) -> List[Dict[str, Any]]:
    """
    Aggregate per-round evaluation checkpoints across valid seeds for each
    `(dataset, model, partition, method, round)` so plotting can run from CSVs alone.
    """
    groups: Dict[Tuple[str, str, str, str], List[LoadedRun]] = {}
    for run in runs:
        if run.is_valid:
            key = (run.dataset, run.model, run.partition, run.method)
            groups.setdefault(key, []).append(run)

    curve_rows: List[Dict[str, Any]] = []
    for (dataset, model, partition, method), valid_runs in groups.items():
        acc_by_r: Dict[int, List[float]] = {}
        loss_by_r: Dict[int, List[float]] = {}
        ppl_by_r: Dict[int, List[float]] = {}

        for r in valid_runs:
            for rnd, acc in r.eval_checkpoints:
                acc_by_r.setdefault(rnd, []).append(acc)
            for rnd, l_val in r.eval_losses:
                loss_by_r.setdefault(rnd, []).append(l_val)
            for rnd, p_val in r.eval_perplexities:
                ppl_by_r.setdefault(rnd, []).append(p_val)

        for rnd in sorted(acc_by_r.keys()):
            acc_s = compute_sample_summary(acc_by_r[rnd])
            loss_s = compute_sample_summary(loss_by_r.get(rnd, []))
            ppl_s = compute_sample_summary(ppl_by_r.get(rnd, []))
            curve_rows.append({
                "dataset": dataset,
                "model": model,
                "partition": partition,
                "method": method,
                "round": rnd,
                "valid_seed_count": acc_s.count,
                "test_accuracy_mean_fraction": _fmt_float(acc_s.mean, 6),
                "test_accuracy_std_fraction": _fmt_float(acc_s.sample_std, 6),
                "test_accuracy_sem_fraction": _fmt_float(acc_s.sem, 6),
                "test_accuracy_mean_percent": _fmt_float(acc_s.mean * 100.0, 4),
                "test_accuracy_std_pp": _fmt_float(acc_s.sample_std * 100.0, 4),
                "test_accuracy_sem_pp": _fmt_float(acc_s.sem * 100.0, 4),
                "test_loss_mean": _fmt_float(loss_s.mean, 6),
                "test_loss_std": _fmt_float(loss_s.sample_std, 6),
                "test_perplexity_mean": _fmt_float(ppl_s.mean, 4),
                "test_perplexity_std": _fmt_float(ppl_s.sample_std, 4),
            })

    return curve_rows
def compute_summary_and_coverage_tables(
    runs: Sequence[LoadedRun],
    auc_lookup: Dict[Tuple[str, str, str, int], float],
    hid_lookup: Dict[Tuple[str, str, str, int], float],
    ftu_lookup: Dict[Tuple[str, str, str, int], float],
    analysis_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate `track1_summary.csv` and `experiment_coverage.csv`."""
    expected_seeds = [int(s) for s in analysis_cfg.get("seeds", [])]
    datasets_cfg = analysis_cfg.get("datasets", {})

    coverage_rows: List[Dict[str, Any]] = []
    for r in runs:
        coverage_rows.append({
            "track": r.track,
            "dataset": r.dataset,
            "model": r.model,
            "partition": r.partition,
            "alpha": _fmt_float(r.alpha, 2) if r.alpha is not None else "",
            "method": r.method,
            "seed": r.seed,
            "status": r.status,
            "included_in_primary_analysis": r.is_valid,
            "status_reason": r.status_reason,
            "completed_rounds": r.completed_rounds,
            "expected_rounds": r.expected_rounds,
            "num_eval_checkpoints": len(r.eval_checkpoints),
            "method_version": r.method_version or "",
            "lr": _fmt_float(r.lr, 4) if r.lr is not None else "",
            "momentum": _fmt_float(r.momentum, 2) if r.momentum is not None else "",
            "run_dir": str(r.run_dir),
            "source_csv": r.source_csv or "",
        })

    groups: Dict[Tuple[str, str, str, str], List[LoadedRun]] = {}
    for r in runs:
        groups.setdefault((r.dataset, r.model, r.partition, r.method), []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    for (dataset, model, partition, method), grp_runs in groups.items():
        valid_runs = [r for r in grp_runs if r.is_valid]
        valid_seeds = [r.seed for r in valid_runs]
        alpha_val = next((r.alpha for r in grp_runs if r.alpha is not None), None)
        ds_spec = datasets_cfg.get(dataset, {})
        iid_part = ds_spec.get("iid_baseline_partition")

        auc_vals = [
            auc_lookup[(dataset, partition, method, s)]
            for s in valid_seeds
            if (dataset, partition, method, s) in auc_lookup
        ]
        ftu_vals = [
            ftu_lookup[(dataset, partition, method, s)]
            for s in valid_seeds
            if (dataset, partition, method, s) in ftu_lookup
        ]
        hid_vals = [
            hid_lookup[(dataset, partition, method, s)]
            for s in valid_seeds
            if (dataset, partition, method, s) in hid_lookup
        ]
        loss_vals = [r.eval_losses[-1][1] for r in valid_runs if r.eval_losses]
        ppl_vals = [
            r.eval_perplexities[-1][1] for r in valid_runs if r.eval_perplexities
        ]
        best_acc_vals = [
            max(x[1] for x in r.eval_checkpoints)
            for r in valid_runs
            if r.eval_checkpoints
        ]

        auc_s = compute_sample_summary(auc_vals)
        ftu_s = compute_sample_summary(ftu_vals)
        hid_s = compute_sample_summary(hid_vals)
        loss_s = compute_sample_summary(loss_vals)
        ppl_s = compute_sample_summary(ppl_vals)
        best_s = compute_sample_summary(best_acc_vals)
        if partition == iid_part:
            hid_status = "REFERENCE_IID"
        elif not iid_part:
            hid_status = "UNAVAILABLE_NO_IID_BASELINE"
        elif hid_s.count > 0:
            hid_status = "VALID"
        else:
            hid_status = "UNAVAILABLE_MISSING_RUNS"

        if ftu_s.count == len(expected_seeds):
            cov_status = "COMPLETE"
        elif ftu_s.count > 0:
            cov_status = "PARTIAL"
        else:
            cov_status = "NONE"

        reasons = sorted({r.status_reason for r in grp_runs if not r.is_valid})

        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "partition": partition,
            "alpha": _fmt_float(alpha_val, 2) if alpha_val is not None else "",
            "method": method,
            "valid_seed_count": ftu_s.count,
            "expected_seed_count": len(expected_seeds),
            "coverage_status": cov_status,
            "learning_auc_mean_fraction": _fmt_float(auc_s.mean, 6),
            "learning_auc_std_fraction": _fmt_float(auc_s.sample_std, 6),
            "learning_auc_mean_percent": _fmt_float(auc_s.mean * 100.0, 4),
            "learning_auc_std_pp": _fmt_float(auc_s.sample_std * 100.0, 4),
            "hid_mean_percent": (
                "0.0000" if partition == iid_part and ftu_s.count > 0 else _fmt_float(hid_s.mean, 4)
            ),
            "hid_std_percent": (
                "0.0000" if partition == iid_part and ftu_s.count >= 2 else _fmt_float(hid_s.sample_std, 4)
            ),
            "hid_paired_seed_count": (
                ftu_s.count if partition == iid_part else hid_s.count
            ),
            "hid_status": hid_status,
            "ftu_mean_fraction": _fmt_float(ftu_s.mean, 6),
            "ftu_std_fraction": _fmt_float(ftu_s.sample_std, 6),
            "ftu_mean_percent": _fmt_float(ftu_s.mean * 100.0, 4),
            "sv_pp": _fmt_float(ftu_s.sample_std * 100.0, 4),
            "best_test_accuracy_mean_percent": _fmt_float(best_s.mean * 100.0, 4),
            "best_test_accuracy_std_pp": _fmt_float(best_s.sample_std * 100.0, 4),
            "final_test_loss_mean": _fmt_float(loss_s.mean, 6),
            "final_test_loss_std": _fmt_float(loss_s.sample_std, 6),
            "final_test_perplexity_mean": _fmt_float(ppl_s.mean, 4),
            "final_test_perplexity_std": _fmt_float(ppl_s.sample_std, 4),
            "excluded_or_missing_reason": "; ".join(reasons),
        })

    return summary_rows, coverage_rows


def run_track1_metrics(analysis_cfg: Dict[str, Any], repo_root: Path) -> Dict[str, Path]:
    """Execute all Track 1 metric and summary table calculations and write CSV outputs."""
    runs = load_track_runs(analysis_cfg, repo_root)
    expected_seeds = [int(s) for s in analysis_cfg.get("seeds", [])]
    out_dirs = analysis_cfg.get("output_dirs", {})
    metrics_dir = repo_root / str(out_dirs.get("metrics", "analysis/track1/outputs/metrics"))
    tables_dir = repo_root / str(out_dirs.get("tables", "analysis/track1/outputs/tables"))

    conv_rows, auc_lookup = compute_convergence_metrics(runs)
    het_rows, hid_lookup = compute_heterogeneity_metrics(runs, auc_lookup, analysis_cfg)
    ftu_rows, ftu_lookup = compute_final_utility_metrics(runs)
    stab_rows = compute_stability_metrics(runs, ftu_lookup, expected_seeds)
    curve_rows = compute_learning_curves_table(runs)
    summary_rows, coverage_rows = compute_summary_and_coverage_tables(
        runs, auc_lookup, hid_lookup, ftu_lookup, analysis_cfg
    )
    t1us_outputs = compute_hierarchical_t1us_scores(runs, auc_lookup, analysis_cfg)

    written: Dict[str, Path] = {}
    if conv_rows:
        written["convergence"] = _write_csv(
            metrics_dir / "convergence.csv", conv_rows, list(conv_rows[0].keys())
        )
    if het_rows:
        written["heterogeneity"] = _write_csv(
            metrics_dir / "heterogeneity.csv", het_rows, list(het_rows[0].keys())
        )
    if ftu_rows:
        written["final_utility"] = _write_csv(
            metrics_dir / "final_utility.csv", ftu_rows, list(ftu_rows[0].keys())
        )
    if stab_rows:
        written["stability"] = _write_csv(
            metrics_dir / "stability.csv", stab_rows, list(stab_rows[0].keys())
        )
    if curve_rows:
        written["learning_curves"] = _write_csv(
            metrics_dir / "learning_curves.csv", curve_rows, list(curve_rows[0].keys())
        )
    if t1us_outputs.get("relative_utility"):
        rel_rows = t1us_outputs["relative_utility"]
        written["relative_utility"] = _write_csv(
            metrics_dir / "relative_utility.csv", rel_rows, list(rel_rows[0].keys())
        )
    if t1us_outputs.get("dataset_scores"):
        ds_rows = t1us_outputs["dataset_scores"]
        written["dataset_scores"] = _write_csv(
            metrics_dir / "dataset_scores.csv", ds_rows, list(ds_rows[0].keys())
        )
    if t1us_outputs.get("modality_scores"):
        mod_rows = t1us_outputs["modality_scores"]
        written["modality_scores"] = _write_csv(
            metrics_dir / "modality_scores.csv", mod_rows, list(mod_rows[0].keys())
        )
    if summary_rows:
        written["track1_summary"] = _write_csv(
            tables_dir / "track1_summary.csv", summary_rows, list(summary_rows[0].keys())
        )
    if coverage_rows:
        written["experiment_coverage"] = _write_csv(
            tables_dir / "experiment_coverage.csv",
            coverage_rows,
            list(coverage_rows[0].keys()),
        )
    if t1us_outputs.get("overall_ranking"):
        rank_rows = t1us_outputs["overall_ranking"]
        written["overall_track1_ranking"] = _write_csv(
            tables_dir / "overall_track1_ranking.csv",
            rank_rows,
            list(rank_rows[0].keys()),
        )

    return written
