"""Dataset-agnostic hierarchical geometric-mean scoring (T1-US) and uncertainty estimation."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.common.loader import LoadedRun
from analysis.common.stats import compute_sample_summary


def apply_utility_transform(
    raw_utility: float,
    transform_spec: Optional[Dict[str, Any]] = None,
    *,
    dataset: str = "",
) -> float:
    """
    Validate and apply an explicitly configured utility transformation to a raw
    Learning AUC measurement.

    Supported `transform_spec["type"]`:
    - `"identity"` (default for accuracy-based Learning AUC in `(0, 1]`):
      verifies `0 < raw_utility <= 1.0` and returns `raw_utility`.
    - `"linear_bounds"`: requires explicit `min_value` and `max_value` (`max > min`)
      and maps `(raw_utility - min_value) / (max_value - min_value)` into `(0, 1]`.
    - `"inverse_positive"`: requires `raw_utility > 0` and maps `1.0 / (1.0 + raw_utility)`.
    """
    if not math.isfinite(raw_utility):
        raise ValueError(f"Nonfinite raw utility ({raw_utility}) on dataset '{dataset}'.")

    spec = transform_spec or {"type": "identity"}
    t_type = str(spec.get("type", "identity")).lower()

    if t_type == "identity":
        min_b = float(spec.get("min_valid", 0.0))
        max_b = float(spec.get("max_valid", 1.0))
        if not (min_b < raw_utility <= max_b):
            raise ValueError(
                f"Dataset '{dataset}' utility={raw_utility} is outside valid identity bounds "
                f"({min_b}, {max_b}]. Configure an explicit `utility_transform` if this dataset "
                "does not use accuracy in (0, 1]."
            )
        return float(raw_utility)

    if t_type == "linear_bounds":
        if "min_value" not in spec or "max_value" not in spec:
            raise ValueError(
                f"Dataset '{dataset}' linear_bounds transform requires explicit 'min_value' and 'max_value'."
            )
        lo = float(spec["min_value"])
        hi = float(spec["max_value"])
        if hi <= lo:
            raise ValueError(
                f"Dataset '{dataset}' linear_bounds requires max_value > min_value, got [{lo}, {hi}]."
            )
        transformed = (float(raw_utility) - lo) / (hi - lo)
        if not (0.0 < transformed <= 1.0):
            raise ValueError(
                f"Dataset '{dataset}' transformed utility={transformed} is outside (0, 1]."
            )
        return transformed

    if t_type == "inverse_positive":
        if raw_utility <= 0.0:
            raise ValueError(
                f"Dataset '{dataset}' inverse_positive transform requires raw_utility > 0, got {raw_utility}."
            )
        return 1.0 / (1.0 + float(raw_utility))

    raise ValueError(
        f"Unsupported or unconfigured utility_transform '{t_type}' for dataset '{dataset}'. "
        "Supported transforms: 'identity', 'linear_bounds', 'inverse_positive'."
    )


def _runs_comparable_to_reference(run: LoadedRun, ref_run: LoadedRun) -> Tuple[bool, str]:
    """Verify that `run` and `ref_run` share identical experimental conditions."""
    if not ref_run.is_valid:
        return False, f"Reference run ({ref_run.method}) invalid: {ref_run.status_reason}"
    if not run.is_valid:
        return False, run.status_reason
    if run.dataset != ref_run.dataset or run.partition != ref_run.partition or run.seed != ref_run.seed:
        return False, "Mismatched dataset, partition, or seed against reference run"
    if run.model != ref_run.model:
        return False, f"Model mismatch ({run.model} vs reference {ref_run.model})"
    if run.expected_rounds != ref_run.expected_rounds or run.eval_interval != ref_run.eval_interval:
        return False, "Training rounds or evaluation interval mismatch against reference run"
    if (
        run.local_epochs != ref_run.local_epochs
        or run.batch_size != ref_run.batch_size
    ):
        return False, "Local training budget (local_epochs or batch_size) mismatch against reference run"
    if (
        run.num_clients != ref_run.num_clients
        or run.clients_per_round != ref_run.clients_per_round
    ):
        return False, "Client pool or participation rate mismatch against reference run"
    return True, "Matched by dataset, model, partition, seed, and training/participation configuration"


def _normalize_weights(keys: Sequence[str], weight_map: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return normalized positive weights summing to 1.0 (equal weights by default)."""
    if not keys:
        return {}
    if not weight_map:
        w = 1.0 / len(keys)
        return {k: w for k in keys}
    raw = {}
    for k in keys:
        val = float(weight_map.get(k, 1.0))
        if val <= 0.0 or not math.isfinite(val):
            raise ValueError(f"Invalid aggregation weight for '{k}': {val}")
        raw[k] = val
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}
def _hierarchical_bootstrap_and_jackknife(
    cell_seed_logs: Dict[Any, Dict[int, float]],
    cell_weights: Dict[Any, float],
    required_seeds: Sequence[int],
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """
    Compute uncertainty for a hierarchical geometric mean score:
        S = 100 * exp( sum_{c in cells} w_c * (1 / K) * sum_{s in seeds} log_q_{c, s} )

    1. Matched-Seed Cluster Bootstrap: resamples K seeds with replacement from
       `required_seeds` (preserving cross-partition/cross-dataset seed pairing
       between algorithm m and the reference algorithm) across `bootstrap_samples`
       replicates to obtain `bootstrap_se`, `ci_low`, and `ci_high`.
    2. Delete-1-Seed Jackknife: deterministically drops one seed s at a time
       across all cells to obtain `jackknife_se`.
    """
    seeds = list(required_seeds)
    k = len(seeds)
    cells = list(cell_weights.keys())
    if k < 2 or not cells:
        nan = float("nan")
        return {
            "bootstrap_se": nan,
            "ci_low": nan,
            "ci_high": nan,
            "jackknife_se": nan,
        }

    # Build matrix L of shape (num_cells, K) and weight vector W of shape (num_cells,)
    w_vec = np.array([cell_weights[c] for c in cells], dtype=np.float64)
    w_vec = w_vec / w_vec.sum()
    l_mat = np.array(
        [[cell_seed_logs[c][s] for s in seeds] for c in cells],
        dtype=np.float64,
    )
    # Per-seed weighted log-ratio across cells: shape (K,)
    seed_weighted_logs = w_vec @ l_mat

    if np.allclose(seed_weighted_logs, seed_weighted_logs[0], atol=1e-15):
        val = float(100.0 * np.exp(seed_weighted_logs.mean()))
        return {
            "bootstrap_se": 0.0,
            "ci_low": val,
            "ci_high": val,
            "jackknife_se": 0.0,
        }

    # 1. Matched-seed bootstrap
    rng = np.random.default_rng(bootstrap_seed)
    boot_indices = rng.integers(0, k, size=(bootstrap_samples, k))
    boot_means = seed_weighted_logs[boot_indices].mean(axis=1)
    boot_scores = 100.0 * np.exp(boot_means)
    boot_se = float(np.std(boot_scores, ddof=1))
    alpha = (1.0 - confidence_level) / 2.0
    ci_low = float(np.quantile(boot_scores, alpha))
    ci_high = float(np.quantile(boot_scores, 1.0 - alpha))

    # 2. Delete-1-seed jackknife
    jack_scores = []
    for idx in range(k):
        mask = [j for j in range(k) if j != idx]
        jack_log = float(seed_weighted_logs[mask].mean())
        jack_scores.append(100.0 * math.exp(jack_log))
    jack_arr = np.asarray(jack_scores, dtype=np.float64)
    jack_mean = float(jack_arr.mean())
    jack_se = float(math.sqrt(((k - 1) / k) * float(np.sum((jack_arr - jack_mean) ** 2))))

    return {
        "bootstrap_se": boot_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "jackknife_se": jack_se,
    }
def _fmt(val: Optional[float], digits: int = 4) -> str:
    if val is None or not math.isfinite(float(val)):
        return ""
    return f"{float(val):.{digits}f}"


def compute_hierarchical_t1us_scores(
    runs: Sequence[LoadedRun],
    auc_lookup: Dict[Tuple[str, str, str, int], float],
    analysis_cfg: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute the dataset-agnostic Track 1 Utility Score (T1-US) hierarchy:
    1. `relative_utility`: per-(method, dataset, partition, seed) q_{m,d,h,s} and log(q_{m,d,h,s}).
    2. `dataset_scores`: per-(method, dataset) geometric-mean T1-US across heterogeneous partitions H_d.
    3. `modality_scores`: per-(method, modality) geometric-mean T1-US across datasets D_g in modality g.
    4. `overall_ranking`: per-method overall T1-US across modalities G, sorted by official rank.
    """
    suite_version = str(analysis_cfg.get("suite_version", "track1_v1.0"))
    scoring_version = str(analysis_cfg.get("scoring_version", "t1us_hierarchical_geomean_v1.0"))
    ref_method = str(analysis_cfg.get("reference_method", "fedavg"))
    methods = list(
        dict.fromkeys(
            [str(m) for m in analysis_cfg.get("methods", [])]
            + [r.method for r in runs]
        )
    )
    required_seeds = [int(s) for s in analysis_cfg.get("seeds", [])]

    agg_cfg = analysis_cfg.get("aggregation", {})
    boot_samples = int(agg_cfg.get("bootstrap_samples", 2000))
    boot_seed = int(agg_cfg.get("bootstrap_seed", 2026))
    conf_level = float(agg_cfg.get("confidence_level", 0.95))
    custom_weights = agg_cfg.get("weights", {}) if isinstance(agg_cfg.get("weights"), dict) else {}

    datasets_cfg = analysis_cfg.get("datasets", {})
    enabled_datasets: Dict[str, Dict[str, Any]] = {}
    modalities_order: List[str] = []
    datasets_by_modality: Dict[str, List[str]] = {}

    for ds_name, ds_spec in datasets_cfg.items():
        if not bool(ds_spec.get("enabled", True)):
            continue
        modality = str(ds_spec.get("modality", "default"))
        enabled_datasets[ds_name] = ds_spec
        if modality not in modalities_order:
            modalities_order.append(modality)
        datasets_by_modality.setdefault(modality, []).append(ds_name)

    modality_weights = _normalize_weights(
        modalities_order, custom_weights.get("modalities")
    )
    dataset_weights_by_mod: Dict[str, Dict[str, float]] = {}
    for mod, ds_list in datasets_by_modality.items():
        dataset_weights_by_mod[mod] = _normalize_weights(
            ds_list, (custom_weights.get("datasets") or {}).get(mod)
        )

    het_partitions_by_ds: Dict[str, List[str]] = {}
    part_weights_by_ds: Dict[str, Dict[str, float]] = {}
    for ds_name, ds_spec in enabled_datasets.items():
        all_parts = [str(p) for p in ds_spec.get("partitions", [])]
        iid_base = ds_spec.get("iid_baseline_partition")
        het_parts = ds_spec.get("heterogeneous_partitions")
        if isinstance(het_parts, list) and het_parts:
            h_list = [str(p) for p in het_parts]
        else:
            h_list = [p for p in all_parts if p != iid_base] or list(all_parts)
        het_partitions_by_ds[ds_name] = h_list
        part_weights_by_ds[ds_name] = _normalize_weights(
            h_list, (custom_weights.get("partitions") or {}).get(ds_name)
        )

    run_lookup = {(r.dataset, r.partition, r.method, r.seed): r for r in runs}
    relative_rows: List[Dict[str, Any]] = []
    log_q_map: Dict[Tuple[str, str, str], Dict[int, float]] = {}
    cell_reasons: Dict[Tuple[str, str, str], List[str]] = {}
    for ds_name, ds_spec in enabled_datasets.items():
        modality = str(ds_spec.get("modality", "default"))
        model_name = str(ds_spec.get("model", ""))
        transform_spec = ds_spec.get("utility_transform") or analysis_cfg.get(
            "utility_transform", {"type": "identity"}
        )
        for part in het_partitions_by_ds[ds_name]:
            for method in methods:
                cell_key = (method, ds_name, part)
                log_q_map[cell_key] = {}
                cell_reasons[cell_key] = []
                for seed in required_seeds:
                    run = run_lookup.get((ds_name, part, method, seed))
                    ref_run = run_lookup.get((ds_name, part, ref_method, seed))
                    raw_c_m = auc_lookup.get((ds_name, part, method, seed))
                    raw_c_ref = auc_lookup.get((ds_name, part, ref_method, seed))

                    if run is None:
                        status, reason = "MISSING", f"Missing run ({method}, {ds_name}, {part}, seed={seed})"
                    elif ref_run is None:
                        status, reason = "MISSING_REFERENCE", f"Missing reference run ({ref_method}, {ds_name}, {part}, seed={seed})"
                    else:
                        ok, cmp_reason = _runs_comparable_to_reference(run, ref_run)
                        if not ok:
                            status, reason = run.status if not run.is_valid else "INCOMPATIBLE", cmp_reason
                        elif raw_c_m is None or raw_c_ref is None:
                            status, reason = "INVALID_UTILITY", "Missing Learning AUC value"
                        else:
                            status, reason = "VALID", cmp_reason

                    if status == "VALID" and raw_c_m is not None and raw_c_ref is not None:
                        u_m = apply_utility_transform(raw_c_m, transform_spec, dataset=ds_name)
                        u_ref = apply_utility_transform(raw_c_ref, transform_spec, dataset=ds_name)
                        q_val = u_m / u_ref
                        log_q = math.log(q_val)
                        log_q_map[cell_key][seed] = log_q
                        relative_rows.append({
                            "suite_version": suite_version,
                            "scoring_version": scoring_version,
                            "modality": modality,
                            "dataset": ds_name,
                            "model": model_name,
                            "partition": part,
                            "method": method,
                            "reference_method": ref_method,
                            "seed": seed,
                            "status": "VALID",
                            "utility_method_fraction": _fmt(u_m, 6),
                            "utility_reference_fraction": _fmt(u_ref, 6),
                            "relative_utility_ratio_q": _fmt(q_val, 6),
                            "log_relative_utility": _fmt(log_q, 6),
                            "relative_score_100": _fmt(100.0 * q_val, 4),
                            "units": "ratio_vs_reference",
                            "reason": reason,
                            "method_source_csv": run.source_csv if run else "",
                            "reference_source_csv": ref_run.source_csv if ref_run else "",
                        })
                    else:
                        cell_reasons[cell_key].append(f"seed_{seed}:{reason}")
                        relative_rows.append({
                            "suite_version": suite_version,
                            "scoring_version": scoring_version,
                            "modality": modality,
                            "dataset": ds_name,
                            "model": model_name,
                            "partition": part,
                            "method": method,
                            "reference_method": ref_method,
                            "seed": seed,
                            "status": status,
                            "utility_method_fraction": _fmt(raw_c_m, 6),
                            "utility_reference_fraction": _fmt(raw_c_ref, 6),
                            "relative_utility_ratio_q": "",
                            "log_relative_utility": "",
                            "relative_score_100": "",
                            "units": "ratio_vs_reference",
                            "reason": reason,
                            "method_source_csv": run.source_csv if run else "",
                            "reference_source_csv": ref_run.source_csv if ref_run else "",
                        })
    # 2. Dataset-level T1-US scores
    dataset_score_rows: List[Dict[str, Any]] = []
    ds_score_lookup: Dict[Tuple[str, str], float] = {}
    ds_eligible_lookup: Dict[Tuple[str, str], bool] = {}
    ds_reason_lookup: Dict[Tuple[str, str], str] = {}

    for ds_name, ds_spec in enabled_datasets.items():
        modality = str(ds_spec.get("modality", "default"))
        model_name = str(ds_spec.get("model", ""))
        h_parts = het_partitions_by_ds[ds_name]
        p_weights = part_weights_by_ds[ds_name]
        expected_cells = len(h_parts) * len(required_seeds)

        for method in methods:
            valid_cells = 0
            complete_parts = 0
            missing_notes: List[str] = []
            cell_logs_for_ds: Dict[str, Dict[int, float]] = {}

            for part in h_parts:
                s_map = log_q_map.get((method, ds_name, part), {})
                valid_cells += len(s_map)
                if len(s_map) == len(required_seeds):
                    complete_parts += 1
                    cell_logs_for_ds[part] = s_map
                else:
                    reasons = cell_reasons.get((method, ds_name, part), [])
                    first_r = reasons[0] if reasons else "incomplete seeds"
                    missing_notes.append(f"{part}({len(s_map)}/{len(required_seeds)} seeds: {first_r})")

            eligible = complete_parts == len(h_parts) and len(h_parts) > 0
            ds_eligible_lookup[(method, ds_name)] = eligible

            if eligible:
                mean_log = sum(
                    p_weights[part] * (sum(cell_logs_for_ds[part].values()) / len(required_seeds))
                    for part in h_parts
                )
                score_val = 100.0 * math.exp(mean_log)
                ds_score_lookup[(method, ds_name)] = score_val
                unc = _hierarchical_bootstrap_and_jackknife(
                    cell_logs_for_ds,
                    p_weights,
                    required_seeds,
                    bootstrap_samples=boot_samples,
                    bootstrap_seed=boot_seed,
                    confidence_level=conf_level,
                )
                reason_str = "Complete heterogeneous partition and seed coverage"
                ds_reason_lookup[(method, ds_name)] = reason_str
                dataset_score_rows.append({
                    "suite_version": suite_version,
                    "scoring_version": scoring_version,
                    "modality": modality,
                    "dataset": ds_name,
                    "model": model_name,
                    "method": method,
                    "reference_method": ref_method,
                    "heterogeneous_partitions": ";".join(h_parts),
                    "completed_partitions": f"{complete_parts}/{len(h_parts)}",
                    "valid_matched_runs": f"{valid_cells}/{expected_cells}",
                    "ranking_eligible": True,
                    "t1us_dataset_score": _fmt(score_val, 4),
                    "mean_log_relative_utility": _fmt(mean_log, 6),
                    "t1us_bootstrap_se": _fmt(unc["bootstrap_se"], 4),
                    "t1us_ci95_low": _fmt(unc["ci_low"], 4),
                    "t1us_ci95_high": _fmt(unc["ci_high"], 4),
                    "t1us_jackknife_se": _fmt(unc["jackknife_se"], 4),
                    "units": "index_ref_100",
                    "ineligibility_reason": "",
                })
            else:
                reason_str = f"Incomplete coverage on {ds_name}: " + "; ".join(missing_notes)
                ds_reason_lookup[(method, ds_name)] = reason_str
                dataset_score_rows.append({
                    "suite_version": suite_version,
                    "scoring_version": scoring_version,
                    "modality": modality,
                    "dataset": ds_name,
                    "model": model_name,
                    "method": method,
                    "reference_method": ref_method,
                    "heterogeneous_partitions": ";".join(h_parts),
                    "completed_partitions": f"{complete_parts}/{len(h_parts)}",
                    "valid_matched_runs": f"{valid_cells}/{expected_cells}",
                    "ranking_eligible": False,
                    "t1us_dataset_score": "",
                    "mean_log_relative_utility": "",
                    "t1us_bootstrap_se": "",
                    "t1us_ci95_low": "",
                    "t1us_ci95_high": "",
                    "t1us_jackknife_se": "",
                    "units": "index_ref_100",
                    "ineligibility_reason": reason_str,
                })
    # 3. Modality-level T1-US scores
    modality_score_rows: List[Dict[str, Any]] = []
    mod_score_lookup: Dict[Tuple[str, str], float] = {}
    mod_eligible_lookup: Dict[Tuple[str, str], bool] = {}

    for mod in modalities_order:
        ds_list = datasets_by_modality[mod]
        d_weights = dataset_weights_by_mod[mod]
        for method in methods:
            complete_ds = [d for d in ds_list if ds_eligible_lookup.get((method, d), False)]
            ineligible_ds = [
                ds_reason_lookup.get((method, d), f"Incomplete dataset {d}")
                for d in ds_list
                if not ds_eligible_lookup.get((method, d), False)
            ]
            eligible = len(complete_ds) == len(ds_list) and len(ds_list) > 0
            mod_eligible_lookup[(method, mod)] = eligible

            if eligible:
                cell_logs_for_mod: Dict[Tuple[str, str], Dict[int, float]] = {}
                cell_weights_for_mod: Dict[Tuple[str, str], float] = {}
                for d in ds_list:
                    for part, pw in part_weights_by_ds[d].items():
                        cell_logs_for_mod[(d, part)] = log_q_map[(method, d, part)]
                        cell_weights_for_mod[(d, part)] = d_weights[d] * pw

                mean_log = sum(
                    w * (sum(cell_logs_for_mod[c].values()) / len(required_seeds))
                    for c, w in cell_weights_for_mod.items()
                )
                score_val = 100.0 * math.exp(mean_log)
                mod_score_lookup[(method, mod)] = score_val
                unc = _hierarchical_bootstrap_and_jackknife(
                    cell_logs_for_mod,
                    cell_weights_for_mod,
                    required_seeds,
                    bootstrap_samples=boot_samples,
                    bootstrap_seed=boot_seed,
                    confidence_level=conf_level,
                )
                modality_score_rows.append({
                    "suite_version": suite_version,
                    "scoring_version": scoring_version,
                    "modality": mod,
                    "datasets": ";".join(ds_list),
                    "completed_datasets": f"{len(complete_ds)}/{len(ds_list)}",
                    "method": method,
                    "reference_method": ref_method,
                    "ranking_eligible": True,
                    "t1us_modality_score": _fmt(score_val, 4),
                    "mean_log_relative_utility": _fmt(mean_log, 6),
                    "t1us_bootstrap_se": _fmt(unc["bootstrap_se"], 4),
                    "t1us_ci95_low": _fmt(unc["ci_low"], 4),
                    "t1us_ci95_high": _fmt(unc["ci_high"], 4),
                    "t1us_jackknife_se": _fmt(unc["jackknife_se"], 4),
                    "units": "index_ref_100",
                    "ineligibility_reason": "",
                })
            else:
                modality_score_rows.append({
                    "suite_version": suite_version,
                    "scoring_version": scoring_version,
                    "modality": mod,
                    "datasets": ";".join(ds_list),
                    "completed_datasets": f"{len(complete_ds)}/{len(ds_list)}",
                    "method": method,
                    "reference_method": ref_method,
                    "ranking_eligible": False,
                    "t1us_modality_score": "",
                    "mean_log_relative_utility": "",
                    "t1us_bootstrap_se": "",
                    "t1us_ci95_low": "",
                    "t1us_ci95_high": "",
                    "t1us_jackknife_se": "",
                    "units": "index_ref_100",
                    "ineligibility_reason": " | ".join(ineligible_ds),
                })
    # 4. Overall Track 1 Utility Score (T1-US) & Official Ranking
    total_het_conditions = sum(len(het_partitions_by_ds[d]) for d in enabled_datasets)
    total_expected_het_runs = total_het_conditions * len(required_seeds)
    overall_candidates: List[Tuple[Optional[float], Dict[str, Any]]] = []

    for method in methods:
        complete_mods = [m for m in modalities_order if mod_eligible_lookup.get((method, m), False)]
        complete_ds_all = [d for d in enabled_datasets if ds_eligible_lookup.get((method, d), False)]
        valid_het_runs = sum(
            len(log_q_map.get((method, d, p), {}))
            for d in enabled_datasets
            for p in het_partitions_by_ds[d]
        )
        eligible = len(complete_mods) == len(modalities_order) and len(modalities_order) > 0

        row_dict: Dict[str, Any] = {
            "official_rank": "",
            "method": method,
            "reference_method": ref_method,
            "ranking_eligible": eligible,
            "t1us_overall": "",
            "t1us_bootstrap_se": "",
            "t1us_ci95_low": "",
            "t1us_ci95_high": "",
            "t1us_jackknife_se": "",
        }
        for mod in modalities_order:
            row_dict[f"t1us_modality_{mod}"] = _fmt(mod_score_lookup.get((method, mod)), 4)
        for ds_name in enabled_datasets:
            row_dict[f"t1us_dataset_{ds_name}"] = _fmt(ds_score_lookup.get((method, ds_name)), 4)

        if eligible:
            cell_logs_all: Dict[Tuple[str, str], Dict[int, float]] = {}
            cell_weights_all: Dict[Tuple[str, str], float] = {}
            for mod in modalities_order:
                mw = modality_weights[mod]
                for d in datasets_by_modality[mod]:
                    dw = dataset_weights_by_mod[mod][d]
                    for part, pw in part_weights_by_ds[d].items():
                        cell_logs_all[(d, part)] = log_q_map[(method, d, part)]
                        cell_weights_all[(d, part)] = mw * dw * pw

            mean_log_all = sum(
                w * (sum(cell_logs_all[c].values()) / len(required_seeds))
                for c, w in cell_weights_all.items()
            )
            overall_val = 100.0 * math.exp(mean_log_all)
            unc_all = _hierarchical_bootstrap_and_jackknife(
                cell_logs_all,
                cell_weights_all,
                required_seeds,
                bootstrap_samples=boot_samples,
                bootstrap_seed=boot_seed,
                confidence_level=conf_level,
            )
            row_dict.update({
                "t1us_overall": _fmt(overall_val, 4),
                "t1us_bootstrap_se": _fmt(unc_all["bootstrap_se"], 4),
                "t1us_ci95_low": _fmt(unc_all["ci_low"], 4),
                "t1us_ci95_high": _fmt(unc_all["ci_high"], 4),
                "t1us_jackknife_se": _fmt(unc_all["jackknife_se"], 4),
            })
            inelig_reason = ""
            sort_score: Optional[float] = overall_val
        else:
            inelig_reasons = [
                ds_reason_lookup.get((method, d), f"Incomplete dataset {d}")
                for d in enabled_datasets
                if not ds_eligible_lookup.get((method, d), False)
            ]
            inelig_reason = " | ".join(inelig_reasons)
            sort_score = None

        row_dict.update({
            "completed_modalities": f"{len(complete_mods)}/{len(modalities_order)}",
            "completed_datasets": f"{len(complete_ds_all)}/{len(enabled_datasets)}",
            "valid_heterogeneous_runs": f"{valid_het_runs}/{total_expected_het_runs}",
            "uncertainty_method": f"matched_seed_bootstrap(B={boot_samples})+delete1_seed_jackknife",
            "suite_version": suite_version,
            "scoring_version": scoring_version,
            "units": "index_ref_100",
            "ineligibility_reason": inelig_reason,
        })
        overall_candidates.append((sort_score, row_dict))

    eligible_rows = sorted(
        [(s, r) for s, r in overall_candidates if s is not None],
        key=lambda x: (-float(x[0]), x[1]["method"]),
    )
    ineligible_rows = [r for s, r in overall_candidates if s is None]

    overall_ranking_rows: List[Dict[str, Any]] = []
    for rank_idx, (_, r) in enumerate(eligible_rows, start=1):
        r["official_rank"] = rank_idx
        overall_ranking_rows.append(r)
    overall_ranking_rows.extend(ineligible_rows)

    return {
        "relative_utility": relative_rows,
        "dataset_scores": dataset_score_rows,
        "modality_scores": modality_score_rows,
        "overall_ranking": overall_ranking_rows,
    }
