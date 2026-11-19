"""Track-specific metric analysis, analytical communication accounting, and rankings."""

import csv
import math
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fl_bench.metrics.schema import (
    METHOD_PASSES_PER_STEP,
    MODEL_STATE_BYTES,
    MODEL_TRAINABLE_BYTES,
    compute_round_comm_bytes,
)

RESOURCE_BASES: Tuple[str, ...] = (
    "rounds",
    "client_updates",
    "upload_bytes",
    "download_bytes",
    "total_bytes",
    "local_steps",
    "forward_passes",
    "backward_passes",
)


def _quartiles_and_iqr(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return `(median, q1, q3, iqr)` using linear quantile interpolation."""
    if not values:
        nan = float("nan")
        return nan, nan, nan, nan
    s = sorted(float(x) for x in values)
    n = len(s)
    if n == 1:
        return s[0], s[0], s[0], 0.0

    def _pct(p: float) -> float:
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return s[lo]
        return s[lo] + (pos - lo) * (s[hi] - s[lo])

    med = _pct(0.5)
    q1 = _pct(0.25)
    q3 = _pct(0.75)
    return med, q1, q3, (q3 - q1)


def _quartiles(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return `(median, q1, q3)` using linear quantile interpolation."""
    med, q1, q3, _ = _quartiles_and_iqr(values)
    return med, q1, q3


def enrich_trajectory_accounting(
    rows: Sequence[Dict[str, Any]],
    *,
    method: Optional[str] = None,
    model_name: Optional[str] = None,
    default_clients_per_round: Optional[int] = None,
    recompute_analytical_bytes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ensure every round row has `cumulative_fit_clients` and method-aware analytical
    `round_upload_bytes`, `round_download_bytes`, `cumulative_upload_bytes`,
    `cumulative_download_bytes`, and `cumulative_local_steps`.
    """
    enriched: List[Dict[str, Any]] = []
    cum_clients = 0
    cum_up = 0
    cum_down = 0
    cum_steps = 0
    cum_fwd = 0
    cum_bwd = 0

    state_b = MODEL_STATE_BYTES.get(model_name or "", 0)
    train_b = MODEL_TRAINABLE_BYTES.get(model_name or "", state_b)
    fwd_mult, bwd_mult = METHOD_PASSES_PER_STEP.get(str(method or "").strip().lower(), (1, 1))

    for r in rows:
        row_copy = dict(r)
        rnd_num = int(float(r.get("round") or 1))
        m_fit = int(float(r.get("num_fit_clients") or default_clients_per_round or 0))
        cum_clients_raw = r.get("cumulative_fit_clients")
        if cum_clients_raw not in (None, ""):
            cum_clients = int(float(cum_clients_raw))
        else:
            cum_clients += m_fit
        row_copy["cumulative_fit_clients"] = cum_clients

        r_steps_raw = r.get("round_local_steps")
        c_steps_raw = r.get("cumulative_local_steps")
        r_steps_val = int(float(r_steps_raw)) if r_steps_raw not in (None, "") else 0
        if c_steps_raw not in (None, ""):
            cum_steps = int(float(c_steps_raw))
        elif r_steps_raw not in (None, ""):
            cum_steps += r_steps_val
            row_copy["cumulative_local_steps"] = cum_steps

        c_fwd_raw = r.get("cumulative_forward_passes")
        c_bwd_raw = r.get("cumulative_backward_passes")
        if c_fwd_raw not in (None, ""):
            cum_fwd = int(float(c_fwd_raw))
        else:
            cum_fwd = cum_steps * fwd_mult
            row_copy["round_forward_passes"] = r_steps_val * fwd_mult
            row_copy["cumulative_forward_passes"] = cum_fwd
        if c_bwd_raw not in (None, ""):
            cum_bwd = int(float(c_bwd_raw))
        else:
            cum_bwd = cum_steps * bwd_mult
            row_copy["round_backward_passes"] = r_steps_val * bwd_mult
            row_copy["cumulative_backward_passes"] = cum_bwd

        has_exact_logged_comm = (
            r.get("round_forward_passes") not in (None, "")
            and r.get("round_upload_bytes") not in (None, "")
            and r.get("round_download_bytes") not in (None, "")
        )
        if not has_exact_logged_comm and recompute_analytical_bytes and method and state_b > 0:
            up_b, down_b = compute_round_comm_bytes(
                m_fit,
                state_b,
                method=method,
                trainable_bytes=train_b,
                round_id=rnd_num,
                model_name=str(model_name or "resnet18"),
            )
            cum_up += up_b
            cum_down += down_b
            row_copy["round_upload_bytes"] = up_b
            row_copy["round_download_bytes"] = down_b
            row_copy["cumulative_upload_bytes"] = cum_up
            row_copy["cumulative_download_bytes"] = cum_down
        else:
            up_b = int(float(r.get("round_upload_bytes") or 0))
            down_b = int(float(r.get("round_download_bytes") or 0))
            c_up_raw = r.get("cumulative_upload_bytes")
            c_down_raw = r.get("cumulative_download_bytes")
            cum_up = int(float(c_up_raw)) if c_up_raw not in (None, "") else (cum_up + up_b)
            cum_down = int(float(c_down_raw)) if c_down_raw not in (None, "") else (cum_down + down_b)
            row_copy["cumulative_upload_bytes"] = cum_up
            row_copy["cumulative_download_bytes"] = cum_down

        row_copy["cumulative_total_bytes"] = cum_up + cum_down
        enriched.append(row_copy)

    return enriched
def evaluate_target_efficiency(
    csv_path: str | Path,
    targets: Sequence[float],
    source_label: Optional[str] = None,
    *,
    sustained_checkpoints: int = 3,
    method: Optional[str] = None,
    model_name: Optional[str] = None,
    default_clients_per_round: Optional[int] = None,
    recompute_analytical_bytes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Evaluate target efficiency for each accuracy threshold in `targets`.

    A run achieves sustained target crossing for target `tau` at evaluation checkpoint
    index `j` iff `W = sustained_checkpoints` consecutive evaluation checkpoints satisfy:
        test_accuracy(r_{k_{j + u}}) >= tau   for all u in {0, ..., W - 1}.

    Records both:
      - `crossing_eval_round`: r_{k_j} (first qualifying checkpoint of the sustained window)
      - `confirmation_eval_round`: r_{k_{j + W - 1}} (checkpoint confirming sustained crossing)
    and extracts cumulative resource costs at `crossing_eval_round`.
    """
    path = Path(csv_path)
    with path.open(newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    if not raw_rows:
        return []

    window_w = max(1, int(sustained_checkpoints))
    rows = enrich_trajectory_accounting(
        raw_rows,
        method=method,
        model_name=model_name,
        default_clients_per_round=default_clients_per_round,
        recompute_analytical_bytes=recompute_analytical_bytes,
    )

    eval_rows: List[Tuple[Dict[str, Any], float]] = []
    has_nonfinite_train = False
    for r in rows:
        try:
            tl = float(r.get("train_loss", "nan"))
            if not math.isfinite(tl):
                has_nonfinite_train = True
        except (ValueError, TypeError):
            has_nonfinite_train = True
        try:
            acc = float(r["test_accuracy"])
            if math.isfinite(acc):
                eval_rows.append((r, acc))
        except (KeyError, ValueError, TypeError):
            continue

    final_utility = eval_rows[-1][1] if eval_rows else float("nan")
    best_utility = max((acc for _, acc in eval_rows), default=float("nan"))
    results: List[Dict[str, Any]] = []

    for target in targets:
        tau = float(target)
        first_single_round: Optional[int] = None
        for r, acc in eval_rows:
            if acc >= tau:
                first_single_round = int(r["round"])
                break

        sustained_idx: Optional[int] = None
        if len(eval_rows) >= window_w:
            for j in range(len(eval_rows) - window_w + 1):
                if all(eval_rows[j + u][1] >= tau for u in range(window_w)):
                    sustained_idx = j
                    break

        if sustained_idx is not None:
            hit_row = eval_rows[sustained_idx][0]
            confirm_row = eval_rows[sustained_idx + window_w - 1][0]
            cross_rnd = int(hit_row["round"])
            confirm_rnd = int(confirm_row["round"])
            cli_updates = int(float(hit_row.get("cumulative_fit_clients", 0)))
            up_b = int(float(hit_row.get("cumulative_upload_bytes", 0)))
            down_b = int(float(hit_row.get("cumulative_download_bytes", 0)))
            steps_raw = hit_row.get("cumulative_local_steps")
            steps = int(float(steps_raw)) if steps_raw not in (None, "") else ""
            fwd_raw = hit_row.get("cumulative_forward_passes")
            fwd_p = int(float(fwd_raw)) if fwd_raw not in (None, "") else ""
            bwd_raw = hit_row.get("cumulative_backward_passes")
            bwd_p = int(float(bwd_raw)) if bwd_raw not in (None, "") else ""
            results.append(
                {
                    "target": tau,
                    "target_reached": 1,
                    "failure_reason": "",
                    "sustained_window_checkpoints": window_w,
                    "first_single_crossing_round": first_single_round if first_single_round is not None else "",
                    "crossing_eval_round": cross_rnd,
                    "confirmation_eval_round": confirm_rnd,
                    "rounds_to_target": cross_rnd,
                    "client_updates_to_target": cli_updates,
                    "upload_bytes_to_target": up_b,
                    "download_bytes_to_target": down_b,
                    "total_bytes_to_target": up_b + down_b,
                    "local_steps_to_target": steps,
                    "forward_passes_to_target": fwd_p,
                    "backward_passes_to_target": bwd_p,
                    "final_utility": final_utility,
                    "best_utility": best_utility,
                    "byte_accounting_type": "analytical_serialized_tensor_estimate",
                    "source_csv": source_label or str(path),
                }
            )
        else:
            if has_nonfinite_train or not eval_rows:
                fail_reason = "diverged_or_invalid_run"
            elif first_single_round is not None:
                fail_reason = f"crossed_at_round_{first_single_round}_but_not_sustained_{window_w}_checkpoints"
            else:
                fail_reason = "never_crossed_target"

            results.append(
                {
                    "target": tau,
                    "target_reached": 0,
                    "failure_reason": fail_reason,
                    "sustained_window_checkpoints": window_w,
                    "first_single_crossing_round": first_single_round if first_single_round is not None else "",
                    "crossing_eval_round": "",
                    "confirmation_eval_round": "",
                    "rounds_to_target": "",
                    "client_updates_to_target": "",
                    "upload_bytes_to_target": "",
                    "download_bytes_to_target": "",
                    "total_bytes_to_target": "",
                    "local_steps_to_target": "",
                    "forward_passes_to_target": "",
                    "backward_passes_to_target": "",
                    "final_utility": final_utility,
                    "best_utility": best_utility,
                    "byte_accounting_type": "analytical_serialized_tensor_estimate",
                    "source_csv": source_label or str(path),
                }
            )

    return results
def aggregate_target_runs(
    per_run_rows: Sequence[Dict[str, Any]],
    *,
    method: str,
    dataset: str,
    partition: str,
    local_setting: str = "epochs1",
    target: float,
    expected_seeds: int = 5,
) -> Dict[str, Any]:
    """
    Aggregate seed-level target-efficiency rows into median, Q1, Q3, and IQR
    for a single `(method, dataset, partition, local_setting, target)` cell.
    Failed runs are never treated as zero cost or included in resource medians.
    """
    matching = [
        r for r in per_run_rows if math.isclose(float(r["target"]), float(target))
    ]
    n_valid = len(matching)
    hits = [r for r in matching if int(r.get("target_reached", 0)) == 1]
    n_succ = len(hits)
    n_fail = n_valid - n_succ
    n_transient = sum(
        1
        for r in matching
        if int(r.get("target_reached", 0)) == 0
        and str(r.get("failure_reason", "")).startswith("crossed_at_round_")
    )
    n_never = sum(
        1
        for r in matching
        if int(r.get("target_reached", 0)) == 0
        and str(r.get("failure_reason", "")) == "never_crossed_target"
    )
    success_rate = (n_succ / n_valid) if n_valid > 0 else 0.0

    if n_valid == 0:
        elig_status = "missing_runs"
        succ_summary = f"0/{expected_seeds} (MISSING)"
    elif n_succ == expected_seeds and n_valid == expected_seeds:
        elig_status = "full_success"
        succ_summary = f"{n_succ}/{n_valid} (100%)"
    elif n_succ > 0:
        elig_status = "partial_success"
        succ_summary = f"{n_succ}/{n_valid} ({100.0 * success_rate:.0f}%, PARTIAL)"
    else:
        elig_status = "zero_success"
        succ_summary = f"0/{n_valid} (0%, FAILED)"

    def _stats_for(col: str) -> Tuple[float, float, float, float]:
        vals = [
            float(r[col])
            for r in hits
            if r.get(col) not in (None, "") and math.isfinite(float(r[col]))
        ]
        return _quartiles_and_iqr(vals)

    r_med, r_q1, r_q3, r_iqr = _stats_for("rounds_to_target")
    c_med, c_q1, c_q3, c_iqr = _stats_for("confirmation_eval_round")
    u_med, u_q1, u_q3, u_iqr = _stats_for("client_updates_to_target")
    up_med, up_q1, up_q3, up_iqr = _stats_for("upload_bytes_to_target")
    dn_med, dn_q1, dn_q3, dn_iqr = _stats_for("download_bytes_to_target")
    t_med, t_q1, t_q3, t_iqr = _stats_for("total_bytes_to_target")
    s_med, s_q1, s_q3, s_iqr = _stats_for("local_steps_to_target")
    fp_med, fp_q1, fp_q3, fp_iqr = _stats_for("forward_passes_to_target")
    bp_med, bp_q1, bp_q3, bp_iqr = _stats_for("backward_passes_to_target")

    utilities = [
        float(r["final_utility"])
        for r in matching
        if r.get("final_utility") not in (None, "")
        and math.isfinite(float(r["final_utility"]))
    ]
    u_mean = statistics.mean(utilities) if utilities else float("nan")
    u_std = (
        statistics.pstdev(utilities)
        if len(utilities) > 1
        else (0.0 if utilities else float("nan"))
    )

    return {
        "algorithm": method,
        "dataset": dataset,
        "partition": partition,
        "local_setting": local_setting,
        "target": target,
        "num_runs": n_valid,
        "num_successful": n_succ,
        "n_seeds_expected": expected_seeds,
        "n_seeds_valid": n_valid,
        "n_seeds_succeeded": n_succ,
        "n_seeds_failed": n_fail,
        "n_seeds_transient_only": n_transient,
        "n_seeds_never_crossed": n_never,
        "success_rate": success_rate,
        "eligibility_status": elig_status,
        "success_summary": succ_summary,
        "rounds_to_target_median": r_med,
        "rounds_to_target_q1": r_q1,
        "rounds_to_target_q3": r_q3,
        "rounds_to_target_iqr": r_iqr,
        "confirmation_round_median": c_med,
        "confirmation_round_q1": c_q1,
        "confirmation_round_q3": c_q3,
        "confirmation_round_iqr": c_iqr,
        "client_updates_to_target_median": u_med,
        "client_updates_to_target_q1": u_q1,
        "client_updates_to_target_q3": u_q3,
        "client_updates_to_target_iqr": u_iqr,
        "upload_bytes_to_target_median": up_med,
        "upload_bytes_to_target_q1": up_q1,
        "upload_bytes_to_target_q3": up_q3,
        "upload_bytes_to_target_iqr": up_iqr,
        "download_bytes_to_target_median": dn_med,
        "download_bytes_to_target_q1": dn_q1,
        "download_bytes_to_target_q3": dn_q3,
        "download_bytes_to_target_iqr": dn_iqr,
        "total_bytes_to_target_median": t_med,
        "total_bytes_to_target_q1": t_q1,
        "total_bytes_to_target_q3": t_q3,
        "total_bytes_to_target_iqr": t_iqr,
        "local_steps_to_target_median": s_med,
        "local_steps_to_target_q1": s_q1,
        "local_steps_to_target_q3": s_q3,
        "local_steps_to_target_iqr": s_iqr,
        "forward_passes_to_target_median": fp_med,
        "forward_passes_to_target_q1": fp_q1,
        "forward_passes_to_target_q3": fp_q3,
        "forward_passes_to_target_iqr": fp_iqr,
        "backward_passes_to_target_median": bp_med,
        "backward_passes_to_target_q1": bp_q1,
        "backward_passes_to_target_q3": bp_q3,
        "backward_passes_to_target_iqr": bp_iqr,
        "final_utility_mean": u_mean,
        "final_utility_std": u_std,
        "byte_accounting_type": "analytical_serialized_tensor_estimate",
    }
def compute_method_rankings(
    aggregated_rows: Sequence[Dict[str, Any]],
    *,
    strict_full_success_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute cost-basis rankings across methods for a given `(dataset, partition,
    local_setting, target)` condition.
    """
    def _rank_basis(rows: Sequence[Dict[str, Any]], median_col: str) -> List[float]:
        qualifying_keys: List[Tuple[int, float, float]] = []
        row_keys: List[Optional[Tuple[int, float, float]]] = []

        for r in rows:
            status = str(r.get("eligibility_status", ""))
            sr = float(r.get("success_rate", 0.0))
            med = float(r.get(median_col, float("nan")))
            if not math.isfinite(med) or sr <= 0.0 or status in ("zero_success", "missing_runs"):
                row_keys.append(None)
                continue
            if strict_full_success_only and status != "full_success":
                row_keys.append(None)
                continue
            tier = 0 if status == "full_success" else 1
            key = (tier, -round(sr, 6), med)
            row_keys.append(key)
            qualifying_keys.append(key)

        sorted_unique = sorted(set(qualifying_keys))
        ranks: List[float] = []
        for key in row_keys:
            if key is None:
                ranks.append(float("nan"))
            else:
                ranks.append(float(sorted_unique.index(key) + 1))
        return ranks

    r_rounds = _rank_basis(aggregated_rows, "rounds_to_target_median")
    r_updates = _rank_basis(aggregated_rows, "client_updates_to_target_median")
    r_up_bytes = _rank_basis(aggregated_rows, "upload_bytes_to_target_median")
    r_dn_bytes = _rank_basis(aggregated_rows, "download_bytes_to_target_median")
    r_tot_bytes = _rank_basis(aggregated_rows, "total_bytes_to_target_median")
    r_steps = _rank_basis(aggregated_rows, "local_steps_to_target_median")
    r_bwd = _rank_basis(aggregated_rows, "backward_passes_to_target_median")

    rankings: List[Dict[str, Any]] = []
    for idx, row in enumerate(aggregated_rows):
        rr = r_rounds[idx]
        ru = r_updates[idx]
        rup = r_up_bytes[idx]
        rdn = r_dn_bytes[idx]
        rb = r_tot_bytes[idx]
        rs = r_steps[idx]
        rbwd = r_bwd[idx]
        primary_ranks = [x for x in (rr, ru, rb, rs, rbwd) if math.isfinite(x)]
        r_range = (max(primary_ranks) - min(primary_ranks)) if len(primary_ranks) >= 2 else float("nan")
        status = str(row.get("eligibility_status", ""))
        partial_flag = status == "partial_success"
        if status == "full_success":
            note = "Ranked (100% seed success)"
        elif status == "partial_success":
            note = (
                f"Excluded from strict ranking ({row.get('success_summary')})"
                if strict_full_success_only
                else f"Tier-2 partial success ({row.get('success_summary')}); ranked below 100%-success methods"
            )
        elif status == "zero_success":
            note = f"Unranked (0/{row.get('n_seeds_valid', 0)} seeds reached sustained target)"
        else:
            note = "Unranked (no valid runs discovered)"

        out_row = dict(row)
        out_row.update(
            {
                "partial_success_flag": partial_flag,
                "ranking_note": note,
                "rank_rounds": rr,
                "rank_client_updates": ru,
                "rank_upload_bytes": rup,
                "rank_download_bytes": rdn,
                "rank_bytes": rb,
                "rank_total_bytes": rb,
                "rank_local_steps": rs,
                "rank_backward_passes": rbwd,
                "rank_range": r_range,
                "has_rank_reversal": bool(math.isfinite(r_range) and r_range > 0.0),
            }
        )
        rankings.append(out_row)

    return rankings
def compute_pairwise_rank_reversals(
    ranking_rows: Sequence[Dict[str, Any]],
    bases: Sequence[Tuple[str, str, str]] = (
        ("rounds", "rank_rounds", "rounds_to_target_median"),
        ("client_updates", "rank_client_updates", "client_updates_to_target_median"),
        ("total_bytes", "rank_total_bytes", "total_bytes_to_target_median"),
        ("local_steps", "rank_local_steps", "local_steps_to_target_median"),
        ("backward_passes", "rank_backward_passes", "backward_passes_to_target_median"),
    ),
) -> List[Dict[str, Any]]:
    """
    Detect all pairwise algorithm rank reversals across resource bases within
    each `(dataset, partition, local_setting, target)` condition.
    """
    reversals: List[Dict[str, Any]] = []
    groups: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]] = {}
    for r in ranking_rows:
        key = (
            str(r.get("dataset", "")),
            str(r.get("partition", "")),
            str(r.get("local_setting", "")),
            float(r.get("target", 0.0)),
        )
        groups.setdefault(key, []).append(r)

    for (dataset, partition, local_setting, target), rows in groups.items():
        for b_idx_1 in range(len(bases)):
            for b_idx_2 in range(b_idx_1 + 1, len(bases)):
                b1_name, b1_rank_col, b1_med_col = bases[b_idx_1]
                b2_name, b2_rank_col, b2_med_col = bases[b_idx_2]
                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        m1, m2 = rows[i], rows[j]
                        r1_a = float(m1.get(b1_rank_col, float("nan")))
                        r2_a = float(m2.get(b1_rank_col, float("nan")))
                        r1_b = float(m1.get(b2_rank_col, float("nan")))
                        r2_b = float(m2.get(b2_rank_col, float("nan")))
                        if not all(math.isfinite(v) for v in (r1_a, r2_a, r1_b, r2_b)):
                            continue
                        diff_a = r1_a - r2_a
                        diff_b = r1_b - r2_b
                        if diff_a * diff_b < 0.0:
                            reversals.append(
                                {
                                    "dataset": dataset,
                                    "partition": partition,
                                    "local_setting": local_setting,
                                    "target": target,
                                    "basis_a": b1_name,
                                    "basis_b": b2_name,
                                    "method_1": m1["algorithm"],
                                    "method_2": m2["algorithm"],
                                    "method_1_success_rate": m1["success_rate"],
                                    "method_2_success_rate": m2["success_rate"],
                                    "method_1_rank_a": r1_a,
                                    "method_2_rank_a": r2_a,
                                    "method_1_rank_b": r1_b,
                                    "method_2_rank_b": r2_b,
                                    "method_1_cost_a_median": m1.get(b1_med_col, ""),
                                    "method_2_cost_a_median": m2.get(b1_med_col, ""),
                                    "method_1_cost_b_median": m1.get(b2_med_col, ""),
                                    "method_2_cost_b_median": m2.get(b2_med_col, ""),
                                }
                            )
    return reversals


def summarize_run_csv(csv_path: str | Path) -> Dict[str, Any]:
    """Compute summary statistics (final, best, last-10 mean accuracy) for a run CSV."""
    path = Path(csv_path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {}

    eval_pairs = []
    for r in rows:
        try:
            rnd = int(r["round"])
            acc = float(r["test_accuracy"])
            loss = float(r["test_loss"])
            if math.isfinite(acc) and math.isfinite(loss):
                eval_pairs.append((rnd, acc, loss))
        except (KeyError, ValueError, TypeError):
            continue

    final_acc = float(rows[-1].get("test_accuracy", "nan"))
    final_loss = float(rows[-1].get("test_loss", "nan"))
    if eval_pairs:
        best_rnd, best_acc, _ = max(eval_pairs, key=lambda x: x[1])
        last10 = [x[1] for x in eval_pairs[-10:]]
        mean_last10 = sum(last10) / len(last10)
    else:
        best_rnd, best_acc, mean_last10 = 0, float("nan"), float("nan")

    return {
        "rounds": len(rows),
        "final_test_accuracy": final_acc,
        "final_test_loss": final_loss,
        "best_test_accuracy": best_acc,
        "best_round": best_rnd,
        "mean_last10_test_accuracy": mean_last10,
    }
