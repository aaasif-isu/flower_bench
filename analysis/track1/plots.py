"""Dataset-agnostic publication-quality figure generation for Track 1."""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.common.plotting import (
    apply_publication_style,
    get_dataset_label,
    get_method_color,
    get_method_label,
    get_method_marker,
    get_modality_label,
    get_partition_label,
    save_figure,
)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required analysis CSV not found: '{path}'. Run `--metrics-only` first."
        )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(val: Optional[str]) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _enabled_datasets(analysis_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    datasets_cfg = analysis_cfg.get("datasets", {})
    return {
        str(k): v
        for k, v in datasets_cfg.items()
        if isinstance(v, dict) and bool(v.get("enabled", True))
    }


def plot_overall_ranking(
    ranking_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Plot Overall Track 1 Algorithm Ranking (T1-US) with 95% bootstrap CIs,
    reference line at 100, and explicit visual indication of excluded methods.
    """
    if not ranking_rows:
        return []

    ref_method = str(analysis_cfg.get("reference_method", "fedavg"))
    ref_label = get_method_label(ref_method, analysis_cfg)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    n_methods = len(ranking_rows)
    x = np.arange(n_methods)

    scores: List[float] = []
    err_low: List[float] = []
    err_high: List[float] = []
    eligible_flags: List[bool] = []
    methods_ordered: List[str] = []

    for r in ranking_rows:
        m = r["method"]
        methods_ordered.append(m)
        elig = str(r.get("ranking_eligible", "")).lower() == "true"
        s_val = _to_float(r.get("t1us_overall"))
        ci_lo = _to_float(r.get("t1us_ci95_low"))
        ci_hi = _to_float(r.get("t1us_ci95_high"))
        eligible_flags.append(elig and math.isfinite(s_val))
        if elig and math.isfinite(s_val):
            scores.append(s_val)
            err_low.append(max(0.0, s_val - ci_lo) if math.isfinite(ci_lo) else 0.0)
            err_high.append(max(0.0, ci_hi - s_val) if math.isfinite(ci_hi) else 0.0)
        else:
            scores.append(0.0)
            err_low.append(0.0)
            err_high.append(0.0)

    finite_scores = [s for s, e in zip(scores, eligible_flags) if e]
    y_max = max(finite_scores + [110.0]) * 1.15 if finite_scores else 120.0

    for idx, (m, s_val, e_lo, e_hi, elig, r) in enumerate(
        zip(methods_ordered, scores, err_low, err_high, eligible_flags, ranking_rows)
    ):
        color = get_method_color(m, idx, analysis_cfg)
        if elig:
            ax.bar(
                x[idx],
                s_val,
                width=0.56,
                yerr=[[e_lo], [e_hi]],
                capsize=4.0,
                color=color,
                edgecolor="#222222",
                linewidth=0.8,
                error_kw={"elinewidth": 1.1, "ecolor": "#222222"},
            )
            rank_str = r.get("official_rank", "")
            badge = f"#{rank_str} " if rank_str else ""
            ax.text(
                x[idx],
                s_val + e_hi + y_max * 0.02,
                f"{badge}{s_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.bar(
                x[idx],
                y_max * 0.35,
                width=0.56,
                color="#F0F0F0",
                edgecolor="#888888",
                linewidth=0.9,
                linestyle="--",
                hatch="//",
            )
            ax.text(
                x[idx],
                y_max * 0.18,
                "Excluded\n(Incomplete Coverage)",
                ha="center",
                va="center",
                fontsize=8,
                color="#555555",
                fontweight="bold",
            )

    ax.axhline(
        100.0,
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        label=f"{ref_label} Reference (100.0)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([get_method_label(m, analysis_cfg) for m in methods_ordered])
    ax.set_ylabel("Track 1 Utility Score, T1-US (Ref = 100) $\\uparrow$")
    ax.set_title("Track 1 Overall Benchmark Ranking: Utility Under Statistical Heterogeneity")
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_t1us_overall_ranking", formats, dpi)
def _plot_hierarchical_score_bars(
    score_rows: Sequence[Dict[str, str]],
    *,
    group_key: str,
    score_col: str,
    group_order: Sequence[str],
    group_label_fn,
    methods: Sequence[str],
    analysis_cfg: Dict[str, Any],
    ylabel: str,
    title: str,
    stem: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Generic grouped bar chart for dataset-level or modality-level T1-US scores."""
    if not score_rows or not group_order:
        return []

    ref_method = str(analysis_cfg.get("reference_method", "fedavg"))
    ref_label = get_method_label(ref_method, analysis_cfg)

    fig_w = max(7.2, 2.6 * len(group_order) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.3))
    x = np.arange(len(group_order))
    width = min(0.14, 0.78 / max(1, len(methods)))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    all_scores = [
        _to_float(r.get(score_col))
        for r in score_rows
        if math.isfinite(_to_float(r.get(score_col)))
    ]
    y_max = max(all_scores + [110.0]) * 1.15 if all_scores else 120.0

    for m_idx, method in enumerate(methods):
        color = get_method_color(method, m_idx, analysis_cfg)
        label = get_method_label(method, analysis_cfg)
        means = []
        err_lo = []
        err_hi = []
        inelig_positions = []

        for g_idx, g_name in enumerate(group_order):
            row = next(
                (r for r in score_rows if r.get(group_key) == g_name and r.get("method") == method),
                None,
            )
            elig = row is not None and str(row.get("ranking_eligible", "")).lower() == "true"
            s_val = _to_float(row.get(score_col)) if row else float("nan")
            ci_l = _to_float(row.get("t1us_ci95_low")) if row else float("nan")
            ci_h = _to_float(row.get("t1us_ci95_high")) if row else float("nan")
            if elig and math.isfinite(s_val):
                means.append(s_val)
                err_lo.append(max(0.0, s_val - ci_l) if math.isfinite(ci_l) else 0.0)
                err_hi.append(max(0.0, ci_h - s_val) if math.isfinite(ci_h) else 0.0)
            else:
                means.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                inelig_positions.append(x[g_idx] + offsets[m_idx])

        ax.bar(
            x + offsets[m_idx],
            means,
            width=width * 0.92,
            yerr=[err_lo, err_hi],
            capsize=2.5,
            color=color,
            edgecolor="#222222",
            linewidth=0.6,
            label=label,
            error_kw={"elinewidth": 0.9, "ecolor": "#222222"},
        )
        for pos in inelig_positions:
            ax.bar(
                pos,
                y_max * 0.16,
                width=width * 0.92,
                color="#F0F0F0",
                edgecolor="#888888",
                linewidth=0.6,
                linestyle="--",
                hatch="//",
            )
            ax.text(
                pos,
                y_max * 0.03,
                "Excl.",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                color="#555555",
            )

    ax.axhline(
        100.0,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label=f"{ref_label} Reference (100)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([group_label_fn(g) for g in group_order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem, formats, dpi)
def plot_learning_curves(
    curves_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    error_band: str = "std",
) -> List[Path]:
    """
    Dynamically generate test-accuracy learning curves (mean ± variability band)
    for all configured datasets and partitions without hardcoding dataset names.
    """
    saved: List[Path] = []
    band_col = "test_accuracy_sem_pp" if error_band == "sem" else "test_accuracy_std_pp"
    enabled_ds = _enabled_datasets(analysis_cfg)

    multi_part_ds: List[Tuple[str, List[str]]] = []
    single_part_ds: List[Tuple[str, str]] = []
    for ds_name, ds_spec in enabled_ds.items():
        parts = [str(p) for p in ds_spec.get("partitions", [])]
        present_parts = [
            p for p in parts
            if any(r["dataset"] == ds_name and r["partition"] == p for r in curves_rows)
        ]
        if len(present_parts) > 1:
            multi_part_ds.append((ds_name, present_parts))
        elif len(present_parts) == 1:
            single_part_ds.append((ds_name, present_parts[0]))

    for ds_name, parts in multi_part_ds:
        ds_spec = enabled_ds[ds_name]
        max_r = int(ds_spec.get("rounds", 500))
        ds_label = get_dataset_label(ds_name, analysis_cfg)
        fig, axes = plt.subplots(
            1, len(parts), figsize=(4.0 * len(parts), 3.8), sharey=True, squeeze=False
        )
        for ax, part in zip(axes[0], parts):
            for m_idx, method in enumerate(methods):
                m_rows = [
                    r for r in curves_rows
                    if r["dataset"] == ds_name and r["partition"] == part and r["method"] == method
                ]
                if not m_rows:
                    continue
                rounds = np.array([int(r["round"]) for r in m_rows])
                means = np.array([_to_float(r["test_accuracy_mean_percent"]) for r in m_rows])
                bands = np.array([_to_float(r[band_col]) for r in m_rows])
                color = get_method_color(method, m_idx, analysis_cfg)
                label = get_method_label(method, analysis_cfg)
                ax.plot(rounds, means, color=color, linewidth=1.6, label=label)
                valid_b = np.isfinite(bands)
                if np.any(valid_b):
                    ax.fill_between(
                        rounds[valid_b],
                        (means - bands)[valid_b],
                        (means + bands)[valid_b],
                        color=color,
                        alpha=0.18,
                    )
            ax.set_title(get_partition_label(part, ds_name, analysis_cfg))
            ax.set_xlabel("Communication Round")
            ax.set_xlim(0, max_r)
        axes[0][0].set_ylabel("Test Accuracy (%)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),
                ncol=len(handles),
            )
        fig.suptitle(f"{ds_label}: Test Accuracy Learning Curves (Mean ± 1 SD)", y=1.15)
        fig.tight_layout()
        saved.extend(save_figure(fig, out_dir, f"fig_learning_curves_{ds_name}", formats, dpi))

    if single_part_ds:
        fig, axes = plt.subplots(
            1, len(single_part_ds), figsize=(6.2 * len(single_part_ds), 4.0), squeeze=False
        )
        for ax, (ds_name, part) in zip(axes[0], single_part_ds):
            ds_spec = enabled_ds[ds_name]
            max_r = int(ds_spec.get("rounds", 1000))
            ds_label = get_dataset_label(ds_name, analysis_cfg)
            part_label = get_partition_label(part, ds_name, analysis_cfg)
            for m_idx, method in enumerate(methods):
                m_rows = [
                    r for r in curves_rows
                    if r["dataset"] == ds_name and r["partition"] == part and r["method"] == method
                ]
                if not m_rows:
                    continue
                rounds = np.array([int(r["round"]) for r in m_rows])
                means = np.array([_to_float(r["test_accuracy_mean_percent"]) for r in m_rows])
                bands = np.array([_to_float(r[band_col]) for r in m_rows])
                color = get_method_color(method, m_idx, analysis_cfg)
                label = get_method_label(method, analysis_cfg)
                ax.plot(rounds, means, color=color, linewidth=1.7, label=label)
                valid_b = np.isfinite(bands)
                if np.any(valid_b):
                    ax.fill_between(
                        rounds[valid_b],
                        (means - bands)[valid_b],
                        (means + bands)[valid_b],
                        color=color,
                        alpha=0.18,
                    )
            ax.set_title(f"{ds_label} — {part_label}")
            ax.set_xlabel("Communication Round")
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_xlim(0, max_r)
            ax.legend(loc="lower right")
        fig.tight_layout()
        stem_suffix = "_".join(d for d, _ in single_part_ds)
        saved.extend(save_figure(fig, out_dir, f"fig_learning_curves_{stem_suffix}", formats, dpi))

    return saved
def _plot_grouped_bars(
    summary_rows: Sequence[Dict[str, str]],
    *,
    dataset: str,
    partitions: Sequence[str],
    methods: Sequence[str],
    analysis_cfg: Dict[str, Any],
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    ax: plt.Axes,
) -> None:
    present_methods = [
        m for m in methods
        if any(
            r["dataset"] == dataset
            and r["partition"] in partitions
            and r["method"] == m
            and math.isfinite(_to_float(r.get(mean_col)))
            for r in summary_rows
        )
    ]
    if not present_methods:
        return

    x = np.arange(len(partitions))
    width = min(0.15, 0.78 / max(1, len(present_methods)))
    offsets = (np.arange(len(present_methods)) - (len(present_methods) - 1) / 2.0) * width

    for m_idx, method in enumerate(present_methods):
        means = []
        stds = []
        for part in partitions:
            row = next(
                (
                    r for r in summary_rows
                    if r["dataset"] == dataset and r["partition"] == part and r["method"] == method
                ),
                None,
            )
            m_val = _to_float(row.get(mean_col)) if row else float("nan")
            s_val = _to_float(row.get(std_col)) if row else 0.0
            means.append(m_val if math.isfinite(m_val) else 0.0)
            stds.append(s_val if math.isfinite(s_val) else 0.0)

        ax.bar(
            x + offsets[m_idx],
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=2.5,
            color=get_method_color(method, m_idx, analysis_cfg),
            edgecolor="#222222",
            linewidth=0.6,
            label=get_method_label(method, analysis_cfg),
            error_kw={"elinewidth": 0.9, "ecolor": "#222222"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([get_partition_label(p, dataset, analysis_cfg) for p in partitions])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")


def _plot_metric_by_dataset_partitions(
    summary_rows: Sequence[Dict[str, str]],
    *,
    analysis_cfg: Dict[str, Any],
    methods: Sequence[str],
    mean_col: str,
    std_col: str,
    ylabel: str,
    title_suffix: str,
    stem_prefix: str,
    out_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """Dynamically plot a metric across multi-partition and single-partition datasets."""
    saved: List[Path] = []
    enabled_ds = _enabled_datasets(analysis_cfg)
    multi_part_ds: List[Tuple[str, List[str]]] = []
    single_part_ds: List[Tuple[str, str]] = []

    for ds_name, ds_spec in enabled_ds.items():
        parts = [str(p) for p in ds_spec.get("partitions", [])]
        present = [
            p for p in parts
            if any(
                r["dataset"] == ds_name
                and r["partition"] == p
                and math.isfinite(_to_float(r.get(mean_col)))
                for r in summary_rows
            )
        ]
        if len(present) > 1:
            multi_part_ds.append((ds_name, present))
        elif len(present) == 1:
            single_part_ds.append((ds_name, present[0]))

    for ds_name, parts in multi_part_ds:
        ds_label = get_dataset_label(ds_name, analysis_cfg)
        fig, ax = plt.subplots(figsize=(max(7.5, 2.2 * len(parts)), 4.2))
        _plot_grouped_bars(
            summary_rows,
            dataset=ds_name,
            partitions=parts,
            methods=methods,
            analysis_cfg=analysis_cfg,
            mean_col=mean_col,
            std_col=std_col,
            ylabel=ylabel,
            title=f"{ds_label}: {title_suffix}",
            ax=ax,
        )
        fig.tight_layout()
        saved.extend(save_figure(fig, out_dir, f"{stem_prefix}_{ds_name}", formats, dpi))

    if single_part_ds:
        fig, axes = plt.subplots(
            1, len(single_part_ds), figsize=(5.6 * len(single_part_ds), 4.0), squeeze=False
        )
        for ax, (ds_name, part) in zip(axes[0], single_part_ds):
            ds_label = get_dataset_label(ds_name, analysis_cfg)
            _plot_grouped_bars(
                summary_rows,
                dataset=ds_name,
                partitions=[part],
                methods=methods,
                analysis_cfg=analysis_cfg,
                mean_col=mean_col,
                std_col=std_col,
                ylabel=ylabel,
                title=f"{ds_label}: {title_suffix}",
                ax=ax,
            )
        fig.tight_layout()
        stem_suffix = "_".join(d for d, _ in single_part_ds)
        saved.extend(save_figure(fig, out_dir, f"{stem_prefix}_{stem_suffix}", formats, dpi))

    return saved
def plot_heterogeneity_hid(
    summary_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Dynamically plot Learning AUC and Heterogeneity-Induced Degradation (HID, %)
    for every enabled dataset that defines an `iid_baseline_partition`.
    """
    saved: List[Path] = []
    enabled_ds = _enabled_datasets(analysis_cfg)

    for ds_name, ds_spec in enabled_ds.items():
        iid_part = ds_spec.get("iid_baseline_partition")
        if not iid_part:
            continue
        regimes = [str(p) for p in ds_spec.get("partitions", [])]
        ds_rows = [r for r in summary_rows if r["dataset"] == ds_name]
        if not ds_rows or len(regimes) < 2:
            continue

        ds_label = get_dataset_label(ds_name, analysis_cfg)
        fig, (ax_auc, ax_hid) = plt.subplots(1, 2, figsize=(11.5, 4.2))
        x_reg = np.arange(len(regimes))

        for m_idx, method in enumerate(methods):
            m_auc = []
            s_auc = []
            m_hid = []
            s_hid = []
            for part in regimes:
                row = next(
                    (r for r in ds_rows if r["partition"] == part and r["method"] == method),
                    None,
                )
                m_auc.append(_to_float(row.get("learning_auc_mean_percent")) if row else float("nan"))
                s_auc.append(_to_float(row.get("learning_auc_std_pp")) if row else float("nan"))
                m_hid.append(_to_float(row.get("hid_mean_percent")) if row else float("nan"))
                s_hid.append(_to_float(row.get("hid_std_percent")) if row else float("nan"))

            if not any(math.isfinite(v) for v in m_auc):
                continue

            color = get_method_color(method, m_idx, analysis_cfg)
            marker = get_method_marker(method, m_idx, analysis_cfg)
            label = get_method_label(method, analysis_cfg)

            ax_auc.errorbar(
                x_reg,
                m_auc,
                yerr=[v if math.isfinite(v) else 0.0 for v in s_auc],
                color=color,
                marker=marker,
                linewidth=1.7,
                capsize=3.0,
                label=label,
            )
            ax_hid.errorbar(
                x_reg,
                m_hid,
                yerr=[v if math.isfinite(v) else 0.0 for v in s_hid],
                color=color,
                marker=marker,
                linewidth=1.7,
                capsize=3.0,
                label=label,
            )

        part_labels = [get_partition_label(p, ds_name, analysis_cfg) for p in regimes]
        ax_auc.set_xticks(x_reg)
        ax_auc.set_xticklabels(part_labels)
        ax_auc.set_xlabel("Statistical Heterogeneity Regime")
        ax_auc.set_ylabel("Normalized Learning AUC (%)")
        ax_auc.set_title("Learning AUC vs. Heterogeneity")
        ax_auc.legend(loc="lower left")

        ax_hid.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, label="IID Baseline (0%)")
        ax_hid.set_xticks(x_reg)
        ax_hid.set_xticklabels(part_labels)
        ax_hid.set_xlabel("Statistical Heterogeneity Regime")
        ax_hid.set_ylabel("Heterogeneity-Induced Degradation, HID (%) $\\downarrow$")
        ax_hid.set_title("Relative AUC Degradation (HID vs. IID)")
        ax_hid.legend(loc="upper left")

        fig.suptitle(f"{ds_label}: Impact of Statistical Heterogeneity on Anytime Utility", y=1.03)
        fig.tight_layout()
        saved.extend(save_figure(fig, out_dir, f"fig_heterogeneity_hid_{ds_name}", formats, dpi))

    return saved
def plot_stability(
    stability_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Dynamically plot Cross-Seed Variability (SV, pp) across conditions
    and the Utility vs. Stability trade-off frontier.
    """
    valid_rows = [
        r for r in stability_rows
        if math.isfinite(_to_float(r.get("sv_pp"))) and math.isfinite(_to_float(r.get("ftu_mean_percent")))
    ]
    if not valid_rows:
        return []

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    enabled_ds = _enabled_datasets(analysis_cfg)

    cond_pairs: List[Tuple[str, str]] = []
    for ds_name, ds_spec in enabled_ds.items():
        for part in ds_spec.get("partitions", []):
            if any(r["dataset"] == ds_name and r["partition"] == str(part) for r in valid_rows):
                cond_pairs.append((ds_name, str(part)))

    present_methods = [
        m for m in methods
        if any(r["method"] == m for r in valid_rows)
    ]
    if present_methods and cond_pairs:
        x = np.arange(len(cond_pairs))
        width = min(0.14, 0.78 / len(present_methods))
        offsets = (np.arange(len(present_methods)) - (len(present_methods) - 1) / 2.0) * width
        for m_idx, method in enumerate(present_methods):
            sv_vals = []
            for ds_name, part in cond_pairs:
                row = next(
                    (
                        r for r in valid_rows
                        if r["dataset"] == ds_name and r["partition"] == part and r["method"] == method
                    ),
                    None,
                )
                sv_vals.append(_to_float(row.get("sv_pp")) if row else 0.0)
            ax_bar.bar(
                x + offsets[m_idx],
                sv_vals,
                width=width * 0.92,
                color=get_method_color(method, m_idx, analysis_cfg),
                edgecolor="#222222",
                linewidth=0.6,
                label=get_method_label(method, analysis_cfg),
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(
            [f"{ds.upper()}\n{get_partition_label(p, ds, analysis_cfg)}" for ds, p in cond_pairs],
            fontsize=8,
        )
        ax_bar.set_ylabel("Cross-Seed Variability, SV (pp) $\\downarrow$")
        ax_bar.set_title("Final Accuracy SD Across Seeds")
        ax_bar.legend(loc="upper left")

    for m_idx, method in enumerate(methods):
        m_rows = [r for r in valid_rows if r["method"] == method]
        if not m_rows:
            continue
        xs = [_to_float(r["sv_pp"]) for r in m_rows]
        ys = [_to_float(r["ftu_mean_percent"]) for r in m_rows]
        ax_scatter.scatter(
            xs,
            ys,
            color=get_method_color(method, m_idx, analysis_cfg),
            marker=get_method_marker(method, m_idx, analysis_cfg),
            s=58,
            alpha=0.88,
            edgecolors="#222222",
            linewidths=0.6,
            label=get_method_label(method, analysis_cfg),
        )
    ax_scatter.set_xlabel("Cross-Seed Variability, SV (percentage points) $\\downarrow$")
    ax_scatter.set_ylabel("Mean Final Test Accuracy, FTU (%) $\\uparrow$")
    ax_scatter.set_title("Utility–Stability Frontier Across Conditions")
    ax_scatter.legend(loc="lower left")

    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_cross_seed_stability", formats, dpi)
def run_track1_plots(analysis_cfg: Dict[str, Any], repo_root: Path) -> List[Path]:
    """Generate all configured Track 1 figures dynamically from analysis CSVs."""
    fig_cfg = analysis_cfg.get("figures", {})
    if not bool(fig_cfg.get("enabled", True)):
        return []

    apply_publication_style()
    out_dirs = analysis_cfg.get("output_dirs", {})
    metrics_dir = repo_root / str(out_dirs.get("metrics", "analysis/track1/outputs/metrics"))
    tables_dir = repo_root / str(out_dirs.get("tables", "analysis/track1/outputs/tables"))
    figures_dir = repo_root / str(out_dirs.get("figures", "analysis/track1/outputs/figures"))

    formats = [str(f) for f in fig_cfg.get("formats", ["pdf", "png"])]
    dpi = int(fig_cfg.get("dpi", 300))
    error_band = str(fig_cfg.get("error_band", "std"))
    plots_enabled = fig_cfg.get("plots", {})
    enabled_ds = _enabled_datasets(analysis_cfg)

    summary_rows = _read_csv_rows(tables_dir / "track1_summary.csv")
    stability_rows = _read_csv_rows(metrics_dir / "stability.csv")
    curves_rows = _read_csv_rows(metrics_dir / "learning_curves.csv")
    ranking_rows = _read_csv_rows(tables_dir / "overall_track1_ranking.csv")
    modality_rows = _read_csv_rows(metrics_dir / "modality_scores.csv")
    dataset_rows = _read_csv_rows(metrics_dir / "dataset_scores.csv")

    methods = list(
        dict.fromkeys(
            [str(m) for m in analysis_cfg.get("methods", [])]
            + [r["method"] for r in summary_rows if r.get("method")]
        )
    )

    generated: List[Path] = []
    if bool(plots_enabled.get("overall_ranking", True)):
        generated.extend(
            plot_overall_ranking(ranking_rows, analysis_cfg, figures_dir, formats, dpi)
        )
    if bool(plots_enabled.get("modality_scores", True)):
        modalities_order = list(dict.fromkeys(r["modality"] for r in modality_rows))
        generated.extend(
            _plot_hierarchical_score_bars(
                modality_rows,
                group_key="modality",
                score_col="t1us_modality_score",
                group_order=modalities_order,
                group_label_fn=lambda m: get_modality_label(m, analysis_cfg),
                methods=methods,
                analysis_cfg=analysis_cfg,
                ylabel="Modality T1-US Score (Ref = 100) $\\uparrow$",
                title="Track 1 Modality-Wise Utility Score Comparison",
                stem="fig_t1us_modality_comparison",
                out_dir=figures_dir,
                formats=formats,
                dpi=dpi,
            )
        )
    if bool(plots_enabled.get("dataset_scores", True)):
        ds_order = [d for d in enabled_ds if any(r["dataset"] == d for r in dataset_rows)]
        generated.extend(
            _plot_hierarchical_score_bars(
                dataset_rows,
                group_key="dataset",
                score_col="t1us_dataset_score",
                group_order=ds_order,
                group_label_fn=lambda d: get_dataset_label(d, analysis_cfg),
                methods=methods,
                analysis_cfg=analysis_cfg,
                ylabel="Dataset T1-US Score (Ref = 100) $\\uparrow$",
                title="Track 1 Dataset-Wise Utility Score Comparison",
                stem="fig_t1us_dataset_comparison",
                out_dir=figures_dir,
                formats=formats,
                dpi=dpi,
            )
        )
    if bool(plots_enabled.get("learning_curves", True)):
        generated.extend(
            plot_learning_curves(
                curves_rows, analysis_cfg, figures_dir, methods, formats, dpi, error_band
            )
        )
    if bool(plots_enabled.get("convergence_auc", True)):
        generated.extend(
            _plot_metric_by_dataset_partitions(
                summary_rows,
                analysis_cfg=analysis_cfg,
                methods=methods,
                mean_col="learning_auc_mean_percent",
                std_col="learning_auc_std_pp",
                ylabel="Normalized Learning AUC (%)",
                title_suffix="Convergence Learning AUC",
                stem_prefix="fig_convergence_auc",
                out_dir=figures_dir,
                formats=formats,
                dpi=dpi,
            )
        )
    if bool(plots_enabled.get("heterogeneity_hid", True)):
        generated.extend(
            plot_heterogeneity_hid(summary_rows, analysis_cfg, figures_dir, methods, formats, dpi)
        )
    if bool(plots_enabled.get("final_accuracy", True)):
        generated.extend(
            _plot_metric_by_dataset_partitions(
                summary_rows,
                analysis_cfg=analysis_cfg,
                methods=methods,
                mean_col="ftu_mean_percent",
                std_col="sv_pp",
                ylabel="Final Test Accuracy, FTU (%)",
                title_suffix="Final Test Utility",
                stem_prefix="fig_final_utility",
                out_dir=figures_dir,
                formats=formats,
                dpi=dpi,
            )
        )
    if bool(plots_enabled.get("stability", True)):
        generated.extend(
            plot_stability(stability_rows, analysis_cfg, figures_dir, methods, formats, dpi)
        )

    return generated

