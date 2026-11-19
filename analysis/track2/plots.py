"""
Publication-grade, dataset-agnostic figure generators for FLBench Track 2
(Efficiency at Matched Model Quality).
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common.plotting import (
    apply_publication_style,
    get_dataset_label,
    get_method_color,
    get_method_label,
    get_method_marker,
    get_partition_label,
    save_figure,
)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(val: Any) -> float:
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _enabled_datasets(analysis_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ds_cfg = analysis_cfg.get("datasets", {})
    return {
        str(k): dict(v)
        for k, v in ds_cfg.items()
        if isinstance(v, dict) and bool(v.get("enabled", True))
    }


def plot_efficiency_to_target(
    summary_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Plot grouped median + IQR [Q1, Q3] resource costs to reach each accuracy target
    across three primary bases:
      1. Communication Rounds to Target
      2. Analytical Total Communication Bytes to Target (GB)
      3. Cumulative Local Optimizer Steps to Target (x10^3)
    Failed (0% success) or missing methods are explicitly annotated rather than omitted.
    """
    saved: List[Path] = []
    enabled_ds = _enabled_datasets(analysis_cfg)
    bases = [
        (
            "rounds_to_target",
            1.0,
            "Rounds to Target (Median ± IQR) $\\downarrow$",
            "Communication Rounds",
        ),
        (
            "total_bytes_to_target",
            1e-9,
            "Analytical Total Bytes (GB, Median ± IQR) $\\downarrow$",
            "Analytical Comm. Payload (GB)",
        ),
        (
            "local_steps_to_target",
            1e-3,
            "Local Optimizer Steps ($\\times 10^3$, Median ± IQR) $\\downarrow$",
            "Local Optimizer Steps ($\\times 10^3$)",
        ),
    ]

    for ds_name, ds_spec in enabled_ds.items():
        ds_rows = [r for r in summary_rows if r.get("dataset") == ds_name]
        if not ds_rows:
            continue
        targets = sorted({ _to_float(r.get("target")) for r in ds_rows if math.isfinite(_to_float(r.get("target"))) })
        if not targets:
            continue

        ds_label = get_dataset_label(ds_name, analysis_cfg)
        fig, axes = plt.subplots(1, len(bases), figsize=(13.8, 4.3), squeeze=False)
        x = np.arange(len(targets))
        width = min(0.13, 0.78 / max(1, len(methods)))
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

        for ax, (prefix, scale, ylabel, panel_title) in zip(axes[0], bases):
            med_col = f"{prefix}_median"
            q1_col = f"{prefix}_q1"
            q3_col = f"{prefix}_q3"

            finite_vals = [
                _to_float(r.get(med_col)) * scale
                for r in ds_rows
                if math.isfinite(_to_float(r.get(med_col)))
            ]
            y_max = max(finite_vals) * 1.25 if finite_vals else 100.0

            for m_idx, method in enumerate(methods):
                color = get_method_color(method, m_idx, analysis_cfg)
                label = get_method_label(method, analysis_cfg)
                meds = []
                err_lo = []
                err_hi = []
                annotations: List[Tuple[float, str, bool]] = []

                for t_idx, tau in enumerate(targets):
                    row = next(
                        (
                            r for r in ds_rows
                            if r.get("algorithm") == method
                            and math.isclose(_to_float(r.get("target")), tau)
                        ),
                        None,
                    )
                    status = str(row.get("eligibility_status", "missing_runs")) if row else "missing_runs"
                    m_val = _to_float(row.get(med_col)) * scale if row else float("nan")
                    q1_val = _to_float(row.get(q1_col)) * scale if row else float("nan")
                    q3_val = _to_float(row.get(q3_col)) * scale if row else float("nan")
                    n_s = int(_to_float(row.get("n_seeds_succeeded"))) if row and math.isfinite(_to_float(row.get("n_seeds_succeeded"))) else 0
                    n_v = int(_to_float(row.get("n_seeds_valid"))) if row and math.isfinite(_to_float(row.get("n_seeds_valid"))) else 0

                    if math.isfinite(m_val) and n_s > 0:
                        meds.append(m_val)
                        err_lo.append(max(0.0, m_val - q1_val) if math.isfinite(q1_val) else 0.0)
                        err_hi.append(max(0.0, q3_val - m_val) if math.isfinite(q3_val) else 0.0)
                        if status == "partial_success":
                            annotations.append((x[t_idx] + offsets[m_idx], f"{n_s}/{n_v}", False))
                    else:
                        meds.append(0.0)
                        err_lo.append(0.0)
                        err_hi.append(0.0)
                        tag = "Miss." if status == "missing_runs" else f"0/{n_v}"
                        annotations.append((x[t_idx] + offsets[m_idx], tag, True))

                ax.bar(
                    x + offsets[m_idx],
                    meds,
                    width=width * 0.92,
                    yerr=[err_lo, err_hi],
                    capsize=2.2,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.6,
                    label=label,
                    error_kw={"elinewidth": 0.85, "ecolor": "#222222"},
                )
                for pos, tag, is_fail in annotations:
                    if is_fail:
                        ax.bar(
                            pos,
                            y_max * 0.14,
                            width=width * 0.92,
                            color="#F2F2F2",
                            edgecolor="#888888",
                            linewidth=0.6,
                            linestyle="--",
                            hatch="//",
                        )
                        ax.text(
                            pos,
                            y_max * 0.02,
                            tag,
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            rotation=90,
                            color="#555555",
                        )
                    else:
                        ax.text(
                            pos,
                            y_max * 0.02,
                            tag,
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            rotation=90,
                            color="#111111",
                            fontweight="bold",
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"Target {100.0 * t:.0f}%" for t in targets])
            ax.set_ylabel(ylabel)
            ax.set_title(panel_title)
            ax.set_ylim(0, y_max)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.07),
                ncol=len(handles),
            )
        fig.suptitle(
            f"{ds_label}: Resource Cost to Sustained Target Crossing (3 Consecutive Evaluations)",
            y=1.14,
        )
        fig.tight_layout()
        saved.extend(save_figure(fig, out_dir, f"fig_efficiency_to_target_{ds_name}", formats, dpi))

    return saved
def plot_accuracy_vs_resources(
    curves_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Plot test accuracy trajectories against cumulative resource consumption:
      (a) Communication Rounds, (b) Analytical Total Bytes (GB), and
      (c) Cumulative Local Optimizer Steps (x10^3), with accuracy target lines.
    """
    saved: List[Path] = []
    enabled_ds = _enabled_datasets(analysis_cfg)
    x_specs = [
        ("round", 1.0, "Communication Round", "Accuracy vs. Communication Rounds"),
        (
            "cumulative_total_bytes",
            1e-9,
            "Analytical Cumulative Payload (GB)",
            "Accuracy vs. Analytical Comm. Bytes",
        ),
        (
            "cumulative_local_steps",
            1e-3,
            "Cumulative Local Optimizer Steps ($\\times 10^3$)",
            "Accuracy vs. Local Optimizer Steps",
        ),
    ]

    for ds_name, ds_spec in enabled_ds.items():
        ds_rows = [r for r in curves_rows if r.get("dataset") == ds_name]
        if not ds_rows:
            continue
        ds_label = get_dataset_label(ds_name, analysis_cfg)
        targets = [float(t) * 100.0 for t in ds_spec.get("targets", [])]

        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), sharey=True, squeeze=False)
        for ax, (x_col, scale, xlabel, panel_title) in zip(axes[0], x_specs):
            for m_idx, method in enumerate(methods):
                m_rows = [r for r in ds_rows if r.get("algorithm") == method]
                if not m_rows:
                    continue
                xs = np.array([_to_float(r.get(x_col)) * scale for r in m_rows])
                ys = np.array([_to_float(r.get("test_accuracy_mean_percent")) for r in m_rows])
                stds = np.array([_to_float(r.get("test_accuracy_std_pp")) for r in m_rows])
                color = get_method_color(method, m_idx, analysis_cfg)
                label = get_method_label(method, analysis_cfg)
                ax.plot(xs, ys, color=color, linewidth=1.7, label=label)
                valid_s = np.isfinite(stds)
                if np.any(valid_s):
                    ax.fill_between(
                        xs[valid_s],
                        (ys - stds)[valid_s],
                        (ys + stds)[valid_s],
                        color=color,
                        alpha=0.16,
                    )
            for tau_pct in targets:
                ax.axhline(
                    tau_pct,
                    color="#555555",
                    linestyle="--",
                    linewidth=0.85,
                    alpha=0.75,
                )
            ax.set_xlabel(xlabel)
            ax.set_title(panel_title)
        axes[0][0].set_ylabel("Test Accuracy (%)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.07),
                ncol=len(handles),
            )
        fig.suptitle(f"{ds_label}: Test Accuracy Trajectories Across Resource Bases", y=1.14)
        fig.tight_layout()
        saved.extend(save_figure(fig, out_dir, f"fig_accuracy_vs_resources_{ds_name}", formats, dpi))

    return saved
def plot_rank_comparison(
    ranking_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Plot algorithm rank bump charts across resource accounting bases
    (`Rounds`, `Client Updates`, `Analytical Bytes`, `Local Steps`) per target.
    """
    saved: List[Path] = []
    enabled_ds = _enabled_datasets(analysis_cfg)
    basis_cols = [
        ("rank_rounds", "Rounds"),
        ("rank_client_updates", "Client\nUpdates"),
        ("rank_total_bytes", "Analytical\nBytes"),
        ("rank_local_steps", "Local\nSteps"),
    ]

    for ds_name in enabled_ds:
        ds_rows = [r for r in ranking_rows if r.get("dataset") == ds_name]
        if not ds_rows:
            continue
        targets = sorted({ _to_float(r.get("target")) for r in ds_rows if math.isfinite(_to_float(r.get("target"))) })
        if not targets:
            continue

        ds_label = get_dataset_label(ds_name, analysis_cfg)
        fig, axes = plt.subplots(1, len(targets), figsize=(4.5 * len(targets), 3.9), sharey=True, squeeze=False)
        x_pos = np.arange(len(basis_cols))

        for ax, tau in zip(axes[0], targets):
            t_rows = [r for r in ds_rows if math.isclose(_to_float(r.get("target")), tau)]
            for m_idx, method in enumerate(methods):
                row = next((r for r in t_rows if r.get("algorithm") == method), None)
                if not row:
                    continue
                ranks = [_to_float(row.get(col)) for col, _ in basis_cols]
                if not any(math.isfinite(v) for v in ranks):
                    continue
                color = get_method_color(method, m_idx, analysis_cfg)
                marker = get_method_marker(method, m_idx, analysis_cfg)
                label = get_method_label(method, analysis_cfg)
                ax.plot(
                    x_pos,
                    ranks,
                    color=color,
                    marker=marker,
                    linewidth=1.8,
                    markersize=6,
                    label=label,
                )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([lbl for _, lbl in basis_cols])
            ax.set_title(f"Target {100.0 * tau:.0f}%")
            ax.set_ylim(len(methods) + 0.5, 0.5)
            ax.set_yticks(range(1, len(methods) + 1))

        axes[0][0].set_ylabel("Efficiency Rank (1 = Best) $\\downarrow$")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),
                ncol=len(handles),
            )
        fig.suptitle(f"{ds_label}: Algorithm Rankings Across Resource Accounting Bases", y=1.15)
        fig.tight_layout()
        saved.extend(save_figure(fig, out_dir, f"fig_rank_comparison_{ds_name}", formats, dpi))

    return saved
def plot_success_rate_heatmap(
    summary_rows: Sequence[Dict[str, str]],
    analysis_cfg: Dict[str, Any],
    out_dir: Path,
    methods: Sequence[str],
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    """
    Plot a heatmap of seed success rates (%) across all `(dataset, target)`
    conditions and algorithms, annotating each cell with `k/N` or `Miss.`.
    """
    if not summary_rows:
        return []

    conditions: List[Tuple[str, str, str, float]] = []
    for r in summary_rows:
        key = (
            str(r.get("dataset", "")),
            str(r.get("partition", "")),
            str(r.get("local_setting", "")),
            _to_float(r.get("target")),
        )
        if math.isfinite(key[3]) and key not in conditions:
            conditions.append(key)
    if not conditions:
        return []

    mat = np.full((len(conditions), len(methods)), np.nan)
    cell_texts: List[List[str]] = []

    for c_idx, (ds, part, budget, tau) in enumerate(conditions):
        row_texts: List[str] = []
        for m_idx, method in enumerate(methods):
            row = next(
                (
                    r for r in summary_rows
                    if r.get("dataset") == ds
                    and r.get("partition") == part
                    and r.get("local_setting") == budget
                    and r.get("algorithm") == method
                    and math.isclose(_to_float(r.get("target")), tau)
                ),
                None,
            )
            status = str(row.get("eligibility_status", "missing_runs")) if row else "missing_runs"
            if row and status != "missing_runs":
                sr = _to_float(row.get("success_rate")) * 100.0
                n_s = int(_to_float(row.get("n_seeds_succeeded")))
                n_v = int(_to_float(row.get("n_seeds_valid")))
                mat[c_idx, m_idx] = sr
                row_texts.append(f"{n_s}/{n_v}\n({sr:.0f}%)")
            else:
                row_texts.append("Miss.")
        cell_texts.append(row_texts)

    fig, ax = plt.subplots(figsize=(max(7.2, 1.35 * len(methods)), max(4.2, 0.55 * len(conditions))))
    cmap = plt.cm.YlGn.copy()
    cmap.set_bad(color="#E8E8E8")
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")

    for c_idx in range(len(conditions)):
        for m_idx in range(len(methods)):
            val = mat[c_idx, m_idx]
            txt_color = "#FFFFFF" if (math.isfinite(val) and val >= 65.0) else "#222222"
            ax.text(
                m_idx,
                c_idx,
                cell_texts[c_idx][m_idx],
                ha="center",
                va="center",
                fontsize=8,
                color=txt_color,
            )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([get_method_label(m, analysis_cfg) for m in methods])
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(
        [
            f"{ds.upper()} ({get_partition_label(part, ds, analysis_cfg)}) — {100.0 * tau:.0f}%"
            for ds, part, _, tau in conditions
        ]
    )
    ax.set_xlabel("Federated Optimization Algorithm")
    ax.set_ylabel("Dataset & Target Accuracy Threshold")
    ax.set_title("Track 2 Seed Success Rate Matrix (Sustained 3-Checkpoint Target Crossing)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Sustained Target Success Rate (%)")
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig_success_rate_heatmap", formats, dpi)


def run_track2_plots(analysis_cfg: Dict[str, Any], repo_root: Path) -> List[Path]:
    """Generate all configured Track 2 figures dynamically from analysis CSVs."""
    fig_cfg = analysis_cfg.get("figures", {})
    if not bool(fig_cfg.get("enabled", True)):
        return []

    apply_publication_style()
    out_dirs = analysis_cfg.get("output_dirs", {})
    metrics_dir = repo_root / str(out_dirs.get("metrics", "analysis/track2/outputs/metrics"))
    tables_dir = repo_root / str(out_dirs.get("tables", "analysis/track2/outputs/tables"))
    figures_dir = repo_root / str(out_dirs.get("figures", "analysis/track2/outputs/figures"))

    formats = [str(f) for f in fig_cfg.get("formats", ["pdf", "png"])]
    dpi = int(fig_cfg.get("dpi", 300))
    plots_enabled = fig_cfg.get("plots", {})

    summary_rows = _read_csv_rows(metrics_dir / "efficiency_summary.csv")
    curves_rows = _read_csv_rows(metrics_dir / "learning_curves.csv")
    ranking_rows = _read_csv_rows(tables_dir / "track2_rankings.csv")
    methods = list(
        dict.fromkeys(
            [str(m) for m in analysis_cfg.get("methods", [])]
            + [r["algorithm"] for r in summary_rows if r.get("algorithm")]
        )
    )

    generated: List[Path] = []
    if bool(plots_enabled.get("efficiency_to_target", True)):
        generated.extend(
            plot_efficiency_to_target(summary_rows, analysis_cfg, figures_dir, methods, formats, dpi)
        )
    if bool(plots_enabled.get("accuracy_vs_resources", True)):
        generated.extend(
            plot_accuracy_vs_resources(curves_rows, analysis_cfg, figures_dir, methods, formats, dpi)
        )
    if bool(plots_enabled.get("rank_comparison", True)):
        generated.extend(
            plot_rank_comparison(ranking_rows, analysis_cfg, figures_dir, methods, formats, dpi)
        )
    if bool(plots_enabled.get("success_rate_heatmap", True)):
        generated.extend(
            plot_success_rate_heatmap(summary_rows, analysis_cfg, figures_dir, methods, formats, dpi)
        )
    return generated
