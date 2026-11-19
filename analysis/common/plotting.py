"""Shared publication-quality Matplotlib styling, palettes, and dynamic label helpers."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_METHOD_LABELS: Dict[str, str] = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedavgm": "FedAvgM",
    "fedadam": "FedAdam",
    "fednova": "FedNova",
    "scaffold": "SCAFFOLD",
    "fedspeed": "FedSpeed",
    "fedlesam": "FedLESAM",
    "fedgps": "FedGPS",
    "fedwmsam": "FedWMSAM",
    "heterofl": "HeteroFL",
}

# Colorblind-safe Okabe-Ito + Paul Tol palette + fallback cycle
DEFAULT_METHOD_COLORS: Dict[str, str] = {
    "fedavg": "#0072B2",    # Blue
    "fedprox": "#56B4E9",   # Sky Blue
    "fedavgm": "#009E73",   # Bluish Green
    "fedadam": "#D55E00",   # Vermilion
    "fednova": "#CC79A7",   # Reddish Purple
    "scaffold": "#E69F00",  # Orange
    "fedspeed": "#882255",  # Wine
    "fedlesam": "#44AA99",  # Teal
    "fedgps": "#332288",    # Indigo
    "fedwmsam": "#117733",  # Forest Green
    "heterofl": "#999933",  # Olive
}

FALLBACK_COLORS: Sequence[str] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#882255",
    "#44AA99",
    "#332288",
    "#117733",
    "#999933",
)

DEFAULT_METHOD_MARKERS: Dict[str, str] = {
    "fedavg": "o",
    "fedprox": "s",
    "fedavgm": "^",
    "fedadam": "D",
    "fednova": "v",
    "scaffold": "P",
    "fedspeed": "X",
    "fedlesam": "*",
    "fedgps": "h",
    "fedwmsam": "p",
    "heterofl": "d",
}

FALLBACK_MARKERS: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X", "*")

METHOD_LABELS = DEFAULT_METHOD_LABELS
METHOD_COLORS = DEFAULT_METHOD_COLORS
METHOD_MARKERS = DEFAULT_METHOD_MARKERS


def get_method_label(method: str, analysis_cfg: Optional[Dict[str, Any]] = None) -> str:
    custom = (analysis_cfg or {}).get("method_labels", {})
    if isinstance(custom, dict) and method in custom:
        return str(custom[method])
    return DEFAULT_METHOD_LABELS.get(method, method)


def get_method_color(method: str, index: int = 0, analysis_cfg: Optional[Dict[str, Any]] = None) -> str:
    custom = (analysis_cfg or {}).get("method_colors", {})
    if isinstance(custom, dict) and method in custom:
        return str(custom[method])
    if method in DEFAULT_METHOD_COLORS:
        return DEFAULT_METHOD_COLORS[method]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def get_method_marker(method: str, index: int = 0, analysis_cfg: Optional[Dict[str, Any]] = None) -> str:
    custom = (analysis_cfg or {}).get("method_markers", {})
    if isinstance(custom, dict) and method in custom:
        return str(custom[method])
    if method in DEFAULT_METHOD_MARKERS:
        return DEFAULT_METHOD_MARKERS[method]
    return FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)]


def get_dataset_label(dataset: str, analysis_cfg: Optional[Dict[str, Any]] = None) -> str:
    ds_cfg = ((analysis_cfg or {}).get("datasets") or {}).get(dataset, {})
    if isinstance(ds_cfg, dict) and ds_cfg.get("display_name"):
        return str(ds_cfg["display_name"])
    model = ds_cfg.get("model") if isinstance(ds_cfg, dict) else None
    return f"{dataset.upper()} ({model})" if model else dataset.upper()


def get_modality_label(modality: str, analysis_cfg: Optional[Dict[str, Any]] = None) -> str:
    mod_cfg = ((analysis_cfg or {}).get("modalities") or {}).get(modality, {})
    if isinstance(mod_cfg, dict) and mod_cfg.get("display_name"):
        return str(mod_cfg["display_name"])
    return modality.replace("_", " ").title()


def get_partition_label(
    partition: str,
    dataset: Optional[str] = None,
    analysis_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    if dataset and analysis_cfg:
        ds_cfg = (analysis_cfg.get("datasets") or {}).get(dataset, {})
        p_labels = ds_cfg.get("partition_labels", {}) if isinstance(ds_cfg, dict) else {}
        if isinstance(p_labels, dict) and partition in p_labels:
            return str(p_labels[partition])
    global_p = (analysis_cfg or {}).get("partition_labels", {})
    if isinstance(global_p, dict) and partition in global_p:
        return str(global_p[partition])
    if partition.lower() == "iid":
        return "IID"
    if partition.lower() == "natural":
        return "Natural Split"
    if partition.startswith("alpha_"):
        val = partition.split("alpha_", 1)[1]
        return rf"Dirichlet ($\alpha={val}$)"
    return partition


def apply_publication_style() -> None:
    """Configure Matplotlib rcParams for clean, publication-grade figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save `fig` in all requested formats (e.g. vector PDF + high-DPI PNG) and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        ext = fmt.lower().lstrip(".")
        target = out_dir / f"{stem}.{ext}"
        fig.savefig(target, format=ext, dpi=dpi if ext == "png" else None)
        saved.append(target)
    plt.close(fig)
    return saved

