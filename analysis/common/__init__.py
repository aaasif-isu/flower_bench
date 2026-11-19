"""Shared data loading, validation, statistical aggregation, and plotting utilities."""

from analysis.common.loader import (
    LoadedRun,
    load_and_validate_run,
    load_track_runs,
    resolve_analysis_methods,
)
from analysis.common.plotting import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    apply_publication_style,
    get_dataset_label,
    get_method_color,
    get_method_label,
    get_method_marker,
    get_modality_label,
    get_partition_label,
    save_figure,
)
from analysis.common.scoring import (
    apply_utility_transform,
    compute_hierarchical_t1us_scores,
)
from analysis.common.stats import (
    SampleSummary,
    compute_sample_summary,
    trapezoidal_learning_auc,
)

__all__ = [
    "LoadedRun",
    "METHOD_COLORS",
    "METHOD_LABELS",
    "METHOD_MARKERS",
    "SampleSummary",
    "apply_publication_style",
    "apply_utility_transform",
    "compute_hierarchical_t1us_scores",
    "compute_sample_summary",
    "get_dataset_label",
    "get_method_color",
    "get_method_label",
    "get_method_marker",
    "get_modality_label",
    "get_partition_label",
    "load_and_validate_run",
    "load_track_runs",
    "resolve_analysis_methods",
    "save_figure",
    "trapezoidal_learning_auc",
]
