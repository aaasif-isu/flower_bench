"""Canonical experiment result loading and strict validation for FLBench analysis."""

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class LoadedRun:
    """Validated representation of a single canonical experiment run directory."""

    track: str
    dataset: str
    model: str
    partition: str
    method: str
    seed: int
    run_dir: Path
    status: str  # "VALID" | "MISSING" | "EXCLUDED_LEGACY" | "EXCLUDED_LEGACY_NONFINITE" | "INVALID_NONFINITE" | "INCOMPLETE_ROUNDS" | "CONFIG_MISMATCH"
    status_reason: str
    expected_rounds: int
    eval_interval: int
    completed_rounds: int = 0
    method_version: Optional[str] = None
    lr: Optional[float] = None
    momentum: Optional[float] = None
    local_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    num_clients: Optional[int] = None
    clients_per_round: Optional[int] = None
    alpha: Optional[float] = None
    source_csv: Optional[str] = None
    eval_checkpoints: List[Tuple[int, float]] = field(default_factory=list)
    eval_losses: List[Tuple[int, float]] = field(default_factory=list)
    eval_perplexities: List[Tuple[int, float]] = field(default_factory=list)
    train_losses: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == "VALID"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else {}
def load_and_validate_run(
    *,
    results_root: Path,
    track: str,
    dataset: str,
    expected_model: str,
    partition: str,
    method: str,
    seed: int,
    expected_rounds: int,
    expected_eval_interval: int,
    expected_num_clients: Optional[int] = None,
    expected_clients_per_round: Optional[int] = None,
    exclude_legacy_scaffold: bool = True,
) -> LoadedRun:
    """
    Load and validate a single canonical run directory:
        `<results_root>/<track>/<dataset>/<partition>/<method>/seed_<seed>/`
    """
    primary_dir = results_root / track / dataset / partition / method / f"seed_{seed}"
    legacy_scaf_dir = (
        results_root / track / dataset / partition / "scaffold_legacy_mom0.9" / f"seed_{seed}"
    )

    is_legacy_fallback = False
    run_dir = primary_dir
    if not primary_dir.exists() and method == "scaffold" and legacy_scaf_dir.exists():
        run_dir = legacy_scaf_dir
        is_legacy_fallback = True

    if not run_dir.exists():
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=expected_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=primary_dir,
            status="MISSING",
            status_reason="Run directory not found",
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
        )

    metrics_csv = run_dir / "metrics.csv"
    config_yaml = run_dir / "config.yaml"
    metadata_json = run_dir / "metadata.json"

    if not metrics_csv.exists():
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=expected_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="MISSING",
            status_reason="metrics.csv missing in run directory",
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
        )

    cfg = _read_yaml(config_yaml)
    meta = _read_json(metadata_json)

    actual_model = str(cfg.get("model") or meta.get("model") or expected_model)
    actual_method = str(cfg.get("method") or meta.get("method") or method)
    actual_part = str(
        cfg.get("partition_label")
        or meta.get("partition_label")
        or cfg.get("partition_type")
        or partition
    )
    actual_seed = int(cfg.get("training_seed") or meta.get("training_seed") or seed)
    method_version = meta.get("method_version") or cfg.get("method_version")
    lr = float(cfg["lr"]) if cfg.get("lr") is not None else None
    momentum = (
        float(meta["momentum"])
        if meta.get("momentum") is not None
        else (float(cfg["momentum"]) if cfg.get("momentum") is not None else None)
    )
    local_epochs = int(cfg["local_epochs"]) if cfg.get("local_epochs") is not None else None
    batch_size = int(cfg["batch_size"]) if cfg.get("batch_size") is not None else None
    num_clients = int(cfg.get("num_clients") or meta.get("num_clients") or 0) or None
    clients_per_round = (
        int(cfg.get("clients_per_round") or meta.get("clients_per_round") or 0) or None
    )
    alpha = (
        float(cfg["alpha"])
        if cfg.get("alpha") is not None
        else (float(meta["alpha"]) if meta.get("alpha") is not None else None)
    )
    source_csv = meta.get("source_csv") or str(metrics_csv)
    # Check configuration consistency
    if (
        actual_model != expected_model
        or actual_method != method
        or actual_part != partition
        or actual_seed != int(seed)
    ):
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=actual_part,
            method=actual_method,
            seed=actual_seed,
            run_dir=run_dir,
            status="CONFIG_MISMATCH",
            status_reason=(
                f"Config mismatch: found (model={actual_model}, method={actual_method}, "
                f"partition={actual_part}, seed={actual_seed})"
            ),
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            method_version=method_version,
            lr=lr,
            momentum=momentum,
            local_epochs=local_epochs,
            batch_size=batch_size,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            alpha=alpha,
            source_csv=source_csv,
        )

    if expected_num_clients is not None and num_clients not in (None, expected_num_clients):
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="CONFIG_MISMATCH",
            status_reason=f"Expected num_clients={expected_num_clients}, found {num_clients}",
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            num_clients=num_clients,
            source_csv=source_csv,
        )

    if (
        expected_clients_per_round is not None
        and clients_per_round not in (None, expected_clients_per_round)
    ):
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="CONFIG_MISMATCH",
            status_reason=(
                f"Expected clients_per_round={expected_clients_per_round}, found {clients_per_round}"
            ),
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            clients_per_round=clients_per_round,
            source_csv=source_csv,
        )

    with metrics_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    completed_rounds = len(rows)
    if completed_rounds != expected_rounds:
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="INCOMPLETE_ROUNDS",
            status_reason=f"Observed {completed_rounds} rows, expected {expected_rounds} rounds",
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            completed_rounds=completed_rounds,
            method_version=method_version,
            lr=lr,
            momentum=momentum,
            local_epochs=local_epochs,
            batch_size=batch_size,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            alpha=alpha,
            source_csv=source_csv,
        )

    last_completed = str(rows[-1].get("run_completed", "")).lower() == "true"
    if not last_completed:
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="INCOMPLETE_ROUNDS",
            status_reason="Final CSV row has run_completed != True",
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            completed_rounds=completed_rounds,
            source_csv=source_csv,
        )
    eval_checkpoints: List[Tuple[int, float]] = []
    eval_losses: List[Tuple[int, float]] = []
    eval_perplexities: List[Tuple[int, float]] = []
    train_losses: List[Tuple[int, float]] = []
    first_nan_round: Optional[int] = None

    for row in rows:
        rnd = int(row["round"])
        tr_l = float(row.get("train_loss", "nan"))
        train_losses.append((rnd, tr_l))
        if rnd % expected_eval_interval == 0 or rnd == expected_rounds:
            te_l = float(row.get("test_loss", "nan"))
            te_a = float(row.get("test_accuracy", "nan"))
            if not (math.isfinite(te_l) and math.isfinite(te_a) and 0.0 <= te_a <= 1.0):
                if first_nan_round is None:
                    first_nan_round = rnd
            else:
                eval_checkpoints.append((rnd, te_a))
                eval_losses.append((rnd, te_l))
                if "test_perplexity" in row and row["test_perplexity"] not in ("", None):
                    ppl = float(row["test_perplexity"])
                    if math.isfinite(ppl):
                        eval_perplexities.append((rnd, ppl))

    is_legacy_scaf = method == "scaffold" and (
        is_legacy_fallback
        or (method_version is not None and str(method_version).startswith("legacy"))
        or (momentum is not None and not math.isclose(momentum, 0.0))
    )

    if is_legacy_scaf and exclude_legacy_scaffold:
        if first_nan_round is not None:
            status = "EXCLUDED_LEGACY_NONFINITE"
            reason = (
                f"Legacy SCAFFOLD (momentum={momentum}) diverged to NaN at round {first_nan_round}; "
                "excluded from primary aggregation"
            )
        else:
            status = "EXCLUDED_LEGACY"
            reason = (
                f"Legacy SCAFFOLD (momentum={momentum}, version={method_version}) "
                "excluded from primary aggregation; awaiting v2_option2_mom0 (momentum=0.0) run"
            )
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status=status,
            status_reason=reason,
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            completed_rounds=completed_rounds,
            method_version=method_version,
            lr=lr,
            momentum=momentum,
            local_epochs=local_epochs,
            batch_size=batch_size,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            alpha=alpha,
            source_csv=source_csv,
            eval_checkpoints=eval_checkpoints,
            eval_losses=eval_losses,
            eval_perplexities=eval_perplexities,
            train_losses=train_losses,
        )
    if first_nan_round is not None:
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="INVALID_NONFINITE",
            status_reason=f"Nonfinite evaluation loss/accuracy starting at round {first_nan_round}",
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            completed_rounds=completed_rounds,
            method_version=method_version,
            lr=lr,
            momentum=momentum,
            local_epochs=local_epochs,
            batch_size=batch_size,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            alpha=alpha,
            source_csv=source_csv,
            eval_checkpoints=eval_checkpoints,
            eval_losses=eval_losses,
            eval_perplexities=eval_perplexities,
            train_losses=train_losses,
        )

    expected_eval_rounds = [
        r
        for r in range(1, expected_rounds + 1)
        if r % expected_eval_interval == 0 or r == expected_rounds
    ]
    observed_eval_rounds = [r for r, _ in eval_checkpoints]
    if observed_eval_rounds != expected_eval_rounds:
        return LoadedRun(
            track=track,
            dataset=dataset,
            model=actual_model,
            partition=partition,
            method=method,
            seed=seed,
            run_dir=run_dir,
            status="INCOMPLETE_ROUNDS",
            status_reason=(
                f"Observed {len(observed_eval_rounds)} evaluation checkpoints, "
                f"expected {len(expected_eval_rounds)}"
            ),
            expected_rounds=expected_rounds,
            eval_interval=expected_eval_interval,
            completed_rounds=completed_rounds,
            source_csv=source_csv,
        )

    return LoadedRun(
        track=track,
        dataset=dataset,
        model=actual_model,
        partition=partition,
        method=method,
        seed=seed,
        run_dir=run_dir,
        status="VALID",
        status_reason="Complete and verified",
        expected_rounds=expected_rounds,
        eval_interval=expected_eval_interval,
        completed_rounds=completed_rounds,
        method_version=method_version,
        lr=lr,
        momentum=momentum,
        local_epochs=local_epochs,
        batch_size=batch_size,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        alpha=alpha,
        source_csv=source_csv,
        eval_checkpoints=eval_checkpoints,
        eval_losses=eval_losses,
        eval_perplexities=eval_perplexities,
        train_losses=train_losses,
    )


def resolve_analysis_methods(
    analysis_cfg: Dict[str, Any],
    results_root: Path,
    track: Optional[str] = None,
) -> List[str]:
    """
    Resolve the ordered list of FL methods to analyze.
    Starts from `analysis_cfg['methods']` (when provided) and, if `methods` is empty/auto
    or `discover_additional_methods: true` is set, discovers any registered methods
    present on disk under `results_root`.
    """
    from fl_bench.methods import METHOD_REGISTRY

    raw_methods = analysis_cfg.get("methods", [])
    if isinstance(raw_methods, list) and raw_methods:
        methods = [str(m) for m in raw_methods]
    else:
        methods = []

    if not methods or bool(analysis_cfg.get("discover_additional_methods", True)):
        track_name = track or str(analysis_cfg.get("track", "track1"))
        track_dir = (
            results_root
            if results_root.name == track_name
            else (results_root / track_name if (results_root / track_name).exists() else results_root)
        )
        discovered: List[str] = []
        datasets_cfg = analysis_cfg.get("datasets", {})
        for ds_name, ds_spec in datasets_cfg.items():
            if not isinstance(ds_spec, dict) or not bool(ds_spec.get("enabled", True)):
                continue
            for part in ds_spec.get("partitions", []):
                part_dir = track_dir / str(ds_name) / str(part)
                if not part_dir.exists():
                    continue
                for child in sorted(part_dir.iterdir()):
                    if child.is_dir() and child.name in METHOD_REGISTRY and child.name not in methods and child.name not in discovered:
                        discovered.append(child.name)
        methods.extend(discovered)

    return methods


def load_track_runs(analysis_cfg: Dict[str, Any], repo_root: Path) -> List[LoadedRun]:
    """Load and validate all expected runs defined in an analysis YAML configuration."""
    results_root = repo_root / str(analysis_cfg.get("results_root", "results"))
    track = str(analysis_cfg.get("track", "track1"))
    methods = resolve_analysis_methods(analysis_cfg, results_root, track=track)
    seeds = [int(s) for s in analysis_cfg.get("seeds", [])]
    exclude_legacy_scaf = bool(analysis_cfg.get("exclude_legacy_scaffold", True))
    datasets_cfg = analysis_cfg.get("datasets", {})

    runs: List[LoadedRun] = []
    for ds_name, ds_spec in datasets_cfg.items():
        if not bool(ds_spec.get("enabled", True)):
            continue
        model = str(ds_spec["model"])
        rounds = int(ds_spec["rounds"])
        eval_interval = int(ds_spec["eval_interval"])
        num_clients = int(ds_spec["num_clients"]) if ds_spec.get("num_clients") else None
        cpr = int(ds_spec["clients_per_round"]) if ds_spec.get("clients_per_round") else None
        partitions = [str(p) for p in ds_spec.get("partitions", [])]

        for part in partitions:
            for method in methods:
                for seed in seeds:
                    runs.append(
                        load_and_validate_run(
                            results_root=results_root,
                            track=track,
                            dataset=ds_name,
                            expected_model=model,
                            partition=part,
                            method=method,
                            seed=seed,
                            expected_rounds=rounds,
                            expected_eval_interval=eval_interval,
                            expected_num_clients=num_clients,
                            expected_clients_per_round=cpr,
                            exclude_legacy_scaffold=exclude_legacy_scaf,
                        )
                    )
    return runs