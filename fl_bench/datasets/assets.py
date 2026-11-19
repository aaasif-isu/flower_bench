"""Unified resolution, content validation, inspection, and preparation of FL assets."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from fl_bench.datasets.leaf import validate_natural_client_file
from fl_bench.datasets.partition import (
    balanced_dirichlet_partition,
    iid_partition,
    load_cifar10_labels,
    save_partition,
    validate_partition_file,
)
from fl_bench.datasets.schedule import (
    save_schedule,
    validate_schedule_file,
)
from fl_bench.utils.io import resolve_repo_path


@dataclass
class AssetStatus:
    """Status of a partition, client list, or participation schedule asset."""

    kind: str
    path: str
    resolved_path: Path
    action: str  # "REUSE" | "GENERATE" | "INVALID"
    identity: str
    error: Optional[str] = None


def _legacy_cifar_alpha_tag(alpha: float) -> str:
    """Return legacy Track 1 CIFAR-10 filename tag (`alpha1`, `alpha03`, `alpha01`)."""
    if math.isclose(alpha, 1.0):
        return "alpha1"
    if math.isclose(alpha, 0.3):
        return "alpha03"
    if math.isclose(alpha, 0.1):
        return "alpha01"
    if math.isclose(alpha, 0.5):
        return "dirichlet05"
    return f"alpha{str(alpha).replace('.', '')}"


def resolve_default_partition_file(
    *,
    track: str,
    dataset: str,
    partition_type: str,
    alpha: Optional[float],
    num_clients: int,
    partition_seed: int,
) -> Optional[str]:
    """
    Resolve the partition file path for `(dataset, partition_type, alpha, num_clients, partition_seed)`.
    Reuses existing Track 1 (100c) and Track 4 (50c/100c/500c) production files when
    parameters match, and uses an unambiguous `{dataset}_{type}_{N}c_seed{S}.npz`
    path otherwise so production files are never overwritten.
    """
    if dataset in ("femnist", "shakespeare"):
        return None

    if track != "track4" and num_clients == 100:
        if partition_type == "iid":
            leg_tag = "iid"
        else:
            assert alpha is not None
            leg_tag = _legacy_cifar_alpha_tag(float(alpha))
        leg_rel = f"tracks/track1/partitions/{dataset}_{leg_tag}_seed{partition_seed}.npz"
        if resolve_repo_path(leg_rel).exists():
            return leg_rel

    if track == "track4" and num_clients in (50, 100, 500):
        if partition_type == "iid" or (
            partition_type == "dirichlet"
            and alpha is not None
            and math.isclose(float(alpha), 0.5)
        ):
            kind = "dirichlet" if partition_type == "dirichlet" else "iid"
            t4_rel = f"tracks/track4/partitions/{kind}_{num_clients}c_seed{partition_seed}.npz"
            if resolve_repo_path(t4_rel).exists():
                return t4_rel

    part_track = "track4" if track == "track4" else "track1"
    if partition_type == "iid":
        return f"tracks/{part_track}/partitions/{dataset}_iid_{num_clients}c_seed{partition_seed}.npz"

    assert alpha is not None
    return (
        f"tracks/{part_track}/partitions/"
        f"{dataset}_dirichlet_alpha{float(alpha):g}_{num_clients}c_seed{partition_seed}.npz"
    )


def resolve_default_schedule_file(
    *,
    track: str,
    dataset: str,
    num_clients: int,
    clients_per_round: int,
    rounds: int,
    schedule_seed: int,
) -> str:
    """
    Resolve the participation schedule JSON path for `(num_clients, clients_per_round, rounds, schedule_seed)`.
    Reuses Track 4 500-round production schedules when `rounds == 500` and encodes `{rounds}r`
    in all other schedule filenames.
    """
    if track == "track4":
        if rounds == 500:
            t4_rel = f"tracks/track4/schedules/{num_clients}c_{clients_per_round}cpr_seed{schedule_seed}.json"
            if resolve_repo_path(t4_rel).exists():
                return t4_rel
        return f"tracks/track4/schedules/{num_clients}c_{clients_per_round}cpr_{rounds}r_seed{schedule_seed}.json"

    if dataset == "cifar10":
        return (
            f"tracks/track1/schedules/"
            f"cifar10_{num_clients}c_{clients_per_round}cpr_{rounds}r_seed{schedule_seed}.json"
        )

    return (
        f"tracks/track1/schedules/{dataset}/"
        f"{dataset}_{num_clients}c_{clients_per_round}cpr_{rounds}r_seed{schedule_seed}.json"
    )
def inspect_config_assets(config) -> Tuple[AssetStatus, AssetStatus]:
    """
    Inspect and validate the partition/client-list asset and schedule asset for
    `config` without generating or modifying any files on disk.
    """
    assert config.num_clients is not None
    assert config.clients_per_round is not None
    assert config.rounds is not None
    assert config.partition_type is not None

    p_seed = int(config.partition_seed if config.partition_seed is not None else config.training_seed)
    s_seed = int(config.schedule_seed if config.schedule_seed is not None else config.training_seed)

    # 1. Partition or LEAF natural client list
    if config.dataset in ("femnist", "shakespeare"):
        c_file = config.client_file or ""
        c_resolved = resolve_repo_path(c_file)
        c_ident = f"dataset={config.dataset}, type=natural, N={config.num_clients}"
        if not c_resolved.exists():
            part_status = AssetStatus(
                kind="client_list",
                path=c_file,
                resolved_path=c_resolved,
                action="INVALID",
                identity=c_ident,
                error=f"LEAF natural client list file not found: {c_file}",
            )
        else:
            try:
                validate_natural_client_file(c_resolved, config.num_clients)
                part_status = AssetStatus(
                    kind="client_list",
                    path=c_file,
                    resolved_path=c_resolved,
                    action="REUSE",
                    identity=c_ident,
                )
            except Exception as exc:
                part_status = AssetStatus(
                    kind="client_list",
                    path=c_file,
                    resolved_path=c_resolved,
                    action="INVALID",
                    identity=c_ident,
                    error=str(exc),
                )
    else:
        p_file = config.partition_file or ""
        p_resolved = resolve_repo_path(p_file)
        alpha_str = f", alpha={config.alpha:g}" if config.alpha is not None else ""
        p_ident = (
            f"dataset={config.dataset}, type={config.partition_type}{alpha_str}, "
            f"N={config.num_clients}, seed={p_seed}"
        )
        if not p_resolved.exists():
            part_status = AssetStatus(
                kind="partition",
                path=p_file,
                resolved_path=p_resolved,
                action="GENERATE",
                identity=p_ident,
            )
        else:
            try:
                validate_partition_file(
                    p_resolved,
                    num_clients=config.num_clients,
                    partition_type=config.partition_type,
                    alpha=config.alpha,
                    seed=p_seed,
                    dataset=config.dataset,
                )
                part_status = AssetStatus(
                    kind="partition",
                    path=p_file,
                    resolved_path=p_resolved,
                    action="REUSE",
                    identity=p_ident,
                )
            except Exception as exc:
                part_status = AssetStatus(
                    kind="partition",
                    path=p_file,
                    resolved_path=p_resolved,
                    action="INVALID",
                    identity=p_ident,
                    error=str(exc),
                )
    # 2. Participation schedule
    s_file = config.schedule_file or ""
    s_resolved = resolve_repo_path(s_file)
    s_ident = (
        f"N={config.num_clients}, M={config.clients_per_round}, "
        f"R={config.rounds}, seed={s_seed}"
    )
    if not s_resolved.exists():
        sched_status = AssetStatus(
            kind="schedule",
            path=s_file,
            resolved_path=s_resolved,
            action="GENERATE",
            identity=s_ident,
        )
    else:
        try:
            validate_schedule_file(
                s_resolved,
                num_clients=config.num_clients,
                clients_per_round=config.clients_per_round,
                num_rounds=config.rounds,
                seed=s_seed,
            )
            sched_status = AssetStatus(
                kind="schedule",
                path=s_file,
                resolved_path=s_resolved,
                action="REUSE",
                identity=s_ident,
            )
        except Exception as exc:
            sched_status = AssetStatus(
                kind="schedule",
                path=s_file,
                resolved_path=s_resolved,
                action="INVALID",
                identity=s_ident,
                error=str(exc),
            )

    return part_status, sched_status


def ensure_config_assets(
    config,
    log_fn: Callable[[str], None] = print,
) -> Tuple[Path, Path]:
    """
    Validate existing assets or generate missing partition and schedule assets
    for `config` before training. Rejects corrupted/incompatible files.
    """
    part_status, sched_status = inspect_config_assets(config)

    if part_status.action == "INVALID":
        raise ValueError(part_status.error or f"Invalid partition asset: {part_status.path}")
    if sched_status.action == "INVALID":
        raise ValueError(sched_status.error or f"Invalid schedule asset: {sched_status.path}")

    p_seed = int(config.partition_seed if config.partition_seed is not None else config.training_seed)
    s_seed = int(config.schedule_seed if config.schedule_seed is not None else config.training_seed)

    if part_status.action == "GENERATE":
        assert config.data_dir is not None
        assert config.num_clients is not None
        log_fn(
            f"[ASSET] Generating partition ({part_status.identity}) -> {part_status.path}"
        )
        labels = load_cifar10_labels(config.data_dir)
        if config.partition_type == "iid":
            parts = iid_partition(labels, config.num_clients, p_seed)
        elif config.partition_type == "dirichlet":
            assert config.alpha is not None
            parts = balanced_dirichlet_partition(
                labels, config.num_clients, float(config.alpha), p_seed
            )
        else:
            raise ValueError(
                f"Cannot generate synthetic partition for partition_type='{config.partition_type}'."
            )
        save_partition(
            part_status.resolved_path,
            parts,
            config.partition_type,
            config.alpha,
            p_seed,
            dataset=config.dataset,
            overwrite=False,
        )
        validate_partition_file(
            part_status.resolved_path,
            num_clients=config.num_clients,
            partition_type=config.partition_type,
            alpha=config.alpha,
            seed=p_seed,
            dataset=config.dataset,
        )
    else:
        log_fn(
            f"[ASSET] Reusing verified {part_status.kind} ({part_status.identity}) -> {part_status.path}"
        )

    if sched_status.action == "GENERATE":
        assert config.num_clients is not None
        assert config.clients_per_round is not None
        assert config.rounds is not None
        log_fn(
            f"[ASSET] Generating schedule ({sched_status.identity}) -> {sched_status.path}"
        )
        save_schedule(
            sched_status.resolved_path,
            num_clients=config.num_clients,
            clients_per_round=config.clients_per_round,
            num_rounds=config.rounds,
            seed=s_seed,
            overwrite=False,
        )
        validate_schedule_file(
            sched_status.resolved_path,
            num_clients=config.num_clients,
            clients_per_round=config.clients_per_round,
            num_rounds=config.rounds,
            seed=s_seed,
        )
    else:
        log_fn(
            f"[ASSET] Reusing verified schedule ({sched_status.identity}) -> {sched_status.path}"
        )

    return part_status.resolved_path, sched_status.resolved_path
