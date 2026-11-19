"""Code provenance, SHA-256 source hashing, and runtime metadata utilities."""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_source_hashes(root: Path = REPO_ROOT) -> Dict[str, str]:
    """Compute SHA-256 digests for all Python files under `fl_bench/`."""
    fl_bench_dir = root / "fl_bench"
    if not fl_bench_dir.exists():
        return {}
    paths = sorted(fl_bench_dir.rglob("*.py"))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def get_git_commit(root: Path = REPO_ROOT) -> Optional[str]:
    """Return the current git commit SHA if available."""
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def file_sha256(path: Optional[str | Path]) -> Optional[str]:
    """Compute SHA-256 digest of a file if it exists on disk."""
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_run_metadata(
    config_dict: Dict[str, Any],
    *,
    run_completed: bool,
    metrics_finite: bool,
    completed_rounds: int,
    elapsed_seconds: Optional[float] = None,
    device_str: Optional[str] = None,
    gpu_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Construct standardized `metadata.json` payload for an experiment run."""
    payload: Dict[str, Any] = {
        "run_id": config_dict.get("run_id") or "",
        "track": config_dict.get("track") or "",
        "experiment_name": config_dict.get("experiment_name") or "",
        "dataset": config_dict.get("dataset"),
        "model": config_dict.get("model"),
        "method": config_dict.get("method"),
        "method_version": config_dict.get("method_version"),
        "lr": config_dict.get("lr"),
        "momentum": config_dict.get("momentum"),
        "partition_type": config_dict.get("partition_type"),
        "partition_label": config_dict.get("partition_label"),
        "alpha": config_dict.get("alpha"),
        "num_clients": config_dict.get("num_clients"),
        "clients_per_round": config_dict.get("clients_per_round"),
        "rounds": config_dict.get("rounds"),
        "training_seed": config_dict.get("training_seed"),
        "partition_file": config_dict.get("partition_file"),
        "schedule_file": config_dict.get("schedule_file"),
        "run_completed": bool(run_completed),
        "metrics_finite": bool(metrics_finite),
        "completed_rounds": int(completed_rounds),
        "elapsed_seconds": (
            round(float(elapsed_seconds), 4)
            if elapsed_seconds is not None
            else None
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "device": device_str,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_commit": get_git_commit(root),
        "source_hashes": compute_source_hashes(root),
    }
    if extra:
        payload.update(extra)
    return payload
