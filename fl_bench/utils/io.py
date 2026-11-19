"""Filesystem I/O, path resolution, completion checks, and run logging."""

import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional, TextIO, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """
    Resolve a repository path, transparently mapping legacy `track{1,2,4}/...`
    references to `tracks/track{1,2,4}/...` or `results/track{1,2,4}/...` when
    files have been organized into the refactored directory hierarchy.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p

    candidate = root / p
    if candidate.exists():
        return candidate

    parts = p.parts
    if parts and parts[0].startswith("track"):
        track_name = parts[0]
        rest = Path(*parts[1:]) if len(parts) > 1 else Path()
        if parts[1:2] == ("results",):
            res_rest = Path(*parts[2:]) if len(parts) > 2 else Path()
            res_candidate = root / "results" / track_name / res_rest
            if res_candidate.exists():
                return res_candidate
        tracks_candidate = root / "tracks" / track_name / rest
        if tracks_candidate.exists():
            return tracks_candidate
        results_candidate = root / "results" / track_name / rest
        if results_candidate.exists():
            return results_candidate

    return candidate


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Atomically write JSON data to `path` via a temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(target)
    return target


def write_yaml_atomic(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Atomically write a configuration dictionary to a YAML file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False)
    except ImportError:
        text = json.dumps(payload, indent=2) + "\n"
    tmp.write_text(text)
    tmp.replace(target)
    return target


def verify_metrics_csv(
    csv_path: str | Path,
    expected_rounds: Optional[int] = None,
    eval_interval: int = 1,
) -> Tuple[bool, bool, int]:
    """
    Inspect a `metrics.csv` file and return `(rows_complete, metrics_finite, num_rows)`.
    """
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return False, False, 0

    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False, False, 0

    if not rows:
        return False, False, 0

    num_rows = len(rows)
    try:
        numbers = [int(r["round"]) for r in rows]
    except (KeyError, ValueError, TypeError):
        return False, False, num_rows

    target_rounds = expected_rounds if expected_rounds is not None else num_rows
    expected_seq = list(range(1, target_rounds + 1))
    last_completed = str(rows[-1].get("run_completed", "")).lower() == "true"
    rows_complete = (
        numbers in (expected_seq, [0] + expected_seq) and last_completed
    )

    metrics_finite = True
    for r in rows:
        rnd = int(r["round"])
        if eval_interval > 1 and (
            rnd % eval_interval != 0 and rnd != target_rounds
        ):
            continue
        try:
            loss = float(r["test_loss"])
            acc = float(r["test_accuracy"])
            if not (
                math.isfinite(loss)
                and math.isfinite(acc)
                and 0.0 <= acc <= 1.0
            ):
                metrics_finite = False
                break
        except (KeyError, ValueError, TypeError):
            metrics_finite = False
            break

    return rows_complete, metrics_finite, num_rows


class RunLock:
    """Exclusive lockfile context manager to prevent concurrent overwrites."""

    def __init__(self, lock_path: str | Path) -> None:
        self.lock_path = Path(lock_path)

    def __enter__(self) -> "RunLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock_path.open("x") as handle:
                job_id = os.environ.get("SLURM_JOB_ID", f"pid-{os.getpid()}")
                handle.write(f"Locked by {job_id}\n")
        except FileExistsError as exc:
            raise RuntimeError(
                f"Run locked: {self.lock_path}. Check active jobs before removing a stale lock."
            ) from exc
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.lock_path.unlink(missing_ok=True)


class TeeLogger:
    """Tee stdout messages to both console and a per-run `run.log` file."""

    def __init__(self, log_path: Optional[str | Path]) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._file: Optional[TextIO] = None

    def __enter__(self) -> "TeeLogger":
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.log_path.open("w", encoding="utf-8")
        return self

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self._file is not None:
            self._file.write(message + "\n")
            self._file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        sys.stdout.flush()
