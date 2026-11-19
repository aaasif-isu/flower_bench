"""
Discrete-event structures, virtual clock, and deterministic priority queue
for FLBench Track 3 (System Heterogeneity, Availability, and Execution Resilience).
"""

from dataclasses import dataclass, field
from enum import Enum
import heapq
from typing import Any, Dict, List, Optional, Tuple

from fl_bench.methods.base import ClientResult


class EventType(str, Enum):
    """Discrete simulation event types processed in virtual-time order."""

    JOB_DISPATCH = "job_dispatch"
    DOWNLOAD_COMPLETE = "download_complete"
    COMPUTE_COMPLETE = "compute_complete"
    CLIENT_FAILURE = "client_failure"
    UPDATE_ARRIVAL = "update_arrival"
    ROUND_TIMEOUT = "round_timeout"
    AGGREGATION = "aggregation"
    EVALUATION = "evaluation"


# Deterministic tie-breaking priority at identical virtual timestamps:
# 1. Phase completions (DOWNLOAD_COMPLETE, COMPUTE_COMPLETE)
# 2. Client failures (CLIENT_FAILURE)
# 3. Update arrivals (UPDATE_ARRIVAL) - processed before timeout at exact boundary
# 4. Synchronous round timeout (ROUND_TIMEOUT)
# 5. Server aggregation (AGGREGATION) - advances server model version
# 6. Checkpoint evaluation (EVALUATION) - evaluates new server model version
# 7. New client dispatches (JOB_DISPATCH) - dispatches on new server model version
EVENT_PRIORITY: Dict[EventType, int] = {
    EventType.DOWNLOAD_COMPLETE: 10,
    EventType.COMPUTE_COMPLETE: 20,
    EventType.CLIENT_FAILURE: 30,
    EventType.UPDATE_ARRIVAL: 40,
    EventType.ROUND_TIMEOUT: 50,
    EventType.AGGREGATION: 60,
    EventType.EVALUATION: 70,
    EventType.JOB_DISPATCH: 80,
}


@dataclass
class ClientJob:
    """
    State and accounting record for a single simulated client training job.

    Each job binds to the immutable server model version (`start_version`)
    available at `dispatch_time`, ensuring zero future information leakage.
    """

    job_id: int
    client_id: int
    cohort_or_step_id: int
    start_version: int
    job_seed: int
    dispatch_time: float
    download_complete_time: float
    compute_complete_time: float
    arrival_time: float
    download_duration: float
    compute_duration: float
    upload_duration: float
    latency_seconds: float
    compute_speed: float
    download_bw_mbps: float
    upload_bw_mbps: float
    download_bytes: int
    upload_bytes: int
    work_units: float
    planned_local_steps: int
    planned_forward_passes: int
    planned_backward_passes: int
    num_examples: int
    model_param_ratio: float = 1.0
    effective_batch_size: float = 64.0
    will_fail: bool = False
    failure_time: Optional[float] = None
    failure_phase: Optional[str] = None
    failure_progress_fraction: float = 1.0

    # Dynamic execution state
    status: str = "dispatched"
    apply_version: Optional[int] = None
    apply_time: Optional[float] = None
    staleness: Optional[int] = None
    staleness_weight: Optional[float] = None
    useful_local_steps: int = 0
    wasted_local_steps: int = 0
    useful_forward_passes: int = 0
    wasted_forward_passes: int = 0
    useful_backward_passes: int = 0
    wasted_backward_passes: int = 0
    useful_compute_time: float = 0.0
    wasted_compute_time: float = 0.0
    actual_download_bytes: int = 0
    actual_upload_bytes: int = 0
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    client_result: Optional[ClientResult] = field(default=None, repr=False)

    def to_event_row(self) -> Dict[str, Any]:
        """Serialize job record for `events.csv` logging."""
        return {
            "job_id": self.job_id,
            "client_id": self.client_id,
            "cohort_or_step_id": self.cohort_or_step_id,
            "start_version": self.start_version,
            "apply_version": "" if self.apply_version is None else self.apply_version,
            "staleness": "" if self.staleness is None else self.staleness,
            "staleness_weight": (
                "" if self.staleness_weight is None else f"{self.staleness_weight:.6f}"
            ),
            "status": self.status,
            "dispatch_time": f"{self.dispatch_time:.6f}",
            "download_complete_time": f"{self.download_complete_time:.6f}",
            "compute_complete_time": f"{self.compute_complete_time:.6f}",
            "arrival_time": f"{self.arrival_time:.6f}",
            "apply_time": "" if self.apply_time is None else f"{self.apply_time:.6f}",
            "failure_time": "" if self.failure_time is None else f"{self.failure_time:.6f}",
            "failure_phase": self.failure_phase or "",
            "download_duration": f"{self.download_duration:.6f}",
            "compute_duration": f"{self.compute_duration:.6f}",
            "upload_duration": f"{self.upload_duration:.6f}",
            "latency_seconds": f"{self.latency_seconds:.6f}",
            "compute_speed": f"{self.compute_speed:.6f}",
            "download_bw_mbps": f"{self.download_bw_mbps:.6f}",
            "upload_bw_mbps": f"{self.upload_bw_mbps:.6f}",
            "download_bytes": self.download_bytes,
            "upload_bytes": self.upload_bytes,
            "actual_download_bytes": self.actual_download_bytes,
            "actual_upload_bytes": self.actual_upload_bytes,
            "planned_local_steps": self.planned_local_steps,
            "useful_local_steps": self.useful_local_steps,
            "wasted_local_steps": self.wasted_local_steps,
            "useful_compute_time": f"{self.useful_compute_time:.6f}",
            "wasted_compute_time": f"{self.wasted_compute_time:.6f}",


EVENT_CSV_FIELDS: Tuple[str, ...] = (
    "job_id",
    "client_id",
    "cohort_or_step_id",
    "start_version",
    "apply_version",
    "staleness",
    "staleness_weight",
    "status",
    "dispatch_time",
    "download_complete_time",
    "compute_complete_time",
    "arrival_time",
    "apply_time",
    "failure_time",
    "failure_phase",
    "download_duration",
    "compute_duration",
    "upload_duration",
    "latency_seconds",
    "compute_speed",
    "download_bw_mbps",
    "upload_bw_mbps",
    "download_bytes",
    "upload_bytes",
    "actual_download_bytes",
    "actual_upload_bytes",
    "planned_local_steps",
    "useful_local_steps",
    "wasted_local_steps",
    "useful_compute_time",
    "wasted_compute_time",
    "num_examples",
    "train_loss",
    "train_accuracy",
)


@dataclass
class SimulationEvent:
    """Single discrete event in the virtual-time simulation queue."""

    timestamp: float
    event_type: EventType
    sequence_id: int
    job_id: Optional[int] = None
    client_id: Optional[int] = None
    round_id: Optional[int] = None
    model_version: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    canceled: bool = False


class VirtualClock:
    """Monotonic virtual clock for deterministic discrete-event simulation."""

    def __init__(self, initial_time: float = 0.0) -> None:
        self._current_time: float = float(initial_time)

    @property
    def current_time(self) -> float:
        return self._current_time

    def advance_to(self, timestamp: float) -> float:
        """Advance virtual time monotonically to `timestamp`."""
        ts = float(timestamp)
        if ts < self._current_time - 1e-12:
            raise RuntimeError(
                f"VirtualClock monotonicity violation: attempted to move backward "
                f"from {self._current_time:.9f}s to {ts:.9f}s."
            )
        if ts > self._current_time:
            self._current_time = ts
        return self._current_time


class EventQueue:
    """
    Deterministic timestamp-ordered priority queue of `SimulationEvent` objects.
    Tie-breaking uses `(timestamp, EVENT_PRIORITY[event_type], sequence_id)`.
    """

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, int, SimulationEvent]] = []
        self._next_seq: int = 0
        self._by_job: Dict[int, List[SimulationEvent]] = {}
        self._by_round: Dict[int, List[SimulationEvent]] = {}

    def push(
        self,
        timestamp: float,
        event_type: EventType,
        *,
        job_id: Optional[int] = None,
        client_id: Optional[int] = None,
        round_id: Optional[int] = None,
        model_version: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> SimulationEvent:
        """Schedule a new event at `timestamp`."""
        self._next_seq += 1
        ev = SimulationEvent(
            timestamp=float(timestamp),
            event_type=event_type,
            sequence_id=self._next_seq,
            job_id=job_id,
            client_id=client_id,
            round_id=round_id,
            model_version=model_version,
            payload=payload,
            canceled=False,
        )
        priority = EVENT_PRIORITY.get(event_type, 100)
        heapq.heappush(self._heap, (ev.timestamp, priority, ev.sequence_id, ev))
        if job_id is not None:
            self._by_job.setdefault(job_id, []).append(ev)
        if round_id is not None:
            self._by_round.setdefault(round_id, []).append(ev)
        return ev

    def cancel_job_events(self, job_id: int) -> int:
        """Cancel all pending future events associated with `job_id`."""
        events = self._by_job.pop(job_id, [])
        count = 0
        for ev in events:
            if not ev.canceled:
                ev.canceled = True
                count += 1
        return count

    def cancel_round_events(
        self,
        round_id: int,
        event_types: Optional[Tuple[EventType, ...]] = None,
    ) -> int:
        """Cancel pending events for `round_id` (optionally filtered by `event_types`)."""
        events = self._by_round.get(round_id, [])
        count = 0
        for ev in events:
            if not ev.canceled and (
                event_types is None or ev.event_type in event_types
            ):
                ev.canceled = True
                count += 1
        return count

    def pop(self) -> Optional[SimulationEvent]:
        """Pop and return the next non-canceled event in chronological order."""
        while self._heap:
            _, _, _, ev = heapq.heappop(self._heap)
            if not ev.canceled:
                return ev
        return None

    def peek_time(self) -> Optional[float]:
        """Return the timestamp of the next non-canceled event without removing it."""
        while self._heap and self._heap[0][3].canceled:
            heapq.heappop(self._heap)
        if not self._heap:
            return None
        return self._heap[0][0]

    def is_empty(self) -> bool:
        return self.peek_time() is None

    def __len__(self) -> int:
        return sum(1 for _, _, _, ev in self._heap if not ev.canceled)

            "num_examples": self.num_examples,
            "train_loss": "" if self.train_loss is None else f"{self.train_loss:.6f}",
            "train_accuracy": (
                "" if self.train_accuracy is None else f"{self.train_accuracy:.6f}"
            ),
        }
