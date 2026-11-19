"""
Simulated wall-clock time modeling, communication/compute resource accounting,
wasted-work tracking, model-version staleness statistics, and target-crossing
evaluation for FLBench Track 3.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fl_bench.config import ExperimentConfig
from fl_bench.datasets import FederatedData
from fl_bench.methods.base import BaseMethod
from fl_bench.metrics.schema import (
    CORE_CSV_FIELDS,
    METHOD_BATCH_SIZE_MULTIPLIER,
    METHOD_PASSES_PER_STEP,
    MODEL_STATE_BYTES,
    MODEL_TRAINABLE_BYTES,
    compute_round_comm_bytes,
    describe_comm_accounting,
)
from fl_bench.simulation.events import ClientJob
from fl_bench.simulation.profiles import JobSystemSample

TRACK3_SYSTEM_CSV_FIELDS: Tuple[str, ...] = (
    "model_version",
    "simulated_time",
    "step_duration",
    "sync_wait_time",
    "cumulative_sync_wait_time",
    "server_idle_time",
    "cumulative_server_idle_time",
    "completed_client_updates",
    "accepted_updates",
    "cumulative_accepted_updates",
    "discarded_updates",
    "cumulative_discarded_updates",
    "client_dropouts",
    "cumulative_client_dropouts",
    "unavailable_selections",
    "cumulative_fit_clients",
    "round_upload_bytes",
    "round_download_bytes",
    "round_local_steps",
    "round_forward_passes",
    "round_backward_passes",
    "cumulative_upload_bytes",
    "cumulative_download_bytes",
    "cumulative_total_bytes",
    "cumulative_local_steps",
    "cumulative_forward_passes",
    "cumulative_backward_passes",
    "useful_compute_time",
    "cumulative_useful_compute_time",
    "wasted_compute_time",
    "cumulative_wasted_compute_time",
    "wasted_local_steps",
    "cumulative_wasted_local_steps",
    "wasted_work_fraction",
    "step_staleness_mean",
    "step_staleness_median",
    "step_staleness_max",
    "cumulative_staleness_mean",
    "cumulative_staleness_median",
    "cumulative_staleness_p95",
)


def build_track3_csv_fields(
    data: FederatedData,
    method: BaseMethod,
) -> List[str]:
    """Construct the ordered CSV column list for Track 3 `metrics.csv`."""
    fields = list(CORE_CSV_FIELDS)
    if data.include_perplexity:
        fields.append("test_perplexity")
    fields.append("num_fit_clients")
    fields.append("num_fit_examples")
    fields.extend(method.extra_csv_fields)
    fields.extend(TRACK3_SYSTEM_CSV_FIELDS)
    fields.append("run_completed")
    return fields


def compute_percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear quantile interpolation for `q` in `[0.0, 1.0]` on pre-sorted values."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, float(q))) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo]))


def compute_staleness_stats(staleness_values: Sequence[int]) -> Dict[str, float]:
    """
    Compute model-version staleness summary statistics (`mean`, `median`, `p95`,
    `min`, `max`). Staleness is strictly measured in integer server model versions.
    """
    if not staleness_values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    vals = sorted(float(int(v)) for v in staleness_values)
    return {
        "mean": sum(vals) / len(vals),
        "median": compute_percentile(vals, 0.50),
        "p95": compute_percentile(vals, 0.95),
        "min": vals[0],
        "max": vals[-1],
    }


class SimulatedResourceAccountant:
    """
    Analytical communication and local-computation duration model for Track 3.

    Models per-client completion time using the serialized transfer + compute equation:
        T_i = (B_{down,i} / v_{down,i}) + L_i + (W_i / c_i) + (B_{up,i} / v_{up,i}) + L_i
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        compute_model_spec: Dict[str, Any],
        model_state_bytes: Optional[int] = None,
        trainable_bytes: Optional[int] = None,
    ) -> None:
        self.config = config
        self.model_name = str(config.model or "resnet18")
        self.method_name = str(config.method or "fedavg").strip().lower()
        self.model_state_bytes = int(
            model_state_bytes
            if model_state_bytes is not None
            else MODEL_STATE_BYTES.get(self.model_name, 0)
        )
        self.trainable_bytes = int(
            trainable_bytes
            if trainable_bytes is not None
            else MODEL_TRAINABLE_BYTES.get(self.model_name, self.model_state_bytes)
        )
        if self.model_state_bytes <= 0:
            raise ValueError(
                f"Positive model_state_bytes required for model '{self.model_name}', got {self.model_state_bytes}."
            )

        required_compute_keys = (
            "reference_forward_pass_seconds",
            "reference_batch_size",
            "forward_pass_weight",
            "backward_pass_weight",
        )
        for req in required_compute_keys:
            if req not in compute_model_spec:
                raise ValueError(
                    f"Track 3 compute_model YAML block is missing required key '{req}'."
                )

        self.ref_fwd_seconds = float(compute_model_spec["reference_forward_pass_seconds"])
        self.ref_batch_size = float(compute_model_spec["reference_batch_size"])
        self.fwd_weight = float(compute_model_spec["forward_pass_weight"])
        self.bwd_weight = float(compute_model_spec["backward_pass_weight"])
        self.method_overheads: Dict[str, float] = {
            str(k).strip().lower(): float(v)
            for k, v in (compute_model_spec.get("method_step_overhead") or {}).items()
        }
        if self.ref_fwd_seconds <= 0.0 or self.ref_batch_size <= 0.0:
            raise ValueError(
                "compute_model reference_forward_pass_seconds and reference_batch_size must be > 0."
            )

    def create_client_job(
        self,
        *,
        job_id: int,
        client_id: int,
        cohort_or_step_id: int,
        start_version: int,
        job_seed: int,
        dispatch_time: float,
        num_examples: int,
        sample: JobSystemSample,
        model_param_ratio: float = 1.0,
        download_bytes_override: Optional[int] = None,
        upload_bytes_override: Optional[int] = None,
    ) -> ClientJob:
        """Instantiate a `ClientJob` with full simulated timing and resource breakdown."""
        batch_size = int(self.config.batch_size or 64)
        n_ex = max(1, int(num_examples))
        steps_per_epoch = max(1, int(math.ceil(n_ex / float(batch_size))))

        if self.config.local_steps is not None:
            local_steps = max(1, int(self.config.local_steps))
            total_examples_processed = 0
            rem = local_steps
            while rem > 0:
                take_steps = min(rem, steps_per_epoch)
                if take_steps == steps_per_epoch:
                    total_examples_processed += n_ex
                else:
                    total_examples_processed += min(n_ex, take_steps * batch_size)
                rem -= take_steps
            mean_batch = float(total_examples_processed) / float(local_steps)
        else:
            epochs = max(1, int(self.config.local_epochs or 1))
            local_steps = epochs * steps_per_epoch
            mean_batch = float(epochs * n_ex) / float(local_steps)

        fwd_mult, bwd_mult = METHOD_PASSES_PER_STEP.get(self.method_name, (1, 1))
        batch_mult = METHOD_BATCH_SIZE_MULTIPLIER.get(self.method_name, 1.0)
        eff_batch = mean_batch * float(batch_mult)

        planned_fwd = local_steps * int(fwd_mult)
        planned_bwd = local_steps * int(bwd_mult)
        opt_overhead = float(self.method_overheads.get(self.method_name, 0.0))

        pass_factor = (
            float(fwd_mult) * self.fwd_weight
            + float(bwd_mult) * self.bwd_weight
            + opt_overhead
        )
        r_model = max(1e-4, float(model_param_ratio))
        step_work = (
            self.ref_fwd_seconds
            * (eff_batch / self.ref_batch_size)
            * r_model
            * pass_factor
        )
        work_units = float(local_steps) * step_work

        comm_round_idx = max(1, int(start_version) + 1)
        default_up_b, default_down_b = compute_round_comm_bytes(
            1,
            self.model_state_bytes,
            method=self.method_name,
            trainable_bytes=self.trainable_bytes,
            round_id=comm_round_idx,
            fedlesam_variant=str(getattr(self.config, "fedlesam_variant", "base") or "base"),
            model_name=self.model_name,
        )
        up_bytes = int(upload_bytes_override if upload_bytes_override is not None else default_up_b)
        down_bytes = int(
            download_bytes_override if download_bytes_override is not None else default_down_b
        )

        # Phase durations: T_i = (B_down / v_down) + L_i + (W_i / c_i) + (B_up / v_up) + L_i
        t_down_dur = (float(down_bytes) / sample.download_bw_bytes_per_sec) + sample.latency_seconds
        t_comp_dur = work_units / sample.compute_speed
        t_up_dur = (float(up_bytes) / sample.upload_bw_bytes_per_sec) + sample.latency_seconds

        t0 = float(dispatch_time)
        t_down_end = t0 + t_down_dur
        t_comp_end = t_down_end + t_comp_dur
        t_arrival = t_comp_end + t_up_dur

        fail_time: Optional[float] = None
        if sample.will_fail and sample.failure_phase is not None:
            frac = max(0.01, min(0.99, float(sample.failure_progress_fraction)))
            if sample.failure_phase == "download":
                fail_time = t0 + frac * t_down_dur
            elif sample.failure_phase == "compute":
                fail_time = t_down_end + frac * t_comp_dur
            else:
                fail_time = t_comp_end + frac * t_up_dur

        return ClientJob(
            job_id=int(job_id),
            client_id=int(client_id),
            cohort_or_step_id=int(cohort_or_step_id),
            start_version=int(start_version),
            job_seed=int(job_seed),
            dispatch_time=t0,
            download_complete_time=t_down_end,
            compute_complete_time=t_comp_end,
            arrival_time=t_arrival,
            download_duration=t_down_dur,
            compute_duration=t_comp_dur,
            upload_duration=t_up_dur,
            latency_seconds=sample.latency_seconds,
            compute_speed=sample.compute_speed,
            download_bw_mbps=sample.download_bw_mbps,
            upload_bw_mbps=sample.upload_bw_mbps,
            download_bytes=down_bytes,
            upload_bytes=up_bytes,
            work_units=work_units,
            planned_local_steps=local_steps,
            planned_forward_passes=planned_fwd,
            planned_backward_passes=planned_bwd,
            num_examples=n_ex,
            model_param_ratio=r_model,
            effective_batch_size=eff_batch,
            will_fail=sample.will_fail,
            failure_time=fail_time,
            failure_phase=sample.failure_phase,
            failure_progress_fraction=sample.failure_progress_fraction,
        )

    @staticmethod
    def record_job_failure(job: ClientJob) -> None:
        """Populate wasted computation and transfer bytes for a mid-job dropout/failure."""
        phase = str(job.failure_phase or "compute").lower()
        frac = max(0.0, min(1.0, float(job.failure_progress_fraction)))
        job.status = f"failed_{phase}"

        if phase == "download":
            job.actual_download_bytes = int(round(frac * float(job.download_bytes)))
            job.actual_upload_bytes = 0
            job.wasted_compute_time = 0.0
            job.wasted_local_steps = 0
            job.wasted_forward_passes = 0
            job.wasted_backward_passes = 0
        elif phase == "compute":
            job.actual_download_bytes = job.download_bytes
            job.actual_upload_bytes = 0
            job.wasted_compute_time = frac * job.compute_duration
            job.wasted_local_steps = int(round(frac * float(job.planned_local_steps)))
            job.wasted_forward_passes = int(round(frac * float(job.planned_forward_passes)))
            job.wasted_backward_passes = int(round(frac * float(job.planned_backward_passes)))
        else:
            job.actual_download_bytes = job.download_bytes
            job.actual_upload_bytes = int(round(frac * float(job.upload_bytes)))
            job.wasted_compute_time = job.compute_duration
            job.wasted_local_steps = job.planned_local_steps
            job.wasted_forward_passes = job.planned_forward_passes
            job.wasted_backward_passes = job.planned_backward_passes

    @staticmethod
    def record_job_cutoff_cancellation(
        job: ClientJob,
        cutoff_time: float,
        *,
        status: str = "discarded_timeout",
    ) -> None:
        """
        Populate wasted computation and transfer bytes when an in-flight job is
        canceled at virtual `cutoff_time`.
        """
        job.status = status
        t_cut = max(job.dispatch_time, float(cutoff_time))

        if t_cut <= job.download_complete_time:
            frac = max(
                0.0,
                min(1.0, (t_cut - job.dispatch_time) / max(1e-12, job.download_duration)),
            )
            job.actual_download_bytes = int(round(frac * float(job.download_bytes)))
            job.actual_upload_bytes = 0
            job.wasted_compute_time = 0.0
            job.wasted_local_steps = 0
            job.wasted_forward_passes = 0
            job.wasted_backward_passes = 0
        elif t_cut <= job.compute_complete_time:
            job.actual_download_bytes = job.download_bytes
            job.actual_upload_bytes = 0
            frac = max(
                0.0,
                min(
                    1.0,
                    (t_cut - job.download_complete_time) / max(1e-12, job.compute_duration),
                ),
            )
            job.wasted_compute_time = frac * job.compute_duration
            job.wasted_local_steps = int(round(frac * float(job.planned_local_steps)))
            job.wasted_forward_passes = int(round(frac * float(job.planned_forward_passes)))
            job.wasted_backward_passes = int(round(frac * float(job.planned_backward_passes)))
        else:
            job.actual_download_bytes = job.download_bytes
            job.wasted_compute_time = job.compute_duration
            job.wasted_local_steps = job.planned_local_steps
            job.wasted_forward_passes = job.planned_forward_passes
            job.wasted_backward_passes = job.planned_backward_passes
            frac = max(
                0.0,
                min(
                    1.0,
                    (t_cut - job.compute_complete_time) / max(1e-12, job.upload_duration),
                ),
            )
            job.actual_upload_bytes = int(round(frac * float(job.upload_bytes)))

    @staticmethod
    def record_job_discard_on_arrival(
        job: ClientJob,
        *,
        status: str = "discarded_staleness",
    ) -> None:
        """
        Populate wasted computation and transfer bytes for a job that completed
        upload and arrived at the server, but was discarded (e.g., due to excessive
        model-version staleness or late arrival after a synchronous round closed).
        """
        job.status = status
        job.actual_download_bytes = job.download_bytes
        job.actual_upload_bytes = job.upload_bytes
        job.wasted_compute_time = job.compute_duration
        job.wasted_local_steps = job.planned_local_steps
        job.wasted_forward_passes = job.planned_forward_passes
        job.wasted_backward_passes = job.planned_backward_passes

    def describe_simulation_accounting(self) -> Dict[str, Any]:
        """Return structured metadata documenting the simulated time and compute model."""
        comm_meta = describe_comm_accounting(
            self.method_name,
            self.model_name,
            model_state_bytes=self.model_state_bytes,
            trainable_bytes=self.trainable_bytes,
        )
        return {
            "time_accounting_type": "analytical_discrete_event_simulation",
            "is_measured_real_world_wall_clock_time": False,
            "completion_time_equation": (
                "T_i = (B_{down,i} / v_{down,i}) + L_i + (W_i / c_i) + (B_{up,i} / v_{up,i}) + L_i"
            ),
            "compute_work_equation": (
                "W_i = K_i * t_ref_fwd * (B_{eff,i} / B_ref) * r_{model,i} * "
                "(m_fwd * alpha_fwd + m_bwd * alpha_bwd + alpha_{opt,method})"
            ),
            "reference_forward_pass_seconds": self.ref_fwd_seconds,
            "reference_batch_size": self.ref_batch_size,
            "forward_pass_weight": self.fwd_weight,
            "backward_pass_weight": self.bwd_weight,
            "method_step_overhead": self.method_overheads.get(self.method_name, 0.0),
            "communication_accounting": comm_meta,
            "limitations": [
                "Simulated durations are analytical discrete-event estimates parameterized by YAML "
                "system profiles and are not hardware-measured wall-clock times.",
                "Local compute work W_i models forward/backward pass counts, effective batch sizes, "
                "and subnetwork parameter ratios, while auxiliary vector operations (e.g., proximal "
                "norm penalties, control-variate additions, or prototype distance losses) are "
                "approximated via configurable method_step_overhead multipliers rather than "
                "instruction-level FLOP profiling.",
            ],
        }


def evaluate_track3_time_to_target(
    rows: Sequence[Dict[str, Any]],
    target: float,
    *,
    sustained_window: int = 3,
) -> Dict[str, Any]:
    """
    Evaluate simulated time and resource cost to reach accuracy `target`.

    Distinguishes:
      1. First target crossing (`first_crossing_time`, `first_crossing_version`).
      2. Sustained target crossing (`sustained_crossing_time`, `sustained_crossing_version`):
         the start checkpoint of the earliest window of `sustained_window` consecutive
         evaluation checkpoints all satisfying `test_accuracy >= target`.
      3. Confirmation time (`confirmation_time`, `confirmation_version`):
         the final (`sustained_window`-th) checkpoint of that window where sustained
         attainment is confirmed.

    Failed targets are never imputed with the final simulation budget.
    """
    w = max(1, int(sustained_window))
    tau = float(target)
    eval_rows: List[Dict[str, Any]] = []
    for r in rows:
        try:
            acc = float(r.get("test_accuracy", ""))
            sim_t = float(r.get("simulated_time", ""))
            if math.isfinite(acc) and math.isfinite(sim_t):
                eval_rows.append(r)
        except (ValueError, TypeError):
            continue

    nan = float("nan")
    res: Dict[str, Any] = {
        "target": tau,
        "sustained_window": w,
        "first_reached": False,
        "first_crossing_version": None,
        "first_crossing_time": nan,
        "sustained_reached": False,
        "sustained_crossing_version": None,
        "sustained_crossing_time": nan,
        "confirmation_version": None,
        "confirmation_time": nan,
        "accepted_updates_to_sustained": nan,
        "accepted_updates_to_confirmation": nan,
        "total_bytes_to_sustained": nan,
        "total_bytes_to_confirmation": nan,
        "useful_compute_time_to_sustained": nan,
        "wasted_compute_time_to_sustained": nan,
    }
    if not eval_rows:
        return res

    for r in eval_rows:
        if float(r["test_accuracy"]) >= tau:
            res["first_reached"] = True
            res["first_crossing_version"] = int(
                float(r.get("model_version") or r.get("round") or 0)
            )
            res["first_crossing_time"] = float(r["simulated_time"])
            break

    if len(eval_rows) >= w:
        for idx in range(len(eval_rows) - w + 1):
            window = eval_rows[idx : idx + w]
            if all(float(x["test_accuracy"]) >= tau for x in window):
                start_r = window[0]
                conf_r = window[-1]
                res["sustained_reached"] = True
                res["sustained_crossing_version"] = int(
                    float(start_r.get("model_version") or start_r.get("round") or 0)
                )
                res["sustained_crossing_time"] = float(start_r["simulated_time"])
                res["confirmation_version"] = int(
                    float(conf_r.get("model_version") or conf_r.get("round") or 0)
                )
                res["confirmation_time"] = float(conf_r["simulated_time"])
                res["accepted_updates_to_sustained"] = float(
                    start_r.get("cumulative_accepted_updates")
                    or start_r.get("cumulative_fit_clients")
                    or 0
                )
                res["accepted_updates_to_confirmation"] = float(
                    conf_r.get("cumulative_accepted_updates")
                    or conf_r.get("cumulative_fit_clients")
                    or 0
                )
                res["total_bytes_to_sustained"] = float(
                    start_r.get("cumulative_total_bytes")
                    or (
                        float(start_r.get("cumulative_upload_bytes", 0))
                        + float(start_r.get("cumulative_download_bytes", 0))
                    )
                )
                res["total_bytes_to_confirmation"] = float(
                    conf_r.get("cumulative_total_bytes")
                    or (
                        float(conf_r.get("cumulative_upload_bytes", 0))
                        + float(conf_r.get("cumulative_download_bytes", 0))
                    )
                )
                res["useful_compute_time_to_sustained"] = float(
                    start_r.get("cumulative_useful_compute_time", 0.0)
                )
                res["wasted_compute_time_to_sustained"] = float(
                    start_r.get("cumulative_wasted_compute_time", 0.0)
                )
                break

    return res


def evaluate_accuracy_at_time_budgets(
    rows: Sequence[Dict[str, Any]],
    time_budgets: Sequence[float],
) -> List[Dict[str, Any]]:
    """
    Evaluate test accuracy and system metrics at fixed simulated-time budgets.
    For each budget `B`, finds the latest evaluation checkpoint with `simulated_time <= B`.
    """
    eval_rows: List[Tuple[float, float, int, Dict[str, Any]]] = []
    for r in rows:
        try:
            sim_t = float(r.get("simulated_time", ""))
            acc = float(r.get("test_accuracy", ""))
            ver = int(float(r.get("model_version") or r.get("round") or 0))
            if math.isfinite(sim_t) and math.isfinite(acc):
                eval_rows.append((sim_t, acc, ver, r))
        except (ValueError, TypeError):
            continue

    eval_rows.sort(key=lambda x: (x[0], x[2]))
    nan = float("nan")
    results: List[Dict[str, Any]] = []

    for b in time_budgets:
        budget = float(b)
        eligible = [x for x in eval_rows if x[0] <= budget + 1e-9]
        if not eligible:
            results.append(
                {
                    "time_budget": budget,
                    "has_checkpoint": False,
                    "checkpoint_version": None,
                    "checkpoint_time": nan,
                    "test_accuracy_at_budget": nan,
                    "best_test_accuracy_by_budget": nan,
                    "accepted_updates_at_budget": nan,
                    "wasted_compute_time_at_budget": nan,
                    "total_bytes_at_budget": nan,
                }
            )
            continue

        last_t, last_acc, last_ver, last_r = eligible[-1]
        best_acc = max(x[1] for x in eligible)
        results.append(
            {
                "time_budget": budget,
                "has_checkpoint": True,
                "checkpoint_version": last_ver,
                "checkpoint_time": last_t,
                "test_accuracy_at_budget": last_acc,
                "best_test_accuracy_by_budget": best_acc,
                "accepted_updates_at_budget": float(
                    last_r.get("cumulative_accepted_updates")
                    or last_r.get("cumulative_fit_clients")
                    or 0
                ),
                "wasted_compute_time_at_budget": float(
                    last_r.get("cumulative_wasted_compute_time", 0.0)
                ),
                "total_bytes_at_budget": float(
                    last_r.get("cumulative_total_bytes")
                    or (
                        float(last_r.get("cumulative_upload_bytes", 0))
                        + float(last_r.get("cumulative_download_bytes", 0))
                    )
                ),
            }
        )
    return results


def trapezoidal_simulated_time_auc(
    rows: Sequence[Dict[str, Any]],
    *,
    max_time_horizon: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute the time-normalized area under the test-accuracy learning curve
    over simulated wall-clock time using composite trapezoidal integration:
        AUC_time = (1 / (t_end - t_start)) * sum_{k=1}^{M-1} 0.5 * (A(t_k) + A(t_{k+1})) * (t_{k+1} - t_k)
    """
    pts: List[Tuple[float, float]] = []
    for r in rows:
        try:
            t_val = float(r.get("simulated_time", ""))
            a_val = float(r.get("test_accuracy", ""))
            if math.isfinite(t_val) and math.isfinite(a_val):
                if max_time_horizon is not None and t_val > float(max_time_horizon) + 1e-9:
                    continue
                pts.append((t_val, a_val))
        except (ValueError, TypeError):
            continue

    pts.sort(key=lambda x: x[0])
    # Deduplicate identical timestamps by keeping the latest accuracy at that timestamp
    dedup: List[Tuple[float, float]] = []
    for t_val, a_val in pts:
        if dedup and math.isclose(dedup[-1][0], t_val, abs_tol=1e-9):
            dedup[-1] = (t_val, a_val)
        else:
            dedup.append((t_val, a_val))

    nan = float("nan")
    if len(dedup) < 2:
        return {
            "auc_fraction": nan,
            "auc_percent": nan,
            "raw_time_integral": nan,
            "eval_start_time": dedup[0][0] if dedup else nan,
            "eval_end_time": dedup[-1][0] if dedup else nan,
            "observed_span_seconds": 0.0,
            "num_eval_checkpoints": len(dedup),
        }

    t_start = dedup[0][0]
    t_end = dedup[-1][0]
    span = t_end - t_start
    if span <= 1e-12:
        return {
            "auc_fraction": dedup[-1][1],
            "auc_percent": dedup[-1][1] * 100.0,
            "raw_time_integral": 0.0,
            "eval_start_time": t_start,
            "eval_end_time": t_end,
            "observed_span_seconds": 0.0,
            "num_eval_checkpoints": len(dedup),
        }

    raw_integral = 0.0
    for (t0, a0), (t1, a1) in zip(dedup[:-1], dedup[1:]):
        dt = t1 - t0
        if dt > 0.0:
            raw_integral += 0.5 * (a0 + a1) * dt

    auc_frac = raw_integral / span
    return {
        "auc_fraction": auc_frac,
        "auc_percent": auc_frac * 100.0,
        "raw_time_integral": raw_integral,
        "eval_start_time": t_start,
        "eval_end_time": t_end,
        "observed_span_seconds": span,
        "num_eval_checkpoints": len(dedup),
    }






