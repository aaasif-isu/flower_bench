"""
Event-driven execution engine for FLBench Track 3 supporting:
  1. Synchronous FL (`synchronous`) with configurable quorum, timeout, and straggler handling.
  2. Asynchronous FL (`fedasync`, Xie et al., arXiv:1903.03934, 2019).
  3. Buffered Asynchronous FL (`fedbuff`, Nguyen et al., AISTATS 2022).

Integrates real PyTorch client training (`method.run_client`) on frozen server
model snapshots (`w^{(start_version)}`) with a deterministic virtual clock and
priority-queue discrete-event loop.
"""

import csv
import gc
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

from fl_bench.config import ExperimentConfig
from fl_bench.datasets import (
    FederatedData,
    ensure_config_assets,
    load_federated_dataset,
    load_schedule,
)
from fl_bench.methods import BaseMethod, ClientResult, StateDict, create_method
from fl_bench.models import state_dict_num_bytes, trainable_num_bytes
from fl_bench.simulation.accounting import (
    SimulatedResourceAccountant,
    build_track3_csv_fields,
    compute_staleness_stats,
    evaluate_accuracy_at_time_budgets,
    evaluate_track3_time_to_target,
    trapezoidal_simulated_time_auc,
)
from fl_bench.simulation.events import (
    EVENT_CSV_FIELDS,
    ClientJob,
    EventQueue,
    EventType,
    SimulationEvent,
    VirtualClock,
)
from fl_bench.simulation.profiles import ClientProfileManager
from fl_bench.simulation.scheduler import ClientScheduler
from fl_bench.utils.io import (
    RunLock,
    TeeLogger,
    verify_metrics_csv,
    write_json_atomic,
    write_yaml_atomic,
)
from fl_bench.utils.provenance import build_run_metadata
from fl_bench.utils.seed import client_round_seed, set_seed

SUPPORTED_EXECUTION_STRATEGIES: Tuple[str, ...] = (
    "synchronous",
    "fedasync",
    "fedbuff",
)

SUPPORTED_STALENESS_MODES: Tuple[str, ...] = (
    "polynomial",
    "hinge",
    "constant",
    "exponential",
)

SUPPORTED_STRATEGY_METHODS: Dict[str, Set[str]] = {
    "synchronous": {
        "fedavg",
        "fedprox",
        "fedavgm",
        "fedadam",
        "fednova",
        "scaffold",
        "fedspeed",
        "fedlesam",
        "fedgps",
        "fedwmsam",
        "heterofl",
    },
    "fedasync": {
        "fedavg",
        "fedprox",
    },
    "fedbuff": {
        "fedavg",
        "fedprox",
    },
}

ASYNC_INCOMPATIBLE_REASONS: Dict[str, str] = {
    "scaffold": (
        "SCAFFOLD maintains coupled global (c) and client (c_i) control variates under a "
        "synchronous zero-sum invariant (c <- c + (1/N) sum Delta c_i). Applying stale "
        "control-variate deltas without an explicit asynchronous control-variate correction "
        "violates variance-reduction semantics."
    ),
    "fedspeed": (
        "FedSpeed maintains coupled per-client dual variables (h_i) and a global dual sum "
        "((1/N) sum_j h_j) that assume round-synchronous primal-dual updates."
    ),
    "fedlesam": (
        "FedLESAM perturbs client weights along the previous synchronous round's global "
        "model difference (w^t - w^{t-1}) and uses synchronous control/dual states in -S/-D."
    ),
    "fedwmsam": (
        "FedWMSAM couples global control variates, server momentum perturbations, and a "
        "round-synchronous adaptive schedule (alpha_t)."
    ),
    "fedgps": (
        "FedGPS computes non-self path rectification from the previous synchronous cohort "
        "((M * Delta w - Delta w_i^{t-1}) / (M - 1)) and synchronous global noise prototypes."
    ),
    "heterofl": (
        "HeteroFL requires cohort-wide parameter-overlap normalization and post-round "
        "Static BatchNorm (sBN) calibration across synchronous subnetwork cohorts."
    ),
    "fednova": (
        "FedNova normalizes cohort updates by a synchronous round effective step factor "
        "(tau_eff = sum_i p_i a_i), which is undefined for single/buffered asynchronous arrivals "
        "without an explicit async FedNova adaptation."
    ),
    "fedavgm": (
        "FedAvgM defines server-side cohort momentum that overlaps with FedAsync/FedBuff "
        "server aggregation rules; use 'fedbuff' with 'fedbuff_server_momentum' + 'fedavg' instead."
    ),
    "fedadam": (
        "FedAdam defines synchronous round-indexed bias-corrected server Adam states (m_t, v_t) "
        "that require an explicit asynchronous adaptive server optimizer formulation."
    ),
}


def validate_execution_compatibility(config: ExperimentConfig) -> None:
    """
    Validate that `config.execution_strategy`, `config.method`, and Track 3 simulation
    hyperparameters form a mathematically valid, supported combination.
    """
    strategy = str(getattr(config, "execution_strategy", "synchronous") or "synchronous").strip().lower()
    if strategy not in SUPPORTED_EXECUTION_STRATEGIES:
        raise ValueError(
            f"Unsupported execution_strategy '{strategy}'. "
            f"Supported: {list(SUPPORTED_EXECUTION_STRATEGIES)}"
        )

    method = str(config.method or "").strip().lower()
    allowed_methods = SUPPORTED_STRATEGY_METHODS.get(strategy, set())
    if method not in allowed_methods:
        reason = ASYNC_INCOMPATIBLE_REASONS.get(
            method,
            f"Method '{method}' is not supported with execution_strategy='{strategy}'.",
        )
        raise ValueError(
            f"Unsupported Track 3 combination: execution_strategy='{strategy}' with method='{method}'. "
            f"Supported methods for '{strategy}': {sorted(allowed_methods)}. "
            f"Reason: {reason}"
        )

    max_conc = getattr(config, "max_concurrency", None)
    if max_conc is not None:
        if int(max_conc) <= 0 or (
            config.num_clients is not None and int(max_conc) > int(config.num_clients)
        ):
            raise ValueError(
                f"max_concurrency ({max_conc}) must be in 1..{config.num_clients}."
            )

    if strategy == "synchronous":
        qf = getattr(config, "quorum_fraction", None)
        if qf is not None and not (0.0 < float(qf) <= 1.0):
            raise ValueError(f"quorum_fraction must be in (0.0, 1.0], got {qf}.")
        rt = getattr(config, "round_timeout", None)
        if rt is not None and float(rt) <= 0.0:
            raise ValueError(f"round_timeout must be > 0 when specified, got {rt}.")
        sp = str(getattr(config, "straggler_policy", "cancel") or "cancel").strip().lower()
        if sp not in ("cancel", "discard_on_arrival"):
            raise ValueError(
                f"straggler_policy must be 'cancel' or 'discard_on_arrival', got '{sp}'."
            )
        ot = str(getattr(config, "on_timeout", "aggregate_partial") or "aggregate_partial").strip().lower()
        if ot not in ("aggregate_partial", "drop_round"):
            raise ValueError(
                f"on_timeout must be 'aggregate_partial' or 'drop_round', got '{ot}'."
            )

    if strategy in ("fedasync", "fedbuff"):
        s_mode = str(getattr(config, "staleness_mode", "polynomial") or "polynomial").strip().lower()
        if s_mode not in SUPPORTED_STALENESS_MODES:
            raise ValueError(
                f"Unsupported staleness_mode '{s_mode}'. Supported: {list(SUPPORTED_STALENESS_MODES)}"
            )
        s_a = getattr(config, "staleness_a", None)
        if s_a is not None and float(s_a) < 0.0:
            raise ValueError(f"staleness_a must be >= 0, got {s_a}.")
        max_s = getattr(config, "max_staleness", None)
        if max_s is not None and int(max_s) < 0:
            raise ValueError(f"max_staleness must be >= 0 when specified, got {max_s}.")

    if strategy == "fedasync":
        alpha = getattr(config, "fedasync_alpha", None)
        if alpha is None or not (0.0 < float(alpha) <= 1.0):
            raise ValueError(
                f"fedasync requires 'fedasync_alpha' in (0.0, 1.0], got {alpha}."
            )

    if strategy == "fedbuff":
        buf = getattr(config, "buffer_size", None)
        if buf is None or int(buf) < 1:
            raise ValueError(
                f"fedbuff requires integer 'buffer_size' >= 1, got {buf}."
            )
        norm_mode = str(
            getattr(config, "fedbuff_normalization", "buffer_size") or "buffer_size"
        ).strip().lower()
        if norm_mode not in ("buffer_size", "normalized_staleness", "sample_weighted"):
            raise ValueError(
                f"Unsupported fedbuff_normalization '{norm_mode}'. "
                "Supported: 'buffer_size', 'normalized_staleness', 'sample_weighted'."
            )


def compute_staleness_weight(
    staleness: int,
    *,
    mode: str = "polynomial",
    a: float = 0.5,
    b: float = 4.0,
) -> float:
    """
    Compute the staleness decay factor g(s) for model-version staleness `s = t - tau >= 0`.

    Follows Xie et al. (arXiv:1903.03934, Section 4.1) and Nguyen et al. (AISTATS 2022):
      - `polynomial`: g(s) = (s + 1)^{-a}
      - `hinge`:      g(s) = 1.0 if s <= b else 1.0 / (a * (s - b) + 1.0)
      - `constant`:   g(s) = 1.0
      - `exponential`: g(s) = exp(-a * s)
    """
    s = max(0, int(staleness))
    m = str(mode or "polynomial").strip().lower()
    if m == "constant" or s == 0:
        return 1.0
    if m == "polynomial":
        return float((float(s) + 1.0) ** (-float(a)))
    if m == "hinge":
        if float(s) <= float(b):
            return 1.0
        return float(1.0 / (float(a) * (float(s) - float(b)) + 1.0))
    if m == "exponential":
        return float(math.exp(-float(a) * float(s)))
    raise ValueError(f"Unknown staleness_mode '{mode}'.")


def fedasync_aggregate_step(
    current_state: StateDict,
    client_result: ClientResult,
    *,
    alpha_eff: float,
) -> StateDict:
    """
    Execute the FedAsync server update (Xie et al., arXiv:1903.03934, Algorithm 1):
        w^{(t+1)} = (1 - alpha_t) * w^{(t)} + alpha_t * w_i^{local}
    where `alpha_t = alpha * g(t - tau_i)`.
    """
    a_t = max(0.0, min(1.0, float(alpha_eff)))
    one_minus_a = 1.0 - a_t
    new_state: StateDict = {}
    local_state = client_result.state

    for key, cur_val in current_state.items():
        loc_val = local_state[key]
        if not torch.is_floating_point(cur_val):
            new_state[key] = torch.maximum(cur_val, loc_val).clone()
            continue

        updated64 = (
            one_minus_a * cur_val.to(torch.float64)
            + a_t * loc_val.to(torch.float64)
        )
        updated = updated64.to(cur_val.dtype)
        if not torch.isfinite(updated).all():
            raise FloatingPointError(
                f"Nonfinite parameter tensor '{key}' in FedAsync server update."
            )
        new_state[key] = updated

    return new_state


def fedbuff_aggregate_step(
    current_state: StateDict,
    buffered_jobs: Sequence[ClientJob],
    snapshots: Dict[int, StateDict],
    trainable_keys: Set[str],
    *,
    server_velocity: Optional[StateDict],
    server_lr: float = 1.0,
    server_momentum: float = 0.0,
    normalization: str = "buffer_size",
) -> Tuple[StateDict, Optional[StateDict]]:
    """
    Execute the FedBuff buffered asynchronous server update
    (Nguyen et al., AISTATS 2022, Algorithm 1):
        Delta_k = w_k^{local} - w^{(tau_k)}
        bar_Delta^{(t)} = (1 / K) * sum_{k=1}^K g(s_k) * Delta_k
        v^{(t+1)} = beta_g * v^{(t)} + bar_Delta^{(t)}
        w^{(t+1)} = w^{(t)} + eta_g * v^{(t+1)}
    """
    k_buf = len(buffered_jobs)
    if k_buf == 0:
        raise RuntimeError("fedbuff_aggregate_step called with an empty buffer.")

    norm_mode = str(normalization or "buffer_size").strip().lower()
    raw_weights: List[float] = []
    for job in buffered_jobs:
        g_s = float(job.staleness_weight if job.staleness_weight is not None else 1.0)
        if norm_mode == "sample_weighted":
            raw_weights.append(g_s * float(max(1, job.num_examples)))
        else:
            raw_weights.append(g_s)

    if norm_mode == "buffer_size":
        # Exact Nguyen et al. (2022) Algorithm 1 normalization: divide by buffer size K
        denom = float(k_buf)
    else:
        denom = max(1e-12, sum(raw_weights))
    eff_weights = [w / denom for w in raw_weights]

    # For BatchNorm running stats, normalize weights to sum to 1.0
    bn_denom = max(1e-12, sum(raw_weights))
    bn_weights = [w / bn_denom for w in raw_weights]

    eta_g = float(server_lr)
    beta_g = float(server_momentum)
    if beta_g > 0.0 and server_velocity is None:
        server_velocity = {
            k: torch.zeros_like(current_state[k], dtype=torch.float64)
            for k in trainable_keys
        }

    new_state: StateDict = {}
    for key, cur_val in current_state.items():
        if not torch.is_floating_point(cur_val):
            max_buf = cur_val.clone()
            for job in buffered_jobs:
                assert job.client_result is not None
                max_buf = torch.maximum(max_buf, job.client_result.state[key])
            new_state[key] = max_buf
            continue

        if key in trainable_keys:
            delta_acc = torch.zeros_like(cur_val, dtype=torch.float64)
            for job, w_k in zip(buffered_jobs, eff_weights):
                assert job.client_result is not None
                base_snap = snapshots[job.start_version]
                delta_k = (
                    job.client_result.state[key].to(torch.float64)
                    - base_snap[key].to(torch.float64)
                )
                delta_acc += w_k * delta_k

            if beta_g > 0.0 and server_velocity is not None:
                server_velocity[key] = beta_g * server_velocity[key] + delta_acc
                step_vec = server_velocity[key]
            else:
                step_vec = delta_acc

            updated64 = cur_val.to(torch.float64) + eta_g * step_vec
            updated = updated64.to(cur_val.dtype)
        else:
            # Non-trainable floating-point BatchNorm buffers (running_mean, running_var)
            bn_acc = torch.zeros_like(cur_val, dtype=torch.float64)
            for job, w_bn in zip(buffered_jobs, bn_weights):
                assert job.client_result is not None
                bn_acc += w_bn * job.client_result.state[key].to(torch.float64)
            updated = bn_acc.to(cur_val.dtype)

        if not torch.isfinite(updated).all():
            raise FloatingPointError(
                f"Nonfinite parameter tensor '{key}' in FedBuff server update."
            )
        new_state[key] = updated

    return new_state, server_velocity


class DiscreteEventFLSimulator:
    """
    Event-driven simulator orchestrating virtual clock progression, client job
    dispatch, download/compute/upload phases, failures, timeouts, and server
    aggregation across `synchronous`, `fedasync`, and `fedbuff`.
    """

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        data: FederatedData,
        method: BaseMethod,
        global_model: nn.Module,
        device: torch.device,
        base_schedule: Dict[str, List[int]],
        logger: TeeLogger,
    ) -> None:
        validate_execution_compatibility(config)
        self.config = config
        self.data = data
        self.method = method
        self.global_model = global_model
        self.device = device
        self.logger = logger

        self.strategy = str(config.execution_strategy or "synchronous").strip().lower()
        self.profile_name = str(config.system_profile or "homogeneous").strip()
        self.max_versions = int(config.rounds or 1)
        self.max_concurrency = int(
            config.max_concurrency
            if config.max_concurrency is not None
            else (config.clients_per_round or 10)
        )
        self.max_simulated_time: Optional[float] = (
            float(config.max_simulated_time)
            if getattr(config, "max_simulated_time", None) is not None
            else None
        )
        self.eval_interval = max(1, int(config.eval_interval or 1))
        self.eval_time_interval: Optional[float] = (
            float(config.eval_time_interval)
            if getattr(config, "eval_time_interval", None) is not None
            else None
        )

        # Synchronous parameters
        self.quorum_fraction = float(
            config.quorum_fraction if config.quorum_fraction is not None else 1.0
        )
        self.round_timeout: Optional[float] = (
            float(config.round_timeout)
            if getattr(config, "round_timeout", None) is not None
            else None
        )
        self.straggler_policy = str(
            getattr(config, "straggler_policy", "cancel") or "cancel"
        ).strip().lower()
        self.on_timeout = str(
            getattr(config, "on_timeout", "aggregate_partial") or "aggregate_partial"
        ).strip().lower()
        self.min_accepted_updates = max(
            1, int(getattr(config, "min_accepted_updates", 1) or 1)
        )

        # Async / Buffered parameters
        self.fedasync_alpha = float(
            config.fedasync_alpha if config.fedasync_alpha is not None else 0.6
        )
        self.staleness_mode = str(
            getattr(config, "staleness_mode", "polynomial") or "polynomial"
        ).strip().lower()
        self.staleness_a = float(
            config.staleness_a if config.staleness_a is not None else 0.5
        )
        self.staleness_b = float(
            config.staleness_b if config.staleness_b is not None else 4.0
        )
        self.max_staleness: Optional[int] = (
            int(config.max_staleness)
            if getattr(config, "max_staleness", None) is not None
            else None
        )
        self.buffer_size = max(1, int(getattr(config, "buffer_size", 5) or 5))
        self.fedbuff_server_lr = float(
            config.fedbuff_server_lr
            if getattr(config, "fedbuff_server_lr", None) is not None
            else (config.server_lr if config.server_lr is not None else 1.0)
        )
        self.fedbuff_server_momentum = float(
            config.fedbuff_server_momentum
            if getattr(config, "fedbuff_server_momentum", None) is not None
            else 0.0
        )
        self.fedbuff_normalization = str(
            getattr(config, "fedbuff_normalization", "buffer_size") or "buffer_size"
        ).strip().lower()

        self.base_lr = float(config.lr or 0.01)
        self.lr_decay = float(getattr(config, "lr_decay", 1.0) or 1.0)

        self.trainable_keys: Set[str] = {
            name for name, param in global_model.named_parameters() if param.requires_grad
        }
        self.model_state_bytes = state_dict_num_bytes(global_model)
        self.trainable_bytes = trainable_num_bytes(global_model)

        profile_seed = int(
            config.profile_seed
            if getattr(config, "profile_seed", None) is not None
            else (config.partition_seed if config.partition_seed is not None else config.training_seed)
        )
        sim_seed = int(
            config.simulation_seed
            if getattr(config, "simulation_seed", None) is not None
            else config.training_seed
        )
        sched_seed = int(
            config.schedule_seed if config.schedule_seed is not None else config.training_seed
        )
        isolate_factor = str(getattr(config, "isolate_factor", "all") or "all")

        all_profiles = getattr(config, "system_profiles", None) or {}
        prof_spec = getattr(config, "system_profile_spec", None) or all_profiles.get(
            self.profile_name
        )
        if not isinstance(prof_spec, dict):
            raise ValueError(
                f"Missing YAML specification for system_profile='{self.profile_name}'."
            )
        homog_spec = all_profiles.get("homogeneous")

        self.profile_manager = ClientProfileManager(
            num_clients=int(config.num_clients or data.num_clients),
            profile_name=self.profile_name,
            profile_spec=prof_spec,
            homogeneous_spec=homog_spec if isinstance(homog_spec, dict) else None,
            profile_seed=profile_seed,
            simulation_seed=sim_seed,
            isolate_factor=isolate_factor,
        )

        compute_spec = getattr(config, "compute_model_spec", None)
        if not isinstance(compute_spec, dict):
            raise ValueError("Missing YAML 'compute_model' specification for Track 3.")

        self.accountant = SimulatedResourceAccountant(
            config=config,
            compute_model_spec=compute_spec,
            model_state_bytes=self.model_state_bytes,
            trainable_bytes=self.trainable_bytes,
        )
        self.scheduler = ClientScheduler(
            num_clients=int(config.num_clients or data.num_clients),
            max_concurrency=self.max_concurrency,
            profile_manager=self.profile_manager,
            schedule_seed=sched_seed,
            base_schedule=base_schedule,
        )

        self.clock = VirtualClock(0.0)
        self.queue = EventQueue()

        self.current_version: int = 0
        self.snapshots: Dict[int, StateDict] = {
            0: {
                k: v.detach().cpu().clone()
                for k, v in self.global_model.state_dict().items()
            }
        }
        self.fedbuff_velocity: Optional[StateDict] = None

        self.next_job_id: int = 0
        self.active_jobs: Dict[int, ClientJob] = {}
        self.all_jobs: List[ClientJob] = []

        # Synchronous round state
        self.current_sync_round: int = 0
        self.sync_round_dispatched_jobs: List[ClientJob] = []
        self.sync_round_arrivals: List[ClientJob] = []
        self.sync_round_quorum: int = 1
        self.sync_round_timed_out: bool = False

        # FedAsync / FedBuff state
        self.pending_async_job: Optional[ClientJob] = None
        self.fedbuff_buffer: List[ClientJob] = []

        # Cumulative & per-step accounting counters
        self.last_step_time: float = 0.0
        self.first_arrival_time_in_step: Optional[float] = None
        self.last_eval_sim_time: float = 0.0
        self.last_eval_loss: Optional[float] = None
        self.last_eval_acc: Optional[float] = None
        self.last_eval_perp: Optional[float] = None

        self.cum_sync_wait_time: float = 0.0
        self.cum_server_idle_time: float = 0.0
        self.cum_completed_updates: int = 0
        self.cum_accepted_updates: int = 0
        self.cum_discarded_updates: int = 0
        self.cum_client_dropouts: int = 0
        self.cum_upload_bytes: int = 0
        self.cum_download_bytes: int = 0
        self.cum_local_steps: int = 0
        self.cum_forward_passes: int = 0
        self.cum_backward_passes: int = 0
        self.cum_useful_compute_time: float = 0.0
        self.cum_wasted_compute_time: float = 0.0
        self.cum_wasted_local_steps: int = 0
        self.all_accepted_staleness: List[int] = []

        # Per-step delta accumulators (reset after each model version step)
        self._step_discarded_updates: int = 0
        self._step_dropouts: int = 0
        self._step_wasted_compute_time: float = 0.0
        self._step_wasted_local_steps: int = 0
        self._step_extra_up_bytes: int = 0
        self._step_extra_down_bytes: int = 0

        self.stopped_by_time_budget: bool = False

    def _prune_snapshots(self) -> None:
        """
        Garbage-collect historical server model snapshots that are no longer
        referenced by any active in-flight job or buffered update.
        """
        needed_versions: Set[int] = {self.current_version}
        for job in self.active_jobs.values():
            needed_versions.add(job.start_version)
        for job in self.fedbuff_buffer:
            needed_versions.add(job.start_version)
        if self.pending_async_job is not None:
            needed_versions.add(self.pending_async_job.start_version)

        stale_keys = [v for v in self.snapshots if v not in needed_versions]
        for v in stale_keys:
            del self.snapshots[v]

    def _execute_real_client_training(self, job: ClientJob) -> ClientResult:
        """
        Execute actual FLBench local client training (`method.run_client`)
        starting from the frozen server model snapshot `w^{(job.start_version)}`
        that was active when `job` was dispatched.
        """
        if job.client_result is not None:
            return job.client_result

        if job.start_version not in self.snapshots:
            raise RuntimeError(
                f"Missing server model snapshot for start_version={job.start_version} "
                f"(job_id={job.job_id}, client_id={job.client_id})."
            )
        start_state = self.snapshots[job.start_version]

        effective_step = max(1, int(job.start_version) + 1)
        self.method.current_round_id = effective_step
        if not math.isclose(self.lr_decay, 1.0):
            self.config.lr = self.base_lr * (self.lr_decay ** max(0, effective_step - 1))
        else:
            self.config.lr = self.base_lr

        res = self.method.run_client(
            cid=job.client_id,
            global_state=start_state,
            client_dataset=self.data.train_clients[job.client_id],
            device=self.device,
            seed=job.job_seed,
        )
        job.client_result = res
        job.train_loss = float(res.train_loss)
        job.train_accuracy = float(res.train_accuracy)
        if res.upload_bytes_override is not None:
            job.upload_bytes = int(res.upload_bytes_override)
        if res.download_bytes_override is not None:
            job.download_bytes = int(res.download_bytes_override)
        return res

    def _estimate_heterofl_subnet_scaling(
        self,
        cid: int,
        round_id: int,
    ) -> Tuple[float, Optional[int], Optional[int]]:
        """
        If `method == 'heterofl'`, return `(param_ratio, down_bytes, up_bytes)`
        for client `cid`'s assigned subnetwork width rate.
        """
        if self.method.name != "heterofl":
            return 1.0, None, None
        client_rates = getattr(self.method, "client_model_rates", None)
        if isinstance(client_rates, list) and 0 <= cid < len(client_rates):
            rate_m = float(client_rates[cid])
        else:
            rate_m = 1.0
        param_ratio = max(1e-4, min(1.0, rate_m * rate_m))
        est_bytes = max(1024, int(round(param_ratio * float(self.model_state_bytes))))
        return param_ratio, est_bytes, est_bytes

    def _dispatch_client(self, cid: int, cohort_or_step_id: int) -> ClientJob:
        """Create and enqueue discrete events for a single client job at `clock.current_time`."""
        self.next_job_id += 1
        job_id = self.next_job_id
        client_job_idx = self.scheduler.mark_dispatched(cid)

        seed_round_idx = (
            int(cohort_or_step_id)
            if self.strategy == "synchronous"
            else int(client_job_idx)
        )
        job_seed = client_round_seed(self.config.training_seed, seed_round_idx, cid)

        sample = self.profile_manager.sample_job_parameters(cid, client_job_idx)
        n_examples = len(self.data.train_clients[cid])
        param_ratio, down_ovr, up_ovr = self._estimate_heterofl_subnet_scaling(
            cid, cohort_or_step_id
        )

        job = self.accountant.create_client_job(
            job_id=job_id,
            client_id=cid,
            cohort_or_step_id=cohort_or_step_id,
            start_version=self.current_version,
            job_seed=job_seed,
            dispatch_time=self.clock.current_time,
            num_examples=n_examples,
            sample=sample,
            model_param_ratio=param_ratio,
            download_bytes_override=down_ovr,
            upload_bytes_override=up_ovr,
        )
        self.active_jobs[job_id] = job
        self.all_jobs.append(job)

        if job.will_fail and job.failure_time is not None:
            t_fail = job.failure_time
            if job.failure_phase in ("compute", "upload"):
                self.queue.push(
                    job.download_complete_time,
                    EventType.DOWNLOAD_COMPLETE,
                    job_id=job_id,
                    client_id=cid,
                    round_id=cohort_or_step_id,
                )
            if job.failure_phase == "upload":
                self.queue.push(
                    job.compute_complete_time,
                    EventType.COMPUTE_COMPLETE,
                    job_id=job_id,
                    client_id=cid,
                    round_id=cohort_or_step_id,
                )
            self.queue.push(
                t_fail,
                EventType.CLIENT_FAILURE,
                job_id=job_id,
                client_id=cid,
                round_id=cohort_or_step_id,
            )
        else:
            self.queue.push(
                job.download_complete_time,
                EventType.DOWNLOAD_COMPLETE,
                job_id=job_id,
                client_id=cid,
                round_id=cohort_or_step_id,
            )
            self.queue.push(
                job.compute_complete_time,
                EventType.COMPUTE_COMPLETE,
                job_id=job_id,
                client_id=cid,
                round_id=cohort_or_step_id,
            )
            self.queue.push(
                job.arrival_time,
                EventType.UPDATE_ARRIVAL,
                job_id=job_id,
                client_id=cid,
                round_id=cohort_or_step_id,
            )
        return job

    def _start_next_synchronous_round(self) -> None:
        """Select a client cohort and dispatch jobs for the next synchronous round."""
        self.current_sync_round += 1
        rnd = self.current_sync_round
        self.sync_round_dispatched_jobs = []
        self.sync_round_arrivals = []
        self.sync_round_timed_out = False
        self.first_arrival_time_in_step = None

        selected_cids, _ = self.scheduler.select_synchronous_cohort(
            rnd, cohort_size=self.max_concurrency
        )
        if not selected_cids:
            # No clients available in the population right now; advance clock by a small retry window
            retry_dt = float(self.round_timeout if self.round_timeout is not None else 10.0)
            self.queue.push(
                self.clock.current_time + retry_dt,
                EventType.AGGREGATION,
                round_id=rnd,
                model_version=self.current_version + 1,
            )
            return

        self.sync_round_quorum = max(
            1, int(math.ceil(self.quorum_fraction * float(len(selected_cids))))
        )
        for cid in selected_cids:
            job = self._dispatch_client(cid, cohort_or_step_id=rnd)
            self.sync_round_dispatched_jobs.append(job)

        if self.round_timeout is not None:
            self.queue.push(
                self.clock.current_time + self.round_timeout,
                EventType.ROUND_TIMEOUT,
                round_id=rnd,
                model_version=self.current_version + 1,
            )

    def _fill_async_concurrency(self, initial: bool = False) -> None:
        """Dispatch available clients to fill all open concurrency slots in `fedasync` / `fedbuff`."""
        selected_cids, _ = self.scheduler.select_async_replacements(
            use_round1_schedule_if_initial=initial
        )
        for cid in selected_cids:
            self._dispatch_client(cid, cohort_or_step_id=self.next_job_id + 1)

    def _accumulate_wasted_job(
        self,
        job: ClientJob,
        *,
        is_dropout: bool = False,
        is_discarded_update: bool = False,
    ) -> None:
        """Record wasted computation, steps, and network bytes from a failed or discarded job."""
        if is_dropout:
            self._step_dropouts += 1
            self.cum_client_dropouts += 1
        if is_discarded_update:
            self._step_discarded_updates += 1
            self.cum_discarded_updates += 1

        self._step_wasted_compute_time += float(job.wasted_compute_time)
        self.cum_wasted_compute_time += float(job.wasted_compute_time)
        self._step_wasted_local_steps += int(job.wasted_local_steps)
        self.cum_wasted_local_steps += int(job.wasted_local_steps)
        self._step_extra_up_bytes += int(job.actual_upload_bytes)
        self._step_extra_down_bytes += int(job.actual_download_bytes)
        self.cum_upload_bytes += int(job.actual_upload_bytes)
        self.cum_download_bytes += int(job.actual_download_bytes)

    def _check_sync_round_ready(self) -> None:
        """
        Check if the active synchronous round has reached quorum or has no remaining
        in-flight jobs, and if so trigger immediate server aggregation.
        """
        rnd = self.current_sync_round
        in_flight_in_round = [
            j
            for j in self.sync_round_dispatched_jobs
            if j.job_id in self.active_jobs
        ]
        quorum_met = len(self.sync_round_arrivals) >= self.sync_round_quorum

        if quorum_met or not in_flight_in_round:
            self.queue.cancel_round_events(
                rnd, event_types=(EventType.ROUND_TIMEOUT, EventType.AGGREGATION)
            )
            self.queue.push(
                self.clock.current_time,
                EventType.AGGREGATION,
                round_id=rnd,
                model_version=self.current_version + 1,
            )

    def _handle_client_failure(self, ev: SimulationEvent) -> None:
        """Process a mid-job client dropout/failure event."""
        assert ev.job_id is not None
        job = self.active_jobs.pop(ev.job_id, None)
        if job is None:
            return
        self.queue.cancel_job_events(job.job_id)
        self.scheduler.mark_completed(job.client_id)
        self.accountant.record_job_failure(job)
        self._accumulate_wasted_job(job, is_dropout=True, is_discarded_update=False)

        if self.strategy == "synchronous":
            if job.cohort_or_step_id == self.current_sync_round:
                self._check_sync_round_ready()
        else:
            if not self._should_stop():
                self._fill_async_concurrency(initial=False)

    def _handle_update_arrival(self, ev: SimulationEvent) -> None:
        """Process a completed client update arriving at the server."""
        assert ev.job_id is not None
        job = self.active_jobs.pop(ev.job_id, None)
        if job is None:
            return
        self.queue.cancel_job_events(job.job_id)
        self.scheduler.mark_completed(job.client_id)

        job.status = "arrived"
        job.actual_download_bytes = job.download_bytes
        job.actual_upload_bytes = job.upload_bytes
        self.cum_completed_updates += 1

        if self.strategy == "synchronous":
            if job.cohort_or_step_id != self.current_sync_round:
                self.accountant.record_job_discard_on_arrival(
                    job, status="discarded_timeout"
                )
                self._accumulate_wasted_job(
                    job, is_dropout=False, is_discarded_update=True
                )
                return

            if self.first_arrival_time_in_step is None:
                self.first_arrival_time_in_step = self.clock.current_time
            self.sync_round_arrivals.append(job)
            self._check_sync_round_ready()
            return

        # Asynchronous (`fedasync`) or Buffered Asynchronous (`fedbuff`)
        staleness = max(0, int(self.current_version - job.start_version))
        job.staleness = staleness
        if self.max_staleness is not None and staleness > self.max_staleness:
            self.accountant.record_job_discard_on_arrival(
                job, status="discarded_staleness"
            )
            self._accumulate_wasted_job(
                job, is_dropout=False, is_discarded_update=True
            )
            if not self._should_stop():
                self._fill_async_concurrency(initial=False)
            return

        if self.strategy == "fedasync":
            if self.first_arrival_time_in_step is None:
                self.first_arrival_time_in_step = self.clock.current_time
            self.pending_async_job = job
            self.queue.push(
                self.clock.current_time,
                EventType.AGGREGATION,
                job_id=job.job_id,
                model_version=self.current_version + 1,
            )
            return

        # `fedbuff` strategy
        if not self.fedbuff_buffer and self.first_arrival_time_in_step is None:
            self.first_arrival_time_in_step = self.clock.current_time

        self._execute_real_client_training(job)
        job.status = "buffered"
        self.fedbuff_buffer.append(job)

        if len(self.fedbuff_buffer) >= self.buffer_size:
            self.queue.push(
                self.clock.current_time,
                EventType.AGGREGATION,
                model_version=self.current_version + 1,
            )
        else:
            if not self._should_stop():
                self._fill_async_concurrency(initial=False)

    def _cancel_remaining_sync_stragglers(self, cutoff_time: float) -> None:
        """
        Handle in-flight stragglers in the current synchronous round when quorum
        or round timeout triggers aggregation at `cutoff_time`.
        """
        in_flight = [
            j
            for j in self.sync_round_dispatched_jobs
            if j.job_id in self.active_jobs
        ]
        if not in_flight:
            return

        if self.straggler_policy == "cancel":
            for job in in_flight:
                self.active_jobs.pop(job.job_id, None)
                self.queue.cancel_job_events(job.job_id)
                self.scheduler.mark_completed(job.client_id)
                self.accountant.record_job_cutoff_cancellation(
                    job, cutoff_time, status="discarded_timeout"
                )
                self._accumulate_wasted_job(
                    job, is_dropout=False, is_discarded_update=True
                )
        else:
            # `discard_on_arrival`: release the client slot for future rounds or
            # let the arrival handler discard the late update when it arrives.
            pass

    def _handle_round_timeout(self, ev: SimulationEvent) -> None:
        """Process synchronous round timeout expiration."""
        if ev.round_id != self.current_sync_round:
            return
        self.sync_round_timed_out = True
        self.queue.cancel_round_events(
            self.current_sync_round, event_types=(EventType.AGGREGATION,)
        )
        self.queue.push(
            self.clock.current_time,
            EventType.AGGREGATION,
            round_id=self.current_sync_round,
            model_version=self.current_version + 1,
        )

    def _should_stop(self) -> bool:
        """Check whether the simulation has reached its target version count or time horizon."""
        if self.current_version >= self.max_versions:
            return True
        if (
            self.max_simulated_time is not None
            and self.clock.current_time >= self.max_simulated_time - 1e-12
        ):
            self.stopped_by_time_budget = True
            return True
        return False

    def _handle_aggregation(
        self,
        ev: SimulationEvent,
        csv_writer: csv.DictWriter,
        csv_file,
    ) -> None:
        """Execute server aggregation, advance model version, and log step metrics."""
        t_now = self.clock.current_time
        step_duration = max(0.0, t_now - self.last_step_time)
        if self.first_arrival_time_in_step is not None:
            step_idle = max(0.0, self.first_arrival_time_in_step - self.last_step_time)
            step_sync_wait = max(0.0, t_now - self.first_arrival_time_in_step)
        else:
            step_idle = step_duration
            step_sync_wait = 0.0

        self.last_step_time = t_now
        self.first_arrival_time_in_step = None
        self.cum_server_idle_time += step_idle
        self.cum_sync_wait_time += step_sync_wait

        cur_state = self.snapshots[self.current_version]
        next_ver = self.current_version + 1
        accepted_jobs: List[ClientJob] = []
        extra_metrics: Dict[str, float] = {}

        if self.strategy == "synchronous":
            self._cancel_remaining_sync_stragglers(t_now)
            arrived = list(self.sync_round_arrivals)
            self.sync_round_arrivals = []

            drop_due_to_timeout = (
                self.sync_round_timed_out
                and self.on_timeout == "drop_round"
                and len(arrived) < self.sync_round_quorum
            )
            if drop_due_to_timeout or len(arrived) < self.min_accepted_updates:
                for job in arrived:
                    self.accountant.record_job_discard_on_arrival(
                        job, status="discarded_round_failed"
                    )
                    self._accumulate_wasted_job(
                        job, is_dropout=False, is_discarded_update=True
                    )
                new_state = {k: v.clone() for k, v in cur_state.items()}
            else:
                accepted_jobs = arrived
                client_results: List[ClientResult] = []
                for job in accepted_jobs:
                    res = self._execute_real_client_training(job)
                    client_results.append(res)
                    job.staleness = max(0, int(self.current_version - job.start_version))
                    job.staleness_weight = 1.0
                self.method.current_round_id = next_ver
                new_state, extra_metrics = self.method.aggregate(
                    round_id=next_ver,
                    global_state=cur_state,
                    client_results=client_results,
                    trainable_keys=self.trainable_keys,
                )

        elif self.strategy == "fedasync":
            assert self.pending_async_job is not None
            job = self.pending_async_job
            self.pending_async_job = None
            res = self._execute_real_client_training(job)
            staleness = max(0, int(self.current_version - job.start_version))
            g_s = compute_staleness_weight(
                staleness,
                mode=self.staleness_mode,
                a=self.staleness_a,
                b=self.staleness_b,
            )
            alpha_eff = self.fedasync_alpha * g_s
            job.staleness = staleness
            job.staleness_weight = alpha_eff
            accepted_jobs = [job]
            new_state = fedasync_aggregate_step(
                cur_state,
                res,
                alpha_eff=alpha_eff,
            )

        else:
            # `fedbuff`
            accepted_jobs = list(self.fedbuff_buffer)
            self.fedbuff_buffer = []
            for job in accepted_jobs:
                self._execute_real_client_training(job)
                staleness = max(0, int(self.current_version - job.start_version))
                g_s = compute_staleness_weight(
                    staleness,
                    mode=self.staleness_mode,
                    a=self.staleness_a,
                    b=self.staleness_b,
                )
                job.staleness = staleness
                job.staleness_weight = g_s

            new_state, self.fedbuff_velocity = fedbuff_aggregate_step(
                cur_state,
                accepted_jobs,
                self.snapshots,
                self.trainable_keys,
                server_velocity=self.fedbuff_velocity,
                server_lr=self.fedbuff_server_lr,
                server_momentum=self.fedbuff_server_momentum,
                normalization=self.fedbuff_normalization,
            )

        self._finalize_aggregation_step(
            next_version=next_ver,
            new_state=new_state,
            accepted_jobs=accepted_jobs,
            extra_metrics=extra_metrics,
            step_duration=step_duration,
            step_idle=step_idle,
            step_sync_wait=step_sync_wait,
            csv_writer=csv_writer,
            csv_file=csv_file,
        )

    def _finalize_aggregation_step(
        self,
        *,
        next_version: int,
        new_state: StateDict,
        accepted_jobs: Sequence[ClientJob],
        extra_metrics: Dict[str, float],
        step_duration: float,
        step_idle: float,
        step_sync_wait: float,
        csv_writer: csv.DictWriter,
        csv_file,
    ) -> None:
        """Update server model version, evaluate if scheduled, write CSV row, and schedule next jobs."""
        from fl_bench.engine.trainer import evaluate

        t_now = self.clock.current_time
        self.current_version = next_version
        self.snapshots[self.current_version] = {
            k: v.detach().cpu().clone() for k, v in new_state.items()
        }
        self.global_model.load_state_dict(new_state)
        self._prune_snapshots()

        step_accepted = len(accepted_jobs)
        step_examples = 0
        step_useful_comp = 0.0
        step_local_steps = 0
        step_fwd_passes = 0
        step_bwd_passes = 0
        step_up_bytes = self._step_extra_up_bytes
        step_down_bytes = self._step_extra_down_bytes
        step_staleness_list: List[int] = []

        weighted_loss_sum = 0.0
        weighted_acc_sum = 0.0

        for job in accepted_jobs:
            assert job.client_result is not None
            res = job.client_result
            job.status = "accepted"
            job.apply_version = next_version
            job.apply_time = t_now
            job.useful_compute_time = job.compute_duration
            job.useful_local_steps = int(res.local_steps or job.planned_local_steps)
            job.useful_forward_passes = int(
                res.forward_passes or job.planned_forward_passes
            )
            job.useful_backward_passes = int(
                res.backward_passes or job.planned_backward_passes
            )

            n_ex = int(res.num_examples)
            step_examples += n_ex
            weighted_loss_sum += float(res.train_loss) * n_ex
            weighted_acc_sum += float(res.train_accuracy) * n_ex

            step_useful_comp += job.useful_compute_time
            step_local_steps += job.useful_local_steps
            step_fwd_passes += job.useful_forward_passes
            step_bwd_passes += job.useful_backward_passes
            step_up_bytes += int(job.actual_upload_bytes)
            step_down_bytes += int(job.actual_download_bytes)
            self.cum_upload_bytes += int(job.actual_upload_bytes)
            self.cum_download_bytes += int(job.actual_download_bytes)

            s_val = int(job.staleness or 0)
            step_staleness_list.append(s_val)
            self.all_accepted_staleness.append(s_val)
            job.client_result = None

        self.cum_accepted_updates += step_accepted
        self.cum_useful_compute_time += step_useful_comp
        self.cum_local_steps += step_local_steps
        self.cum_forward_passes += step_fwd_passes
        self.cum_backward_passes += step_bwd_passes










