"""
YAML-driven client system heterogeneity profiles (compute speed, upload/download
bandwidth, latency, availability, and mid-job dropout/failure behavior) for Track 3.

All distribution parameters are parsed from YAML configuration rather than hardcoded.
Supports both combined heterogeneity profiles (`homogeneous`, `mild`, `strong`, `severe`)
and isolated variation of individual factors (`compute`, `network`, `availability`, `dropout`).
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

SUPPORTED_PROFILE_NAMES: Tuple[str, ...] = (
    "homogeneous",
    "mild",
    "strong",
    "severe",
)

SUPPORTED_ISOLATE_FACTORS: Tuple[str, ...] = (
    "all",
    "compute",
    "network",
    "availability",
    "dropout",
)

FAILURE_PHASES: Tuple[str, ...] = ("download", "compute", "upload")


@dataclass(frozen=True)
class ClientSystemProfile:
    """Static baseline system profile assigned to a simulated client."""

    client_id: int
    profile_name: str
    base_compute_speed: float
    base_download_bw_mbps: float
    base_upload_bw_mbps: float
    base_latency_seconds: float
    availability_prob: float
    dropout_prob: float
    runtime_jitter_sigma: float
    failure_phase_probs: Tuple[float, float, float]

    @property
    def compute_multiplier(self) -> float:
        return 1.0 / max(1e-12, self.base_compute_speed)


@dataclass(frozen=True)
class JobSystemSample:
    """Realized per-job system and failure parameters for one client dispatch."""

    client_id: int
    compute_speed: float
    compute_multiplier: float
    download_bw_mbps: float
    upload_bw_mbps: float
    download_bw_bytes_per_sec: float
    upload_bw_bytes_per_sec: float
    latency_seconds: float
    will_fail: bool
    failure_phase: Optional[str]
    failure_progress_fraction: float


def _clip_value(val: float, spec: Dict[str, Any]) -> float:
    lo = spec.get("min_val")
    hi = spec.get("max_val")
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def sample_from_distribution(
    rng: np.random.Generator,
    spec: Any,
    *,
    field_name: str = "parameter",
) -> float:
    """
    Sample a single scalar from a YAML distribution specification dictionary.
    Accepts a numeric scalar (treated as a constant) or a mapping with `dist`.
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, dict):
        raise ValueError(
            f"Invalid distribution specification for '{field_name}': "
            f"expected dict or float, got {type(spec).__name__}."
        )

    dist = str(spec.get("dist", "constant")).strip().lower()
    if dist in ("constant", "fixed"):
        if "value" not in spec:
            raise ValueError(f"Distribution '{field_name}' (dist={dist}) requires 'value'.")
        return _clip_value(float(spec["value"]), spec)

    if dist == "lognormal":
        sigma = float(spec.get("sigma", 0.0))
        if sigma < 0.0:
            raise ValueError(f"Lognormal sigma for '{field_name}' must be >= 0, got {sigma}.")
        if "median" in spec:
            median = float(spec["median"])
            if median <= 0.0:
                raise ValueError(f"Lognormal median for '{field_name}' must be > 0, got {median}.")
            mu = math.log(median)
        elif "mu" in spec:
            mu = float(spec["mu"])
        elif "mean" in spec:
            mean_val = float(spec["mean"])
            if mean_val <= 0.0:
                raise ValueError(f"Lognormal mean for '{field_name}' must be > 0, got {mean_val}.")
            mu = math.log(mean_val) - 0.5 * (sigma ** 2)
        else:
            raise ValueError(
                f"Lognormal distribution for '{field_name}' requires 'median', 'mu', or 'mean'."
            )
        val = float(rng.lognormal(mean=mu, sigma=sigma)) if sigma > 0.0 else math.exp(mu)
        return _clip_value(val, spec)

    if dist == "uniform":
        low = float(spec["low"])
        high = float(spec["high"])
        if high < low:
            raise ValueError(f"Uniform distribution for '{field_name}' has high ({high}) < low ({low}).")
        val = float(rng.uniform(low, high)) if high > low else low
        return _clip_value(val, spec)

    if dist in ("normal", "truncated_normal", "gaussian"):
        mean_val = float(spec["mean"])
        std_val = float(spec.get("std", 0.0))
        if std_val < 0.0:
            raise ValueError(f"Normal std for '{field_name}' must be >= 0, got {std_val}.")
        val = float(rng.normal(loc=mean_val, scale=std_val)) if std_val > 0.0 else mean_val
        return _clip_value(val, spec)

    if dist == "beta":
        alpha = float(spec["alpha"])
        beta = float(spec["beta"])
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError(f"Beta parameters for '{field_name}' must be > 0, got ({alpha}, {beta}).")
        raw = float(rng.beta(alpha, beta))
        scale_lo = float(spec.get("scale_low", 0.0))
        scale_hi = float(spec.get("scale_high", 1.0))
        return _clip_value(scale_lo + (scale_hi - scale_lo) * raw, spec)

    if dist == "pareto":
        alpha = float(spec["alpha"])
        scale = float(spec.get("scale", 1.0))
        if alpha <= 0.0 or scale <= 0.0:
            raise ValueError(f"Pareto parameters for '{field_name}' must be > 0.")
        val = float(scale * (1.0 + rng.pareto(alpha)))
        return _clip_value(val, spec)

    if dist == "gamma":
        shape = float(spec["shape"])
        scale = float(spec.get("scale", 1.0))
        val = float(rng.gamma(shape, scale))
        return _clip_value(val, spec)

    if dist in ("categorical", "discrete"):
        values = [float(x) for x in spec["values"]]
        probs = spec.get("probabilities")
        if probs is not None:
            p_arr = np.asarray([float(p) for p in probs], dtype=np.float64)
            p_arr = p_arr / p_arr.sum()
            idx = int(rng.choice(len(values), p=p_arr))
        else:
            idx = int(rng.choice(len(values)))
        return _clip_value(values[idx], spec)

    raise ValueError(
        f"Unsupported distribution type '{dist}' for '{field_name}'. "
        "Supported: constant, lognormal, uniform, normal, beta, pareto, gamma, categorical."
    )


def resolve_effective_profile_spec(
    profile_name: str,
    profile_spec: Dict[str, Any],
    homogeneous_spec: Optional[Dict[str, Any]] = None,
    isolate_factor: str = "all",
) -> Dict[str, Any]:
    """
    Construct the effective profile specification, optionally isolating a single
    heterogeneity factor (`compute`, `network`, `availability`, or `dropout`)
    while holding non-isolated factors at their `homogeneous` baseline specification.
    """
    factor = str(isolate_factor or "all").strip().lower()
    if factor not in SUPPORTED_ISOLATE_FACTORS:
        raise ValueError(
            f"Unsupported isolate_factor '{isolate_factor}'. "
            f"Supported: {list(SUPPORTED_ISOLATE_FACTORS)}"
        )

    required_keys = (
        "compute_speed",
        "download_bandwidth_mbps",
        "upload_bandwidth_mbps",
        "latency_seconds",
        "availability_prob",
        "dropout_prob",
    )
    for req in required_keys:
        if req not in profile_spec:
            raise ValueError(
                f"System profile '{profile_name}' is missing required YAML field '{req}'."
            )

    if factor == "all" or profile_name == "homogeneous":
        return dict(profile_spec)

    if not homogeneous_spec:
        raise ValueError(
            f"Cannot apply isolate_factor='{factor}' without a 'homogeneous' baseline profile specification."
        )
    for req in required_keys:
        if req not in homogeneous_spec:
            raise ValueError(
                f"Baseline 'homogeneous' system profile is missing required YAML field '{req}'."
            )

    effective = dict(homogeneous_spec)
    if factor == "compute":
        effective["compute_speed"] = profile_spec["compute_speed"]
        if "runtime_jitter_sigma" in profile_spec:
            effective["runtime_jitter_sigma"] = profile_spec["runtime_jitter_sigma"]
    elif factor == "network":
        effective["download_bandwidth_mbps"] = profile_spec["download_bandwidth_mbps"]
        effective["upload_bandwidth_mbps"] = profile_spec["upload_bandwidth_mbps"]
        effective["latency_seconds"] = profile_spec["latency_seconds"]
    elif factor == "availability":
        effective["availability_prob"] = profile_spec["availability_prob"]
    elif factor == "dropout":
        effective["dropout_prob"] = profile_spec["dropout_prob"]
        if "failure_phase_probs" in profile_spec:
            effective["failure_phase_probs"] = profile_spec["failure_phase_probs"]

    return effective


def _parse_failure_phase_probs(spec: Dict[str, Any]) -> Tuple[float, float, float]:
    raw = spec.get("failure_phase_probs")
    if raw is None:
        raise ValueError(
            "System profile specification is missing required 'failure_phase_probs' "
            "(mapping with 'download', 'compute', 'upload')."
        )
    if isinstance(raw, dict):
        p_down = float(raw.get("download", 0.0))
        p_comp = float(raw.get("compute", 0.0))
        p_up = float(raw.get("upload", 0.0))
    elif isinstance(raw, (list, tuple)) and len(raw) == 3:
        p_down, p_comp, p_up = float(raw[0]), float(raw[1]), float(raw[2])
    else:
        raise ValueError(
            f"Invalid 'failure_phase_probs': expected dict or 3-tuple, got {raw!r}."
        )
    tot = p_down + p_comp + p_up
    if tot <= 0.0:
        raise ValueError("Sum of 'failure_phase_probs' must be positive.")
    return (p_down / tot, p_comp / tot, p_up / tot)


class ClientProfileManager:
    """
    Deterministic generator and manager for client system heterogeneity profiles.

    Given `profile_seed`, generates identical per-client hardware, network,
    availability, and dropout traits across algorithms and execution strategies.
    """

    def __init__(
        self,
        *,
        num_clients: int,
        profile_name: str,
        profile_spec: Dict[str, Any],
        homogeneous_spec: Optional[Dict[str, Any]] = None,
        profile_seed: int = 42,
        simulation_seed: int = 42,
        isolate_factor: str = "all",
        bits_per_megabit: float = 1_000_000.0,
    ) -> None:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}.")
        self.num_clients = int(num_clients)
        self.profile_name = str(profile_name)
        self.profile_seed = int(profile_seed)
        self.simulation_seed = int(simulation_seed)
        self.isolate_factor = str(isolate_factor or "all").strip().lower()
        self.bytes_per_mbps = float(bits_per_megabit) / 8.0

        self.effective_spec = resolve_effective_profile_spec(
            self.profile_name,
            profile_spec,
            homogeneous_spec=homogeneous_spec,
            isolate_factor=self.isolate_factor,
        )
        self.clients: Dict[int, ClientSystemProfile] = self._build_client_profiles()

    def _build_client_profiles(self) -> Dict[int, ClientSystemProfile]:
        rng = np.random.default_rng(self.profile_seed)
        phase_probs = _parse_failure_phase_probs(self.effective_spec)
        jitter_sigma = float(self.effective_spec.get("runtime_jitter_sigma", 0.0) or 0.0)

        profiles: Dict[int, ClientSystemProfile] = {}
        for cid in range(self.num_clients):
            comp = sample_from_distribution(
                rng,
                self.effective_spec["compute_speed"],
                field_name=f"{self.profile_name}.compute_speed",
            )
            down_bw = sample_from_distribution(
                rng,
                self.effective_spec["download_bandwidth_mbps"],
                field_name=f"{self.profile_name}.download_bandwidth_mbps",
            )
            up_bw = sample_from_distribution(
                rng,
                self.effective_spec["upload_bandwidth_mbps"],
                field_name=f"{self.profile_name}.upload_bandwidth_mbps",
            )
            latency = sample_from_distribution(
                rng,
                self.effective_spec["latency_seconds"],
                field_name=f"{self.profile_name}.latency_seconds",
            )
            avail = sample_from_distribution(
                rng,
                self.effective_spec["availability_prob"],
                field_name=f"{self.profile_name}.availability_prob",
            )
            dropout = sample_from_distribution(
                rng,
                self.effective_spec["dropout_prob"],
                field_name=f"{self.profile_name}.dropout_prob",
            )

            if comp <= 0.0:
                raise ValueError(f"Sampled non-positive compute_speed={comp} for client {cid}.")
            if down_bw <= 0.0 or up_bw <= 0.0:
                raise ValueError(
                    f"Sampled non-positive bandwidth (down={down_bw}, up={up_bw}) for client {cid}."
                )
            if latency < 0.0:
                raise ValueError(f"Sampled negative latency={latency} for client {cid}.")

            profiles[cid] = ClientSystemProfile(
                client_id=cid,
                profile_name=self.profile_name,
                base_compute_speed=float(comp),
                base_download_bw_mbps=float(down_bw),
                base_upload_bw_mbps=float(up_bw),
                base_latency_seconds=float(latency),
                availability_prob=float(min(1.0, max(0.0, avail))),
                dropout_prob=float(min(1.0, max(0.0, dropout))),
                runtime_jitter_sigma=max(0.0, jitter_sigma),
                failure_phase_probs=phase_probs,
            )
        return profiles

    def get_client_profile(self, client_id: int) -> ClientSystemProfile:
        return self.clients[int(client_id)]

    def is_client_available(self, client_id: int, check_index: int) -> bool:
        """
        Deterministically sample whether `client_id` is available at selection
        attempt `check_index`.
        """
        prof = self.clients[int(client_id)]
        if prof.availability_prob >= 1.0:
            return True
        if prof.availability_prob <= 0.0:
            return False
        rng = np.random.default_rng(
            [self.simulation_seed, self.profile_seed, 101, int(client_id), int(check_index)]
        )
        return bool(float(rng.random()) < prof.availability_prob)

    def sample_job_parameters(self, client_id: int, client_job_index: int) -> JobSystemSample:
        """
        Deterministically sample the realized compute speed, upload/download bandwidths,
        latency, and failure behavior for `client_id`'s `client_job_index`-th dispatched job.
        """
        prof = self.clients[int(client_id)]
        rng = np.random.default_rng(
            [self.simulation_seed, self.profile_seed, 202, int(client_id), int(client_job_index)]
        )

        if prof.runtime_jitter_sigma > 0.0:
            s_sigma = prof.runtime_jitter_sigma
            mu_zero_mean = -0.5 * (s_sigma ** 2)
            comp_jitter = float(rng.lognormal(mean=mu_zero_mean, sigma=s_sigma))
            down_jitter = float(rng.lognormal(mean=mu_zero_mean, sigma=s_sigma))
            up_jitter = float(rng.lognormal(mean=mu_zero_mean, sigma=s_sigma))
            lat_jitter = float(rng.lognormal(mean=mu_zero_mean, sigma=0.5 * s_sigma))
        else:
            comp_jitter = 1.0
            down_jitter = 1.0
            up_jitter = 1.0
            lat_jitter = 1.0

        comp_speed = max(1e-6, prof.base_compute_speed * comp_jitter)
        down_mbps = max(1e-3, prof.base_download_bw_mbps * down_jitter)
        up_mbps = max(1e-3, prof.base_upload_bw_mbps * up_jitter)
        latency_s = max(0.0, prof.base_latency_seconds * lat_jitter)

        will_fail = False
        failure_phase: Optional[str] = None
        failure_progress = 1.0
        if prof.dropout_prob > 0.0:
            if float(rng.random()) < prof.dropout_prob:
                will_fail = True
                phase_idx = int(rng.choice(3, p=prof.failure_phase_probs))
                failure_phase = FAILURE_PHASES[phase_idx]
                failure_progress = float(rng.uniform(0.05, 0.95))

        return JobSystemSample(
            client_id=int(client_id),
            compute_speed=comp_speed,
            compute_multiplier=1.0 / comp_speed,
            download_bw_mbps=down_mbps,
            upload_bw_mbps=up_mbps,
            download_bw_bytes_per_sec=down_mbps * self.bytes_per_mbps,
            upload_bw_bytes_per_sec=up_mbps * self.bytes_per_mbps,
            latency_seconds=latency_s,
            will_fail=will_fail,
            failure_phase=failure_phase,
            failure_progress_fraction=failure_progress,
        )

    def summarize_population(self) -> Dict[str, Any]:
        """Return summary statistics of the generated client population profiles."""
        comp = [p.base_compute_speed for p in self.clients.values()]
        down = [p.base_download_bw_mbps for p in self.clients.values()]
        up = [p.base_upload_bw_mbps for p in self.clients.values()]
        lat = [p.base_latency_seconds for p in self.clients.values()]
        avail = [p.availability_prob for p in self.clients.values()]
        drop = [p.dropout_prob for p in self.clients.values()]

        def _stats(arr: Sequence[float]) -> Dict[str, float]:
            a = np.asarray(arr, dtype=np.float64)
            return {
                "mean": round(float(np.mean(a)), 6),
                "std": round(float(np.std(a, ddof=1 if len(a) > 1 else 0)), 6),
                "min": round(float(np.min(a)), 6),
                "median": round(float(np.median(a)), 6),
                "p90": round(float(np.percentile(a, 90)), 6),
                "max": round(float(np.max(a)), 6),
            }

        return {
            "profile_name": self.profile_name,
            "isolate_factor": self.isolate_factor,
            "profile_seed": self.profile_seed,
            "simulation_seed": self.simulation_seed,
            "num_clients": self.num_clients,
            "compute_speed": _stats(comp),
            "download_bandwidth_mbps": _stats(down),
            "upload_bandwidth_mbps": _stats(up),
            "latency_seconds": _stats(lat),
            "availability_prob": _stats(avail),
            "dropout_prob": _stats(drop),
        }


