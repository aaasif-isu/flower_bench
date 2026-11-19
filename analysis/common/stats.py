"""Statistical aggregation and trapezoidal learning-curve integration utilities."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SampleSummary:
    """Summary statistics across K independent seeds (sample std uses K-1)."""

    count: int
    mean: float
    sample_std: float
    sem: float
    minimum: float
    maximum: float


def compute_sample_summary(values: Sequence[float]) -> SampleSummary:
    """
    Compute mean, sample standard deviation (denominator K-1 for K >= 2, else NaN),
    standard error of the mean, min, and max over finite `values`.
    """
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    k = len(clean)
    if k == 0:
        nan = float("nan")
        return SampleSummary(
            count=0,
            mean=nan,
            sample_std=nan,
            sem=nan,
            minimum=nan,
            maximum=nan,
        )

    mean_val = sum(clean) / k
    if k >= 2:
        var = sum((x - mean_val) ** 2 for x in clean) / (k - 1)
        std_val = math.sqrt(max(0.0, var))
        sem_val = std_val / math.sqrt(k)
    else:
        std_val = float("nan")
        sem_val = float("nan")

    return SampleSummary(
        count=k,
        mean=mean_val,
        sample_std=std_val,
        sem=sem_val,
        minimum=min(clean),
        maximum=max(clean),
    )


def trapezoidal_learning_auc(
    checkpoints: Sequence[Tuple[int, float]],
    expected_final_round: int,
) -> Dict[str, object]:
    """
    Compute the normalized area under the test-accuracy learning curve using
    composite trapezoidal integration over actual observed evaluation checkpoints:

        auc_observed = (1 / (r_end - r_start)) * sum_{k=1}^{M-1} 0.5 * (A(r_k) + A(r_{k+1})) * (r_{k+1} - r_k)
        auc_nominal_r = (1 / R) * sum_{k=1}^{M-1} 0.5 * (A(r_k) + A(r_{k+1})) * (r_{k+1} - r_k)

    Does NOT invent or extrapolate Round 0 if Round 0 is absent from `checkpoints`.
    """
    if len(checkpoints) < 2:
        raise ValueError(
            f"At least 2 evaluation checkpoints are required for trapezoidal AUC, got {len(checkpoints)}."
        )

    pts: List[Tuple[int, float]] = sorted(
        [(int(r), float(a)) for r, a in checkpoints],
        key=lambda x: x[0],
    )
    for r, a in pts:
        if not math.isfinite(a):
            raise ValueError(f"Nonfinite accuracy ({a}) at round {r}.")

    r_start = pts[0][0]
    r_end = pts[-1][0]
    if r_end != int(expected_final_round):
        raise ValueError(
            f"Last evaluation checkpoint round ({r_end}) does not match expected final round ({expected_final_round})."
        )

    span = r_end - r_start
    if span <= 0:
        raise ValueError(f"Invalid evaluation round span: {r_start}..{r_end}.")

    raw_integral = 0.0
    for (r0, a0), (r1, a1) in zip(pts[:-1], pts[1:]):
        dr = r1 - r0
        if dr <= 0:
            raise ValueError(
                f"Duplicate or non-increasing evaluation rounds: {r0} -> {r1}."
            )
        raw_integral += 0.5 * (a0 + a1) * dr

    round_0_present = r_start == 0
    auc_observed = raw_integral / float(span)
    auc_nominal_r = raw_integral / float(expected_final_round)

    return {
        "auc_fraction": auc_observed,
        "auc_percent": auc_observed * 100.0,
        "auc_nominal_r_fraction": auc_nominal_r,
        "raw_integral": raw_integral,
        "eval_start_round": r_start,
        "eval_end_round": r_end,
        "observed_span_rounds": span,
        "nominal_total_rounds": int(expected_final_round),
        "num_eval_checkpoints": len(pts),
        "round_0_in_csv": round_0_present,
        "integration_note": (
            "Integrated over [0, R]"
            if round_0_present
            else f"Round 0 absent from CSV; trapezoidal integral over observed interval [{r_start}, {r_end}] normalized by (R - r_start) = {span}"
        ),
    }
