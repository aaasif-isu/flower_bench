"""Standardized CSV metric schemas and analytical communication/step accounting."""

from typing import Any, Dict, List, Optional, Tuple

from fl_bench.datasets import FederatedData
from fl_bench.methods.base import BaseMethod

CORE_CSV_FIELDS: Tuple[str, ...] = (
    "round",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
)

COMM_ACCOUNTING_FIELDS: Tuple[str, ...] = (
    "cumulative_fit_clients",
    "round_upload_bytes",
    "round_download_bytes",
    "round_local_steps",
    "round_forward_passes",
    "round_backward_passes",
    "cumulative_upload_bytes",
    "cumulative_download_bytes",
    "cumulative_local_steps",
    "cumulative_forward_passes",
    "cumulative_backward_passes",
)

# Analytical serialized `state_dict` byte sizes (float32 parameters + BatchNorm buffers)
MODEL_STATE_BYTES: Dict[str, int] = {
    "resnet18": 44_734_408,
    "femnist_cnn": 26_414_840,
    "shakespeare_lstm": 3_279_680,
}

# Analytical serialized trainable-parameter byte sizes (`requires_grad=True`, used for control variates / perturbations)
MODEL_TRAINABLE_BYTES: Dict[str, int] = {
    "resnet18": 44_696_104,
    "femnist_cnn": 26_414_840,
    "shakespeare_lstm": 3_279_680,
}

# Analytical penultimate feature dimensions and class counts for FedGPS prototype payload estimation
MODEL_PROTOTYPE_BYTES: Dict[str, int] = {
    "resnet18": 10 * (512 * 4 + 8),         # 10 classes * (512 float32 + 8B weight) = 20,560 B
    "femnist_cnn": 62 * (2048 * 4 + 8),     # 62 classes * (2048 float32 + 8B weight) = 508,400 B
    "shakespeare_lstm": 80 * (256 * 4 + 8), # 80 classes * (256 float32 + 8B weight) = 82,560 B
}

# Forward/backward pass multiplier per local optimizer step by method
METHOD_PASSES_PER_STEP: Dict[str, Tuple[int, int]] = {
    "fedavg": (1, 1),
    "fedprox": (1, 1),
    "fedavgm": (1, 1),
    "fedadam": (1, 1),
    "fednova": (1, 1),
    "scaffold": (1, 1),
    "fedspeed": (2, 2),   # ESAM ascent + descent: 2 sequential forward/backward passes per step
    "fedlesam": (1, 1),   # Single-backprop perturbation along global update
    "fedwmsam": (1, 1),   # Single-backprop perturbation along personalized momentum
    "fedgps": (1, 1),     # 1 forward/backward pass per step (on concatenated [real, surrogate] batch)
}

# Effective batch-size multiplier per forward/backward pass (1.0 = real batch only; 2.0 = real + surrogate batch)
METHOD_BATCH_SIZE_MULTIPLIER: Dict[str, float] = {
    "fedgps": 2.0,
}

# FedNova uploads one float64 scalar effective local step weight `a_i` per participating client
FEDNOVA_SCALAR_UPLOAD_BYTES: int = 8


def build_csv_fields(
    data: FederatedData,
    method: BaseMethod,
    include_comm_accounting: bool = False,
) -> List[str]:
    """
    Construct the exact CSV column order for the dataset, method, and track.
    """
    fields = list(CORE_CSV_FIELDS)
    if data.include_perplexity:
        fields.append("test_perplexity")
    fields.append("num_fit_clients")
    if data.include_num_fit_examples or include_comm_accounting:
        fields.append("num_fit_examples")
    fields.extend(method.extra_csv_fields)
    if include_comm_accounting:
        fields.extend(COMM_ACCOUNTING_FIELDS)
    fields.append("run_completed")
    return fields


def compute_round_comm_bytes(
    num_fit_clients: int,
    model_state_bytes: int,
    method: str = "fedavg",
    trainable_bytes: Optional[int] = None,
    *,
    round_id: int = 2,
    extra_upload_bytes: Optional[int] = None,
    extra_download_bytes: Optional[int] = None,
    fedlesam_variant: str = "base",
    model_name: str = "resnet18",
) -> Tuple[int, int]:
    """
    Compute analytical `(round_upload_bytes, round_download_bytes)` for a synchronous FL round.

    NOTE: These values are analytical serialized tensor payload estimates (not measured
    network socket traffic):
      - All methods download `model_state_bytes` and upload `model_state_bytes` per active client.
      - `scaffold`: downloads `c` (`trainable_bytes`) and uploads `delta_c_i` (`trainable_bytes`) per client.
      - `fednova`: uploads scalar `a_i` (+8 bytes) per client.
      - `fedspeed`: downloads `Local_dual_correction` (`trainable_bytes`) per client (`server/FedSpeed.py`).
      - `fedlesam`: downloads `global_update` (`trainable_bytes` for round >= 2) plus `Local_VR_correction`
        or `Local_dual_correction` (`trainable_bytes`) when `fedlesam_variant` is `"s"` or `"d"`.
      - `fedwmsam`: downloads personalized `Client_momentum` (`trainable_bytes`) per client (`server/FedWMSAM.py`).
      - `fedgps`: downloads `GLOBAL_UPDATE_PARAM` (`trainable_bytes` for round >= 2) + `GLOBAL_NOISE_PROTOS_PARAM`,
        and uploads `LOCAL_NOISE_PROTOS_PARAM` + `PROTO_AGG_WEIGHT`.
    """
    m_clients = int(num_fit_clients)
    base_bytes = int(model_state_bytes)
    cv_bytes = int(trainable_bytes) if trainable_bytes is not None else base_bytes
    method_key = str(method).strip().lower()

    if extra_upload_bytes is not None or extra_download_bytes is not None:
        return (
            m_clients * base_bytes + int(extra_upload_bytes or 0),
            m_clients * base_bytes + int(extra_download_bytes or 0),
        )

    per_client_down = base_bytes
    per_client_up = base_bytes

    if method_key == "scaffold":
        per_client_down += cv_bytes
        per_client_up += cv_bytes
    elif method_key == "fednova":
        per_client_up += FEDNOVA_SCALAR_UPLOAD_BYTES
    elif method_key == "fedspeed":
        per_client_down += cv_bytes
    elif method_key == "fedlesam":
        if int(round_id) >= 2:
            per_client_down += cv_bytes
        if str(fedlesam_variant).strip().lower() in ("s", "d"):
            per_client_down += cv_bytes
    elif method_key == "fedwmsam":
        per_client_down += cv_bytes
    elif method_key == "fedgps":
        proto_b = MODEL_PROTOTYPE_BYTES.get(model_name, 20_560)
        if int(round_id) >= 2:
            per_client_down += cv_bytes + proto_b
        per_client_up += proto_b

    return m_clients * per_client_up, m_clients * per_client_down


def describe_comm_accounting(
    method: str,
    model_name: str,
    model_state_bytes: Optional[int] = None,
    trainable_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Return structured metadata describing the analytical communication and compute model."""
    state_b = int(
        model_state_bytes
        if model_state_bytes is not None
        else MODEL_STATE_BYTES.get(model_name, 0)
    )
    train_b = int(
        trainable_bytes
        if trainable_bytes is not None
        else MODEL_TRAINABLE_BYTES.get(model_name, state_b)
    )
    up_1, down_1 = compute_round_comm_bytes(
        1,
        state_b,
        method=method,
        trainable_bytes=train_b,
        round_id=2,
        model_name=model_name,
    )
    fwd_mult, bwd_mult = METHOD_PASSES_PER_STEP.get(str(method).strip().lower(), (1, 1))
    batch_mult = METHOD_BATCH_SIZE_MULTIPLIER.get(str(method).strip().lower(), 1.0)
    return {
        "accounting_type": "analytical_serialized_tensor_estimate",
        "is_measured_network_traffic": False,
        "model": model_name,
        "method": method,
        "model_state_bytes": state_b,
        "trainable_parameter_bytes": train_b,
        "per_client_upload_bytes": up_1,
        "per_client_download_bytes": down_1,
        "per_client_total_bytes": up_1 + down_1,
        "forward_passes_per_step": fwd_mult,
        "backward_passes_per_step": bwd_mult,
        "effective_batch_size_multiplier": batch_mult,
    }
