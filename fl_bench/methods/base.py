"""Base FL method interface and shared aggregation primitives."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from fl_bench.config import ExperimentConfig

StateDict = Dict[str, torch.Tensor]


@dataclass
class ClientResult:
    """Result returned by a single client after local training."""

    state: StateDict
    num_examples: int
    train_loss: float
    train_accuracy: float
    local_normalizer: float = 0.0
    local_steps: int = 0
    c_delta: Optional[StateDict] = None
    cid: Optional[int] = None
    forward_passes: int = 0
    backward_passes: int = 0
    extra_upload_bytes: int = 0
    extra_download_bytes: int = 0
    upload_bytes_override: Optional[int] = None
    download_bytes_override: Optional[int] = None
    aux_payload: Optional[Dict[str, Any]] = None


def fedavg_aggregate(client_results: Sequence[ClientResult]) -> StateDict:
    """
    Sample-weighted FedAvg aggregation in float64.
    Non-floating-point buffers (e.g. BatchNorm `num_batches_tracked`) are
    cloned from the first selected client.
    """
    total_examples = sum(r.num_examples for r in client_results)
    first_state = client_results[0].state
    averaged: StateDict = {}

    for key, value in first_state.items():
        if not torch.is_floating_point(value):
            averaged[key] = value.clone()
            continue

        accumulator = torch.zeros_like(value, dtype=torch.float64)
        for result in client_results:
            accumulator += (
                result.state[key].to(torch.float64)
                * (result.num_examples / total_examples)
            )
        averaged[key] = accumulator.to(value.dtype)

    return averaged


def tensor_dict_norm(values: StateDict) -> float:
    """Compute the L2 norm of a dictionary of tensors in float64."""
    total = 0.0
    for value in values.values():
        total += value.to(torch.float64).pow(2).sum().item()
    return total ** 0.5


class BaseMethod:
    """Base interface for Federated Learning methods."""

    name: str = "base"
    extra_csv_fields: Tuple[str, ...] = ()
    cleanup_after_round: bool = False

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.current_round_id: int = 1

    @classmethod
    def validate_compatibility(cls, config: ExperimentConfig) -> None:
        """
        Validate that `config` satisfies this algorithm's mathematical and architectural
        requirements (model architecture, input data modality, client participation,
        and auxiliary data). Raises `ValueError` on unsupported combinations during
        config resolution and `--plan` dry runs.
        """

    def current_client_lr(self) -> float:
        """Return the effective client learning rate for `self.current_round_id` (`lr * lr_decay^(r-1)`)."""
        assert self.config.lr is not None
        decay = float(getattr(self.config, "lr_decay", 1.0) or 1.0)
        if math.isclose(decay, 1.0):
            return float(self.config.lr)
        return float(self.config.lr) * (decay ** max(0, int(self.current_round_id) - 1))

    def create_global_model(self, device: torch.device) -> nn.Module:
        """Instantiate the global model for this method on `device`."""
        from fl_bench.models import create_model

        assert self.config.model is not None
        return create_model(self.config.model).to(device)

    def init_server(self, global_model: nn.Module) -> None:
        """Initialize any server-side state from the initial global model."""

    def run_client(
        self,
        cid: int,
        global_state: StateDict,
        client_dataset: Dataset,
        device: torch.device,
        seed: int,
    ) -> ClientResult:
        """Execute local client update for client `cid`."""
        from fl_bench.engine.trainer import train_client

        return train_client(
            cid=cid,
            global_state=global_state,
            client_dataset=client_dataset,
            device=device,
            config=self.config,
            seed=seed,
        )

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        """Aggregate client updates into a new global state and extra metrics."""
        raise NotImplementedError

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        """Optional extra console log string for method-specific metrics."""
        return ""


BaseAlgorithm = BaseMethod
