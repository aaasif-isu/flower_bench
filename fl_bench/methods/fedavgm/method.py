"""FedAvgM (Server Momentum) method implementation (Hsu et al., 2019)."""

from typing import Dict, Optional, Sequence, Set, Tuple

import torch

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    fedavg_aggregate,
)


class FedAvgMMethod(BaseMethod):
    """FedAvg with server-side momentum on trainable parameters."""

    name = "fedavgm"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.server_velocity: Optional[StateDict] = None

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        averaged_state = fedavg_aggregate(client_results)

        if self.server_velocity is None:
            self.server_velocity = {
                key: torch.zeros_like(global_state[key])
                for key in trainable_keys
            }

        assert self.config.server_lr is not None
        new_state: StateDict = {}

        for key, current_value in global_state.items():
            if key not in trainable_keys:
                new_state[key] = averaged_state[key].clone()
                continue

            delta = current_value - averaged_state[key]
            self.server_velocity[key] = (
                self.config.server_momentum * self.server_velocity[key]
                + delta
            )
            new_state[key] = (
                current_value
                - self.config.server_lr * self.server_velocity[key]
            )

        return new_state, {}


FedAvgMAlgorithm = FedAvgMMethod
