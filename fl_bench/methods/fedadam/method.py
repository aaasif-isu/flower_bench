"""FedAdam method implementation (Reddi et al., 2021)."""

from typing import Dict, Optional, Sequence, Set, Tuple

import torch

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    fedavg_aggregate,
)


class FedAdamMethod(BaseMethod):
    """Adaptive Federated Optimization using server-side Adam."""

    name = "fedadam"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.server_m: Optional[StateDict] = None
        self.server_v: Optional[StateDict] = None

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        averaged_state = fedavg_aggregate(client_results)

        if self.server_m is None or self.server_v is None:
            self.server_m = {
                key: torch.zeros_like(global_state[key])
                for key in trainable_keys
            }
            self.server_v = {
                key: torch.zeros_like(global_state[key])
                for key in trainable_keys
            }

        assert self.config.server_lr is not None
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        tau = self.config.tau

        # Bias-corrected server learning rate (matches Flower FedAdam: round_id + 1).
        eta_norm = (
            self.config.server_lr
            * ((1.0 - beta2 ** (round_id + 1)) ** 0.5)
            / (1.0 - beta1 ** (round_id + 1))
        )

        new_state: StateDict = {}

        for key, current_value in global_state.items():
            if key not in trainable_keys:
                new_state[key] = averaged_state[key].clone()
                continue

            delta = averaged_state[key] - current_value

            self.server_m[key] = beta1 * self.server_m[key] + (1.0 - beta1) * delta
            self.server_v[key] = (
                beta2 * self.server_v[key] + (1.0 - beta2) * (delta * delta)
            )

            new_state[key] = (
                current_value
                + eta_norm
                * self.server_m[key]
                / (torch.sqrt(self.server_v[key]) + tau)
            )

        return new_state, {}


FedAdamAlgorithm = FedAdamMethod
