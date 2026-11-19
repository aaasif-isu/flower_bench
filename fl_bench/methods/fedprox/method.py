"""FedProx method implementation (Li et al., 2020)."""

from typing import Dict, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    fedavg_aggregate,
)


class FedProxMethod(BaseMethod):
    """FedProx: local objective includes `(mu / 2) * ||w - w_global||^2`."""

    name = "fedprox"

    def run_client(
        self,
        cid: int,
        global_state: StateDict,
        client_dataset: Dataset,
        device: torch.device,
        seed: int,
    ) -> ClientResult:
        from fl_bench.engine.trainer import train_client

        return train_client(
            cid=cid,
            global_state=global_state,
            client_dataset=client_dataset,
            device=device,
            config=self.config,
            seed=seed,
            mu=self.config.mu,
        )

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        return fedavg_aggregate(client_results), {}


FedProxAlgorithm = FedProxMethod
