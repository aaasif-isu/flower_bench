"""Federated Averaging (FedAvg) method implementation (McMahan et al., 2017)."""

from typing import Dict, Sequence, Set, Tuple

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    fedavg_aggregate,
)


class FedAvgMethod(BaseMethod):
    """Standard McMahan et al. (2017) Federated Averaging."""

    name = "fedavg"

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        return fedavg_aggregate(client_results), {}


FedAvgAlgorithm = FedAvgMethod
