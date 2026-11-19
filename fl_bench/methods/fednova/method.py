"""FedNova normalized averaging method implementation (Wang et al., 2020)."""

from typing import Dict, Sequence, Set, Tuple

import torch

from fl_bench.methods.base import BaseMethod, ClientResult, StateDict


class FedNovaMethod(BaseMethod):
    """FedNova: normalized averaging to eliminate objective inconsistency."""

    name = "fednova"
    extra_csv_fields = (
        "tau_eff",
        "mean_local_normalizer",
        "mean_local_steps",
    )

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        total_examples = sum(r.num_examples for r in client_results)
        weights = [r.num_examples / total_examples for r in client_results]
        normalizers = [r.local_normalizer for r in client_results]

        tau_eff = sum(w * a_i for w, a_i in zip(weights, normalizers))
        if tau_eff <= 0:
            raise RuntimeError(f"Invalid FedNova tau_eff: {tau_eff}")

        new_state: StateDict = {}
        first_state = client_results[0].state

        for key, global_value in global_state.items():
            if not torch.is_floating_point(global_value):
                new_state[key] = first_state[key].clone()
                continue

            if key in trainable_keys:
                accumulator = torch.zeros_like(
                    global_value, dtype=torch.float64
                )
                global64 = global_value.to(torch.float64)

                for result, weight, a_i in zip(
                    client_results, weights, normalizers
                ):
                    delta = result.state[key].to(torch.float64) - global64
                    scale = weight * tau_eff / a_i
                    accumulator += scale * delta

                new_state[key] = (global64 + accumulator).to(
                    global_value.dtype
                )
            else:
                accumulator = torch.zeros_like(
                    global_value, dtype=torch.float64
                )
                for result, weight in zip(client_results, weights):
                    accumulator += result.state[key].to(torch.float64) * weight
                new_state[key] = accumulator.to(global_value.dtype)

        mean_normalizer = sum(normalizers) / len(client_results)
        mean_steps = (
            sum(r.local_steps for r in client_results) / len(client_results)
        )

        return new_state, {
            "tau_eff": tau_eff,
            "mean_local_normalizer": mean_normalizer,
            "mean_local_steps": mean_steps,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"tau_eff={extra_metrics['tau_eff']:.4f} "
            f"a_i={extra_metrics['mean_local_normalizer']:.4f} "
            f"steps={extra_metrics['mean_local_steps']:.1f}"
        )


FedNovaAlgorithm = FedNovaMethod
