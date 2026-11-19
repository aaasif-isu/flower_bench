"""SCAFFOLD control-variate method implementation (Karimireddy et al., 2020)."""

import math
from typing import Dict, Sequence, Set, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    tensor_dict_norm,
)


def zero_control(global_model: nn.Module) -> StateDict:
    """Create a zero-initialized control variate dict on CPU."""
    return {
        name: torch.zeros_like(param.detach().cpu())
        for name, param in global_model.named_parameters()
        if param.requires_grad
    }


class ScaffoldMethod(BaseMethod):
    """
    Paper-faithful SCAFFOLD (Karimireddy et al., ICML 2020) with Option II
    control-variate updates, vanilla local SGD (`momentum=0.0`), and lazy
    CPU per-client control-variate storage.
    """

    name = "scaffold"
    extra_csv_fields = (
        "mean_local_steps",
        "c_global_norm",
        "mean_c_delta_norm",
    )
    cleanup_after_round = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.c_global: StateDict = {}
        self.client_controls: Dict[int, StateDict] = {}

    def init_server(self, global_model: nn.Module) -> None:
        self.c_global = zero_control(global_model)
        self.client_controls = {}

    def run_client(
        self,
        cid: int,
        global_state: StateDict,
        client_dataset: Dataset,
        device: torch.device,
        seed: int,
    ) -> ClientResult:
        from fl_bench.engine.trainer import train_client

        if cid not in self.client_controls:
            self.client_controls[cid] = {
                name: torch.zeros_like(val)
                for name, val in self.c_global.items()
            }

        c_local = self.client_controls[cid]
        result = train_client(
            cid=cid,
            global_state=global_state,
            client_dataset=client_dataset,
            device=device,
            config=self.config,
            seed=seed,
            c_global=self.c_global,
            c_local=c_local,
        )

        assert result.c_delta is not None
        self.client_controls[cid] = {
            name: (
                c_local[name].to(torch.float64)
                + result.c_delta[name].to(torch.float64)
            ).to(c_local[name].dtype)
            for name in self.c_global
        }
        return result

    def aggregate(
        self,
        round_id: int,
        global_state: StateDict,
        client_results: Sequence[ClientResult],
        trainable_keys: Set[str],
    ) -> Tuple[StateDict, Dict[str, float]]:
        num_selected = len(client_results)
        if num_selected == 0:
            raise RuntimeError("No selected clients")

        assert self.config.num_clients is not None
        uniform_weight = 1.0 / num_selected
        new_state: StateDict = {}

        for key, global_value in global_state.items():
            if not torch.is_floating_point(global_value):
                # Integer buffers (e.g. BatchNorm num_batches_tracked)
                new_state[key] = client_results[0].state[key].clone()
                continue

            if key in trainable_keys:
                # Algorithm 1 server update: x <- x + (1 / |S|) * sum_{i in S} (y_i - x)
                avg_delta = torch.zeros_like(
                    global_value, dtype=torch.float64
                )
                for result in client_results:
                    avg_delta += (
                        (
                            result.state[key].to(torch.float64)
                            - global_value.to(torch.float64)
                        )
                        * uniform_weight
                    )
                updated_param = (
                    global_value.to(torch.float64) + avg_delta
                ).to(global_value.dtype)
            else:
                # Non-trainable floating-point BatchNorm buffers (running_mean, running_var)
                # aggregated with the same uniform client weighting (1 / |S|) as trainable params.
                accumulator = torch.zeros_like(
                    global_value, dtype=torch.float64
                )
                for result in client_results:
                    accumulator += (
                        result.state[key].to(torch.float64) * uniform_weight
                    )
                updated_param = accumulator.to(global_value.dtype)

            if not torch.isfinite(updated_param).all():
                raise FloatingPointError(
                    f"Round {round_id}: Nonfinite global model tensor '{key}' in SCAFFOLD aggregation."
                )
            new_state[key] = updated_param

        # Algorithm 1 global control variate update:
        # c <- c + (|S| / N) * (1 / |S|) * sum_{i in S} c_delta_i
        participation_ratio = num_selected / self.config.num_clients
        new_c_global: StateDict = {}

        for name, old_c in self.c_global.items():
            avg_c_delta = torch.zeros_like(old_c, dtype=torch.float64)
            for result in client_results:
                assert result.c_delta is not None
                avg_c_delta += (
                    result.c_delta[name].to(torch.float64) * uniform_weight
                )
            updated_c = (
                old_c.to(torch.float64) + participation_ratio * avg_c_delta
            ).to(old_c.dtype)
            if not torch.isfinite(updated_c).all():
                raise FloatingPointError(
                    f"Round {round_id}: Nonfinite global control variate '{name}' in SCAFFOLD aggregation."
                )
            new_c_global[name] = updated_c

        self.c_global = new_c_global

        mean_steps = (
            sum(r.local_steps for r in client_results) / num_selected
        )
        c_global_norm = tensor_dict_norm(self.c_global)
        mean_c_delta_norm = (
            sum(
                tensor_dict_norm(r.c_delta)
                for r in client_results
                if r.c_delta is not None
            )
            / num_selected
        )

        if not (math.isfinite(c_global_norm) and math.isfinite(mean_c_delta_norm)):
            raise FloatingPointError(
                f"Round {round_id}: Nonfinite SCAFFOLD control-variate norms "
                f"(c_global_norm={c_global_norm}, mean_c_delta_norm={mean_c_delta_norm})."
            )

        return new_state, {
            "mean_local_steps": mean_steps,
            "c_global_norm": c_global_norm,
            "mean_c_delta_norm": mean_c_delta_norm,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"steps={extra_metrics['mean_local_steps']:.1f} "
            f"c_global_norm={extra_metrics['c_global_norm']:.4f} "
            f"c_delta_norm={extra_metrics['mean_c_delta_norm']:.4f}"
        )


ScaffoldAlgorithm = ScaffoldMethod
