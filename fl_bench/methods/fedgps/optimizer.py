"""
FedGPS path-rectification optimizer and leave-one-out non-self global update
primitives (Yang et al., NeurIPS 2025).
Reference: https://github.com/visitworld123/FedGPS/blob/main/utils/optim.py
and https://github.com/visitworld123/FedGPS/blob/main/FedGPS/FedGPSClient.py
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from fl_bench.methods.base import StateDict


def compute_fedgps_perturbation(
    *,
    source_update: Optional[StateDict],
    client_prev_delta: Optional[StateDict],
    trainable_names: Iterable[str],
    clients_per_round: int,
    pre_flag: bool,
    pre_selected_client_use_minus: bool = True,
    require_prev_delta: bool = True,
    use_minus_lesam: bool = False,
    rho: float = 0.5,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Compute the FedGPS non-self global update direction `g_other` and perturbation `e_w`
    matching `FedGPSClient.train()` (`FedGPSClient.py:222-246`) and `FedGPSOptimizer.first_step()`
    (`utils/optim.py:34-54`).

    In `FedGPSClient.py:232`:
      `if self.global_update != None and len(self.model_delta) > 0:`
    When `require_prev_delta=True` (reference default), perturbation is active only if
    the client has participated in at least one previous round (`client_prev_delta is not None`).
    """
    if source_update is None or rho <= 0.0:
        return {}
    if require_prev_delta and not client_prev_delta:
        return {}

    use_minus = pre_flag if pre_selected_client_use_minus else (not pre_flag)
    m_cpr = max(2, int(clients_per_round))
    names = list(trainable_names)

    g_other: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in names:
            if name not in source_update:
                continue
            u_param = source_update[name].to(torch.float64)
            if use_minus and client_prev_delta is not None and name in client_prev_delta:
                prev_d = client_prev_delta[name].to(torch.float64)
                g_vec = (float(m_cpr) * u_param - prev_d) / float(m_cpr - 1)
            else:
                g_vec = u_param
            g_other[name] = g_vec.to(source_update[name].dtype)

        # In visitworld123/FedGPS/utils/optim.py:
        #   grad_norm = 0; grad_norm -= g_update[idx].norm(p=2)
        #   scale = rho / (grad_norm + 1e-7); e_w = -g_update[idx] * scale
        # Because grad_norm is negative (-sum_norm), -g_update * (rho / (-sum_norm)) = +g_update * (rho / sum_norm)
        # ("minuse_is_positive: True" in conf/algorithms/fedgps.yaml).
        sum_norm = sum(float(v.norm(p=2).item()) for v in g_other.values())
        if sum_norm <= 0.0:
            return {}

        if use_minus_lesam:
            scale = -float(rho) / (sum_norm + 1e-7)
        else:
            grad_norm = -sum_norm
            scale = -float(rho) / (grad_norm + 1e-7)

        return {
            name: (g_vec * scale).to(device)
            for name, g_vec in g_other.items()
        }


class FedGPSOptimizer(torch.optim.Optimizer):
    """
    FedGPS path-rectification optimizer matching `visitworld123/FedGPS/utils/optim.py`.
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.5,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid perturbation rate, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho

    @torch.no_grad()
    def first_step(
        self,
        named_trainable: List[Tuple[str, nn.Parameter]],
        e_w_device: Dict[str, torch.Tensor],
    ) -> None:
        if not e_w_device:
            return
        for name, p in named_trainable:
            if name in e_w_device:
                p.add_(e_w_device[name])

    @torch.no_grad()
    def second_step(
        self,
        named_trainable: List[Tuple[str, nn.Parameter]],
        e_w_device: Dict[str, torch.Tensor],
    ) -> None:
        if not e_w_device:
            return
        for name, p in named_trainable:
            if name in e_w_device:
                p.sub_(e_w_device[name])
