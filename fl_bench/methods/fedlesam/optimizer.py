"""
LESAM, LESAM-S, and LESAM-D perturbation and optimizer primitives
(Fan et al., ICML 2024 Spotlight).
Reference: https://github.com/MediaBrain-SJTU/FedLESAM/tree/main/optimizer
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


def compute_lesam_perturbation(
    g_update: Optional[Dict[str, torch.Tensor]],
    trainable_names: Iterable[str],
    rho: float,
    device: torch.device,
    norm_mode: str = "reference_sum_l2",
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Compute the LESAM parameter perturbation `e_w = -g_update * (rho / (grad_norm + 1e-7))`.

    Supports both:
      - `norm_mode == "reference_sum_l2"` (default): matches `LESAM.py:29` in the official
        repository (`grad_norm = sum_p ||g_update[p]||_2`).
      - `norm_mode == "paper_global_l2"`: matches the ICML 2024 paper equation
        (`grad_norm = sqrt(sum_p ||g_update[p]||_2^2)`).
    """
    if g_update is None or rho <= 0.0:
        return {}, 0.0

    names = list(trainable_names)
    with torch.no_grad():
        if norm_mode == "paper_global_l2":
            sq_sum = sum(
                float(g_update[n].to(torch.float64).pow(2).sum().item())
                for n in names
                if n in g_update
            )
            grad_norm = sq_sum ** 0.5
        else:
            grad_norm = sum(
                float(g_update[n].norm(p=2).item())
                for n in names
                if n in g_update
            )

        if grad_norm <= 0.0:
            return {}, 0.0

        scale = float(rho) / (grad_norm + 1e-7)
        e_w_device: Dict[str, torch.Tensor] = {}
        pert_norm_sq = 0.0
        for n in names:
            if n not in g_update:
                continue
            ew = (-g_update[n] * scale).to(device)
            e_w_device[n] = ew
            pert_norm_sq += float(ew.to(torch.float64).pow(2).sum().item())

        return e_w_device, (pert_norm_sq ** 0.5)


class LESAM(torch.optim.Optimizer):
    """
    Locally Estimated Global Perturbation SAM optimizer (`LESAM`, `LESAM_S`, `LESAM_D`)
    matching `MediaBrain-SJTU/FedLESAM/optimizer/LESAM.py`.
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.1,
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
