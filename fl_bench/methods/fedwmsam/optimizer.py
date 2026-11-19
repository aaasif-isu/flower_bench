"""
WMSAM optimizer, `loss_gamma` objective, and alignment score primitives
for FedWMSAM (Li et al., NeurIPS 2025).
Reference: https://github.com/Huang-Yongzhi/NeurlPS_FedWMSAM
(optimizer/WMSAM.py, client/fedwmsam.py, server/FedWMSAM.py).
"""

import math
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from fl_bench.methods.base import StateDict


def wmsam_alignment_score(delta_dict: StateDict, mom_dict: StateDict) -> float:
    """
    Compute `cosine_similarity(delta_i, momentum) + 1.0` (or `0.0` when either norm is 0),
    matching `cosine_similarity` in `server/FedWMSAM.py`.
    """
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for k, a in delta_dict.items():
        b = mom_dict[k]
        a64 = a.to(torch.float64).reshape(-1)
        b64 = b.to(torch.float64).reshape(-1)
        dot += float(torch.dot(a64, b64).item())
        norm_a_sq += float(torch.dot(a64, a64).item())
        norm_b_sq += float(torch.dot(b64, b64).item())
    denom = (norm_a_sq ** 0.5) * (norm_b_sq ** 0.5)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return (dot / denom) + 1.0


def loss_gamma(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    named_trainable: List[Tuple[str, nn.Parameter]],
    client_momentum: Dict[str, torch.Tensor],
    alpha: float,
) -> Tuple[torch.Tensor, float]:
    """
    Compute the FedWMSAM weighted objective (`loss_gamma` in `client/fedwmsam.py`):
        L = alpha * CE(predictions, labels) + (1 - alpha) * sum_p <p, v_i[p]>
    Returns `(total_loss, ce_loss_value)`.
    """
    ce_loss = F.cross_entropy(predictions, labels, reduction="mean")
    ce_val = float(ce_loss.item())
    mom_dot = torch.zeros((), device=predictions.device)
    for n, p in named_trainable:
        mom_dot = mom_dot + torch.sum(p * client_momentum[n])
    total_loss = alpha * ce_loss + (1.0 - alpha) * mom_dot
    return total_loss, ce_val


class WMSAM(torch.optim.Optimizer):
    """
    Weighted Momentum Sharpness-Aware Minimization (`WMSAM`) optimizer
    matching `Huang-Yongzhi/NeurlPS_FedWMSAM/optimizer/WMSAM.py`.
    """

    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        norm_mode: str = "reference_sum_l2",
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid perturbation rate, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = float(rho)
        self.norm_mode = str(norm_mode).strip().lower()
        for group in self.param_groups:
            group["rho"] = rho

    @torch.no_grad()
    def first_step(
        self,
        named_trainable: List[Tuple[str, nn.Parameter]],
        target_model: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perturb model parameters along `differ = target_model - w`:
            e_w = -differ * (rho / (grad_norm + 1e-7))
        """
        if self.rho <= 0.0:
            return {}
        diff_map: Dict[str, torch.Tensor] = {
            n: (target_model[n] - p.detach()) for n, p in named_trainable
        }
        if self.norm_mode == "paper_global_l2":
            sq_sum = sum(
                float(d.to(torch.float64).pow(2).sum().item()) for d in diff_map.values()
            )
            grad_norm = sq_sum ** 0.5
        else:
            grad_norm = sum(float(d.norm(p=2).item()) for d in diff_map.values())

        if grad_norm <= 0.0:
            return {}

        scale = self.rho / (grad_norm + 1e-7)
        e_w_map: Dict[str, torch.Tensor] = {}
        for n, p in named_trainable:
            ew = -diff_map[n] * scale
            p.add_(ew)
            e_w_map[n] = ew
        return e_w_map

    @torch.no_grad()
    def second_step(
        self,
        named_trainable: List[Tuple[str, nn.Parameter]],
        e_w_map: Dict[str, torch.Tensor],
    ) -> None:
        if not e_w_map:
            return
        for n, p in named_trainable:
            if n in e_w_map:
                p.sub_(e_w_map[n])
