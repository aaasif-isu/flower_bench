"""
ESAM optimizer, BatchNorm running-stat preservation, and proximal-dual correction
primitives for FedSpeed (Sun et al., ICLR 2023).
Reference: https://github.com/woodenchild95/FL-Simulator/blob/main/optimizer/ESAM.py
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn


@contextmanager
def preserve_batchnorm_stats(model: nn.Module, enabled: bool = True) -> Iterator[None]:
    """
    Save and restore BatchNorm `running_mean`, `running_var`, and `num_batches_tracked`
    around the SAM perturbed forward pass so adversarial perturbation activations do not
    double-update or skew BatchNorm running statistics on BatchNorm models (e.g. ResNet-18).
    """
    if not enabled:
        yield
        return

    saved: List[Tuple[nn.modules.batchnorm._BatchNorm, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            rm = m.running_mean.detach().clone() if m.running_mean is not None else None
            rv = m.running_var.detach().clone() if m.running_var is not None else None
            nbt = m.num_batches_tracked.detach().clone() if m.num_batches_tracked is not None else None
            saved.append((m, rm, rv, nbt))
    try:
        yield
    finally:
        with torch.no_grad():
            for m, rm, rv, nbt in saved:
                if rm is not None and m.running_mean is not None:
                    m.running_mean.copy_(rm)
                if rv is not None and m.running_var is not None:
                    m.running_var.copy_(rv)
                if nbt is not None and m.num_batches_tracked is not None:
                    m.num_batches_tracked.copy_(nbt)


class ESAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer (`ESAM`) from `woodenchild95/FL-Simulator/optimizer/ESAM.py`,
    supporting both the reference `alpha=1.0` perturbed gradient and the paper's general
    quasi-gradient `(1 - alpha) * g_1 + alpha * g_2` (Sun et al., ICLR 2023, Eq. 4).
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
            group["adaptive"] = adaptive

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        norms = [
            ((torch.abs(p) if group.get("adaptive", False) else 1.0) * p.grad).norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, save_clean_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            adaptive = bool(group.get("adaptive", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if save_clean_grad:
                    self.state[p]["g_clean"] = p.grad.detach().clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self, alpha: float = 1.0) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state:
                    continue
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]
                if alpha < 1.0 and "g_clean" in self.state[p]:
                    p.grad.add_((1.0 - alpha) * self.state[p]["g_clean"])
                    del self.state[p]["g_clean"]
