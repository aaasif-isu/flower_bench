"""
FedLESAM: Locally Estimated Global Perturbations are Better than Local Perturbations
for Federated Sharpness-Aware Minimization (Fan et al., ICML 2024 Spotlight).
Reference implementation: https://github.com/MediaBrain-SJTU/FedLESAM
(server/FedLESAM.py, server/FedLESAM_S.py, server/FedLESAM_D.py, optimizer/LESAM.py).
"""

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    tensor_dict_norm,
)
from fl_bench.methods.fedlesam.optimizer import LESAM, compute_lesam_perturbation
from fl_bench.models import create_model
from fl_bench.utils.seed import set_seed


class FedLESAMMethod(BaseMethod):
    """
    Paper- and reference-faithful FedLESAM supporting all three official variants
    via `config.fedlesam_variant`:
      - `"base"` (default): FedLESAM with single-backprop global-update perturbation
      - `"s"`: FedLESAM-S (FedLESAM + SCAFFOLD control-variate correction)
      - `"d"`: FedLESAM-D (FedLESAM + FedDyn dual variable correction)
    """

    name = "fedlesam"
    extra_csv_fields = (
        "mean_local_steps",
        "global_update_norm",
        "perturbation_norm",
    )
    cleanup_after_round = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.global_update: Optional[StateDict] = None
        self.client_prev_global: Dict[int, StateDict] = {}
        self.c_global: StateDict = {}
        self.client_controls: Dict[int, StateDict] = {}
        self.h_clients: Dict[int, StateDict] = {}
        self.h_sum: StateDict = {}
        self.trainable_bytes: int = 0

    def init_server(self, global_model: nn.Module) -> None:
        self.global_update = None
        self.client_prev_global = {}
        self.c_global = {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }
        self.client_controls = {}
        self.h_clients = {}
        self.h_sum = {
            name: torch.zeros_like(param.detach().cpu(), dtype=torch.float64)
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }
        self.trainable_bytes = sum(
            int(p.numel()) * int(p.element_size())
            for p in global_model.parameters()
            if p.requires_grad
        )

    def run_client(
        self,
        cid: int,
        global_state: StateDict,
        client_dataset: Dataset,
        device: torch.device,
        seed: int,
    ) -> ClientResult:
        set_seed(seed)

        cfg = self.config
        assert cfg.model is not None
        assert cfg.batch_size is not None
        assert cfg.local_epochs is not None
        assert cfg.lr is not None
        assert cfg.weight_decay is not None

        variant = str(getattr(cfg, "fedlesam_variant", "base") or "base").strip().lower()
        norm_mode = str(getattr(cfg, "fedlesam_norm_mode", "reference_sum_l2") or "reference_sum_l2").strip().lower()
        pert_target = str(
            getattr(cfg, "fedlesam_perturbation_target", "server_global_update") or "server_global_update"
        ).strip().lower()
        prox_mode = str(getattr(cfg, "fedlesam_prox_mode", "reference_simulator") or "reference_simulator").strip().lower()
        rho = float(getattr(cfg, "rho", 0.1) or 0.1)
        lamb = float(getattr(cfg, "lamb", 0.1) or 0.1)
        max_norm = float(cfg.max_grad_norm if cfg.max_grad_norm is not None else 10.0)
        mom = float(cfg.momentum if cfg.momentum is not None else 0.0)

        model = create_model(cfg.model).to(device)
        model.load_state_dict(global_state)

        wd = float(cfg.weight_decay) + (
            lamb if (variant == "d" and prox_mode == "reference_simulator") else 0.0
        )
        lr_cur = self.current_client_lr()
        base_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr_cur,
            momentum=mom,
            weight_decay=wd,
        )
        lesam_optimizer = LESAM(model.parameters(), base_optimizer, rho=rho)

        # Prepare variant-specific correction vector (`Local_VR_correction` or `Local_dual_correction`)
        correction_device: Optional[Dict[str, torch.Tensor]] = None
        if variant == "s":
            if cid not in self.client_controls:
                self.client_controls[cid] = {
                    name: torch.zeros_like(val) for name, val in self.c_global.items()
                }
            c_local = self.client_controls[cid]
            correction_device = {
                name: (self.c_global[name] - c_local[name]).to(device)
                for name in self.c_global
            }
        elif variant == "d":
            if cid not in self.h_clients:
                self.h_clients[cid] = {
                    name: torch.zeros_like(val) for name, val in self.c_global.items()
                }
            h_local = self.h_clients[cid]
            correction_device = {
                name: (h_local[name] - global_state[name].detach().cpu()).to(device)
                for name in self.c_global
            }

        # Determine global update direction:
        # - "server_global_update" (default, matches MediaBrain-SJTU/FedLESAM/server/FedLESAM.py):
        #   uses server-transmitted `global_update = w^t - w^{t-1}`.
        # - "client_prev_global" (matches ICML 2024 paper Algorithm 1):
        #   client computes `w^t - w^{t_i^prev}` from its cached global model at previous active round.
        effective_g_update: Optional[StateDict] = None
        if pert_target == "client_prev_global":
            if cid in self.client_prev_global:
                prev_g = self.client_prev_global[cid]
                effective_g_update = {
                    name: (global_state[name].detach().cpu() - prev_g[name])
                    for name in self.c_global
                }
            self.client_prev_global[cid] = {
                name: global_state[name].detach().cpu().clone()
                for name in self.c_global
            }
        else:
            effective_g_update = self.global_update

        e_w_device, pert_norm = compute_lesam_perturbation(
            effective_g_update,
            self.c_global.keys(),
            rho=rho,
            device=device,
            norm_mode=norm_mode,
        )
        loader = DataLoader(
            client_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        criterion = nn.CrossEntropyLoss()
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        local_steps = 0
        forward_passes = 0
        backward_passes = 0

        target_steps = int(cfg.local_steps) if getattr(cfg, "local_steps", None) is not None else None
        max_passes = (
            max(1, math.ceil(target_steps / max(1, len(loader))))
            if target_steps is not None
            else int(cfg.local_epochs)
        )

        named_trainable: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]

        for _ in range(max_passes):
            for x, y in loader:
                if target_steps is not None and local_steps >= target_steps:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                lesam_optimizer.zero_grad()

                # LESAM first_step: perturb parameters by e_w = -g_update * scale
                lesam_optimizer.first_step(named_trainable, e_w_device)

                # Single forward & backward pass at w + e_w
                logits = model(x)
                ce_loss = criterion(logits, y)
                loss_val = ce_loss.item()
                if not math.isfinite(loss_val):
                    raise FloatingPointError(
                        f"FedLESAM client {cid} encountered nonfinite loss ({loss_val}) at step {local_steps + 1}."
                    )

                if correction_device is not None:
                    reg_term = torch.zeros((), device=device)
                    for n, p in named_trainable:
                        reg_term = reg_term + torch.sum(p * correction_device[n])
                    loss = ce_loss + lamb * reg_term
                else:
                    loss = ce_loss

                loss.backward()
                forward_passes += 1
                backward_passes += 1

                # LESAM second_step: restore parameters to w while retaining gradients
                lesam_optimizer.second_step(named_trainable, e_w_device)

                if variant == "d" and prox_mode == "paper_symmetric":
                    with torch.no_grad():
                        for n, p in named_trainable:
                            if p.grad is not None:
                                p.grad.add_(lamb * p.detach())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                base_optimizer.step()

                local_steps += 1
                batch_n = y.size(0)
                total_loss += loss_val * batch_n
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_seen += batch_n

            if target_steps is not None and local_steps >= target_steps:
                break
        local_state: StateDict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        c_delta: Optional[StateDict] = None
        if variant == "s":
            if local_steps == 0:
                raise RuntimeError(f"FedLESAM-S client {cid} performed zero local steps")
            coef = 1.0 / (float(local_steps) * float(lr_cur))
            c_delta = {}
            c_local = self.client_controls[cid]
            with torch.no_grad():
                for name in self.c_global:
                    dw64 = local_state[name].to(torch.float64) - global_state[name].to(torch.float64)
                    cdel64 = -self.c_global[name].to(torch.float64) - coef * dw64
                    c_delta[name] = cdel64.to(c_local[name].dtype)
                    self.client_controls[cid][name] = (
                        c_local[name].to(torch.float64) + cdel64
                    ).to(c_local[name].dtype)
        elif variant == "d":
            c_delta = {}
            h_local = self.h_clients[cid]
            with torch.no_grad():
                for name in self.c_global:
                    dw64 = local_state[name].to(torch.float64) - global_state[name].to(torch.float64)
                    c_delta[name] = dw64.to(h_local[name].dtype)
                    self.h_clients[cid][name] = (
                        h_local[name].to(torch.float64) + dw64
                    ).to(h_local[name].dtype)
                    self.h_sum[name] += dw64

        extra_down = (
            (self.trainable_bytes if (pert_target == "server_global_update" and self.global_update is not None) else 0)
            + (self.trainable_bytes if variant in ("s", "d") else 0)
        )
        return ClientResult(
            state=local_state,
            num_examples=total_seen,
            train_loss=total_loss / max(1, total_seen),
            train_accuracy=total_correct / max(1, total_seen),
            local_normalizer=pert_norm,
            local_steps=local_steps,
            c_delta=c_delta,
            cid=cid,
            forward_passes=forward_passes,
            backward_passes=backward_passes,
            extra_download_bytes=extra_down,
            extra_upload_bytes=0,
        )

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
        assert self.config.server_lr is not None

        variant = str(getattr(self.config, "fedlesam_variant", "base") or "base").strip().lower()
        uniform_w = 1.0 / num_selected
        inv_n_clients = 1.0 / float(self.config.num_clients)
        eta_g = float(self.config.server_lr)

        new_state: StateDict = {}
        new_global_update: StateDict = {}

        for key, global_val in global_state.items():
            if not torch.is_floating_point(global_val):
                new_state[key] = client_results[0].state[key].clone()
                continue

            if key in trainable_keys:
                avg_delta = torch.zeros_like(global_val, dtype=torch.float64)
                for r in client_results:
                    avg_delta += (
                        r.state[key].to(torch.float64) - global_val.to(torch.float64)
                    ) * uniform_w

                if variant == "d":
                    h_mean = self.h_sum[key] * inv_n_clients
                    g_upd64 = avg_delta + h_mean
                    updated64 = global_val.to(torch.float64) + g_upd64
                else:
                    g_upd64 = eta_g * avg_delta
                    updated64 = global_val.to(torch.float64) + g_upd64

                new_global_update[key] = g_upd64.to(global_val.dtype)
                updated = updated64.to(global_val.dtype)
            else:
                avg_buf = torch.zeros_like(global_val, dtype=torch.float64)
                for r in client_results:
                    avg_buf += r.state[key].to(torch.float64) * uniform_w
                updated = avg_buf.to(global_val.dtype)

            if not torch.isfinite(updated).all():
                raise FloatingPointError(
                    f"Round {round_id}: Nonfinite global parameter '{key}' in FedLESAM aggregation."
                )
            new_state[key] = updated

        if variant == "s":
            part_ratio = num_selected / float(self.config.num_clients)
            for name, old_c in self.c_global.items():
                avg_cd = torch.zeros_like(old_c, dtype=torch.float64)
                for r in client_results:
                    assert r.c_delta is not None
                    avg_cd += r.c_delta[name].to(torch.float64) * uniform_w
                self.c_global[name] = (old_c.to(torch.float64) + part_ratio * avg_cd).to(old_c.dtype)

        self.global_update = new_global_update
        mean_steps = sum(r.local_steps for r in client_results) / num_selected
        g_upd_norm = tensor_dict_norm(self.global_update)
        pert_norm = sum(r.local_normalizer for r in client_results) / num_selected

        return new_state, {
            "mean_local_steps": mean_steps,
            "global_update_norm": g_upd_norm,
            "perturbation_norm": pert_norm,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"steps={extra_metrics['mean_local_steps']:.1f} "
            f"g_upd_norm={extra_metrics['global_update_norm']:.4f} "
            f"pert_norm={extra_metrics['perturbation_norm']:.4f}"
        )


FedLESAMAlgorithm = FedLESAMMethod
