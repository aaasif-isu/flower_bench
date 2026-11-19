"""
FedWMSAM: Fast and Flat Federated Learning via Weighted Momentum and Sharpness-Aware Minimization
(Li et al., NeurIPS 2025).
Reference implementation: https://github.com/Huang-Yongzhi/NeurlPS_FedWMSAM
(server/FedWMSAM.py, client/fedwmsam.py, optimizer/WMSAM.py).
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
from fl_bench.methods.fedwmsam.optimizer import WMSAM, loss_gamma, wmsam_alignment_score
from fl_bench.models import create_model
from fl_bench.utils.seed import set_seed


class FedWMSAMMethod(BaseMethod):
    """
    Paper- and reference-faithful FedWMSAM (NeurIPS 2025) implementing:
      - C1 (Momentum-guided global perturbation): `differ = target_model - w`,
        `e_w = -differ * (rho / (||differ|| + 1e-7))`, single backward pass per step.
      - C2 (Personalized momentum): `v_i = m + (alpha / (1 - alpha)) * (c - c_i)`,
        weighted objective `alpha * CE(w + e_w) + (1 - alpha) * <w + e_w, v_i>`,
        and target model advancement `target_model <- target_model + v_i`.
      - C3 (Adaptive schedule): updates `alpha <- 0.99 * alpha + 0.01 * clip(mean_align, 0.1, 0.9)`,
        global control variate `c`, and server momentum `m`.
    """

    name = "fedwmsam"
    extra_csv_fields = (
        "wmsam_alpha",
        "mean_alignment",
        "momentum_norm",
        "c_global_norm",
    )
    cleanup_after_round = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.c_global: StateDict = {}
        self.client_controls: Dict[int, StateDict] = {}
        self.momentum: StateDict = {}
        self.alpha_val: float = 0.5
        self.trainable_bytes: int = 0

    def init_server(self, global_model: nn.Module) -> None:
        self.c_global = {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }
        self.client_controls = {}
        self.momentum = {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in global_model.named_parameters()
            if param.requires_grad
        }
        init_alpha = float(
            getattr(self.config, "wmsam_alpha", 0.5)
            if getattr(self.config, "wmsam_alpha", None) is not None
            else 0.5
        )
        self.alpha_val = min(0.9, max(0.1, init_alpha))
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

        rho = float(getattr(cfg, "rho", 0.05) if getattr(cfg, "rho", None) is not None else 0.05)
        norm_mode = str(getattr(cfg, "wmsam_norm_mode", "reference_sum_l2") or "reference_sum_l2").strip().lower()
        max_norm = float(cfg.max_grad_norm if cfg.max_grad_norm is not None else 10.0)
        mom = float(cfg.momentum if cfg.momentum is not None else 0.0)
        alpha_cur = float(self.alpha_val)

        if cid not in self.client_controls:
            self.client_controls[cid] = {
                name: torch.zeros_like(val) for name, val in self.c_global.items()
            }
        c_local = self.client_controls[cid]

        # C2: Personalized client momentum v_i = m + (alpha / (1 - alpha)) * (c - c_i)
        ratio = alpha_cur / max(1e-6, 1.0 - alpha_cur)
        client_mom_device: Dict[str, torch.Tensor] = {
            name: (
                self.momentum[name].to(torch.float64)
                + ratio * (self.c_global[name].to(torch.float64) - c_local[name].to(torch.float64))
            ).to(global_state[name].dtype).to(device)
            for name in self.c_global
        }

        model = create_model(cfg.model).to(device)
        model.load_state_dict(global_state)

        target_model: Dict[str, torch.Tensor] = {
            name: global_state[name].detach().to(device).clone()
            for name in self.c_global
        }

        loader = DataLoader(
            client_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        lr_cur = self.current_client_lr()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr_cur,
            momentum=mom,
            weight_decay=cfg.weight_decay,
        )
        sam_optimizer = WMSAM(
            model.parameters(),
            optimizer,
            rho=rho,
            norm_mode=norm_mode,
        )
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

                sam_optimizer.zero_grad()

                # C1: Momentum-guided global perturbation (WMSAM.first_step)
                e_w_map = sam_optimizer.first_step(named_trainable, target_model)

                # Single forward & backward pass with loss_gamma:
                # alpha * CE(predictions, labels) + (1 - alpha) * sum(param * client_momentum)
                logits = model(x)
                loss, ce_val = loss_gamma(
                    logits, y, named_trainable, client_mom_device, alpha_cur
                )
                if not math.isfinite(ce_val):
                    raise FloatingPointError(
                        f"FedWMSAM client {cid} encountered nonfinite CE loss ({ce_val}) at step {local_steps + 1}."
                    )
                loss.backward()
                forward_passes += 1
                backward_passes += 1

                # WMSAM.second_step: restore parameters from w + e_w to w
                sam_optimizer.second_step(named_trainable, e_w_map)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

                # Advance target_model += Client_momentum
                with torch.no_grad():
                    for n in target_model:
                        target_model[n].add_(client_mom_device[n])

                local_steps += 1
                batch_n = y.size(0)
                total_loss += ce_val * batch_n
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_seen += batch_n

            if target_steps is not None and local_steps >= target_steps:
                break
        local_state: StateDict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        if local_steps == 0:
            raise RuntimeError(f"FedWMSAM client {cid} performed zero local steps")

        model_delta: StateDict = {}
        c_delta: StateDict = {}
        coef = 1.0 / (float(local_steps) * float(lr_cur))
        with torch.no_grad():
            for name in self.c_global:
                dw64 = local_state[name].to(torch.float64) - global_state[name].to(torch.float64)
                model_delta[name] = dw64.to(local_state[name].dtype)
                # updated_c_i = c_i - c - delta_w / (K_i * lr)  =>  delta_c_i = -c - delta_w / (K_i * lr)
                cdel64 = -self.c_global[name].to(torch.float64) - coef * dw64
                c_delta[name] = cdel64.to(c_local[name].dtype)
                self.client_controls[cid][name] = (
                    c_local[name].to(torch.float64) + cdel64
                ).to(c_local[name].dtype)

        align_val = wmsam_alignment_score(model_delta, self.momentum)

        return ClientResult(
            state=local_state,
            num_examples=total_seen,
            train_loss=total_loss / max(1, total_seen),
            train_accuracy=total_correct / max(1, total_seen),
            local_normalizer=align_val,
            local_steps=local_steps,
            c_delta=c_delta,
            cid=cid,
            forward_passes=forward_passes,
            backward_passes=backward_passes,
            extra_download_bytes=self.trainable_bytes,
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
        assert self.config.lr is not None

        uniform_w = 1.0 / num_selected
        eta_g = float(self.config.server_lr)
        lr_cur = self.current_client_lr()
        lr_decay = float(getattr(self.config, "lr_decay", 1.0) or 1.0)
        mean_steps = sum(r.local_steps for r in client_results) / num_selected

        # C3: Adaptive training schedule update for alpha
        mean_align = sum(r.local_normalizer for r in client_results) / num_selected
        clamped_align = max(0.1, min(0.9, mean_align))
        self.alpha_val = 0.99 * self.alpha_val + 0.01 * clamped_align

        # Update global control variate c <- c + (1 / N) * sum_{i in S_t} delta_c_i
        part_ratio = num_selected / float(self.config.num_clients)
        for name, old_c in self.c_global.items():
            avg_cd = torch.zeros_like(old_c, dtype=torch.float64)
            for r in client_results:
                assert r.c_delta is not None
                avg_cd += r.c_delta[name].to(torch.float64) * uniform_w
            self.c_global[name] = (old_c.to(torch.float64) + part_ratio * avg_cd).to(old_c.dtype)

        new_state: StateDict = {}
        new_momentum: StateDict = {}
        inv_step_lr = -lr_decay / max(1e-12, float(mean_steps) * lr_cur)

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
                updated64 = global_val.to(torch.float64) + eta_g * avg_delta
                new_momentum[key] = (avg_delta * inv_step_lr).to(global_val.dtype)
                updated = updated64.to(global_val.dtype)
            else:
                avg_buf = torch.zeros_like(global_val, dtype=torch.float64)
                for r in client_results:
                    avg_buf += r.state[key].to(torch.float64) * uniform_w
                updated = avg_buf.to(global_val.dtype)

            if not torch.isfinite(updated).all():
                raise FloatingPointError(
                    f"Round {round_id}: Nonfinite global parameter '{key}' in FedWMSAM aggregation."
                )
            new_state[key] = updated

        self.momentum = new_momentum
        mom_norm = tensor_dict_norm(self.momentum)
        c_norm = tensor_dict_norm(self.c_global)

        return new_state, {
            "wmsam_alpha": self.alpha_val,
            "mean_alignment": mean_align,
            "momentum_norm": mom_norm,
            "c_global_norm": c_norm,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"alpha={extra_metrics['wmsam_alpha']:.4f} "
            f"align={extra_metrics['mean_alignment']:.4f} "
            f"mom_norm={extra_metrics['momentum_norm']:.4f}"
        )


FedWMSAMAlgorithm = FedWMSAMMethod
