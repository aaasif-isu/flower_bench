"""
FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy
(Sun et al., ICLR 2023, arXiv:2302.10429).
Reference implementation: https://github.com/woodenchild95/FL-Simulator (server/FedSpeed.py, client/fedspeed.py, optimizer/ESAM.py)
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
from fl_bench.methods.fedspeed.optimizer import ESAM, preserve_batchnorm_stats
from fl_bench.models import create_model
from fl_bench.utils.seed import set_seed


class FedSpeedMethod(BaseMethod):
    """
    Paper- and reference-faithful FedSpeed implementation combining:
      1. Sharpness-aware perturbation step (ESAM, 2 forward/backward passes per step)
         with optional quasi-gradient mixing coefficient `fedspeed_alpha` (default 1.0).
      2. Prox-term weight decay (`weight_decay + lamb`) and local dual correction
         `delta_i = h_i - w_{i,0}` (`loss_correct = lamb * <w, h_i - w_{i,0}>`).
      3. Dual variable update `h_i <- h_i + (w_{i,K} - w_{i,0})` and server global
         update `w^{t+1} = (1/|S_t|) sum_{i in S_t} w_{i,K} + (1/N) sum_{j=1}^N h_j`.
    """

    name = "fedspeed"
    extra_csv_fields = (
        "mean_local_steps",
        "h_mean_norm",
        "mean_update_norm",
    )
    cleanup_after_round = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.h_clients: Dict[int, StateDict] = {}
        self.h_sum: StateDict = {}
        self.prev_client_models: Dict[int, StateDict] = {}
        self.trainable_bytes: int = 0

    def init_server(self, global_model: nn.Module) -> None:
        self.h_clients = {}
        self.prev_client_models = {}
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

        rho = float(getattr(cfg, "rho", 0.1) or 0.1)
        lamb = float(getattr(cfg, "lamb", 0.1) or 0.1)
        sam_alpha = float(getattr(cfg, "fedspeed_alpha", 1.0) if getattr(cfg, "fedspeed_alpha", None) is not None else 1.0)
        use_ri = bool(getattr(cfg, "use_ri", False))
        ri_beta = float(getattr(cfg, "ri_beta", 0.1) or 0.1)
        prox_mode = str(getattr(cfg, "fedspeed_prox_mode", "paper_symmetric") or "paper_symmetric").strip().lower()
        comm_mode = str(getattr(cfg, "fedspeed_comm_mode", "reference_simulator") or "reference_simulator").strip().lower()
        freeze_bn = bool(getattr(cfg, "fedspeed_freeze_bn_on_perturb", True))
        max_norm = float(cfg.max_grad_norm if cfg.max_grad_norm is not None else 10.0)
        mom = float(cfg.momentum if cfg.momentum is not None else 0.0)

        if cid not in self.h_clients:
            self.h_clients[cid] = {
                name: torch.zeros_like(global_state[name].detach().cpu())
                for name in self.h_sum
            }
        h_local = self.h_clients[cid]

        init_state: StateDict = {}
        for k, v in global_state.items():
            if use_ri and cid in self.prev_client_models and k in self.h_sum:
                prev_w = self.prev_client_models[cid][k]
                init_state[k] = (v.detach().cpu() + ri_beta * (v.detach().cpu() - prev_w)).to(v.dtype)
            else:
                init_state[k] = v.detach().cpu().clone()

        model = create_model(cfg.model).to(device)
        model.load_state_dict(init_state)

        delta_device: Dict[str, torch.Tensor] = {
            name: (h_local[name] - init_state[name]).to(device)
            for name in self.h_sum
        }

        loader = DataLoader(
            client_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        sgd_wd = float(cfg.weight_decay) + (lamb if prox_mode == "reference_simulator" else 0.0)
        lr_cur = self.current_client_lr()
        base_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr_cur,
            momentum=mom,
            weight_decay=sgd_wd,
        )
        esam_optimizer = ESAM(model.parameters(), base_optimizer, rho=rho)
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

                # --- Step 1: First forward/backward pass at w_{i,k} (ESAM.first_step) ---
                esam_optimizer.zero_grad()
                logits_1 = model(x)
                loss_1 = criterion(logits_1, y)
                loss_val = loss_1.item()
                if not math.isfinite(loss_val):
                    raise FloatingPointError(
                        f"FedSpeed client {cid} encountered nonfinite loss ({loss_val}) at step {local_steps + 1}."
                    )
                loss_1.backward()
                forward_passes += 1
                backward_passes += 1

                esam_optimizer.first_step(save_clean_grad=(sam_alpha < 1.0))

                # --- Step 2: Second forward/backward pass at w_{i,k} + e_w (ESAM.second_step) ---
                esam_optimizer.zero_grad()
                with preserve_batchnorm_stats(model, enabled=freeze_bn):
                    logits_2 = model(x)
                    loss_2 = sam_alpha * criterion(logits_2, y)
                loss_2.backward()
                forward_passes += 1
                backward_passes += 1

                esam_optimizer.second_step(alpha=sam_alpha)

                with torch.no_grad():
                    for n, p in named_trainable:
                        if p.grad is not None:
                            if prox_mode == "paper_symmetric":
                                p.grad.add_(lamb * (p.detach() + delta_device[n]))
                            else:
                                p.grad.add_(lamb * delta_device[n])

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                base_optimizer.step()

                local_steps += 1
                batch_n = y.size(0)
                total_loss += loss_val * batch_n
                total_correct += (logits_1.argmax(dim=1) == y).sum().item()
                total_seen += batch_n

            if target_steps is not None and local_steps >= target_steps:
                break
        local_state: StateDict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        # Update dual variable h_i <- h_i + (w_{i,K} - w_{i,0})
        update_delta: StateDict = {}
        with torch.no_grad():
            for name in self.h_sum:
                dw64 = local_state[name].to(torch.float64) - init_state[name].to(torch.float64)
                update_delta[name] = dw64.to(local_state[name].dtype)
                self.h_clients[cid][name] = (
                    h_local[name].to(torch.float64) + dw64
                ).to(h_local[name].dtype)
                self.h_sum[name] += dw64

        if use_ri:
            self.prev_client_models[cid] = {
                name: local_state[name].clone() for name in self.h_sum
            }

        extra_down = self.trainable_bytes if comm_mode == "reference_simulator" else 0
        return ClientResult(
            state=local_state,
            num_examples=total_seen,
            train_loss=total_loss / max(1, total_seen),
            train_accuracy=total_correct / max(1, total_seen),
            local_normalizer=float(local_steps),
            local_steps=local_steps,
            c_delta=update_delta,
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
        uniform_w = 1.0 / num_selected
        inv_n_clients = 1.0 / float(self.config.num_clients)

        new_state: StateDict = {}
        for key, global_val in global_state.items():
            if not torch.is_floating_point(global_val):
                new_state[key] = client_results[0].state[key].clone()
                continue

            avg_model = torch.zeros_like(global_val, dtype=torch.float64)
            for r in client_results:
                avg_model += r.state[key].to(torch.float64) * uniform_w

            if key in trainable_keys:
                # ServerOpt: w(t+1) = average_{i in S_t}[w_i(t)] + (1/N) sum_{j=1}^N h_j(t)
                updated = (avg_model + self.h_sum[key] * inv_n_clients).to(global_val.dtype)
            else:
                updated = avg_model.to(global_val.dtype)

            if not torch.isfinite(updated).all():
                raise FloatingPointError(
                    f"Round {round_id}: Nonfinite global parameter '{key}' in FedSpeed aggregation."
                )
            new_state[key] = updated

        mean_steps = sum(r.local_steps for r in client_results) / num_selected
        h_mean_dict = {k: v * inv_n_clients for k, v in self.h_sum.items()}
        h_mean_norm = tensor_dict_norm(h_mean_dict)
        mean_update_norm = (
            sum(tensor_dict_norm(r.c_delta) for r in client_results if r.c_delta is not None)
            / num_selected
        )
        return new_state, {
            "mean_local_steps": mean_steps,
            "h_mean_norm": h_mean_norm,
            "mean_update_norm": mean_update_norm,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"steps={extra_metrics['mean_local_steps']:.1f} "
            f"h_norm={extra_metrics['h_mean_norm']:.4f} "
            f"upd_norm={extra_metrics['mean_update_norm']:.4f}"
        )


FedSpeedAlgorithm = FedSpeedMethod
