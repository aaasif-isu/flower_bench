"""
FedGPS: Statistical Rectification Against Data Heterogeneity in Federated Learning
(Yang et al., NeurIPS 2025).
Reference implementation: https://github.com/visitworld123/FedGPS
(FedGPS/FedGPSClient.py, FedGPS/FedGPSServer.py, FedGPS/utils.py, utils/optim.py, conf/algorithms/fedgps.yaml).
"""

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from fl_bench.methods.base import (
    BaseMethod,
    ClientResult,
    StateDict,
    tensor_dict_norm,
)
from fl_bench.methods.fedgps.losses import (
    ProxyAlignLoss,
    TripletLoss,
    local_proto_agg,
    proto_aggregation,
)
from fl_bench.methods.fedgps.optimizer import (
    FedGPSOptimizer,
    compute_fedgps_perturbation,
)
from fl_bench.methods.fedgps.surrogate import (
    FedGPSFeatureCatcher,
    build_fedgps_model,
    generate_surrogate_batch,
)
from fl_bench.utils.seed import set_seed
class FedGPSMethod(BaseMethod):
    """
    Paper- and reference-faithful FedGPS (Goal-Path Synergy, Yang et al., NeurIPS 2025):
      1. Shared `(C + C_surrogate)`-class Classifier Head (`create_global_model`):
         Expands the classification head to `C + C_surrogate` (`2C` when `C_surrogate == C`)
         so real classes `0 .. C-1` and shifted surrogate classes `C .. C + C_surrogate - 1`
         are trained jointly under a single softmax classifier without class-index collision.
      2. Path Rectification: Perturbs local model parameters along the non-self global update
         direction `g_other = (M * global_update - delta_w_i^{t-1}) / (M - 1)` when client `i`
         participated in the previous round (or `global_update` otherwise), gated on prior
         client participation (`len(self.model_delta) > 0` in `FedGPSClient.py:232`).
      3. Goal Rectification: Modifies the local learning objective using surrogate statistical
         data (`loss_aux`), supervised contrastive alignment (`ProxyAlignLoss`), and global
         surrogate prototype alignment (`proto_aggregation` + MSE/Triplet prototype loss).
    """

    name = "fedgps"
    extra_csv_fields = (
        "mean_local_steps",
        "global_update_norm",
        "num_global_protos",
        "mean_proto_loss",
    )
    cleanup_after_round = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.global_update: Optional[StateDict] = None
        self.global_ema_update: Optional[StateDict] = None
        self.global_noise_protos: Dict[int, torch.Tensor] = {}
        self.prev_selected_clients: Set[int] = set()
        self.current_round_clients: Set[int] = set()
        self.client_prev_deltas: Dict[int, StateDict] = {}
        self.trainable_bytes: int = 0
        self.trainable_names: Set[str] = set()
        self.real_num_classes: int = 10
        self.surrogate_num: int = 10
        self.feature_dim: int = 512

    @classmethod
    def validate_compatibility(cls, config) -> None:
        from pathlib import Path

        mode = str(getattr(config, "fedgps_surrogate_mode", "gaussian") or "gaussian").strip().lower()
        if mode not in ("gaussian", "dataset", "none"):
            raise ValueError(
                f"Invalid fedgps_surrogate_mode '{mode}'. Expected 'gaussian', 'dataset', or 'none'."
            )
        if mode != "none" and str(config.model or "").strip().lower() == "shakespeare_lstm":
            raise ValueError(
                "Method 'fedgps' with surrogate data enabled (`fedgps_surrogate_mode != 'none'`) "
                "requires continuous floating-point inputs and is incompatible with discrete token "
                "sequence model 'shakespeare_lstm'. Set `fedgps_surrogate_mode: none` and "
                "`fedgps_label_shift: false` for Shakespeare."
            )
        if mode == "dataset":
            s_path = getattr(config, "fedgps_surrogate_path", None)
            if not s_path or not Path(str(s_path)).exists():
                raise ValueError(
                    f"Method 'fedgps' configured with `fedgps_surrogate_mode='dataset'`, but "
                    f"`fedgps_surrogate_path` ('{s_path}') does not exist."
                )
        use_minus = bool(getattr(config, "fedgps_use_minus", True))
        if use_minus and int(config.clients_per_round or 0) < 2:
            raise ValueError(
                f"Method 'fedgps' with `fedgps_use_minus=True` requires `clients_per_round >= 2` "
                f"for leave-one-out non-self global update computation, got {config.clients_per_round}."
            )

    def _surrogate_settings(self) -> Tuple[bool, bool, Optional[int], str]:
        cfg = self.config
        mode = str(getattr(cfg, "fedgps_surrogate_mode", "gaussian") or "gaussian").strip().lower()
        surr_enabled = mode != "none"
        label_shift = bool(
            getattr(cfg, "fedgps_label_shift", True)
            if getattr(cfg, "fedgps_label_shift", None) is not None
            else True
        )
        surr_num_raw = getattr(cfg, "fedgps_surrogate_num", None)
        surr_num = int(surr_num_raw) if (surr_num_raw is not None and int(surr_num_raw) > 0) else None
        return surr_enabled, label_shift, surr_num, mode

    def create_global_model(self, device: torch.device) -> nn.Module:
        assert self.config.model is not None
        surr_enabled, label_shift, surr_num, _ = self._surrogate_settings()
        model, real_c, eff_surr, feat_d = build_fedgps_model(
            self.config.model,
            surrogate_enabled=surr_enabled,
            surrogate_label_shift=label_shift,
            surrogate_num=surr_num,
            seed=int(self.config.training_seed),
        )
        self.real_num_classes = real_c
        self.surrogate_num = eff_surr
        self.feature_dim = feat_d
        return model.to(device)
    def init_server(self, global_model: nn.Module) -> None:
        self.global_update = None
        self.global_ema_update = None
        self.global_noise_protos = {}
        self.prev_selected_clients = set()
        self.current_round_clients = set()
        self.client_prev_deltas = {}
        self.trainable_names = {
            name for name, param in global_model.named_parameters() if param.requires_grad
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
        self.current_round_clients.add(cid)

        cfg = self.config
        assert cfg.model is not None
        assert cfg.batch_size is not None
        assert cfg.local_epochs is not None
        assert cfg.lr is not None
        assert cfg.weight_decay is not None
        assert cfg.clients_per_round is not None

        surr_enabled, label_shift, surr_num, surrogate_mode = self._surrogate_settings()
        rho = float(getattr(cfg, "rho", 0.5) if getattr(cfg, "rho", None) is not None else 0.5)
        max_norm = float(cfg.max_grad_norm if cfg.max_grad_norm is not None else 10.0)
        mom = float(cfg.momentum if cfg.momentum is not None else 0.9)
        use_ema = bool(getattr(cfg, "fedgps_use_ema", False))
        use_minus = bool(getattr(cfg, "fedgps_use_minus", True))
        require_prev_delta = bool(
            getattr(cfg, "fedgps_require_prev_delta", True)
            if getattr(cfg, "fedgps_require_prev_delta", None) is not None
            else True
        )
        use_minus_lesam = bool(getattr(cfg, "fedgps_use_minus_lesam", False))
        surrogate_path = getattr(cfg, "fedgps_surrogate_path", None)
        surrogate_batch_size = getattr(cfg, "fedgps_surrogate_batch_size", None)
        surrogate_alpha = float(getattr(cfg, "fedgps_surrogate_alpha", 1.0) or 1.0)
        align_loss_weight = float(getattr(cfg, "fedgps_align_loss_weight", 0.1) or 0.1)
        proto_align_method = str(getattr(cfg, "fedgps_proto_align_method", "mse") or "mse").strip().lower()

        model, real_classes, eff_surr_num, _ = build_fedgps_model(
            cfg.model,
            surrogate_enabled=surr_enabled,
            surrogate_label_shift=label_shift,
            surrogate_num=surr_num,
            seed=seed,
        )
        model = model.to(device)
        model.load_state_dict(global_state)

        lr_cur = self.current_client_lr()
        base_opt = torch.optim.SGD(
            model.parameters(),
            lr=lr_cur,
            momentum=mom,
            weight_decay=cfg.weight_decay,
        )
        gps_opt = FedGPSOptimizer(model.parameters(), base_optimizer=base_opt, rho=rho)

        pre_flag = cid in self.prev_selected_clients
        source_update = (
            self.global_ema_update
            if (use_ema and self.global_ema_update is not None)
            else self.global_update
        )
        e_w_device = compute_fedgps_perturbation(
            source_update=source_update,
            client_prev_delta=self.client_prev_deltas.get(cid),
            trainable_names=self.trainable_names,
            clients_per_round=int(cfg.clients_per_round),
            pre_flag=pre_flag,
            pre_selected_client_use_minus=use_minus,
            require_prev_delta=require_prev_delta,
            use_minus_lesam=use_minus_lesam,
            rho=rho,
            device=device,
        )
        loader = DataLoader(
            client_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        proxy_align = ProxyAlignLoss(
            inter_domain_mapping=False,
            inter_domain_class_match=True,
            noise_feat_detach=True,
            noise_contrastive=True,
            inter_domain_weight=0.0,
            inter_class_weight=1.0,
            noise_supcon_weight=0.1,
            noise_label_shift=(real_classes if label_shift else 0),
        ).to(device)
        triplet_loss_fn = TripletLoss(margin=1.0)
        catcher = FedGPSFeatureCatcher(model)
        model.train()

        total_loss = 0.0
        total_proto_loss = 0.0
        total_correct = 0
        total_seen = 0
        local_steps = 0
        forward_passes = 0
        backward_passes = 0
        agg_noise_protos_label: Dict[int, List[torch.Tensor]] = {}

        target_steps = int(cfg.local_steps) if getattr(cfg, "local_steps", None) is not None else None
        max_passes = (
            max(1, math.ceil(target_steps / max(1, len(loader))))
            if target_steps is not None
            else int(cfg.local_epochs)
        )

        named_trainable: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]

        try:
            for _ in range(max_passes):
                for x, y in loader:
                    if target_steps is not None and local_steps >= target_steps:
                        break
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    real_bs = int(x.size(0))

                    gps_opt.zero_grad()
                    gps_opt.first_step(named_trainable, e_w_device)

                    if surr_enabled:
                        aux_x, sampled_label = generate_surrogate_batch(
                            x,
                            real_num_classes=real_classes,
                            surrogate_num=eff_surr_num,
                            surrogate_label_shift=label_shift,
                            surrogate_mode=surrogate_mode,
                            surrogate_path=surrogate_path,
                            surrogate_batch_size=surrogate_batch_size,
                            step_seed=seed + local_steps * 131,
                        )
                        x_cat = torch.cat((x, aux_x), dim=0)
                        outputs = model(x_cat)
                        feat = catcher.features
                        assert feat is not None

                        ce_loss = F.cross_entropy(outputs[:real_bs], y)
                        ce_val = ce_loss.item()
                        if not math.isfinite(ce_val):
                            raise FloatingPointError(
                                f"FedGPS client {cid} encountered nonfinite CE loss ({ce_val}) at step {local_steps + 1}."
                            )
                        loss_aux = F.cross_entropy(outputs[real_bs:], sampled_label)
                        align_loss, _, _, _ = proxy_align(
                            feat, torch.cat([y, sampled_label], dim=0), real_bs
                        )
                        loss = ce_loss + surrogate_alpha * loss_aux + align_loss
                        proto_val = 0.0
                        if self.global_noise_protos:
                            noise_feat_dev = feat[real_bs:]
                            if proto_align_method == "triplet":
                                pos_feat = torch.zeros_like(noise_feat_dev)
                                neg_feat = torch.zeros_like(noise_feat_dev)
                                valid_mask = []
                                for idx_s, s_lbl in enumerate(sampled_label):
                                    lbl_int = int(s_lbl.item())
                                    neg_cls = (
                                        (lbl_int - real_classes + int(torch.randint(1, max(2, eff_surr_num), (1,)).item()))
                                        % eff_surr_num
                                    ) + (real_classes if label_shift else 0)
                                    if lbl_int in self.global_noise_protos and neg_cls in self.global_noise_protos:
                                        pos_feat[idx_s] = self.global_noise_protos[lbl_int].to(device)
                                        neg_feat[idx_s] = self.global_noise_protos[neg_cls].to(device)
                                        valid_mask.append(idx_s)
                                if valid_mask:
                                    v_idx = torch.tensor(valid_mask, dtype=torch.long, device=device)
                                    p_loss = triplet_loss_fn(
                                        noise_feat_dev[v_idx], pos_feat[v_idx], neg_feat[v_idx]
                                    )
                                    proto_val = float(p_loss.item())
                                    loss = loss + align_loss_weight * p_loss
                            else:
                                proto_target = torch.zeros_like(noise_feat_dev)
                                valid_mask = []
                                for idx_s, s_lbl in enumerate(sampled_label):
                                    lbl_int = int(s_lbl.item())
                                    if lbl_int in self.global_noise_protos:
                                        proto_target[idx_s] = self.global_noise_protos[lbl_int].to(device)
                                        valid_mask.append(idx_s)
                                if valid_mask:
                                    v_idx = torch.tensor(valid_mask, dtype=torch.long, device=device)
                                    p_loss = F.mse_loss(proto_target[v_idx], noise_feat_dev[v_idx])
                                    proto_val = float(p_loss.item())
                                    loss = loss + align_loss_weight * p_loss

                        loss.backward()
                        # 1 forward pass and 1 backward pass per step on concatenated batch [x, aux_x]
                        forward_passes += 1
                        backward_passes += 1

                        noise_feat_cpu = feat[real_bs:].detach().cpu()
                        for i_s in range(int(sampled_label.size(0))):
                            lbl_int = int(sampled_label[i_s].item())
                            agg_noise_protos_label.setdefault(lbl_int, []).append(noise_feat_cpu[i_s])
                        logits_real = outputs[:real_bs]
                    else:
                        logits_real = model(x)
                        ce_loss = F.cross_entropy(logits_real, y)
                        ce_val = ce_loss.item()
                        if not math.isfinite(ce_val):
                            raise FloatingPointError(
                                f"FedGPS client {cid} encountered nonfinite CE loss ({ce_val}) at step {local_steps + 1}."
                            )
                        proto_val = 0.0
                        ce_loss.backward()
                        forward_passes += 1
                        backward_passes += 1

                    gps_opt.second_step(named_trainable, e_w_device)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    base_opt.step()

                    local_steps += 1
                    total_loss += ce_val * real_bs
                    total_proto_loss += proto_val * real_bs
                    total_correct += (logits_real.argmax(dim=1) == y).sum().item()
                    total_seen += real_bs

                if target_steps is not None and local_steps >= target_steps:
                    break
        finally:
            catcher.close()

        local_state: StateDict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        client_delta: StateDict = {}
        with torch.no_grad():
            for name in self.trainable_names:
                client_delta[name] = (
                    local_state[name].to(torch.float64) - global_state[name].to(torch.float64)
                ).to(local_state[name].dtype)
        self.client_prev_deltas[cid] = client_delta

        local_protos, proto_weights = local_proto_agg(agg_noise_protos_label)
        down_proto_bytes = sum(
            int(p.numel()) * int(p.element_size()) for p in self.global_noise_protos.values()
        )
        extra_down = (
            (self.trainable_bytes if self.global_update is not None else 0)
            + (self.trainable_bytes if (use_ema and self.global_ema_update is not None) else 0)
            + down_proto_bytes
        )
        extra_up = sum(
            int(p.numel()) * int(p.element_size()) + 8 for p in local_protos.values()
        )

        return ClientResult(
            state=local_state,
            num_examples=total_seen,
            train_loss=total_loss / max(1, total_seen),
            train_accuracy=total_correct / max(1, total_seen),
            local_normalizer=total_proto_loss / max(1, total_seen),
            local_steps=local_steps,
            c_delta=client_delta,
            cid=cid,
            forward_passes=forward_passes,
            backward_passes=backward_passes,
            extra_download_bytes=extra_down,
            extra_upload_bytes=extra_up,
            aux_payload={
                "local_noise_protos": local_protos,
                "proto_agg_weights": proto_weights,
            },
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
        assert self.config.server_lr is not None

        weighted_sum = bool(getattr(self.config, "fedgps_weighted_sum", False))
        ema_decay = float(getattr(self.config, "fedgps_ema_decay", 0.9) or 0.9)
        eta_g = float(self.config.server_lr)

        if weighted_sum:
            tot_ex = float(sum(r.num_examples for r in client_results))
            weights = [float(r.num_examples) / max(1.0, tot_ex) for r in client_results]
        else:
            weights = [1.0 / float(num_selected) for _ in client_results]

        c_protos = [
            r.aux_payload["local_noise_protos"]
            for r in client_results
            if r.aux_payload and "local_noise_protos" in r.aux_payload
        ]
        c_pweights = [
            r.aux_payload["proto_agg_weights"]
            for r in client_results
            if r.aux_payload and "proto_agg_weights" in r.aux_payload
        ]
        if c_protos:
            self.global_noise_protos = proto_aggregation(c_protos, c_pweights)

        new_state: StateDict = {}
        new_global_update: StateDict = {}

        for key, global_val in global_state.items():
            if not torch.is_floating_point(global_val):
                new_state[key] = client_results[0].state[key].clone()
                continue

            avg_delta = torch.zeros_like(global_val, dtype=torch.float64)
            for r, w_i in zip(client_results, weights):
                avg_delta += (
                    r.state[key].to(torch.float64) - global_val.to(torch.float64)
                ) * w_i

            if key in trainable_keys:
                scaled_delta = eta_g * avg_delta
                new_global_update[key] = scaled_delta.to(global_val.dtype)
                updated = (global_val.to(torch.float64) + scaled_delta).to(global_val.dtype)
            else:
                updated = (global_val.to(torch.float64) + avg_delta).to(global_val.dtype)

            if not torch.isfinite(updated).all():
                raise FloatingPointError(
                    f"Round {round_id}: Nonfinite global parameter '{key}' in FedGPS aggregation."
                )
            new_state[key] = updated

        self.global_update = new_global_update
        if self.global_ema_update is None:
            self.global_ema_update = {
                k: v.clone() for k, v in new_global_update.items()
            }
        else:
            for k, v in new_global_update.items():
                self.global_ema_update[k] = (
                    ema_decay * self.global_ema_update[k].to(torch.float64)
                    + (1.0 - ema_decay) * v.to(torch.float64)
                ).to(v.dtype)

        self.prev_selected_clients = set(self.current_round_clients)
        self.current_round_clients = set()

        mean_steps = sum(r.local_steps for r in client_results) / num_selected
        g_upd_norm = tensor_dict_norm(self.global_update)
        mean_proto_loss = sum(r.local_normalizer for r in client_results) / num_selected

        return new_state, {
            "mean_local_steps": mean_steps,
            "global_update_norm": g_upd_norm,
            "num_global_protos": float(len(self.global_noise_protos)),
            "mean_proto_loss": mean_proto_loss,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"steps={extra_metrics['mean_local_steps']:.1f} "
            f"g_upd_norm={extra_metrics['global_update_norm']:.4f} "
            f"protos={int(extra_metrics['num_global_protos'])} "
            f"proto_loss={extra_metrics['mean_proto_loss']:.4f}"
        )


FedGPSAlgorithm = FedGPSMethod
