"""
HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients
(Diao, Ding, and Tarokh, ICLR 2021, https://openreview.net/pdf?id=TNkPBBYFkXg).
Official reference: https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients
"""

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from fl_bench.methods.base import BaseMethod, ClientResult, StateDict
from fl_bench.methods.heterofl.capacity import (
    assign_client_model_rates,
    parse_model_mode,
)
from fl_bench.methods.heterofl.models import (
    calibrate_static_bn,
    create_heterofl_model,
)
from fl_bench.methods.heterofl.slicing import (
    ParamIndex,
    combine_subnetwork_states,
    compute_subnetwork_indices,
    extract_subnetwork_state,
)
from fl_bench.utils.seed import set_seed


def _extract_client_label_split(client_dataset: Dataset) -> torch.Tensor:
    """Return sorted unique class labels present in `client_dataset`."""
    labels_set: Set[int] = set()
    for i in range(len(client_dataset)):
        _, y = client_dataset[i]
        lbl = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
        labels_set.add(lbl)
    return torch.tensor(sorted(labels_set), dtype=torch.long)


class HeteroFLMethod(BaseMethod):
    """
    Paper- and reference-faithful HeteroFL (Diao et al., ICLR 2021):
      1. Width-scaled subnetworks (`a=1.0, b=0.5, c=0.25, d=0.125, e=0.0625` or custom tiers)
         assigned via fixed (`fix`) or per-round (`dynamic`) client capacity profiles.
      2. Activation scaling (`Scaler(rate)`) and Static Batch Normalization (`sBN`:
         `track_running_stats=False` during local training + `calibrate_static_bn` before global eval).
      3. Optional masked cross-entropy (`heterofl_mask=True`) over each client's local label set.
      4. Parameter-aligned overlap server aggregation (`combine_subnetwork_states`) normalizing
         each tensor coordinate `(i, j)` by the count of active clients whose subnetwork covers `(i, j)`.
    """

    name = "heterofl"
    extra_csv_fields = (
        "mean_model_rate",
        "mean_subnet_params",
        "mean_subnet_param_ratio",
        "mean_local_steps",
    )
    cleanup_after_round = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.global_model_rate: float = 1.0
        self.client_model_rates: List[float] = []
        self.client_label_splits: Dict[int, torch.Tensor] = {}
        self.client_datasets_cache: Dict[int, Dataset] = {}
        self.global_model_ref: Optional[nn.Module] = None
        self.global_total_params: int = 1
        self._last_dynamic_round: int = -1

    @classmethod
    def validate_compatibility(cls, config) -> None:
        from fl_bench.methods.heterofl.models import SUPPORTED_HETEROF_MODELS

        model_name = str(config.model or "").strip().lower()
        if model_name not in SUPPORTED_HETEROF_MODELS:
            raise ValueError(
                f"Method 'heterofl' does not support model '{config.model}' (dataset='{config.dataset}'). "
                f"Supported width-scalable models: {list(SUPPORTED_HETEROF_MODELS)}. "
                "Note: 'shakespeare_lstm' uses a fused 4-gate PyTorch nn.LSTM recurrent weight layout "
                "([4 * hidden_size, ...]) that cannot be top-left prefix-sliced by HeteroFL."
            )
        split_mode = str(getattr(config, "heterofl_split_mode", "fix") or "fix").strip().lower()
        if split_mode not in ("fix", "dynamic"):
            raise ValueError(
                f"Invalid heterofl_split_mode '{split_mode}'. Expected 'fix' or 'dynamic'."
            )
        model_mode = str(
            getattr(config, "heterofl_model_mode", "a2-b2-c2-d2-e2") or "a2-b2-c2-d2-e2"
        ).strip()
        parse_model_mode(
            model_mode,
            custom_split_rates=getattr(config, "heterofl_custom_rates", None),
        )

    def _heterofl_settings(self) -> Tuple[str, str, bool, bool, bool]:
        cfg = self.config
        split_mode = str(getattr(cfg, "heterofl_split_mode", "fix") or "fix").strip().lower()
        model_mode = str(
            getattr(cfg, "heterofl_model_mode", "a2-b2-c2-d2-e2") or "a2-b2-c2-d2-e2"
        ).strip()
        use_scaler = bool(
            getattr(cfg, "heterofl_scaler", True)
            if getattr(cfg, "heterofl_scaler", None) is not None
            else True
        )
        use_mask = bool(
            getattr(cfg, "heterofl_mask", True)
            if getattr(cfg, "heterofl_mask", None) is not None
            else True
        )
        use_sbn = bool(
            getattr(cfg, "heterofl_sbn", True)
            if getattr(cfg, "heterofl_sbn", None) is not None
            else True
        )
        return split_mode, model_mode, use_scaler, use_mask, use_sbn

    def create_global_model(self, device: torch.device) -> nn.Module:
        assert self.config.model is not None
        _, model_mode, use_scaler, _, _ = self._heterofl_settings()
        global_rate, _, _, _ = parse_model_mode(
            model_mode,
            custom_split_rates=getattr(self.config, "heterofl_custom_rates", None),
        )
        self.global_model_rate = global_rate
        model = create_heterofl_model(
            self.config.model,
            model_rate=global_rate,
            global_model_rate=global_rate,
            track_running_stats=True,
            use_scaler=use_scaler,
        ).to(device)
        return model

    def init_server(self, global_model: nn.Module) -> None:
        assert self.config.num_clients is not None
        split_mode, model_mode, _, _, _ = self._heterofl_settings()
        global_rate, rates = assign_client_model_rates(
            int(self.config.num_clients),
            split_mode=split_mode,
            model_mode=model_mode,
            seed=int(self.config.training_seed),
            round_id=1,
            custom_split_rates=getattr(self.config, "heterofl_custom_rates", None),
        )
        self.global_model_rate = global_rate
        self.client_model_rates = rates
        self.client_label_splits = {}
        self.client_datasets_cache = {}
        self.global_model_ref = global_model
        self.global_total_params = max(
            1, sum(int(p.numel()) for p in global_model.parameters() if p.requires_grad)
        )
        self._last_dynamic_round = 1
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
        assert cfg.num_clients is not None

        split_mode, model_mode, use_scaler, use_mask, _ = self._heterofl_settings()
        if split_mode == "dynamic" and self._last_dynamic_round != self.current_round_id:
            _, self.client_model_rates = assign_client_model_rates(
                int(cfg.num_clients),
                split_mode="dynamic",
                model_mode=model_mode,
                seed=int(cfg.training_seed),
                round_id=self.current_round_id,
                custom_split_rates=getattr(cfg, "heterofl_custom_rates", None),
            )
            self._last_dynamic_round = self.current_round_id

        self.client_datasets_cache[cid] = client_dataset
        if cid not in self.client_label_splits:
            self.client_label_splits[cid] = _extract_client_label_split(client_dataset)
        label_split = self.client_label_splits[cid]

        rate_m = float(self.client_model_rates[cid])
        scaler_rate = rate_m / float(self.global_model_rate)

        param_idx = compute_subnetwork_indices(
            global_state, cfg.model, scaler_rate=scaler_rate
        )
        local_sub_state, subnet_params, subnet_bytes = extract_subnetwork_state(
            global_state, param_idx
        )

        # Static BatchNorm (`track_running_stats=False`) during local training (`src/models/resnet.py`)
        model = create_heterofl_model(
            cfg.model,
            model_rate=rate_m,
            global_model_rate=self.global_model_rate,
            track_running_stats=False,
            use_scaler=use_scaler,
        ).to(device)
        model.load_state_dict(local_sub_state, strict=True)

        lr_cur = self.current_client_lr()
        mom = float(cfg.momentum if cfg.momentum is not None else 0.9)
        max_norm = float(cfg.max_grad_norm if cfg.max_grad_norm is not None else 1.0)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr_cur,
            momentum=mom,
            weight_decay=cfg.weight_decay,
        )
        loader = DataLoader(
            client_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        model.train()

        num_classes = 10 if cfg.model == "resnet18" else 62
        label_mask_dev: Optional[torch.Tensor] = None
        if use_mask and label_split.numel() > 0:
            label_mask_dev = torch.zeros(num_classes, device=device)
            label_mask_dev[label_split.to(device)] = 1.0

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        local_steps = 0

        target_steps = int(cfg.local_steps) if getattr(cfg, "local_steps", None) is not None else None
        max_passes = (
            max(1, math.ceil(target_steps / max(1, len(loader))))
            if target_steps is not None
            else int(cfg.local_epochs)
        )

        for _ in range(max_passes):
            for x, y in loader:
                if target_steps is not None and local_steps >= target_steps:
                    break
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(x)
                if label_mask_dev is not None:
                    logits = logits.masked_fill(label_mask_dev == 0.0, 0.0)
                loss = F.cross_entropy(logits, y)
                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    raise FloatingPointError(
                        f"HeteroFL client {cid} (rate={rate_m}) encountered nonfinite loss ({loss_val}) at step {local_steps + 1}."
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

                local_steps += 1
                batch_n = int(y.size(0))
                total_loss += loss_val * batch_n
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total_seen += batch_n

            if target_steps is not None and local_steps >= target_steps:
                break
        updated_sub_state: StateDict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        return ClientResult(
            state=updated_sub_state,
            num_examples=total_seen,
            train_loss=total_loss / max(1, total_seen),
            train_accuracy=total_correct / max(1, total_seen),
            local_normalizer=rate_m,
            local_steps=local_steps,
            cid=cid,
            forward_passes=local_steps,
            backward_passes=local_steps,
            upload_bytes_override=subnet_bytes,
            download_bytes_override=subnet_bytes,
            aux_payload={
                "param_idx": param_idx,
                "label_split": label_split,
                "model_rate": rate_m,
                "subnet_params": subnet_params,
                "subnet_bytes": subnet_bytes,
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
        assert self.config.model is not None

        _, _, _, use_mask, use_sbn = self._heterofl_settings()
        local_states = [r.state for r in client_results]
        param_indices = [
            r.aux_payload["param_idx"]
            for r in client_results
            if r.aux_payload is not None
        ]
        label_splits = [
            r.aux_payload["label_split"]
            for r in client_results
            if r.aux_payload is not None
        ]

        new_state = combine_subnetwork_states(
            global_state,
            local_states,
            param_indices,
            label_splits,
            model_name=self.config.model,
            use_label_mask=use_mask,
        )

        # Static BatchNorm (`sBN`) calibration pass (`stats()` in `src/train_classifier_fed.py:134-144`)
        if use_sbn and self.global_model_ref is not None:
            dev = next(self.global_model_ref.parameters()).device
            self.global_model_ref.load_state_dict(new_state, strict=False)
            selected_cids = [int(r.cid) for r in client_results if r.cid is not None]
            calibrate_static_bn(
                self.global_model_ref,
                self.client_datasets_cache,
                selected_cids,
                device=dev,
                batch_size=int(self.config.batch_size or 64),
            )
            calibrated_sd = self.global_model_ref.state_dict()
            for k in new_state:
                if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                    new_state[k] = calibrated_sd[k].detach().cpu().clone()

        mean_rate = sum(r.local_normalizer for r in client_results) / num_selected
        mean_params = (
            sum(
                float(r.aux_payload["subnet_params"])
                for r in client_results
                if r.aux_payload is not None
            )
            / num_selected
        )
        param_ratio = mean_params / float(self.global_total_params)
        mean_steps = sum(r.local_steps for r in client_results) / num_selected

        return new_state, {
            "mean_model_rate": mean_rate,
            "mean_subnet_params": mean_params,
            "mean_subnet_param_ratio": param_ratio,
            "mean_local_steps": mean_steps,
        }

    def format_extra_log(self, extra_metrics: Dict[str, float]) -> str:
        return (
            f"rate={extra_metrics['mean_model_rate']:.3f} "
            f"subnet_ratio={extra_metrics['mean_subnet_param_ratio']:.3f} "
            f"steps={extra_metrics['mean_local_steps']:.1f}"
        )


HeteroFLAlgorithm = HeteroFLMethod
