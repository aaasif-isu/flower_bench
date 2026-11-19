"""
Statistical goal-rectification losses (`SupConLoss`, `ProxyAlignLoss`, `TripletLoss`)
and prototype aggregation primitives (`local_proto_agg`, `proto_aggregation`) for FedGPS
(Yang et al., NeurIPS 2025).
Reference: https://github.com/visitworld123/FedGPS/blob/main/FedGPS/utils.py
and https://github.com/visitworld123/FedGPS/blob/main/FedGPS/FedGPSClient.py
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss (`SupConLoss` in `FedGPS/utils.py`).
    """

    def __init__(self, contrast_mode: str = "all", base_temperature: float = 0.07) -> None:
        super().__init__()
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = features.device
        if features.dim() < 3:
            features = features.unsqueeze(1)
        if features.dim() > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.zeros((), device=device)

        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels_col = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels_col, labels_col.T).float().to(device)
        else:
            assert mask is not None
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0.0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -(temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()
class ProxyAlignLoss(nn.Module):
    """
    Surrogate statistical alignment objective (`proxy_align_loss` in `FedGPS/utils.py:51-138`).
    Note: `labels` has shape `[real_batch_size + noise_batch_size]` where
    `labels[:real_batch_size]` are in `0..C-1` and `labels[real_batch_size:]` are shifted
    by `+ noise_label_shift` (`C..C+C_surr-1`), ensuring disjoint label spaces between
    real and surrogate samples in `align_cls_loss`.
    """

    def __init__(
        self,
        *,
        inter_domain_mapping: bool = False,
        inter_domain_class_match: bool = True,
        noise_feat_detach: bool = True,
        noise_contrastive: bool = True,
        inter_domain_mapping_matrix: Optional[torch.Tensor] = None,
        inter_domain_weight: float = 0.0,
        inter_class_weight: float = 1.0,
        noise_supcon_weight: float = 0.1,
        noise_label_shift: int = 10,
    ) -> None:
        super().__init__()
        self.inter_domain_mapping = inter_domain_mapping
        self.inter_domain_class_match = inter_domain_class_match
        self.noise_feat_detach = noise_feat_detach
        self.noise_contrastive = noise_contrastive
        self.inter_domain_mapping_matrix = inter_domain_mapping_matrix
        self.inter_domain_weight = inter_domain_weight
        self.inter_class_weight = inter_class_weight
        self.noise_supcon_weight = noise_supcon_weight
        self.noise_label_shift = int(noise_label_shift)
        self.supcon_loss = SupConLoss(contrast_mode="all", base_temperature=0.07)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        real_batch_size: int,
    ) -> Tuple[torch.Tensor, float, float, float]:
        real_features = features[:real_batch_size]
        noise_features = features[real_batch_size:]
        noise_batch_size = int(noise_features.shape[0])
        if real_batch_size == 0 or noise_batch_size == 0:
            zero = torch.zeros((), device=features.device)
            return zero, 0.0, 0.0, 0.0

        if self.inter_domain_mapping and self.inter_domain_mapping_matrix is not None:
            mapped_noise = torch.matmul(
                noise_features, self.inter_domain_mapping_matrix.to(features.device)
            )
            new_features = torch.cat([real_features, mapped_noise], dim=0)
        elif self.noise_feat_detach:
            new_features = torch.cat(
                [real_features, noise_features.clone().detach()], dim=0
            )
        else:
            new_features = features

        min_bs = min(real_batch_size, noise_batch_size)
        align_domain_loss = (
            torch.linalg.norm(
                real_features[:min_bs] - noise_features[:min_bs], ord=2, dim=1
            ).sum()
            / float(real_batch_size)
        )

        norm_new = F.normalize(new_features, dim=1, eps=1e-8).unsqueeze(1)
        norm_noise = F.normalize(noise_features, dim=1, eps=1e-8).unsqueeze(1)

        align_cls_loss = self.supcon_loss(norm_new, labels=labels, temperature=0.07)

        if self.noise_contrastive:
            noise_labels = labels[real_batch_size:] - self.noise_label_shift
            noise_cls_loss = self.supcon_loss(
                norm_noise, labels=noise_labels, temperature=0.07
            )
            noise_cls_val = float(noise_cls_loss.item())
        else:
            noise_cls_loss = torch.zeros((), device=features.device)
            noise_cls_val = 0.0

        total = (
            self.inter_domain_weight * align_domain_loss
            + self.inter_class_weight * align_cls_loss
            + self.noise_supcon_weight * noise_cls_loss
        )
        return (
            total,
            float(align_domain_loss.item()),
            float(align_cls_loss.item()),
            noise_cls_val,
        )
class TripletLoss(nn.Module):
    """Triplet loss for prototype alignment (`TripletLoss` in `FedGPSClient.py:424-434`)."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        return torch.relu(d_pos - d_neg + self.margin).mean()


def local_proto_agg(
    protos: Dict[int, List[torch.Tensor]],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    """Client-side prototype averaging (`local_proto_agg` in `FedGPSClient.py:403-421`)."""
    new_protos: Dict[int, torch.Tensor] = {}
    proto_agg_weights: Dict[int, int] = {}
    for label, proto_list in protos.items():
        if not proto_list:
            continue
        if len(proto_list) > 1:
            acc = torch.zeros_like(proto_list[0].data, dtype=torch.float64)
            for p in proto_list:
                acc += p.data.to(torch.float64)
            new_protos[label] = (acc / float(len(proto_list))).to(proto_list[0].dtype)
            proto_agg_weights[label] = len(proto_list)
        else:
            new_protos[label] = proto_list[0].detach().clone()
            proto_agg_weights[label] = 1
    return new_protos, proto_agg_weights


def proto_aggregation(
    clients_noise_protos: Sequence[Dict[int, torch.Tensor]],
    clients_proto_agg_weights: Sequence[Dict[int, int]],
) -> Dict[int, torch.Tensor]:
    """Server-side sample-weighted prototype aggregation (`proto_aggregation` in `FedGPS/utils.py:25-49`)."""
    agg_protos_label: Dict[int, List[torch.Tensor]] = {}
    agg_weights_label: Dict[int, List[float]] = {}

    for c_protos, c_weights in zip(clients_noise_protos, clients_proto_agg_weights):
        for label, proto in c_protos.items():
            agg_protos_label.setdefault(label, []).append(proto.detach().cpu())
            agg_weights_label.setdefault(label, []).append(float(c_weights.get(label, 0)))

    weighted_agg_protos: Dict[int, torch.Tensor] = {}
    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            proto_stack = torch.stack([p.to(torch.float64) for p in proto_list], dim=0)
            weights = torch.tensor(
                agg_weights_label[label], dtype=torch.float64, device=proto_stack.device
            )
            w_sum = weights.sum().clamp_min(1e-12)
            weights = weights / w_sum
            weighted_proto = (weights.unsqueeze(1) * proto_stack).sum(dim=0)
            weighted_agg_protos[label] = weighted_proto.to(proto_list[0].dtype)
        else:
            weighted_agg_protos[label] = proto_list[0].clone()
    return weighted_agg_protos
