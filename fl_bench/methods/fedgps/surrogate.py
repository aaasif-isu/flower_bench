"""
FedGPS model head expansion (`C + C_surrogate` shared softmax classifier) and
surrogate dataset / synthetic Gaussian batch loader
(Yang et al., NeurIPS 2025).
Reference: https://github.com/visitworld123/FedGPS/blob/main/FedGPS/surrogate_dataset.py
and https://github.com/visitworld123/FedGPS/blob/main/FedGPS/FedGPSClient.py
"""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn

from fl_bench.models import create_model
from fl_bench.utils.seed import set_seed


def build_fedgps_model(
    model_name: str,
    *,
    surrogate_enabled: bool = True,
    surrogate_label_shift: bool = True,
    surrogate_num: Optional[int] = None,
    seed: int = 42,
) -> Tuple[nn.Module, int, int, int]:
    """
    Instantiate `model_name` and, when `surrogate_enabled and surrogate_label_shift`
    (the reference default in `FedGPSClient._initial_setup` and `FedGPSServer._initial_setup`),
    expand the final `nn.Linear` classification head in-place from `C` classes to
    `C + C_surrogate` classes so real samples (`0 .. C-1`) and shifted surrogate samples
    (`C .. C + C_surrogate - 1`) share the exact single `(C + C_surrogate)`-way softmax
    classifier head required by FedGPS.

    Returns `(model, real_num_classes, effective_surrogate_num, feature_dim)`.
    """
    set_seed(seed)
    model = create_model(model_name)

    last_name: Optional[str] = None
    last_linear: Optional[nn.Linear] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_linear = module

    if last_linear is None or last_name is None:
        raise RuntimeError(f"Model '{model_name}' has no nn.Linear classifier head for FedGPS.")

    real_classes = int(last_linear.out_features)
    feat_dim = int(last_linear.in_features)
    n_surr = int(surrogate_num) if (surrogate_num is not None and int(surrogate_num) > 0) else real_classes

    if surrogate_enabled and surrogate_label_shift:
        total_classes = real_classes + n_surr
        new_linear = nn.Linear(
            feat_dim,
            total_classes,
            bias=(last_linear.bias is not None),
        )
        with torch.no_grad():
            new_linear.weight[:real_classes].copy_(last_linear.weight)
            if last_linear.bias is not None and new_linear.bias is not None:
                new_linear.bias[:real_classes].copy_(last_linear.bias)

        parts = last_name.split(".")
        parent: nn.Module = model
        for p_attr in parts[:-1]:
            parent = getattr(parent, p_attr)
        setattr(parent, parts[-1], new_linear)

    return model, real_classes, n_surr, feat_dim


class FedGPSFeatureCatcher:
    """
    Forward pre-hook capturing the penultimate feature vector `feat` entering the final
    `nn.Linear` classification layer, matching `feat, out = self._forward(inputs)` in
    `visitworld123/FedGPS/model/ResNet.py:100-112`.
    """

    def __init__(self, model: nn.Module) -> None:
        self.features: Optional[torch.Tensor] = None
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            raise RuntimeError("Model has no nn.Linear classifier head for FedGPS.")
        self._handle = linear_layers[-1].register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        if inputs:
            feat = inputs[0]
            if feat.dim() > 2:
                feat = feat.reshape(feat.size(0), -1)
            self.features = feat

    def close(self) -> None:
        self._handle.remove()
def generate_surrogate_batch(
    real_x: torch.Tensor,
    *,
    real_num_classes: int,
    surrogate_num: int,
    surrogate_label_shift: bool,
    surrogate_mode: str,
    surrogate_path: Optional[str],
    surrogate_batch_size: Optional[int],
    step_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate or load a surrogate batch `(aux_data, sampled_label)` matching
    `FedGPSClient.generate_noise_data(noise_label_style="extra")` (`FedGPSClient.py:114-156`).

    When `surrogate_label_shift=True` (reference default), `sampled_label` is shifted by
    `+ real_num_classes` so surrogate labels lie in `{C, ..., C + C_surrogate - 1}`,
    completely disjoint from real labels `{0, ..., C - 1}`.
    """
    device = real_x.device
    aux_bs = int(surrogate_batch_size) if (surrogate_batch_size is not None and int(surrogate_batch_size) > 0) else int(real_x.size(0))
    n_surr = max(1, int(surrogate_num))
    label_shift = int(real_num_classes) if surrogate_label_shift else 0

    chunk_size = aux_bs // n_surr
    chunks = [chunk_size] * n_surr
    for i in range(aux_bs - n_surr * chunk_size):
        chunks[i] += 1
    base_labels: List[int] = []
    for cls_idx, count in enumerate(chunks):
        base_labels.extend([cls_idx] * count)

    mode = str(surrogate_mode).strip().lower()
    if mode == "dataset":
        if not surrogate_path or not Path(surrogate_path).exists():
            raise FileNotFoundError(
                f"FedGPS configured with `fedgps_surrogate_mode='dataset'`, but surrogate dataset "
                f"directory '{surrogate_path}' does not exist. Provide the surrogate dataset directory "
                f"or set `fedgps_surrogate_mode='gaussian'`."
            )
        from torchvision import datasets, transforms

        train_dir = Path(surrogate_path) / "train"
        root_dir = train_dir if train_dir.exists() else Path(surrogate_path)
        h, w = int(real_x.size(-2)), int(real_x.size(-1))
        in_ch = int(real_x.size(1)) if real_x.dim() == 4 else 3
        tf_list = []
        if in_ch == 1:
            tf_list.append(transforms.Grayscale(num_output_channels=1))
            mean_t, std_t = (0.5,), (0.25,)
        else:
            mean_t, std_t = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
        tf_list.extend([
            transforms.Resize((h, w)),
            transforms.RandomCrop((h, w), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_t, std_t),
        ])
        ds = datasets.ImageFolder(str(root_dir), transform=transforms.Compose(tf_list))
        gen = torch.Generator().manual_seed(step_seed)
        idxs = torch.randint(0, len(ds), (aux_bs,), generator=gen).tolist()
        imgs = torch.stack([ds[i][0] for i in idxs], dim=0).to(device=device, dtype=real_x.dtype)
        lbls = torch.tensor(
            [(int(ds[i][1]) % n_surr) + label_shift for i in idxs],
            dtype=torch.long,
            device=device,
        )
        return imgs, lbls

    if not real_x.is_floating_point():
        raise ValueError(
            "FedGPS surrogate data generation requires continuous floating-point input tensors "
            "(e.g., CIFAR-10 / FEMNIST images). For discrete token sequence models such as "
            "Shakespeare LSTM, set `fedgps_surrogate_mode: none`."
        )

    gen = torch.Generator(device="cpu")
    aux_shape = (aux_bs,) + tuple(real_x.shape[1:])
    aux_cpu = torch.empty(aux_shape, dtype=real_x.dtype, device="cpu")
    for i in range(aux_bs):
        cls_id = int(base_labels[i])
        gen.manual_seed(10_000 + cls_id * 997)
        cls_template = torch.normal(
            mean=0.0, std=0.25, size=real_x.shape[1:], generator=gen, dtype=real_x.dtype
        )
        gen.manual_seed(step_seed + i * 31 + cls_id)
        noise = torch.normal(
            mean=0.0, std=0.25, size=real_x.shape[1:], generator=gen, dtype=real_x.dtype
        )
        aux_cpu[i] = (0.5 + cls_template + noise - 0.5) / 0.25

    shifted_labels = torch.tensor(
        [int(c) + label_shift for c in base_labels[:aux_bs]],
        dtype=torch.long,
        device=device,
    )
    return aux_cpu.to(device, non_blocking=True), shifted_labels
