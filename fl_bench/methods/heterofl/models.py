"""
Width-scaled subnetwork architectures (`resnet18` and `femnist_cnn`), `Scaler` module,
and Static Batch Normalization (`sBN`) calibration for HeteroFL (Diao et al., ICLR 2021).
Reference: https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/blob/master/src/models/resnet.py
and `src/modules/modules.py`.
"""

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import BasicBlock

SUPPORTED_HETEROF_MODELS: Tuple[str, ...] = ("resnet18", "femnist_cnn")


class Scaler(nn.Module):
    """
    HeteroFL activation scaler (`src/modules/modules.py`):
    divides activations by `rate` during training (`x / rate`) so activation
    magnitude is invariant across subnetwork width rates `rate in (0, 1]`.
    """

    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / self.rate) if (self.training and self.rate != 1.0) else x


class _ScaledBasicBlock(nn.Module):
    """
    Width-scaled ResNet BasicBlock matching FLBench's `resnet18` state_dict key
    hierarchy (`conv1`, `bn1`, `conv2`, `bn2`, `downsample.0`, `downsample.1`)
    with HeteroFL `Scaler(scaler_rate)` and Static BatchNorm (`track_running_stats`).
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        scaler_rate: float = 1.0,
        track_running_stats: bool = False,
        use_scaler: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.scaler1 = Scaler(scaler_rate) if use_scaler else nn.Identity()
        self.bn1 = nn.BatchNorm2d(
            planes, momentum=None, track_running_stats=track_running_stats
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scaler2 = Scaler(scaler_rate) if use_scaler else nn.Identity()
        self.bn2 = nn.BatchNorm2d(
            planes, momentum=None, track_running_stats=track_running_stats
        )
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(
                    planes, momentum=None, track_running_stats=track_running_stats
                ),
            )
        self.scaler_down = Scaler(scaler_rate) if use_scaler else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            d_conv = self.downsample[0](x)
            identity = self.downsample[1](self.scaler_down(d_conv))

        out = self.conv1(x)
        out = self.bn1(self.scaler1(out))
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(self.scaler2(out))

        out += identity
        out = self.relu(out)
        return out
class HeteroFLResNet18(nn.Module):
    """
    Width-scaled CIFAR-10 ResNet-18 for HeteroFL (`model_rate in (0, 1]`),
    preserving exact `state_dict` key alignment with `fl_bench/models/resnet18.py`
    while inserting `Scaler(scaler_rate)` and configurable `track_running_stats` (`sBN`).
    """

    BASE_HIDDEN_SIZES: Tuple[int, int, int, int] = (64, 128, 256, 512)

    def __init__(
        self,
        *,
        model_rate: float = 1.0,
        global_model_rate: float = 1.0,
        num_classes: int = 10,
        track_running_stats: bool = False,
        use_scaler: bool = True,
    ) -> None:
        super().__init__()
        self.model_rate = float(model_rate)
        self.scaler_rate = float(model_rate) / float(global_model_rate)
        self.track_running_stats = bool(track_running_stats)
        self.use_scaler = bool(use_scaler)

        hidden = [max(1, int(math.ceil(self.model_rate * h))) for h in self.BASE_HIDDEN_SIZES]
        self.in_planes = hidden[0]

        self.conv1 = nn.Conv2d(
            3, hidden[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scaler0 = Scaler(self.scaler_rate) if self.use_scaler else nn.Identity()
        self.bn1 = nn.BatchNorm2d(
            hidden[0], momentum=None, track_running_stats=self.track_running_stats
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(hidden[0], 2, stride=1)
        self.layer2 = self._make_layer(hidden[1], 2, stride=2)
        self.layer3 = self._make_layer(hidden[2], 2, stride=2)
        self.layer4 = self._make_layer(hidden[3], 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden[3], num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(
                _ScaledBasicBlock(
                    self.in_planes,
                    planes,
                    stride=s,
                    scaler_rate=self.scaler_rate,
                    track_running_stats=self.track_running_stats,
                    use_scaler=self.use_scaler,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(self.scaler0(out))
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
class HeteroFLFEMNISTCNN(nn.Module):
    """
    Width-scaled FEMNIST 2-Conv CNN for HeteroFL (`model_rate in (0, 1]`),
    preserving exact `state_dict` key alignment with `fl_bench/models/femnist_cnn.py`
    (`features.0`, `features.3`, `classifier.1`, `classifier.3`) while scaling
    channel/hidden dimensions by `ceil(model_rate * H)` and applying `Scaler(scaler_rate)`.
    """

    BASE_CHANNELS: Tuple[int, int] = (32, 64)
    BASE_FC_HIDDEN: int = 2048

    def __init__(
        self,
        *,
        model_rate: float = 1.0,
        global_model_rate: float = 1.0,
        num_classes: int = 62,
        use_scaler: bool = True,
    ) -> None:
        super().__init__()
        self.model_rate = float(model_rate)
        self.scaler_rate = float(model_rate) / float(global_model_rate)
        c1 = max(1, int(math.ceil(self.model_rate * self.BASE_CHANNELS[0])))
        c2 = max(1, int(math.ceil(self.model_rate * self.BASE_CHANNELS[1])))
        h1 = max(1, int(math.ceil(self.model_rate * self.BASE_FC_HIDDEN)))

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * c2, h1),
            nn.ReLU(),
            nn.Linear(h1, num_classes),
        )
        self.scaler = Scaler(self.scaler_rate) if use_scaler else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features[0](x)
        x = self.features[1](self.scaler(x))
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](self.scaler(x))
        x = self.features[5](x)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](self.scaler(x))
        x = self.classifier[3](x)
        return x


def create_heterofl_model(
    model_name: str,
    *,
    model_rate: float = 1.0,
    global_model_rate: float = 1.0,
    track_running_stats: bool = False,
    use_scaler: bool = True,
) -> nn.Module:
    """
    Instantiate a width-scaled HeteroFL model for `model_name`.
    Explicitly rejects unsupported architectures (such as `shakespeare_lstm`).
    """
    name = str(model_name).strip().lower()
    if name == "resnet18":
        return HeteroFLResNet18(
            model_rate=model_rate,
            global_model_rate=global_model_rate,
            num_classes=10,
            track_running_stats=track_running_stats,
            use_scaler=use_scaler,
        )
    if name == "femnist_cnn":
        return HeteroFLFEMNISTCNN(
            model_rate=model_rate,
            global_model_rate=global_model_rate,
            num_classes=62,
            use_scaler=use_scaler,
        )
    raise ValueError(
        f"Model '{model_name}' is not supported by HeteroFL width-scaled subnetwork slicing. "
        f"Supported models: {list(SUPPORTED_HETEROF_MODELS)}. "
        "Note: 'shakespeare_lstm' uses a fused 4-gate PyTorch nn.LSTM recurrent weight layout "
        "([4 * hidden_size, ...]) and is not supported by HeteroFL's top-left subnetwork "
        "prefix slicing (Diao et al., ICLR 2021)."
    )


@torch.no_grad()
def calibrate_static_bn(
    global_model: nn.Module,
    client_datasets: Dict[int, Dataset],
    selected_clients: Sequence[int],
    device: torch.device,
    batch_size: int = 64,
    max_batches_per_client: int = 4,
) -> None:
    """
    Static Batch Normalization (`sBN`) calibration pass (`stats()` in `src/train_classifier_fed.py:134-144`).
    """
    bn_layers = [
        m for m in global_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_layers:
        return

    for m in bn_layers:
        m.track_running_stats = True
        m.momentum = None
        if m.running_mean is not None:
            m.running_mean.zero_()
        if m.running_var is not None:
            m.running_var.fill_(1.0)
        if m.num_batches_tracked is not None:
            m.num_batches_tracked.zero_()

    was_training = global_model.training
    global_model.train(True)
    for cid in selected_clients:
        if cid not in client_datasets:
            continue
        loader = DataLoader(
            client_datasets[cid],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        for b_idx, (x, _) in enumerate(loader):
            if b_idx >= max_batches_per_client:
                break
            global_model(x.to(device, non_blocking=True))
    global_model.train(was_training)
