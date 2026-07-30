import torch
import torch.nn as nn
import torch.nn.functional as F


def _scaled_channels(base_channels, model_rate):
    return [max(1, int(c * model_rate)) for c in base_channels]


class Scaler(nn.Module):
    """
    HeteroFL Scaler.

    During training, smaller subnetworks scale activations by 1 / model_rate.
    During evaluation, the global model is used directly without scaling.
    """
    def __init__(self, model_rate=1.0):
        super().__init__()
        self.model_rate = float(model_rate)

    def forward(self, x):
        if self.training and self.model_rate > 0:
            return x / self.model_rate
        return x


def sbn(num_features):
    """
    Static BatchNorm approximation:
    affine=True, but no running mean/var tracking.

    This matches the HeteroFL idea that BN statistics should not be globally
    tracked during heterogeneous local training.
    """
    return nn.BatchNorm2d(
        num_features,
        affine=True,
        track_running_stats=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, model_rate=1.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.scaler1 = Scaler(model_rate)
        self.bn1 = sbn(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scaler2 = Scaler(model_rate)
        self.bn2 = sbn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                Scaler(model_rate),
                sbn(planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.scaler1(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.scaler2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)

        return out


class HeteroResNet10(nn.Module):
    """
    ResNet10-style CIFAR-10 model.

    Layers: [1, 1, 1, 1]
    This is lighter than PreResNet18 but much closer to the paper setup than
    the earlier 3-layer CNN.
    """
    def __init__(self, model_rate=1.0, num_classes=10):
        super().__init__()

        c1, c2, c3, c4 = _scaled_channels([64, 128, 256, 512], model_rate)

        self.in_planes = c1
        self.model_rate = float(model_rate)

        self.conv1 = nn.Conv2d(
            3,
            c1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.scaler1 = Scaler(model_rate)
        self.bn1 = sbn(c1)

        self.layer1 = self._make_layer(BasicBlock, c1, num_blocks=1, stride=1, model_rate=model_rate)
        self.layer2 = self._make_layer(BasicBlock, c2, num_blocks=1, stride=2, model_rate=model_rate)
        self.layer3 = self._make_layer(BasicBlock, c3, num_blocks=1, stride=2, model_rate=model_rate)
        self.layer4 = self._make_layer(BasicBlock, c4, num_blocks=1, stride=2, model_rate=model_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(c4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, model_rate):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, model_rate))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.scaler1(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out


def create_model(model_rate=1.0, num_classes=10):
    return HeteroResNet10(model_rate=model_rate, num_classes=num_classes)
