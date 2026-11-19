"""CIFAR-10 ResNet-18 model definition."""

from torch import nn
from torchvision.models import resnet18


def create_resnet18(num_classes: int = 10) -> nn.Module:
    """CIFAR-10 ResNet-18 shared across all CIFAR-10 FL benchmark runs."""
    model = resnet18(weights=None, num_classes=num_classes)

    # Standard CIFAR adaptation: 32x32 images use a 3x3 stride-1 stem
    # without initial max pooling.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    return model
