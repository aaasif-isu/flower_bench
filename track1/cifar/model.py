import torch
from torch import nn
from torchvision.models import resnet18


def create_resnet18(num_classes=10):
    """CIFAR-10 ResNet-18 shared by every Track 1 algorithm."""

    model = resnet18(weights=None, num_classes=num_classes)

    # Standard CIFAR adaptation:
    # 32x32 images do not need ImageNet's 7x7 stride-2 stem.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    # Remove initial max pooling for CIFAR resolution.
    model.maxpool = nn.Identity()

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = create_resnet18()
    print(model)
    print("Trainable parameters:", count_parameters(model))

    x = torch.randn(4, 3, 32, 32)
    y = model(x)

    print("Input:", x.shape)
    print("Output:", y.shape)
