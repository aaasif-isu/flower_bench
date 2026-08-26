import torch
from torch import nn


class FEMNISTCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                7 * 7 * 64,
                2048,
            ),
            nn.ReLU(),
            nn.Linear(
                2048,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def create_femnist_cnn():
    return FEMNISTCNN(num_classes=62)
