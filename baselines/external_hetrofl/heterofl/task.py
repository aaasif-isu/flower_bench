import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from .model import create_model


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_transforms():
    return Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def load_datasets(num_clients, batch_size):
    data_dir = os.environ.get("FLOWER_DATA_DIR", "./data")

    trainset = CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=get_transforms(),
    )

    testset = CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=get_transforms(),
    )

    indices = np.arange(len(trainset))
    client_indices = np.array_split(indices, num_clients)

    trainloaders = [
        DataLoader(
            Subset(trainset, idxs.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        for idxs in client_indices
    ]

    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return trainloaders, testloader


def train(model, trainloader, epochs, device):
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(1, total_examples)
    avg_acc = total_correct / max(1, total_examples)

    return avg_loss, avg_acc


def test(model, testloader, device):
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(1, total_examples)
    avg_acc = total_correct / max(1, total_examples)

    return avg_loss, avg_acc


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict(
        {key: torch.tensor(value) for key, value in zip(keys, parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def parameters_to_state_dict(model, parameters):
    keys = list(model.state_dict().keys())
    return OrderedDict(
        {key: torch.tensor(value) for key, value in zip(keys, parameters)}
    )


def state_dict_to_parameters(state_dict):
    return [val.cpu().numpy() for _, val in state_dict.items()]
