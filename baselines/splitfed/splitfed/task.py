from collections import OrderedDict
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from splitfed.leaf_femnist import load_leaf_femnist
from torchvision import datasets, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet10ClientSide(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, stride=1)

    def _make_layer(self, planes, stride):
        block = BasicBlock(self.in_planes, planes, stride)
        self.in_planes = planes
        return block

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        return x


class ResNet10ServerSide(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, stride):
        block = BasicBlock(self.in_planes, planes, stride)
        self.in_planes = planes
        return block

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SplitFedResNet10(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.client_side = ResNet10ClientSide(input_channels=input_channels)
        self.server_side = ResNet10ServerSide(num_classes=num_classes)

    def forward(self, x):
        smashed_data = self.client_side(x)
        return self.server_side(smashed_data)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


@lru_cache(maxsize=8)
def _get_partitions(
    num_partitions: int,
    train_fraction: float,
    data_dir: str = "./data",
    dataset_name="cifar10"):
    """Cache CIFAR-10 partitions so every client does not reload/re-split data."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    dataset_name = str(dataset_name).lower()

    if dataset_name in ["leaf_femnist", "femnist"]:
        client_ids, trainloaders, testloaders = load_leaf_femnist(
            leaf_root=leaf_root,
            partition=partition,
            num_clients=num_partitions,
            batch_size=batch_size,
            seed=seed,
            max_iid_source_clients=max_iid_source_clients,
        )

        key = str(partition_id)

        if key not in trainloaders:
            raise ValueError(f"Client {key} not found in LEAF FEMNIST trainloaders")

        trainloader = trainloaders[key]
        testloader = testloaders.get(key, trainloader)

        return trainloader, testloader

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
    full_test = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

    train_size = int(len(full_train) * train_fraction)
    train_indices = np.random.default_rng(1234).choice(len(full_train), train_size, replace=False)
    train_subset = Subset(full_train, train_indices)

    train_partition_size = len(train_subset) // num_partitions
    test_partition_size = len(full_test) // num_partitions

    train_lengths = [train_partition_size] * num_partitions
    test_lengths = [test_partition_size] * num_partitions

    train_lengths[-1] += len(train_subset) - sum(train_lengths)
    test_lengths[-1] += len(full_test) - sum(test_lengths)

    train_parts = random_split(
        train_subset,
        train_lengths,
        generator=torch.Generator().manual_seed(1234),
    )
    test_parts = random_split(
        full_test,
        test_lengths,
        generator=torch.Generator().manual_seed(1234),
    )

    return train_parts, test_parts


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    train_fraction: float,
    data_dir: str = "./data",
    dataset_name: str = "cifar10",
    partition: str = "niid",
    leaf_root: str = "/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/external/leaf/data/femnist",
    seed: int = 0,
    max_iid_source_clients=None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_name = str(dataset_name).lower()

    if max_iid_source_clients in [None, "", "None", "none"]:
        max_iid_source_clients = None
    else:
        max_iid_source_clients = int(max_iid_source_clients)

    if dataset_name in ["leaf_femnist", "femnist"]:
        client_ids, trainloaders, testloaders = load_leaf_femnist(
            leaf_root=leaf_root,
            partition=partition,
            num_clients=num_partitions,
            batch_size=batch_size,
            seed=seed,
            max_iid_source_clients=max_iid_source_clients,
        )

        key = str(partition_id)
        if key not in trainloaders:
            raise ValueError(f"Client {key} not found in LEAF FEMNIST trainloaders")

        trainloader = trainloaders[key]
        testloader = testloaders.get(key, trainloader)
        return trainloader, testloader

    train_parts, test_parts = _get_partitions(
        num_partitions=num_partitions,
        train_fraction=train_fraction,
        data_dir=data_dir,
        dataset_name=dataset_name)

    trainloader = DataLoader(
        train_parts[partition_id],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = DataLoader(
        test_parts[partition_id],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return trainloader, testloader


def train(model, trainloader, epochs):
    model.train()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def test(model, testloader):
    model.eval()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
