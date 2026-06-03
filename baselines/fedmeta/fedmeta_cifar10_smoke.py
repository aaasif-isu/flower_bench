import argparse
import csv
import os
import time
from collections import OrderedDict
from copy import deepcopy

import matplotlib.pyplot as plt

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

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
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 1, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)

        return self.linear(out)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)

    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )

    net.load_state_dict(state_dict, strict=True)



def arrays_size_mb(arrays):
    """Return total size of a list of numpy arrays/tensors in MB."""
    total_bytes = 0
    for arr in arrays:
        if hasattr(arr, "nbytes"):
            total_bytes += arr.nbytes
        elif hasattr(arr, "numel") and hasattr(arr, "element_size"):
            total_bytes += arr.numel() * arr.element_size()
        else:
            total_bytes += np.asarray(arr).nbytes
    return total_bytes / (1024 * 1024)


def train_meta_first_order(
    net,
    supportloader,
    queryloader,
    inner_lr,
    device,
    gradient_steps,
):
    criterion = torch.nn.CrossEntropyLoss()

    train_net = deepcopy(net).to(device)
    train_net.train()

    for _ in range(gradient_steps):
        for images, labels in supportloader:
            images = images.to(device)
            labels = labels.to(device)

            loss = criterion(train_net(images), labels)

            grads = torch.autograd.grad(
                loss,
                list(train_net.parameters()),
                retain_graph=False,
            )

            for param, grad in zip(train_net.parameters(), grads):
                param.data = param.data - inner_lr * grad

    query_loss = 0.0
    query_count = 0

    for images, labels in queryloader:
        images = images.to(device)
        labels = labels.to(device)

        loss = criterion(train_net(images), labels)

        query_loss = query_loss + loss * labels.size(0)
        query_count += labels.size(0)

    query_loss = query_loss / query_count

    grads = torch.autograd.grad(
        query_loss,
        list(train_net.parameters()),
    )

    grads_np = [g.detach().cpu().numpy() for g in grads]

    return float(query_loss.detach().cpu()), grads_np, query_count


def evaluate_model(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()

    net.eval()

    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            batch_loss = criterion(outputs, labels)

            loss += batch_loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = loss / total
    accuracy = correct / total

    return avg_loss, accuracy


class CifarFedMetaClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        supportloader,
        queryloader,
        inner_lr,
        gradient_steps,
    ):
        self.cid = int(cid)
        self.supportloader = supportloader
        self.queryloader = queryloader
        self.inner_lr = inner_lr
        self.gradient_steps = gradient_steps

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.net = ResNet10().to(self.device)

    def get_parameters(self, config):
        return get_weights(self.net)

    def set_parameters(self, parameters):
        set_weights(self.net, parameters)

    def fit(self, parameters, config):
        fit_start = time.perf_counter()
        fit_param_down_mb = arrays_size_mb(parameters)

        self.set_parameters(parameters)

        train_start = time.perf_counter()
        loss, grads, num_examples = train_meta_first_order(
            self.net,
            self.supportloader,
            self.queryloader,
            self.inner_lr,
            self.device,
            self.gradient_steps,
        )
        train_time_sec = time.perf_counter() - train_start

        # FedMeta returns gradients as the "parameters" payload.
        # So the upload size is the gradient payload size.
        fit_grad_up_mb = arrays_size_mb(grads)
        fit_total_time_sec = time.perf_counter() - fit_start

        return grads, num_examples, {
            "loss": float(loss),
            "client_train_time_sec": float(train_time_sec),
            "client_fit_total_time_sec": float(fit_total_time_sec),
            "fit_param_down_mb": float(fit_param_down_mb),
            "fit_param_up_mb": float(fit_grad_up_mb),
            "fit_grad_up_mb": float(fit_grad_up_mb),
            "fit_param_total_comm_mb": float(fit_param_down_mb + fit_grad_up_mb),
        }

    def evaluate(self, parameters, config):
        eval_start = time.perf_counter()
        eval_param_down_mb = arrays_size_mb(parameters)

        self.set_parameters(parameters)

        loss, acc = evaluate_model(
            self.net,
            self.queryloader,
            self.device,
        )

        eval_time_sec = time.perf_counter() - eval_start

        return float(loss), len(self.queryloader.dataset), {
            "accuracy": float(acc),
            "client_eval_time_sec": float(eval_time_sec),
            "eval_param_down_mb": float(eval_param_down_mb),
            "eval_param_up_mb": 0.0,
            "eval_param_total_comm_mb": float(eval_param_down_mb),
        }


class CifarFedMetaStrategy(FedAvg):
    def __init__(
        self,
        net,
        outer_lr,
        weight_decay,
        testloader=None,
        metrics_csv="fedmeta_cifar10_resnet10_detailed.csv",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.net = deepcopy(net).cpu()
        self.outer_lr = outer_lr
        self.weight_decay = weight_decay
        self.current_weights = get_weights(self.net)

        self.testloader = testloader
        self.metrics_csv = metrics_csv

        self.round_stats = {}
        self.cumulative_total_comm_mb = 0.0
        self.cumulative_round_wall_time_sec = 0.0

        self.csv_fieldnames = [
            "method",
            "dataset",
            "model",
            "round",
            "train_loss",
            "train_accuracy",
            "test_loss",
            "test_accuracy",
            "num_fit_clients",
            "fit_failures",
            "fit_param_down_mb",
            "fit_param_up_mb",
            "fit_param_total_comm_mb",
            "split_train_smashed_forward_mb",
            "split_train_smashed_backward_mb",
            "split_train_label_mb",
            "split_train_total_comm_mb",
            "fit_total_comm_mb",
            "client_train_time_mean_sec",
            "client_train_time_max_sec",
            "fit_total_time_mean_sec",
            "fit_total_time_max_sec",
            "fit_wall_time_sec",
            "num_eval_clients",
            "eval_failures",
            "eval_param_down_mb",
            "eval_param_up_mb",
            "eval_param_total_comm_mb",
            "split_eval_smashed_forward_mb",
            "split_eval_label_mb",
            "split_eval_total_comm_mb",
            "eval_total_comm_mb",
            "client_eval_time_mean_sec",
            "client_eval_time_max_sec",
            "eval_total_time_mean_sec",
            "eval_total_time_max_sec",
            "eval_wall_time_sec",
            "round_param_comm_mb",
            "round_splitfed_comm_mb",
            "round_total_comm_mb",
            "cumulative_total_comm_mb",
            "round_wall_time_sec",
            "cumulative_round_wall_time_sec",
        ]

        if self.metrics_csv is not None:
            with open(self.metrics_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()

    def configure_fit(self, server_round, parameters, client_manager):
        round_start = time.perf_counter()

        fit_ins_list = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

        fit_param_down_mb_per_client = arrays_size_mb(parameters_to_ndarrays(parameters))
        num_fit_clients = len(fit_ins_list)

        self.round_stats[server_round] = {
            "round_start_time": round_start,
            "fit_param_down_mb": fit_param_down_mb_per_client * num_fit_clients,
            "fit_param_up_mb": 0.0,
            "fit_param_total_comm_mb": 0.0,
            "fit_total_comm_mb": 0.0,
            "num_fit_clients": num_fit_clients,
            "fit_failures": 0,
            "client_train_times": [],
            "client_fit_total_times": [],
            "train_losses": [],
            "fit_wall_time_sec": 0.0,
            "eval_wall_time_sec": 0.0,
        }

        return fit_ins_list

    def evaluate_global_model(self):
        """Evaluate current global model on centralized CIFAR-10 test set."""

        if self.testloader is None:
            return None, None

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        eval_net = ResNet10().to(device)
        set_weights(eval_net, self.current_weights)

        test_loss, test_acc = evaluate_model(
            eval_net,
            self.testloader,
            device,
        )

        return test_loss, test_acc

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        if server_round not in self.round_stats:
            self.round_stats[server_round] = {
                "round_start_time": time.perf_counter(),
                "fit_param_down_mb": 0.0,
            }

        stats = self.round_stats[server_round]
        stats["num_fit_clients"] = len(results)
        stats["fit_failures"] = len(failures)

        fit_param_up_mb = 0.0
        client_train_times = []
        client_fit_total_times = []
        losses = []

        for _, fit_res in results:
            # In this FedMeta implementation, fit_res.parameters is the gradient payload.
            fit_param_up_mb += arrays_size_mb(parameters_to_ndarrays(fit_res.parameters))

            if "client_train_time_sec" in fit_res.metrics:
                client_train_times.append(float(fit_res.metrics["client_train_time_sec"]))

            if "client_fit_total_time_sec" in fit_res.metrics:
                client_fit_total_times.append(float(fit_res.metrics["client_fit_total_time_sec"]))

            if "loss" in fit_res.metrics:
                losses.append(float(fit_res.metrics["loss"]))

        stats["fit_param_up_mb"] = fit_param_up_mb
        stats["client_train_times"] = client_train_times
        stats["client_fit_total_times"] = client_fit_total_times
        stats["train_losses"] = losses

        down_mb = float(stats.get("fit_param_down_mb", 0.0))
        stats["fit_param_total_comm_mb"] = down_mb + fit_param_up_mb
        stats["fit_total_comm_mb"] = down_mb + fit_param_up_mb

        grad_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
            )
            for _, fit_res in results
        ]

        gradients_aggregated = aggregate(grad_results)

        set_weights(self.net, self.current_weights)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.outer_lr,
            weight_decay=self.weight_decay,
        )

        for param, grad in zip(self.net.parameters(), gradients_aggregated):
            param.grad = torch.tensor(grad, dtype=param.dtype)

        optimizer.step()
        optimizer.zero_grad()

        self.current_weights = get_weights(self.net)

        avg_client_query_loss = float(sum(losses) / len(losses)) if losses else 0.0

        eval_start = time.perf_counter()
        test_loss, test_acc = self.evaluate_global_model()
        eval_wall_time_sec = time.perf_counter() - eval_start

        stats["eval_wall_time_sec"] = eval_wall_time_sec

        if "round_start_time" in stats:
            stats["fit_wall_time_sec"] = time.perf_counter() - stats["round_start_time"]

        self._write_round_csv_row(
            server_round=server_round,
            train_loss=avg_client_query_loss,
            test_loss=float(test_loss),
            test_accuracy=float(test_acc),
        )

        print(
            f"[FedMeta] round={server_round}, "
            f"avg_client_query_loss={avg_client_query_loss:.4f}, "
            f"test_loss={test_loss:.4f}, "
            f"test_accuracy={test_acc:.4f}",
            flush=True,
        )

        return ndarrays_to_parameters(self.current_weights), {
            "avg_client_query_loss": avg_client_query_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

    def _write_round_csv_row(
        self,
        server_round,
        train_loss,
        test_loss,
        test_accuracy,
    ):
        stats = self.round_stats.get(server_round, {})

        round_wall_time_sec = 0.0
        if "round_start_time" in stats:
            round_wall_time_sec = time.perf_counter() - stats["round_start_time"]

        self.cumulative_round_wall_time_sec += round_wall_time_sec

        fit_param_down_mb = float(stats.get("fit_param_down_mb", 0.0))
        fit_param_up_mb = float(stats.get("fit_param_up_mb", 0.0))
        fit_param_total_comm_mb = fit_param_down_mb + fit_param_up_mb

        # Centralized evaluation happens on server, so no client eval communication.
        eval_param_down_mb = 0.0
        eval_param_up_mb = 0.0
        eval_param_total_comm_mb = 0.0

        round_param_comm_mb = fit_param_total_comm_mb + eval_param_total_comm_mb
        round_splitfed_comm_mb = 0.0
        round_total_comm_mb = round_param_comm_mb + round_splitfed_comm_mb

        self.cumulative_total_comm_mb += round_total_comm_mb

        client_train_times = stats.get("client_train_times", [])
        client_fit_total_times = stats.get("client_fit_total_times", [])

        client_train_time_mean_sec = (
            sum(client_train_times) / len(client_train_times)
            if client_train_times
            else 0.0
        )
        client_train_time_max_sec = max(client_train_times) if client_train_times else 0.0

        fit_total_time_mean_sec = (
            sum(client_fit_total_times) / len(client_fit_total_times)
            if client_fit_total_times
            else 0.0
        )
        fit_total_time_max_sec = (
            max(client_fit_total_times) if client_fit_total_times else 0.0
        )

        row = {
            "method": "FedMeta",
            "dataset": "CIFAR10",
            "model": "ResNet10",
            "round": server_round,
            "train_loss": float(train_loss),
            "train_accuracy": 0.0,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "num_fit_clients": int(stats.get("num_fit_clients", 0)),
            "fit_failures": int(stats.get("fit_failures", 0)),
            "fit_param_down_mb": fit_param_down_mb,
            "fit_param_up_mb": fit_param_up_mb,
            "fit_param_total_comm_mb": fit_param_total_comm_mb,
            "split_train_smashed_forward_mb": 0.0,
            "split_train_smashed_backward_mb": 0.0,
            "split_train_label_mb": 0.0,
            "split_train_total_comm_mb": 0.0,
            "fit_total_comm_mb": fit_param_total_comm_mb,
            "client_train_time_mean_sec": client_train_time_mean_sec,
            "client_train_time_max_sec": client_train_time_max_sec,
            "fit_total_time_mean_sec": fit_total_time_mean_sec,
            "fit_total_time_max_sec": fit_total_time_max_sec,
            "fit_wall_time_sec": float(stats.get("fit_wall_time_sec", 0.0)),
            "num_eval_clients": 0,
            "eval_failures": 0,
            "eval_param_down_mb": eval_param_down_mb,
            "eval_param_up_mb": eval_param_up_mb,
            "eval_param_total_comm_mb": eval_param_total_comm_mb,
            "split_eval_smashed_forward_mb": 0.0,
            "split_eval_label_mb": 0.0,
            "split_eval_total_comm_mb": 0.0,
            "eval_total_comm_mb": eval_param_total_comm_mb,
            "client_eval_time_mean_sec": 0.0,
            "client_eval_time_max_sec": 0.0,
            "eval_total_time_mean_sec": 0.0,
            "eval_total_time_max_sec": 0.0,
            "eval_wall_time_sec": float(stats.get("eval_wall_time_sec", 0.0)),
            "round_param_comm_mb": round_param_comm_mb,
            "round_splitfed_comm_mb": round_splitfed_comm_mb,
            "round_total_comm_mb": round_total_comm_mb,
            "cumulative_total_comm_mb": self.cumulative_total_comm_mb,
            "round_wall_time_sec": round_wall_time_sec,
            "cumulative_round_wall_time_sec": self.cumulative_round_wall_time_sec,
        }

        if self.metrics_csv is not None:
            with open(self.metrics_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writerow(row)

        print(f"[FEDMETA CSV] wrote round {server_round} to {self.metrics_csv}", flush=True)


def make_client_loaders(args):
    transform = Compose(
        [
            ToTensor(),
            Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    trainset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    testset = CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    rng = np.random.default_rng(args.seed)

    train_indices = rng.permutation(len(trainset))
    client_chunks = np.array_split(train_indices, args.num_clients)

    supportloaders = []
    queryloaders = []

    for chunk in client_chunks:
        chunk = np.array(chunk)
        rng.shuffle(chunk)

        if args.samples_per_client > 0:
            chunk = chunk[: args.samples_per_client]

        split = max(1, len(chunk) // 2)

        support_idx = chunk[:split]
        query_idx = chunk[split:]

        if len(query_idx) == 0:
            query_idx = support_idx

        supportloaders.append(
            DataLoader(
                Subset(trainset, support_idx.tolist()),
                batch_size=args.batch_size,
                shuffle=True,
            )
        )

        queryloaders.append(
            DataLoader(
                Subset(trainset, query_idx.tolist()),
                batch_size=args.batch_size,
                shuffle=True,
            )
        )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return supportloaders, queryloaders, testloader


def plot_round_metrics(
    csv_path="fedmeta_cifar10_resnet10_detailed.csv",
    loss_png="fedmeta_round_loss_graph.png",
    accuracy_png="fedmeta_round_accuracy_graph.png",
):
    df = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            df.append(row)

    if not df:
        print("No rows found for plotting.")
        return

    rounds = [int(row["round"]) for row in df]
    train_losses = [float(row["train_loss"]) for row in df]
    test_losses = [float(row["test_loss"]) for row in df]
    test_accuracies = [float(row["test_accuracy"]) for row in df]
    round_times = [float(row["round_wall_time_sec"]) for row in df]
    cumulative_times = [float(row["cumulative_round_wall_time_sec"]) for row in df]
    round_comm = [float(row["round_total_comm_mb"]) for row in df]
    cumulative_comm = [float(row["cumulative_total_comm_mb"]) for row in df]
    client_train_mean = [float(row["client_train_time_mean_sec"]) for row in df]
    client_train_max = [float(row["client_train_time_max_sec"]) for row in df]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, train_losses, marker="o", label="Avg Client Query Loss")
    plt.plot(rounds, test_losses, marker="o", label="Centralized Test Loss")
    plt.title("FedMeta CIFAR-10 ResNet10 Loss")
    plt.xlabel("Global Round")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, test_accuracies, marker="o", label="Centralized Test Accuracy")
    plt.title("FedMeta CIFAR-10 ResNet10 Accuracy")
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_png, dpi=200)
    plt.close()

    plots = [
        ("fedmeta_round_time_graph.png", round_times, "Round Wall Time", "Seconds"),
        ("fedmeta_cumulative_time_graph.png", cumulative_times, "Cumulative Wall Time", "Seconds"),
        ("fedmeta_round_communication_graph.png", round_comm, "Round Communication", "MB"),
        ("fedmeta_cumulative_communication_graph.png", cumulative_comm, "Cumulative Communication", "MB"),
    ]

    for filename, values, title, ylabel in plots:
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, values, marker="o")
        plt.title(f"FedMeta CIFAR-10 ResNet10 {title}")
        plt.xlabel("Global Round")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, client_train_mean, marker="o", label="Mean Client Train Time")
    plt.plot(rounds, client_train_max, marker="o", label="Max Client Train Time")
    plt.title("FedMeta CIFAR-10 ResNet10 Client Train Time")
    plt.xlabel("Global Round")
    plt.ylabel("Seconds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fedmeta_client_train_time_graph.png", dpi=200)
    plt.close()

    print(f"Saved loss graph to: {loss_png}")
    print(f"Saved accuracy graph to: {accuracy_png}")
    print("Saved time, communication, and client train time graphs.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-client", type=int, default=200)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=0.001)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--client-cpus", type=float, default=2)
    parser.add_argument("--client-gpus", type=float, default=0.0)
    parser.add_argument("--ray-cpus", type=int, default=16)
    parser.add_argument("--ray-gpus", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    supportloaders, queryloaders, testloader = make_client_loaders(args)

    initial_net = ResNet10()
    initial_parameters = ndarrays_to_parameters(get_weights(initial_net))

    metrics_csv = "fedmeta_cifar10_resnet10_detailed.csv"

    strategy = CifarFedMetaStrategy(
        net=initial_net,
        outer_lr=args.outer_lr,
        weight_decay=0.0001,
        testloader=testloader,
        metrics_csv=metrics_csv,
        fraction_fit=args.clients_per_round / args.num_clients,
        fraction_evaluate=0.0,
        min_fit_clients=args.clients_per_round,
        min_available_clients=args.num_clients,
        initial_parameters=initial_parameters,
    )

    def client_fn(cid: str):
        return CifarFedMetaClient(
            cid=cid,
            supportloader=supportloaders[int(cid)],
            queryloader=queryloaders[int(cid)],
            inner_lr=args.inner_lr,
            gradient_steps=args.gradient_steps,
        )

    ray_tmp = (
        f"/tmp/ray_fedmeta_cifar10_"
        f"{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )

    os.makedirs(ray_tmp, exist_ok=True)

    print("Starting FedMeta CIFAR-10 ResNet10 smoke benchmark")
    print(
        f"num_clients={args.num_clients}, "
        f"clients_per_round={args.clients_per_round}"
    )
    print(
        f"rounds={args.rounds}, "
        f"client_gpus={args.client_gpus}, "
        f"ray_tmp={ray_tmp}"
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": args.client_cpus,
            "num_gpus": args.client_gpus,
        },
        ray_init_args={
            "include_dashboard": False,
            "_temp_dir": ray_tmp,
            "num_cpus": args.ray_cpus,
            "num_gpus": args.ray_gpus,
        },
    )

    plot_round_metrics(
        csv_path=metrics_csv,
        loss_png="fedmeta_round_loss_graph.png",
        accuracy_png="fedmeta_round_accuracy_graph.png",
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    final_net = ResNet10().to(device)
    set_weights(final_net, strategy.current_weights)

    test_loss, test_acc = evaluate_model(
        final_net,
        testloader,
        device,
    )

    print(f"Final centralized CIFAR-10 test loss: {test_loss:.4f}")
    print(f"Final centralized CIFAR-10 test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()