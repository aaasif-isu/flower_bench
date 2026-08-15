import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import create_resnet18


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_partition(path, num_clients):
    data = np.load(path)

    return [
        np.asarray(data[f"client_{cid}"], dtype=np.int64)
        for cid in range(num_clients)
    ]


def load_schedule(path):
    with open(path, "r") as f:
        payload = json.load(f)

    return payload["schedule"]


def get_datasets(data_dir):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    trainset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform,
    )

    testset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform,
    )

    return trainset, testset


def zero_control(global_model):
    return {
        name: torch.zeros_like(
            param.detach().cpu()
        )
        for name, param in global_model.named_parameters()
        if param.requires_grad
    }


def train_client(
    cid,
    global_state,
    c_global,
    c_local,
    client_indices,
    trainset,
    device,
    batch_size,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    seed,
):
    """
    SCAFFOLD client update.

    Local gradient correction:

        g <- g + c_global - c_local

    After K local optimizer steps:

        y_delta = w_local - w_global

        c_plus =
            c_local
            - c_global
            - y_delta / (K * lr)

        c_delta = c_plus - c_local
    """

    set_seed(seed)

    model = create_resnet18().to(device)
    model.load_state_dict(global_state)

    loader = DataLoader(
        Subset(trainset, client_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    local_steps = 0

    # Move the control variates for this client onto
    # the training device once.
    cg_device = {
        name: value.to(device)
        for name, value in c_global.items()
    }

    cl_device = {
        name: value.to(device)
        for name, value in c_local.items()
    }

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            # SCAFFOLD gradient correction:
            #
            # grad <- grad + c_global - c_local
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if (
                        param.requires_grad
                        and param.grad is not None
                    ):
                        param.grad.add_(
                            cg_device[name]
                            - cl_device[name]
                        )

            optimizer.step()

            local_steps += 1

            batch_n = y.size(0)

            total_loss += (
                loss.item() * batch_n
            )

            total_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            total_seen += batch_n

    if local_steps == 0:
        raise RuntimeError(
            f"Client {cid} performed zero local steps"
        )

    local_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    y_delta = {}
    c_plus = {}
    c_delta = {}

    coef = 1.0 / (
        local_steps * lr
    )

    with torch.no_grad():
        for name in c_global:
            delta = (
                local_state[name]
                - global_state[name]
            )

            y_delta[name] = delta.clone()

            new_c = (
                c_local[name]
                - c_global[name]
                - coef * delta
            )

            c_plus[name] = new_c.clone()

            c_delta[name] = (
                new_c
                - c_local[name]
            )

    return {
        "state": local_state,
        "num_examples": total_seen,
        "train_loss": (
            total_loss / total_seen
        ),
        "train_accuracy": (
            total_correct / total_seen
        ),
        "y_delta": y_delta,
        "c_plus": c_plus,
        "c_delta": c_delta,
        "local_steps": local_steps,
    }


def aggregate_scaffold(
    global_state,
    c_global,
    client_results,
    num_clients_total,
    trainable_keys,
):
    """
    Reference SCAFFOLD server update:

        w <- w + average(y_delta)

        c <- c + (M / N) * average(c_delta)

    M = participating clients this round
    N = total clients

    Non-trainable buffers use ordinary weighted averaging.
    """

    num_selected = len(client_results)

    if num_selected == 0:
        raise RuntimeError(
            "No selected clients"
        )

    new_state = {}

    # Reference implementation uses equal averaging across
    # participating clients for trainable parameters.
    uniform_weight = (
        1.0 / num_selected
    )

    for key, global_value in global_state.items():

        if not torch.is_floating_point(
            global_value
        ):
            # e.g. BatchNorm num_batches_tracked
            new_state[key] = (
                client_results[0]["state"][key]
                .clone()
            )
            continue

        if key in trainable_keys:
            avg_delta = torch.zeros_like(
                global_value,
                dtype=torch.float64,
            )

            for result in client_results:
                avg_delta += (
                    result["y_delta"][key]
                    .to(torch.float64)
                    * uniform_weight
                )

            new_state[key] = (
                global_value.to(
                    torch.float64
                )
                + avg_delta
            ).to(global_value.dtype)

        else:
            # Non-trainable floating buffers use sample
            # weighted averaging, consistent with our
            # other Track-1 runners.
            total_examples = sum(
                result["num_examples"]
                for result in client_results
            )

            accumulator = torch.zeros_like(
                global_value,
                dtype=torch.float64,
            )

            for result in client_results:
                weight = (
                    result["num_examples"]
                    / total_examples
                )

                accumulator += (
                    result["state"][key]
                    .to(torch.float64)
                    * weight
                )

            new_state[key] = accumulator.to(
                global_value.dtype
            )

    # Global control variate update.
    #
    # c_global += M/N * avg(c_delta)
    participation_ratio = (
        num_selected
        / num_clients_total
    )

    new_c_global = {}

    for name, old_c in c_global.items():
        avg_c_delta = torch.zeros_like(
            old_c,
            dtype=torch.float64,
        )

        for result in client_results:
            avg_c_delta += (
                result["c_delta"][name]
                .to(torch.float64)
                * uniform_weight
            )

        new_c_global[name] = (
            old_c.to(torch.float64)
            + participation_ratio
            * avg_c_delta
        ).to(old_c.dtype)

    return new_state, new_c_global


def tensor_dict_norm(values):
    total = 0.0

    for value in values.values():
        total += (
            value.to(torch.float64)
            .pow(2)
            .sum()
            .item()
        )

    return total ** 0.5


@torch.no_grad()
def evaluate(
    model,
    testset,
    device,
    batch_size=256,
):
    loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(x)
        loss = criterion(logits, y)

        batch_n = y.size(0)

        total_loss += (
            loss.item()
            * batch_n
        )

        total_correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total_seen += batch_n

    return (
        total_loss / total_seen,
        total_correct / total_seen,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--partition-file",
        required=True,
    )

    parser.add_argument(
        "--schedule-file",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        default="track1/data",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.training_seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    trainset, testset = get_datasets(
        args.data_dir
    )

    partitions = load_partition(
        args.partition_file,
        args.num_clients,
    )

    schedule = load_schedule(
        args.schedule_file
    )

    global_model = (
        create_resnet18()
        .to(device)
    )

    trainable_keys = set(
        dict(
            global_model.named_parameters()
        ).keys()
    )

    # Global control variate c.
    c_global = zero_control(
        global_model
    )

    # Persistent client control variates c_i.
    #
    # Only allocate a client's control state when
    # that client is first selected.
    client_controls = {}

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "round",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "num_fit_clients",
        "mean_local_steps",
        "c_global_norm",
        "mean_c_delta_norm",
        "run_completed",
    ]

    with output.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        test_loss, test_acc = evaluate(
            global_model,
            testset,
            device,
        )

        print(
            f"Round 0 | "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f}"
        )

        for round_id in range(
            1,
            args.rounds + 1,
        ):
            selected = schedule[
                str(round_id)
            ]

            print(
                f"Round {round_id} | "
                f"selected={selected}"
            )

            global_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in global_model
                .state_dict()
                .items()
            }

            client_results = []

            for cid in selected:

                if cid not in client_controls:
                    client_controls[cid] = {
                        name: torch.zeros_like(
                            value
                        )
                        for name, value
                        in c_global.items()
                    }

                result = train_client(
                    cid=cid,
                    global_state=global_state,
                    c_global=c_global,
                    c_local=(
                        client_controls[cid]
                    ),
                    client_indices=(
                        partitions[cid]
                    ),
                    trainset=trainset,
                    device=device,
                    batch_size=(
                        args.batch_size
                    ),
                    local_epochs=(
                        args.local_epochs
                    ),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=(
                        args.weight_decay
                    ),
                    seed=(
                        args.training_seed
                        + round_id * 10000
                        + cid
                    ),
                )

                # Persist updated local control c_i.
                client_controls[cid] = (
                    result["c_plus"]
                )

                client_results.append(
                    result
                )

            new_state, c_global = (
                aggregate_scaffold(
                    global_state=global_state,
                    c_global=c_global,
                    client_results=(
                        client_results
                    ),
                    num_clients_total=(
                        args.num_clients
                    ),
                    trainable_keys=(
                        trainable_keys
                    ),
                )
            )

            global_model.load_state_dict(
                new_state
            )

            total_examples = sum(
                result["num_examples"]
                for result in client_results
            )

            train_loss = sum(
                result["train_loss"]
                * result["num_examples"]
                for result in client_results
            ) / total_examples

            train_acc = sum(
                result["train_accuracy"]
                * result["num_examples"]
                for result in client_results
            ) / total_examples

            mean_steps = sum(
                result["local_steps"]
                for result in client_results
            ) / len(client_results)

            c_global_norm = (
                tensor_dict_norm(
                    c_global
                )
            )

            mean_c_delta_norm = (
                sum(
                    tensor_dict_norm(
                        result["c_delta"]
                    )
                    for result
                    in client_results
                )
                / len(client_results)
            )

            test_loss, test_acc = evaluate(
                global_model,
                testset,
                device,
            )

            print(
                f"Round {round_id} | "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.4f} "
                f"steps={mean_steps:.1f} "
                f"c_global_norm={c_global_norm:.4f} "
                f"c_delta_norm={mean_c_delta_norm:.4f}"
            )

            writer.writerow(
                {
                    "round": round_id,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "num_fit_clients": (
                        len(selected)
                    ),
                    "mean_local_steps": (
                        mean_steps
                    ),
                    "c_global_norm": (
                        c_global_norm
                    ),
                    "mean_c_delta_norm": (
                        mean_c_delta_norm
                    ),
                    "run_completed": (
                        round_id
                        == args.rounds
                    ),
                }
            )

            f.flush()

    print("Saved:", output)


if __name__ == "__main__":
    main()
