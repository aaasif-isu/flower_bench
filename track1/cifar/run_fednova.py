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


def train_client(
    global_state,
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

    # FedNova normalization state.
    #
    # Matches the recurrence in the reference implementation:
    #
    # if momentum != 0:
    #     local_counter = local_counter * momentum + 1
    #     local_normalizing_vec += local_counter
    #
    # otherwise:
    #     local_normalizing_vec += 1
    local_counter = 0.0
    local_normalizing_vec = 0.0
    local_steps = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            if momentum != 0:
                local_counter = (
                    local_counter * momentum + 1.0
                )
                local_normalizing_vec += local_counter
            else:
                local_normalizing_vec += 1.0

            local_steps += 1

            batch_n = y.size(0)

            total_loss += loss.item() * batch_n
            total_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()
            total_seen += batch_n

    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    return (
        state,
        total_seen,
        total_loss / total_seen,
        total_correct / total_seen,
        local_normalizing_vec,
        local_steps,
    )


def fednova(
    global_state,
    client_results,
    trainable_keys,
):
    """
    Sequential equivalent of FedNova normalized aggregation.

    Reference implementation:

        tau_eff = sum_i p_i * a_i

        scaled_update_i =
            p_i * (tau_eff / a_i) * update_i

    where

        update_i = local_model_i - global_model

    The new global model is

        w_new = w_global + sum_i scaled_update_i

    BatchNorm/non-trainable floating buffers use ordinary
    weighted averaging. Integer buffers are copied from the
    first client, consistent with our other Track-1 runners.
    """

    total_examples = sum(
        result[1] for result in client_results
    )

    weights = [
        result[1] / total_examples
        for result in client_results
    ]

    normalizers = [
        result[4]
        for result in client_results
    ]

    tau_eff = sum(
        weight * a_i
        for weight, a_i in zip(
            weights,
            normalizers,
        )
    )

    if tau_eff <= 0:
        raise RuntimeError(
            f"Invalid FedNova tau_eff: {tau_eff}"
        )

    new_state = {}

    first_state = client_results[0][0]

    for key, global_value in global_state.items():
        # Integer buffers, e.g. num_batches_tracked.
        if not torch.is_floating_point(global_value):
            new_state[key] = first_state[key].clone()
            continue

        # FedNova normalization applies to trainable model
        # parameters.
        if key in trainable_keys:
            accumulator = torch.zeros_like(
                global_value,
                dtype=torch.float64,
            )

            global64 = global_value.to(
                torch.float64
            )

            for (
                result,
                weight,
                a_i,
            ) in zip(
                client_results,
                weights,
                normalizers,
            ):
                local_state = result[0]

                delta = (
                    local_state[key].to(
                        torch.float64
                    )
                    - global64
                )

                scale = (
                    weight
                    * tau_eff
                    / a_i
                )

                accumulator += scale * delta

            new_state[key] = (
                global64 + accumulator
            ).to(global_value.dtype)

        else:
            # BatchNorm running_mean/running_var and other
            # floating buffers use ordinary FedAvg.
            accumulator = torch.zeros_like(
                global_value,
                dtype=torch.float64,
            )

            for result, weight in zip(
                client_results,
                weights,
            ):
                accumulator += (
                    result[0][key].to(
                        torch.float64
                    )
                    * weight
                )

            new_state[key] = accumulator.to(
                global_value.dtype
            )

    return new_state, tau_eff


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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_n = y.size(0)

        total_loss += loss.item() * batch_n
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

    global_model = create_resnet18().to(
        device
    )

    trainable_keys = set(
        dict(
            global_model.named_parameters()
        ).keys()
    )

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
        "tau_eff",
        "mean_local_normalizer",
        "mean_local_steps",
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
                result = train_client(
                    global_state=global_state,
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

                client_results.append(
                    result
                )

            new_state, tau_eff = fednova(
                global_state=global_state,
                client_results=(
                    client_results
                ),
                trainable_keys=(
                    trainable_keys
                ),
            )

            global_model.load_state_dict(
                new_state
            )

            total_examples = sum(
                x[1]
                for x in client_results
            )

            train_loss = sum(
                x[2] * x[1]
                for x in client_results
            ) / total_examples

            train_acc = sum(
                x[3] * x[1]
                for x in client_results
            ) / total_examples

            mean_normalizer = (
                sum(
                    x[4]
                    for x in client_results
                )
                / len(client_results)
            )

            mean_steps = (
                sum(
                    x[5]
                    for x in client_results
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
                f"tau_eff={tau_eff:.4f} "
                f"a_i={mean_normalizer:.4f} "
                f"steps={mean_steps:.1f}"
            )

            writer.writerow(
                {
                    "round": round_id,
                    "train_loss": (
                        train_loss
                    ),
                    "train_accuracy": (
                        train_acc
                    ),
                    "test_loss": (
                        test_loss
                    ),
                    "test_accuracy": (
                        test_acc
                    ),
                    "num_fit_clients": (
                        len(selected)
                    ),
                    "tau_eff": tau_eff,
                    "mean_local_normalizer": (
                        mean_normalizer
                    ),
                    "mean_local_steps": (
                        mean_steps
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
