import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from data import load_shakespeare
from model import create_shakespeare_lstm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_schedule(path):
    with open(path) as f:
        return json.load(f)["schedule"]


def train_client(
    global_state,
    dataset,
    device,
    batch_size,
    local_epochs,
    lr,
    momentum,
    weight_decay,
    seed,
):
    set_seed(seed)

    model = create_shakespeare_lstm().to(device)
    model.load_state_dict(global_state)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
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

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            # Keeps recurrent training stable.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
            )

            optimizer.step()

            n = y.size(0)

            total_loss += (
                loss.item() * n
            )

            total_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            total_seen += n

    state = {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }

    return (
        state,
        total_seen,
        total_loss / total_seen,
        total_correct / total_seen,
    )


def fedavg(results):
    total_examples = sum(
        result[1]
        for result in results
    )

    first = results[0][0]

    averaged = {}

    for key, value in first.items():

        if not torch.is_floating_point(value):
            averaged[key] = value.clone()
            continue

        accumulator = torch.zeros_like(
            value,
            dtype=torch.float64,
        )

        for state, n, _, _ in results:
            accumulator += (
                state[key].to(torch.float64)
                * (n / total_examples)
            )

        averaged[key] = accumulator.to(
            value.dtype
        )

    return averaged


@torch.no_grad()
def evaluate(
    model,
    dataset,
    device,
    batch_size=256,
):
    loader = DataLoader(
        dataset,
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

        n = y.size(0)

        total_loss += (
            loss.item() * n
        )

        total_correct += (
            logits.argmax(dim=1) == y
        ).sum().item()

        total_seen += n

    loss = total_loss / total_seen
    accuracy = total_correct / total_seen

    # Natural-log cross entropy -> perplexity.
    perplexity = math.exp(
        min(loss, 50.0)
    )

    return (
        loss,
        accuracy,
        perplexity,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--leaf-root",
        default=(
            "external/leaf/data/"
            "shakespeare"
        ),
    )

    parser.add_argument(
        "--client-file",
        default=(
            "track1/shakespeare/"
            "eligible_clients.txt"
        ),
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
        default=32,
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
        "--server-lr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=1e-9,
    )

    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    set_seed(
        args.training_seed
    )

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

    users, train_clients, test_clients = (
        load_shakespeare(
            args.leaf_root,
            args.client_file,
        )
    )

    print(
        "Clients:",
        len(users),
    )

    print(
        "Train examples:",
        sum(
            len(dataset)
            for dataset in train_clients
        ),
    )

    print(
        "Test examples:",
        sum(
            len(dataset)
            for dataset in test_clients
        ),
    )

    schedule = load_schedule(
        args.schedule_file
    )

    global_model = (
        create_shakespeare_lstm()
        .to(device)
    )

    global_test = ConcatDataset(
        test_clients
    )

    output = Path(
        args.output
    )

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
        "test_perplexity",
        "num_fit_clients",
        "num_fit_examples",
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

        test_loss, test_acc, test_ppl = evaluate(
            global_model,
            global_test,
            device,
        )

        print(
            f"Round 0 | "
            f"test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f} "
            f"ppl={test_ppl:.2f}"
        )

        server_m = None
        server_v = None

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

            results = []

            for cid in selected:
                result = train_client(
                    global_state=global_state,
                    dataset=(
                        train_clients[cid]
                    ),
                    device=device,
                    batch_size=(
                        args.batch_size
                    ),
                    local_epochs=(
                        args.local_epochs
                    ),
                    lr=args.lr,
                    momentum=(
                        args.momentum
                    ),
                    weight_decay=(
                        args.weight_decay
                    ),
                    seed=(
                        args.training_seed
                        + round_id * 10000
                        + cid
                    ),
                )

                results.append(
                    result
                )

            averaged_state = fedavg(
                results
            )

            current_state = {
                key: value.detach().cpu().clone()
                for key, value
                in global_model.state_dict().items()
            }

            trainable_keys = set(
                dict(
                    global_model.named_parameters()
                ).keys()
            )

            if server_m is None:
                server_m = {}
                server_v = {}

                for key in trainable_keys:
                    server_m[key] = torch.zeros_like(
                        current_state[key]
                    )
                    server_v[key] = torch.zeros_like(
                        current_state[key]
                    )

            new_state = {}

            for key, current_value in current_state.items():

                if key not in trainable_keys:
                    new_state[key] = (
                        averaged_state[key].clone()
                    )
                    continue

                delta = (
                    averaged_state[key]
                    - current_value
                )

                server_m[key] = (
                    args.beta1
                    * server_m[key]
                    + (1.0 - args.beta1)
                    * delta
                )

                server_v[key] = (
                    args.beta2
                    * server_v[key]
                    + (1.0 - args.beta2)
                    * (delta * delta)
                )

                eta_norm = (
                    args.server_lr
                    * (
                        1.0
                        - args.beta2 ** (round_id + 1)
                    ) ** 0.5
                    / (
                        1.0
                        - args.beta1 ** (round_id + 1)
                    )
                )

                new_state[key] = (
                    current_value
                    + eta_norm
                    * server_m[key]
                    / (
                        torch.sqrt(server_v[key])
                        + args.tau
                    )
                )

            global_model.load_state_dict(
                new_state
            )

            total_examples = sum(
                result[1]
                for result in results
            )

            train_loss = sum(
                result[2] * result[1]
                for result in results
            ) / total_examples

            train_acc = sum(
                result[3] * result[1]
                for result in results
            ) / total_examples

            if (
                round_id % args.eval_interval == 0
                or round_id == args.rounds
            ):
                (
                    test_loss,
                    test_acc,
                    test_ppl,
                ) = evaluate(
                    global_model,
                    global_test,
                    device,
                )
            else:
                test_loss = float("nan")
                test_acc = float("nan")
                test_ppl = float("nan")

            print(
                f"Round {round_id} | "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.4f} "
                f"ppl={test_ppl:.2f} "
                f"examples={total_examples}"
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
                    "test_perplexity": (
                        test_ppl
                    ),
                    "num_fit_clients": (
                        len(selected)
                    ),
                    "num_fit_examples": (
                        total_examples
                    ),
                    "run_completed": (
                        round_id
                        == args.rounds
                    ),
                }
            )

            f.flush()

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
