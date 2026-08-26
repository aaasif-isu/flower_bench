import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset

from data import load_femnist
from model import create_femnist_cnn


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

    model = create_femnist_cnn().to(device)
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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            n = y.size(0)

            total_loss += loss.item() * n
            total_correct += (
                logits.argmax(1) == y
            ).sum().item()
            total_seen += n

    state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    return (
        state,
        total_seen,
        total_loss / total_seen,
        total_correct / total_seen,
    )


def fedavg(results):
    total_examples = sum(r[1] for r in results)

    first = results[0][0]
    averaged = {}

    for key, value in first.items():
        if not torch.is_floating_point(value):
            averaged[key] = value.clone()
            continue

        acc = torch.zeros_like(
            value,
            dtype=torch.float64,
        )

        for state, n, _, _ in results:
            acc += (
                state[key].to(torch.float64)
                * (n / total_examples)
            )

        averaged[key] = acc.to(value.dtype)

    return averaged


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=256):
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        n = y.size(0)

        total_loss += loss.item() * n
        total_correct += (
            logits.argmax(1) == y
        ).sum().item()
        total_seen += n

    return (
        total_loss / total_seen,
        total_correct / total_seen,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--leaf-root",
        default="external/leaf/data/femnist",
    )
    parser.add_argument(
        "--client-file",
        default="track1/femnist/eligible_clients.txt",
    )
    parser.add_argument("--schedule-file", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.training_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    users, train_clients, test_clients = load_femnist(
        args.leaf_root,
        args.client_file,
    )

    print("Clients:", len(users))
    print(
        "Train examples:",
        sum(len(d) for d in train_clients),
    )
    print(
        "Test examples:",
        sum(len(d) for d in test_clients),
    )

    schedule = load_schedule(args.schedule_file)

    global_model = create_femnist_cnn().to(device)

    global_test = ConcatDataset(test_clients)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "round",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "num_fit_clients",
        "num_fit_examples",
        "run_completed",
    ]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        test_loss, test_acc = evaluate(
            global_model,
            global_test,
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
            selected = schedule[str(round_id)]

            print(
                f"Round {round_id} | "
                f"selected={selected}"
            )

            global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            }

            results = []

            for cid in selected:
                result = train_client(
                    global_state=global_state,
                    dataset=train_clients[cid],
                    device=device,
                    batch_size=args.batch_size,
                    local_epochs=args.local_epochs,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    seed=(
                        args.training_seed
                        + round_id * 10000
                        + cid
                    ),
                )

                results.append(result)

            new_state = fedavg(results)
            global_model.load_state_dict(new_state)

            total_examples = sum(r[1] for r in results)

            train_loss = sum(
                r[2] * r[1] for r in results
            ) / total_examples

            train_acc = sum(
                r[3] * r[1] for r in results
            ) / total_examples

            if (
                round_id % args.eval_interval == 0
                or round_id == args.rounds
            ):
                test_loss, test_acc = evaluate(
                    global_model,
                    global_test,
                    device,
                )
            else:
                test_loss = float("nan")
                test_acc = float("nan")

            print(
                f"Round {round_id} | "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.4f} "
                f"examples={total_examples}"
            )

            writer.writerow(
                {
                    "round": round_id,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "num_fit_clients": len(selected),
                    "num_fit_examples": total_examples,
                    "run_completed": (
                        round_id == args.rounds
                    ),
                }
            )

            f.flush()

    print("Saved:", output)


if __name__ == "__main__":
    main()
