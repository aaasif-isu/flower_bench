import argparse
import csv
import gc
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


def zero_control(global_model):
    return {
        name: torch.zeros_like(
            param.detach().cpu()
        )
        for name, param
        in global_model.named_parameters()
        if param.requires_grad
    }


def train_client(
    cid,
    global_state,
    c_global,
    c_local,
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

    cg_device = {
        name: value.to(device)
        for name, value in c_global.items()
    }

    cl_device = {
        name: value.to(device)
        for name, value in c_local.items()
    }

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
    local_steps = 0

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

            n = y.size(0)

            total_loss += (
                loss.item() * n
            )

            total_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()

            total_seen += n

    if local_steps == 0:
        raise RuntimeError(
            f"Client {cid} performed zero local steps"
        )

    local_state = {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }

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

            c_delta[name] = (
                -c_global[name]
                - coef * delta
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
    num_selected = len(client_results)

    if num_selected == 0:
        raise RuntimeError(
            "No selected clients"
        )

    new_state = {}
    uniform_weight = 1.0 / num_selected

    for key, global_value in global_state.items():

        if not torch.is_floating_point(
            global_value
        ):
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
                    (
                        result["state"][key]
                        - global_state[key]
                    )
                    .to(torch.float64)
                    * uniform_weight
                )

            new_state[key] = (
                global_value.to(torch.float64)
                + avg_delta
            ).to(global_value.dtype)

        else:
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

    trainable_keys = set(
        dict(global_model.named_parameters()).keys()
    )

    c_global = zero_control(
        global_model
    )

    client_controls = {}

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
                    c_local=client_controls[cid],
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

                client_controls[cid] = {
                    name: (
                        client_controls[cid][name]
                        + result["c_delta"][name]
                    )
                    for name in c_global
                }

                results.append(
                    result
                )

            new_state, c_global = (
                aggregate_scaffold(
                    global_state=global_state,
                    c_global=c_global,
                    client_results=results,
                    num_clients_total=args.num_clients,
                    trainable_keys=trainable_keys,
                )
            )

            global_model.load_state_dict(
                new_state
            )

            total_examples = sum(
                result["num_examples"]
                for result in results
            )

            train_loss = sum(
                result["train_loss"]
                * result["num_examples"]
                for result in results
            ) / total_examples

            train_acc = sum(
                result["train_accuracy"]
                * result["num_examples"]
                for result in results
            ) / total_examples

            mean_steps = sum(
                result["local_steps"]
                for result in results
            ) / len(results)

            c_global_norm = tensor_dict_norm(
                c_global
            )

            mean_c_delta_norm = (
                sum(
                    tensor_dict_norm(
                        result["c_delta"]
                    )
                    for result in results
                )
                / len(results)
            )

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
                f"steps={mean_steps:.1f} "
                f"c_global_norm={c_global_norm:.4f} "
                f"c_delta_norm={mean_c_delta_norm:.4f} "
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

            del results
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
