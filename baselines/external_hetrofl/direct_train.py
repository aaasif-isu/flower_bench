import argparse
import csv
import os
import time

from heterofl.model import create_model
from heterofl.split import extract_submodel_state, average_submodels
from heterofl.task import get_device, load_datasets, train, test


MODEL_RATE_MAP = {
    "a": 1.0,
    "b": 0.5,
    "c": 0.25,
    "d": 0.125,
    "e": 0.0625,
}


def parse_model_mode(model_mode):
    levels = []
    for part in model_mode.split("-"):
        level = part[0]
        count = int(part[1:])
        levels.extend([level] * count)
    return levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-mode", type=str, default="a1-b1-c1")
    parser.add_argument("--csv-path", type=str, default="results/external_hetrofl_direct_3r.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    device = get_device()
    print("device:", device)

    trainloaders, testloader = load_datasets(args.num_clients, args.batch_size)

    global_model = create_model(model_rate=1.0)
    global_state = global_model.state_dict()

    mode_levels = parse_model_mode(args.model_mode)

    with open(args.csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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
            "model_mode",
            "avg_model_rate",
            "round_wall_time_sec",
            "cumulative_round_wall_time_sec",
        ])

    cumulative_time = 0.0

    for rnd in range(1, args.num_rounds + 1):
        start = time.time()

        client_states = []
        train_losses = []
        train_accs = []
        model_rates = []

        selected_clients = [
            (rnd - 1 + i) % args.num_clients
            for i in range(args.clients_per_round)
        ]

        for i, cid in enumerate(selected_clients):
            level = mode_levels[(rnd + i - 1) % len(mode_levels)]
            model_rate = MODEL_RATE_MAP[level]
            model_rates.append(model_rate)

            local_model = create_model(model_rate=model_rate)
            local_state = local_model.state_dict()

            sub_state = extract_submodel_state(global_state, local_state)
            local_model.load_state_dict(sub_state, strict=True)

            train_loss, train_acc = train(
                model=local_model,
                trainloader=trainloaders[cid],
                epochs=args.local_epochs,
                device=device,
            )

            client_states.append(local_model.state_dict())
            train_losses.append(train_loss)
            train_accs.append(train_acc)

        new_global_state = average_submodels(global_state, client_states)
        global_model.load_state_dict(new_global_state, strict=True)
        global_state = global_model.state_dict()

        test_loss, test_acc = test(global_model, testloader, device)

        round_time = time.time() - start
        cumulative_time += round_time

        train_loss_mean = sum(train_losses) / len(train_losses)
        train_acc_mean = sum(train_accs) / len(train_accs)
        avg_model_rate = sum(model_rates) / len(model_rates)

        with open(args.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ExternalHeteroFL-Direct",
                "CIFAR10",
                "HeteroResNet10_sBN_Scaler",
                rnd,
                train_loss_mean,
                train_acc_mean,
                test_loss,
                test_acc,
                len(selected_clients),
                0,
                args.model_mode,
                avg_model_rate,
                round_time,
                cumulative_time,
            ])

        print(
            f">>> Round {rnd}: "
            f"train_acc={train_acc_mean:.4f}, "
            f"test_acc={test_acc:.4f}, "
            f"avg_model_rate={avg_model_rate:.4f}, "
            f"round_time={round_time:.2f}s"
        )

    print("wrote", args.csv_path)


if __name__ == "__main__":
    main()
