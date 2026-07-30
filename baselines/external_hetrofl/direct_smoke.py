import csv
import os
import time

import torch

from heterofl.model import create_model
from heterofl.split import extract_submodel_state, average_submodels
from heterofl.task import (
    get_device,
    load_datasets,
    set_parameters,
    get_parameters,
    parameters_to_state_dict,
    state_dict_to_parameters,
    train,
    test,
)

MODEL_RATES = [1.0, 0.5]


def main():
    num_clients = 2
    clients_per_round = 2
    num_rounds = 1
    local_epochs = 1
    batch_size = 64
    csv_path = "results/external_hetrofl_direct_smoke_1r.csv"

    os.makedirs("results", exist_ok=True)

    device = get_device()
    print("device:", device)

    trainloaders, testloader = load_datasets(num_clients, batch_size)

    global_model = create_model(model_rate=1.0)
    global_state = global_model.state_dict()

    with open(csv_path, "w", newline="") as f:
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
            "avg_model_rate",
            "round_wall_time_sec",
        ])

    for rnd in range(1, num_rounds + 1):
        start = time.time()

        client_states = []
        train_losses = []
        train_accs = []
        model_rates = []

        for cid in range(clients_per_round):
            model_rate = MODEL_RATES[cid % len(MODEL_RATES)]
            model_rates.append(model_rate)

            local_model = create_model(model_rate=model_rate)
            local_state = local_model.state_dict()

            sub_state = extract_submodel_state(global_state, local_state)
            local_model.load_state_dict(sub_state, strict=True)

            train_loss, train_acc = train(
                model=local_model,
                trainloader=trainloaders[cid],
                epochs=local_epochs,
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

        row = [
            "ExternalHeteroFL-Direct",
            "CIFAR10",
            "HeteroCifarCNN",
            rnd,
            sum(train_losses) / len(train_losses),
            sum(train_accs) / len(train_accs),
            test_loss,
            test_acc,
            clients_per_round,
            0,
            sum(model_rates) / len(model_rates),
            round_time,
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(
            f">>> Round {rnd}: "
            f"train_acc={row[5]:.4f}, test_acc={test_acc:.4f}, "
            f"avg_model_rate={row[10]:.4f}, round_time={round_time:.2f}s"
        )

    print("wrote", csv_path)


if __name__ == "__main__":
    main()
