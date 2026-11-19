import csv
import os
import time
from collections import OrderedDict

import flwr as fl
import torch
from flwr.common import Context, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import Strategy

from .model import create_model
from .split import extract_submodel_state, average_submodels
from .strategy import sample_model_rate
from .task import (
    get_device,
    get_parameters,
    load_datasets,
    parameters_to_state_dict,
    set_parameters,
    state_dict_to_parameters,
    test,
)


class HeteroFLStrategy(Strategy):
    def __init__(
        self,
        num_clients,
        clients_per_round,
        num_rounds,
        local_epochs,
        batch_size,
        model_mode,
        csv_path,
    ):
        self.num_clients = int(num_clients)
        self.clients_per_round = int(clients_per_round)
        self.num_rounds = int(num_rounds)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.model_mode = str(model_mode)
        self.csv_path = str(csv_path)

        self.global_model = create_model(model_rate=1.0)
        self.global_parameters = ndarrays_to_parameters(get_parameters(self.global_model))

        self.device = get_device()
        _, self.testloader = load_datasets(
            num_clients=self.num_clients,
            batch_size=self.batch_size,
        )

        self.round_start_time = None
        self.cumulative_time = 0.0

        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
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

    def initialize_parameters(self, client_manager):
        return self.global_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()

        available_clients = client_manager.all()
        client_ids = list(available_clients.keys())

        if len(client_ids) < self.clients_per_round:
            selected_ids = client_ids
        else:
            selected_ids = client_ids[: self.clients_per_round]

        fit_ins = []

        global_state = self.global_model.state_dict()

        for cid in selected_ids:
            client_proxy = available_clients[cid]
            model_rate, model_level = sample_model_rate(self.model_mode)

            local_model = create_model(model_rate=model_rate)
            local_state = local_model.state_dict()

            sub_state = extract_submodel_state(global_state, local_state)
            sub_parameters = ndarrays_to_parameters(state_dict_to_parameters(sub_state))

            config = {
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "model_rate": model_rate,
                "model_level": model_level,
            }

            fit_ins.append((client_proxy, FitIns(sub_parameters, config)))

        return fit_ins

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return self.global_parameters, {}

        client_states = []
        train_losses = []
        train_accs = []
        model_rates = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)

            model_rate = float(fit_res.metrics.get("model_rate", 1.0))
            local_model = create_model(model_rate=model_rate)
            client_state = parameters_to_state_dict(local_model, arrays)
            client_states.append(client_state)

            if "train_loss" in fit_res.metrics:
                train_losses.append(float(fit_res.metrics["train_loss"]))

            if "train_accuracy" in fit_res.metrics:
                train_accs.append(float(fit_res.metrics["train_accuracy"]))

            model_rates.append(model_rate)

        new_global_state = average_submodels(
            self.global_model.state_dict(),
            client_states,
        )

        self.global_model.load_state_dict(new_global_state)
        self.global_parameters = ndarrays_to_parameters(
            state_dict_to_parameters(new_global_state)
        )

        test_loss, test_acc = test(
            model=self.global_model,
            testloader=self.testloader,
            device=self.device,
        )

        round_time = time.time() - self.round_start_time if self.round_start_time else 0.0
        self.cumulative_time += round_time

        train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        train_acc = sum(train_accs) / len(train_accs) if train_accs else 0.0
        avg_model_rate = sum(model_rates) / len(model_rates) if model_rates else 0.0

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "ExternalHeteroFL",
                "CIFAR10",
                "HeteroCifarCNN",
                server_round,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                len(results),
                len(failures),
                self.model_mode,
                avg_model_rate,
                round_time,
                self.cumulative_time,
            ])

        print(
            f">>> Round {server_round}: "
            f"train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}, "
            f"avg_model_rate={avg_model_rate:.4f}, "
            f"failures={len(failures)}, "
            f"round_time={round_time:.2f}s"
        )

        return self.global_parameters, {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


def server_fn(context: Context):
    run_config = context.run_config

    strategy = HeteroFLStrategy(
        num_clients=int(run_config.get("num-clients", 10)),
        clients_per_round=int(run_config.get("clients-per-round", 5)),
        num_rounds=int(run_config.get("num-server-rounds", 3)),
        local_epochs=int(run_config.get("local-epochs", 1)),
        batch_size=int(run_config.get("batch-size", 64)),
        model_mode=str(run_config.get("model-mode", "a1-b1-c1")),
        csv_path=str(run_config.get("csv-path", "results/external_hetrofl_cifar10_smoke.csv")),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(
            num_rounds=int(run_config.get("num-server-rounds", 3))
        ),
    )


app = fl.server.ServerApp(server_fn=server_fn)
