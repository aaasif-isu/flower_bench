import os
from typing import List, Tuple

import pandas as pd
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from splitfed.task import SplitFedResNet10, get_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)

    accuracy = sum(num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics) / total_examples

    return {"accuracy": accuracy}


def weighted_fit_average(results):
    total_examples = sum(fit_res.num_examples for _, fit_res in results)

    train_loss = sum(
        fit_res.num_examples * fit_res.metrics.get("train_loss", 0.0)
        for _, fit_res in results
    ) / total_examples

    train_acc = sum(
        fit_res.num_examples * fit_res.metrics.get("train_accuracy", 0.0)
        for _, fit_res in results
    ) / total_examples

    return train_loss, train_acc


class CsvFedAvg(FedAvg):
    def __init__(self, method_name: str, csv_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = method_name
        self.csv_path = csv_path
        self.rows = []
        self.fit_cache = {}

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if results:
            train_loss, train_acc = weighted_fit_average(results)
            self.fit_cache[server_round] = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
            }

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        fit_metrics = self.fit_cache.get(server_round, {})

        row = {
            "method": self.method_name,
            "dataset": "CIFAR10",
            "model": "ResNet10",
            "round": server_round,
            "train_loss": fit_metrics.get("train_loss"),
            "train_accuracy": fit_metrics.get("train_accuracy"),
            "test_loss": loss_aggregated,
            "test_accuracy": metrics_aggregated.get("accuracy") if metrics_aggregated else None,
        }

        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.csv_path, index=False)

        print(f"[CSV] Saved round {server_round} metrics to {self.csv_path}")
        print(row)

        return loss_aggregated, metrics_aggregated


def server_fn(context: Context):
    num_rounds = int(context.run_config["num-server-rounds"])
    num_clients = int(context.run_config["num-clients"])
    clients_per_round = int(context.run_config.get("clients-per-round", num_clients))
    method_name = str(context.run_config["method"])
    csv_path = str(context.run_config["csv-path"])

    model = SplitFedResNet10(num_classes=10)
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = CsvFedAvg(
        method_name=method_name,
        csv_path=csv_path,
        fraction_fit=clients_per_round / num_clients,
        fraction_evaluate=clients_per_round / num_clients,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=clients_per_round,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
