"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import fedmeta.client as client
from fedmeta.dataset import load_datasets
from fedmeta.fedmeta_client_manager import FedmetaClientManager
from fedmeta.strategy import weighted_average
from fedmeta.utils import plot_from_pkl, save_graph_params


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.

        algo : FedAvg, FedAvg(Meta), FedMeta(MAML), FedMeta(Meta-SGD)
        data : Femnist, Shakespeare
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, valloaders, _ = load_datasets(config=cfg.data, path=cfg.path)

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        learning_rate=cfg.algo[cfg.data.data].alpha,
        model=cfg.data.model,
        gradient_step=cfg.data.gradient_step,
    )

    # prepare strategy function
    strategy = instantiate(
        cfg.strategy,
        evaluate_metrics_aggregation_fn=weighted_average,
        alpha=cfg.algo[cfg.data.data].alpha,
        beta=cfg.algo[cfg.data.data].beta,
        data=cfg.data.data,
        algo=cfg.algo.algo,
    )

    # Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(trainloaders["sup"]),
        config=fl.server.ServerConfig(num_rounds=cfg.data.num_rounds),
        client_resources={
            "num_cpus": cfg.data.client_resources.num_cpus,
            "num_gpus": cfg.data.client_resources.num_gpus,
        },
        client_manager=FedmetaClientManager(valid_client=len(valloaders["qry"])),
        strategy=strategy,
    )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir

    print("................")
    print(history)
    output_path = HydraConfig.get().runtime.output_dir

    data_params = {
        "algo": cfg.algo.algo,
        "data": cfg.data.data,
        "loss": history.losses_distributed,
        "accuracy": history.metrics_distributed,
        "path": output_path,
    }

    save_graph_params(data_params)
    plot_from_pkl(directory=output_path)
    print("................")

def plot_round_metrics(
    csv_path="fedmeta_cifar10_resnet10_round_metrics.csv",
    loss_png="fedmeta_round_loss_graph.png",
    accuracy_png="fedmeta_round_accuracy_graph.png",
):
    rounds = []
    query_losses = []
    test_losses = []
    test_accuracies = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rounds.append(int(row["round"]))
            query_losses.append(float(row["avg_client_query_loss"]))
            test_losses.append(float(row["test_loss"]))
            test_accuracies.append(float(row["test_accuracy"]))

    # Loss graph
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, query_losses, marker="o", label="Avg Client Query Loss")
    plt.plot(rounds, test_losses, marker="o", label="Centralized Test Loss")
    plt.title("FedMeta CIFAR-10 ResNet10: Loss per Round")
    plt.xlabel("Global Round")
    plt.ylabel("Loss")
    plt.xticks(rounds)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png, dpi=200)
    plt.close()

    # Accuracy graph
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, test_accuracies, marker="o")
    plt.title("FedMeta CIFAR-10 ResNet10: Accuracy per Round")
    plt.xlabel("Global Round")
    plt.ylabel("Centralized Test Accuracy")
    plt.xticks(rounds)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_png, dpi=200)
    plt.close()

    print(f"Saved loss graph to: {loss_png}")
    print(f"Saved accuracy graph to: {accuracy_png}")

if __name__ == "__main__":
    main()
