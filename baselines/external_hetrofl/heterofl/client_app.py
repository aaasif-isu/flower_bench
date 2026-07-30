import flwr as fl
from flwr.common import Context

from .model import create_model
from .task import get_device, load_datasets, set_parameters, get_parameters, train


class HeteroFLClient(fl.client.NumPyClient):
    def __init__(self, cid, num_clients, batch_size):
        self.cid = int(cid)
        self.num_clients = int(num_clients)
        self.batch_size = int(batch_size)
        self.device = get_device()

        self.trainloaders, _ = load_datasets(
            num_clients=self.num_clients,
            batch_size=self.batch_size,
        )

    def fit(self, parameters, config):
        model_rate = float(config.get("model_rate", 1.0))
        local_epochs = int(config.get("local_epochs", 1))

        model = create_model(model_rate=model_rate)
        set_parameters(model, parameters)

        train_loss, train_acc = train(
            model=model,
            trainloader=self.trainloaders[self.cid],
            epochs=local_epochs,
            device=self.device,
        )

        return (
            get_parameters(model),
            len(self.trainloaders[self.cid].dataset),
            {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "model_rate": float(model_rate),
            },
        )


def client_fn(context: Context):
    cid = context.node_config["partition-id"]

    num_clients = int(context.run_config.get("num-clients", 10))
    batch_size = int(context.run_config.get("batch-size", 64))

    return HeteroFLClient(
        cid=cid,
        num_clients=num_clients,
        batch_size=batch_size,
    ).to_client()


app = fl.client.ClientApp(client_fn=client_fn)
