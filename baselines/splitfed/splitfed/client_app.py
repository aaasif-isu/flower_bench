from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from splitfed.task import SplitFedResNet10, get_parameters, set_parameters, load_data, train, test


class SplitFedClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        train_loss, train_acc = train(
            self.model,
            self.trainloader,
            epochs=self.local_epochs,
        )

        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        loss, accuracy = test(self.model, self.testloader)

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy)},
        )


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])

    num_clients = int(context.run_config["num-clients"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    train_fraction = float(context.run_config["train-fraction"])

    model = SplitFedResNet10(num_classes=10)

    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_clients,
        batch_size=batch_size,
        train_fraction=train_fraction,
    )

    return SplitFedClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        local_epochs=local_epochs,
    ).to_client()


app = ClientApp(client_fn=client_fn)
