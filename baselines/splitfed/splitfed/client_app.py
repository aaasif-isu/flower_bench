import time

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from splitfed.task import SplitFedResNet10, get_parameters, set_parameters, load_data, train, test


def arrays_size_mb(arrays):
    total = 0
    for arr in arrays:
        if hasattr(arr, "nbytes"):
            total += arr.nbytes
        elif hasattr(arr, "numel") and hasattr(arr, "element_size"):
            total += arr.numel() * arr.element_size()
    return total / (1024 * 1024)


class SplitFedClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs

        # Smashed activation shape after client side: (batch_size, 64, 32, 32) float32
        # This is what gets sent to server in true split learning
        self._batch_size = trainloader.batch_size
        self._smashed_mb_per_batch = (self._batch_size * 64 * 32 * 32 * 4) / (1024 * 1024)
        self._grad_mb_per_batch = self._smashed_mb_per_batch  # same size

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        fit_start = time.perf_counter()
        fit_param_down_mb = arrays_size_mb(parameters)
        set_parameters(self.model, parameters)

        train_start = time.perf_counter()
        train_loss, train_acc = train(self.model, self.trainloader, epochs=self.local_epochs)
        client_train_time_sec = time.perf_counter() - train_start

        out_params = get_parameters(self.model)
        fit_param_up_mb = arrays_size_mb(out_params)
        client_fit_total_time_sec = time.perf_counter() - fit_start

        # Calculate smashed data communication (what would happen in true split learning)
        # Each forward pass: (batch_size, 64, 32, 32) activations sent client→server
        # Each backward pass: same-shape gradients sent server→client
        # Labels also sent client→server once per batch
        num_train_samples = len(self.trainloader.dataset)
        num_batches = len(self.trainloader)  # per epoch
        total_batches = num_batches * self.local_epochs

        smashed_forward_mb = self._smashed_mb_per_batch * total_batches
        smashed_backward_mb = self._grad_mb_per_batch * total_batches
        # Labels: int64 (8 bytes) per sample, per epoch
        label_mb = (num_train_samples * self.local_epochs * 8) / (1024 * 1024)

        return (
            out_params,
            num_train_samples,
            {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "client_train_time_sec": float(client_train_time_sec),
                "client_fit_total_time_sec": float(client_fit_total_time_sec),
                "fit_param_down_mb": float(fit_param_down_mb),
                "fit_param_up_mb": float(fit_param_up_mb),
                "fit_param_total_comm_mb": float(fit_param_down_mb + fit_param_up_mb),
                # SplitFed-specific communication
                "smashed_forward_mb": float(smashed_forward_mb),
                "smashed_backward_mb": float(smashed_backward_mb),
                "label_mb": float(label_mb),
            },
        )

    def evaluate(self, parameters, config):
        eval_start = time.perf_counter()
        eval_param_down_mb = arrays_size_mb(parameters)
        set_parameters(self.model, parameters)

        loss, accuracy = test(self.model, self.testloader)
        client_eval_time_sec = time.perf_counter() - eval_start

        # Eval smashed comm: forward pass only (no backward needed for inference)
        num_eval_samples = len(self.testloader.dataset)
        num_eval_batches = len(self.testloader)
        eval_smashed_forward_mb = self._smashed_mb_per_batch * num_eval_batches
        eval_label_mb = (num_eval_samples * 8) / (1024 * 1024)

        return (
            float(loss),
            num_eval_samples,
            {
                "accuracy": float(accuracy),
                "client_eval_time_sec": float(client_eval_time_sec),
                "eval_param_down_mb": float(eval_param_down_mb),
                "eval_param_up_mb": 0.0,
                "eval_param_total_comm_mb": float(eval_param_down_mb),
                "eval_smashed_forward_mb": float(eval_smashed_forward_mb),
                "eval_label_mb": float(eval_label_mb),
            },
        )


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    num_clients = int(context.run_config["num-clients"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    train_fraction = float(context.run_config["train-fraction"])

    dataset_name = str(context.run_config.get("dataset-name", "cifar10")).lower()

    if dataset_name in ["leaf_femnist", "femnist"]:
        input_channels = 1
        num_classes = 62
    else:
        input_channels = 3
        num_classes = 10

    model = SplitFedResNet10(num_classes=num_classes, input_channels=input_channels)
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_clients,
        batch_size=batch_size,
        dataset_name=dataset_name,
        partition=str(context.run_config.get("partition", "niid")),
        leaf_root=str(context.run_config.get("leaf-root", "/lustre/hdd/LAS/jannesar-lab/aadishah/flower_bench/external/leaf/data/femnist")),
        seed=int(context.run_config.get("seed", 0)),
        max_iid_source_clients=context.run_config.get("max-iid-source-clients", None),
        train_fraction=train_fraction,
    )

    return SplitFedClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        local_epochs=local_epochs,
    ).to_client()


app = ClientApp(client_fn=client_fn)
