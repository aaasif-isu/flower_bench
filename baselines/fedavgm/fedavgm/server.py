"""Define the Flower Server and function to instantiate it."""

from keras.utils import to_categorical
from omegaconf import DictConfig


def get_on_fit_config(config: DictConfig):
    """Generate the function for config.

    The config dict is sent to the client fit() method.
    """

    def fit_config_fn(server_round: int):  # pylint: disable=unused-argument
        return {
            "local_epochs": config.local_epochs,
            "batch_size": config.batch_size,
        }

    return fit_config_fn


def get_evaluate_fn(model, x_test, y_test, num_rounds, num_classes):
    """Generate the function for server global model evaluation.

    The evaluate_fn runs after global model aggregation.
    This version evaluates after EVERY round, not only the final round.
    """

    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    def evaluate_fn(
        server_round: int, parameters, config
    ):  # pylint: disable=unused-argument

        # Skip initial evaluation before round 1
        if server_round == 0:
            return None

        # Set global model weights after aggregation
        model.set_weights(parameters)

        # Evaluate global model on centralized test set
        loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=False)

        # Print every round so it appears in the log
        print(
            f">>> Round {server_round}: "
            f"test_loss={loss:.4f}, test_accuracy={accuracy:.4f}",
            flush=True,
        )

        # Return metrics so Flower stores them in history
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate_fn