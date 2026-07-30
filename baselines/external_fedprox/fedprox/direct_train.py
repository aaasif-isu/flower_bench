import os

# Set before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------
# Make the existing Flower FedAvgM package importable.
#
# direct_train.py:
# flower_bench/baselines/external_fedprox/fedprox/direct_train.py
#
# parents[3]:
# flower_bench/
# ---------------------------------------------------------------------
FLOWER_ROOT = Path(__file__).resolve().parents[3]
FEDAVGM_ROOT = FLOWER_ROOT / "baselines" / "fedavgm"

if str(FEDAVGM_ROOT) not in sys.path:
    sys.path.insert(0, str(FEDAVGM_ROOT))

from fedavgm.common import create_lda_partitions
from fedavgm.dataset import cifar10
from fedavgm.models import resnet10


def make_model(input_shape, num_classes, learning_rate):
    """Create the common CIFAR-10 ResNet10 model."""

    K.clear_session()
    gc.collect()

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    model = resnet10(
        input_shape=tuple(input_shape),
        num_classes=int(num_classes),
        learning_rate=float(learning_rate),
    )

    # Explicitly disable JIT where supported.
    try:
        model.compile(
            optimizer=keras.optimizers.SGD(
                learning_rate=float(learning_rate)
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=False,
        )
    except TypeError:
        model.compile(
            optimizer=keras.optimizers.SGD(
                learning_rate=float(learning_rate)
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def weighted_average_weights(client_weights, client_sizes):
    """Sample-weighted FedAvg aggregation used by FedProx."""

    if not client_weights:
        raise ValueError("No client weights were provided for aggregation")

    total_examples = float(sum(client_sizes))
    if total_examples <= 0:
        raise ValueError("Total client sample count must be positive")

    averaged = []

    for layer_weights in zip(*client_weights):
        layer_average = np.zeros_like(layer_weights[0], dtype=np.float64)

        for weights, num_examples in zip(layer_weights, client_sizes):
            layer_average += (
                weights.astype(np.float64)
                * (float(num_examples) / total_examples)
            )

        averaged.append(
            layer_average.astype(layer_weights[0].dtype, copy=False)
        )

    return averaged


def model_size_mb(weights):
    return sum(np.asarray(weight).nbytes for weight in weights) / (1024**2)


def save_weights(path, weights):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        **{
            f"arr_{index}": np.asarray(weight)
            for index, weight in enumerate(weights)
        },
    )


def load_weights(path):
    data = np.load(path)

    keys = sorted(
        data.files,
        key=lambda key: int(key.split("_")[1]),
    )

    return [data[key] for key in keys]


def completed_rounds(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return 0

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        return 0

    return int(dataframe["round"].max())


def append_row(csv_path, fieldnames, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    exists = csv_path.exists()

    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)
        csv_file.flush()


def local_fedprox_train(
    model,
    x_train,
    y_train,
    global_trainable_weights,
    epochs,
    batch_size,
    learning_rate,
    mu,
    seed,
):
    """Train one client using the official FedProx objective.

    Objective:

        task_loss + (mu / 2) * ||w_local - w_global||^2
    """

    optimizer = keras.optimizers.SGD(
        learning_rate=float(learning_rate)
    )

    loss_fn = keras.losses.CategoricalCrossentropy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    )

    dataset = dataset.shuffle(
        buffer_size=max(len(x_train), 1),
        seed=int(seed),
        reshuffle_each_iteration=True,
    )

    dataset = dataset.batch(
        int(batch_size),
        drop_remainder=False,
    )

    total_task_loss = 0.0
    total_correct = 0
    total_seen = 0

    for _ in range(int(epochs)):
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                predictions = model(
                    images,
                    training=True,
                )

                task_loss = loss_fn(
                    labels,
                    predictions,
                )

                # Match model.fit behaviour by including the model's L2 losses.
                if model.losses:
                    task_loss += tf.add_n(model.losses)

                proximal_terms = [
                    tf.reduce_sum(
                        tf.square(
                            local_variable
                            - tf.cast(
                                global_variable,
                                local_variable.dtype,
                            )
                        )
                    )
                    for local_variable, global_variable in zip(
                        model.trainable_variables,
                        global_trainable_weights,
                    )
                ]

                if proximal_terms:
                    proximal_loss = tf.add_n(proximal_terms)
                else:
                    proximal_loss = tf.constant(
                        0.0,
                        dtype=task_loss.dtype,
                    )

                total_loss = (
                    task_loss
                    + 0.5 * float(mu) * proximal_loss
                )

            gradients = tape.gradient(
                total_loss,
                model.trainable_variables,
            )

            gradient_variable_pairs = [
                (gradient, variable)
                for gradient, variable in zip(
                    gradients,
                    model.trainable_variables,
                )
                if gradient is not None
            ]

            optimizer.apply_gradients(
                gradient_variable_pairs
            )

            batch_size_actual = int(
                tf.shape(labels)[0].numpy()
            )

            predicted_labels = tf.argmax(
                predictions,
                axis=1,
                output_type=tf.int64,
            )

            true_labels = tf.argmax(
                labels,
                axis=1,
                output_type=tf.int64,
            )

            batch_correct = int(
                tf.reduce_sum(
                    tf.cast(
                        tf.equal(
                            predicted_labels,
                            true_labels,
                        ),
                        tf.int32,
                    )
                ).numpy()
            )

            total_task_loss += (
                float(task_loss.numpy())
                * batch_size_actual
            )
            total_correct += batch_correct
            total_seen += batch_size_actual

    if total_seen == 0:
        return float("nan"), float("nan")

    return (
        total_task_loss / total_seen,
        total_correct / total_seen,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "External FedProx direct benchmark using CIFAR-10 "
            "and the common ResNet10 model"
        )
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--client-dropout-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.num_clients <= 0:
        parser.error("--num-clients must be positive")

    if not 1 <= args.clients_per_round <= args.num_clients:
        parser.error(
            "--clients-per-round must be between 1 and num-clients"
        )

    if args.num_rounds <= 0:
        parser.error("--num-rounds must be positive")

    if args.local_epochs <= 0:
        parser.error("--local-epochs must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    if args.mu < 0:
        parser.error("--mu cannot be negative")

    if args.dirichlet_alpha <= 0:
        parser.error("--dirichlet-alpha must be positive")

    if not 0.0 <= args.client_dropout_ratio < 1.0:
        parser.error(
            "--client-dropout-ratio must be in [0, 1)"
        )

    return args


def main():
    args = parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except Exception:
            pass

    print(">>> External FedProx direct benchmark")
    print(">>> Device:", "GPU" if gpus else "CPU")
    print(">>> GPUs:", gpus)
    print(">>> FedProx mu:", args.mu)
    print(">>> Dirichlet alpha:", args.dirichlet_alpha)
    print(">>> Client dropout:", args.client_dropout_ratio)

    (
        x_train,
        y_train,
        x_test,
        y_test,
        input_shape,
        num_classes,
    ) = cifar10(
        num_classes=10,
        input_shape=(32, 32, 3),
    )

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    train_partitions, _ = create_lda_partitions(
        dataset=(x_train, y_train),
        num_partitions=args.num_clients,
        concentration=args.dirichlet_alpha,
        accept_imbalanced=False,
        seed=args.seed,
    )

    y_test_categorical = to_categorical(
        y_test,
        num_classes=num_classes,
    )

    csv_path = Path(args.csv_path)
    checkpoint_path = Path(args.checkpoint_path)

    done = completed_rounds(csv_path)

    if done > 0 and checkpoint_path.exists():
        print(
            f">>> Resuming from completed round {done}"
        )
        global_weights = load_weights(
            checkpoint_path
        )
    else:
        print(">>> Starting from round 1")

        initial_model = make_model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
        )

        global_weights = [
            weight.copy()
            for weight in initial_model.get_weights()
        ]

        del initial_model
        K.clear_session()
        gc.collect()

    parameter_size_mb = model_size_mb(
        global_weights
    )

    print(
        f">>> Model parameter size: "
        f"{parameter_size_mb:.4f} MB"
    )
    print(
        f">>> Number of train partitions: "
        f"{len(train_partitions)}"
    )
    print(
        f">>> Test shape: "
        f"{x_test.shape}, {y_test.shape}"
    )

    fieldnames = [
        "method",
        "dataset",
        "model",
        "round",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "num_selected_clients",
        "num_fit_clients",
        "fit_failures",
        "dropped_clients",
        "mu",
        "dirichlet_alpha",
        "local_epochs",
        "batch_size",
        "learning_rate",
        "fit_param_down_mb",
        "fit_param_up_mb",
        "fit_param_total_comm_mb",
        "round_total_comm_mb",
        "cumulative_total_comm_mb",
        "round_wall_time_sec",
        "cumulative_round_wall_time_sec",
    ]

    if csv_path.exists() and done > 0:
        previous = pd.read_csv(csv_path)

        cumulative_communication = float(
            previous[
                "cumulative_total_comm_mb"
            ].iloc[-1]
        )

        cumulative_time = float(
            previous[
                "cumulative_round_wall_time_sec"
            ].iloc[-1]
        )
    else:
        cumulative_communication = 0.0
        cumulative_time = 0.0

    for round_number in range(
        done + 1,
        args.num_rounds + 1,
    ):
        round_start = time.time()

        # Resume-safe deterministic client selection.
        selection_rng = np.random.default_rng(
            args.seed + round_number
        )

        selected_clients = selection_rng.choice(
            args.num_clients,
            size=args.clients_per_round,
            replace=False,
        ).tolist()

        dropout_rng = np.random.default_rng(
            args.seed + 10000 + round_number
        )

        num_dropped = int(
            round(
                len(selected_clients)
                * args.client_dropout_ratio
            )
        )

        if num_dropped > 0:
            dropped_clients = set(
                dropout_rng.choice(
                    selected_clients,
                    size=num_dropped,
                    replace=False,
                ).tolist()
            )
        else:
            dropped_clients = set()

        fit_clients = [
            client_id
            for client_id in selected_clients
            if client_id not in dropped_clients
        ]

        print(
            f">>> Round {round_number}: "
            f"selected={len(selected_clients)}, "
            f"dropped={len(dropped_clients)}, "
            f"fit={len(fit_clients)}"
        )

        client_weights = []
        client_sizes = []

        weighted_train_losses = []
        weighted_train_accuracies = []

        failures = 0

        for client_id in fit_clients:
            local_model = None

            try:
                (
                    client_x_train,
                    client_y_train,
                ) = train_partitions[client_id]

                client_y_train = np.asarray(
                    client_y_train
                ).reshape(-1)

                client_y_categorical = to_categorical(
                    client_y_train,
                    num_classes=num_classes,
                )

                local_model = make_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    learning_rate=args.learning_rate,
                )

                # Initialize the client from the current global model.
                local_model.set_weights(global_weights)

                # Snapshot only trainable variables for the proximal term.
                global_trainable_weights = [
                    tf.identity(variable)
                    for variable
                    in local_model.trainable_variables
                ]

                train_loss, train_accuracy = (
                    local_fedprox_train(
                        model=local_model,
                        x_train=client_x_train,
                        y_train=client_y_categorical,
                        global_trainable_weights=(
                            global_trainable_weights
                        ),
                        epochs=args.local_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        mu=args.mu,
                        seed=(
                            args.seed
                            + round_number * 1000
                            + client_id
                        ),
                    )
                )

                number_of_examples = len(
                    client_x_train
                )

                client_weights.append(
                    [
                        weight.copy()
                        for weight
                        in local_model.get_weights()
                    ]
                )

                client_sizes.append(
                    number_of_examples
                )

                weighted_train_losses.append(
                    train_loss * number_of_examples
                )

                weighted_train_accuracies.append(
                    train_accuracy * number_of_examples
                )

            except Exception as error:
                failures += 1

                print(
                    f">>> Client {client_id} failed: "
                    f"{repr(error)}"
                )

            finally:
                if local_model is not None:
                    del local_model

                K.clear_session()
                gc.collect()

        if client_weights:
            global_weights = weighted_average_weights(
                client_weights,
                client_sizes,
            )
        else:
            print(
                ">>> Warning: no successful client updates; "
                "global model remains unchanged"
            )

        evaluation_model = make_model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
        )

        evaluation_model.set_weights(
            global_weights
        )

        test_loss, test_accuracy = (
            evaluation_model.evaluate(
                x_test,
                y_test_categorical,
                batch_size=args.batch_size,
                verbose=0,
            )
        )

        del evaluation_model
        K.clear_session()
        gc.collect()

        total_fit_examples = sum(client_sizes)

        if total_fit_examples > 0:
            train_loss = (
                sum(weighted_train_losses)
                / total_fit_examples
            )

            train_accuracy = (
                sum(weighted_train_accuracies)
                / total_fit_examples
            )
        else:
            train_loss = float("nan")
            train_accuracy = float("nan")

        successful_clients = len(client_sizes)

        fit_parameter_download_mb = (
            successful_clients
            * parameter_size_mb
        )

        fit_parameter_upload_mb = (
            successful_clients
            * parameter_size_mb
        )

        fit_total_communication_mb = (
            fit_parameter_download_mb
            + fit_parameter_upload_mb
        )

        cumulative_communication += (
            fit_total_communication_mb
        )

        round_time = time.time() - round_start
        cumulative_time += round_time

        row = {
            "method": "ExternalFedProx-Direct",
            "dataset": "CIFAR10",
            "model": "ResNet10",
            "round": round_number,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "num_selected_clients": len(
                selected_clients
            ),
            "num_fit_clients": successful_clients,
            "fit_failures": failures,
            "dropped_clients": len(
                dropped_clients
            ),
            "mu": args.mu,
            "dirichlet_alpha": (
                args.dirichlet_alpha
            ),
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": (
                args.learning_rate
            ),
            "fit_param_down_mb": (
                fit_parameter_download_mb
            ),
            "fit_param_up_mb": (
                fit_parameter_upload_mb
            ),
            "fit_param_total_comm_mb": (
                fit_total_communication_mb
            ),
            "round_total_comm_mb": (
                fit_total_communication_mb
            ),
            "cumulative_total_comm_mb": (
                cumulative_communication
            ),
            "round_wall_time_sec": round_time,
            "cumulative_round_wall_time_sec": (
                cumulative_time
            ),
        }

        append_row(
            csv_path=csv_path,
            fieldnames=fieldnames,
            row=row,
        )

        save_weights(
            checkpoint_path,
            global_weights,
        )

        print(
            f">>> Round {round_number}: "
            f"train_acc={train_accuracy:.4f}, "
            f"test_acc={float(test_accuracy):.4f}, "
            f"test_loss={float(test_loss):.4f}, "
            f"failures={failures}, "
            f"round_time={round_time:.2f}s"
        )

    print(">>> Training complete")
    print(">>> CSV:", csv_path)
    print(">>> Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
