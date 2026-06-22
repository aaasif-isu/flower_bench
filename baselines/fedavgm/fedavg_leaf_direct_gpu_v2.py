import os

# Must be set before TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from fedavgm.dataset import leaf_femnist
from fedavgm.models import resnet10


def make_model(input_shape, num_classes, lr):
    K.clear_session()
    gc.collect()

    tf.config.optimizer.set_jit(False)

    model = resnet10(
        input_shape=tuple(input_shape),
        num_classes=int(num_classes),
        learning_rate=float(lr),
    )

    # Force no XLA/JIT compile
    try:
        opt = model.optimizer
        loss = model.loss if model.loss is not None else "categorical_crossentropy"
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy"],
            jit_compile=False,
        )
    except TypeError:
        opt = model.optimizer
        loss = model.loss if model.loss is not None else "categorical_crossentropy"
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=["accuracy"],
        )

    return model


def weighted_average_weights(client_weights, client_sizes):
    total = float(sum(client_sizes))
    return [
        sum(w * (n / total) for w, n in zip(weights_tuple, client_sizes))
        for weights_tuple in zip(*client_weights)
    ]


def model_size_mb(weights):
    return sum(w.nbytes for w in weights) / (1024 * 1024)


def save_weights(path, weights):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"arr_{i}": w for i, w in enumerate(weights)})


def load_weights(path):
    data = np.load(path)
    keys = sorted(data.files, key=lambda x: int(x.split("_")[1]))
    return [data[k] for k in keys]


def completed_rounds(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return 0
    return int(df["round"].max())


def append_row(csv_path, fieldnames, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf-root", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--clients-per-round", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    print(">>> Direct FedAvg GPU v2 resume-safe run")
    print(">>> GPUs:", gpus)

    train_partitions, _, x_test, y_test, input_shape, num_classes = leaf_femnist(
        root=args.leaf_root,
        num_clients=args.num_clients,
        partition_name="niid",
        input_shape=(28, 28, 1),
        num_classes=62,
    )

    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    ckpt = Path(args.checkpoint_path)
    done = completed_rounds(args.csv_path)

    if done > 0 and ckpt.exists():
        print(f">>> Resuming from completed round {done}")
        global_weights = load_weights(ckpt)
    else:
        print(">>> Starting from round 1")
        init_model = make_model(input_shape, num_classes, args.lr)
        global_weights = [w.copy() for w in init_model.get_weights()]
        del init_model
        K.clear_session()
        gc.collect()

    param_mb = model_size_mb(global_weights)

    print(f">>> Model parameter size: {param_mb:.4f} MB")
    print(f">>> Train clients: {len(train_partitions)}")
    print(f">>> Test shape: {x_test.shape}, {y_test.shape}")

    fieldnames = [
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
        "num_eval_clients",
        "eval_failures",
        "fit_param_down_mb",
        "fit_param_up_mb",
        "fit_param_total_comm_mb",
        "round_total_comm_mb",
        "cumulative_total_comm_mb",
        "round_wall_time_sec",
        "cumulative_round_wall_time_sec",
    ]

    if Path(args.csv_path).exists() and done > 0:
        prev = pd.read_csv(args.csv_path)
        cumulative_comm = float(prev["cumulative_total_comm_mb"].iloc[-1])
        cumulative_time = float(prev["cumulative_round_wall_time_sec"].iloc[-1])
    else:
        cumulative_comm = 0.0
        cumulative_time = 0.0

    for rnd in range(done + 1, args.rounds + 1):
        round_start = time.time()

        # Deterministic per-round selection/dropout, resume-safe
        rng = np.random.default_rng(args.seed + rnd)
        selected = rng.choice(
            args.num_clients,
            size=args.clients_per_round,
            replace=False,
        ).tolist()

        drop_rng = np.random.default_rng(args.seed + 10000 + rnd)
        num_drop = int(round(len(selected) * args.dropout))
        drop_set = set(drop_rng.choice(selected, size=num_drop, replace=False).tolist()) if num_drop > 0 else set()
        fit_clients = [cid for cid in selected if cid not in drop_set]

        print(f">>> Round {rnd}: selected={len(selected)}, dropped={len(drop_set)}, fit={len(fit_clients)}")

        client_weights = []
        client_sizes = []
        train_losses = []
        train_accs = []
        failures = 0

        for cid in fit_clients:
            try:
                x_train, y_train = train_partitions[cid]
                y_train_cat = to_categorical(y_train, num_classes=num_classes)

                local_model = make_model(input_shape, num_classes, args.lr)
                local_model.set_weights(global_weights)

                hist = local_model.fit(
                    x_train,
                    y_train_cat,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose=0,
                )

                loss = float(hist.history["loss"][-1])
                acc_key = "accuracy" if "accuracy" in hist.history else "acc"
                acc = float(hist.history[acc_key][-1])

                client_weights.append([w.copy() for w in local_model.get_weights()])
                client_sizes.append(len(x_train))
                train_losses.append(loss * len(x_train))
                train_accs.append(acc * len(x_train))

                del local_model
                K.clear_session()
                gc.collect()

            except Exception as e:
                failures += 1
                print(f">>> Client {cid} failed: {repr(e)}")
                K.clear_session()
                gc.collect()

        if client_weights:
            global_weights = weighted_average_weights(client_weights, client_sizes)

        eval_model = make_model(input_shape, num_classes, args.lr)
        eval_model.set_weights(global_weights)
        test_loss, test_acc = eval_model.evaluate(
            x_test,
            y_test_cat,
            batch_size=args.batch_size,
            verbose=0,
        )

        del eval_model
        K.clear_session()
        gc.collect()

        fit_examples = sum(client_sizes) if client_sizes else 1
        train_loss = sum(train_losses) / fit_examples if train_losses else float("nan")
        train_acc = sum(train_accs) / fit_examples if train_accs else float("nan")

        fit_down = len(fit_clients) * param_mb
        fit_up = len(fit_clients) * param_mb
        fit_total = fit_down + fit_up
        cumulative_comm += fit_total

        round_time = time.time() - round_start
        cumulative_time += round_time

        row = {
            "method": "FedAvg-DirectGPU-v2",
            "dataset": "LEAF_FEMNIST_NIID",
            "model": "resnet10",
            "round": rnd,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "num_fit_clients": len(fit_clients),
            "fit_failures": failures,
            "num_eval_clients": 0,
            "eval_failures": 0,
            "fit_param_down_mb": fit_down,
            "fit_param_up_mb": fit_up,
            "fit_param_total_comm_mb": fit_total,
            "round_total_comm_mb": fit_total,
            "cumulative_total_comm_mb": cumulative_comm,
            "round_wall_time_sec": round_time,
            "cumulative_round_wall_time_sec": cumulative_time,
        }

        append_row(args.csv_path, fieldnames, row)
        save_weights(args.checkpoint_path, global_weights)

        print(
            f">>> Round {rnd}: train_acc={train_acc:.4f}, "
            f"test_acc={float(test_acc):.4f}, failures={failures}, "
            f"round_time={round_time:.2f}s"
        )

    print(f">>> Saved CSV to {args.csv_path}")


if __name__ == "__main__":
    main()
