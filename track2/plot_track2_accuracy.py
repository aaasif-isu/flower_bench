from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "results" / "raw"
OUTPUT_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")


DATASETS = {
    "cifar10": {
        "partition": "iid",
        "title": "CIFAR-10",
    },
    "femnist": {
        "partition": "natural",
        "title": "FEMNIST",
    },
    "shakespeare": {
        "partition": "natural",
        "title": "Shakespeare",
    },
}


ALGORITHMS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fednova": "FedNova",
    "fedavgm": "FedAvgM",
    "fedadam": "FedAdam",
}


SEEDS = [42, 43, 44, 45, 46]


def load_runs(dataset, partition, algorithm):
    runs = []

    for seed in SEEDS:
        filename = f"{algorithm}_{partition}_seed{seed}.csv"
        path = RAW_DIR / dataset / filename

        if not path.exists():
            print(f"[WARNING] Missing: {path}")
            continue

        df = pd.read_csv(path)

        df = df.dropna(subset=["test_accuracy"]).copy()

        if df.empty:
            print(f"[WARNING] No test accuracy in: {path}")
            continue

        df["total_cumulative_bytes"] = (
            df["cumulative_upload_bytes"]
            + df["cumulative_download_bytes"]
        )

        df["total_cumulative_gb"] = (
            df["total_cumulative_bytes"] / (1024 ** 3)
        )

        df["seed"] = seed

        runs.append(df)

    return runs


def aggregate_by_round(runs):
    combined = pd.concat(runs, ignore_index=True)

    summary = (
        combined
        .groupby("round")
        .agg(
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
        )
        .reset_index()
    )

    summary["std_accuracy"] = summary["std_accuracy"].fillna(0)

    return summary


def aggregate_by_cost(runs, x_column):
    all_runs = []

    for df in runs:
        temp = df[[x_column, "test_accuracy"]].copy()
        all_runs.append(temp)

    combined = pd.concat(all_runs, ignore_index=True)

    # Different seeds can have slightly different x-values,
    # so bin them into common points for comparison.
    combined = combined.sort_values(x_column)

    return combined


def plot_rounds(dataset, config):
    plt.figure(figsize=(12, 7))

    palette = sns.color_palette("tab10", len(ALGORITHMS))

    for color, (algorithm, label) in zip(
        palette,
        ALGORITHMS.items()
    ):
        runs = load_runs(
            dataset,
            config["partition"],
            algorithm
        )

        if not runs:
            continue

        summary = aggregate_by_round(runs)

        x = summary["round"].to_numpy()
        mean = summary["mean_accuracy"].to_numpy()
        std = summary["std_accuracy"].to_numpy()

        plt.plot(
            x,
            mean,
            label=label,
            linewidth=2.3,
            color=color,
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
        )

    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.title(
        f"Track 2: {config['title']} Test Accuracy vs Communication Round"
    )

    plt.legend(title="Algorithm")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    output = OUTPUT_DIR / f"{dataset}_accuracy_vs_rounds.png"

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {output}")
    plt.close()


def plot_cost_curve(
    dataset,
    config,
    x_column,
    xlabel,
    filename_suffix,
):
    plt.figure(figsize=(12, 7))

    palette = sns.color_palette("tab10", len(ALGORITHMS))

    for color, (algorithm, label) in zip(
        palette,
        ALGORITHMS.items()
    ):
        runs = load_runs(
            dataset,
            config["partition"],
            algorithm
        )

        if not runs:
            continue

        # Plot each seed faintly
        for run in runs:
            plt.plot(
                run[x_column],
                run["test_accuracy"],
                color=color,
                alpha=0.12,
                linewidth=1,
            )

        # Build mean curve using round alignment
        merged = None

        for i, run in enumerate(runs):
            temp = run[
                ["round", x_column, "test_accuracy"]
            ].copy()

            temp = temp.rename(
                columns={
                    x_column: f"x_{i}",
                    "test_accuracy": f"acc_{i}",
                }
            )

            if merged is None:
                merged = temp
            else:
                merged = pd.merge(
                    merged,
                    temp,
                    on="round",
                    how="inner",
                )

        x_cols = [
            col for col in merged.columns
            if col.startswith("x_")
        ]

        acc_cols = [
            col for col in merged.columns
            if col.startswith("acc_")
        ]

        mean_x = merged[x_cols].mean(axis=1)
        mean_acc = merged[acc_cols].mean(axis=1)

        plt.plot(
            mean_x,
            mean_acc,
            label=label,
            linewidth=2.5,
            color=color,
        )

    plt.xlabel(xlabel)
    plt.ylabel("Test Accuracy")
    plt.title(
        f"Track 2: {config['title']} Test Accuracy vs {xlabel}"
    )

    plt.legend(title="Algorithm")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    output = (
        OUTPUT_DIR
        / f"{dataset}_accuracy_vs_{filename_suffix}.png"
    )

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {output}")
    plt.close()


def main():
    print("Generating Track 2 plots...\n")

    for dataset, config in DATASETS.items():

        print(f"Processing {config['title']}...")

        # 1. Accuracy vs communication rounds
        plot_rounds(dataset, config)

        # 2. Accuracy vs total cumulative communication bytes
        plot_cost_curve(
            dataset,
            config,
            x_column="total_cumulative_gb",
            xlabel="Cumulative Communication (GB)",
            filename_suffix="communication_gb",
        )

        # 3. Accuracy vs cumulative local steps
        plot_cost_curve(
            dataset,
            config,
            x_column="cumulative_local_steps",
            xlabel="Cumulative Local Steps",
            filename_suffix="local_steps",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
