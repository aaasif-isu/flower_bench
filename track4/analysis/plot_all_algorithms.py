from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("track4/results")
PLOTS_DIR = Path("track4/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = [
    "fedavg",
    "fedprox",
    "fednova",
    "fedavgm",
    "fedadam",
    "scaffold",
]

CONFIGS = {
    "C1": 50,
    "C2": 100,
    "C3": 500,
    "C4": 1000,
}

# ---------------------------------------------------------
# Plot 1: Accuracy vs rounds, one plot per algorithm
# ---------------------------------------------------------

for algorithm in ALGORITHMS:
    plt.figure(figsize=(9, 6))
    found = False

    for config, clients in CONFIGS.items():
        path = RESULTS_DIR / f"{config.lower()}_{algorithm}_50r.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)

        plt.plot(
            df["round"],
            df["test_accuracy"],
            label=f"{config} ({clients} clients)",
        )

        found = True

    if found:
        plt.xlabel("Communication Round")
        plt.ylabel("Test Accuracy")
        plt.title(f"{algorithm.upper()}: Test Accuracy vs Communication Round")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output = PLOTS_DIR / f"{algorithm}_accuracy_vs_rounds.png"
        plt.savefig(output, dpi=300)
        plt.close()

        print(f"Saved: {output}")
    else:
        plt.close()


# ---------------------------------------------------------
# Plot 2: Final accuracy vs total clients
# ---------------------------------------------------------

for algorithm in ALGORITHMS:
    client_counts = []
    final_accuracies = []

    for config, clients in CONFIGS.items():
        path = RESULTS_DIR / f"{config.lower()}_{algorithm}_50r.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)

        client_counts.append(clients)
        final_accuracies.append(df.iloc[-1]["test_accuracy"])

    if client_counts:
        plt.figure(figsize=(8, 6))

        plt.plot(
            client_counts,
            final_accuracies,
            marker="o",
        )

        plt.xlabel("Total Number of Clients")
        plt.ylabel("Final Test Accuracy")
        plt.title(f"{algorithm.upper()}: Final Accuracy vs Client Population")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output = PLOTS_DIR / f"{algorithm}_final_accuracy_vs_clients.png"
        plt.savefig(output, dpi=300)
        plt.close()

        print(f"Saved: {output}")


# ---------------------------------------------------------
# Plot 3: Cross-algorithm comparison
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))
comparison_found = False

for algorithm in ALGORITHMS:
    x = []
    y = []

    for config, clients in CONFIGS.items():
        path = RESULTS_DIR / f"{config.lower()}_{algorithm}_50r.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)

        x.append(clients)
        y.append(df.iloc[-1]["test_accuracy"])

    if x:
        plt.plot(x, y, marker="o", label=algorithm.upper())
        comparison_found = True

if comparison_found:
    plt.xlabel("Total Number of Clients")
    plt.ylabel("Final Test Accuracy")
    plt.title("Track 4: Final Accuracy Across Algorithms")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = PLOTS_DIR / "all_algorithms_final_accuracy_vs_clients.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")
else:
    plt.close()
