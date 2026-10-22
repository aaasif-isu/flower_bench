from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("track4/results")
PLOTS_DIR = Path("track4/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "C1 (50 clients)": RESULTS_DIR / "c1_fedavg_50r.csv",
    "C2 (100 clients)": RESULTS_DIR / "c2_fedavg_50r.csv",
    "C3 (500 clients)": RESULTS_DIR / "c3_fedavg_50r.csv",
    "C4 (1000 clients)": RESULTS_DIR / "c4_fedavg_50r.csv",
}


# ---------------------------------------------------------
# Plot 1: Test accuracy vs communication round
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

found = False

for label, path in FILES.items():
    if not path.exists():
        print(f"Skipping missing file: {path}")
        continue

    df = pd.read_csv(path)

    plt.plot(
        df["round"],
        df["test_accuracy"],
        label=label,
    )

    found = True

if found:
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.title("FedAvg Scaling: Test Accuracy vs Communication Round")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = PLOTS_DIR / "fedavg_accuracy_vs_rounds.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")
else:
    plt.close()


# ---------------------------------------------------------
# Plot 2: Final test accuracy vs total clients
# ---------------------------------------------------------

client_counts = []
final_accuracies = []

config_clients = {
    "C1 (50 clients)": 50,
    "C2 (100 clients)": 100,
    "C3 (500 clients)": 500,
    "C4 (1000 clients)": 1000,
}

for label, path in FILES.items():
    if not path.exists():
        continue

    df = pd.read_csv(path)

    client_counts.append(config_clients[label])
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
    plt.title("FedAvg Scaling: Final Accuracy vs Client Population")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = PLOTS_DIR / "fedavg_final_accuracy_vs_clients.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")
