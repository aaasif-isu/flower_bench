from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("track4/results")

CONFIGS = {
    "C1": {
        "clients": 50,
        "clients_per_round": 10,
        "file": "c1_fedavg_50r.csv",
    },
    "C2": {
        "clients": 100,
        "clients_per_round": 20,
        "file": "c2_fedavg_50r.csv",
    },
    "C3": {
        "clients": 500,
        "clients_per_round": 100,
        "file": "c3_fedavg_50r.csv",
    },
    "C4": {
        "clients": 1000,
        "clients_per_round": 200,
        "file": "c4_fedavg_50r.csv",
    },
}

rows = []

for config, info in CONFIGS.items():
    path = RESULTS_DIR / info["file"]

    if not path.exists():
        print(f"{config}: missing -> {path}")
        continue

    df = pd.read_csv(path)

    if df.empty:
        print(f"{config}: file exists but is empty")
        continue

    final = df.iloc[-1]

    rows.append(
        {
            "config": config,
            "total_clients": info["clients"],
            "clients_per_round": info["clients_per_round"],
            "rounds_completed": int(final["round"]),
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "final_test_loss": final["test_loss"],
            "final_test_accuracy": final["test_accuracy"],
            "run_completed": final.get("run_completed", None),
        }
    )

if not rows:
    print("No completed result files found.")
else:
    summary = pd.DataFrame(rows)

    print("\nTrack 4 FedAvg Summary\n")
    print(summary.to_string(index=False))

    output = RESULTS_DIR / "fedavg_scaling_summary.csv"
    summary.to_csv(output, index=False)

    print(f"\nSaved summary to: {output}")
