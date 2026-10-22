from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("track4/results")

ALGORITHMS = [
    "fedavg",
    "fedprox",
    "fednova",
    "fedavgm",
    "fedadam",
    "scaffold",
]

CONFIGS = {
    "C1": {"clients": 50, "clients_per_round": 10},
    "C2": {"clients": 100, "clients_per_round": 20},
    "C3": {"clients": 500, "clients_per_round": 100},
    "C4": {"clients": 1000, "clients_per_round": 200},
}

rows = []

for algorithm in ALGORITHMS:
    for config, info in CONFIGS.items():
        filename = f"{config.lower()}_{algorithm}_50r.csv"
        path = RESULTS_DIR / filename

        if not path.exists():
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"Skipping empty file: {path}")
            continue

        final = df.iloc[-1]

        rows.append(
            {
                "algorithm": algorithm,
                "config": config,
                "total_clients": info["clients"],
                "clients_per_round": info["clients_per_round"],
                "rounds_completed": int(final["round"]),
                "final_train_loss": final.get("train_loss"),
                "final_train_accuracy": final.get("train_accuracy"),
                "final_test_loss": final.get("test_loss"),
                "final_test_accuracy": final.get("test_accuracy"),
                "run_completed": final.get("run_completed"),
                "result_file": filename,
            }
        )

if not rows:
    print("No Track 4 algorithm results found.")
else:
    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        by=["algorithm", "total_clients"]
    ).reset_index(drop=True)

    print("\nTrack 4 Algorithm Summary\n")
    print(summary.to_string(index=False))

    output = RESULTS_DIR / "track4_all_algorithms_summary.csv"
    summary.to_csv(output, index=False)

    print(f"\nSaved summary to: {output}")
