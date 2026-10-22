from pathlib import Path
import pandas as pd

TIMING_DIR = Path("track4/results/timing")
TIMING_DIR.mkdir(parents=True, exist_ok=True)

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

        path = TIMING_DIR / f"{config.lower()}_{algorithm}_timing.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)

        if df.empty or "round_seconds" not in df.columns:
            continue

        round_times = pd.to_numeric(
            df["round_seconds"],
            errors="coerce"
        ).dropna()

        if round_times.empty:
            continue

        row = {
            "algorithm": algorithm,
            "config": config,
            "total_clients": info["clients"],
            "clients_per_round": info["clients_per_round"],
            "rounds_measured": len(round_times),
            "mean_round_seconds": round_times.mean(),
            "median_round_seconds": round_times.median(),
            "min_round_seconds": round_times.min(),
            "max_round_seconds": round_times.max(),
            "total_runtime_seconds": round_times.sum(),
            "total_runtime_minutes": round_times.sum() / 60.0,
        }

        if "test_accuracy" in df.columns:
            acc = pd.to_numeric(
                df["test_accuracy"],
                errors="coerce"
            ).dropna()

            if not acc.empty:
                row["final_test_accuracy"] = acc.iloc[-1]

                for threshold in [0.50, 0.60, 0.65, 0.70]:
                    reached = df[
                        pd.to_numeric(
                            df["test_accuracy"],
                            errors="coerce"
                        ) >= threshold
                    ]

                    if not reached.empty:
                        row[f"round_to_{int(threshold * 100)}pct"] = int(
                            reached.iloc[0]["round"]
                        )
                    else:
                        row[f"round_to_{int(threshold * 100)}pct"] = None

        rows.append(row)

if not rows:
    print("No timing CSV files found yet.")
else:
    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["algorithm", "total_clients"]
    ).reset_index(drop=True)

    print("\nTrack 4 Timing Summary\n")
    print(summary.to_string(index=False))

    output = TIMING_DIR / "track4_timing_summary.csv"
    summary.to_csv(output, index=False)

    print(f"\nSaved: {output}")
