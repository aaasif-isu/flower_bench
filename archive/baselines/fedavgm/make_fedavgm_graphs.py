import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("outputs/2026-06-03/11-44-59/fedavgm_round_metrics.csv")
out_dir = Path("results/fedavgm_benchmark_graphs")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# Clean numeric columns
for col in df.columns:
    if col not in ["method", "dataset", "model"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Save clean copy
clean_csv = out_dir / "fedavgm_cifar10_resnet10_50rounds_clean.csv"
df.to_csv(clean_csv, index=False)

# 1. Test Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["test_accuracy"], marker="o")
plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.title("FedAvg - CIFAR-10 ResNet10 Test Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_test_accuracy.png", dpi=300)
plt.close()

# 2. Test Loss
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["test_loss"], marker="o")
plt.xlabel("Round")
plt.ylabel("Test Loss")
plt.title("FedAvg - CIFAR-10 ResNet10 Test Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_test_loss.png", dpi=300)
plt.close()

# 3. Train Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["train_accuracy"], marker="o")
plt.xlabel("Round")
plt.ylabel("Train Accuracy")
plt.title("FedAvg - CIFAR-10 ResNet10 Train Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_train_accuracy.png", dpi=300)
plt.close()

# 4. Train Loss
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["train_loss"], marker="o")
plt.xlabel("Round")
plt.ylabel("Train Loss")
plt.title("FedAvg - CIFAR-10 ResNet10 Train Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_train_loss.png", dpi=300)
plt.close()

# 5. Round Wall Time
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["round_wall_time_sec"], marker="o")
plt.xlabel("Round")
plt.ylabel("Round Wall Time (sec)")
plt.title("FedAvg - Round Wall Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_round_wall_time.png", dpi=300)
plt.close()

# 6. Cumulative Time
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["cumulative_round_wall_time_sec"], marker="o")
plt.xlabel("Round")
plt.ylabel("Cumulative Time (sec)")
plt.title("FedAvg - Cumulative Runtime")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_cumulative_time.png", dpi=300)
plt.close()

# 7. Communication
plt.figure(figsize=(10, 6))
plt.plot(df["round"], df["round_total_comm_mb"], marker="o")
plt.xlabel("Round")
plt.ylabel("Communication (MB)")
plt.title("FedAvg - Per-Round Communication")
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir / "fedavg_round_communication.png", dpi=300)
plt.close()

# 8. Summary text
summary = {
    "final_round": int(df["round"].max()),
    "final_test_accuracy": float(df["test_accuracy"].iloc[-1]),
    "final_test_loss": float(df["test_loss"].iloc[-1]),
    "final_train_accuracy": float(df["train_accuracy"].iloc[-1]),
    "final_train_loss": float(df["train_loss"].iloc[-1]),
    "total_time_sec": float(df["cumulative_round_wall_time_sec"].iloc[-1]),
    "total_time_min": float(df["cumulative_round_wall_time_sec"].iloc[-1] / 60),
    "total_comm_mb": float(df["cumulative_total_comm_mb"].iloc[-1]),
}

summary_path = out_dir / "fedavg_summary.txt"
with open(summary_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("Saved graphs and clean CSV to:", out_dir)
print("Clean CSV:", clean_csv)
print("Summary:", summary_path)
print(summary)
