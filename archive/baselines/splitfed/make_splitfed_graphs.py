import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "results/splitfed_cifar10_resnet10_50rounds_v2.csv"
OUT_DIR = "results/splitfed_graphs_v2"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# Keep only fully valid benchmark rounds.
# Round 50 had eval failures and test_accuracy=0.0, so exclude failure rows.
clean = df[
    (df["fit_failures"] == 0)
    & (df["eval_failures"] == 0)
    & (df["test_accuracy"] > 0)
].copy()

clean_csv = os.path.join(OUT_DIR, "splitfed_cifar10_resnet10_49rounds_clean.csv")
clean.to_csv(clean_csv, index=False)

summary_path = os.path.join(OUT_DIR, "splitfed_summary.txt")

best_acc_row = clean.loc[clean["test_accuracy"].idxmax()]
final_row = clean.loc[clean["round"].idxmax()]
best_loss_row = clean.loc[clean["test_loss"].idxmin()]

with open(summary_path, "w") as f:
    f.write("SplitFed CIFAR10 ResNet10 Benchmark Summary\n")
    f.write("==========================================\n\n")
    f.write(f"Input CSV: {INPUT_CSV}\n")
    f.write(f"Clean CSV: {clean_csv}\n")
    f.write(f"Valid rounds used: {int(clean['round'].min())} to {int(clean['round'].max())}\n")
    f.write("Excluded rows: any row with fit/eval failures or test_accuracy <= 0\n\n")

    f.write("Final valid round:\n")
    f.write(f"  Round: {int(final_row['round'])}\n")
    f.write(f"  Train loss: {final_row['train_loss']:.6f}\n")
    f.write(f"  Train accuracy: {final_row['train_accuracy']:.6f}\n")
    f.write(f"  Test loss: {final_row['test_loss']:.6f}\n")
    f.write(f"  Test accuracy: {final_row['test_accuracy']:.6f}\n")
    f.write(f"  Cumulative communication MB: {final_row['cumulative_total_comm_mb']:.3f}\n")
    f.write(f"  Cumulative wall time sec: {final_row['cumulative_round_wall_time_sec']:.3f}\n\n")

    f.write("Best test accuracy:\n")
    f.write(f"  Round: {int(best_acc_row['round'])}\n")
    f.write(f"  Test accuracy: {best_acc_row['test_accuracy']:.6f}\n")
    f.write(f"  Test loss: {best_acc_row['test_loss']:.6f}\n\n")

    f.write("Best test loss:\n")
    f.write(f"  Round: {int(best_loss_row['round'])}\n")
    f.write(f"  Test loss: {best_loss_row['test_loss']:.6f}\n")
    f.write(f"  Test accuracy: {best_loss_row['test_accuracy']:.6f}\n")

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved", path)

# 1. Accuracy vs round
plt.figure(figsize=(9, 5))
plt.plot(clean["round"], clean["train_accuracy"], marker="o", label="Train Accuracy")
plt.plot(clean["round"], clean["test_accuracy"], marker="o", label="Test Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("SplitFed CIFAR-10 ResNet10 Accuracy vs Round")
plt.grid(True)
plt.legend()
savefig("01_accuracy_vs_round.png")

# 2. Loss vs round
plt.figure(figsize=(9, 5))
plt.plot(clean["round"], clean["train_loss"], marker="o", label="Train Loss")
plt.plot(clean["round"], clean["test_loss"], marker="o", label="Test Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("SplitFed CIFAR-10 ResNet10 Loss vs Round")
plt.grid(True)
plt.legend()
savefig("02_loss_vs_round.png")

# 3. Communication per round
plt.figure(figsize=(9, 5))
plt.plot(clean["round"], clean["round_total_comm_mb"], marker="o", label="Total Communication")
plt.plot(clean["round"], clean["round_param_comm_mb"], marker="o", label="Parameter Communication")
plt.plot(clean["round"], clean["round_splitfed_comm_mb"], marker="o", label="SplitFed Smashed-data Communication")
plt.xlabel("Round")
plt.ylabel("Communication (MB)")
plt.title("SplitFed Communication per Round")
plt.grid(True)
plt.legend()
savefig("03_communication_per_round.png")

# 4. Cumulative communication
plt.figure(figsize=(9, 5))
plt.plot(clean["round"], clean["cumulative_total_comm_mb"], marker="o")
plt.xlabel("Round")
plt.ylabel("Cumulative Communication (MB)")
plt.title("SplitFed Cumulative Communication")
plt.grid(True)
savefig("04_cumulative_communication.png")

# 5. Round wall time
plt.figure(figsize=(9, 5))
plt.plot(clean["round"], clean["round_wall_time_sec"], marker="o")
plt.xlabel("Round")
plt.ylabel("Wall Time (seconds)")
plt.title("SplitFed Wall Time per Round")
plt.grid(True)
savefig("05_wall_time_per_round.png")

# 6. Cumulative wall time
plt.figure(figsize=(9, 5))
plt.plot(clean["round"], clean["cumulative_round_wall_time_sec"], marker="o")
plt.xlabel("Round")
plt.ylabel("Cumulative Wall Time (seconds)")
plt.title("SplitFed Cumulative Wall Time")
plt.grid(True)
savefig("06_cumulative_wall_time.png")

# 7. Accuracy vs cumulative communication
plt.figure(figsize=(9, 5))
plt.plot(clean["cumulative_total_comm_mb"], clean["test_accuracy"], marker="o")
plt.xlabel("Cumulative Communication (MB)")
plt.ylabel("Test Accuracy")
plt.title("SplitFed Test Accuracy vs Communication")
plt.grid(True)
savefig("07_accuracy_vs_communication.png")

# 8. Accuracy vs cumulative time
plt.figure(figsize=(9, 5))
plt.plot(clean["cumulative_round_wall_time_sec"], clean["test_accuracy"], marker="o")
plt.xlabel("Cumulative Wall Time (seconds)")
plt.ylabel("Test Accuracy")
plt.title("SplitFed Test Accuracy vs Time")
plt.grid(True)
savefig("08_accuracy_vs_time.png")

# Zip everything
zip_path = "results/splitfed_graphs_v2.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for filename in os.listdir(OUT_DIR):
        z.write(os.path.join(OUT_DIR, filename), arcname=filename)

print("\nDONE")
print("Clean CSV:", clean_csv)
print("Summary:", summary_path)
print("Zip:", zip_path)
print("\nFinal valid round:")
print(final_row[["round", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "cumulative_total_comm_mb", "cumulative_round_wall_time_sec"]])
print("\nBest accuracy round:")
print(best_acc_row[["round", "test_loss", "test_accuracy"]])
