import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to YOLO training results
# Path to YOLO training results
results_file = "runs/classify/crop_model_runs/yolo11m_staples_480_split70/results.csv"

# Folder to save evaluation charts
output_dir = "reports/figures"

# Check if results.csv exists
if not os.path.exists(results_file):
    print(f"❌ File not found: {results_file}")
    exit()

# Load training results
data = pd.read_csv(results_file)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Create output folder if needed
os.makedirs(output_dir, exist_ok=True)

# Plot training and validation loss
plt.figure(figsize=(10, 6))

if "train/loss" in data.columns:
    plt.plot(data["epoch"], data["train/loss"], label="Training Loss")

if "val/loss" in data.columns:
    plt.plot(data["epoch"], data["val/loss"], label="Validation Loss")

plt.title("YOLO11m Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save loss chart
plt.savefig(
    "reports/figures/training_loss.png",
    dpi=300
)

plt.show()


# Plot classification accuracy
if "metrics/accuracy_top1" in data.columns:
    plt.figure(figsize=(10, 6))

    plt.plot(
        data["epoch"],
        data["metrics/accuracy_top1"],
        label="Top-1 Accuracy"
    )

    plt.title("YOLO11m Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save accuracy chart
    plt.savefig(
        "reports/figures/training_accuracy.png",
        dpi=300
    )

    plt.show()

print("✅ Training history charts saved in reports/figures.")