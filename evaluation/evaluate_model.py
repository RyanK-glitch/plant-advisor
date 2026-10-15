"""Evaluate a trained YOLO classifier on the held-out test split.

Reports accuracy, per-class precision/recall/F1, macro and weighted averages,
a confusion matrix and the list of misclassified images.

Usage (from the project folder, with the venv active):
    python evaluation/evaluate_model.py
    python evaluation/evaluate_model.py --weights runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt"
DEFAULT_TEST_DIR = PROJECT_DIR / "data/processed/test"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def canonical(label):
    # Same mapping as parse_prediction() in app.py, so raw model labels
    # such as "corn__maize__common_rust_" match test folders such as "maize_common_rust"
    parts = [part for part in re.split(r"_+", label.lower().strip()) if part]
    original_crop = parts.pop(0)
    crop = "maize" if original_crop == "corn" else original_crop
    while parts and parts[0] in {original_crop, crop}:
        parts.pop(0)
    condition = "_".join(parts)
    if crop == "rice" and condition in {"blast", "rice_blast"}:
        condition = "leaf_blast"
    return f"{crop}_{condition}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    weights, test_dir = Path(args.weights), Path(args.test_dir)
    run_name = weights.parent.parent.name
    out_dir = PROJECT_DIR / "reports" / f"evaluation_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect test images; the folder name is the true class
    images, y_true = [], []
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() in IMAGE_SUFFIXES:
                images.append(str(image))
                y_true.append(canonical(class_dir.name))
    print(f"Model: {weights}\nTest images: {len(images)} in {len(set(y_true))} classes")

    model = YOLO(str(weights))
    imgsz = model.overrides.get("imgsz", 480)

    y_pred = []
    for start in range(0, len(images), args.batch):
        for result in model.predict(images[start:start + args.batch], imgsz=imgsz, verbose=False):
            y_pred.append(canonical(result.names[result.probs.top1]))
        print(f"  predicted {len(y_pred)}/{len(images)}", end="\r")
    print()

    labels = sorted(set(y_true) | set(y_pred))

    # Overall and per-class metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    ).T
    report_df.to_csv(out_dir / "classification_report.csv")

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(report)

    # Confusion matrix, row-normalised so each row shows recall for that class
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if matrix[i, j]:
                ax.text(j, i, f"{normalised[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if normalised[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Test-set confusion matrix (row-normalised) - accuracy {accuracy:.2%}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)

    # Misclassified images, for error analysis in the report
    errors = pd.DataFrame({"image": images, "true": y_true, "predicted": y_pred})
    errors = errors[errors.true != errors.predicted]
    errors.to_csv(out_dir / "misclassified.csv", index=False)

    print(f"Misclassified: {len(errors)} of {len(images)}")
    print(f"Saved report, confusion matrix and error list to {out_dir}")


if __name__ == "__main__":
    main()
