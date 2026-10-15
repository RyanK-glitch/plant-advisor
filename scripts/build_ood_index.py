"""Build and evaluate the unknown-photo (out-of-distribution) check used by the app.

Extracts classifier features for the training images (the reference bank), sets the
rejection threshold so that a target share of genuine validation images is accepted,
and measures how many leaves of unsupported crops are rejected.

Out-of-scope images come from PlantVillage crops the model was never trained on. They
are split by crop into two disjoint groups: one to choose the scoring method, one to
report the final result, so the reported numbers come from crops never used for tuning.

Usage (from the project folder, with the venv active):
    python scripts/build_ood_index.py
"""
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from src.input_guard import DEFAULT_INDEX, InputGuard, normalise, ood_scores  # noqa: E402
from src.labels import parse_prediction  # noqa: E402

WEIGHTS = PROJECT_DIR / "runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt"
PROCESSED = PROJECT_DIR / "data/processed"
PLANTVILLAGE = PROJECT_DIR / "data/raw/PlantVillage"
REPORT = PROJECT_DIR / "reports/tables/ood_evaluation.csv"

# Unsupported crops, split into a method-selection group and a held-out test group
SELECTION_CROPS = ("Apple", "Blueberry", "Cherry", "Grape", "Peach", "Raspberry", "Strawberry")
TEST_CROPS = ("Orange", "Pepper", "Soybean", "Squash", "Tomato")
PER_CLASS = 100

TPR_TARGET = 0.95  # share of genuine validation images that must be accepted
K = 10
METHODS = ("msp", "energy", "knn")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def images_in(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def processed_split(split):
    paths, crops = [], []
    for class_dir in sorted(p for p in (PROCESSED / split).iterdir() if p.is_dir()):
        for path in images_in(class_dir):
            paths.append(str(path))
            crops.append(parse_prediction(class_dir.name)[0])
    return paths, crops


def unsupported_images(crop_names, rng):
    paths = []
    for class_dir in sorted(p for p in (PLANTVILLAGE / "train").iterdir() if p.is_dir()):
        if class_dir.name.split("_")[0].split(",")[0] in crop_names:
            images = images_in(class_dir)
            paths.extend(str(p) for p in rng.sample(images, min(PER_CLASS, len(images))))
    return paths


def main():
    rng = random.Random(42)
    guard = InputGuard(YOLO(str(WEIGHTS)), index_path=Path("/nonexistent"))

    sets = {}
    for split in ("train", "val", "test"):
        paths, crops = processed_split(split)
        sets[split] = {"paths": paths, "crops": crops}
    sets["ood_selection"] = {"paths": unsupported_images(SELECTION_CROPS, rng)}
    sets["ood_test"] = {"paths": unsupported_images(TEST_CROPS, rng)}

    for name, data in sets.items():
        print(f"Extracting features: {name} ({len(data['paths'])} images)")
        data["logits"], data["features"] = guard.forward(data["paths"])

    bank = normalise(sets["train"]["features"]).astype(np.float32)

    def scores(method, name):
        return ood_scores(method, sets[name]["logits"], sets[name]["features"], bank, K)

    rows = []
    for method in METHODS:
        val = scores(method, "val")
        threshold = float(np.quantile(val, 1 - TPR_TARGET))
        selection = scores(method, "ood_selection")
        test, ood_test = scores(method, "test"), scores(method, "ood_test")

        is_plantvillage = np.array(["___" in Path(p).name for p in sets["test"]["paths"]])
        rows.append({
            "method": method,
            "threshold": threshold,
            "selection_auroc": roc_auc_score(
                np.r_[np.ones(len(val)), np.zeros(len(selection))], np.r_[val, selection]
            ),
            "test_auroc": roc_auc_score(
                np.r_[np.ones(len(test)), np.zeros(len(ood_test))], np.r_[test, ood_test]
            ),
            "test_supported_accepted": float((test >= threshold).mean()),
            "test_plantvillage_accepted": float((test[is_plantvillage] >= threshold).mean()),
            "test_field_accepted": float((test[~is_plantvillage] >= threshold).mean()),
            "test_unsupported_rejected": float((ood_test < threshold).mean()),
        })

    report = pd.DataFrame(rows)
    best = report.loc[report.selection_auroc.idxmax()]

    # How often the true crop would fail the crop-selector check on genuine test images
    test_probs = np.exp(sets["test"]["logits"] - sets["test"]["logits"].max(1, keepdims=True))
    test_probs /= test_probs.sum(1, keepdims=True)
    true_mass = np.array([
        test_probs[i, guard.class_crops == crop].sum() for i, crop in enumerate(sets["test"]["crops"])
    ])
    mismatch_rate = float((true_mass < InputGuard.CROP_MASS_MIN).mean())

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT, index=False)
    np.savez_compressed(
        DEFAULT_INDEX,
        method=best.method, threshold=best.threshold, k=K, bank=bank.astype(np.float16),
        tpr_target=TPR_TARGET, built=str(date.today()), weights=str(WEIGHTS.relative_to(PROJECT_DIR)),
    )

    pd.set_option("display.width", 200)
    print("\n" + report.round(4).to_string(index=False))
    print(f"\nSelected method: {best.method} (threshold {best.threshold:.4f})")
    print(f"Genuine test images that would fail the crop check: {mismatch_rate:.2%}")
    print(f"Saved index to {DEFAULT_INDEX} and report to {REPORT}")


if __name__ == "__main__":
    main()
