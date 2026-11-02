"""Old model vs field-fine-tuned model on the same test images.

Compares accuracy and macro-F1 (with stratified bootstrap 95% CIs), per-class F1 and
confusion matrices on two test sets:
    lab test    - data/processed/test (1,779 lab images; neither model trained on them)
    field test  - plant-field-training/test_field (847 field photos; neither model trained on them)

Reads caches written by extract_outputs.py (raw predictions, no calibration, so this
measures the models themselves):
    old model: reports/reliability/cache/test.npz and test_field.npz
    new model: reports/reliability_field_ft/cache/test_lab.npz and test_field.npz

Usage:
    python scripts/evaluation/compare_models.py
Outputs: reports/model_comparison/ (summary.csv, per_class_f1.csv, confusion_matrices.png, summary.md)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "scripts/evaluation"))
from field_test import bootstrap_ci  # noqa: E402


def load(path):
    d = np.load(path)
    keys = list(d["class_keys"])
    y = d["labels"]
    pred = d["logits"].argmax(axis=1)
    return [keys[i] for i in y], [keys[i] for i in pred], keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cache", default=str(PROJECT_DIR / "reports/reliability/cache"))
    parser.add_argument("--new-cache", default=str(PROJECT_DIR / "reports/reliability_field_ft/cache"))
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "reports/model_comparison"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = {
        ("Old model", "Lab test"): Path(args.old_cache) / "test.npz",
        ("New model", "Lab test"): Path(args.new_cache) / "test_lab.npz",
        ("Old model", "Field test"): Path(args.old_cache) / "test_field.npz",
        ("New model", "Field test"): Path(args.new_cache) / "test_field.npz",
    }
    rows, f1_rows, matrices = [], {}, {}
    for (model, test_set), path in runs.items():
        if not path.is_file():
            print(f"missing {path} - skipped")
            continue
        y, pred, keys = load(path)
        present = sorted(set(y), key=keys.index)
        acc_ci, f1_ci = bootstrap_ci(y, pred, present)
        rows.append({
            "model": model, "test_set": test_set, "images": len(y),
            "accuracy": float(np.mean(np.array(y) == np.array(pred))),
            "accuracy_ci95": f"{acc_ci[0]:.4f}-{acc_ci[1]:.4f}",
            "macro_f1": f1_score(y, pred, labels=present, average="macro", zero_division=0),
            "macro_f1_ci95": f"{f1_ci[0]:.4f}-{f1_ci[1]:.4f}",
        })
        per = f1_score(y, pred, labels=present, average=None, zero_division=0)
        for cls, v in zip(present, per):
            f1_rows.setdefault((test_set, cls), {})[model] = v
        matrices[(model, test_set)] = (confusion_matrix(y, pred, labels=keys), keys, present)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    per_class = pd.DataFrame([{"test_set": t, "class": c, **v} for (t, c), v in f1_rows.items()])
    if {"Old model", "New model"} <= set(per_class.columns):
        per_class["change"] = per_class["New model"] - per_class["Old model"]
    per_class.to_csv(out / "per_class_f1.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print()
    print(per_class.round(3).to_string(index=False))

    # 2 x 2 grid of row-normalised confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, key in zip(axes.flat, [("Old model", "Lab test"), ("New model", "Lab test"),
                                   ("Old model", "Field test"), ("New model", "Field test")]):
        if key not in matrices:
            ax.axis("off")
            continue
        cm, keys, present = matrices[key]
        rows_idx = [keys.index(c) for c in present]
        cm_rows = cm[rows_idx]
        norm = cm_rows / cm_rows.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(keys)), keys, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(present)), present, fontsize=7)
        for i in range(len(present)):
            for j in range(len(keys)):
                if cm_rows[i, j]:
                    ax.text(j, i, cm_rows[i, j], ha="center", va="center", fontsize=6,
                            color="white" if norm[i, j] > 0.5 else "black")
        acc = summary[(summary.model == key[0]) & (summary.test_set == key[1])].accuracy.iloc[0]
        ax.set_title(f"{key[0]} · {key[1]} · accuracy {acc:.1%}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=170)
    plt.close(fig)

    lines = ["# Old vs field-fine-tuned model", "", "| Model | Test set | Images | Accuracy (95% CI) | Macro-F1 (95% CI) |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['test_set']} | {r['images']} | {r['accuracy']:.2%} ({r['accuracy_ci95']}) "
                     f"| {r['macro_f1']:.3f} ({r['macro_f1_ci95']}) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved results in {out}")


if __name__ == "__main__":
    main()
