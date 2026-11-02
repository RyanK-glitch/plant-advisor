"""Cross-dataset field test: how the model does on photos from a different source.

Steps
    1. Remove near-duplicates: any field image whose perceptual hash (pHash, 64-bit) is within
       --max-hamming bits of an image in data/processed/{train,val,test} is excluded, so the
       field score cannot be inflated by images the model has already seen.
    2. Score the remaining images in two modes:
         auto-detect    - the model chooses among all 14 classes
         crop selected  - the farmer picks the right crop (as InputGuard does in the app)
    3. Report accuracy and macro-F1 with stratified bootstrap 95% CIs, per-class P/R/F1,
       calibration (ECE) on field images, and what the reliability gate does to them,
       next to the in-domain test score for the SAME classes.

Needs the cache from extract_outputs.py, e.g. for PlantDoc:
    python scripts/evaluation/extract_outputs.py --images path/to/PlantDoc --name field_plantdoc \
        --mapping scripts/evaluation/config/plantdoc_mapping.json
    python scripts/evaluation/field_test.py --field field_plantdoc

Outputs go to reports/reliability/field_<name>/.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.reliability import DEFAULT_BANK, DEFAULT_CONFIG, KNNScorer, calibration_metrics, softmax  # noqa: E402

CACHE = PROJECT_DIR / "reports/reliability/cache"
REFERENCE_DIRS = ["data/processed/train", "data/processed/val", "data/processed/test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def phash_table(paths):
    import imagehash
    from PIL import Image
    rows = []
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                rows.append({"path": str(path), "phash": str(imagehash.phash(image.convert("RGB")))})
        except Exception as error:  # unreadable file: keep it out of the comparison
            print(f"  could not hash {path}: {error}")
        if i % 500 == 0:
            print(f"  hashed {i}/{len(paths)}", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


def reference_hashes(cache_file):
    if cache_file.is_file():
        return pd.read_csv(cache_file)
    paths = [p for d in REFERENCE_DIRS for p in (PROJECT_DIR / d).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    print(f"Hashing {len(paths)} training/val/test images (cached afterwards)...")
    table = phash_table(paths)
    table.to_csv(cache_file, index=False)
    return table


def hamming_min(field_hashes, ref_hashes):
    ref = np.array([int(h, 16) for h in ref_hashes], dtype=np.uint64)
    out = []
    for h in field_hashes:
        x = np.bitwise_xor(ref, np.uint64(int(h, 16)))
        bits = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        out.append(int(bits.min()))
    return np.array(out)


def bootstrap_ci(y_true, y_pred, labels, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    accs, f1s = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        accs.append(np.mean(y_true[idx] == y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def evaluate(labels, probs, keys, present):
    pred = probs.argmax(axis=1)
    y_true = [keys[i] for i in labels]
    y_pred = [keys[i] for i in pred]
    acc = float(np.mean(labels == pred))
    f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    acc_ci, f1_ci = bootstrap_ci(y_true, y_pred, present)
    return {"accuracy": acc, "accuracy_ci95": [float(x) for x in acc_ci],
            "macro_f1": float(f1), "macro_f1_ci95": [float(x) for x in f1_ci]}, y_true, y_pred


def restrict_to_true_crop(probs, labels, keys):
    crops = np.array([k.split("_")[0] for k in keys])
    mask = crops[None, :] == crops[labels][:, None]
    p = np.where(mask, probs, 0.0)
    return p / p.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", required=True, help="cache name, e.g. field_plantdoc")
    parser.add_argument("--max-hamming", type=int, default=8, help="pHash distance treated as a duplicate")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--reference-hashes", help="CSV of path,phash for the training/val/test images")
    parser.add_argument("--cache-dir", default=None, help="default reports/reliability/cache")
    parser.add_argument("--out-dir", default=None, help="results go to <out-dir>/<field>; default reports/reliability")
    parser.add_argument("--in-domain", default="test", help="lab test cache to compare against")
    args = parser.parse_args()

    global CACHE
    CACHE = Path(args.cache_dir) if args.cache_dir else CACHE
    out = (Path(args.out_dir) if args.out_dir else PROJECT_DIR / "reports/reliability") / args.field
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE / f"{args.field}.npz")
    keys = list(data["class_keys"])
    paths, labels, logits = data["paths"], data["labels"], data["logits"].astype(np.float64)
    features = data["features"].astype(np.float32)

    # 1. Near-duplicate removal --------------------------------------------------------
    keep = np.ones(len(paths), dtype=bool)
    if not args.skip_dedup:
        ref_file = Path(args.reference_hashes) if args.reference_hashes else CACHE / "phash_reference.csv"
        ref = reference_hashes(ref_file)
        field = phash_table(paths)
        distance = hamming_min(field.phash, ref.phash)
        dup = pd.DataFrame({"path": field.path, "min_hamming_to_train_val_test": distance})
        dup["duplicate"] = dup.min_hamming_to_train_val_test <= args.max_hamming
        dup.to_csv(out / "dedup_check.csv", index=False)
        duplicated = set(dup.path[dup.duplicate])
        keep = np.array([str(p) not in duplicated for p in paths])
        print(f"Near-duplicates removed (pHash Hamming <= {args.max_hamming}): {(~keep).sum()} of {len(paths)}")
    paths, labels, logits, features = paths[keep], labels[keep], logits[keep], features[keep]

    config = json.loads(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.is_file() else {}
    temperature = float(config.get("temperature", 1.0))
    probs = softmax(logits, temperature)
    present = sorted({keys[i] for i in labels})

    # 2. Field scores --------------------------------------------------------------------
    summary = {"field_set": args.field, "images_after_dedup": int(len(labels)),
               "classes": present, "temperature": temperature}
    reports = {}
    for mode, p in [("auto-detect", probs), ("crop selected", restrict_to_true_crop(probs, labels, keys))]:
        summary[mode], y_true, y_pred = evaluate(labels, p, keys, present)
        reports[mode] = pd.DataFrame(classification_report(
            y_true, y_pred, labels=present, output_dict=True, zero_division=0)).T
        reports[mode].to_csv(out / f"per_class_{mode.replace(' ', '_')}.csv")
        if mode == "auto-detect":
            all_labels = sorted(set(y_true) | set(y_pred), key=keys.index)
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)
            pd.DataFrame(cm, index=all_labels, columns=all_labels).to_csv(out / "confusion_matrix.csv")
            fig, ax = plt.subplots(figsize=(9, 6))
            rows = [all_labels.index(c) for c in present]
            cm_rows = cm[rows] / cm[rows].sum(axis=1, keepdims=True)
            ax.imshow(cm_rows, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(all_labels)), all_labels, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(present)), present, fontsize=8)
            for i in range(len(present)):
                for j in range(len(all_labels)):
                    if cm[rows[i], j]:
                        ax.text(j, i, cm[rows[i], j], ha="center", va="center", fontsize=7,
                                color="white" if cm_rows[i, j] > 0.5 else "black")
            ax.set_title(f"{args.field}: field confusion (auto-detect), accuracy {summary[mode]['accuracy']:.1%}",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "confusion_matrix.png", dpi=200)
            plt.close(fig)

    # Calibration on field images (does temperature scaling still hold under shift?)
    summary["ece_raw"] = calibration_metrics(softmax(logits), labels)["ece"]
    summary["ece_temperature_scaled"] = calibration_metrics(probs, labels)["ece"]
    summary["mean_confidence"] = float(probs.max(axis=1).mean())

    # In-domain comparison on the same classes
    test = np.load(CACHE / f"{args.in_domain}.npz")
    idx = [keys.index(c) for c in present]
    mask = np.isin(test["labels"], idx)
    if mask.any():
        tp = softmax(test["logits"][mask].astype(np.float64), temperature)
        summary["in_domain_test_same_classes"], _, _ = evaluate(test["labels"][mask], tp, keys, present)
        summary["drop_accuracy_points"] = 100 * (summary["in_domain_test_same_classes"]["accuracy"]
                                                 - summary["auto-detect"]["accuracy"])

    # What the reliability gate does to field photos
    if "knn_threshold" in config and DEFAULT_BANK.is_file():
        knn = KNNScorer(np.load(DEFAULT_BANK), k=int(config.get("knn_k", 10)))
        unknown = knn.score(features) < float(config["knn_threshold"])
        low = (~unknown) & (probs.max(axis=1) < float(config.get("abstain_threshold", 0)))
        answered = ~unknown & ~low
        correct = probs.argmax(axis=1) == labels
        summary["gate"] = {
            "rejected_unknown": float(unknown.mean()), "abstained_low_confidence": float(low.mean()),
            "answered": float(answered.mean()),
            "accuracy_when_answered": float(correct[answered].mean()) if answered.any() else None,
            "wrong_answers_given": float((answered & ~correct).mean()),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results in {out}")


if __name__ == "__main__":
    main()
