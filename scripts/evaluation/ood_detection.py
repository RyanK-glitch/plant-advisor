"""Unknown-input (out-of-distribution) detection: compare scores, pick a threshold, test the gate.

Reads caches written by extract_outputs.py:
    train.npz     feature bank for kNN / Mahalanobis (the training images)
    val.npz       in-distribution images used to SET the threshold
    test.npz      in-distribution images used to REPORT results
    ood_*.npz     images the model was never trained for (other crops, non-leaf photos, ...)

Scores compared (higher = more like the training data):
    MSP (raw and temperature-scaled), MaxLogit, Energy (Liu et al., 2020),
    Mahalanobis with Ledoit-Wolf shrinkage (Lee et al., 2018), kNN distance (Sun et al., 2022)

Outputs (reports/reliability/):
    ood_metrics.csv          AUROC and FPR@95TPR for every score and OOD set
    ood_auroc.png            bar chart of AUROC
    ood_knn_histogram.png    kNN score distributions, ID test vs each OOD set
    gate_outcomes.csv        what the full gate (unknown -> abstain -> answer) does to each set
and updates models/reliability.json and models/ood_feature_bank.npy for the app.

Usage:
    python scripts/evaluation/ood_detection.py
    python scripts/evaluation/ood_detection.py --k 10 --id-keep 0.95
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
from sklearn.covariance import LedoitWolf

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.reliability import (DEFAULT_BANK, DEFAULT_CONFIG, KNNScorer, energy_score,  # noqa: E402
                             l2_normalise, maxlogit_score, msp_score, ood_metrics, softmax)

CACHE = PROJECT_DIR / "reports/reliability/cache"
OUT = PROJECT_DIR / "reports/reliability"


def load(names):
    """One cache name, or a comma list that is concatenated (e.g. 'train_lab,train_field')."""
    parts = [np.load(CACHE / f"{n}.npz") for n in names.split(",")]
    return {"logits": np.concatenate([d["logits"] for d in parts]).astype(np.float64),
            "features": np.concatenate([d["features"] for d in parts]).astype(np.float32),
            "labels": np.concatenate([d["labels"] for d in parts]),
            "sets": np.concatenate([[n] * len(d["labels"]) for n, d in zip(names.split(","), parts)])}


def subset(data, mask):
    return {k: v[mask] for k, v in data.items()}


class Mahalanobis:
    def __init__(self, features, labels):
        feats = l2_normalise(features)
        self.means = np.stack([feats[labels == c].mean(axis=0) for c in np.unique(labels)])
        centred = feats - self.means[np.searchsorted(np.unique(labels), labels)]
        self.precision = LedoitWolf().fit(centred).precision_

    def score(self, features):
        feats = l2_normalise(features)
        best = np.full(len(feats), np.inf)
        for mean in self.means:
            d = feats - mean
            best = np.minimum(best, np.einsum("ij,jk,ik->i", d, self.precision, d))
        return -best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="neighbour used by the app's kNN gate")
    parser.add_argument("--id-keep", type=float, default=0.95,
                        help="share of genuine val images the unknown-input gate must accept")
    parser.add_argument("--cache-dir", default=None, help="default reports/reliability/cache")
    parser.add_argument("--out-dir", default=None, help="default reports/reliability")
    parser.add_argument("--train", default="train", help="feature-bank cache name(s), comma-separated")
    parser.add_argument("--val", default="val", help="cache name(s) used to set the threshold")
    parser.add_argument("--test", default="test", help="cache name(s) used to report")
    parser.add_argument("--model", help="weights path this calibration belongs to (must match reliability.json)")
    args = parser.parse_args()
    global CACHE, OUT
    CACHE = Path(args.cache_dir) if args.cache_dir else CACHE
    OUT = Path(args.out_dir) if args.out_dir else OUT
    OUT.mkdir(parents=True, exist_ok=True)

    config = json.loads(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.is_file() else {}
    if args.model and config.get("model") != args.model:
        sys.exit(f"{DEFAULT_CONFIG} belongs to {config.get('model')}; run calibration_abstention.py "
                 f"--model {args.model} first.")
    temperature = float(config.get("temperature", 1.0))
    abstain = float(config.get("abstain_threshold", 0.0))

    train, val, test = load(args.train), load(args.val), load(args.test)
    ood_sets = {p.stem.removeprefix("ood_"): load(p.stem) for p in sorted(CACHE.glob("ood_*.npz"))}
    if not ood_sets:
        sys.exit("No ood_*.npz caches found. Run extract_outputs.py with --ood first.")

    knn = {k: KNNScorer(train["features"], k=k) for k in sorted({1, 5, args.k, 50})}
    maha = Mahalanobis(train["features"], train["labels"])
    scorers = {
        "MSP (raw)": lambda d: msp_score(d["logits"]),
        f"MSP (T={temperature:.2f})": lambda d: msp_score(d["logits"], temperature),
        "MaxLogit": lambda d: maxlogit_score(d["logits"]),
        "Energy": lambda d: energy_score(d["logits"]),
        "Mahalanobis": lambda d: maha.score(d["features"]),
        **{f"kNN (k={k})": (lambda d, s=s: s.score(d["features"])) for k, s in knn.items()},
    }

    rows = []
    for method, fn in scorers.items():
        id_scores = fn(test)
        for ood_name, ood in ood_sets.items():
            rows.append({"method": method, "ood_set": ood_name, "n_id": len(id_scores),
                         "n_ood": len(ood["labels"]), **ood_metrics(id_scores, fn(ood))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "ood_metrics.csv", index=False)
    print(metrics.drop(columns="threshold_at_95tpr").round(4).to_string(index=False))

    pivot = metrics.pivot(index="method", columns="ood_set", values="auroc").loc[list(scorers)]
    ax = pivot.plot.barh(figsize=(8, 4.8), width=0.8)
    ax.axvline(0.5, ls=":", color="#777")
    ax.set_xlabel("AUROC (1.0 = perfect separation, 0.5 = chance)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Unknown-input detection on the test split vs OOD sets", fontsize=10)
    ax.invert_yaxis()
    ax.legend(title="OOD set", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "ood_auroc.png", dpi=200)
    plt.close()

    # Threshold for the app: kNN(k) score that keeps `id_keep` of genuine VAL images
    gate_knn = knn[args.k]
    knn_threshold = float(np.quantile(gate_knn.score(val["features"]), 1 - args.id_keep))

    fig, ax = plt.subplots(figsize=(7, 4))
    all_scores = np.concatenate([gate_knn.score(test["features"])] +
                                [gate_knn.score(o["features"]) for o in ood_sets.values()])
    bins = np.linspace(np.quantile(all_scores, 0.002) - 0.02, all_scores.max() + 0.02, 60)
    ax.hist(gate_knn.score(test["features"]), bins=bins, alpha=0.6, density=True, label="ID test (14 classes)")
    for name, ood in ood_sets.items():
        ax.hist(gate_knn.score(ood["features"]), bins=bins, alpha=0.45, density=True, label=f"OOD: {name}")
    ax.axvline(knn_threshold, color="black", ls="--", label=f"gate threshold ({args.id_keep:.0%} of val kept)")
    ax.set_xlabel(f"kNN score (minus distance to {args.k}th nearest training feature)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "ood_knn_histogram.png", dpi=200)
    plt.close(fig)

    # Full gate: unknown-input check first, then abstention on calibrated confidence
    def outcomes(name, data, in_dist):
        unknown = gate_knn.score(data["features"]) < knn_threshold
        probs = softmax(data["logits"], temperature)
        low = (~unknown) & (probs.max(axis=1) < abstain)
        answered = ~unknown & ~low
        row = {"set": name, "images": len(unknown), "rejected_unknown": unknown.mean(),
               "abstained_low_confidence": low.mean(), "answered": answered.mean()}
        if in_dist:
            correct = probs.argmax(axis=1) == data["labels"]
            row["answered_correct"] = (answered & correct).mean()
            row["answered_wrong"] = (answered & ~correct).mean()
            row["accuracy_when_answered"] = correct[answered].mean() if answered.any() else np.nan
        else:
            row["answered_wrong"] = answered.mean()  # any answer to an unknown image is unsafe
        return row

    per_test = [] if len(set(test["sets"])) < 2 else [
        outcomes(f"ID test: {n}", subset(test, test["sets"] == n), True) for n in dict.fromkeys(test["sets"])]
    gate = pd.DataFrame([outcomes("ID test", test, True)] + per_test +
                        [outcomes(f"OOD: {n}", d, False) for n, d in ood_sets.items()])
    gate.to_csv(OUT / "gate_outcomes.csv", index=False)
    print("\nGate outcomes (shares of each set):")
    print(gate.round(4).to_string(index=False))

    np.save(DEFAULT_BANK, train["features"].astype(np.float16))
    config.update({"knn_k": args.k, "knn_threshold": round(knn_threshold, 6), "knn_id_keep_on_val": args.id_keep,
                   "knn_bank_images": int(len(train["labels"]))})
    DEFAULT_CONFIG.write_text(json.dumps(config, indent=2))
    print(f"\nSaved {DEFAULT_CONFIG} and {DEFAULT_BANK}")


if __name__ == "__main__":
    main()
