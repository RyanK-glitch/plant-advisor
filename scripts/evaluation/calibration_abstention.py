"""Calibration (temperature scaling) and abstention (risk-coverage) for the YOLO11-cls model.

Reads the logit caches written by extract_outputs.py:
    reports/reliability/cache/val.npz   - used to FIT the temperature and CHOOSE thresholds
    reports/reliability/cache/test.npz  - held out, used only to REPORT results

Outputs (reports/reliability/):
    calibration_metrics.csv          accuracy, ECE, MCE, NLL, Brier before/after, val and test
    reliability_diagram.png          test-set reliability diagram before vs after scaling
    risk_coverage.png                selective risk vs coverage (auto-detect and crop-selected modes)
    abstention_thresholds.csv        thresholds chosen on val for each target risk, applied to test
    abstention_per_class.csv         how often each class is withheld at the chosen threshold
and updates models/reliability.json, which the app reads at start-up.

Usage:
    python scripts/evaluation/calibration_abstention.py
    python scripts/evaluation/calibration_abstention.py --target-risk 0.01
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.reliability import (DEFAULT_CONFIG, calibration_metrics, fit_temperature,  # noqa: E402
                             risk_coverage, selective_summary, softmax, threshold_for_risk)

CACHE = PROJECT_DIR / "reports/reliability/cache"
OUT = PROJECT_DIR / "reports/reliability"
TARGETS = (0.005, 0.01, 0.02, 0.03)


def load(names):
    """One cache name, or a comma list that is concatenated (e.g. 'val_lab,val_field')."""
    parts = [np.load(CACHE / f"{n}.npz") for n in names.split(",")]
    logits = np.concatenate([d["logits"] for d in parts]).astype(np.float64)
    labels = np.concatenate([d["labels"] for d in parts])
    sets = np.concatenate([[n] * len(d["labels"]) for n, d in zip(names.split(","), parts)])
    return logits, labels, list(parts[0]["class_keys"]), sets


def crop_restricted(probs, labels, keys):
    """Simulate a farmer who selects the correct crop: keep that crop's classes and renormalise."""
    crops = np.array([key.split("_")[0] for key in keys])
    mask = crops[None, :] == crops[labels][:, None]
    restricted = np.where(mask, probs, 0.0)
    return restricted / restricted.sum(axis=1, keepdims=True)


def reliability_plot(raw, cal, temperature, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (title, metrics) in zip(axes, [("Before (T = 1)", raw), (f"After temperature scaling (T = {temperature:.2f})", cal)]):
        bins = [b for b in metrics["bins"] if b["count"]]
        centres = [(b["lo"] + b["hi"]) / 2 for b in bins]
        width = bins[0]["hi"] - bins[0]["lo"] if bins else 1 / 15
        acc = np.array([b["accuracy"] for b in bins])
        conf = np.array([b["confidence"] for b in bins])
        ax.bar(centres, acc, width=width * 0.95, color="#2a9d8f", edgecolor="white", label="Accuracy in bin")
        # Gap drawn between accuracy and mean confidence, whichever is higher
        ax.bar(centres, np.abs(conf - acc), bottom=np.minimum(acc, conf), width=width * 0.95,
               facecolor="none", edgecolor="#e76f51", hatch="///", lw=0.8, label="Gap to mean confidence")
        for x, a, c, b in zip(centres, acc, conf, bins):
            ax.text(x, max(a, c) + 0.015, str(b["count"]), ha="center", va="bottom", fontsize=6, color="#444")
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Perfect calibration")
        ax.set_title(f"{title}\nECE {metrics['ece']:.4f} · accuracy {metrics['accuracy']:.4f}", fontsize=10)
        ax.set_xlabel("Confidence (top-1 probability)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Test-set reliability diagram (15 equal-width bins; numbers = images per bin)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-risk", type=float, default=0.01,
                        help="maximum error rate among answered images, chosen on val (default 1%%)")
    parser.add_argument("--cache-dir", default=None, help="default reports/reliability/cache")
    parser.add_argument("--out-dir", default=None, help="default reports/reliability")
    parser.add_argument("--val", default="val", help="cache name(s) used to fit, comma-separated")
    parser.add_argument("--test", default="test", help="cache name(s) used to report, comma-separated")
    parser.add_argument("--model", help="weights path (relative to the project) this calibration belongs to")
    args = parser.parse_args()
    global CACHE, OUT
    CACHE = Path(args.cache_dir) if args.cache_dir else CACHE
    OUT = Path(args.out_dir) if args.out_dir else OUT
    OUT.mkdir(parents=True, exist_ok=True)

    val_logits, val_labels, keys, _ = load(args.val)
    test_logits, test_labels, _, test_sets = load(args.test)
    test_parts = [] if len(set(test_sets)) < 2 else [
        (f"test: {n}", test_logits[test_sets == n], test_labels[test_sets == n]) for n in dict.fromkeys(test_sets)]

    # 1. Calibration -----------------------------------------------------------------
    temperature = fit_temperature(val_logits, val_labels)
    rows, test_metrics = [], {}
    for split, logits, labels in [("val", val_logits, val_labels), ("test", test_logits, test_labels)] + test_parts:
        for name, t in [("raw (T=1)", 1.0), ("temperature scaled", temperature)]:
            m = calibration_metrics(softmax(logits, t), labels)
            test_metrics[(split, name)] = m
            rows.append({"split": split, "model": name, "temperature": t,
                         **{k: v for k, v in m.items() if k != "bins"}})
    calib = pd.DataFrame(rows)
    calib.to_csv(OUT / "calibration_metrics.csv", index=False)
    reliability_plot(test_metrics[("test", "raw (T=1)")], test_metrics[("test", "temperature scaled")],
                     temperature, OUT / "reliability_diagram.png")
    print(f"Temperature fitted on val: T = {temperature:.4f}\n")
    print(calib.round(4).to_string(index=False))

    # 2. Abstention --------------------------------------------------------------------
    modes = {}
    for mode in ("auto-detect", "crop selected"):
        per_split = {}
        for split, logits, labels in [("val", val_logits, val_labels), ("test", test_logits, test_labels)]:
            probs = softmax(logits, temperature)
            if mode == "crop selected":
                probs = crop_restricted(probs, labels, keys)
            per_split[split] = (probs.max(axis=1), (probs.argmax(axis=1) == labels).astype(int))
        modes[mode] = per_split

    threshold_rows, chosen = [], {}
    for mode, per_split in modes.items():
        for target in sorted(set(TARGETS) | {args.target_risk}):
            t = threshold_for_risk(*per_split["val"], target)
            val_s = selective_summary(*per_split["val"], t)
            test_s = selective_summary(*per_split["test"], t)
            threshold_rows.append({
                "mode": mode, "target_risk_on_val": target, "threshold": t,
                "val_coverage": val_s["coverage"], "val_risk": val_s["selective_risk"],
                "test_coverage": test_s["coverage"], "test_risk": test_s["selective_risk"],
                "test_accuracy_answered": test_s["selective_accuracy"],
                "test_abstained": test_s["abstained"],
                "test_errors_caught": test_s["errors_caught"], "test_total_errors": test_s["total_errors"],
            })
            if target == args.target_risk:
                chosen[mode] = t
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT / "abstention_thresholds.csv", index=False)
    print("\nAbstention thresholds (chosen on val, reported on test):")
    print(thresholds.round(4).to_string(index=False))

    # Risk-coverage curves on test
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    aurc_rows = []
    for mode, colour in [("auto-detect", "#264653"), ("crop selected", "#e9c46a")]:
        conf, correct = modes[mode]["test"]
        rc = risk_coverage(conf, correct)
        aurc_rows.append({"mode": mode, "aurc": rc["aurc"], "e_aurc": rc["e_aurc"], "full_risk": rc["full_risk"]})
        ax.plot(rc["coverage"], rc["risk"] * 100, color=colour, lw=2,
                label=f"{mode} (AURC {rc['aurc'] * 1000:.2f}×10⁻³)")
        t = chosen[mode]
        s = selective_summary(conf, correct, t)
        ax.scatter([s["coverage"]], [s["selective_risk"] * 100], color=colour, edgecolor="black", zorder=3)
    ax.axhline(args.target_risk * 100, ls=":", color="#e76f51", label=f"target risk {args.target_risk:.1%}")
    ax.set_xlabel("Coverage (share of images answered)")
    ax.set_ylabel("Selective risk (% errors among answered)")
    ax.set_title("Risk–coverage on the test split (calibrated confidence)", fontsize=10)
    ax.set_xlim(0.5, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(aurc_rows).to_csv(OUT / "risk_coverage_aurc.csv", index=False)

    # Per-class abstention at the chosen auto-detect threshold
    conf, correct = modes["auto-detect"]["test"]
    keep = conf >= chosen["auto-detect"]
    per_class = pd.DataFrame({
        "class": [keys[i] for i in test_labels], "abstained": ~keep, "correct": correct,
    }).groupby("class").agg(images=("correct", "size"), abstained=("abstained", "sum"),
                            accuracy_all=("correct", "mean"))
    per_class["abstention_rate"] = per_class.abstained / per_class.images
    per_class.to_csv(OUT / "abstention_per_class.csv")

    # Chosen threshold on each test set separately (e.g. lab vs field photos)
    by_set = []
    for name in dict.fromkeys(test_sets):
        m = test_sets == name
        by_set.append({"test_set": name, "images": int(m.sum()),
                       **selective_summary(conf[m], correct[m], chosen["auto-detect"])})
    pd.DataFrame(by_set).to_csv(OUT / "abstention_by_test_set.csv", index=False)
    print("\nChosen threshold per test set (auto-detect):")
    print(pd.DataFrame(by_set).round(4).to_string(index=False))

    # 3. Save the runtime configuration ---------------------------------------------------
    config = json.loads(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.is_file() else {}
    if args.model and config.get("model") != args.model:
        config = {"model": args.model}  # settings of another model must not be mixed in
    config.update({
        "created": date.today().isoformat(),
        "temperature": round(temperature, 6),
        "target_risk": args.target_risk,
        "abstain_threshold": round(chosen["auto-detect"], 6),
        "abstain_threshold_crop_selected": round(chosen["crop selected"], 6),
        "calibration_fitted_on": args.val,
    })
    DEFAULT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG.write_text(json.dumps(config, indent=2))
    print(f"\nSaved {DEFAULT_CONFIG}\nFigures and tables in {OUT}")


if __name__ == "__main__":
    main()
