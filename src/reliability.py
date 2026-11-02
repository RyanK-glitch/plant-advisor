"""Calibration, abstention and unknown-input (OOD) checks for the YOLO11-cls model.

Ultralytics' predict() keeps only softmax probabilities, so LogitCapture hooks the
Classify head's final linear layer to recover the raw logits and the 1280-d feature
vector that feeds it. Everything else here is plain numpy so it can be unit tested.

Decision order used by the app (ReliabilityGate.assess):
    1. Unknown input  - kNN feature distance too far from every training image -> no advice
    2. Low confidence - calibrated confidence below the abstention threshold   -> no advice
    3. Otherwise      - diagnosis continues to InputGuard, RAG and the rule engine
"""
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_DIR / "models" / "reliability.json"
DEFAULT_BANK = PROJECT_DIR / "models" / "ood_feature_bank.npy"


# --------------------------------------------------------------------------- logits

class LogitCapture:
    """Records the input (feature) and output (raw logits) of Classify.linear on every forward pass."""

    def __init__(self, yolo, imgsz=None, device=None):
        # predict() runs a separate copy of the network held by yolo.predictor, so run one
        # warm-up prediction to create it, then hook that copy rather than yolo.model
        self.yolo = yolo
        if getattr(yolo, "predictor", None) is None:
            yolo.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False, device=device,
                         imgsz=imgsz or yolo.overrides.get("imgsz", 224))
        self.features, self.logits = [], []
        self.linear, self.handle = None, None
        self.sync()

    def sync(self):
        """Re-attach if Ultralytics rebuilt its predictor (e.g. after a device change)."""
        linear = self.yolo.predictor.model.model.model[-1].linear
        if linear is not self.linear:
            if self.handle is not None:
                self.handle.remove()
            self.linear = linear
            self.handle = linear.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.features.append(inputs[0].detach().float().cpu().numpy())
        self.logits.append(output.detach().float().cpu().numpy())

    def pop(self):
        """Return (logits, features) captured since the last pop, then clear the buffers."""
        if not self.logits:
            raise RuntimeError("No forward pass was captured; check the model is a YOLO classify model.")
        logits, features = np.concatenate(self.logits), np.concatenate(self.features)
        self.features, self.logits = [], []
        return logits, features

    def remove(self):
        self.handle.remove()


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------- calibration

def nll(logits, labels, temperature=1.0):
    p = softmax(logits, temperature)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None))))


def fit_temperature(logits, labels, low=0.05, high=20.0):
    """One temperature T minimising validation NLL (Guo et al., 2017). Golden-section search on log T."""
    a, b = np.log(low), np.log(high)
    ratio = (np.sqrt(5) - 1) / 2
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    for _ in range(80):
        if nll(logits, labels, np.exp(c)) < nll(logits, labels, np.exp(d)):
            b = d
        else:
            a = c
        c, d = b - ratio * (b - a), a + ratio * (b - a)
    return float(np.exp((a + b) / 2))


def calibration_metrics(probs, labels, n_bins=15):
    """Accuracy, ECE, MCE (equal-width bins on top-1 confidence), NLL and multiclass Brier score."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == labels).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            bins.append({"lo": lo, "hi": hi, "count": 0, "accuracy": np.nan, "confidence": np.nan})
            continue
        acc, avg_conf = correct[mask].mean(), conf[mask].mean()
        gap = abs(acc - avg_conf)
        ece += mask.mean() * gap
        mce = max(mce, gap)
        bins.append({"lo": lo, "hi": hi, "count": int(mask.sum()), "accuracy": acc, "confidence": avg_conf})

    onehot = np.eye(probs.shape[1])[labels]
    return {
        "accuracy": float(correct.mean()),
        "ece": float(ece),
        "mce": float(mce),
        "nll": float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)))),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "mean_confidence": float(conf.mean()),
        "bins": bins,
    }


# ------------------------------------------------------------------------ abstention

def risk_coverage(confidence, correct):
    """Selective risk at every coverage level when the least confident images are rejected first."""
    order = np.argsort(-np.asarray(confidence), kind="stable")
    errors = 1 - np.asarray(correct, dtype=float)[order]
    n = len(errors)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    aurc = float(np.mean(risk))
    # Excess AURC over an oracle that rejects every error first (Geifman et al., 2019)
    err_rate = errors.mean()
    oracle = np.maximum(0, (np.arange(1, n + 1) - (n - errors.sum())) / np.arange(1, n + 1))
    return {"coverage": coverage, "risk": risk, "aurc": aurc,
            "e_aurc": float(aurc - oracle.mean()), "full_risk": float(err_rate)}


def threshold_for_risk(confidence, correct, target_risk):
    """Lowest confidence threshold whose accepted set has selective risk <= target (max coverage)."""
    confidence = np.asarray(confidence)
    correct = np.asarray(correct, dtype=float)
    best = None
    for t in np.unique(confidence):
        keep = confidence >= t
        if keep.sum() == 0:
            continue
        risk = 1 - correct[keep].mean()
        if risk <= target_risk:
            best = float(t)
            break
    return best if best is not None else float(confidence.max())


def selective_summary(confidence, correct, threshold):
    keep = np.asarray(confidence) >= threshold
    correct = np.asarray(correct, dtype=float)
    return {
        "threshold": float(threshold),
        "coverage": float(keep.mean()),
        "selective_risk": float(1 - correct[keep].mean()) if keep.any() else float("nan"),
        "selective_accuracy": float(correct[keep].mean()) if keep.any() else float("nan"),
        "abstained": int((~keep).sum()),
        "errors_caught": int(((~keep) & (correct == 0)).sum()),
        "total_errors": int((correct == 0).sum()),
    }


# ---------------------------------------------------------------------- OOD scores
# Convention: higher score = more in-distribution (more like the training data).

def msp_score(logits, temperature=1.0):
    return softmax(logits, temperature).max(axis=1)


def maxlogit_score(logits):
    return np.asarray(logits).max(axis=1)


def energy_score(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max(axis=1, keepdims=True)
    return (temperature * (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))).ravel()


def l2_normalise(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class KNNScorer:
    """Deep nearest-neighbour OOD score (Sun et al., 2022): minus the distance to the k-th
    nearest normalised training feature."""

    def __init__(self, bank, k=10):
        self.bank = l2_normalise(bank)
        self.k = k

    def score(self, features, batch=512, exclude_self=False):
        feats = l2_normalise(features)
        out = np.empty(len(feats), dtype=np.float32)
        kth = self.k + (1 if exclude_self else 0)
        for start in range(0, len(feats), batch):
            sims = feats[start:start + batch] @ self.bank.T
            dist = np.sqrt(np.clip(2 - 2 * sims, 0, None))
            out[start:start + batch] = -np.partition(dist, kth - 1, axis=1)[:, kth - 1]
        return out


def ood_metrics(id_scores, ood_scores, tpr=0.95):
    """AUROC with in-distribution as the positive class, and the share of OOD images still
    accepted when the threshold keeps 95% of in-distribution images (FPR@95TPR)."""
    from sklearn.metrics import roc_auc_score
    id_scores, ood_scores = np.asarray(id_scores), np.asarray(ood_scores)
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    threshold = np.quantile(id_scores, 1 - tpr)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "fpr_at_95tpr": float(np.mean(ood_scores >= threshold)),
        "threshold_at_95tpr": float(threshold),
    }


# --------------------------------------------------------------------- runtime gate

class ReliabilityGate:
    OK = "OK"
    UNKNOWN_INPUT = "UNKNOWN_INPUT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

    def __init__(self, config_path=DEFAULT_CONFIG, bank_path=DEFAULT_BANK):
        self.config = json.loads(Path(config_path).read_text())
        self.temperature = float(self.config.get("temperature", 1.0))
        self.abstain_threshold = float(self.config.get("abstain_threshold", 0.0))
        self.abstain_threshold_crop = float(
            self.config.get("abstain_threshold_crop_selected", self.abstain_threshold))
        self.knn_threshold = self.config.get("knn_threshold")
        self.knn = None
        if self.knn_threshold is not None and Path(bank_path).is_file():
            self.knn = KNNScorer(np.load(bank_path), k=int(self.config.get("knn_k", 10)))

    @classmethod
    def load_if_available(cls, config_path=DEFAULT_CONFIG, bank_path=DEFAULT_BANK, model_path=None):
        """Load the gate only if it was calibrated for the model the app is using.

        Temperature, thresholds and the kNN feature bank all belong to one set of weights;
        used with another model they would refuse almost every photo. On a mismatch the app
        runs without the reliability checks (as before calibration) and prints a warning.
        """
        if not Path(config_path).is_file():
            return None
        calibrated_for = json.loads(Path(config_path).read_text()).get("model")
        if model_path is not None and calibrated_for:
            if Path(model_path).resolve() != (PROJECT_DIR / calibrated_for).resolve():
                print(f"Reliability checks OFF: {Path(config_path).name} was calibrated for {calibrated_for}, "
                      f"not {model_path}. Re-run the calibration scripts for this model.")
                return None
        return cls(config_path, bank_path)

    def calibrated(self, logits):
        return softmax(np.atleast_2d(logits), self.temperature)[0]

    def is_unknown(self, feature):
        if self.knn is None:
            return False, None
        score = float(self.knn.score(np.atleast_2d(feature))[0])
        return score < float(self.knn_threshold), score

    def threshold(self, crop_selected):
        return self.abstain_threshold_crop if crop_selected else self.abstain_threshold
