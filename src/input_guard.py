from pathlib import Path

import numpy as np
import torch

from src.labels import parse_prediction

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = PROJECT_DIR / "models" / "ood_index.npz"


def ood_scores(method, logits, features, bank=None, k=10):
    # Higher score = more like the training leaves; each row is one image
    if method == "msp":
        return torch.softmax(torch.from_numpy(logits), dim=1).max(dim=1).values.numpy()
    if method == "energy":
        return torch.logsumexp(torch.from_numpy(logits), dim=1).numpy()
    if method == "knn":
        # Cosine similarity to the k-th nearest training image (Sun et al., 2022)
        sims = normalise(features) @ bank.T
        return np.sort(sims, axis=1)[:, -k]
    raise ValueError(f"Unknown OOD method: {method}")


def normalise(features):
    return features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-12)


class InputGuard:
    """Checks that a photo is a leaf of a supported crop before any advice is given."""

    OK = "OK"
    UNKNOWN = "UNKNOWN"
    CROP_MISMATCH = "CROP_MISMATCH"

    # The selected crop must hold at least half of the model's probability
    CROP_MASS_MIN = 0.5

    def __init__(self, model, index_path=DEFAULT_INDEX):
        self.model = model
        self.class_keys = [parse_prediction(model.names[i]) for i in range(len(model.names))]
        self.class_crops = np.array([crop for crop, _ in self.class_keys])

        # Capture the classifier's input features and raw logits on every forward pass
        self._captured = []
        model.model.model[-1].linear.register_forward_hook(
            lambda module, inputs, output: self._captured.append(
                (output.detach().float().cpu().numpy(), inputs[0].detach().float().cpu().numpy())
            )
        )

        self.index = None
        if Path(index_path).is_file():
            data = np.load(index_path)
            self.index = {
                "method": str(data["method"]),
                "threshold": float(data["threshold"]),
                "k": int(data["k"]),
                "bank": data["bank"].astype(np.float32),
            }

    @property
    def ood_ready(self):
        return self.index is not None

    def forward(self, images, batch=32):
        # Returns (logits, features) for a list of images or image paths
        logits, features = [], []
        for start in range(0, len(images), batch):
            self._captured.clear()
            self.model.predict(images[start:start + batch], batch=batch, verbose=False)
            logits.extend(item[0] for item in self._captured)
            features.extend(item[1] for item in self._captured)
        return np.concatenate(logits), np.concatenate(features)

    def crop_mass(self, probs):
        return {crop: float(probs[self.class_crops == crop].sum()) for crop in dict.fromkeys(self.class_crops)}

    def analyse(self, image, crop=None):
        logits, features = self.forward([image])
        probs = torch.softmax(torch.from_numpy(logits[0]), dim=0).numpy()
        masses = self.crop_mass(probs)
        likely_crop = max(masses, key=masses.get)

        result = {"crop_mass": masses, "likely_crop": likely_crop, "ood_score": None}

        if self.ood_ready:
            score = float(ood_scores(
                self.index["method"], logits, features, self.index["bank"], self.index["k"]
            )[0])
            result["ood_score"] = score
            if score < self.index["threshold"]:
                return {**result, "status": self.UNKNOWN}

        if crop and masses.get(crop, 0.0) < self.CROP_MASS_MIN:
            return {**result, "status": self.CROP_MISMATCH, "selected_crop": crop}

        # Restrict the prediction to the selected crop's classes
        allowed = self.class_crops == crop if crop else np.ones(len(probs), dtype=bool)
        restricted = np.where(allowed, probs, 0.0)
        top = int(restricted.argmax())
        crop_key, disease_key = self.class_keys[top]

        return {
            **result,
            "status": self.OK,
            "label": self.model.names[top],
            "crop": crop_key,
            "disease_key": disease_key,
            "confidence": float(restricted[top] / restricted.sum()),
        }
