import numpy as np

from src.labels import parse_prediction
from src.reliability import LogitCapture, ReliabilityGate


class InputGuard:
    """Checks a photo before any advice is given.

    Order of checks (each one stops the advice pipeline):
        1. UNKNOWN_INPUT  - the image is far from every training image (kNN feature distance),
                            e.g. another crop, an unseen disease or not a leaf at all
        2. CROP_MISMATCH  - the selected crop holds less than half of the model's probability
        3. LOW_CONFIDENCE - calibrated confidence is below the abstention threshold
    The reliability checks (1 and 3) switch on only when models/reliability.json exists,
    so the app behaves exactly as before until calibration_abstention.py has been run.
    """

    OK = "OK"
    CROP_MISMATCH = "CROP_MISMATCH"
    UNKNOWN_INPUT = ReliabilityGate.UNKNOWN_INPUT
    LOW_CONFIDENCE = ReliabilityGate.LOW_CONFIDENCE

    # The selected crop must hold at least half of the model's probability
    CROP_MASS_MIN = 0.5

    def __init__(self, model, gate="auto", model_path=None):
        self.model = model
        self.class_keys = [parse_prediction(model.names[i]) for i in range(len(model.names))]
        self.class_crops = np.array([crop for crop, _ in self.class_keys])
        self.gate = ReliabilityGate.load_if_available(model_path=model_path) if gate == "auto" else gate
        self.capture = LogitCapture(model) if self.gate is not None else None

    def crop_mass(self, probs):
        return {crop: float(probs[self.class_crops == crop].sum()) for crop in dict.fromkeys(self.class_crops)}

    def _probabilities(self, image):
        if self.capture is None:
            probs = self.model(image, verbose=False)[0].probs.data.float().cpu().numpy()
            return probs, probs, None
        self.capture.sync()
        raw = self.model(image, verbose=False)[0].probs.data.float().cpu().numpy()
        logits, features = self.capture.pop()
        return self.gate.calibrated(logits[0]), raw, features[0]

    def analyse(self, image, crop=None):
        probs, raw_probs, feature = self._probabilities(image)
        masses = self.crop_mass(probs)
        likely_crop = max(masses, key=masses.get)
        result = {"crop_mass": masses, "likely_crop": likely_crop, "calibrated": self.gate is not None}

        if self.gate is not None:
            unknown, knn_score = self.gate.is_unknown(feature)
            result["knn_score"] = knn_score
            if unknown:
                return {**result, "status": self.UNKNOWN_INPUT}

        if crop and masses.get(crop, 0.0) < self.CROP_MASS_MIN:
            return {**result, "status": self.CROP_MISMATCH, "selected_crop": crop}

        # Restrict the prediction to the selected crop's classes
        allowed = self.class_crops == crop if crop else np.ones(len(probs), dtype=bool)
        restricted = np.where(allowed, probs, 0.0)
        restricted = restricted / restricted.sum()
        order = np.argsort(-restricted)
        top = int(order[0])
        crop_key, disease_key = self.class_keys[top]

        result = {
            **result,
            "status": self.OK,
            "label": self.model.names[top],
            "crop": crop_key,
            "disease_key": disease_key,
            "confidence": float(restricted[top]),
            "raw_confidence": float(raw_probs[top]),
            "alternatives": [(self.class_keys[i][1], float(restricted[i])) for i in order[1:3] if restricted[i] > 0],
        }

        if self.gate is not None and result["confidence"] < self.gate.threshold(crop_selected=bool(crop)):
            result["status"] = self.LOW_CONFIDENCE
            result["threshold"] = self.gate.threshold(crop_selected=bool(crop))
        return result
