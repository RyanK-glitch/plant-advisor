import numpy as np

from src.labels import parse_prediction


class InputGuard:
    """Checks a photo against the farmer's selected crop before any advice is given."""

    OK = "OK"
    CROP_MISMATCH = "CROP_MISMATCH"

    # The selected crop must hold at least half of the model's probability
    CROP_MASS_MIN = 0.5

    def __init__(self, model):
        self.model = model
        self.class_keys = [parse_prediction(model.names[i]) for i in range(len(model.names))]
        self.class_crops = np.array([crop for crop, _ in self.class_keys])

    def crop_mass(self, probs):
        return {crop: float(probs[self.class_crops == crop].sum()) for crop in dict.fromkeys(self.class_crops)}

    def analyse(self, image, crop=None):
        probs = self.model(image, verbose=False)[0].probs.data.float().cpu().numpy()
        masses = self.crop_mass(probs)
        likely_crop = max(masses, key=masses.get)

        result = {"crop_mass": masses, "likely_crop": likely_crop}

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
