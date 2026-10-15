import re

SUPPORTED_CROPS = ("maize", "potato", "rice", "wheat")


def parse_prediction(label):
    # Convert duplicate PlantVillage labels into the verified canonical key
    parts = [part for part in re.split(r"_+", label.lower().strip()) if part]
    original_crop = parts.pop(0)
    crop = "maize" if original_crop == "corn" else original_crop

    while parts and parts[0] in {original_crop, crop}:
        parts.pop(0)

    condition = "_".join(parts)
    if crop == "rice" and condition in {"blast", "rice_blast"}:
        condition = "leaf_blast"

    return crop, f"{crop}_{condition}"
