"""Summarise farmers' "Was this correct?" answers and export the shared photos for retraining.

Writes reports/tables/feedback_summary.csv (field accuracy per predicted class) and copies every
shared photo with a known true class into data/feedback_dataset/<class>/, the same folder layout
as data/processed. Review the photos by hand before adding them to the training data.

    python scripts/analysis/export_feedback.py
"""
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.feedback import CORRECT, FEEDBACK_DIR, UNSURE, WRONG, load_feedback  # noqa: E402

EXPORT_DIR = Path("data/feedback_dataset")
REPORT_PATH = Path("reports/tables/feedback_summary.csv")


def true_class(record):
    # Correct answers confirm the prediction; wrong answers name the real class unless unknown
    if record["verdict"] == CORRECT:
        return record["predicted_key"]
    if record["verdict"] == WRONG and record["correct_key"] not in (None, "other"):
        return record["correct_key"]
    return None


def main():
    records = load_feedback()
    if not records:
        raise SystemExit(f"No feedback found in {FEEDBACK_DIR}.")

    counts = {}
    for record in records:
        counts.setdefault(record["predicted_key"], Counter())[record["verdict"]] += 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted_class", "correct", "wrong", "unsure", "field_accuracy"])
        for key, count in sorted(counts.items()):
            judged = count[CORRECT] + count[WRONG]
            accuracy = f"{count[CORRECT] / judged:.3f}" if judged else ""
            writer.writerow([key, count[CORRECT], count[WRONG], count[UNSURE], accuracy])

    exported = 0
    for record in records:
        label = true_class(record)
        if label and record["image_file"]:
            source = FEEDBACK_DIR / "images" / record["image_file"]
            if source.is_file():
                target_dir = EXPORT_DIR / label
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target_dir / record["image_file"])
                exported += 1

    print(f"📊 {len(records)} answers summarised in {REPORT_PATH}")
    print(f"📁 {exported} shared photos exported to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
