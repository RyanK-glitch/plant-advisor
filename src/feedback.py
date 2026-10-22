"""Stores farmers' "Was this correct?" answers so field photos can be reviewed and used for retraining.

Records go to a local SQLite database. Photos are kept only when the farmer agrees, are resized
and are saved without EXIF metadata, so phone GPS coordinates are never stored. The data folder
is gitignored; on Streamlit Community Cloud the disk is reset on every restart, so copy the
database out regularly or point FEEDBACK_DIR at persistent storage.
"""
import hashlib
import io
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", PROJECT_DIR / "data" / "feedback"))

CORRECT = "correct"
WRONG = "wrong"
UNSURE = "unsure"
VERDICTS = (CORRECT, WRONG, UNSURE)

MAX_PHOTO_SIDE = 1024
MAX_COMMENT_LENGTH = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    verdict TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    predicted_key TEXT NOT NULL,
    confidence REAL NOT NULL,
    selected_crop TEXT,
    correct_key TEXT,
    comment TEXT,
    place TEXT,
    report_language TEXT,
    model_name TEXT,
    image_sha256 TEXT,
    image_file TEXT
)
"""


def image_digest(image):
    # Same photo -> same digest, so repeated submissions can be spotted
    return hashlib.sha256(image.tobytes()).hexdigest()


def _connect(directory):
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / "feedback.db")
    connection.execute(SCHEMA)
    return connection


def _save_photo(image, directory, record_id):
    photo = image.convert("RGB")
    photo.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE))

    # Re-encoding without the exif argument drops all metadata, including GPS
    buffer = io.BytesIO()
    photo.save(buffer, format="JPEG", quality=90)

    images_dir = directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{record_id}.jpg"
    (images_dir / file_name).write_bytes(buffer.getvalue())
    return file_name


def save_feedback(*, verdict, prediction, image=None, share_photo=False, correct_key=None,
                  comment="", selected_crop=None, place=None, report_language=None,
                  model_name=None, directory=FEEDBACK_DIR):
    """Stores one answer and returns its id. prediction holds label, disease_key and confidence."""
    if verdict not in VERDICTS:
        raise ValueError(f"Unknown verdict: {verdict}")

    directory = Path(directory)
    record_id = uuid.uuid4().hex
    image_file = _save_photo(image, directory, record_id) if image is not None and share_photo else None

    connection = _connect(directory)
    try:
        with connection:
            connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    verdict,
                    prediction["label"],
                    prediction["disease_key"],
                    float(prediction["confidence"]),
                    selected_crop,
                    correct_key if verdict == WRONG else None,
                    (comment or "").strip()[:MAX_COMMENT_LENGTH] or None,
                    place,
                    report_language,
                    model_name,
                    image_digest(image) if image is not None else None,
                    image_file,
                ),
            )
    finally:
        connection.close()
    return record_id


def load_feedback(directory=FEEDBACK_DIR):
    """All answers as a list of dicts, newest first."""
    path = Path(directory) / "feedback.db"
    if not path.is_file():
        return []
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]
