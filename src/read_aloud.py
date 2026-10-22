"""Read the advisory report aloud in the farmer's language with Google Text-to-Speech (gTTS).

gTTS needs an internet connection and has no voice for Odia, so the app shows the written
report alone in those cases.
"""
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor

from gtts import gTTS, gTTSError

PIECE_CHARS = 400
WORKERS = 6
RETRIES = 2

# Report language name -> gTTS language code (Odia has no gTTS voice)
VOICE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
}


def has_voice(language):
    return language in VOICE_CODES


def speakable_text(markdown):
    # Strip Markdown and URLs so the voice reads words, not symbols or web addresses
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"`+|\*+|_{2,}|^#+\s*|^\s*[-•]\s+|^>\s*|\|", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)

    # End headings and list items with a full stop so the voice pauses between them
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(line if line[-1] in ".!?:।॥" else line + "." for line in lines)


def _pieces(text, size=PIECE_CHARS):
    # Group whole lines into pieces of about `size` characters so they can be fetched in parallel
    pieces, current = [], ""
    for line in text.splitlines():
        if current and len(current) + len(line) > size:
            pieces.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    return pieces + [current] if current else pieces


def _synthesize_piece(piece, code):
    # gTTS sends one request per ~100 characters, so a single throttled request is retried
    for attempt in range(RETRIES + 1):
        try:
            buffer = io.BytesIO()
            # The co.in domain gives English an Indian accent; other languages ignore it
            gTTS(text=piece, lang=code, tld="co.in", timeout=15).write_to_fp(buffer)
            return buffer.getvalue()
        except gTTSError:
            if attempt == RETRIES:
                raise
            time.sleep(1 + attempt)


def synthesize(text, language):
    """MP3 bytes of the report read aloud, or None when the language has no voice."""
    code = VOICE_CODES.get(language)
    if code is None or not text.strip():
        return None

    # A full report is about 60 requests; fetching pieces in parallel cuts ~20 s to a few seconds.
    # MP3 files can be joined end to end, so the pieces play as one recording
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return b"".join(pool.map(lambda piece: _synthesize_piece(piece, code), _pieces(text)))
