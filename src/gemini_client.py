"""Gemini text generation with model fallback (shared by app.py and the evaluation scripts)."""
from google import genai
from google.genai import errors, types

# Free-tier Gemini text models, tried in order until one responds
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]

# Busy, rate-limited, retired or timed-out models: move to the next one
FALLBACK_CODES = {404, 408, 429, 500, 502, 503, 504}


def generate_with_fallback(api_key, prompt, models=None):
    # Give each model 60 seconds so a stalled one does not block the fallback
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
    failures = []

    for model_name in models or GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response.text:
                return response.text, model_name, failures
            failures.append(f"{model_name}: empty response")
        except errors.APIError as error:
            # Other client errors (bad key, bad request) affect every model, so stop
            if error.code not in FALLBACK_CODES:
                raise
            failures.append(f"{model_name}: {error.code} {error.status}")
        except Exception as error:
            # Network timeouts and dropped connections
            failures.append(f"{model_name}: {type(error).__name__}")

    raise RuntimeError("All Gemini models are currently unavailable. " + "; ".join(failures))
