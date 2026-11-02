FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_MODEL_PATH=/app/runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt \
    FEEDBACK_DIR=/app/data/feedback

# OpenCV (pulled in by ultralytics) needs these shared libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

# Bake the RAG embedding model into the image so startup does not download 1.3 GB
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
# Use the baked-in copy instead of checking the Hub on every start
ENV HF_HUB_OFFLINE=1

# --chmod=755 throughout: several host files are owner-only and the app runs as a non-root user.
# (755, not 644: the mode also applies to folders COPY creates, which need +x to be entered.)
COPY --chmod=755 .streamlit/config.toml .streamlit/config.toml
COPY --chmod=755 assets/ assets/
# 3D leaf model and still image, served at /app/static/ (the .blend source is excluded in .dockerignore)
COPY --chmod=755 static/ static/
COPY --chmod=755 src/ src/
COPY chroma_db/ chroma_db/
# The field-fine-tuned model the app uses; reliability.json and the feature bank were fitted to it.
# Kept at its project path: the reliability gate switches off unless the model path matches reliability.json.
COPY --chmod=755 runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt \
    runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt
# Calibration, abstention and unknown-input settings (scripts/evaluation/*)
COPY --chmod=755 models/reliability.json models/ood_feature_bank.npy models/
COPY --chmod=755 app.py .

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/feedback \
    && chown -R appuser /app/data /app/chroma_db
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--server.enableXsrfProtection=false"]
