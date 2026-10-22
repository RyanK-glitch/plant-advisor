FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_MODEL_PATH=/app/models/best.pt \
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

COPY .streamlit/config.toml .streamlit/config.toml
COPY assets/ assets/
COPY src/ src/
COPY chroma_db/ chroma_db/
COPY runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt models/best.pt
COPY app.py .

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
