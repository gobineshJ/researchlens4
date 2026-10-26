# One image for both the API and the Celery worker (docker-compose.yml
# runs it with two different commands).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles ~2 GB of CUDA
# libraries that a CPU container never uses.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Slow or flaky connections: wait up to 2 minutes on a stalled download
# (pip's default is 15 s, Hugging Face's 10 s) and retry, instead of
# failing the build. Set here, below the torch layer, so changing them
# never invalidates that ~200 MB download.
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ETAG_TIMEOUT=60

COPY requirements.txt .
# The cache mount keeps every wheel that finished downloading across
# builds, so a build that fails halfway resumes rather than starting the
# list over. It lives outside the image (PIP_NO_CACHE_DIR still keeps
# the image itself free of a pip cache).
RUN --mount=type=cache,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR pip install -r requirements.txt

# Bake the embedding and reranker models into the image, so the first
# report doesn't stall on a download (and the worker can run offline).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app

RUN useradd --create-home appuser \
    && mkdir -p /data \
    && chown -R appuser /data /models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
