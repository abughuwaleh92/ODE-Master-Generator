# syntax=docker/dockerfile:1.5

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000 \
    WEB_CONCURRENCY=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
EXPOSE 8000

# Healthcheck expects /health or /healthz in your api_server.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["bash", "services/api/start-api.sh"]
