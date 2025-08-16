# syntax=docker/dockerfile:1.5

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS deps needed by numpy/pandas/plotly/sympy; keep slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ curl \
    libgfortran5 libopenblas0 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy application
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit config via env
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1

# Railway sets $PORT at runtime
ENV PORT=8501
EXPOSE 8501

# Healthcheck uses Streamlit's internal health
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

CMD ["bash", "services/ui/start-ui.sh"]
