# ----- Dockerfile -----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages: nginx, supervisor, envsubst (from gettext-base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor gettext-base ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install (single Streamlit range; no conflicts)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (adjust/add paths as needed)
# Expect these files to be present in the repo root:
#   - master_generators_app.py (Streamlit UI)
#   - api_server.py            (FastAPI backend)
#   - core_master_generators.py (the core)
COPY master_generators_app.py /app/master_generators_app.py
COPY api_server.py            /app/api_server.py
COPY core_master_generators.py /app/core_master_generators.py
# If you’ve packaged the core as a module dir, include it too:
# COPY mg_core /app/mg_core

# Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

# Nginx & Supervisor configs and entrypoint
COPY deploy/nginx.conf.template /etc/nginx/conf.d/default.conf.template
COPY deploy/supervisord.conf    /etc/supervisor/conf.d/supervisord.conf
COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
 && rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf || true

# Railway sets $PORT; default to 8080 for local use
ENV PORT=8080 \
    MG_API_BASE=/api

# Logs to stdout/stderr
ENV STREAMLIT_SERVER_HEADLESS=true

# Start: render nginx with $PORT, run supervisor (nginx+uvicorn+streamlit)
CMD ["bash", "-lc", "docker-entrypoint.sh"]
