#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=8000}"

# If you use Gunicorn+Uvicorn workers:
# exec gunicorn api_server:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:"${PORT}" --workers "${WEB_CONCURRENCY}"

# Plain uvicorn:
exec uvicorn api_server:app --host 0.0.0.0 --port "${PORT}" --workers "${WEB_CONCURRENCY}"
