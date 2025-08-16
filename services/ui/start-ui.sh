#!/usr/bin/env bash
set -euo pipefail

# Optional preflight
if [[ -f startup_check.py ]]; then
  python startup_check.py || true
fi

# PORT is provided by Railway
: "${PORT:=8501}"

# Streamlit must listen on 0.0.0.0 and $PORT
exec streamlit run master_generators_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --browser.gatherUsageStats=false
