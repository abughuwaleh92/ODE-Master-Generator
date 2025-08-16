#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REDIS_URL:-}" ]]; then
  echo "ERROR: REDIS_URL is not set"; exit 1
fi

echo "Starting RQ worker on queues: ${RQ_QUEUES}"
# Use the correct scheme: redis:// or rediss://
exec rq worker -u "${REDIS_URL}" ${RQ_QUEUES}
