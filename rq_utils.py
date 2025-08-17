# rq_utils.py
"""
RQ helpers — robust, queue-consistent, with persistent progress.
Default queue is 'ode_jobs' (override with env RQ_QUEUE).
"""

import os
import time
import json
from typing import Any, Dict, Optional

import redis
from rq import Queue, Connection
from rq.job import Job
from rq.registry import (StartedJobRegistry, FinishedJobRegistry,
                         FailedJobRegistry, ScheduledJobRegistry, DeferredJobRegistry)

# ---- configuration ----
REDIS_URL = (
    os.getenv("REDIS_URL")
    or os.getenv("UPSTASH_REDIS_URL")
    or os.getenv("REDIS_TLS_URL")
    or ""
)
RQ_QUEUE = os.getenv("RQ_QUEUE", "ode_jobs").strip() or "ode_jobs"

# optional: longer defaults for heavy training
DEFAULT_RESULT_TTL = int(os.getenv("RQ_RESULT_TTL", "86400"))    # 24h
DEFAULT_FAILURE_TTL = int(os.getenv("RQ_FAILURE_TTL", "604800")) # 7d
DEFAULT_JOB_TIMEOUT = int(os.getenv("RQ_JOB_TIMEOUT", "1800"))   # 30m default

# ---- connection ----
def _conn() -> redis.Redis:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is not set")
    # keep connections reliable on Railway
    return redis.from_url(
        REDIS_URL,
        decode_responses=True,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_timeout=30,
        retry_on_timeout=True,
    )

def has_redis() -> bool:
    try:
        r = _conn()
        r.ping()
        return True
    except Exception:
        return False

# ---- enqueue/fetch ----
def enqueue_job(
    fn_path: str,
    payload: Dict[str, Any],
    *,
    queue: Optional[str] = None,
    description: Optional[str] = None,
    job_timeout: Optional[int] = None,
    result_ttl: Optional[int] = None,
    failure_ttl: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Enqueue a Python callable by dotted path.
    Returns job_id.
    """
    qname = (queue or RQ_QUEUE).strip()
    with Connection(_conn()):
        q = Queue(qname, default_timeout=job_timeout or DEFAULT_JOB_TIMEOUT)
        job = q.enqueue(
            fn_path,
            payload,
            description=description or fn_path,
            job_timeout=job_timeout or DEFAULT_JOB_TIMEOUT,
            result_ttl=result_ttl if result_ttl is not None else DEFAULT_RESULT_TTL,
            failure_ttl=failure_ttl if failure_ttl is not None else DEFAULT_FAILURE_TTL,
            meta=meta or {},
        )
        return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a compact, JSON‑safe status dict for a job_id (None if unknown)."""
    if not job_id:
        return None
    with Connection(_conn()):
        try:
            job = Job.fetch(job_id, connection=_conn())
        except Exception:
            return None

        # compact snapshot + meta
        info = {
            "id": job.id,
            "status": job.get_status(refresh=True),
            "created_at": str(job.created_at) if job.created_at else None,
            "started_at": str(job.started_at) if job.started_at else None,
            "ended_at": str(job.ended_at) if job.ended_at else None,
            "description": job.description,
            "origin": job.origin,
            "result": job.result if job.is_finished else None,
            "meta": job.meta or {},
            "exc_info": job.exc_info,
        }
        # add stage if present in meta
        stage = (job.meta or {}).get("stage")
        if stage:
            info["stage"] = stage
        return info

# ---- observability (used by app's Jobs & Workers panel) ----
def rq_inspect() -> Dict[str, Any]:
    """Summarize queues and workers; safe to call from Streamlit."""
    summary: Dict[str, Any] = {"queues": [], "workers": []}
    try:
        with Connection(_conn()):
            from rq import Worker
            from rq.command import send_shutdown_command

            # queues
            for qname in sorted({RQ_QUEUE, "default"}):
                q = Queue(qname)
                summary["queues"].append({
                    "name": qname,
                    "count": q.count,
                    "deferred": DeferredJobRegistry(qname).count,
                    "scheduled": ScheduledJobRegistry(qname).count,
                    "started": StartedJobRegistry(qname).count,
                    "finished": FinishedJobRegistry(qname).count,
                    "failed": FailedJobRegistry(qname).count,
                })

            # workers
            for w in Worker.all():
                summary["workers"].append({
                    "name": w.name,
                    "state": w.state,
                    "queues": [qq.name for qq in w.queues],
                    "birth_date": str(w.birth_date) if w.birth_date else None,
                })
    except Exception as e:
        summary["error"] = str(e)
    return summary