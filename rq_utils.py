# rq_utils.py
import os
from typing import Any, Dict, Optional

from rq import Queue
from rq.job import Job
from redis import Redis

def _redis_url_from_env() -> Optional[str]:
    # Try several well-known env names
    for key in ("REDIS_URL", "REDIS_TLS_URL", "UPSTASH_REDIS_URL"):
        val = os.getenv(key)
        if val:
            return val

    # Optional: read from a secrets file if you can't put it directly in env
    path = os.getenv("REDIS_URL_FILE")
    if path and os.path.exists(path):
        try:
            return open(path, "r", encoding="utf-8").read().strip()
        except Exception:
            pass

    # Optional: Streamlit secrets (only in the web app process)
    try:
        import streamlit as st
        for key in ("REDIS_URL", "REDIS_TLS_URL", "UPSTASH_REDIS_URL"):
            if key in st.secrets:
                return st.secrets[key]
    except Exception:
        pass

    return None

def _redis_kwargs(url: str) -> dict:
    # Allow insecure TLS (for some managed hosts / Upstash if cert path breaks)
    kwargs = {"decode_responses": False}
    if url.startswith("rediss://") and os.getenv("REDIS_INSECURE_TLS", "false").lower() == "true":
        kwargs["ssl_cert_reqs"] = None  # disable cert verification (use only if you must)
    return kwargs

def _redis_rq() -> Optional[Redis]:
    url = _redis_url_from_env()
    if not url:
        return None
    return Redis.from_url(url, **_redis_kwargs(url))

def has_redis() -> bool:
    r = _redis_rq()
    if not r:
        return False
    try:
        return r.ping()
    except Exception:
        return False

def _queue(name: Optional[str] = None) -> Queue:
    conn = _redis_rq()
    if conn is None:
        raise RuntimeError("REDIS_URL not set or Redis unavailable.")
    qname = name or os.getenv("RQ_QUEUE", "ode_jobs")
    return Queue(name=qname, connection=conn)

def enqueue_job(func_path: str, payload: Dict[str, Any], **job_opts) -> Optional[str]:
    q = _queue(job_opts.pop("queue", None))
    allowed = {"job_timeout", "result_ttl", "ttl", "description", "meta", "depends_on"}
    opts = {k: v for k, v in job_opts.items() if k in allowed}
    opts.setdefault("job_timeout", int(os.getenv("RQ_JOB_TIMEOUT", "86400")))
    opts.setdefault("result_ttl", int(os.getenv("RQ_RESULT_TTL", "604800")))
    opts.setdefault("ttl", int(os.getenv("RQ_TTL", "86400")))
    job = q.enqueue(func_path, payload, **opts)
    return job.id if job else None

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _redis_rq()
    if conn is None:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return None
    info = {
        "id": job.id,
        "status": job.get_status(),
        "origin": job.origin,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
        "meta": job.meta or {},
        "exc_info": job.exc_info,
        "result": job.result if job.is_finished else None,
    }
    # Optional: attach plain-text logs created by worker.py
    try:
        url = _redis_url_from_env()
        rtext = Redis.from_url(url, decode_responses=True, **_redis_kwargs(url))
        logs = rtext.lrange(f"job:{job.id}:logs", 0, -1)
        info["logs"] = logs or []
    except Exception:
        info["logs"] = []
    return info

# Small helper exposed for the UI diagnostics
def redis_status() -> Dict[str, Any]:
    url = _redis_url_from_env()
    if not url:
        return {"ok": False, "reason": "No REDIS_URL/TLS/UPSTASH env or secret found."}
    try:
        r = Redis.from_url(url, **_redis_kwargs(url))
        pong = r.ping()
        return {
            "ok": bool(pong),
            "url_preview": _mask_url(url),
            "queue": os.getenv("RQ_QUEUE", "ode_jobs"),
        }
    except Exception as e:
        return {"ok": False, "reason": str(e), "url_preview": _mask_url(url)}

def _mask_url(url: str) -> str:
    # Hide password in logs/UI
    try:
        import re
        return re.sub(r"(?<=://.*:)[^@]+(?=@)", "***", url)
    except Exception:
        return url