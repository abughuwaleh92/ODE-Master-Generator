# rq_utils.py
import os
from typing import Any, Dict, Optional, Tuple

from rq import Queue
from rq.job import Job
from redis import Redis

PLACEHOLDER_TOKENS = ("${{", "}}", "$(")

def _is_placeholder(val: str) -> bool:
    v = (val or "").strip()
    if not v:
        return False
    # Railway/CI-style unresolved placeholders or empty scheme
    return any(tok in v for tok in PLACEHOLDER_TOKENS) or v.startswith("http") or "://" not in v

def _trim(val: Optional[str]) -> Optional[str]:
    return val.strip() if isinstance(val, str) else val

def _redis_url_from_env() -> Optional[str]:
    # Preferred keys
    for key in ("REDIS_URL", "REDIS_TLS_URL", "UPSTASH_REDIS_URL"):
        raw = os.getenv(key)
        if raw:
            raw = _trim(raw)
            if raw and not _is_placeholder(raw):
                return raw

    # Scan all env if user used a custom name (e.g., RAILWAY_REDIS_URL)
    for k, v in os.environ.items():
        if "REDIS" in k.upper() and isinstance(v, str):
            vv = _trim(v)
            if vv and ("redis://" in vv or "rediss://" in vv) and not _is_placeholder(vv):
                return vv

    # Optional secrets file
    path = os.getenv("REDIS_URL_FILE")
    if path and os.path.exists(path):
        try:
            vv = _trim(open(path, "r", encoding="utf-8").read())
            if vv and not _is_placeholder(vv):
                return vv
        except Exception:
            pass

    # Optional Streamlit secrets (only on web)
    try:
        import streamlit as st
        for key in ("REDIS_URL", "REDIS_TLS_URL", "UPSTASH_REDIS_URL"):
            if key in st.secrets:
                vv = _trim(st.secrets[key])
                if vv and not _is_placeholder(vv):
                    return vv
    except Exception:
        pass

    return None

def _redis_kwargs(url: str) -> dict:
    kwargs = {"decode_responses": False}
    if url.startswith("rediss://"):
        # Allow insecure TLS if requested (managed hosts sometimes need this)
        if os.getenv("REDIS_INSECURE_TLS", "false").lower() in ("1", "true", "yes"):
            kwargs["ssl_cert_reqs"] = None
    return kwargs

def _redis_conn() -> Optional[Redis]:
    url = _redis_url_from_env()
    if not url:
        return None
    try:
        return Redis.from_url(url, **_redis_kwargs(url))
    except Exception:
        return None

def has_redis() -> bool:
    r = _redis_conn()
    if not r:
        return False
    try:
        return bool(r.ping())
    except Exception:
        return False

def _mask_url(url: str) -> str:
    try:
        import re
        return re.sub(r"(?<=://.*:)[^@]+(?=@)", "***", url)
    except Exception:
        return url

def redis_status() -> Dict[str, Any]:
    url = _redis_url_from_env()
    if not url:
        return {"ok": False, "reason": "No REDIS_* env/secret found or value is a placeholder."}
    try:
        r = Redis.from_url(url, **_redis_kwargs(url))
        ok = r.ping()
        return {"ok": bool(ok), "url_preview": _mask_url(url), "queue": os.getenv("RQ_QUEUE", "ode_jobs")}
    except Exception as e:
        return {"ok": False, "url_preview": _mask_url(url), "reason": str(e)}

def _queue(name: Optional[str] = None) -> Queue:
    conn = _redis_conn()
    if conn is None:
        raise RuntimeError("REDIS_URL not set or Redis unavailable.")
    qname = name or os.getenv("RQ_QUEUE", "ode_jobs")
    return Queue(name=qname, connection=conn)

def enqueue_job(func_path: str, payload: Dict[str, Any], **job_opts) -> Optional[str]:
    """
    Allowed job_opts: job_timeout, result_ttl, ttl, description, meta, depends_on, queue
    Defaults are long so you don't lose info after a few minutes.
    """
    q = _queue(job_opts.pop("queue", None))
    allowed = {"job_timeout", "result_ttl", "ttl", "description", "meta", "depends_on"}
    opts = {k: v for k, v in job_opts.items() if k in allowed}

    # Long-lived defaults
    opts.setdefault("job_timeout", int(os.getenv("RQ_JOB_TIMEOUT", "86400")))   # 24h
    opts.setdefault("result_ttl", int(os.getenv("RQ_RESULT_TTL", "604800")))   # 7 days
    opts.setdefault("ttl", int(os.getenv("RQ_TTL", "172800")))                 # 2 days

    job = q.enqueue(func_path, payload, **opts)
    return job.id if job else None

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _redis_conn()
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

    # Try to attach worker logs
    try:
        url = _redis_url_from_env()
        rtxt = Redis.from_url(url, decode_responses=True, **_redis_kwargs(url))
        info["logs"] = rtxt.lrange(f"job:{job.id}:logs", 0, -1)
    except Exception:
        info["logs"] = []

    return info