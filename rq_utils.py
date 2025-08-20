# rq_utils.py
import os
import importlib
from typing import Any, Dict, Optional

try:
    import redis
    from rq import Queue
    from rq.job import Job
    from rq.worker import Worker
except Exception:  # allow app to run without RQ installed
    redis = None
    Queue = Job = Worker = None

# --- env & defaults ---
REDIS_URL = os.getenv("REDIS_URL", "").strip()
QUEUE_NAME = os.getenv("RQ_QUEUE", "ode_jobs").strip() or "ode_jobs"

# Defaults tuned so jobs don't "disappear" too soon
DEFAULT_TIMEOUT     = int(os.getenv("RQ_TIMEOUT", "3600"))     # 1h max runtime
DEFAULT_RESULT_TTL  = int(os.getenv("RQ_RESULT_TTL", "86400")) # 24h keep results
DEFAULT_TTL         = int(os.getenv("RQ_TTL", "604800"))       # 7 days keep job data

_conn = None

def has_redis() -> bool:
    return bool(REDIS_URL and redis is not None)

def _conn_ok():
    try:
        if not has_redis():
            return False
        c = get_connection()
        c.ping()
        return True
    except Exception:
        return False

def get_connection():
    global _conn
    if _conn is not None:
        return _conn
    if not has_redis():
        return None
    # decode_responses=False is important (jobs are pickled)
    _conn = redis.from_url(REDIS_URL, decode_responses=False)
    return _conn

def _import_callable(func_path: str):
    """Import a callable from dotted path 'pkg.mod.func'."""
    mod_path, fn_name = func_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)

def enqueue_job(
    func_path: str,
    payload: Dict[str, Any],
    description: str = "",
    timeout: Optional[int] = None,
    result_ttl: Optional[int] = None,
    ttl: Optional[int] = None,
) -> Optional[str]:
    """
    RQ-1.x compatible enqueue that uses 'timeout', 'result_ttl', 'ttl' (NOT 'job_timeout').
    We import the callable so pickling works the same across web/worker.
    """
    if not _conn_ok():
        return None
    func = _import_callable(func_path)
    q = Queue(QUEUE_NAME, connection=get_connection(), default_timeout=timeout or DEFAULT_TIMEOUT)
    # enqueue_call signature is stable across RQ 1.x if we avoid job_timeout
    job = q.enqueue_call(
        func=func,
        kwargs={"payload": payload},
        timeout=timeout or DEFAULT_TIMEOUT,
        result_ttl=result_ttl or DEFAULT_RESULT_TTL,
        ttl=ttl or DEFAULT_TTL,
        description=description or func_path,
    )
    return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a job safely from the same Redis connection.
    Returns a normalized dict or None if not found.
    """
    if not _conn_ok() or not job_id:
        return None
    try:
        job = Job.fetch(job_id, connection=get_connection())
    except Exception:
        return None

    info = {
        "id": job.id,
        "status": job.get_status(refresh=False),
        "origin": job.origin,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
        "meta": job.meta or {},
        "exc_info": job.exc_info,
    }
    # If finished & result_ttl not expired, include result
    try:
        if info["status"] == "finished":
            info["result"] = job.result
    except Exception:
        info["result"] = None

    # Optional logs your worker can append into meta["logs"]
    info["logs"] = (job.meta or {}).get("logs", [])
    return info

def redis_status() -> Dict[str, Any]:
    """Small diag block for the UI Settings page."""
    out = {
        "ok": _conn_ok(),
        "queue": QUEUE_NAME,
        "url_present": bool(REDIS_URL),
        "workers": [],
        "ping": None,
    }
    if not _conn_ok():
        return out
    try:
        out["ping"] = get_connection().ping()
    except Exception:
        pass
    try:
        # list worker names attached to this Redis (not necessarily this queue)
        ws = Worker.all(connection=get_connection()) or []
        out["workers"] = [getattr(w, "name", "worker") for w in ws]
    except Exception:
        pass
    return out