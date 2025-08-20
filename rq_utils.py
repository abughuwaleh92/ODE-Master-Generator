# rq_utils.py
import os
from typing import Any, Dict, List, Optional

from rq import Queue
from rq.job import Job

try:
    import redis
except Exception as e:
    redis = None

def _redis_url() -> Optional[str]:
    return os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or os.getenv("UPSTASH_REDIS_URL")

def _conn():
    """
    Create a Redis connection compatible with RQ pickling (binary).
    decode_responses=False is important to avoid breaking RQ payload/result handling.
    """
    url = _redis_url()
    if not url or not redis:
        return None
    return redis.from_url(url, decode_responses=False)

def has_redis() -> bool:
    return _conn() is not None

def _queue_name(explicit: Optional[str] = None) -> str:
    return explicit or os.getenv("RQ_QUEUE") or "ode_jobs"

def get_queue(name: Optional[str] = None) -> Optional[Queue]:
    conn = _conn()
    if not conn:
        return None
    return Queue(_queue_name(name), connection=conn)

def enqueue_job(func_path: str, payload: dict, queue: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Enqueue a job on the RQ queue. Accepts RQ options via **kwargs:
      - job_timeout
      - result_ttl
      - ttl
      - retry
      - at, in_, etc.
    """
    q = get_queue(queue)
    if not q:
        return None
    # payload is passed as a single positional arg to target function
    job = q.enqueue(func_path, payload, **kwargs)
    return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    if not conn:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return None

    info: Dict[str, Any] = {
        "id": job.id,
        "status": job.get_status(),
        "origin": job.origin,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
    }
    # Job meta (progress, artifacts, etc.)
    try:
        info["meta"] = job.meta or {}
    except Exception:
        info["meta"] = {}

    # Result if finished
    try:
        if job.is_finished:
            info["result"] = job.result
    except Exception:
        info["result"] = None

    if job.is_failed:
        info["exc_info"] = job.exc_info

    return info

def get_progress(job_id: str) -> Dict[str, Any]:
    conn = _conn()
    if not conn:
        return {}
    try:
        job = Job.fetch(job_id, connection=conn)
        return job.meta.get("progress", {}) or {}
    except Exception:
        return {}

def get_artifacts(job_id: str) -> Dict[str, Any]:
    conn = _conn()
    if not conn:
        return {}
    try:
        job = Job.fetch(job_id, connection=conn)
        return job.meta.get("artifacts", {}) or {}
    except Exception:
        return {}

def get_logs(job_id: str, start: int = 0, end: int = -1) -> List[str]:
    """
    Logs are pushed by the worker to Redis list: key = f"job:{job_id}:logs".
    We read and decode them here.
    """
    conn = _conn()
    if not conn:
        return []
    key = f"job:{job_id}:logs"
    try:
        # Python Redis expects end inclusive; if -1, fetch many
        if end == -1:
            end = start + 2000
        raw = conn.lrange(key, start, end)
        out: List[str] = []
        for item in raw:
            if isinstance(item, bytes):
                try:
                    out.append(item.decode("utf-8", "ignore"))
                except Exception:
                    out.append(str(item))
            else:
                out.append(str(item))
        return out
    except Exception:
        return []