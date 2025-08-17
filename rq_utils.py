# rq_utils.py
import os
from typing import Optional, Dict, Any, List
try:
    from redis import Redis
    from rq import Queue
    from rq.job import Job
except Exception:
    Redis = None
    Queue = None
    Job = None

from shared.training_registry import (
    get_progress, get_logs, get_metrics, list_artifacts, get_artifact,
    list_completed, get_state
)

def has_redis() -> bool:
    return bool(os.getenv("REDIS_URL"))

def _redis() -> Optional["Redis"]:
    if Redis is None:
        return None
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    return Redis.from_url(url)

def _queue() -> Optional["Queue"]:
    if Queue is None:
        return None
    r = _redis()
    if not r: return None
    name = os.getenv("RQ_QUEUE", "default")
    return Queue(name, connection=r)

def enqueue_job(func_path: str, payload: dict, **rq_kwargs) -> Optional[str]:
    """
    Enqueue RQ job with flexible options.
    Examples of rq_kwargs: job_timeout=..., ttl=..., result_ttl=..., failure_ttl=...
    """
    q = _queue()
    if not q: return None
    # default long timeout if not provided
    rq_kwargs.setdefault("job_timeout", int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "86400")))
    job = q.enqueue(func_path, payload, **rq_kwargs)
    return job.id if job else None

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return minimal RQ job status; combine with persistent registry for richer UI."""
    if Job is None: return None
    r = _redis()
    if not r: return None
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        return {"status": get_state(job_id) or "unknown"}
    status = job.get_status(refresh=True)
    return {"status": status, "id": job_id}

# ---- Persistent training helpers for Streamlit ----
def training_progress(job_id: str) -> Optional[dict]:
    return get_progress(job_id)

def training_logs(job_id: str, last_n: int = 200) -> List[str]:
    return get_logs(job_id, last_n)

def training_metrics(job_id: str) -> Optional[dict]:
    return get_metrics(job_id)

def training_artifacts(job_id: str) -> List[str]:
    return list_artifacts(job_id)

def download_artifact(job_id: str, name: str) -> Optional[bytes]:
    return get_artifact(job_id, name)

def completed_trainings() -> List[str]:
    return list_completed()