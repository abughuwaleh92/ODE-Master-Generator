# rq_utils.py (excerpt)
import os
from typing import Optional, Dict, Any
from rq import Queue
from redis import Redis

def has_redis() -> bool:
    url = os.getenv("REDIS_URL", "")
    return url.startswith("redis://") or url.startswith("rediss://") or url.startswith("unix://")

def _conn() -> Optional[Redis]:
    url = os.getenv("REDIS_URL", "")
    if not url:
        return None
    return Redis.from_url(url)

def enqueue_job(func_path: str, payload: dict, **rq_kwargs) -> Optional[str]:
    """
    Enqueue a job on RQ. Accepts passthrough kwargs like:
      job_timeout, ttl, result_ttl, failure_ttl, depends_on, at_front, on_success, on_failure, meta, etc.
    """
    r = _conn()
    if not r:
        return None
    qname = os.getenv("RQ_QUEUE", "ode_jobs")
    q = Queue(qname, connection=r)
    # func_path is like "worker.compute_job"
    job = q.enqueue(func_path, kwargs={"payload": payload}, **rq_kwargs)
    return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    r = _conn()
    if not r:
        return None
    from rq.job import Job
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        return None
    out: Dict[str, Any] = {
        "id": job_id,
        "status": job.get_status(refresh=False),
        "meta": dict(job.meta or {}),
    }
    if job.is_finished:
        out["result"] = job.result
    elif job.is_failed:
        try:
            out["error"] = str(job.exc_info or job._exc_info)  # rq stores traceback
        except Exception:
            out["error"] = "failed"
    return out