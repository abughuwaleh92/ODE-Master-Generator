# rq_utils.py
import os
from typing import Any, Dict, Optional

from rq import Queue
from rq.job import Job
from redis import Redis

# RQ connection used for jobs (binary safe for pickling)
def _redis_rq() -> Optional[Redis]:
    url = (
        os.getenv("REDIS_URL")
        or os.getenv("REDIS_TLS_URL")
        or os.getenv("UPSTASH_REDIS_URL")
    )
    if not url:
        return None
    # decode_responses=False is important for pickling payloads
    return Redis.from_url(url, decode_responses=False)

def has_redis() -> bool:
    return _redis_rq() is not None

def _queue(name: Optional[str] = None) -> Queue:
    conn = _redis_rq()
    if conn is None:
        raise RuntimeError("REDIS_URL not set or Redis unavailable.")
    qname = name or os.getenv("RQ_QUEUE", "ode_jobs")
    return Queue(name=qname, connection=conn)

def enqueue_job(func_path: str, payload: Dict[str, Any], **job_opts) -> Optional[str]:
    """
    Enqueue a job. Accepts common options if supported by the installed RQ:
    - job_timeout, result_ttl, ttl, description, meta
    """
    q = _queue(job_opts.pop("queue", None))
    # filter known enqueue options
    allowed = {"job_timeout", "result_ttl", "ttl", "description", "meta", "depends_on"}
    opts = {k: v for k, v in job_opts.items() if k in allowed}
    # sensible defaults
    opts.setdefault("job_timeout", int(os.getenv("RQ_JOB_TIMEOUT", "86400")))     # 24h
    opts.setdefault("result_ttl", int(os.getenv("RQ_RESULT_TTL", "604800")))     # 7 days
    opts.setdefault("ttl", int(os.getenv("RQ_TTL", "86400")))                    # 1 day in queue
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
    # attach plain-text logs if we used worker.py's logger
    try:
        rtext = Redis.from_url(
            os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or os.getenv("UPSTASH_REDIS_URL"),
            decode_responses=True,
        )
        logs = rtext.lrange(f"job:{job.id}:logs", 0, -1)
        info["logs"] = logs or []
    except Exception:
        info["logs"] = []
    return info