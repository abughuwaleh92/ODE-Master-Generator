# rq_utils.py
import os
import logging
from typing import Optional, Dict, Any

import redis
from rq import Queue, Worker
from rq.job import Job

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "")
RQ_QUEUE = os.environ.get("RQ_QUEUE", "ode_jobs")


def _get_redis() -> Optional[redis.Redis]:
    """Return a live Redis connection or None."""
    if not REDIS_URL:
        return None
    try:
        r = redis.from_url(REDIS_URL)
        # Light health check; don't be too aggressive in PaaS.
        r.ping()
        return r
    except Exception as e:
        logger.warning(f"Redis connection issue: {e}")
        return None


def has_redis() -> bool:
    """True if REDIS_URL is configured and usable."""
    return _get_redis() is not None


def enqueue_job(
    func_path: str,
    payload: Dict[str, Any],
    queue: Optional[str] = None,
    **rq_kwargs: Any,
) -> Optional[str]:
    """
    Enqueue a job by function import path (e.g., 'worker.compute_job').
    - Ensures it is enqueued to the intended queue (default: env RQ_QUEUE or 'ode_jobs').
    - Forwards any RQ enqueue kwargs (job_timeout, result_ttl, failure_ttl, description, depends_on, at_front, meta, etc.).
    """
    r = _get_redis()
    if r is None:
        return None

    qname = queue or RQ_QUEUE
    q = Queue(qname, connection=r)

    # Ensure we always have a sensible initial meta
    meta = rq_kwargs.pop("meta", {}) or {}
    meta.setdefault("stage", "enqueued")

    job = q.enqueue(
        func_path,
        kwargs={"payload": payload},
        meta=meta,
        **rq_kwargs,
    )
    return job.get_id()


def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a snapshot of job status/fields or None if not found."""
    r = _get_redis()
    if r is None:
        return None
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        return None

    snap = {
        "id": job.id,
        "status": job.get_status(),
        "origin": job.origin,
        "description": getattr(job, "description", "") or "",
        "created_at": getattr(job, "created_at", None),
        "enqueued_at": getattr(job, "enqueued_at", None),
        "started_at": getattr(job, "started_at", None),
        "ended_at": getattr(job, "ended_at", None),
        "result": job.result,
        "meta": dict(job.meta or {}),
        "exc_info": job.exc_info,
    }
    return snap


def rq_inspect() -> Dict[str, Any]:
    """
    Lightweight queues/workers snapshot for the UI panel.
    """
    r = _get_redis()
    if r is None:
        return {"error": "No Redis connection (set REDIS_URL)."}

    try:
        # Queues
        queues = []
        for q in Queue.all(connection=r):
            try:
                count = q.count
            except Exception:
                count = None
            queues.append({"name": q.name, "count": count})

        # Workers
        workers = []
        for w in Worker.all(connection=r):
            try:
                qnames = [qq.name for qq in w.queues]
            except Exception:
                qnames = []
            try:
                state = w.get_state()
            except Exception:
                state = "unknown"
            workers.append({"name": w.name, "state": state, "queues": ",".join(qnames)})

        return {"queues": queues, "workers": workers}
    except Exception as e:
        return {"error": str(e)}