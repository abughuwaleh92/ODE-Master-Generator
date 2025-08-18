# rq_utils.py
"""
RQ helper utilities (RQ 2.x compatible).
- No 'Connection' import (removed in RQ 2).
- Uses redis-py client for connections.
- Provides: has_redis, get_redis, get_queue, enqueue_job, fetch_job, rq_inspect.
"""

import os
from typing import Any, Dict, Optional, Tuple

from redis import Redis
from rq import Queue
from rq.job import Job

# ----------------------------- Redis / Queue -----------------------------

def _redis_url() -> Optional[str]:
    # Prefer REDIS_URL; allow RQ_REDIS_URL fallback
    return os.getenv("REDIS_URL") or os.getenv("RQ_REDIS_URL")

def get_redis() -> Optional[Redis]:
    url = _redis_url()
    if not url:
        return None
    try:
        r = Redis.from_url(url)
        # verify quickly
        r.ping()
        return r
    except Exception:
        return None

def has_redis() -> bool:
    return get_redis() is not None

def get_queue(name: Optional[str] = None) -> Optional[Queue]:
    """Return Queue instance bound to configured Redis."""
    conn = get_redis()
    if not conn:
        return None
    qname = name or os.getenv("RQ_QUEUE") or "ode_jobs"
    # give each job a sane default timeout unless overridden when enqueuing
    return Queue(qname, connection=conn, default_timeout=1800)

# ------------------------------ Enqueue ---------------------------------

def enqueue_job(
    func_path: str,
    payload: Dict[str, Any],
    queue: Optional[str] = None,
    job_timeout: int = 1800,
    result_ttl: int = 86400,
    description: str = "",
) -> Optional[str]:
    """
    Enqueue a job by function import path string (e.g., 'worker.compute_job').
    Returns job id or None.
    """
    q = get_queue(queue)
    if not q:
        return None
    try:
        # Use import string so the worker doesn't need the object pickled from web
        job = q.enqueue(
            func_path,
            args=(payload,),
            job_timeout=job_timeout,
            result_ttl=result_ttl,
            failure_ttl=max(result_ttl, 7 * 24 * 3600),
            description=description or "RQ job",
            retry=None,
        )
        return job.get_id()
    except Exception:
        return None

# ------------------------------ Fetch -----------------------------------

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a JSON-safe snapshot of an RQ job."""
    conn = get_redis()
    if not conn:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
        info = {
            "id": job.get_id(),
            "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
            "started_at": str(job.started_at) if job.started_at else None,
            "ended_at": str(job.ended_at) if job.ended_at else None,
            "status": job.get_status(),
            "origin": job.origin,
            "description": job.description,
            "meta": dict(job.meta or {}),
        }
        # Only attach result safely after finished
        if job.is_finished:
            info["result"] = job.result
        if job.is_failed:
            info["exc_string"] = job.exc_info
        return info
    except Exception:
        return None

# ------------------------------ Inspect ---------------------------------

def rq_inspect(current_queue: Optional[str] = None) -> Dict[str, Any]:
    """
    Lightweight view of queues & workers (no deprecated Connection).
    """
    from rq import Queue
    from rq.worker import Worker

    res = {"ok": False, "queues": [], "workers": [], "using_queue": None}
    conn = get_redis()
    if not conn:
        return res

    try:
        qname = current_queue or os.getenv("RQ_QUEUE") or "ode_jobs"
        res["using_queue"] = qname

        # queues
        q_all = [Queue(qname, connection=conn)]
        # also show 'default' if different
        if qname != "default":
            q_all.append(Queue("default", connection=conn))

        for q in q_all:
            try:
                res["queues"].append({"name": q.name, "count": q.count})
            except Exception:
                res["queues"].append({"name": q.name, "count": None})

        # workers (simple snapshot)
        for w in Worker.all(connection=conn):
            try:
                res["workers"].append({
                    "name": w.name,
                    "state": w.get_state(),
                    "queues": [qq.name for qq in w.queues]
                })
            except Exception:
                pass

        res["ok"] = True
        return res
    except Exception:
        return res