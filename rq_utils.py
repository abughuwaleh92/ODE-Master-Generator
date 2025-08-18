# rq_utils.py
"""
RQ helper utilities for RQ 2.x.
Provides: has_redis, get_redis, get_queue, enqueue_job, fetch_job, rq_inspect.
"""

import os
from typing import Any, Dict, Optional

from redis import Redis
from rq import Queue
from rq.job import Job

def _redis_url() -> Optional[str]:
    return os.getenv("REDIS_URL") or os.getenv("RQ_REDIS_URL")

def get_redis() -> Optional[Redis]:
    url = _redis_url()
    if not url:
        return None
    try:
        r = Redis.from_url(url)
        r.ping()
        return r
    except Exception:
        return None

def has_redis() -> bool:
    return get_redis() is not None

def get_queue(name: Optional[str] = None) -> Optional[Queue]:
    conn = get_redis()
    if not conn:
        return None
    qname = name or os.getenv("RQ_QUEUE") or "ode_jobs"
    return Queue(qname, connection=conn, default_timeout=1800)

def enqueue_job(
    func_path: str,
    payload: Dict[str, Any],
    queue: Optional[str] = None,
    job_timeout: int = 1800,
    result_ttl: int = 86400,
    description: str = "",
) -> Optional[str]:
    q = get_queue(queue)
    if not q:
        return None
    try:
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

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_redis()
    if not conn:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
        info = {
            "id": job.get_id(),
            "status": job.get_status(),
            "origin": job.origin,
            "description": job.description,
            "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
            "started_at": str(job.started_at) if job.started_at else None,
            "ended_at": str(job.ended_at) if job.ended_at else None,
            "meta": dict(job.meta or {}),
        }
        if job.is_finished:
            info["result"] = job.result
        if job.is_failed:
            info["exc_string"] = job.exc_info
        return info
    except Exception:
        return None

def rq_inspect(current_queue: Optional[str] = None) -> Dict[str, Any]:
    from rq.worker import Worker
    res = {"ok": False, "queues": [], "workers": [], "using_queue": None}
    conn = get_redis()
    if not conn:
        return res
    try:
        qname = current_queue or os.getenv("RQ_QUEUE") or "ode_jobs"
        res["using_queue"] = qname

        q_main = Queue(qname, connection=conn)
        res["queues"].append({"name": q_main.name, "count": q_main.count})
        if qname != "default":
            q_def = Queue("default", connection=conn)
            res["queues"].append({"name": q_def.name, "count": q_def.count})

        for w in Worker.all(connection=conn):
            res["workers"].append({
                "name": w.name,
                "state": w.get_state(),
                "queues": [qq.name for qq in w.queues],
            })
        res["ok"] = True
        return res
    except Exception:
        return res