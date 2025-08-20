# rq_utils.py
import os, json
from datetime import datetime
from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from rq.suspension import is_suspended

REDIS_URL = os.getenv('REDIS_URL', '')
QUEUE_NAME = os.getenv('RQ_QUEUE', 'ode_jobs')
DEFAULT_TIMEOUT = int(os.getenv('RQ_DEFAULT_TIMEOUT', '3600'))       # 1h
RESULT_TTL     = int(os.getenv('RQ_RESULT_TTL', '604800'))           # 7 days
JOB_TTL        = int(os.getenv('RQ_JOB_TTL', '86400'))               # 1 day

def _conn():
    url = REDIS_URL
    if not url or not url.startswith(('redis://','rediss://','unix://')):
        return None
    try:
        r = Redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None

def has_redis():
    return _conn() is not None

def enqueue_job(func_path: str, payload: dict, description: str | None = None):
    r = _conn()
    if not r:
        return None
    q = Queue(name=QUEUE_NAME, connection=r, default_timeout=DEFAULT_TIMEOUT)
    job = q.enqueue_call(
        func=func_path,
        kwargs={'payload': payload},
        description=description or func_path,
        result_ttl=RESULT_TTL,
        job_timeout=DEFAULT_TIMEOUT,
        ttl=JOB_TTL
    )
    job.meta['progress'] = {'stage': 'queued'}
    job.save_meta()
    return job.id

def fetch_job(job_id: str) -> dict | None:
    r = _conn()
    if not r:
        return None
    try:
        job = Job.fetch(job_id, connection=r)
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
    }
    if job.is_finished:
        try:
            info["result"] = job.result
        except Exception:
            info["result"] = None
    if job.is_failed:
        try:
            info["exc_info"] = job.exc_info
        except Exception:
            pass
    # optional convenience: cheap "logs" via meta
    info["logs"] = (job.meta or {}).get("logs", [])
    return info

def redis_status():
    r = _conn()
    if not r:
        return {"ok": False, "reason": "REDIS_URL not set or unreachable"}
    try:
        workers = [w.name for w in Worker.all(connection=r)]
        return {
            "ok": True,
            "queue": QUEUE_NAME,
            "suspended": bool(is_suspended(r)),
            "workers": workers,
            "ping": True
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}