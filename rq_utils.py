# rq_utils.py
import os
from datetime import datetime
from redis import Redis
from rq import Queue, Worker
from rq.job import Job
from rq.suspension import is_suspended

# Environment
REDIS_URL       = os.getenv("REDIS_URL", "")
QUEUE_NAME      = os.getenv("RQ_QUEUE", "ode_jobs")
DEFAULT_TIMEOUT = int(os.getenv("RQ_DEFAULT_TIMEOUT", "3600"))   # seconds
RESULT_TTL      = int(os.getenv("RQ_RESULT_TTL", "604800"))      # 7 days
JOB_TTL         = int(os.getenv("RQ_JOB_TTL", "86400"))          # 1 day

def _conn():
    """Return a working Redis connection or None."""
    url = REDIS_URL
    if not url or not url.startswith(("redis://", "rediss://", "unix://")):
        return None
    try:
        r = Redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None

def has_redis() -> bool:
    return _conn() is not None

def enqueue_job(func_path: str, payload: dict, description: str | None = None,
                timeout: int | None = None, result_ttl: int | None = None, ttl: int | None = None):
    """
    Enqueue an RQ job using parameters compatible with older/newer RQ versions.
    NOTE: RQ expects `timeout`, not `job_timeout`.
    """
    r = _conn()
    if not r:
        return None

    q = Queue(name=QUEUE_NAME, connection=r, default_timeout=DEFAULT_TIMEOUT)
    timeout    = DEFAULT_TIMEOUT if timeout is None else timeout
    result_ttl = RESULT_TTL if result_ttl is None else result_ttl
    ttl        = JOB_TTL if ttl is None else ttl

    # Build common kwargs
    call_kwargs = dict(
        func=func_path,
        kwargs={"payload": payload},
        description=description or func_path,
        timeout=timeout,            # <-- correct name across RQ versions
        result_ttl=result_ttl,
        ttl=ttl,
        meta={"progress": {"stage": "queued"}}
    )

    # Some very old RQ versions may not accept ttl/result_ttl/meta; guard with try/except.
    try:
        job = q.enqueue_call(**call_kwargs)
    except TypeError:
        # Fallback: only pass what’s sure to exist
        job = q.enqueue_call(
            func=func_path,
            kwargs={"payload": payload},
            description=description or func_path,
            timeout=timeout
        )
        # Manually set meta if possible
        try:
            job.meta["progress"] = {"stage": "queued"}
            job.save_meta()
        except Exception:
            pass

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
    # Surface logs if we wrote them in worker
    info["logs"] = (job.meta or {}).get("logs", [])
    return info

def redis_status():
    r = _conn()
    if not r:
        return {"ok": False, "reason": "REDIS_URL missing/invalid or Redis unreachable"}
    try:
        workers = [w.name for w in Worker.all(connection=r)]
        return {
            "ok": True,
            "queue": QUEUE_NAME,
            "suspended": bool(is_suspended(r)),
            "workers": workers,
            "ping": True,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}