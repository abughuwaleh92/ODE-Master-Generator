# rq_utils.py
import os
import json
from typing import Optional, Dict, Any, List

import redis
from rq import Queue
from rq.job import Job

REDIS_URL = (
    os.getenv("REDIS_URL")
    or os.getenv("UPSTASH_REDIS_URL")
    or os.getenv("REDISCLOUD_URL")
    or ""
)

def has_redis() -> bool:
    return bool(REDIS_URL and (REDIS_URL.startswith("redis://") or REDIS_URL.startswith("rediss://") or REDIS_URL.startswith("unix://")))

def _redis() -> Optional[redis.Redis]:
    if not has_redis():
        return None
    return redis.from_url(REDIS_URL, decode_responses=True)

def _queue(name: str = "default") -> Queue:
    r = _redis()
    return Queue(name, connection=r, default_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "86400")))

def enqueue_job(func_path: str, payload: dict, **rq_kwargs) -> Optional[str]:
    """
    Enqueue with flexible kwargs (fixes 'unexpected keyword' error).
    e.g. enqueue_job("worker.train_job", payload, job_timeout=86400, result_ttl=604800)
    """
    if not has_redis():
        return None
    q = _queue("default")
    try:
        job = q.enqueue(func_path, payload, **rq_kwargs)
    except TypeError:
        # Fall back to minimal call if local RQ lacks some kwarg
        job = q.enqueue(func_path, payload)
    return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    if not has_redis(): return None
    r = _redis()
    try:
        job = Job.fetch(job_id, connection=r)
    except Exception:
        return None
    meta = job.meta or {}
    return {
        "id": job.id,
        "status": job.get_status(refresh=True),
        "meta": meta,
        "created_at": str(job.created_at) if job.created_at else None,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
        "result": job.result if job.is_finished else None,
    }

# ---------- persistent progress in Redis ----------
def _key(job_id: str, which: str) -> str:
    return f"mg:train:{job_id}:{which}"

def push_log(job_id: str, line: str):
    r = _redis()
    if not r: return
    r.rpush(_key(job_id, "log"), line)

def get_logs(job_id: str, start: int = 0, end: int = -1) -> List[str]:
    r = _redis()
    if not r: return []
    return r.lrange(_key(job_id, "log"), start, end)

def set_progress(job_id: str, progress: Dict[str, Any]):
    r = _redis()
    if not r: return
    r.hset(_key(job_id, "prog"), mapping={k: json.dumps(v) for k, v in progress.items()})

def get_progress(job_id: str) -> Dict[str, Any]:
    r = _redis()
    if not r: return {}
    raw = r.hgetall(_key(job_id, "prog"))
    return {k: json.loads(v) for k, v in raw.items()}

def set_artifacts(job_id: str, paths: Dict[str, str]):
    r = _redis()
    if not r: return
    r.hset(_key(job_id, "artifacts"), mapping=paths)

def get_artifacts(job_id: str) -> Dict[str, str]:
    r = _redis()
    if not r: return {}
    return r.hgetall(_key(job_id, "artifacts"))