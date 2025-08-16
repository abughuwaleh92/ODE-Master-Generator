# rq_utils.py
import os
import json
from typing import Optional, Any, Dict

def has_redis() -> bool:
    return bool(os.environ.get("REDIS_URL"))

# Only import redis/rq if present, to allow local sync mode.
_redis = None
_rq = None
if has_redis():
    import redis as _redis
    import rq as _rq

def get_queue():
    if not has_redis():
        return None
    conn = _redis.from_url(os.environ["REDIS_URL"])
    return _rq.Queue("ode_jobs", connection=conn), conn

def enqueue_job(func_path: str, payload: Dict[str, Any]) -> Optional[str]:
    """
    func_path: e.g. 'worker.compute_job'
    """
    if not has_redis():
        return None
    q, _ = get_queue()
    job = q.enqueue(func_path, payload)
    return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    if not has_redis():
        return None
    _, conn = get_queue()
    job = _rq.job.Job.fetch(job_id, connection=conn)
    if job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)}
    if job.is_finished:
        return {"status": "finished", "result": job.result}
    return {"status": "queued"}
