# rq_utils.py
import os
from typing import Optional, List, Dict, Any
import redis
from rq import Queue
from rq.job import Job

# ---------- Redis connection ----------
def _redis_url() -> str:
    url = (
        os.getenv("REDIS_URL")
        or os.getenv("REDIS_TLS_URL")   # some providers expose only TLS url
        or os.getenv("UPSTASH_REDIS_URL")
        or os.getenv("REDIS_URI")
    )
    if not url:
        raise RuntimeError("REDIS_URL (or REDIS_TLS_URL) not set")
    return url

def _redis_conn():
    # decode_responses=False => keep pickled objects intact
    return redis.from_url(_redis_url(), decode_responses=False)

def has_redis() -> bool:
    try:
        _redis_conn().ping()
        return True
    except Exception:
        return False

def _queue_name(name: Optional[str]) -> str:
    return name or os.getenv("RQ_QUEUE", "ode_jobs")

def _queue(name: Optional[str] = None) -> Queue:
    qname = _queue_name(name)
    default_timeout = int(os.getenv("RQ_DEFAULT_TIMEOUT", "3600"))
    return Queue(qname, connection=_redis_conn(), default_timeout=default_timeout)

# ---------- Enqueue / Fetch ----------
def enqueue_job(
    func_path: str,
    payload: Optional[Dict[str, Any]] = None,
    queue_name: Optional[str] = None,
    job_timeout: Optional[int] = None,
    result_ttl: Optional[int] = None,
    failure_ttl: Optional[int] = None,
    job_id: Optional[str] = None,
) -> Optional[str]:
    """
    Enqueue a job onto the configured queue.
    - func_path: 'module.function' string (e.g., 'worker.compute_job')
    - payload: dict passed as the single argument to the function
    """
    if not has_redis():
        return None
    q = _queue(queue_name)
    if job_timeout is None:
        job_timeout = int(os.getenv("RQ_JOB_TIMEOUT", "1800"))  # 30m default
    if result_ttl is None:
        result_ttl = int(os.getenv("RQ_RESULT_TTL", "86400"))   # 24h
    if failure_ttl is None:
        failure_ttl = int(os.getenv("RQ_FAILURE_TTL", "86400")) # 24h
    job = q.enqueue(
        func_path,
        payload or {},
        job_id=job_id,
        job_timeout=job_timeout,
        result_ttl=result_ttl,
        failure_ttl=failure_ttl,
        at_front=False,
    )
    return job.id

def fetch_job(job_id: str) -> Dict[str, Any]:
    """
    Return a small status dict with 'status' and 'result' (if finished),
    plus timing and error info.
    """
    try:
        job = Job.fetch(job_id, connection=_redis_conn())
    except Exception as e:
        return {"status": "unknown", "error": str(e)}
    status = job.get_status()
    out: Dict[str, Any] = {
        "id": job.id,
        "status": status,
        "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
        "started_at": str(job.started_at) if job.started_at else None,
        "ended_at": str(job.ended_at) if job.ended_at else None,
        "exc_info": job.exc_info.decode() if isinstance(job.exc_info, bytes) else job.exc_info,
    }
    if status == "finished":
        try:
            out["result"] = job.result
        except Exception:
            out["result"] = None
    return out

# ---------- Optional training helpers (no-ops here if you already have your own) ----------
def training_progress(job_id: str) -> Optional[Dict[str, Any]]:
    # if you already implemented progress via Redis keys, keep your own version.
    return None

def training_logs(job_id: str, last_n: int = 200) -> List[str]:
    return []

def training_metrics(job_id: str) -> Optional[Dict[str, Any]]:
    return None

def training_artifacts(job_id: str) -> List[str]:
    return []

def download_artifact(job_id: str, name: str) -> Optional[bytes]:
    return None

def completed_trainings() -> List[str]:
    return []