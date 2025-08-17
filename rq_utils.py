# rq_utils.py
import os
import json
import logging
import importlib
from typing import Optional, Dict, Any, List, Tuple, Set

import redis
from rq import Queue, Worker
from rq.job import Job

logger = logging.getLogger("rq_utils")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------- Defaults ----------------
_DEFAULT_RESULT_TTL = int(os.getenv("RQ_RESULT_TTL", str(7 * 24 * 3600)))
_DEFAULT_FAILURE_TTL = int(os.getenv("RQ_FAILURE_TTL", str(7 * 24 * 3600)))
_DEFAULT_COMPUTE_TIMEOUT = int(os.getenv("RQ_COMPUTE_TIMEOUT", "3600"))
_DEFAULT_TRAIN_TIMEOUT   = int(os.getenv("RQ_TRAIN_TIMEOUT", "86400"))

def _coalesce_url() -> Optional[str]:
    for k in ("REDIS_URL", "REDIS_TLS_URL", "UPSTASH_REDIS_URL", "REDIS_URI", "Redis.REDIS_URL"):
        v = os.getenv(k)
        if v:
            return v
    return None

def get_connection() -> Optional[redis.Redis]:
    url = _coalesce_url()
    if not url:
        return None
    try:
        return redis.from_url(url, decode_responses=False)
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        return None

def has_redis() -> bool:
    conn = get_connection()
    if not conn:
        return False
    try:
        conn.ping()
        return True
    except Exception:
        return False

# ---------------- Worker/Queue discovery ----------------
def _workers_queues() -> Set[str]:
    conn = get_connection()
    names: Set[str] = set()
    if not conn:
        return names
    try:
        for w in Worker.all(connection=conn):
            for q in getattr(w, "queues", []):
                try:
                    names.add(getattr(q, "name", str(q)))
                except Exception:
                    pass
    except Exception:
        pass
    return names

def list_queues() -> List[Tuple[str, int]]:
    conn = get_connection()
    if not conn:
        return []
    out = []
    try:
        for q in Queue.all(connection=conn):
            try:
                out.append((q.name, q.count))
            except Exception:
                pass
    except Exception:
        pass
    return out

def list_workers() -> List[Dict[str, Any]]:
    conn = get_connection()
    if not conn:
        return []
    info = []
    try:
        for w in Worker.all(connection=conn):
            info.append({
                "name": getattr(w, "name", "?"),
                "state": getattr(w, "state", "?"),
                "queues": [getattr(qq, "name", str(qq)) for qq in getattr(w, "queues", [])]
            })
    except Exception:
        pass
    return info

# ---------------- Persistent training logs ----------------
def _log_key(job_id: str) -> str:
    return f"rq:log:{job_id}"

def append_training_log(job_id: str, event: Dict[str, Any]) -> None:
    try:
        conn = get_connection()
        if not conn:
            return
        conn.rpush(_log_key(job_id), json.dumps(event, ensure_ascii=False))
        conn.ltrim(_log_key(job_id), -2000, -1)
    except Exception as e:
        logger.debug(f"append_training_log error: {e}")

def read_training_log(job_id: str, start: int = 0, stop: int = -1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        conn = get_connection()
        if not conn:
            return out
        items = conn.lrange(_log_key(job_id), start, stop)
        for raw in items:
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode()
                out.append(json.loads(raw))
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"read_training_log error: {e}")
    return out

# ---------------- Queue selection (dynamic & robust) ----------------
def _decide_queue(requested: Optional[str]) -> str:
    """
    Priority:
      1) RQ_FORCE_QUEUE (if set)  -> always use it
      2) requested (explicit)     -> use it
      3) RQ_QUEUE (env)           -> use if any worker listens to it
      4) if a worker listens to 'ode_jobs' -> use it
      5) if a worker listens to 'default'  -> use it
      6) fallback to RQ_QUEUE or 'ode_jobs'
    """
    force = os.getenv("RQ_FORCE_QUEUE")
    if force:
        return force

    if requested:
        return requested

    env_q = os.getenv("RQ_QUEUE")
    workers = _workers_queues()
    if env_q and env_q in workers:
        return env_q
    if "ode_jobs" in workers:
        return "ode_jobs"
    if "default" in workers:
        return "default"
    # No visible workers: prefer env or ode_jobs
    return env_q or "ode_jobs"

def get_queue(queue_name: Optional[str] = None) -> Optional[Queue]:
    conn = get_connection()
    if not conn:
        return None
    chosen = _decide_queue(queue_name)
    return Queue(name=chosen, connection=conn)

# ---------------- Enqueue / Fetch ----------------
def _import_callable(func_path: str):
    mod_name, func_name = func_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, func_name)

def enqueue_job(func_path: str,
                payload: Dict[str, Any],
                queue_name: Optional[str] = None,
                job_timeout: Optional[int] = None,
                result_ttl: Optional[int] = None,
                failure_ttl: Optional[int] = None,
                description: Optional[str] = None,
                *compat_args, **compat_kwargs) -> Optional[str]:
    """
    Backward-compatible enqueuer. Chooses the correct queue automatically.
    """
    try:
        q = get_queue(queue_name)
        if not q:
            logger.error("enqueue_job: No queue (Redis missing?)")
            return None

        func = _import_callable(func_path)
        is_train = func_path.endswith("train_job")
        jt = job_timeout if job_timeout is not None else (_DEFAULT_TRAIN_TIMEOUT if is_train else _DEFAULT_COMPUTE_TIMEOUT)
        rt = _DEFAULT_RESULT_TTL if result_ttl is None else int(result_ttl)
        ft = _DEFAULT_FAILURE_TTL if failure_ttl is None else int(failure_ttl)

        meta = {"kind": "train" if is_train else "compute", "status": "enqueued"}

        job = q.enqueue(
            func,
            args=(payload,),
            job_timeout=jt,
            result_ttl=rt,
            failure_ttl=ft,
            meta=meta,
            description=description or (f"{'TRAIN' if is_train else 'COMPUTE'}: {func_path}")
        )
        return job.id
    except Exception as e:
        logger.error(f"enqueue_job error: {e}")
        # Ultra-compatible fallback
        try:
            q = get_queue(queue_name)
            func = _import_callable(func_path)
            job = q.enqueue(func, payload)
            return job.id
        except Exception as ee:
            logger.error(f"enqueue_job fallback failed: {ee}")
            return None

def fetch_job(job_id: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        conn = get_connection()
        if not conn:
            return {"status": "unknown", "error": "No Redis"}
        job = Job.fetch(job_id, connection=conn)
        status = job.get_status()
        origin = getattr(job, "origin", None)

        info = {
            "id": job.id,
            "status": status,
            "origin": origin,  # <-- show which queue the job sits in
            "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
            "started_at": str(job.started_at) if job.started_at else None,
            "ended_at": str(job.ended_at) if job.ended_at else None,
            "description": getattr(job, "description", None),
            "meta": dict(getattr(job, "meta", {}) or {}),
        }
        if status == "finished":
            try:
                info["result"] = job.result
            except Exception:
                info["result"] = None
        if job.exc_info:
            exc_info = job.exc_info.decode() if isinstance(job.exc_info, bytes) else job.exc_info
            info["exc_info"] = exc_info
        # Add last logs
        try:
            tail = read_training_log(job_id, -50, -1)
            if tail:
                info["logs_tail"] = tail
        except Exception:
            pass
        return info
    except Exception as e:
        return {"status": "unknown", "error": str(e)}

def get_queue_stats() -> Dict[str, Any]:
    qs = list_queues()
    ws = list_workers()
    return {"queues": [{"name": n, "count": c} for (n, c) in qs], "workers": ws}