# rq_utils.py
import os
import json
import logging
import importlib
from typing import Optional, Dict, Any, List, Tuple

import redis
from rq import Queue, Worker
from rq.job import Job

logger = logging.getLogger("rq_utils")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------- Environment / Defaults ----------
_DEFAULT_QUEUE = os.getenv("RQ_QUEUE", "ode_jobs")  # change to "default" if your worker listens to default
_DEFAULT_RESULT_TTL = int(os.getenv("RQ_RESULT_TTL", str(7 * 24 * 3600)))   # 7 days
_DEFAULT_FAILURE_TTL = int(os.getenv("RQ_FAILURE_TTL", str(7 * 24 * 3600))) # 7 days
_DEFAULT_COMPUTE_TIMEOUT = int(os.getenv("RQ_COMPUTE_TIMEOUT", "3600"))     # 1h
_DEFAULT_TRAIN_TIMEOUT   = int(os.getenv("RQ_TRAIN_TIMEOUT", "86400"))      # 24h

# ---------- Redis connection ----------
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

# ---------- Queues / Workers ----------
def get_queue(queue_name: Optional[str] = None) -> Optional[Queue]:
    conn = get_connection()
    if not conn:
        return None
    return Queue(name=(queue_name or _DEFAULT_QUEUE), connection=conn)

def list_queues() -> List[Tuple[str, int]]:
    conn = get_connection()
    if not conn:
        return []
    out = []
    try:
        for qname in Queue.all(connection=conn):
            try:
                out.append((qname.name, qname.count))
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
            try:
                info.append({
                    "name": getattr(w, "name", "?"),
                    "state": getattr(w, "state", "?"),
                    "queues": [getattr(qq, "name", str(qq)) for qq in getattr(w, "queues", [])]
                })
            except Exception:
                pass
    except Exception:
        pass
    return info

# ---------- Logging helpers (persistent in Redis) ----------
def _log_key(job_id: str) -> str:
    return f"rq:log:{job_id}"

def append_training_log(job_id: str, event: Dict[str, Any]) -> None:
    """
    Append a JSON-serializable dict to Redis list for persistent viewing.
    """
    try:
        conn = get_connection()
        if not conn: 
            return
        conn.rpush(_log_key(job_id), json.dumps(event, ensure_ascii=False))
        # optional trim to last N entries to avoid unbounded growth
        conn.ltrim(_log_key(job_id), -2000, -1)
    except Exception as e:
        logger.debug(f"append_training_log error: {e}")

def read_training_log(job_id: str, start: int = 0, stop: int = -1) -> List[Dict[str, Any]]:
    """
    Read persistent logs. stop=-1 returns all.
    """
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

# ---------- Enqueue / Fetch ----------
def _import_callable(func_path: str):
    """
    func_path like 'worker.compute_job' or 'worker.train_job'
    """
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
    Backward-compatible enqueuer. Accepts old and new signatures.
    Returns job_id or None.
    """
    try:
        # resolve queue and function
        q = get_queue(queue_name)
        if not q:
            logger.error("enqueue_job: No queue (Redis missing?)")
            return None
        func = _import_callable(func_path)

        # default TTL/timeouts
        is_train = func_path.endswith("train_job")
        jt = job_timeout if job_timeout is not None else (_DEFAULT_TRAIN_TIMEOUT if is_train else _DEFAULT_COMPUTE_TIMEOUT)
        rt = _DEFAULT_RESULT_TTL if result_ttl is None else int(result_ttl)
        ft = _DEFAULT_FAILURE_TTL if failure_ttl is None else int(failure_ttl)
        meta = {
            "kind": "train" if is_train else "compute",
            "status": "enqueued",
        }

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
        # Try extremely compatible path: enqueue_call with minimal args
        try:
            q = get_queue(queue_name)
            func = _import_callable(func_path)
            job = q.enqueue(func, payload)
            return job.id
        except Exception as ee:
            logger.error(f"enqueue_job fallback failed: {ee}")
            return None

def fetch_job(job_id: str) -> Dict[str, Any]:
    """
    Fetch job info and status in a JSON-friendly dict.
    """
    info: Dict[str, Any] = {}
    try:
        conn = get_connection()
        if not conn:
            return {"status": "unknown", "error": "No Redis"}
        job = Job.fetch(job_id, connection=conn)
        status = job.get_status()
        info = {
            "id": job.id,
            "status": status,
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
        # attach last 50 logs
        try:
            logs = read_training_log(job_id, -50, -1)
            if logs:
                info["logs_tail"] = logs
        except Exception:
            pass
        # exception info
        try:
            exc_info = job.exc_info
            if isinstance(exc_info, bytes):
                exc_info = exc_info.decode()
            if exc_info:
                info["exc_info"] = exc_info
        except Exception:
            pass
        return info
    except Exception as e:
        return {"status": "unknown", "error": str(e)}

def get_queue_stats() -> Dict[str, Any]:
    """
    Small pack of queue + workers diagnostics for UI.
    """
    qs = list_queues()
    ws = list_workers()
    return {"queues": [{"name": n, "count": c} for (n, c) in qs], "workers": ws}