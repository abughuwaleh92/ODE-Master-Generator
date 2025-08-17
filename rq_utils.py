# rq_utils.py
import os, json, time
from typing import Optional, Dict, Any, List
import redis as _redis
from rq import Queue
from rq.job import Job

# ---------- Redis connection ----------
def get_redis_conn():
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL")
    if not url:
        return None
    try:
        return _redis.from_url(url)
    except Exception:
        return None

def has_redis() -> bool:
    return get_redis_conn() is not None

# ---------- Helper: Queue ----------
def _get_queue(name: str = "default") -> Optional[Queue]:
    conn = get_redis_conn()
    if not conn:
        return None
    return Queue(name, connection=conn, default_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600")))

# ---------- Job enqueue/fetch/list ----------
def enqueue_job(func_path: str,
                payload: dict,
                queue_name: str = "default",
                job_timeout: Optional[int] = None,
                result_ttl: Optional[int] = None,
                description: Optional[str] = None,
                meta: Optional[dict] = None) -> Optional[str]:
    """
    Enqueue a job with optional timeout/result_ttl and meta.
    Returns job_id or None if Redis unavailable.
    """
    q = _get_queue(queue_name)
    if not q:
        return None
    kwargs = {
        "args": (payload,),
        "meta": meta or {},
        "description": description or "",
    }
    if job_timeout is not None:
        kwargs["job_timeout"] = job_timeout
    if result_ttl is not None:
        kwargs["result_ttl"] = result_ttl
    else:
        kwargs["result_ttl"] = int(os.getenv("RQ_RESULT_TTL", "604800"))  # 7 days default

    job = q.enqueue(func_path, **kwargs)
    # create a persistent run record for training jobs
    if func_path.endswith("train_job"):
        _register_run(job.id, {
            "job_id": job.id,
            "queue": queue_name,
            "func": func_path,
            "status": "queued",
            "created_at": time.time(),
            "meta": kwargs["meta"],
            "description": kwargs["description"]
        })
    return job.id

def fetch_job(job_id: str, queue_name: str = "default") -> Optional[Dict[str, Any]]:
    conn = get_redis_conn()
    if not conn:
        return None
    try:
        job = Job.fetch(job_id, connection=conn)
    except Exception:
        return _load_run(job_id)  # fall back to persisted run record if job expired
    data = {
        "id": job.get_id(),
        "status": job.get_status(),
        "created_at": getattr(job, "created_at", None),
        "enqueued_at": getattr(job, "enqueued_at", None),
        "started_at": getattr(job, "started_at", None),
        "ended_at": getattr(job, "ended_at", None),
        "meta": job.meta or {},
        "description": job.description or "",
    }
    if job.is_finished:
        try:
            data["result"] = job.result
        except Exception:
            data["result"] = None
    if job.is_failed:
        try:
            data["exc_info"] = job.exc_info
        except Exception:
            data["exc_info"] = None
    return data

def list_jobs(queue_name: str = "default",
              statuses: Optional[List[str]] = None,
              limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return a lightweight list of recent jobs by reading our run registry and merging with live RQ when available.
    """
    conn = get_redis_conn()
    runs = _list_runs()
    out = []
    if conn:
        # Try to merge with live info if the job still exists
        for r in runs:
            if statuses and r.get("status") not in statuses:
                continue
            try:
                job = Job.fetch(r["job_id"], connection=conn)
                r = {**r, "status": job.get_status(), "meta": job.meta or r.get("meta", {})}
            except Exception:
                pass
            out.append(r)
    else:
        # Offline: show registry only
        out = [r for r in runs if (not statuses or r.get("status") in statuses)]
    return out[:limit]

# ---------- Logging helpers ----------
def append_log(job: Job, line: str, keep: int = 200):
    """
    Append a short line to job.meta['log_tail'] and save_meta.
    """
    try:
        tail = job.meta.get("log_tail", [])
        tail.append(line)
        if len(tail) > keep:
            tail = tail[-keep:]
        job.meta["log_tail"] = tail
        job.save_meta()
    except Exception:
        pass

# ---------- Run registry (Redis + file mirror) ----------
_RUNS_KEY = "mg:runs:index"
_RUNS_FILE = os.path.join("checkpoints", "runs_index.json")

def _register_run(job_id: str, run: Dict[str, Any]):
    conn = get_redis_conn()
    if conn:
        try:
            conn.hset(_RUNS_KEY, job_id, json.dumps(run))
        except Exception:
            pass
    _mirror_runs_to_file(job_id, run)

def _update_run(job_id: str, patch: Dict[str, Any]):
    run = _load_run(job_id) or {"job_id": job_id}
    run.update(patch)
    conn = get_redis_conn()
    if conn:
        try:
            conn.hset(_RUNS_KEY, job_id, json.dumps(run))
        except Exception:
            pass
    _mirror_runs_to_file(job_id, run)

def _load_run(job_id: str) -> Optional[Dict[str, Any]]:
    # from Redis
    conn = get_redis_conn()
    if conn:
        try:
            raw = conn.hget(_RUNS_KEY, job_id)
            if raw:
                return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            pass
    # from file
    try:
        if os.path.exists(_RUNS_FILE):
            with open(_RUNS_FILE, "r") as f:
                idx = json.load(f)
            return idx.get(job_id)
    except Exception:
        pass
    return None

def _list_runs() -> List[Dict[str, Any]]:
    # prefer Redis
    conn = get_redis_conn()
    out = []
    if conn:
        try:
            entries = conn.hgetall(_RUNS_KEY)
            for k, v in entries.items():
                try:
                    out.append(json.loads(v.decode("utf-8") if isinstance(v, bytes) else v))
                except Exception:
                    continue
            # fall through to merge file if any extra
        except Exception:
            pass
    # merge from file
    try:
        if os.path.exists(_RUNS_FILE):
            with open(_RUNS_FILE, "r") as f:
                idx = json.load(f)
            # index is dict job_id -> run
            existing_ids = {r["job_id"] for r in out if "job_id" in r}
            for jid, run in idx.items():
                if jid not in existing_ids:
                    out.append(run)
    except Exception:
        pass
    # sort by created_at desc if available
    out.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return out

def _mirror_runs_to_file(job_id: str, run: Dict[str, Any]):
    try:
        os.makedirs("checkpoints", exist_ok=True)
        idx = {}
        if os.path.exists(_RUNS_FILE):
            with open(_RUNS_FILE, "r") as f:
                idx = json.load(f)
        idx[job_id] = run
        with open(_RUNS_FILE, "w") as f:
            json.dump(idx, f, indent=2)
    except Exception:
        pass

# Public re-exports for app pages
register_run = _register_run
update_run = _update_run
load_run = _load_run
list_runs = _list_runs