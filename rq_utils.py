# rq_utils.py
import os
import json
from typing import Optional, Dict, Any, List

import redis
from rq import Queue, Connection
from rq.job import Job
from rq.registry import (
    StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry,
    DeferredJobRegistry, ScheduledJobRegistry
)

def _redis_url() -> Optional[str]:
    return os.getenv("REDIS_URL") or os.getenv("UPSTASH_REDIS_URL")

def _conn():
    url = _redis_url()
    if not url:
        return None
    return redis.from_url(url, decode_responses=False)

def has_redis() -> bool:
    try:
        r = _conn()
        if not r:
            return False
        r.ping()
        return True
    except Exception:
        return False

def _queue_name(fallback: str = "ode_jobs") -> str:
    # Global fallback queue, can be overridden per call
    return os.getenv("RQ_QUEUE", fallback)

def enqueue_job(
    func_path: str,
    payload: Dict[str, Any],
    *,
    queue: Optional[str] = None,
    job_timeout: Optional[int] = None,
    result_ttl: Optional[int] = None,
    description: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Enqueue a job on a specific queue (defaults to ode_jobs).
    Compatible with Railway/Upstash Redis.
    """
    r = _conn()
    if not r:
        return None

    qname = (queue or _queue_name("ode_jobs")).strip()
    jt = int(job_timeout if job_timeout is not None else os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600"))
    rt = int(result_ttl if result_ttl is not None else os.getenv("RQ_RESULT_TTL", "604800"))

    with Connection(r):
        q = Queue(name=qname)
        job = q.enqueue(
            func_path,
            payload,
            job_timeout=jt,
            result_ttl=rt,
            description=description or func_path,
        )
        if meta:
            job.meta.update(meta)
            job.save_meta()
        return job.id

def fetch_job(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Force-refresh job state and return a JSON-friendly dict.
    """
    r = _conn()
    if not r:
        return None
    with Connection(r):
        try:
            job = Job.fetch(job_id)
            # Force-refresh internal status
            status = job.get_status(refresh=True)
            info = {
                "job_id": job.id,
                "status": status,
                "description": job.description,
                "created_at": str(job.created_at) if job.created_at else None,
                "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
                "started_at": str(job.started_at) if job.started_at else None,
                "ended_at": str(job.ended_at) if job.ended_at else None,
                "meta": job.meta or {},
                "queue": getattr(job, "origin", None),
            }
            if status == "finished":
                try:
                    info["result"] = job.return_value()
                except Exception:
                    info["result"] = job.result
            if job.exc_info:
                info["exc_info"] = job.exc_info
            return info
        except Exception:
            return None

def _collect_ids_from_registry(reg) -> List[str]:
    try:
        return reg.get_job_ids()[:200]
    except Exception:
        return []

def list_runs(queues: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Summarize jobs across registries for the given queues (or env RQ_QUEUES or default).
    """
    r = _conn()
    if not r:
        return []

    qnames = queues or os.getenv("RQ_QUEUES", "").split(",")
    qnames = [q.strip() for q in qnames if q.strip()]
    if not qnames:
        # default set covers compute/train/reverse
        qnames = ["ode_jobs", "ml_jobs", "reverse_jobs"]

    runs: List[Dict[str, Any]] = []
    with Connection(r):
        for qn in qnames:
            q = Queue(name=qn)
            # Registries
            regs = [
                ("started", StartedJobRegistry(qn)),
                ("finished", FinishedJobRegistry(qn)),
                ("failed", FailedJobRegistry(qn)),
                ("deferred", DeferredJobRegistry(qn)),
                ("scheduled", ScheduledJobRegistry(qn)),
            ]
            for label, reg in regs:
                for jid in _collect_ids_from_registry(reg):
                    try:
                        job = Job.fetch(jid)
                        status = job.get_status(refresh=False) or label
                        summary = job.meta.get("summary", {}) if job.meta else {}
                        runs.append({
                            "job_id": jid,
                            "queue": qn,
                            "status": status,
                            "description": job.description,
                            "summary": summary,
                            "created_at": str(job.created_at) if job.created_at else None,
                            "enqueued_at": str(job.enqueued_at) if job.enqueued_at else None,
                            "started_at": str(job.started_at) if job.started_at else None,
                            "ended_at": str(job.ended_at) if job.ended_at else None,
                        })
                    except Exception:
                        continue
    # de-dup by job_id keeping the most recent record
    unique = {}
    for r0 in runs:
        unique[r0["job_id"]] = r0
    return list(unique.values())

def load_run(best_model_path: str) -> Dict[str, Any]:
    """Tiny helper if you need to read a JSON sidecar next to a model file, optional."""
    sidecar = best_model_path + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}