# shared/training_registry.py
import os, time, json, base64
from typing import Optional, Dict, Any, List, Tuple

try:
    import redis
except Exception:
    redis = None

PREFIX = "mg:train"  # master-generators training namespace

def _ttl_seconds() -> int:
    days = int(os.getenv("MG_ARTIFACT_TTL_DAYS", "7"))
    return max(1, days) * 24 * 3600

def _r() -> Optional["redis.Redis"]:
    if redis is None:
        return None
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    # decode_responses=False to allow binary artifacts
    return redis.from_url(url, decode_responses=False)

def _k(job_id: str, suffix: str) -> str:
    return f"{PREFIX}:{job_id}:{suffix}"

def init_run(job_id: str, config: Dict[str, Any]) -> None:
    r = _r()
    if not r: return
    r.hset(_k(job_id, "status"), mapping={
        b"state": b"running",
        b"start_ts": str(time.time()).encode(),
    })
    r.hset(_k(job_id, "config"), mapping={b"json": json.dumps(config).encode()})
    r.expire(_k(job_id, "status"), _ttl_seconds())
    r.expire(_k(job_id, "config"), _ttl_seconds())

def set_state(job_id: str, state: str) -> None:
    r = _r()
    if not r: return
    r.hset(_k(job_id, "status"), b"state", state.encode())
    r.expire(_k(job_id, "status"), _ttl_seconds())

def get_state(job_id: str) -> Optional[str]:
    r = _r()
    if not r: return None
    v = r.hget(_k(job_id, "status"), b"state")
    return v.decode() if v else None

def append_log(job_id: str, line: str) -> None:
    r = _r()
    if not r: return
    r.rpush(_k(job_id, "logs"), line.encode())
    # keep last 2000 lines
    r.ltrim(_k(job_id, "logs"), -2000, -1)
    r.expire(_k(job_id, "logs"), _ttl_seconds())

def get_logs(job_id: str, last_n: int = 200) -> List[str]:
    r = _r()
    if not r: return []
    n = max(1, last_n)
    arr = r.lrange(_k(job_id, "logs"), -n, -1)
    return [x.decode(errors="ignore") for x in arr] if arr else []

def update_progress(job_id: str, epoch: int, total_epochs: int,
                    train_loss: float, val_loss: float) -> None:
    r = _r()
    if not r: return
    now = time.time()
    prog = {
        "epoch": epoch, "total": total_epochs,
        "train": float(train_loss), "val": float(val_loss), "ts": now
    }
    r.hset(_k(job_id, "progress"), mapping={b"json": json.dumps(prog).encode()})
    # append to history
    r.rpush(_k(job_id, "history"), json.dumps(prog).encode())
    r.expire(_k(job_id, "progress"), _ttl_seconds())
    r.expire(_k(job_id, "history"), _ttl_seconds())

def get_progress(job_id: str) -> Optional[Dict[str, Any]]:
    r = _r()
    if not r: return None
    v = r.hget(_k(job_id, "progress"), b"json")
    return json.loads(v.decode()) if v else None

def get_history(job_id: str, last_n: int = 1000) -> List[Dict[str, Any]]:
    r = _r()
    if not r: return []
    arr = r.lrange(_k(job_id, "history"), -last_n, -1)
    return [json.loads(x.decode()) for x in arr] if arr else []

def set_metrics(job_id: str, metrics: Dict[str, Any]) -> None:
    r = _r()
    if not r: return
    r.hset(_k(job_id, "metrics"), mapping={b"json": json.dumps(metrics).encode()})
    r.expire(_k(job_id, "metrics"), _ttl_seconds())

def get_metrics(job_id: str) -> Optional[Dict[str, Any]]:
    r = _r()
    if not r: return None
    v = r.hget(_k(job_id, "metrics"), b"json")
    return json.loads(v.decode()) if v else None

def set_artifact(job_id: str, name: str, data: bytes) -> None:
    r = _r()
    if not r: return
    r.hset(_k(job_id, "artifacts"), name.encode(), data)
    r.expire(_k(job_id, "artifacts"), _ttl_seconds())

def list_artifacts(job_id: str) -> List[str]:
    r = _r()
    if not r: return []
    keys = r.hkeys(_k(job_id, "artifacts"))
    return [k.decode() for k in keys] if keys else []

def get_artifact(job_id: str, name: str) -> Optional[bytes]:
    r = _r()
    if not r: return None
    v = r.hget(_k(job_id, "artifacts"), name.encode())
    return v

def mark_finished(job_id: str, ok: bool, best_val: float) -> None:
    r = _r()
    if not r: return
    r.hset(_k(job_id, "status"), mapping={
        b"state": b"finished" if ok else b"failed",
        b"end_ts": str(time.time()).encode(),
        b"best_val": str(best_val).encode()
    })
    r.sadd(f"{PREFIX}:completed", job_id.encode())
    r.expire(_k(job_id, "status"), _ttl_seconds())

def list_completed() -> List[str]:
    r = _r()
    if not r: return []
    arr = r.smembers(f"{PREFIX}:completed")
    return sorted([a.decode() for a in arr]) if arr else []