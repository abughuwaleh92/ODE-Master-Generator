# worker.py
import os
import json
from datetime import datetime
import sympy as sp

# Optional imports of your src libs
try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
except Exception:
    BasicFunctions = SpecialFunctions = None

try:
    from src.ml.trainer import MLTrainer
except Exception:
    MLTrainer = None

from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str
from redis import Redis

# --- small logger that writes plain text lines into Redis list per job
def _get_log_conn() -> Redis:
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or os.getenv("UPSTASH_REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL missing in worker environment.")
    kwargs = {}
    if url.startswith("rediss://") and os.getenv("REDIS_INSECURE_TLS", "false").lower() in ("1","true","yes"):
        kwargs["ssl_cert_reqs"] = None
    return Redis.from_url(url, decode_responses=True, **kwargs)

def _log(job_id: str, line: str):
    try:
        r = _get_log_conn()
        r.rpush(f"job:{job_id}:logs", f"[{datetime.utcnow().isoformat()}Z] {line}")
        # keep up to 5000 lines
        r.ltrim(f"job:{job_id}:logs", -5000, -1)
    except Exception:
        pass

# RQ will call this: "worker.ping_job"
def ping_job(payload: dict) -> dict:
    return {"ok": True, "echo": payload, "ts": datetime.utcnow().isoformat() + "Z"}

# RQ will call this: "worker.compute_job"
def compute_job(payload: dict) -> dict:
    """
    payload should include:
      func_name, alpha, beta, n, M, use_exact, simplify_level,
      lhs_source in {"constructor","freeform","arbitrary"},
      constructor_lhs (stringified SymPy) when lhs_source == "constructor"
      freeform_terms, arbitrary_lhs_text, function_library ("Basic" or "Special")
    """
    job_id = os.getenv("RQ_JOB_ID", "unknown")
    try:
        _log(job_id, f"compute_job: payload={json.dumps(payload)[:500]}")

        # Instantiate libraries (optional)
        basic_lib = BasicFunctions() if BasicFunctions else None
        special_lib = SpecialFunctions() if SpecialFunctions else None

        # Rebuild constructor LHS if provided
        constructor_lhs = None
        if payload.get("lhs_source") == "constructor":
            lhs_text = payload.get("constructor_lhs")
            if lhs_text:
                try:
                    constructor_lhs = sp.sympify(lhs_text)
                    _log(job_id, f"constructor_lhs reconstructed: {str(constructor_lhs)[:120]}")
                except Exception as e:
                    _log(job_id, f"constructor_lhs parse error: {e}")

        p = ComputeParams(
            func_name      = payload.get("func_name", "exp(z)"),
            alpha          = payload.get("alpha", 1),
            beta           = payload.get("beta", 1),
            n              = int(payload.get("n", 1)),
            M              = payload.get("M", 0),
            use_exact      = bool(payload.get("use_exact", True)),
            simplify_level = payload.get("simplify_level","light"),
            lhs_source     = payload.get("lhs_source","constructor"),
            constructor_lhs= constructor_lhs,
            freeform_terms = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library = payload.get("function_library","Basic"),
            basic_lib = basic_lib,
            special_lib = special_lib,
        )

        res = compute_ode_full(p)

        # Make JSON-safe
        safe = {
            **res,
            "generator": expr_to_str(res.get("generator")),
            "rhs":       expr_to_str(res.get("rhs")),
            "solution":  expr_to_str(res.get("solution")),
            "f_expr_preview": expr_to_str(res.get("f_expr_preview")),
            "timestamp": datetime.utcnow().isoformat()+"Z",
        }
        _log(job_id, "compute_job: finished")
        return safe
    except Exception as e:
        _log(job_id, f"compute_job failed: {e}")
        return {"error": str(e)}

# --- Training (kept; only used if you click Train via RQ)
def train_job(payload: dict) -> dict:
    job_id = os.getenv("RQ_JOB_ID", "unknown")
    _log(job_id, f"train_job: start payload={json.dumps(payload)}")

    if MLTrainer is None:
        _log(job_id, "train_job: MLTrainer not importable")
        return {"error": "MLTrainer not available in worker."}

    # Build trainer safely from payload (no 'config' param)
    ctor_kwargs = {
        "model_type": payload.get("model_type", "pattern_learner"),
        "input_dim": payload.get("input_dim", 12),
        "hidden_dim": payload.get("hidden_dim", 128),
        "output_dim": payload.get("output_dim", 12),
        "learning_rate": payload.get("learning_rate", 1e-3),
        "device": payload.get("device"),  # None|cpu|cuda
        "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
        "enable_mixed_precision": bool(payload.get("enable_mixed_precision", False)),
    }

    try:
        trainer = MLTrainer(**ctor_kwargs)
    except TypeError as e:
        _log(job_id, f"train_job: MLTrainer ctor error: {e}")
        return {"error": f"MLTrainer init error: {e}"}

    train_kwargs = {
        "epochs": int(payload.get("epochs", 100)),
        "batch_size": int(payload.get("batch_size", 32)),
        "samples": int(payload.get("samples", 1000)),
        "validation_split": float(payload.get("validation_split", 0.2)),
        "use_generator": bool(payload.get("use_generator", True)),
        "checkpoint_interval": int(payload.get("checkpoint_interval", 10)),
        "gradient_accumulation_steps": int(payload.get("gradient_accumulation_steps", 1)),
        # DO NOT pass 'save_best' if your trainer signature doesn’t accept it
    }

    try:
        trainer.train(**train_kwargs)
        # Save best model path if your trainer does that internally
        artifact = os.path.join(trainer.checkpoint_dir, f"{trainer.model_type}_best.pth")
        _log(job_id, f"train_job: finished, artifact={artifact}")
        return {
            "ok": True,
            "history": trainer.history,
            "artifact": artifact if os.path.exists(artifact) else None,
        }
    except Exception as e:
        _log(job_id, f"train_job failed: {e}")
        return {"error": str(e)}