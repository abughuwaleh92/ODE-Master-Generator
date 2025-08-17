# worker.py
"""
RQ worker functions for the Master Generator app.
- compute_job(payload): ODE generation (Master Theorem path)
- train_job(payload): model training with persistent progress
This file can be imported by RQ OR run directly for local tests.
"""

import os
import json
from datetime import datetime
import logging
from typing import Any, Dict, Optional

import sympy as sp
from rq import get_current_job

# Light logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

# Optional GPU off on small hosts
if os.getenv("DISABLE_CUDA", "1") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# --- import shared core (no UI) ---
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# --- optional src imports (safe fallback if absent) ---
try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    HAVE_LIBS = True
except Exception:
    BasicFunctions = SpecialFunctions = None
    HAVE_LIBS = False

# ---------------- ODE compute ----------------
def compute_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background ODE generation.
    `payload` mirrors ComputeParams fields (see app).
    """
    job = get_current_job()
    try:
        if job:
            job.meta.update({"stage": "started", "desc": "ODE compute"})
            job.save_meta()

        basic_lib = BasicFunctions() if HAVE_LIBS else None
        special_lib = SpecialFunctions() if HAVE_LIBS else None

        p = ComputeParams(
            func_name=payload.get("func_name", "exp(z)"),
            alpha=payload.get("alpha", 1),
            beta=payload.get("beta", 1),
            n=int(payload.get("n", 1)),
            M=payload.get("M", 0),
            use_exact=bool(payload.get("use_exact", True)),
            simplify_level=payload.get("simplify_level", "light"),
            lhs_source=payload.get("lhs_source", "constructor"),
            constructor_lhs=None,  # worker doesn't have session constructor
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )

        result = compute_ode_full(p)

        safe = {
            **result,
            "generator": expr_to_str(result["generator"]),
            "rhs":       expr_to_str(result["rhs"]),
            "solution":  expr_to_str(result["solution"]),
            "f_expr_preview": expr_to_str(result.get("f_expr_preview")),
            "timestamp": datetime.now().isoformat(),
        }

        if job:
            job.meta.update({"stage": "finished"})
            job.save_meta()
        return safe

    except Exception as e:
        if job:
            job.meta.update({"stage": "failed"})
            job.save_meta()
        return {"error": str(e)}

# ---------------- Training ----------------
def train_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background model training with persistent progress saved in job.meta.
    payload:
      model_type, epochs, batch_size, samples, validation_split, learning_rate, device, use_generator
    """
    job = get_current_job()
    try:
        if job:
            job.meta.update({
                "stage": "started",
                "desc": "training",
                "epoch": 0,
                "train_loss": None,
                "val_loss": None,
            })
            job.save_meta()

        # Lazy import to keep worker light
        from src.ml.trainer import MLTrainer

        trainer = MLTrainer(
            model_type=payload.get("model_type", "pattern_learner"),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            device=payload.get("device", "cpu"),
            enable_mixed_precision=bool(payload.get("mixed_precision", False)),
        )

        epochs = int(payload.get("epochs", 100))
        batch_size = int(payload.get("batch_size", 32))
        samples = int(payload.get("samples", 1000))
        validation_split = float(payload.get("validation_split", 0.2))
        use_generator = bool(payload.get("use_generator", True))

        # Progress callback updates job.meta every epoch
        def on_progress(epoch: int, total: int, train_loss: Optional[float]=None, val_loss: Optional[float]=None):
            if not get_current_job():
                return
            j = get_current_job()
            j.meta.update({
                "stage": "running",
                "epoch": epoch,
                "total_epochs": total,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
            j.save_meta()

        # Patch trainer to call our callback with losses
        _orig_log = trainer.logger.info if hasattr(trainer, "logger") else None

        def progress_proxy(e, t):
            # find latest losses if available
            tl = trainer.history["train_loss"][-1] if trainer.history["train_loss"] else None
            vl = trainer.history["val_loss"][-1] if trainer.history["val_loss"] else None
            on_progress(e, t, tl, vl)

        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            validation_split=validation_split,
            use_generator=use_generator,
            progress_callback=progress_proxy,
            save_best=True,
        )

        # Best checkpoint path (as trainer saved it)
        ckpt_dir = getattr(trainer, "checkpoint_dir", "checkpoints")
        best_name = f"{trainer.model_type}_best.pth"
        model_path = os.path.join(ckpt_dir, best_name)

        result = {
            "ok": True,
            "trained": True,
            "epochs": trainer.history.get("epochs", epochs),
            "best_val_loss": trainer.history.get("best_val_loss"),
            "model_path": model_path,
            "history": trainer.history,
            "timestamp": datetime.now().isoformat(),
        }

        if job:
            job.meta.update({"stage": "finished"})
            job.save_meta()

        return result

    except Exception as e:
        if job:
            job.meta.update({"stage": "failed", "error": str(e)})
            job.save_meta()
        return {"ok": False, "error": str(e)}

# ---- optional: run a worker directly (local dev) ----
if __name__ == "__main__":
    # Example: python worker.py (starts an RQ worker)
    import redis
    from rq import Worker, Queue, Connection

    REDIS_URL = os.getenv("REDIS_URL")
    RQ_QUEUES = [q.strip() for q in os.getenv("RQ_QUEUES", os.getenv("RQ_QUEUE", "ode_jobs")).split(",") if q.strip()]
    if not REDIS_URL:
        raise SystemExit("Set REDIS_URL first")
    conn = redis.from_url(REDIS_URL, decode_responses=True)
    with Connection(conn):
        qs = [Queue(name) for name in RQ_QUEUES]
        log.info(f"Starting worker on queues: {', '.join([q.name for q in qs])}")
        w = Worker(qs)
        w.work(with_scheduler=True)