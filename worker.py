# worker.py
"""
RQ worker entry points.
- compute_job(payload): builds ODE via shared.ode_core and returns JSON-safe result.
- train_job(payload): trains ML model with persistent progress saved in job.meta.
Designed for RQ >= 2, Redis URL from env: REDIS_URL (or RQ_REDIS_URL).
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

# RQ >= 2
from rq.job import get_current_job

# ---- Optional heavy deps guarded ----
import sympy as sp

# core compute helpers (must exist in your repo)
from shared.ode_core import (
    ComputeParams,
    compute_ode_full,
    expr_to_str,   # stringifier safe for SymPy; falls back to str() if needed
)

# -------------------- Helpers --------------------

def _to_safe(obj: Any) -> Any:
    """SymPy/NumPy -> JSON safe."""
    try:
        return expr_to_str(obj)
    except Exception:
        try:
            if isinstance(obj, (list, tuple)):
                return [_to_safe(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_safe(v) for k, v in obj.items()}
            return str(obj)
        except Exception:
            return str(obj)

def _update_meta(stage: str, **kwargs):
    job = get_current_job()
    if not job:
        return
    job.meta = job.meta or {}
    job.meta["stage"] = stage
    job.meta["heartbeat"] = datetime.utcnow().isoformat()
    for k, v in kwargs.items():
        job.meta[k] = v
    job.save_meta()

# -------------------- ODE compute --------------------

def compute_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create ODE from Theorem page request.
    Payload fields mirror ComputeParams plus library hints.
    """
    _update_meta("received", desc="ODE compute")
    try:
        # Try to instantiate libraries here, but compute_ode_full can work with basic names
        basic_lib = None
        special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
        except Exception:
            pass

        p = ComputeParams(
            func_name=payload.get("func_name", "exp(z)"),
            alpha=payload.get("alpha", 1),
            beta=payload.get("beta", 1),
            n=int(payload.get("n", 1)),
            M=payload.get("M", 0),
            use_exact=bool(payload.get("use_exact", True)),
            simplify_level=payload.get("simplify_level", "light"),
            lhs_source=payload.get("lhs_source", "constructor"),
            constructor_lhs=None,  # constructor expression is session-bound; worker uses freeform/arbitrary text if given
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )

        _update_meta("computing")
        res = compute_ode_full(p)

        safe = {k: _to_safe(v) for k, v in res.items()}
        safe["timestamp"] = datetime.utcnow().isoformat()

        _update_meta("finished")
        return safe

    except Exception as e:
        _update_meta("failed", error=str(e))
        return {"error": str(e), "stage": "failed"}

# -------------------- Training job --------------------

def train_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train ML/DL model with persistent progress.
    payload = {model_type, epochs, batch_size, samples, validation_split, use_generator, learning_rate, device, mixed_precision}
    """
    _update_meta("received", desc="ML training", epoch=0, total_epochs=payload.get("epochs", 100))

    # Import lazily to keep worker import cost low
    from src.ml.trainer import MLTrainer

    model_type = payload.get("model_type", "pattern_learner")
    epochs = int(payload.get("epochs", 100))
    batch_size = int(payload.get("batch_size", 32))
    samples = int(payload.get("samples", 1000))
    validation_split = float(payload.get("validation_split", 0.2))
    use_generator = bool(payload.get("use_generator", True))
    lr = float(payload.get("learning_rate", 1e-3))
    device = payload.get("device") or ("cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu")
    mp = bool(payload.get("mixed_precision", False))

    # Make sure checkpoint dir exists on worker FS
    ckpt_dir = payload.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build trainer
    trainer = MLTrainer(
        model_type=model_type,
        learning_rate=lr,
        device=device,
        checkpoint_dir=ckpt_dir,
        enable_mixed_precision=mp,
    )

    # callback: persist epoch + losses in job.meta
    def on_progress(ep: int, total: int):
        hist = getattr(trainer, "history", {})
        tr = hist.get("train_loss", [])
        vl = hist.get("val_loss", [])
        train_loss = float(tr[-1]) if tr else None
        val_loss = float(vl[-1]) if vl else None
        _update_meta(
            "training",
            epoch=int(ep),
            total_epochs=int(total),
            train_loss=train_loss,
            val_loss=val_loss,
        )

    # Train
    _update_meta("training", epoch=0, total_epochs=epochs)
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        samples=samples,
        validation_split=validation_split,
        use_generator=use_generator,
        save_best=True,
        progress_callback=on_progress,
    )

    # Locate best checkpoint
    best_path = os.path.join(ckpt_dir, f"{model_type}_best.pth")
    info = {
        "model_type": model_type,
        "device": device,
        "epochs": epochs,
        "checkpoint": best_path if os.path.exists(best_path) else None,
        "history": getattr(trainer, "history", {}),
        "saved_at": datetime.utcnow().isoformat(),
    }

    _update_meta("finished", trained=True, checkpoint=info["checkpoint"])
    return info