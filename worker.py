# worker.py
import os
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional

import sympy as sp
from rq import get_current_job

from rq_utils import append_training_log
from shared.ode_core import (
    ComputeParams, compute_ode_full, expr_to_str
)

# Optional torch imports guarded in train path
ML_IMPORT_ERROR = None
try:
    import torch
except Exception as _e:
    ML_IMPORT_ERROR = _e
    torch = None

logger = logging.getLogger("worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- Utility ----------------
def _to_json_safe(res: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(res)
    # SymPy -> strings where needed
    for k in ("generator", "rhs", "solution", "f_expr_preview"):
        if k in out:
            out[k] = expr_to_str(out[k])
    out["timestamp"] = datetime.utcnow().isoformat()
    return out

def _update_job_meta(meta: Dict[str, Any]):
    job = get_current_job()
    if not job:
        return
    job.meta.update(meta or {})
    try:
        job.save_meta()
    except Exception:
        # older RQ
        try:
            job.save()
        except Exception:
            pass

def _log(job_id: Optional[str], msg: Dict[str, Any]):
    if job_id:
        append_training_log(job_id, msg)

# ---------------- Compute job ----------------
def compute_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute ODE in worker. Payload mirrors ComputeParams fields used by shared.ode_core.
    """
    job = get_current_job()
    job_id = job.id if job else None
    _update_job_meta({"status": "computing"})
    try:
        # Optional library load for worker-side function resolution
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
            constructor_lhs=None,  # unknown in worker session
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )
        res = compute_ode_full(p)
        out = _to_json_safe(res)
        _update_job_meta({"status": "finished", "ok": True})
        return out
    except Exception as e:
        _update_job_meta({"status": "failed", "ok": False, "error": str(e)})
        return {"error": str(e)}

# ---------------- Training job ----------------
def train_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train ML model in background with persistent progress + checkpoints.
    payload:
      model_type: "pattern_learner" | "vae" | "transformer"
      epochs, batch_size, samples, validation_split, learning_rate
      device: "cuda"|"cpu"|None
      resume_from (optional): checkpoint path to resume
    """
    job = get_current_job()
    job_id = job.id if job else None

    # Log start
    _update_job_meta({"status": "starting", "kind": "train"})
    _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "train_start", "payload": payload})

    if ML_IMPORT_ERROR:
        msg = f"PyTorch unavailable in worker: {ML_IMPORT_ERROR}"
        _update_job_meta({"status": "failed", "error": msg})
        _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "error", "message": msg})
        return {"error": msg}

    # Lazy import trainer to avoid import cost for compute jobs
    try:
        from src.ml.trainer import MLTrainer
    except Exception as e:
        _update_job_meta({"status": "failed", "error": f"Import MLTrainer failed: {e}"})
        _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "error", "message": f"Import MLTrainer failed: {e}"})
        return {"error": f"Import MLTrainer failed: {e}"}

    # Extract params with defaults
    model_type       = payload.get("model_type", "pattern_learner")
    epochs           = int(payload.get("epochs", 100))
    batch_size       = int(payload.get("batch_size", 32))
    samples          = int(payload.get("samples", 1000))
    validation_split = float(payload.get("validation_split", 0.2))
    learning_rate    = float(payload.get("learning_rate", 1e-3))
    enable_amp       = bool(payload.get("enable_mixed_precision", False))
    device           = payload.get("device")
    checkpoint_dir   = payload.get("checkpoint_dir", CHECKPOINT_DIR)
    resume_from      = payload.get("resume_from")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Select device
    if not device:
        device = "cuda" if (torch.cuda.is_available()) else "cpu"

    trainer = MLTrainer(
        model_type=model_type,
        learning_rate=learning_rate,
        device=device,
        enable_mixed_precision=enable_amp,
        checkpoint_dir=checkpoint_dir,
    )

    # Resume if a checkpoint is provided
    if resume_from:
        try:
            last_epoch = trainer.load_checkpoint(resume_from)
            _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "resume", "checkpoint": resume_from, "epoch": last_epoch})
        except Exception as e:
            _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "resume_failed", "error": str(e)})

    # Progress callback
    def on_progress(ep: int, total: int):
        hist = getattr(trainer, "history", {})
        train_loss = (hist.get("train_loss") or [None])[-1]
        val_loss   = (hist.get("val_loss") or [None])[-1]
        pct = float(ep) / float(total)
        _update_job_meta({
            "status": "training",
            "epoch": ep,
            "epochs": total,
            "progress": pct,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        _log(job_id, {
            "ts": datetime.utcnow().isoformat(),
            "event": "epoch",
            "epoch": ep,
            "epochs": total,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    # Run training
    try:
        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            validation_split=validation_split,
            use_generator=True,
            save_best=True,
            checkpoint_interval=10,
            progress_callback=on_progress
        )
    except Exception as e:
        _update_job_meta({"status": "failed", "error": str(e)})
        _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "error", "message": str(e)})
        return {"error": str(e)}

    # Save best model & history artifacts
    best_name = f"{model_type}_best.pth"
    best_path = os.path.join(checkpoint_dir, best_name)
    try:
        # trainer already saved best; ensure it exists
        if not os.path.exists(best_path):
            trainer.save_model(best_path)
    except Exception:
        pass

    hist_path = os.path.join(checkpoint_dir, f"history_{job_id}.json")
    try:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(getattr(trainer, "history", {}), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    result = {
        "ok": True,
        "status": "finished",
        "job_id": job_id,
        "model_type": model_type,
        "checkpoint_dir": checkpoint_dir,
        "best_model_path": best_path if os.path.exists(best_path) else None,
        "history_path": hist_path if os.path.exists(hist_path) else None,
        "history": getattr(trainer, "history", {}),
        "device": device
    }
    _update_job_meta({"status": "finished", "ok": True, "best_model_path": result["best_model_path"]})
    _log(job_id, {"ts": datetime.utcnow().isoformat(), "event": "train_done", "artifacts": {"best": result["best_model_path"], "history": result["history_path"]}})
    return result