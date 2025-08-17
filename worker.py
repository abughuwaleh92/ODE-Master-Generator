# worker.py
import os
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional

import sympy as sp
from rq import get_current_job

from rq_utils import append_training_log
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional torch lazy import
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

def _to_json_safe(res: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(res)
    for k in ("generator", "rhs", "solution", "f_expr_preview"):
        if k in out:
            try:
                out[k] = expr_to_str(out[k])
            except Exception:
                out[k] = str(out[k])
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
        try:
            job.save()
        except Exception:
            pass

def _log(event: Dict[str, Any]):
    job = get_current_job()
    if not job:
        return
    append_training_log(job.id, {"ts": datetime.utcnow().isoformat(), **event})

# ---------------- Compute job ----------------
def compute_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    job = get_current_job()
    origin = getattr(job, "origin", None) if job else None
    _update_job_meta({"status": "computing", "origin": origin})
    try:
        # Optional function libs
        basic_lib = special_lib = None
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
            constructor_lhs=None,
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
    job = get_current_job()
    origin = getattr(job, "origin", None) if job else None
    _update_job_meta({"status": "starting", "kind": "train", "origin": origin})
    _log({"event": "train_start", "payload": payload, "origin": origin})

    if ML_IMPORT_ERROR:
        msg = f"PyTorch unavailable in worker: {ML_IMPORT_ERROR}"
        _update_job_meta({"status": "failed", "error": msg})
        _log({"event": "error", "message": msg})
        return {"error": msg}

    try:
        from src.ml.trainer import MLTrainer
    except Exception as e:
        _update_job_meta({"status": "failed", "error": f"Import MLTrainer failed: {e}"})
        _log({"event": "error", "message": f"Import MLTrainer failed: {e}"})
        return {"error": f"Import MLTrainer failed: {e}"}

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
    if not device:
        device = "cuda" if (torch.cuda.is_available()) else "cpu"

    trainer = MLTrainer(
        model_type=model_type,
        learning_rate=learning_rate,
        device=device,
        enable_mixed_precision=enable_amp,
        checkpoint_dir=checkpoint_dir,
    )

    if resume_from:
        try:
            ep = trainer.load_checkpoint(resume_from)
            _log({"event": "resume", "checkpoint": resume_from, "epoch": ep})
        except Exception as e:
            _log({"event": "resume_failed", "error": str(e)})

    def on_progress(ep: int, total: int):
        hist = getattr(trainer, "history", {})
        tr = (hist.get("train_loss") or [None])[-1]
        vl = (hist.get("val_loss") or [None])[-1]
        _update_job_meta({
            "status": "training",
            "epoch": ep, "epochs": total,
            "progress": float(ep)/float(total),
            "train_loss": tr, "val_loss": vl,
            "origin": origin
        })
        _log({"event": "epoch", "epoch": ep, "epochs": total, "train_loss": tr, "val_loss": vl})

    try:
        trainer.train(
            epochs=epochs, batch_size=batch_size, samples=samples,
            validation_split=validation_split, use_generator=True,
            save_best=True, checkpoint_interval=10, progress_callback=on_progress
        )
    except Exception as e:
        _update_job_meta({"status": "failed", "error": str(e)})
        _log({"event": "error", "message": str(e)})
        return {"error": str(e)}

    best = os.path.join(checkpoint_dir, f"{model_type}_best.pth")
    try:
        if not os.path.exists(best):
            trainer.save_model(best)
    except Exception:
        pass

    hist_path = os.path.join(checkpoint_dir, f"history_{job.id if job else 'local'}.json")
    try:
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(getattr(trainer, "history", {}), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    result = {
        "ok": True, "status": "finished",
        "job_id": job.id if job else None,
        "model_type": model_type,
        "checkpoint_dir": checkpoint_dir,
        "best_model_path": best if os.path.exists(best) else None,
        "history_path": hist_path if os.path.exists(hist_path) else None,
        "history": getattr(trainer, "history", {}),
        "device": device, "origin": origin
    }
    _update_job_meta({"status": "finished", "ok": True, "best_model_path": result["best_model_path"], "origin": origin})
    _log({"event": "train_done", "artifacts": {"best": result["best_model_path"], "history": result["history_path"]}})
    return result