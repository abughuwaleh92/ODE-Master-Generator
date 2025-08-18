# worker.py
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import sympy as sp
from rq import get_current_job

from rq_utils import push_log, set_progress, set_artifacts
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional: ML imports inside worker to keep Streamlit light
from src.ml.trainer import MLTrainer, TrainConfig

LOG_DIR = os.getenv("LOG_DIR", "logs")
ART_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
CKPT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ------------- helpers -------------
def _job_log(job_id: str, msg: str):
    line = f"[{datetime.now().isoformat()}] {msg}"
    push_log(job_id, line)

def _job_progress(job_id: str, payload: Dict[str, Any]):
    set_progress(job_id, payload)

def _attach_hooks(trainer: MLTrainer, job_id: str):
    def log_hook(msg: str):
        _job_log(job_id, msg)
    def progress_hook(p: Dict[str, Any]):
        # mirror into job.meta too (survives for result_ttl)
        job = get_current_job()
        if job is not None:
            job.meta.update({"progress": p})
            job.save_meta()
        _job_progress(job_id, p)
    trainer.log_hook = log_hook
    trainer.progress_hook = progress_hook

# ======================
# 1) ODE compute job
# ======================
def compute_job(payload: dict):
    """
    Same payload you use already from Streamlit (Apply Master Theorem).
    """
    try:
        # Optional src libs:
        basic_lib = special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
        except Exception:
            pass

        p = ComputeParams(
            func_name      = payload.get("func_name", "exp(z)"),
            alpha          = payload.get("alpha", 1),
            beta           = payload.get("beta", 1),
            n              = int(payload.get("n", 1)),
            M              = payload.get("M", 0),
            use_exact      = bool(payload.get("use_exact", True)),
            simplify_level = payload.get("simplify_level","light"),
            lhs_source     = payload.get("lhs_source","constructor"),
            constructor_lhs= None,
            freeform_terms = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library = payload.get("function_library","Basic"),
            basic_lib = basic_lib,
            special_lib = special_lib,
        )
        res = compute_ode_full(p)
        safe = {
            **res,
            "generator": expr_to_str(res["generator"]),
            "rhs":       expr_to_str(res["rhs"]),
            "solution":  expr_to_str(res["solution"]),
            "f_expr_preview": expr_to_str(res.get("f_expr_preview", "")),
            "timestamp": datetime.now().isoformat()
        }
        return safe
    except Exception as e:
        return {"error": str(e)}

# ======================
# 2) Training job
# ======================
def train_job(payload: dict):
    """
    Long-running ML training with persistent progress + logs + artifacts.
    Expects payload fields (all optional defaults provided):
      - model_type, input_dim, hidden_dim, output_dim, learning_rate,
        enable_mixed_precision, normalize, beta_vae, kl_anneal, kl_max_beta, kl_warmup_epochs,
        early_stop_patience, loss_weights
      - epochs, batch_size, samples, validation_split, use_generator, resume_from
    """
    job = get_current_job()
    job_id = job.id if job else f"nojob-{datetime.now().timestamp()}"

    # where to store artifacts for this job
    job_art_dir = os.path.join(ART_DIR, f"train_{job_id}")
    os.makedirs(job_art_dir, exist_ok=True)

    # configure trainer
    cfg = TrainConfig(
        model_type = payload.get("model_type", "pattern_learner"),
        input_dim  = int(payload.get("input_dim", 12)),
        hidden_dim = int(payload.get("hidden_dim", 128)),
        output_dim = int(payload.get("output_dim", 12)),
        learning_rate = float(payload.get("learning_rate", 1e-3)),
        enable_mixed_precision = bool(payload.get("enable_mixed_precision", False)),
        normalize = bool(payload.get("normalize", False)),
        beta_vae  = float(payload.get("beta_vae", 1.0)),
        kl_anneal = payload.get("kl_anneal", "linear"),
        kl_max_beta = float(payload.get("kl_max_beta", 1.0)),
        kl_warmup_epochs = int(payload.get("kl_warmup_epochs", 10)),
        early_stop_patience = int(payload.get("early_stop_patience", 12)),
        loss_weights = payload.get("loss_weights"),
        device = payload.get("device", None) or ("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir = CKPT_DIR,
    )
    trainer = MLTrainer(cfg)
    _attach_hooks(trainer, job_id)

    # save initial meta
    if job:
        job.meta["status"] = "running"
        job.meta["config"] = cfg.__dict__
        job.save_meta()

    # training args
    epochs = int(payload.get("epochs", 100))
    batch_size = int(payload.get("batch_size", 32))
    samples = int(payload.get("samples", 1000))
    validation_split = float(payload.get("validation_split", 0.2))
    use_generator = bool(payload.get("use_generator", True))
    resume_from = payload.get("resume_from")

    # run training
    try:
        _job_log(job_id, f"Starting training: {cfg.model_type}, epochs={epochs}, batch={batch_size}, samples={samples}")
        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            validation_split=validation_split,
            use_generator=use_generator,
            resume_from=resume_from,
            checkpoint_interval=max(1, epochs // 5),
        )

        # persist artifacts
        best_path = trainer.history.get("best_model_path")
        trainer.save_session_dir(job_art_dir, best_path=best_path)
        zip_path = os.path.join(job_art_dir, "session_artifacts.zip")
        trainer.export_session_zip(zip_path, best_path=best_path)

        set_artifacts(job_id, {
            "session_dir": job_art_dir,
            "session_zip": zip_path,
            "best_model": best_path or "",
        })

        # mark finished
        if job:
            job.meta["status"] = "finished"
            job.meta["trained"] = True
            job.meta["history"] = trainer.history
            job.save_meta()

        _job_log(job_id, "Training finished successfully.")
        return {
            "ok": True,
            "best_model": best_path,
            "session_zip": zip_path,
            "history": trainer.history,
        }
    except Exception as e:
        if job:
            job.meta["status"] = "failed"
            job.meta["error"] = str(e)
            job.save_meta()
        _job_log(job_id, f"Training failed: {e}")
        return {"ok": False, "error": str(e)}