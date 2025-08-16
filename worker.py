# worker.py
import os
import json
import logging
from datetime import datetime
import sympy as sp

from rq import get_current_job

from shared.ode_core import (
    ComputeParams,
    compute_ode_full,
    expr_to_str,
)

log = logging.getLogger("worker")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))


# ---------------------------------------------------------------------
# ODE COMPUTE JOB: "worker.compute_job"
# ---------------------------------------------------------------------
def compute_job(payload: dict):
    """
    Background job: build y(x) with Theorem 4.1 and apply LHS to get RHS.
    Payload mirrors ComputeParams (plus simple library hints).
    """
    try:
        # Optional: import libs inside job to avoid heavy imports at worker boot
        basic_lib = None
        special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
        except Exception:
            # libs are optional; compute_ode_full can work with None
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
            constructor_lhs=None,  # worker does not know Streamlit session constructor
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )

        res = compute_ode_full(p)

        # JSON-safe (convert SymPy to strings)
        safe = {
            **res,
            "generator":       expr_to_str(res["generator"]),
            "rhs":             expr_to_str(res["rhs"]),
            "solution":        expr_to_str(res["solution"]),
            "f_expr_preview":  expr_to_str(res.get("f_expr_preview", "")),
            "timestamp":       datetime.now().isoformat(),
        }
        return safe

    except Exception as e:
        # Prefer to fail the job so UI sees status == "failed"
        log.exception("compute_job failed")
        raise


# ---------------------------------------------------------------------
# ML TRAINING JOB: "worker.train_job"
# ---------------------------------------------------------------------
def train_job(payload: dict):
    """
    Background ML training with progress updates.

    Expected payload keys:
      model_type: "pattern_learner" | "vae" | "transformer" (default "pattern_learner")
      learning_rate: float (default 1e-3)
      epochs: int (default 100)
      batch_size: int (default 32)
      samples: int (default 1000)        # synthetic data generator count
      validation_split: float (default .2)
      use_generator: bool (default True) # memory-efficient generator
      enable_mixed_precision: bool (default False)
      checkpoint_dir: str (default $CHECKPOINT_DIR or "checkpoints")
      input_dim, hidden_dim, output_dim: optional ints
      device: "cuda" | "cpu" | None      # None lets trainer pick automatically
    """
    job = get_current_job()

    try:
        # Lazy import so the worker can boot even if torch is heavy
        from src.ml.trainer import MLTrainer
        try:
            import torch
        except Exception:
            torch = None

        # ------------- read config -------------
        model_type = payload.get("model_type", "pattern_learner")
        learning_rate = float(payload.get("learning_rate", 1e-3))
        epochs = int(payload.get("epochs", 100))
        batch_size = int(payload.get("batch_size", 32))
        samples = int(payload.get("samples", 1000))
        validation_split = float(payload.get("validation_split", 0.2))
        use_generator = bool(payload.get("use_generator", True))
        enable_mixed_precision = bool(payload.get("enable_mixed_precision", False))
        checkpoint_dir = payload.get("checkpoint_dir", os.getenv("CHECKPOINT_DIR", "checkpoints"))

        # optional dims
        input_dim = int(payload.get("input_dim", 12))
        hidden_dim = int(payload.get("hidden_dim", 128))
        output_dim = int(payload.get("output_dim", 12))

        # device preference
        device = payload.get("device")
        if device == "cuda":
            # If CUDA requested but not available, fall back to CPU
            if not (torch and torch.cuda.is_available()):
                device = "cpu"

        os.makedirs(checkpoint_dir, exist_ok=True)

        log.info(
            "Starting training: type=%s, epochs=%d, batch_size=%d, samples=%d, device=%s, ckpt=%s",
            model_type, epochs, batch_size, samples, device, checkpoint_dir
        )

        # ------------- create trainer -------------
        trainer = MLTrainer(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=learning_rate,
            device=device,  # MLTrainer will select cuda when None and available
            checkpoint_dir=checkpoint_dir,
            enable_mixed_precision=enable_mixed_precision,
        )

        # ------------- progress callback -------------
        def progress_cb(epoch, total_epochs):
            if job:  # update job meta so Streamlit can poll
                job.meta["progress"] = {
                    "epoch": int(epoch),
                    "total_epochs": int(total_epochs),
                }
                job.save_meta()

        # ------------- train -------------
        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            validation_split=validation_split,
            save_best=True,
            use_generator=use_generator,
            checkpoint_interval=10,
            gradient_accumulation_steps=1,
            progress_callback=progress_cb,
        )

        best_path = os.path.join(checkpoint_dir, f"{model_type}_best.pth")

        # ------------- return JSON-safe summary -------------
        # history is already JSON-friendly (floats & lists)
        result = {
            "ok": True,
            "model_type": model_type,
            "history": trainer.history,
            "best_val_loss": trainer.history.get("best_val_loss"),
            "best_model_path": best_path,
            "device": str(trainer.device),
            "timestamp": datetime.now().isoformat(),
        }
        return result

    except Exception as e:
        log.exception("train_job failed")
        # Let RQ mark the job as failed so UI can display the error clearly
        raise


# ---------------------------------------------------------------------
# Optional: quick health check
# ---------------------------------------------------------------------
def ping(payload=None):
    return {"ok": True, "time": datetime.now().isoformat()}