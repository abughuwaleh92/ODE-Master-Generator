# worker.py
import os
from datetime import datetime

import sympy as sp

from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional ML imports inside train_job to keep cold start small
# from src.ml.trainer import MLTrainer  # imported lazily in train_job

# get_current_job import that works across RQ versions
try:
    from rq import get_current_job  # RQ 1.x/2.x
except Exception:
    try:
        from rq.job import get_current_job  # fallback
    except Exception:
        get_current_job = None


def _update_meta(**fields):
    """Safely update current job's meta."""
    if not get_current_job:
        return
    try:
        job = get_current_job()
        if job:
            m = job.meta or {}
            m.update(fields)
            job.meta = m
            job.save_meta()
    except Exception:
        # Avoid crashing the job solely due to meta updates
        pass


# ---------------------------------------------------------------------
# ODE compute job
# ---------------------------------------------------------------------
def compute_job(payload: dict):
    """
    Compute an ODE using shared.ode_core on the worker.
    The UI polls job.meta (stage, etc.) + job.result for rendering.
    """
    try:
        _update_meta(stage="received")

        # Try loading libraries if available in worker image
        basic_lib = None
        special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions

            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
        except Exception:
            # Keep running, compute_ode_full can handle None libs when using simple functions
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
            constructor_lhs=None,  # session-bound, not accessible here
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )

        _update_meta(stage="computing")
        res = compute_ode_full(p)

        # JSON-safe result (convert sympy to string)
        safe = {
            **res,
            "generator": expr_to_str(res.get("generator")),
            "rhs": expr_to_str(res.get("rhs")),
            "solution": expr_to_str(res.get("solution")),
            "f_expr_preview": expr_to_str(res.get("f_expr_preview")),
            "timestamp": datetime.now().isoformat(),
        }

        _update_meta(stage="finished")
        return safe

    except Exception as e:
        _update_meta(stage="failed", error=str(e))
        # Re-raise so RQ marks the job as failed and captures exc_info
        raise


# ---------------------------------------------------------------------
# Training job with persistent progress
# ---------------------------------------------------------------------
def train_job(payload: dict):
    """
    Train ML model in the background.
    Persist progress in job.meta: stage, epoch, total_epochs, train_loss, val_loss.
    Return a summary dict (history, saved model path hint for the worker container).
    """
    try:
        total_epochs = int(payload.get("epochs", 100))
        _update_meta(stage="starting", epoch=0, total_epochs=total_epochs)

        model_type = payload.get("model_type", "pattern_learner")
        learning_rate = float(payload.get("learning_rate", 1e-3))
        device = payload.get("device", "cpu")
        epochs = total_epochs
        batch_size = int(payload.get("batch_size", 32))
        samples = int(payload.get("samples", 1000))
        validation_split = float(payload.get("validation_split", 0.2))
        use_generator = bool(payload.get("use_generator", True))

        # Local import here to keep worker startup light
        from src.ml.trainer import MLTrainer

        trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
        os.makedirs("checkpoints", exist_ok=True)

        def progress_callback(epoch, total):
            # Called at the end of each epoch in MLTrainer.train(...)
            tr = None
            vl = None
            try:
                tr = float(trainer.history.get("train_loss", [])[-1])
            except Exception:
                pass
            try:
                vl = float(trainer.history.get("val_loss", [])[-1])
            except Exception:
                pass
            _update_meta(
                stage="training",
                epoch=int(epoch),
                total_epochs=int(total),
                train_loss=tr,
                val_loss=vl,
            )

        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            validation_split=validation_split,
            use_generator=use_generator,
            progress_callback=progress_callback,
            save_best=True,
        )

        _update_meta(stage="saving")
        # Ensure we have a saved model on the worker filesystem
        model_path = os.path.join("checkpoints", f"{trainer.model_type}_best.pth")
        try:
            trainer.save_model(model_path)
        except Exception:
            # If already saved inside trainer, ignore
            pass

        _update_meta(stage="finished", epoch=epochs)

        return {
            "trained": True,
            "history": trainer.history,
            "model_path": model_path if os.path.exists(model_path) else None,
            "finished_at": datetime.now().isoformat(),
        }

    except Exception as e:
        _update_meta(stage="failed", error=str(e))
        raise