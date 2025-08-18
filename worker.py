# worker.py
import os
from datetime import datetime
import sympy as sp

from rq import get_current_job

from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# ---------- Compute job (already in your setup) ----------
def compute_job(payload: dict):
    """
    RQ job entry to compute an ODE with shared.ode_core (Theorem engine).
    Worker loads function libraries if available.
    """
    try:
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
            constructor_lhs=None,  # worker cannot access Streamlit session object
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )
        res = compute_ode_full(p)

        safe = {
            **res,
            "generator": expr_to_str(res["generator"]),
            "rhs": expr_to_str(res["rhs"]),
            "solution": expr_to_str(res["solution"]),
            "f_expr_preview": expr_to_str(res.get("f_expr_preview")),
            "timestamp": datetime.now().isoformat(),
        }
        return safe
    except Exception as e:
        return {"error": str(e)}


# ---------- Training job (new) ----------
def train_job(payload: dict):
    """
    RQ job entry to train ML model. Uses MLTrainer (fixed collate_fn).
    Updates job.meta['progress'] per epoch for UI polling.
    """
    from src.ml.trainer import MLTrainer  # import here so worker picks the updated file

    # Read payload
    model_type = payload.get("model_type", "pattern_learner")
    learning_rate = float(payload.get("learning_rate", 1e-3))
    epochs = int(payload.get("epochs", 100))
    batch_size = int(payload.get("batch_size", 32))
    samples = int(payload.get("samples", 1000))
    validation_split = float(payload.get("validation_split", 0.2))
    use_generator = bool(payload.get("use_generator", True))
    enable_mixed_precision = bool(payload.get("enable_mixed_precision", False))
    checkpoint_dir = payload.get("checkpoint_dir", os.getenv("CHECKPOINT_DIR", "checkpoints"))
    device = payload.get("device")  # "cuda" / "cpu" or None

    os.makedirs(checkpoint_dir, exist_ok=True)

    job = get_current_job()
    def cb(epoch: int, total: int):
        if job:
            job.meta["progress"] = {"epoch": int(epoch), "total_epochs": int(total)}
            job.save_meta()

    trainer = MLTrainer(
        model_type=model_type,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir,
        enable_mixed_precision=enable_mixed_precision,
    )

    # Run training
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        samples=samples,
        validation_split=validation_split,
        use_generator=use_generator,
        progress_callback=cb,
        save_best=True,
        checkpoint_interval=max(epochs // 5, 1),
    )

    best_model_path = os.path.join(checkpoint_dir, f"{model_type}_best.pth")
    result = {
        "history": trainer.history,
        "best_model_path": best_model_path if os.path.exists(best_model_path) else None,
        "finished_at": datetime.now().isoformat(),
    }
    return result