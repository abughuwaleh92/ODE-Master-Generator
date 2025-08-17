# worker.py
import os, json, time
from datetime import datetime
from typing import Dict, Any, Optional

import sympy as sp
from rq import get_current_job

from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str
from shared.reverse_engineering import reverse_engineer
from rq_utils import update_run

# ---------------- compute job (unchanged behavior) ----------------
def compute_job(payload: Dict[str, Any]) -> Dict[str, Any]:
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
            func_name      = payload.get("func_name", "exp(z)"),
            alpha          = payload.get("alpha", 1),
            beta           = payload.get("beta", 1),
            n              = int(payload.get("n", 1)),
            M              = payload.get("M", 0),
            use_exact      = bool(payload.get("use_exact", True)),
            simplify_level = payload.get("simplify_level", "light"),
            lhs_source     = payload.get("lhs_source", "constructor"),
            constructor_lhs= None,  # worker doesn't own constructor session
            freeform_terms = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library = payload.get("function_library", "Basic"),
            basic_lib = basic_lib,
            special_lib = special_lib,
        )
        res = compute_ode_full(p)
        safe = {
            **res,
            "generator": expr_to_str(res["generator"]),
            "rhs":       expr_to_str(res["rhs"]),
            "solution":  expr_to_str(res["solution"]),
            "f_expr_preview": expr_to_str(res["f_expr_preview"]),
            "timestamp": datetime.now().isoformat()
        }
        return safe
    except Exception as e:
        return {"error": str(e)}

# ---------------- training job with robust progress ----------------
def train_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heavy training in worker with periodic progress pushes to job.meta and run registry.
    """
    job = get_current_job()
    job.meta["state"] = "starting"
    job.meta["progress"] = {"epoch": 0, "total": payload.get("epochs", 100)}
    job.save_meta()

    # create run directory
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"run_{ts}_{job.id[:8]}")
    os.makedirs(run_dir, exist_ok=True)

    # import here (fast path in worker)
    from src.ml.trainer import MLTrainer

    # config
    model_type = payload.get("model_type", "pattern_learner")
    device = payload.get("device") or ("cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu")
    lr = float(payload.get("learning_rate", 1e-3))
    epochs = int(payload.get("epochs", 100))
    batch_size = int(payload.get("batch_size", 32))
    samples = int(payload.get("samples", 1000))
    val_split = float(payload.get("validation_split", 0.2))
    use_generator = bool(payload.get("use_generator", True))
    hidden_dim = int(payload.get("hidden_dim", 128))
    input_dim = int(payload.get("input_dim", 12))
    output_dim = int(payload.get("output_dim", 12))
    early_stop_patience = int(payload.get("early_stop_patience", 10))
    enable_amp = bool(payload.get("enable_mixed_precision", False))
    normalize = bool(payload.get("normalize", False))
    loss_weights = payload.get("loss_weights")  # list of floats or None

    # setup trainer
    trainer = MLTrainer(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        learning_rate=lr,
        device=device,
        checkpoint_dir=run_dir,
        enable_mixed_precision=enable_amp
    )
    trainer.normalize = normalize
    if loss_weights:
        import torch
        lw = torch.tensor(loss_weights, dtype=torch.float32).view(1, -1).to(trainer.device)
        trainer.loss_weights = lw

    # progress callback
    def progress_cb(epoch: int, total: int, stats: Optional[Dict[str, float]] = None):
        job.meta["state"] = "running"
        job.meta["progress"] = {"epoch": epoch, "total": total}
        if stats:
            job.meta["stats"] = stats
        job.save_meta()

    # train
    started = time.time()
    trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        samples=samples,
        validation_split=val_split,
        save_best=True,
        use_generator=use_generator,
        checkpoint_interval=max(epochs // 5, 1),
        gradient_accumulation_steps=1,
        progress_callback=progress_cb,
        early_stop_patience=early_stop_patience
    )
    dur = time.time() - started

    # save artifacts (history.json, config.json) and return summary
    artifacts = trainer.save_artifacts(run_dir)
    best_model = os.path.join(run_dir, f"{model_type}_best.pth")
    exists = os.path.exists(best_model)

    summary = {
        "job_id": job.id,
        "run_dir": run_dir,
        "best_model_path": best_model if exists else None,
        "history_path": artifacts.get("history_path"),
        "config_path": artifacts.get("config_path"),
        "epochs_done": trainer.history.get("epochs", epochs),
        "best_val": trainer.history.get("best_val_loss", None),
        "duration_sec": dur,
        "model_type": model_type
    }

    # mark finished
    job.meta["state"] = "finished"
    job.meta["summary"] = summary
    job.save_meta()

    # persist in run registry too
    update_run(job.id, {
        "status": "finished",
        "finished_at": time.time(),
        "summary": summary
    })

    return summary

# ---------------- reverse engineering job ----------------
def reverse_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      - {"ode_text": "..."}  (LaTeX-like or SymPy-parseable string)
      - {"ode_lhs": "...", "ode_rhs": "..."}
      - or {"samples": {"x": [...], "y": [...]}, "template": {...}} for numeric fits.
    Returns predicted parameters, function name, type, order, and a reconstructed ODE preview.
    """
    job = get_current_job()
    if job:
        job.meta["state"] = "running"
        job.save_meta()
    try:
        result = reverse_engineer(payload)
        if job:
            job.meta["state"] = "finished"
            job.meta["summary"] = result
            job.save_meta()
        return result
    except Exception as e:
        if job:
            job.meta["state"] = "failed"
            job.meta["error"] = str(e)
            job.save_meta()
        return {"error": str(e)}