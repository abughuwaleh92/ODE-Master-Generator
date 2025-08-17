# worker.py
import os, io, json, time, logging, base64
from datetime import datetime

import sympy as sp
from rq import get_current_job

from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str
from shared.training_registry import (
    init_run, set_state, append_log, update_progress,
    set_metrics, set_artifact, mark_finished
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("worker")

# ---------------- Compute job (unchanged behavior) ----------------
def compute_job(payload: dict):
    try:
        basic_lib = None; special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions(); special_lib = SpecialFunctions()
        except Exception:
            pass

        p = ComputeParams(
            func_name      = payload.get("func_name","exp(z)"),
            alpha          = payload.get("alpha",1),
            beta           = payload.get("beta",1),
            n              = int(payload.get("n",1)),
            M              = payload.get("M",0),
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
            "f_expr_preview": expr_to_str(res["f_expr_preview"]),
            "timestamp": datetime.now().isoformat()
        }
        return safe
    except Exception as e:
        return {"error": str(e)}

# ---------------- Training job (NEW) ----------------
def train_job(payload: dict):
    """
    Trains ML model with persistent visibility via Redis and produces artifacts:
      - best model (.pth)
      - full session ZIP (model+opt+history+config)
      - metrics.json
    Optional resume: payload["resume_session_b64"] = base64(zip bytes)
    """
    from src.ml.trainer import MLTrainer

    job = get_current_job()
    job_id = job.get_id() if job else f"local-{int(time.time())}"

    # init persistent run
    init_run(job_id, config=payload)
    append_log(job_id, f"[{datetime.now().isoformat()}] Training started. job_id={job_id}")

    # trainer config
    model_type = payload.get("model_type","pattern_learner")
    learning_rate = float(payload.get("learning_rate", 1e-3))
    input_dim = int(payload.get("input_dim", 12))
    hidden_dim = int(payload.get("hidden_dim", 128))
    output_dim = int(payload.get("output_dim", 12))
    device = payload.get("device","cuda" if torch.cuda.is_available() else "cpu") if payload.get("device") is None else payload.get("device")
    enable_mixed_precision = bool(payload.get("enable_mixed_precision", False))
    beta_kl = float(payload.get("beta_kl", 1.0))
    kl_warmup_epochs = int(payload.get("kl_warmup_epochs", 5))

    trainer = MLTrainer(
        model_type=model_type,
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
        learning_rate=learning_rate, device=device,
        checkpoint_dir=payload.get("checkpoint_dir","checkpoints"),
        enable_mixed_precision=enable_mixed_precision,
        beta_kl=beta_kl, kl_warmup_epochs=kl_warmup_epochs
    )

    # optional knobs
    trainer.normalize = bool(payload.get("normalize", False))
    loss_weights = payload.get("loss_weights")
    if loss_weights:
        w = torch.tensor(loss_weights, dtype=torch.float32).view(1, -1)
        trainer.loss_weights = w.to(trainer.device)

    # optional resume from uploaded session
    try:
        b64 = payload.get("resume_session_b64")
        if b64:
            trainer.load_session_bytes(base64.b64decode(b64))
            append_log(job_id, "Resumed from uploaded session.")
    except Exception as e:
        append_log(job_id, f"Resume failed: {e}")

    # progress hook -> persistent registry
    def on_progress(msg: dict):
        update_progress(job_id,
                        epoch=msg.get("epoch", 0),
                        total_epochs=msg.get("total_epochs", 0),
                        train_loss=msg.get("train_loss", 0.0),
                        val_loss=msg.get("val_loss", 0.0))
        append_log(job_id, f"E{msg.get('epoch')}/{msg.get('total_epochs')} tr={msg.get('train_loss'):.4f} val={msg.get('val_loss'):.4f}")

    # run training
    ok = True
    best_val = None
    try:
        trainer.train(
            epochs=int(payload.get("epochs",100)),
            batch_size=int(payload.get("batch_size",32)),
            samples=int(payload.get("samples",1000)),
            validation_split=float(payload.get("validation_split",0.2)),
            save_best=True,
            use_generator=bool(payload.get("use_generator", True)),
            checkpoint_interval=max(1, int(payload.get("checkpoint_interval", 10))),
            gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 1)),
            progress_callback=on_progress,
            early_stop_patience=int(payload.get("early_stop_patience", 10))
        )
        best_val = float(trainer.history.get("best_val_loss", 0.0))
    except Exception as e:
        ok = False
        append_log(job_id, f"Training failed: {e}")

    # artifacts
    try:
        # Model best checkpoint path (already saved by trainer)
        best_name = f"{model_type}_best.pth"
        best_path = os.path.join(trainer.checkpoint_dir, best_name)
        if os.path.exists(best_path):
            with open(best_path, "rb") as f:
                set_artifact(job_id, best_name, f.read())
        # Full session ZIP
        session_zip = trainer.save_session_bytes()
        set_artifact(job_id, "session.zip", session_zip)
        # Metrics
        metrics = {"history": trainer.history, "best_val_loss": trainer.history.get("best_val_loss")}
        set_metrics(job_id, metrics)
    except Exception as e:
        append_log(job_id, f"Artifact packaging failed: {e}")
        ok = False if best_val is None else ok

    # mark finished
    mark_finished(job_id, ok=ok, best_val=best_val if best_val is not None else -1.0)
    set_state(job_id, "finished" if ok else "failed")
    append_log(job_id, f"[{datetime.now().isoformat()}] Training {'finished' if ok else 'failed'} (best={best_val})")

    return {"ok": ok, "job_id": job_id, "best_val_loss": best_val}