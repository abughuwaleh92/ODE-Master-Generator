# worker.py# worker.py
import os
import json
import zipfile
from datetime import datetime

import sympy as sp
from rq import get_current_job

# Core
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional trainer (for train_job)
try:
    from src.ml.trainer import MLTrainer, TrainConfig
except Exception:
    MLTrainer = None
    TrainConfig = None

# Redis helper (for logs)
try:
    import redis
except Exception:
    redis = None

def _redis():
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or os.getenv("UPSTASH_REDIS_URL")
    if not url or not redis:
        return None
    # For direct log strings, decode_responses=True is OK; RQ uses its own connection for pickles.
    return redis.from_url(url, decode_responses=True)

def _log(job, msg: str):
    try:
        conn = _redis()
        if not conn or not job:
            return
        key = f"job:{job.id}:logs"
        conn.rpush(key, f"[{datetime.utcnow().isoformat()}Z] {msg}")
        # keep list bounded
        conn.ltrim(key, -5000, -1)
    except Exception:
        pass

# ----------------- Compute (Generate ODE) -----------------
def compute_job(payload: dict):
    """
    Called via RQ as "worker.compute_job".
    - Does not rely on session-only objects (constructor_lhs is None on worker).
    - Converts SymPy expressions to JSON-safe strings.
    """
    job = get_current_job()
    try:
        _log(job, f"compute_job: start payload={payload}")

        # Try to load function libraries (optional)
        basic_lib = None
        special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
            _log(job, "Loaded BasicFunctions & SpecialFunctions.")
        except Exception as e:
            _log(job, f"Library load skipped: {e}")

        p = ComputeParams(
            func_name       = payload.get("func_name", "exp(z)"),
            alpha           = payload.get("alpha", 1),
            beta            = payload.get("beta", 1),
            n               = int(payload.get("n", 1)),
            M               = payload.get("M", 0),
            use_exact       = bool(payload.get("use_exact", True)),
            simplify_level  = payload.get("simplify_level", "light"),
            lhs_source      = payload.get("lhs_source", "constructor"),
            constructor_lhs = None,  # worker has no access to UI constructor session
            freeform_terms  = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library= payload.get("function_library", "Basic"),
            basic_lib       = basic_lib,
            special_lib     = special_lib,
        )

        res = compute_ode_full(p)

        safe = {
            **res,
            "generator":       expr_to_str(res.get("generator")),
            "rhs":             expr_to_str(res.get("rhs")),
            "solution":        expr_to_str(res.get("solution")),
            "f_expr_preview":  expr_to_str(res.get("f_expr_preview")),
            "timestamp":       datetime.utcnow().isoformat() + "Z"
        }

        # Update meta so UI can show terminal state even if result not pulled yet
        job.meta["progress"] = {"stage": "finished"}
        job.meta["artifacts"] = {}
        job.save_meta()

        _log(job, "compute_job: finished.")
        return safe

    except Exception as e:
        if job:
            job.meta["progress"] = {"stage": "failed", "error": str(e)}
            job.save_meta()
        _log(job, f"compute_job: failed: {e}")
        raise


# ----------------- Train (Persistent RQ) -----------------
def train_job(payload: dict):
    """
    Background training with persistent progress/logging/artifacts.
    Expects MLTrainer + TrainConfig to be available in src/ml/trainer.py
    """
    job = get_current_job()
    if MLTrainer is None or TrainConfig is None:
        _log(job, "train_job: Trainer not available.")
        raise RuntimeError("MLTrainer/TrainConfig not available.")

    try:
        _log(job, f"train_job: start payload={payload}")

        cfg = TrainConfig(
            model_type=payload.get("model_type", "pattern_learner"),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            normalize=bool(payload.get("normalize", False)),
            beta_vae=float(payload.get("beta_vae", 1.0)),
            kl_anneal=str(payload.get("kl_anneal", "linear")),
            kl_max_beta=float(payload.get("kl_max_beta", 1.0)),
            kl_warmup_epochs=int(payload.get("kl_warmup_epochs", 10)),
            early_stop_patience=int(payload.get("early_stop_patience", 12)),
            loss_weights=payload.get("loss_weights", None),
            enable_mixed_precision=bool(payload.get("enable_mixed_precision", False)),
        )
        trainer = MLTrainer(cfg)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        epochs            = int(payload.get("epochs", 100))
        batch_size        = int(payload.get("batch_size", 32))
        samples           = int(payload.get("samples", 1000))
        validation_split  = float(payload.get("validation_split", 0.2))
        use_generator     = bool(payload.get("use_generator", True))
        resume_from       = payload.get("resume_from", None)

        if resume_from and os.path.exists(resume_from):
            try:
                trainer.load_checkpoint(resume_from)
                _log(job, f"Resumed from checkpoint: {resume_from}")
            except Exception as e:
                _log(job, f"Resume failed: {e}")

        def progress_callback(epoch, total_epochs):
            # Trainer may not provide metrics; we record epoch counters reliably
            job.meta["progress"] = {"epoch": int(epoch), "total": int(total_epochs)}
            job.save_meta()
            _log(job, f"epoch {epoch}/{total_epochs}")

        # Train
        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            validation_split=validation_split,
            save_best=True,
            use_generator=use_generator,
            checkpoint_interval=5,
            progress_callback=progress_callback
        )

        # Best model path heuristic (works with the trainer you shared)
        best_model = os.path.join("checkpoints", f"{cfg.model_type}_best.pth")
        if not os.path.exists(best_model):
            # try to find any best file
            try:
                import glob
                cand = sorted(glob.glob("checkpoints/*best*.pth"))
                if cand:
                    best_model = cand[-1]
            except Exception:
                pass

        # Create a session ZIP (fallback if trainer doesn't provide helper)
        session_zip = os.path.join("artifacts", f"session_{job.id}.zip")
        try:
            with zipfile.ZipFile(session_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                # include best model if exists
                if best_model and os.path.exists(best_model):
                    zf.write(best_model, arcname=os.path.basename(best_model))
                # include history
                zf.writestr("history.json", json.dumps(getattr(trainer, "history", {}), indent=2))
                # include config
                zf.writestr("config.json", json.dumps(cfg.__dict__, indent=2))
        except Exception as e:
            _log(job, f"session zip error: {e}")
            session_zip = None

        job.meta["artifacts"] = {
            "best_model": best_model if os.path.exists(best_model) else None,
            "session_zip": session_zip if session_zip and os.path.exists(session_zip) else None
        }
        job.save_meta()

        _log(job, f"train_job: finished. best={job.meta['artifacts']['best_model']} zip={job.meta['artifacts']['session_zip']}")
        return job.meta["artifacts"]

    except Exception as e:
        if job:
            job.meta["progress"] = {"stage": "failed", "error": str(e)}
            job.save_meta()
        _log(job, f"train_job: failed: {e}")
        raise
import os
from datetime import datetime
import sympy as sp
from rq import get_current_job
from shared.ode_core import ComputeParams, compute_ode_full

def _log(msg: str):
    job = get_current_job()
    if not job:
        return
    meta = job.meta or {}
    logs = meta.get("logs", [])
    logs.append(f"[{datetime.utcnow().isoformat()}Z] {msg}")
    meta["logs"] = logs
    job.meta = meta
    job.save_meta()

def compute_job(payload: dict):
    """RQ entrypoint: worker.compute_job — generates an ODE via Master Theorem path."""
    _log(f"compute_job: payload keys={list(payload.keys())}")
    try:
        # Optionally load libraries if present in worker image
        basic_lib = special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
            _log("Loaded Basic/Special libraries")
        except Exception:
            _log("Basic/Special libraries not available in worker image")

        constructor_lhs = payload.get("constructor_lhs")
        if constructor_lhs:
            try:
                constructor_lhs = sp.sympify(constructor_lhs)
            except Exception:
                _log("Failed to sympify constructor_lhs; ignoring it")
                constructor_lhs = None

        p = ComputeParams(
            func_name          = payload.get("func_name", "exp(z)"),
            alpha              = payload.get("alpha", 1),
            beta               = payload.get("beta", 1),
            n                  = int(payload.get("n", 1)),
            M                  = payload.get("M", 0),
            use_exact          = bool(payload.get("use_exact", True)),
            simplify_level     = payload.get("simplify_level", "light"),
            lhs_source         = payload.get("lhs_source", "constructor"),
            constructor_lhs    = constructor_lhs,
            freeform_terms     = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library   = payload.get("function_library", "Basic"),
            basic_lib          = basic_lib,
            special_lib        = special_lib,
        )

        _log("compute_ode_full starting")
        res = compute_ode_full(p)
        _log("compute_ode_full finished")

        # JSON-safe return
        safe = {
            **res,
            "generator": str(res.get("generator", "")),
            "rhs":       str(res.get("rhs", "")),
            "solution":  str(res.get("solution", "")),
            "f_expr_preview": str(res.get("f_expr_preview", "")),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        job = get_current_job()
        if job:
            job.meta["progress"] = {"stage": "finished"}
            job.save_meta()
        return safe

    except Exception as e:
        job = get_current_job()
        if job:
            job.meta["progress"] = {"stage": "failed", "error": str(e)}
            job.save_meta()
        raise

def ping_job(payload: dict):
    _log("ping_job")
    return {"ok": True, "echo": payload, "ts": datetime.utcnow().isoformat() + "Z"}