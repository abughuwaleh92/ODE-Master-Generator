# worker.py
import os
import json
import glob
import zipfile
import inspect
from datetime import datetime

import sympy as sp
from rq import get_current_job

# ---- Core compute (ODE generation) ----
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# ---- Optional Trainer (present in your repo) ----
try:
    from src.ml.trainer import MLTrainer  # your trainer class
except Exception:
    MLTrainer = None

# ---- Optional: some repos define a TrainConfig; we handle both cases ----
try:
    from src.ml.trainer import TrainConfig
except Exception:
    TrainConfig = None

# ---- Redis for logs (separate connection from RQ’s internal one) ----
try:
    import redis
except Exception:
    redis = None


def _redis():
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or os.getenv("UPSTASH_REDIS_URL")
    if not url or not redis:
        return None
    # decode_responses=True is fine here because we only push/read plain text log lines
    return redis.from_url(url, decode_responses=True)


def _log(job, msg: str):
    """Append a log line to Redis (job-specific list), keep last ~5000 lines."""
    try:
        conn = _redis()
        if not conn or not job:
            return
        key = f"job:{job.id}:logs"
        conn.rpush(key, f"[{datetime.utcnow().isoformat()}Z] {msg}")
        conn.ltrim(key, -5000, -1)
    except Exception:
        pass


def _set_progress(job, **fields):
    """Persist progress in job.meta['progress'] so UI remains visible across epochs."""
    if not job:
        return
    try:
        job.meta.setdefault("progress", {})
        job.meta["progress"].update(fields)
        job.save_meta()
    except Exception:
        pass


def _set_artifacts(job, **fields):
    if not job:
        return
    try:
        job.meta.setdefault("artifacts", {})
        job.meta["artifacts"].update(fields)
        job.save_meta()
    except Exception:
        pass


# ----------------- Compute (Generate ODE) -----------------
def compute_job(payload: dict):
    """
    Called via RQ: "worker.compute_job"
    Does not rely on UI session-only objects (constructor_lhs is None on worker).
    Converts SymPy to JSON-safe strings before returning.
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
            func_name        = payload.get("func_name", "exp(z)"),
            alpha            = payload.get("alpha", 1),
            beta             = payload.get("beta", 1),
            n                = int(payload.get("n", 1)),
            M                = payload.get("M", 0),
            use_exact        = bool(payload.get("use_exact", True)),
            simplify_level   = payload.get("simplify_level", "light"),
            lhs_source       = payload.get("lhs_source", "constructor"),
            constructor_lhs  = None,  # worker cannot access UI constructor session
            freeform_terms   = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library = payload.get("function_library", "Basic"),
            basic_lib        = basic_lib,
            special_lib      = special_lib,
        )

        res = compute_ode_full(p)

        safe = {
            **res,
            "generator":      expr_to_str(res.get("generator")),
            "rhs":            expr_to_str(res.get("rhs")),
            "solution":       expr_to_str(res.get("solution")),
            "f_expr_preview": expr_to_str(res.get("f_expr_preview")),
            "timestamp":      datetime.utcnow().isoformat() + "Z",
        }

        _set_progress(job, stage="finished")
        _log(job, "compute_job: finished.")
        return safe

    except Exception as e:
        _set_progress(job, stage="failed", error=str(e))
        _log(job, f"compute_job: failed: {e}")
        raise


# ----------------- Helpers for trainer compatibility -----------------
def _supports(fn, param_name: str) -> bool:
    """Return True if callable `fn` supports a keyword arg `param_name`."""
    try:
        sig = inspect.signature(fn)
        return param_name in sig.parameters
    except Exception:
        return False


def _filter_kwargs(fn, kwargs: dict) -> dict:
    """Keep only kwargs supported by callable `fn`."""
    try:
        sig = inspect.signature(fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        # Fallback: pass nothing if we can't safely inspect
        return {}


def _build_trainer(payload: dict):
    """
    Instantiate MLTrainer safely across versions:
      - If TrainConfig exists and MLTrainer accepts it, use it.
      - Else pass (model_type, hidden_dim, enable_mixed_precision, etc.) filtered by __init__ signature.
    """
    if MLTrainer is None:
        raise RuntimeError("MLTrainer not available in worker environment.")

    model_type = payload.get("model_type", "pattern_learner")
    hidden_dim = int(payload.get("hidden_dim", 128))
    enable_amp = bool(payload.get("enable_mixed_precision", False))

    # Prefer TrainConfig if available AND MLTrainer.__init__ supports it.
    if TrainConfig is not None:
        try:
            cfg = TrainConfig(
                model_type=model_type,
                hidden_dim=hidden_dim,
                normalize=bool(payload.get("normalize", False)),
                beta_vae=float(payload.get("beta_vae", 1.0)) if "beta_vae" in payload else 1.0,
                kl_anneal=str(payload.get("kl_anneal", "linear")) if "kl_anneal" in payload else "linear",
                kl_max_beta=float(payload.get("kl_max_beta", 1.0)) if "kl_max_beta" in payload else 1.0,
                kl_warmup_epochs=int(payload.get("kl_warmup_epochs", 10)) if "kl_warmup_epochs" in payload else 10,
                early_stop_patience=int(payload.get("early_stop_patience", 12)) if "early_stop_patience" in payload else 12,
                loss_weights=payload.get("loss_weights", None),
                enable_mixed_precision=enable_amp,
            )
            # If __init__ supports a single config object:
            if _supports(MLTrainer.__init__, "cfg"):
                return MLTrainer(cfg)
            # Else pass expanded fields through signature filter:
            init_kwargs = {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "enable_mixed_precision": enable_amp,
                # add common kwargs your older MLTrainer may accept:
                "learning_rate": payload.get("learning_rate", 0.001),
                "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
            }
            init_kwargs = _filter_kwargs(MLTrainer.__init__, init_kwargs)
            return MLTrainer(**init_kwargs)
        except Exception:
            # Fall back to signature-filtered init below
            pass

    # No TrainConfig path → call MLTrainer with filtered kwargs
    init_kwargs = {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "enable_mixed_precision": enable_amp,
        "learning_rate": payload.get("learning_rate", 0.001),
        "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
        "device": payload.get("device", None),
        "input_dim": payload.get("input_dim", 12),
        "output_dim": payload.get("output_dim", 12),
    }
    init_kwargs = _filter_kwargs(MLTrainer.__init__, init_kwargs)
    return MLTrainer(**init_kwargs)


# ----------------- Train (Persistent RQ) -----------------
def train_job(payload: dict):
    """
    Background training with persistent progress/logging/artifacts.
    Works with *any* version of your MLTrainer by filtering args.
    """
    job = get_current_job()

    try:
        _log(job, f"train_job: start payload={payload}")
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        trainer = _build_trainer(payload)

        # Collect train kwargs then filter them against trainer.train signature
        train_kwargs = {
            "epochs":            int(payload.get("epochs", 100)),
            "batch_size":        int(payload.get("batch_size", 32)),
            "samples":           int(payload.get("samples", 1000)),
            "validation_split":  float(payload.get("validation_split", 0.2)),
            "use_generator":     bool(payload.get("use_generator", True)),
            # These may or may not exist depending on trainer version:
            "save_best":         True,
            "checkpoint_interval": 5,
        }

        # Progress callback (only pass if supported)
        def progress_callback(epoch, total_epochs):
            _set_progress(job, epoch=int(epoch), total=int(total_epochs))
            _log(job, f"epoch {epoch}/{total_epochs}")

        if _supports(trainer.train, "progress_callback"):
            train_kwargs["progress_callback"] = progress_callback

        # Filter kwargs to what this trainer actually supports
        safe_train_kwargs = _filter_kwargs(trainer.train, train_kwargs)

        # Resume support (if your trainer has a load_checkpoint method)
        resume_from = payload.get("resume_from", None)
        if resume_from and hasattr(trainer, "load_checkpoint") and os.path.exists(resume_from):
            try:
                trainer.load_checkpoint(resume_from)
                _log(job, f"Resumed from checkpoint: {resume_from}")
            except Exception as e:
                _log(job, f"Resume failed: {e}")

        # ---- Train ----
        _set_progress(job, stage="running", epoch=0, total=safe_train_kwargs.get("epochs", 0))
        trainer.train(**safe_train_kwargs)

        # ---- Locate a best model (robust) ----
        best_model = None
        # Prefer an explicit attribute if your trainer exposes it
        if hasattr(trainer, "best_model_path"):
            best_model = getattr(trainer, "best_model_path")

        if not best_model or not os.path.exists(best_model):
            # Common convention in your repo:
            mt = getattr(trainer, "model_type", None) or payload.get("model_type", "pattern_learner")
            candidate = os.path.join("checkpoints", f"{mt}_best.pth")
            if os.path.exists(candidate):
                best_model = candidate

        if not best_model:
            # Last resort: newest "*best*.pth"
            cands = sorted(glob.glob("checkpoints/*best*.pth"))
            if cands:
                best_model = cands[-1]

        # ---- Save a session ZIP for portability ----
        session_zip = os.path.join("artifacts", f"session_{job.id}.zip")
        try:
            with zipfile.ZipFile(session_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                if best_model and os.path.exists(best_model):
                    zf.write(best_model, arcname=os.path.basename(best_model))
                # history.json
                hist = getattr(trainer, "history", None)
                if hist is not None:
                    zf.writestr("history.json", json.dumps(hist, indent=2))
                # config-like info for reproducibility
                cfg_dict = {
                    "model_type": payload.get("model_type", "pattern_learner"),
                    "hidden_dim": int(payload.get("hidden_dim", 128)),
                    "normalize":  bool(payload.get("normalize", False)),
                }
                zf.writestr("config.json", json.dumps(cfg_dict, indent=2))
        except Exception as e:
            _log(job, f"session zip error: {e}")
            session_zip = None

        _set_artifacts(job,
            best_model=best_model if best_model and os.path.exists(best_model) else None,
            session_zip=session_zip if session_zip and os.path.exists(session_zip) else None
        )

        _set_progress(job, stage="finished")
        _log(job, f"train_job: finished. best={best_model} zip={session_zip}")
        # Return artifacts so RQ result panel shows them even before UI polls meta
        return {"best_model": best_model, "session_zip": session_zip}

    except Exception as e:
        _set_progress(job, stage="failed", error=str(e))
        _log(job, f"train_job: failed: {e}")
        raise