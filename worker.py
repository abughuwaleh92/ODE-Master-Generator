# worker.py
import os
import json
import glob
import zipfile
import inspect
from datetime import datetime

import sympy as sp
from rq import get_current_job

# Core ODE compute
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional ML trainer (handles old/new trainer APIs)
try:
    from src.ml.trainer import MLTrainer, TrainConfig
except Exception:
    MLTrainer = None
    TrainConfig = None

# Direct Redis for logs (RQ uses its own connection; we use a separate one for plain text logs)
try:
    import redis
except Exception:
    redis = None


def _redis():
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or os.getenv("UPSTASH_REDIS_URL")
    if not url or not redis:
        return None
    # Here decode_responses=True is OK because we only push/read plain strings for logs
    return redis.from_url(url, decode_responses=True)


def _log(job, msg: str):
    """Append a log line for this job into Redis list job:{id}:logs (bounded)."""
    try:
        conn = _redis()
        if not conn or not job:
            return
        key = f"job:{job.id}:logs"
        conn.rpush(key, f"[{datetime.utcnow().isoformat()}Z] {msg}")
        conn.ltrim(key, -5000, -1)
    except Exception:
        pass


def _accepted_kwargs(callable_obj, **kwargs):
    """
    Filter kwargs so we pass only what the target callable actually accepts.
    Works for functions and bound methods. Uses inspect.signature safely.
    """
    try:
        sig = inspect.signature(callable_obj)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return accepted
    except Exception:
        # If introspection fails, pass nothing extra
        return {}


# ----------------- Compute (Generate ODE) -----------------
# Called by RQ as "worker.compute_job"
def compute_job(payload: dict):
    """
    Stateless compute on worker. Builds ComputeParams from payload,
    calls compute_ode_full, returns JSON-safe result.
    """
    job = get_current_job()
    try:
        _log(job, f"compute_job: start payload={payload}")

        # Best effort: load function libraries so worker can resolve f(z)
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
            constructor_lhs = None,  # worker can't access UI constructor session
            freeform_terms  = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library= payload.get("function_library", "Basic"),
            basic_lib       = basic_lib,
            special_lib     = special_lib,
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
# Called by RQ as "worker.train_job"
def train_job(payload: dict):
    """
    Background training. Robust to different MLTrainer versions:
    - Instantiates MLTrainer either with TrainConfig or with direct kwargs (introspection).
    - Calls trainer.train() with only the kwargs it accepts (introspection).
    - Persists progress in job.meta and writes logs to Redis.
    - Exports best checkpoint (if any) and a session ZIP with history+config.
    """
    job = get_current_job()
    if MLTrainer is None:
        _log(job, "train_job: MLTrainer not available on worker PYTHONPATH.")
        raise RuntimeError("MLTrainer not available.")

    try:
        _log(job, f"train_job: start payload={payload}")

        # ----------------- Instantiate Trainer (compatible with both APIs) -----------------
        # Preferred path: TrainConfig exists and MLTrainer.__init__ accepts "cfg"
        trainer = None
        try:
            init_sig = inspect.signature(MLTrainer.__init__)
            init_params = init_sig.parameters
        except Exception:
            init_params = {}

        if "cfg" in init_params and TrainConfig is not None:
            # Build TrainConfig from payload (fallback defaults)
            cfg = TrainConfig(
                model_type         = payload.get("model_type", "pattern_learner"),
                hidden_dim         = int(payload.get("hidden_dim", 128)),
                normalize          = bool(payload.get("normalize", False)),
                beta_vae           = float(payload.get("beta_vae", 1.0)),
                kl_anneal          = str(payload.get("kl_anneal", "linear")),
                kl_max_beta        = float(payload.get("kl_max_beta", 1.0)),
                kl_warmup_epochs   = int(payload.get("kl_warmup_epochs", 10)),
                early_stop_patience= int(payload.get("early_stop_patience", 12)),
                loss_weights       = payload.get("loss_weights", None),
                enable_mixed_precision = bool(payload.get("enable_mixed_precision", False)),
                checkpoint_dir     = payload.get("checkpoint_dir", "checkpoints"),
                learning_rate      = float(payload.get("learning_rate", 1e-3)),
                input_dim          = int(payload.get("input_dim", 12)),
                output_dim         = int(payload.get("output_dim", 12)),
            )
            trainer = MLTrainer(cfg)
            _log(job, "MLTrainer instantiated with TrainConfig.")
        else:
            # Older trainer API: pass only args that __init__ supports
            default_ctor_kwargs = {
                "model_type": payload.get("model_type", "pattern_learner"),
                "input_dim": int(payload.get("input_dim", 12)),
                "hidden_dim": int(payload.get("hidden_dim", 128)),
                "output_dim": int(payload.get("output_dim", 12)),
                "learning_rate": float(payload.get("learning_rate", 1e-3)),
                "device": payload.get("device", None),
                "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
                "enable_mixed_precision": bool(payload.get("enable_mixed_precision", False)),
            }
            ctor_kwargs = _accepted_kwargs(MLTrainer.__init__, **default_ctor_kwargs)
            trainer = MLTrainer(**ctor_kwargs)
            _log(job, f"MLTrainer instantiated with kwargs={ctor_kwargs}.")

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        # ----------------- Training parameters -----------------
        epochs           = int(payload.get("epochs", 100))
        batch_size       = int(payload.get("batch_size", 32))
        samples          = int(payload.get("samples", 1000))
        validation_split = float(payload.get("validation_split", 0.2))
        use_generator    = bool(payload.get("use_generator", True))
        resume_from      = payload.get("resume_from", None)

        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            try:
                trainer.load_checkpoint(resume_from)
                _log(job, f"Resumed from checkpoint: {resume_from}")
            except Exception as e:
                _log(job, f"Resume failed: {e}")

        # Progress callback: persists epoch info
        def progress_callback(epoch, total_epochs):
            job.meta["progress"] = {"epoch": int(epoch), "total": int(total_epochs)}
            job.save_meta()
            _log(job, f"epoch {epoch}/{total_epochs}")

        # ----------------- Call train() with only accepted kwargs -----------------
        # These are candidate kwargs; we will filter to what trainer.train actually accepts.
        candidate_train_kwargs = {
            "epochs": epochs,
            "batch_size": batch_size,
            "samples": samples,
            "validation_split": validation_split,
            # Optional knobs (will be dropped if not supported by this trainer):
            "save_best": True,
            "use_generator": use_generator,
            "checkpoint_interval": 5,
            "progress_callback": progress_callback,
        }
        train_kwargs = _accepted_kwargs(trainer.train, **candidate_train_kwargs)
        _log(job, f"Calling trainer.train with kwargs={train_kwargs}")
        trainer.train(**train_kwargs)

        # ----------------- Collect artifacts -----------------
        # Heuristic to find "best" checkpoint produced by various trainer versions
        best_model = payload.get("best_model", None)
        if not best_model:
            # First try common name
            model_type = payload.get("model_type", "pattern_learner")
            default_best = os.path.join("checkpoints", f"{model_type}_best.pth")
            if os.path.exists(default_best):
                best_model = default_best
            else:
                # Fallback: any "*best*.pth"
                cand = sorted(glob.glob(os.path.join("checkpoints", "*best*.pth")))
                best_model = cand[-1] if cand else None

        # Build a session ZIP with history + config + best checkpoint (if found)
        session_zip = os.path.join("artifacts", f"session_{job.id}.zip")
        try:
            with zipfile.ZipFile(session_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                # include best model if exists
                if best_model and os.path.exists(best_model):
                    zf.write(best_model, arcname=os.path.basename(best_model))
                # include training history if available
                hist = getattr(trainer, "history", {})
                zf.writestr("history.json", json.dumps(hist, indent=2))
                # include config (if we had TrainConfig; else summarize ctor kwargs)
                if TrainConfig is not None and "cfg" in init_params:
                    # Retrieve the internal cfg if trainer exposes it; else log payload snapshot
                    cfg_dict = getattr(getattr(trainer, "cfg", None), "__dict__", None)
                    if cfg_dict is None:
                        cfg_dict = {
                            "model_type": payload.get("model_type", "pattern_learner"),
                            "hidden_dim": int(payload.get("hidden_dim", 128)),
                            "normalize": bool(payload.get("normalize", False)),
                        }
                    zf.writestr("config.json", json.dumps(cfg_dict, indent=2))
                else:
                    # Older API: store what we used
                    zf.writestr("config.json", json.dumps({
                        "ctor_kwargs": _accepted_kwargs(MLTrainer.__init__, **{
                            "model_type": payload.get("model_type", "pattern_learner"),
                            "input_dim": int(payload.get("input_dim", 12)),
                            "hidden_dim": int(payload.get("hidden_dim", 128)),
                            "output_dim": int(payload.get("output_dim", 12)),
                            "learning_rate": float(payload.get("learning_rate", 1e-3)),
                            "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
                            "enable_mixed_precision": bool(payload.get("enable_mixed_precision", False)),
                        })
                    }, indent=2))
        except Exception as e:
            _log(job, f"session zip error: {e}")
            session_zip = None

        job.meta["artifacts"] = {
            "best_model": best_model if best_model and os.path.exists(best_model) else None,
            "session_zip": session_zip if session_zip and os.path.exists(session_zip) else None,
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