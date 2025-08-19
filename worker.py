# worker.py
import os
import json
import glob
import zipfile
import inspect
from datetime import datetime

import sympy as sp
from rq import get_current_job

# ODE compute core
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional ML trainer (support both "config-style" and "kwargs-style")
try:
    from src.ml.trainer import MLTrainer, TrainConfig  # your code may or may not export TrainConfig
except Exception:
    MLTrainer = None
    TrainConfig = None

# Plain Redis (for separate text logs)
try:
    import redis
except Exception:
    redis = None


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def _redis():
    url = (
        os.getenv("REDIS_URL")
        or os.getenv("REDIS_TLS_URL")
        or os.getenv("UPSTASH_REDIS_URL")
    )
    if not url or not redis:
        return None
    # decode_responses=True for human logs (strings)
    return redis.from_url(url, decode_responses=True)


def _append_log(job, msg: str):
    """Append a single log line to Redis list job:{id}:logs (bounded)."""
    try:
        r = _redis()
        if not r or not job:
            return
        key = f"job:{job.id}:logs"
        r.rpush(key, f"[{datetime.utcnow().isoformat()}Z] {msg}")
        r.ltrim(key, -5000, -1)
    except Exception:
        pass


def _accepted_kwargs(callable_obj, **kwargs):
    """Filter kwargs to only those accepted by callable_obj."""
    try:
        sig = inspect.signature(callable_obj)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return {}


def _mk_trainconfig_from_payload(payload):
    """
    Build a TrainConfig object (if available), using only parameters TrainConfig accepts.
    If TrainConfig isn't available, return a dict config.
    """
    # Candidate configuration fields we know about; add more as your trainer grows.
    raw = {
        "model_type": payload.get("model_type", "pattern_learner"),
        "hidden_dim": int(payload.get("hidden_dim", 128)),
        "normalize": bool(payload.get("normalize", False)),
        "beta_vae": float(payload.get("beta_vae", 1.0)),
        "kl_anneal": str(payload.get("kl_anneal", "linear")),
        "kl_max_beta": float(payload.get("kl_max_beta", 1.0)),
        "kl_warmup_epochs": int(payload.get("kl_warmup_epochs", 10)),
        "early_stop_patience": int(payload.get("early_stop_patience", 12)),
        "loss_weights": payload.get("loss_weights", None),
        "enable_mixed_precision": bool(payload.get("enable_mixed_precision", False)),
        "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
        "learning_rate": float(payload.get("learning_rate", 1e-3)),
        "input_dim": int(payload.get("input_dim", 12)),
        "output_dim": int(payload.get("output_dim", 12)),
        # (extend with any other fields your TrainConfig supports)
    }

    if TrainConfig is None:
        return raw  # fall back to dict

    try:
        sig = inspect.signature(TrainConfig.__init__)
        only = {k: v for k, v in raw.items() if k in sig.parameters}
        return TrainConfig(**only)
    except Exception:
        # If TrainConfig exists but signature probing fails, still try a best-effort dict.
        return raw


def _instantiate_trainer(job, payload):
    """
    Instantiate MLTrainer robustly across versions:
    - If trainer requires a single config argument, try ('cfg', 'config', 'conf') and positional.
    - Else pass filtered kwargs to __init__.
    """
    if MLTrainer is None:
        raise RuntimeError("MLTrainer is not importable on the worker.")

    cfg_obj_or_dict = _mk_trainconfig_from_payload(payload)
    attempts = []
    made = None

    # 1) Try *config object/dict* under common names and as positional
    # If it's a real object (dataclass), try that first; else dict is fine too.
    for label, kwargs in [
        ("MLTrainer(cfg=...)", {"cfg": cfg_obj_or_dict}),
        ("MLTrainer(config=...)", {"config": cfg_obj_or_dict}),
        ("MLTrainer(conf=...)", {"conf": cfg_obj_or_dict}),
    ]:
        try:
            # Only pass kwargs that __init__ accepts
            filtered = _accepted_kwargs(MLTrainer.__init__, **kwargs)
            if filtered:
                made = MLTrainer(**filtered)
                _append_log(job, f"Trainer created via {label}")
                return made
            attempts.append(f"{label}: not accepted by __init__")
        except Exception as e:
            attempts.append(f"{label}: {e}")

    # Try positional single argument (some versions do MLTrainer(config))
    try:
        made = MLTrainer(cfg_obj_or_dict)  # positional
        _append_log(job, "Trainer created via MLTrainer(<config> positional).")
        return made
    except Exception as e:
        attempts.append(f"MLTrainer(<config> positional): {e}")

    # 2) Fall back to kwargs style (older trainers)
    ctor_candidates = {
        "model_type": payload.get("model_type", "pattern_learner"),
        "input_dim": int(payload.get("input_dim", 12)),
        "hidden_dim": int(payload.get("hidden_dim", 128)),
        "output_dim": int(payload.get("output_dim", 12)),
        "learning_rate": float(payload.get("learning_rate", 1e-3)),
        "device": payload.get("device", None),
        "checkpoint_dir": payload.get("checkpoint_dir", "checkpoints"),
        "enable_mixed_precision": bool(payload.get("enable_mixed_precision", False)),
    }
    try:
        filtered = _accepted_kwargs(MLTrainer.__init__, **ctor_candidates)
        made = MLTrainer(**filtered)
        _append_log(job, f"Trainer created with kwargs={filtered}")
        return made
    except Exception as e:
        attempts.append(f"kwargs ctor: {e}")

    # If we reach here, all attempts failed
    raise RuntimeError(
        "Failed to instantiate MLTrainer. Attempts:\n- " + "\n- ".join(attempts)
    )


# -------------------------------------------------------------------
# Compute Job (Generate ODE)
# RQ path: "worker.compute_job"
# -------------------------------------------------------------------
def compute_job(payload: dict):
    job = get_current_job()
    try:
        # Mark running for the UI
        if job:
            job.meta["progress"] = {"stage": "running"}
            job.save_meta()

        _append_log(job, f"compute_job: start payload={payload}")

        # Try to load function libraries so worker can resolve f(z)
        basic_lib = special_lib = None
        try:
            from src.functions.basic_functions import BasicFunctions
            from src.functions.special_functions import SpecialFunctions
            basic_lib = BasicFunctions()
            special_lib = SpecialFunctions()
            _append_log(job, "Loaded BasicFunctions & SpecialFunctions.")
        except Exception as e:
            _append_log(job, f"Library load skipped: {e}")

        p = ComputeParams(
            func_name=payload.get("func_name", "exp(z)"),
            alpha=payload.get("alpha", 1),
            beta=payload.get("beta", 1),
            n=int(payload.get("n", 1)),
            M=payload.get("M", 0),
            use_exact=bool(payload.get("use_exact", True)),
            simplify_level=payload.get("simplify_level", "light"),
            lhs_source=payload.get("lhs_source", "constructor"),
            constructor_lhs=None,  # worker doesn't know UI constructor session
            freeform_terms=payload.get("freeform_terms"),
            arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
            function_library=payload.get("function_library", "Basic"),
            basic_lib=basic_lib,
            special_lib=special_lib,
        )

        res = compute_ode_full(p)

        safe = {
            **res,
            "generator": expr_to_str(res.get("generator")),
            "rhs": expr_to_str(res.get("rhs")),
            "solution": expr_to_str(res.get("solution")),
            "f_expr_preview": expr_to_str(res.get("f_expr_preview")),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if job:
            job.meta["progress"] = {"stage": "finished"}
            job.meta["artifacts"] = {}
            job.save_meta()

        _append_log(job, "compute_job: finished.")
        return safe

    except Exception as e:
        if job:
            job.meta["progress"] = {"stage": "failed", "error": str(e)}
            job.save_meta()
        _append_log(job, f"compute_job: failed: {e}")
        raise


# -------------------------------------------------------------------
# Train Job (Persistent, version-robust)
# RQ path: "worker.train_job"
# -------------------------------------------------------------------
def train_job(payload: dict):
    job = get_current_job()
    if MLTrainer is None:
        _append_log(job, "train_job: MLTrainer not importable.")
        raise RuntimeError("MLTrainer not available on worker.")

    try:
        if job:
            job.meta["progress"] = {"stage": "running", "epoch": 0}
            job.save_meta()

        _append_log(job, f"train_job: start payload={payload}")

        # Instantiate trainer in a version-robust way
        trainer = _instantiate_trainer(job, payload)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        # Training args
        epochs = int(payload.get("epochs", 100))
        batch_size = int(payload.get("batch_size", 32))
        samples = int(payload.get("samples", 1000))
        validation_split = float(payload.get("validation_split", 0.2))
        use_generator = bool(payload.get("use_generator", True))
        resume_from = payload.get("resume_from", None)

        # Resume if given
        if resume_from and os.path.exists(resume_from):
            try:
                trainer.load_checkpoint(resume_from)
                _append_log(job, f"Resumed from {resume_from}")
            except Exception as e:
                _append_log(job, f"Resume failed: {e}")

        # Progress callback
        def progress_callback(epoch, total_epochs):
            if job:
                job.meta["progress"] = {
                    "stage": "running",
                    "epoch": int(epoch),
                    "total": int(total_epochs),
                }
                job.save_meta()
            _append_log(job, f"epoch {epoch}/{total_epochs}")

        # Call train() with only the kwargs it accepts
        candidate = {
            "epochs": epochs,
            "batch_size": batch_size,
            "samples": samples,
            "validation_split": validation_split,
            "use_generator": use_generator,
            "checkpoint_interval": 5,
            "save_best": True,
            "progress_callback": progress_callback,
        }
        train_kwargs = _accepted_kwargs(trainer.train, **candidate)
        _append_log(job, f"trainer.train kwargs={train_kwargs}")
        trainer.train(**train_kwargs)

        # Locate best checkpoint
        model_type = payload.get("model_type", "pattern_learner")
        default_best = os.path.join("checkpoints", f"{model_type}_best.pth")
        best_model = default_best if os.path.exists(default_best) else None
        if not best_model:
            cand = sorted(glob.glob(os.path.join("checkpoints", "*best*.pth")))
            best_model = cand[-1] if cand else None

        # Build session ZIP (history + config + best checkpoint)
        session_zip = os.path.join("artifacts", f"session_{job.id}.zip")
        try:
            with zipfile.ZipFile(session_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                if best_model and os.path.exists(best_model):
                    zf.write(best_model, arcname=os.path.basename(best_model))
                hist = getattr(trainer, "history", {})
                zf.writestr("history.json", json.dumps(hist, indent=2))
                # Include whichever config we used
                cfg_payload = _mk_trainconfig_from_payload(payload)
                # If it’s an object, try its __dict__
                if hasattr(cfg_payload, "__dict__"):
                    cfg_dump = {k: v for k, v in vars(cfg_payload).items() if not k.startswith("_")}
                else:
                    cfg_dump = cfg_payload
                zf.writestr("config.json", json.dumps(cfg_dump, indent=2))
        except Exception as e:
            _append_log(job, f"session zip error: {e}")
            session_zip = None

        if job:
            job.meta["progress"] = {"stage": "finished", "epoch": epochs, "total": epochs}
            job.meta["artifacts"] = {
                "best_model": best_model if best_model and os.path.exists(best_model) else None,
                "session_zip": session_zip if session_zip and os.path.exists(session_zip) else None,
            }
            job.save_meta()

        _append_log(job, f"train_job: finished best={best_model} zip={session_zip}")
        return job.meta["artifacts"]

    except Exception as e:
        if job:
            job.meta["progress"] = {"stage": "failed", "error": str(e)}
            job.save_meta()
        _append_log(job, f"train_job: failed: {e}")
        raise