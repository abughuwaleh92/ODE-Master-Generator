# worker.py
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