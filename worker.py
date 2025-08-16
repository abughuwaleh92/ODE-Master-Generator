# worker.py
import os
import json
from datetime import datetime
import sympy as sp

from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# This is the function path used by RQ: "worker.compute_job"
def compute_job(payload: dict):
    """
    payload fields mirror ComputeParams, plus hints for libraries.
    NOTE: On Railway worker, we usually don't load heavy src libs; if needed, set flags and import.
    """
    try:
        # Optional: import src libraries here if you want factory behavior
        basic_lib = None
        special_lib = None
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
            simplify_level = payload.get("simplify_level","light"),
            lhs_source     = payload.get("lhs_source","constructor"),
            constructor_lhs= None,  # worker doesn't know the session constructor; leave None
            freeform_terms = payload.get("freeform_terms"),
            arbitrary_lhs_text = payload.get("arbitrary_lhs_text"),
            function_library = payload.get("function_library","Basic"),
            basic_lib = basic_lib,
            special_lib = special_lib,
        )
        res = compute_ode_full(p)

        # JSON-safe (convert sympy to string)
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
