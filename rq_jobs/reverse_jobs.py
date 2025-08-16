# rq_jobs/reverse_jobs.py
import sympy as sp
from typing import Any, Dict

from shared.inverse_core import (
    DEFAULT_SEARCH,
    infer_params_from_solution,
    infer_from_ode_single,
    infer_params_multi_blocks,
    infer_from_ode_multi,
)

# ── Single‑block jobs ─────────────────────────────────────────────────────────
def inverse_solution_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    x = sp.Symbol(payload.get("x_symbol", "x"), real=True)
    y_expr = sp.sympify(payload["y_expr"])
    libs   = payload.get("libraries", ["Basic","Special","Phi"])
    search = payload.get("search", DEFAULT_SEARCH)
    topk   = int(payload.get("topk", 5))
    res = infer_params_from_solution(y_expr, libraries=libs, search=search, x_symbol=x, topk=topk)
    out = []
    for r in res:
        out.append({
            "function_name": r.function_name,
            "alpha": r.alpha, "beta": r.beta, "n": r.n, "M": r.M,
            "scale_C": r.scale_C, "rmse": r.rmse,
            "candidate_expr": str(r.candidate_expr)
        })
    return {"status": "ok", "candidates": out}

def inverse_ode_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    x = sp.Symbol(payload.get("x_symbol", "x"), real=True)
    lhs = sp.sympify(payload["lhs"])
    rhs = sp.sympify(payload["rhs"])
    libs   = payload.get("libraries", ["Basic","Special","Phi"])
    search = payload.get("search", DEFAULT_SEARCH)
    topk   = int(payload.get("topk", 5))
    res = infer_from_ode_single(lhs, rhs, libraries=libs, search=search, x_symbol=x, topk=topk)
    out = []
    for r in res:
        out.append({
            "function_name": r.function_name,
            "alpha": r.alpha, "beta": r.beta, "n": r.n, "M": r.M,
            "scale_C": r.scale_C, "rmse": r.rmse,
            "candidate_expr": str(r.candidate_expr)
        })
    return {"status": "ok", "candidates": out}

# ── Multi‑block jobs ──────────────────────────────────────────────────────────
def inverse_solution_multiblock_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    x = sp.Symbol(payload.get("x_symbol", "x"), real=True)
    y_expr = sp.sympify(payload["y_expr"])
    libs   = payload.get("libraries", ["Basic","Special","Phi"])
    search = payload.get("search", DEFAULT_SEARCH)
    mode   = payload.get("mode", "product")
    J      = int(payload.get("J", 2))
    topk   = int(payload.get("topk", 5))
    pool   = int(payload.get("pool_size", 20))
    combos = int(payload.get("max_combos", 100))
    res = infer_params_multi_blocks(
        y_expr, J=J, mode=mode, libraries=libs, search=search, x_symbol=x,
        topk=topk, pool_size=pool, max_combos=combos
    )
    out = []
    for c in res:
        out.append({
            "blocks": [{
                "function_name": r.function_name, "alpha": r.alpha, "beta": r.beta,
                "n": r.n, "M": r.M
            } for r in c.blocks],
            "scales": c.scales,
            "rmse": c.rmse,
            "expr": str(c.expr)
        })
    return {"status": "ok", "candidates": out}

def inverse_ode_multiblock_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    x = sp.Symbol(payload.get("x_symbol", "x"), real=True)
    lhs = sp.sympify(payload["lhs"])
    rhs = sp.sympify(payload["rhs"])
    libs   = payload.get("libraries", ["Basic","Special","Phi"])
    search = payload.get("search", DEFAULT_SEARCH)
    mode   = payload.get("mode", "product")
    J      = int(payload.get("J", 2))
    topk   = int(payload.get("topk", 5))
    pool   = int(payload.get("pool_size", 20))
    combos = int(payload.get("max_combos", 100))
    res = infer_from_ode_multi(
        lhs, rhs, J=J, mode=mode, libraries=libs, search=search, x_symbol=x,
        topk=topk, pool_size=pool, max_combos=combos
    )
    out = []
    for c in res:
        out.append({
            "blocks": [{
                "function_name": r.function_name, "alpha": r.alpha, "beta": r.beta,
                "n": r.n, "M": r.M
            } for r in c.blocks],
            "scales": c.scales,
            "rmse": c.rmse,
            "expr": str(c.expr)
        })
    return {"status": "ok", "candidates": out}
