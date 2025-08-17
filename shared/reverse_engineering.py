# shared/reverse_engineering.py
"""
Reverse engineering pipeline
- Heuristic parser for linear/nonlinear, order
- Estimate (alpha, beta, n, M) from structure when possible
- Numeric fit scaffold with multi-block factors:
    y(x) ≈ x^M * prod_j [ phi_j( alpha_j * x**beta_j ) ] ** n_j
- Inner adapters: allow scaled inner u = a*x + b and exp-of-exp with scale 'a'
"""
import json, math, random
from typing import Dict, Any, Optional, List
import numpy as np
import sympy as sp
from sympy import Eq
from shared.phi_lib import phi as phi_fn

x = sp.Symbol('x', real=True)
y = sp.Function('y')

def _to_sympy(expr_or_text) -> sp.Expr:
    if isinstance(expr_or_text, (sp.Expr, sp.Equality)):
        return expr_or_text
    s = str(expr_or_text or "").strip()
    try:
        return sp.sympify(s)
    except Exception:
        # try as equation "lhs = rhs"
        try:
            lhs, rhs = s.split("=")
            return sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
        except Exception:
            raise ValueError("Unable to parse ODE text/expression.")

def _detect_order_and_linearity(eq: Eq) -> Dict[str, Any]:
    # crude: scan highest derivative of y wrt x
    order = 0
    linear = True
    if isinstance(eq, sp.Equality):
        lhs = eq.lhs - eq.rhs
    else:
        lhs = eq
    # scan derivatives
    for i in range(1, 7):
        di = sp.Derivative(y(x), (x, i))
        if lhs.has(di):
            order = max(order, i)
    # nonlinearity check: any product of y, y' terms or y**k (k!=1)
    ys = [y(x)] + [sp.Derivative(y(x), (x, i)) for i in range(1, order+1)]
    syms = set()
    for t in ys:
        syms |= set(lhs.atoms(type(t)))
    # heuristic nonlinear detection
    # If powers other than 1 or products of ys exist, mark nonlinear
    for t in ys:
        for p in lhs.atoms(sp.Pow):
            if p.has(t) and (p.exp != 1):
                linear = False
    for m in lhs.atoms(sp.Mul):
        if sum(1 for t in ys if m.has(t)) >= 2:
            linear = False
    return {"order": order or 1, "linear": linear}

def _guess_function(eq: Eq) -> str:
    # look for sin, cos, exp, erf, logistic, sinh, cosh
    s = str(eq)
    for name in ["erf", "erfc", "logistic", "sigmoid", "sinh", "cosh", "tanh", "sin", "cos", "tan", "exp", "log"]:
        if name in s:
            return "logistic" if name in ("logistic","sigmoid") else name
    return "exp"

def _estimate_params_from_structure(eq: Eq, order: int) -> Dict[str, Any]:
    # very rough heuristic
    # α in [-5,5], β in [0.1,5], n in [1,5], M in [-5,5]
    return {
        "alpha": 1.0,
        "beta": 1.0,
        "n": max(1, min(5, order)),
        "M": 0.0
    }

def _grid_numeric_fit(xs: np.ndarray, ys: np.ndarray, J: int = 1,
                      candidate_funcs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Multi-block fit scaffold:
    y(x) ≈ x^M * prod_{j=1}^J [ phi_j( alpha_j * (a_j*x + b_j)**beta_j ) ] ** n_j
    We grid-search rough alphas/betas/n and do a linear fit for outer scale.
    Returns best configuration (coarse).
    """
    if candidate_funcs is None:
        candidate_funcs = ["exp", "sin", "cos", "erf", "logistic", "sinh", "cosh"]
    best = {"loss": float("inf")}
    # simple coarse grid (keep small to avoid long runs)
    M_grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    alpha_grid = [0.5, 1.0, 2.0]
    beta_grid = [0.5, 1.0, 2.0]
    n_grid = [1, 2, 3]
    a_grid = [0.5, 1.0, 2.0]
    b_grid = [0.0]

    for M in M_grid:
        xf = np.power(np.clip(xs, 1e-6, None), M)  # x^M
        # single block J=1 for runtime; you can loop J>1 similarly
        for phi_name in candidate_funcs:
            for alpha in alpha_grid:
                for beta in beta_grid:
                    for n in n_grid:
                        for a in a_grid:
                            for b in b_grid:
                                # build block value
                                u = a*xs + b
                                try:
                                    # numeric phi evaluation
                                    if phi_name in ("exp","log"):
                                        inner = np.power(np.clip(u, 1e-6, None), beta)
                                    else:
                                        inner = np.power(np.abs(u), beta) * np.sign(u)  # odd/even safe-ish
                                    if phi_name == "exp":
                                        block = np.exp(alpha * inner)
                                    elif phi_name == "sin":
                                        block = np.sin(alpha * inner)
                                    elif phi_name == "cos":
                                        block = np.cos(alpha * inner)
                                    elif phi_name == "erf":
                                        from math import erf
                                        block = np.array([erf(alpha*val) for val in inner])
                                    elif phi_name in ("logistic","sigmoid"):
                                        block = 1/(1 + np.exp(-alpha * inner))
                                    elif phi_name == "sinh":
                                        block = np.sinh(alpha * inner)
                                    elif phi_name == "cosh":
                                        block = np.cosh(alpha * inner)
                                    else:  # tan, tanh etc could be added similarly
                                        block = np.exp(alpha * inner)
                                    model = xf * np.power(block, n)
                                    # solve outer scale c in least squares: ys ≈ c*model
                                    A = model.reshape(-1, 1)
                                    c, *_ = np.linalg.lstsq(A, ys, rcond=None)
                                    yhat = c[0] * model
                                    loss = float(np.mean((yhat - ys) ** 2))
                                    if loss < best["loss"]:
                                        best = {
                                            "loss": loss, "M": M, "funcs": [phi_name],
                                            "alpha": [alpha], "beta": [beta], "n":[n], "a":[a], "b":[b],
                                            "outer_scale": float(c[0]),
                                        }
                                except Exception:
                                    continue
    return best

def reverse_engineer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try: heuristic parse; else numeric fit if samples provided.
    Returns:
      {
        "linear": bool, "order": int,
        "alpha": float, "beta": float, "n": int, "M": float,
        "function_name": str,
        "reconstructed_preview": "latex or text",
        "fit": {...}   # if numeric fitting used
      }
    """
    # 1) Path: samples for numeric fit
    samples = payload.get("samples")
    if samples and "x" in samples and "y" in samples:
        xs = np.asarray(samples["x"], dtype=float)
        ys = np.asarray(samples["y"], dtype=float)
        best = _grid_numeric_fit(xs, ys, J=1)
        fn = best.get("funcs", ["exp"])[0]
        alpha = best.get("alpha", [1.0])[0]
        beta  = best.get("beta", [1.0])[0]
        n     = best.get("n", [1])[0]
        M     = best.get("M", 0.0)
        return {
            "linear": True, "order": 1,
            "alpha": alpha, "beta": beta, "n": n, "M": M,
            "function_name": fn,
            "fit": best,
            "reconstructed_preview": f"x^{M} * {fn}( {alpha} * (a x + b)^{beta} )^{n}"
        }

    # 2) Path: equation text
    ode_text = payload.get("ode_text")
    lhs_text, rhs_text = payload.get("ode_lhs"), payload.get("ode_rhs")
    if ode_text or (lhs_text and rhs_text):
        if ode_text:
            expr = _to_sympy(ode_text)
        else:
            expr = sp.Eq(sp.sympify(lhs_text), sp.sympify(rhs_text))
        # detect order/linearity
        det = _detect_order_and_linearity(expr if isinstance(expr, sp.Equality) else sp.Eq(expr, 0))
        order, linear = det["order"], det["linear"]
        fn = _guess_function(expr)
        est = _estimate_params_from_structure(expr, order)
        # reconstruct preview string
        preview = f"order={order}, linear={linear}, phi={fn}, alpha={est['alpha']}, beta={est['beta']}, n={est['n']}, M={est['M']}"
        return {
            "linear": linear, "order": order,
            "alpha": est["alpha"], "beta": est["beta"], "n": est["n"], "M": est["M"],
            "function_name": fn,
            "reconstructed_preview": preview
        }

    raise ValueError("reverse_job requires 'samples' or 'ode_text' or ('ode_lhs','ode_rhs').")