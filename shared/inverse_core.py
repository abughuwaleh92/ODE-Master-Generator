# shared/inverse_core.py
import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

# Optional function libraries
try:
    from src.functions.basic_functions import BasicFunctions
except Exception:
    BasicFunctions = None

try:
    from src.functions.special_functions import SpecialFunctions
except Exception:
    SpecialFunctions = None

try:
    from src.functions.phi_library import PhiLibrary
except Exception:
    PhiLibrary = None

# Reuse your typed get_function_expr if present; otherwise a simple fallback
def _fallback_get_expr(lib, name: str) -> sp.Expr:
    z = sp.Symbol("z", real=True)
    if hasattr(lib, "get_function"):
        try:
            f = lib.get_function(name)
            return sp.sympify(f)
        except Exception:
            pass
    return z  # identity

def get_function_expr_any(libraries: List[str], name: str) -> sp.Expr:
    """
    Extended resolver across Basic / Special / Phi libraries.
    """
    if ("Basic" in libraries) and BasicFunctions:
        b = BasicFunctions()
        if name in b.get_function_names():
            return _fallback_get_expr(b, name)
    if ("Special" in libraries) and SpecialFunctions:
        s = SpecialFunctions()
        if name in s.get_function_names():
            return _fallback_get_expr(s, name)
    if ("Phi" in libraries) and PhiLibrary:
        p = PhiLibrary()
        if name in p.get_function_names():
            return _fallback_get_expr(p, name)
    raise KeyError(f"Function '{name}' not found in selected libraries: {libraries}")

def list_function_candidates(libraries: List[str]) -> List[str]:
    names: List[str] = []
    if ("Basic" in libraries) and BasicFunctions:
        try:
            names += BasicFunctions().get_function_names()
        except Exception:
            pass
    if ("Special" in libraries) and SpecialFunctions:
        try:
            names += SpecialFunctions().get_function_names()[:50]
        except Exception:
            pass
    if ("Phi" in libraries) and PhiLibrary:
        try:
            names += PhiLibrary().get_function_names()
        except Exception:
            pass
    # ensure unique order-preserving
    seen = set(); uniq = []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    if not uniq:
        uniq = ["id"]  # fallback
    return uniq

@dataclass
class ReverseFit:
    function_name: str
    alpha: float
    beta: float
    n: int
    M: float
    scale_C: float
    rmse: float
    candidate_expr: sp.Expr  # SymPy y_fit(x)

DEFAULT_SEARCH = {
    "alpha": {"min": 0.1, "max": 5.0, "steps": 7, "allow_negative": True},
    "beta":  {"min": 0.5, "max": 3.0, "steps": 6, "allow_negative": False},
    "n":     {"min": 1,   "max": 6,  "steps": 6},   # integer sweep
    "M":     {"min": -4.0,"max": 4.0,"steps": 7}
}

def _grid(values: Dict, integer: bool = False, allow_negative: bool = False) -> List[float]:
    lo, hi, steps = float(values["min"]), float(values["max"]), int(values["steps"])
    steps = max(2, steps)
    arr = np.linspace(lo, hi, steps)
    if allow_negative:
        arr = np.unique(np.concatenate([-arr, arr]))
    return [int(round(v)) for v in arr] if integer else arr.tolist()

def _best_C(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return 0.0, float("inf")
    yt = y_true[mask]; yp = y_pred[mask]
    denom = float(np.dot(yp, yp))
    if denom <= 1e-18:
        return 0.0, float("inf")
    C = float(np.dot(yp, yt) / denom)
    rmse = float(np.sqrt(np.mean((yt - C*yp)**2)))
    return C, rmse

def _lambdify(expr: sp.Expr, x: sp.Symbol):
    try:
        f = sp.lambdify([x], expr, "numpy")
        return lambda xs: np.array(f(xs), dtype=float)
    except Exception:
        return lambda xs: np.full_like(xs, np.nan, dtype=float)

def _phi_block_expr(fname: str, alpha: float, beta: float, n: int, M: float, libraries: List[str]) -> sp.Expr:
    """
    Build a single block: x^M * (f(alpha*x^beta))^n
    """
    x = sp.Symbol("x", real=True)
    fz = get_function_expr_any(libraries, fname)
    arg = sp.Float(alpha) * sp.Pow(x, sp.Float(beta))
    block = sp.Pow(sp.simplify(fz.subs({sp.Symbol("z"): arg})), int(n))
    return sp.simplify(sp.Pow(x, sp.Float(M)) * block)

# ── SINGLE-BLOCK FIT ───────────────────────────────────────────────────────────
def infer_params_from_solution(
    y_expr: sp.Expr,
    libraries: Optional[List[str]] = None,
    search: Optional[Dict] = None,
    x_symbol: Optional[sp.Symbol] = None,
    topk: int = 5,
    sample_range: Tuple[float, float] = (0.25, 2.5),
    samples: int = 200
) -> List[ReverseFit]:
    """
    Fit y(x) ≈ C * x^M * (f(alpha*x^beta))^n, sweeping candidate f and parameters.
    Returns top-k candidates sorted by RMSE (ascending).
    """
    libraries = libraries or ["Basic", "Special", "Phi"]
    search = search or DEFAULT_SEARCH
    x = x_symbol or sp.Symbol("x", real=True)
    y_expr = sp.simplify(y_expr)
    y_num = _lambdify(y_expr, x)

    xs = np.linspace(sample_range[0], sample_range[1], int(samples)).astype(float)
    y_true = y_num(xs)

    cand_names = list_function_candidates(libraries)
    alpha_vals = _grid(search["alpha"], integer=False, allow_negative=bool(search["alpha"].get("allow_negative", False)))
    beta_vals  = _grid(search["beta"],  integer=False, allow_negative=bool(search["beta"].get("allow_negative", False)))
    n_vals     = _grid(search["n"],     integer=True)
    M_vals     = _grid(search["M"],     integer=False)

    results: List[ReverseFit] = []
    eps = 1e-12

    for fname in cand_names:
        # compile numeric f
        try:
            fz = get_function_expr_any(libraries, fname)
            z = sp.Symbol("z", real=True)
            fn = sp.lambdify([z], fz, "numpy")
        except Exception:
            continue

        for n in n_vals:
            if not (1 <= int(n) <= 10):
                continue
            for M in M_vals:
                xM = np.power(xs + eps, float(M))
                for alpha in alpha_vals:
                    for beta in beta_vals:
                        try:
                            arg = float(alpha) * np.power(xs + eps, float(beta))
                            inside = fn(arg)
                            powpart = np.power(inside + eps, int(n))
                            y_pred  = xM * powpart
                            C, rmse = _best_C(y_true, y_pred)
                            if math.isfinite(rmse):
                                # build symbolic candidate expr (for top-k later)
                                results.append(ReverseFit(
                                    function_name=fname, alpha=float(alpha), beta=float(beta),
                                    n=int(n), M=float(M), scale_C=float(C), rmse=float(rmse),
                                    candidate_expr=sp.Symbol("y_fit")
                                ))
                        except Exception:
                            continue

    if not results:
        return []
    results.sort(key=lambda r: r.rmse)
    top = results[:max(1, int(topk))]

    # finalize symbolic y_fit for top-k
    x = sp.Symbol("x", real=True)
    for r in top:
        try:
            block = _phi_block_expr(r.function_name, r.alpha, r.beta, r.n, r.M, libraries)
            r.candidate_expr = sp.simplify(sp.Float(r.scale_C) * block)
        except Exception:
            r.candidate_expr = sp.Symbol("y_fit")
    return top

# ── MULTI‑BLOCK FITS ──────────────────────────────────────────────────────────
@dataclass
class MultiBlockCandidate:
    blocks: List[ReverseFit]      # list of single-block params
    scales: List[float]           # [C] for product, [C_j] for sum
    rmse: float
    expr: sp.Expr

def _design_matrix(blocks_phi: List[np.ndarray]) -> np.ndarray:
    """
    Stack block columns for least squares in 'sum' mode.
    """
    return np.column_stack(blocks_phi)  # [N x J]

def infer_params_multi_blocks(
    y_expr: sp.Expr,
    J: int = 2,
    mode: str = "product",   # "product" or "sum"
    libraries: Optional[List[str]] = None,
    search: Optional[Dict] = None,
    x_symbol: Optional[sp.Symbol] = None,
    topk: int = 5,
    pool_size: int = 20,     # take best K single-blocks, then combine
    max_combos: int = 100,   # cap combinations for tractability
    sample_range: Tuple[float, float] = (0.25, 2.5),
    samples: int = 200
) -> List[MultiBlockCandidate]:
    """
    Fit using multiple blocks.
      - product: y ≈ C * ∏_{j=1}^J block_j(x)
      - sum:     y ≈ Σ_{j=1}^J C_j * block_j(x)

    Strategy:
      1) get top 'pool_size' single-block candidates (fast coarse fit)
      2) form up to 'max_combos' combinations of size J
      3) evaluate and solve outer scales (C or C_j) by least squares
    """
    assert J >= 1
    libraries = libraries or ["Basic", "Special", "Phi"]
    x = x_symbol or sp.Symbol("x", real=True)

    # Step 1: single-block pool
    base = infer_params_from_solution(
        y_expr, libraries=libraries, search=search or DEFAULT_SEARCH, x_symbol=x,
        topk=int(pool_size), sample_range=sample_range, samples=samples
    )
    if not base:
        return []

    # build numeric y_true
    y_num = _lambdify(sp.simplify(y_expr), x)
    xs = np.linspace(sample_range[0], sample_range[1], int(samples)).astype(float)
    y_true = y_num(xs)

    # Precompute numeric versions for each single-block
    eps = 1e-12
    block_num: List[np.ndarray] = []
    for r in base:
        try:
            block = _phi_block_expr(r.function_name, r.alpha, r.beta, r.n, r.M, libraries)
            f = _lambdify(block, x)
            block_num.append(f(xs))
        except Exception:
            block_num.append(np.full_like(xs, np.nan, dtype=float))

    # Choose combinations of blocks
    combos = list(itertools.combinations(range(len(base)), J))
    if len(combos) > max_combos:
        # sample a subset for tractability
        rng = np.random.default_rng(123)
        combos = list(rng.choice(combos, size=max_combos, replace=False))

    out: List[MultiBlockCandidate] = []

    if mode.lower() == "product":
        for idxs in combos:
            try:
                phi = np.ones_like(xs, dtype=float)
                expr_sym = sp.Integer(1)
                blocks: List[ReverseFit] = []
                for k in idxs:
                    r = base[k]; blocks.append(r)
                    phi *= block_num[k]
                    expr_sym = sp.simplify(expr_sym * _phi_block_expr(r.function_name, r.alpha, r.beta, r.n, r.M, libraries))

                C, rmse = _best_C(y_true, phi)
                if math.isfinite(rmse):
                    expr = sp.simplify(sp.Float(C) * expr_sym)
                    out.append(MultiBlockCandidate(blocks=blocks, scales=[C], rmse=rmse, expr=expr))
            except Exception:
                continue

    else:  # "sum" mode
        for idxs in combos:
            try:
                cols = [block_num[k] for k in idxs]
                A = _design_matrix(cols)     # [N x J]
                # Solve A*C = y in least squares
                Cvec, *_ = np.linalg.lstsq(A, y_true, rcond=None)
                y_hat = A @ Cvec
                rmse = float(np.sqrt(np.mean((y_true - y_hat)**2)))

                # build symbolic expr
                expr_sym = sp.Integer(0)
                blocks: List[ReverseFit] = []
                for j, k in enumerate(idxs):
                    r = base[k]; blocks.append(r)
                    expr_sym += sp.Float(Cvec[j]) * _phi_block_expr(r.function_name, r.alpha, r.beta, r.n, r.M, libraries)
                out.append(MultiBlockCandidate(blocks=blocks, scales=[float(c) for c in Cvec], rmse=rmse, expr=sp.simplify(expr_sym)))
            except Exception:
                continue

    if not out:
        return []

    out.sort(key=lambda t: t.rmse)
    return out[:max(1, int(topk))]

# ── ODE → solution → fit ──────────────────────────────────────────────────────
def _freeze_dsolve_constants(expr: sp.Expr) -> sp.Expr:
    syms = list(expr.free_symbols)
    subs = {}
    for s in syms:
        if s.name.startswith("C"):
            idx = int(''.join(ch for ch in s.name if ch.isdigit()) or "1")
            subs[s] = 1.0 if idx == 1 else 0.5
    return sp.simplify(expr.subs(subs)) if subs else expr

def infer_from_ode_single(
    lhs_expr: sp.Expr,
    rhs_expr: sp.Expr,
    libraries: Optional[List[str]] = None,
    search: Optional[Dict] = None,
    x_symbol: Optional[sp.Symbol] = None,
    **kwargs
) -> List[ReverseFit]:
    x = x_symbol or sp.Symbol("x", real=True)
    try:
        sol = sp.dsolve(sp.Eq(lhs_expr, rhs_expr))
        y_expr = sol.rhs if hasattr(sol, "rhs") else (sol if isinstance(sol, sp.Expr) else None)
        if y_expr is None:
            return []
        y_expr = _freeze_dsolve_constants(sp.simplify(y_expr))
        return infer_params_from_solution(y_expr, libraries=libraries, search=search, x_symbol=x, **kwargs)
    except Exception:
        return []

def infer_from_ode_multi(
    lhs_expr: sp.Expr,
    rhs_expr: sp.Expr,
    J: int = 2,
    mode: str = "product",
    libraries: Optional[List[str]] = None,
    search: Optional[Dict] = None,
    x_symbol: Optional[sp.Symbol] = None,
    **kwargs
) -> List[MultiBlockCandidate]:
    x = x_symbol or sp.Symbol("x", real=True)
    try:
        sol = sp.dsolve(sp.Eq(lhs_expr, rhs_expr))
        y_expr = sol.rhs if hasattr(sol, "rhs") else (sol if isinstance(sol, sp.Expr) else None)
        if y_expr is None:
            return []
        y_expr = _freeze_dsolve_constants(sp.simplify(y_expr))
        return infer_params_multi_blocks(y_expr, J=J, mode=mode, libraries=libraries, search=search, x_symbol=x, **kwargs)
    except Exception:
        return []