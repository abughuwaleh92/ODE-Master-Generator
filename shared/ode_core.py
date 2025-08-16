# shared/ode_core.py
# Core math engine: robust Theorem 4.1/4.2, active LHS application, ICs, free-form parsing.

from __future__ import annotations
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy as sp
from sympy.core.function import AppliedUndef

# --------------------------- utilities ---------------------------

def simplify_expr(expr: sp.Expr, level: str = "light") -> sp.Expr:
    if level == "none":
        return expr
    try:
        e = sp.together(expr)
        e = sp.cancel(e)
        e = sp.simplify(e)
        if level == "aggressive":
            try:
                e = sp.nsimplify(e, [sp.E, sp.pi, sp.I], rational=True, maxsteps=50)
            except Exception:
                pass
        return e
    except Exception:
        return expr

def to_exact(v):
    try:
        return sp.nsimplify(v, rational=True)
    except Exception:
        return sp.sympify(v)

@lru_cache(maxsize=256)
def omega_list(n: int):
    return tuple(sp.Rational(2*s-1, 2*n)*sp.pi for s in range(1, int(n)+1))

@lru_cache(maxsize=256)
def stirling_row(m: int):
    S = sp.functions.combinatorial.numbers.stirling
    return tuple(S(int(m), j, kind=2) for j in range(1, int(m)+1))

# --------------------------- f(z) resolver ---------------------------

def get_function_expr(source_lib, func_name: str) -> sp.Expr:
    z = sp.Symbol("z", real=True)
    f_obj = None
    if source_lib is not None:
        try:
            f_obj = source_lib.get_function(func_name)
        except Exception:
            f_obj = None

    if f_obj is None:
        # Accept raw SymPy expression like "exp(z) + z**2"
        return sp.sympify(func_name, locals={"z": z, "E": sp.E, "pi": sp.pi})

    if isinstance(f_obj, sp.Expr):
        frees = list(f_obj.free_symbols)
        if not frees:
            return sp.sympify(f_obj)
        g = f_obj
        for s in frees:
            if s != z:
                g = g.subs(s, z)
        return g

    if callable(f_obj):
        try:
            v = f_obj(z)
            if isinstance(v, sp.Expr):
                return v
        except Exception:
            pass

    if isinstance(f_obj, str):
        return sp.sympify(f_obj, locals={"z": z})

    raise TypeError(f"Unsupported function object for '{func_name}'")

# --------------------------- Theorem 4.1 ---------------------------

def theorem_4_1_solution_expr(
    f_expr: sp.Expr, alpha, beta, n: int, M, x: sp.Symbol, simplify_level="light"
) -> sp.Expr:
    """
    y(x) = (π/2n) Σ_s [ 2 f(α+β) - f(α + β ζ_s(x)) - f(α + β \overline{ζ_s(x)}) ] + π M
    where ζ_s(x) = e^{-x sin ω_s} e^{i x cos ω_s}, ω_s = (2s-1)π/(2n)
    """
    z = sp.Symbol("z", real=True)
    base = f_expr.subs(z, alpha + beta)
    terms = []
    for ω in omega_list(int(n)):
        exp_pos = sp.exp(sp.I*x*sp.cos(ω) - x*sp.sin(ω))
        exp_neg = sp.exp(-sp.I*x*sp.cos(ω) - x*sp.sin(ω))
        psi = f_expr.subs(z, alpha + beta*exp_pos)
        phi = f_expr.subs(z, alpha + beta*exp_neg)
        terms.append(2*base - (psi + phi))
    y = sp.pi/(2*n) * sp.Add(*terms) + sp.pi*M
    return simplify_expr(y, level=simplify_level)

# --------------------------- Theorem 4.2 (Stirling) ---------------------------

def theorem_4_2_y_m_expr(
    f_expr: sp.Expr, alpha_value, beta, n: int, m: int, x: sp.Symbol, simplify_level="light"
) -> sp.Expr:
    """
    Compact Stirling-number form described in your note. Matches your odd/even cases.
    """
    z = sp.Symbol("z", real=True)
    αsym = sp.Symbol("alpha_sym", real=True)
    total = 0
    Srow = stirling_row(int(m))

    for ω in omega_list(int(n)):
        lam = sp.exp(sp.I*(sp.pi/2 + ω))
        zeta = sp.exp(-x*sp.sin(ω)) * sp.exp(sp.I*x*sp.cos(ω))
        zetab = sp.conjugate(zeta)

        psi = f_expr.subs(z, αsym + beta*zeta)
        phi = f_expr.subs(z, αsym + beta*zetab)

        s1 = 0
        s2 = 0
        for j, S in enumerate(Srow, start=1):
            s1 += S * (beta*zeta)**j  * sp.diff(psi, αsym, j)
            s2 += S * (beta*zetab)**j * sp.diff(phi, αsym, j)

        total += lam**m * s1 + sp.conjugate(lam)**m * s2

    y_m = -sp.pi/(2*n) * total
    y_m = y_m.subs(αsym, alpha_value)
    return simplify_expr(y_m, level=simplify_level)

# --------------------------- LHS application (with chain rule & delay) ---------------------------

def apply_lhs_to_solution(
    lhs_expr: sp.Expr, solution_y: sp.Expr, x: sp.Symbol, y_name: str = "y",
    simplify_level="light", max_unique_derivs: int = 100
) -> sp.Expr:
    """
    Replace y(·) and derivatives with the closed-form 'solution_y' with
    full chain rule support for y(g(x)) by substituting x→g(x) first then differentiating.
    """
    subs_map = {}
    yfun = sp.Function(y_name)

    needs = []
    seen = set()
    # collect y(·) atoms and Derivative(y(·), x, k) atoms
    for node in lhs_expr.atoms(AppliedUndef, sp.Derivative):
        if isinstance(node, AppliedUndef) and node.func == yfun and len(node.args) == 1:
            arg = node.args[0]
            key = ("y", sp.srepr(arg), 0)
            if key not in seen:
                needs.append((node, arg, 0, False))
                seen.add(key)
        elif isinstance(node, sp.Derivative):
            base = node.expr
            if isinstance(base, AppliedUndef) and base.func == yfun and len(base.args) == 1:
                arg = base.args[0]
                try:
                    # modern SymPy (variable_count gives (sym, count)) or fallback
                    order = sum(c for v, c in node.variable_count if v == x)
                except Exception:
                    order = sum(1 for v in node.variables if v == x)
                key = ("dy", sp.srepr(arg), int(order))
                if key not in seen:
                    needs.append((node, arg, int(order), True))
                    seen.add(key)

    if len(needs) > max_unique_derivs:
        raise RuntimeError(f"Too many distinct derivatives requested ({len(needs)} > {max_unique_derivs}).")

    # fast path when arg == x
    max_order_x = max((o for _, a, o, isd in needs if isd and a == x), default=0)
    d_cache = {0: solution_y}
    for k in range(1, max_order_x+1):
        d_cache[k] = sp.diff(solution_y, (x, k))

    local_cache = {}
    def diff_after_sub(arg_expr: sp.Expr, order: int) -> sp.Expr:
        """
        Compute d^order/dx^order [ solution_y(x) | x→arg_expr(x) ] .
        This yields correct chain rule behavior for delays/advances/affine reparametrizations.
        """
        key = (sp.srepr(arg_expr), order)
        if key in local_cache: return local_cache[key]
        val = solution_y.subs(x, arg_expr)
        if order > 0:
            val = sp.diff(val, (x, order))
        local_cache[key] = val
        return val

    for node, arg, order, is_deriv in needs:
        if arg == x:
            subs_map[node] = d_cache[order]
        else:
            subs_map[node] = diff_after_sub(arg, order)

    try:
        out = lhs_expr.xreplace(subs_map)
    except Exception:
        out = lhs_expr.subs(subs_map)
    return simplify_expr(out, level=simplify_level)

# --------------------------- Free-form LHS (builder + arbitrary text) ---------------------------

_EXTRA_WRAPPERS = {
    "id": lambda u, eps: u,
    "exp": lambda u, eps: sp.exp(u),
    "sin": lambda u, eps: sp.sin(u),
    "cos": lambda u, eps: sp.cos(u),
    "tan": lambda u, eps: sp.tan(u),
    "sinh": lambda u, eps: sp.sinh(u),
    "cosh": lambda u, eps: sp.cosh(u),
    "tanh": lambda u, eps: sp.tanh(u),
    "log": lambda u, eps: sp.log(eps + sp.Abs(u)),
    "abs": lambda u, eps: sp.Abs(u),
    "asin": lambda u, eps: sp.asin(u),
    "acos": lambda u, eps: sp.acos(u),
    "atan": lambda u, eps: sp.atan(u),
    "asinh": lambda u, eps: sp.asinh(u),
    "acosh": lambda u, eps: sp.acosh(u),
    "atanh": lambda u, eps: sp.atanh(u),
    "erf": lambda u, eps: sp.erf(u),
    "erfc": lambda u, eps: sp.erfc(u),
}

def build_freeform_term(
    x: sp.Symbol, coef=1, inner_order=0, wrapper="id", power=1,
    arg_scale=None, arg_shift=None, outer_order=0,
    y_name="y", ln_eps: sp.Symbol = sp.Symbol("epsilon", positive=True)
) -> sp.Expr:
    yfun = sp.Function(y_name)
    arg = x if arg_scale in (None, 0) and arg_shift in (None, 0) else (x/(arg_scale or 1) + (arg_shift or 0))
    base = yfun(arg)
    if inner_order > 0:
        base = sp.diff(base, (x, int(inner_order)))
    wrap = _EXTRA_WRAPPERS.get(str(wrapper).lower(), _EXTRA_WRAPPERS["id"])
    core = wrap(base, ln_eps)
    term = sp.Integer(1)*coef * (core**power)
    if outer_order > 0:
        term = sp.diff(term, (x, int(outer_order)))
    return term

def build_freeform_lhs(x: sp.Symbol, terms: List[dict], y_name="y") -> sp.Expr:
    if not terms:
        return sp.Symbol("LHS")
    return sp.Add(*[build_freeform_term(x, **t, y_name=y_name) for t in terms])

def parse_arbitrary_lhs(expr_text: str, y_name="y") -> sp.Expr:
    """
    Accept arbitrary SymPy expression for LHS, e.g.:
      "sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1)"
    Safe locals expose x, y, Derivative, functions, etc.
    """
    x = sp.Symbol("x", real=True)
    y = sp.Function(y_name)
    safe = { "x": x, y_name: y, "y": y, "Derivative": sp.Derivative, "diff": sp.diff,
             "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
             "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "Abs": sp.Abs,
             "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "asinh": sp.asinh,
             "acosh": sp.acosh, "atanh": sp.atanh, "erf": sp.erf, "erfc": sp.erfc,
             "E": sp.E, "pi": sp.pi, "I": sp.I }
    return sp.sympify(expr_text, locals=safe)

# --------------------------- type/order inference and ICs ---------------------------

def infer_type_from_lhs(lhs: sp.Expr, y_name="y") -> str:
    # if polynomial in y and its derivatives of degree 1 -> linear
    # rough heuristic: check powers of y(·) and Derivative(y,*)
    yfun = sp.Function(y_name)
    poly = sp.Poly
    try:
        # try to detect nonlinearity by power >1 or wrapped nonlinearity
        for a in lhs.atoms(yfun, sp.Derivative):
            p = lhs.as_poly(a)
            if p is not None and p.total_degree() > 1:
                return "nonlinear"
        # look for nonlinear functions of y(·)
        nonlin_funcs = (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sinh, sp.cosh, sp.tanh,
                        sp.asin, sp.acos, sp.atan, sp.asinh, sp.acosh, sp.atanh, sp.Abs)
        for f in lhs.atoms(*nonlin_funcs):
            for arg in f.args:
                if arg.has(yfun):
                    return "nonlinear"
        return "linear"
    except Exception:
        return "nonlinear"

def infer_order_from_lhs(lhs: sp.Expr, x: sp.Symbol, y_name="y") -> int:
    max_order = 0
    yfun = sp.Function(y_name)
    for node in lhs.atoms(sp.Derivative):
        base = node.expr
        if isinstance(base, AppliedUndef) and base.func == yfun and len(base.args) == 1:
            try:
                order = sum(c for v, c in node.variable_count if v == x)
            except Exception:
                order = sum(1 for v in node.variables if v == x)
            max_order = max(max_order, int(order))
    return max_order

def compute_initial_conditions(solution_y: sp.Expr, max_order: int, x: sp.Symbol) -> Dict[str, Any]:
    """
    Return { 'y(0)': val, 'y\'(0)': val, ..., up to order-1 }
    """
    ics: Dict[str, Any] = {}
    try:
        y0 = simplify_expr(solution_y.subs(x, 0), level="light")
        ics["y(0)"] = y0
    except Exception:
        pass
    for k in range(1, max(0, int(max_order)) + 1):
        try:
            val = sp.diff(solution_y, (x, k)).subs(x, 0)
            ics[f"y^{k}(0)"] = simplify_expr(val, level="light")
        except Exception:
            break
    return ics

# --------------------------- Orchestrator used by UI/Worker ---------------------------

@dataclass
class ComputeParams:
    func_name: str
    alpha: Union[float, int, sp.Expr]
    beta: Union[float, int, sp.Expr]
    n: int
    M: Union[float, int, sp.Expr]
    use_exact: bool = True
    simplify_level: str = "light"
    # LHS choice
    lhs_source: str = "constructor"  # "constructor"|"freeform"|"arbitrary"
    constructor_lhs: Optional[sp.Expr] = None
    freeform_terms: Optional[List[dict]] = None
    arbitrary_lhs_text: Optional[str] = None
    # libraries
    basic_lib: Any = None
    special_lib: Any = None
    # meta
    function_library: str = "Basic"  # or "Special"

def compute_ode_full(p: ComputeParams) -> Dict[str, Any]:
    """
    1) Resolve f(z)
    2) Build y(x) using Theorem 4.1 with exact or float parameters
    3) Determine LHS (active)
    4) RHS = LHS[y] (identity if none ⇒ RHS = y)
    5) Order/type from LHS, ICs up to order-1
    """
    x = sp.Symbol("x", real=True)
    # 1) f(z)
    source_lib = p.basic_lib if p.function_library == "Basic" else p.special_lib
    f_expr = get_function_expr(source_lib, p.func_name)

    # 2) exact/floats
    α = to_exact(p.alpha) if p.use_exact else sp.Float(p.alpha)
    β = to_exact(p.beta)  if p.use_exact else sp.Float(p.beta)
    𝑀 = to_exact(p.M)     if p.use_exact else sp.Float(p.M)

    # y(x)
    y_expr = theorem_4_1_solution_expr(f_expr, α, β, int(p.n), 𝑀, x, p.simplify_level)

    # 3) Active LHS
    lhs = None
    if p.lhs_source == "constructor" and p.constructor_lhs is not None:
        lhs = p.constructor_lhs
    elif p.lhs_source == "freeform" and p.freeform_terms:
        lhs = build_freeform_lhs(x, p.freeform_terms)
    elif p.lhs_source == "arbitrary" and p.arbitrary_lhs_text:
        lhs = parse_arbitrary_lhs(p.arbitrary_lhs_text)
    else:
        lhs = sp.Function("L")(sp.Function("y")(x))  # placeholder

    # If no meaningful LHS, default to identity L[y]=y(x) so RHS is solution
    identity = False
    if lhs == sp.Symbol("LHS") or str(lhs).startswith("L("):
        lhs = sp.Function("y")(x)
        identity = True

    # 4) RHS
    rhs = apply_lhs_to_solution(lhs, y_expr, x, y_name="y", simplify_level=p.simplify_level)

    # 5) order/type/ICs
    ode_order = infer_order_from_lhs(lhs, x, "y")
    ode_type  = infer_type_from_lhs(lhs, "y")
    ics = compute_initial_conditions(y_expr, ode_order, x)

    result = {
        "generator": lhs,
        "rhs": rhs,
        "solution": y_expr,
        "parameters": {"alpha": α, "beta": β, "n": int(p.n), "M": 𝑀},
        "function_used": str(p.func_name),
        "type": ode_type,
        "order": int(ode_order),
        "initial_conditions": ics,
        "lhs_is_identity": identity,
        "f_expr_preview": f_expr,
    }
    return result

# --------------------------- small helpers for JSON export ---------------------------

def expr_to_str(obj: Any) -> Any:
    try:
        if isinstance(obj, sp.Basic):
            return str(obj)
        return obj
    except Exception:
        return obj
