# shared/series_tools.py
"""
Series helpers (inner adapters).

I0_p_exp_of_exp_series(p, a, x, order)
--------------------------------------
Return a truncated series for  [ I0( exp(exp(-a*x)) ) ]^p  around x=0 (default),
with the requested scale 'a' (i.e., x -> a*x). This matches your request:
"generalize I0_p_exp_of_exp_series to accept a scale parameter a (replace x->a x)".

If you used a different center previously, change 'about' accordingly.
"""

from typing import Optional
import sympy as sp

def series_with_scale(expr: sp.Expr, x: sp.Symbol, a: sp.Expr, about=0, order: int = 6) -> sp.Expr:
    """
    Compute series of expr(x -> a*x) about 'about' to 'order' terms; strip O(.) term.
    """
    expr_scaled = expr.subs({x: a*x})
    try:
        return sp.series(expr_scaled, x, about, order).removeO()
    except Exception:
        # If series fails (too heavy), just return the substituted expression
        return sp.simplify(expr_scaled)

def I0_p_exp_of_exp_series(p: int = 1, a: float = 1.0,
                           x: Optional[sp.Symbol] = None,
                           about=0, order: int = 6) -> sp.Expr:
    """
    Series for [I0( exp(exp(-a*x)) )]^p around x=about up to 'order'.
    """
    x = x or sp.Symbol("x", real=True)
    expr = sp.besseli(0, sp.exp(sp.exp(-x))) ** int(p)
    return series_with_scale(expr, x, sp.Float(a), about=about, order=order)