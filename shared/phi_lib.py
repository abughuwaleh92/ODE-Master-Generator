# shared/phi_lib.py
import sympy as sp

x = sp.Symbol('x', real=True)
u = sp.Symbol('u', real=True)

def phi(u_expr, name: str):
    name = (name or "").lower()
    if name in ("exp",):
        return sp.exp(u_expr)
    if name in ("sin",):
        return sp.sin(u_expr)
    if name in ("cos",):
        return sp.cos(u_expr)
    if name in ("tan",):
        return sp.tan(u_expr)
    if name in ("sinh",):
        return sp.sinh(u_expr)
    if name in ("cosh",):
        return sp.cosh(u_expr)
    if name in ("tanh",):
        return sp.tanh(u_expr)
    if name in ("log", "ln"):
        return sp.log(u_expr)
    if name in ("erf",):
        return sp.erf(u_expr)
    if name in ("erfc",):
        return sp.erfc(u_expr)
    if name in ("logistic", "sigmoid"):
        return 1/(1 + sp.exp(-u_expr))
    # default
    return sp.exp(u_expr)

def phi_prime(u_expr, name: str):
    """Return d/du phi(u). Chain rule handled by caller if needed."""
    name = (name or "").lower()
    if name in ("exp",):
        return sp.exp(u_expr)          # phi'(u)=exp(u)
    if name in ("sin",):
        return sp.cos(u_expr)
    if name in ("cos",):
        return -sp.sin(u_expr)
    if name in ("tan",):
        return 1/sp.cos(u_expr)**2
    if name in ("sinh",):
        return sp.cosh(u_expr)
    if name in ("cosh",):
        return sp.sinh(u_expr)
    if name in ("tanh",):
        return 1/sp.cosh(u_expr)**2
    if name in ("log", "ln"):
        return 1/u_expr
    if name in ("erf",):
        return 2/sp.sqrt(sp.pi) * sp.exp(-u_expr**2)
    if name in ("erfc",):
        return -2/sp.sqrt(sp.pi) * sp.exp(-u_expr**2)
    if name in ("logistic", "sigmoid"):
        s = 1/(1 + sp.exp(-u_expr))
        return s*(1-s)
    return sp.exp(u_expr)