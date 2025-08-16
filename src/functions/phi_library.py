# src/functions/phi_library.py
"""
PhiLibrary — a small, explicit registry of φ(u) and its derivative σ(u)=φ'(u).
Integrates with your ecosystem the same way BasicFunctions/SpecialFunctions do:
- get_function_names()
- get_function(name) -> SymPy expression in 'z' (for consistency with other libs)
- get_phi(name)  -> callable returning SymPy expr φ(u)
- get_sig(name)  -> callable returning SymPy expr σ(u)

Supported names (case-sensitive):
  id, exp, sin, cos, tanh, sinh, cosh, logistic, erf, log

Notes:
- 'logistic(u) = 1/(1 + exp(-u))' with σ(u) = logistic(u)*(1 - logistic(u))
- 'log' uses a small ε>0 to avoid singularities in derivative.
"""

from typing import Callable, Dict, List, Tuple
import sympy as sp

class PhiLibrary:
    def __init__(self):
        u = sp.Symbol("u", real=True)
        eps = sp.Symbol("epsilon", positive=True)  # safeguard (e.g., for log)
        logistic = lambda x: 1/(1 + sp.exp(-x))

        self._phi: Dict[str, Callable[[sp.Expr], sp.Expr]] = {
            "id":       (lambda x: x),
            "exp":      sp.exp,
            "sin":      sp.sin,
            "cos":      sp.cos,
            "tanh":     sp.tanh,
            "sinh":     sp.sinh,
            "cosh":     sp.cosh,
            "logistic": logistic,
            "erf":      sp.erf,
            "log":      (lambda x: sp.log(eps + sp.Abs(x))),
        }
        self._sig: Dict[str, Callable[[sp.Expr], sp.Expr]] = {
            "id":       (lambda x: 1),
            "exp":      sp.exp,
            "sin":      sp.cos,
            "cos":      (lambda x: -sp.sin(x)),
            "tanh":     (lambda x: sp.sech(x)**2),
            "sinh":     sp.cosh,
            "cosh":     sp.sinh,
            "logistic": (lambda x: logistic(x) * (1 - logistic(x))),
            "erf":      (lambda x: 2/sp.sqrt(sp.pi) * sp.exp(-x**2)),
            "log":      (lambda x: 1/(sp.Symbol("epsilon", positive=True) + sp.Abs(x))),
        }

    # ── API similar to Basic/Special ────────────────────────────────────────────
    def get_function_names(self) -> List[str]:
        return list(self._phi.keys())

    def get_function(self, name: str):
        """
        Return a SymPy expression in the symbol 'z' consistent with other libs.
        Example: for 'exp', returns exp(z), for 'logistic', returns 1/(1+exp(-z))
        """
        if name not in self._phi:
            raise KeyError(f"Unknown phi function '{name}'")
        z = sp.Symbol("z", real=True)
        return self._phi[name](z)

    def get_phi(self, name: str) -> Callable[[sp.Expr], sp.Expr]:
        if name not in self._phi:
            raise KeyError(f"Unknown phi function '{name}'")
        return self._phi[name]

    def get_sig(self, name: str) -> Callable[[sp.Expr], sp.Expr]:
        if name not in self._sig:
            raise KeyError(f"Unknown phi derivative for '{name}'")
        return self._sig[name]