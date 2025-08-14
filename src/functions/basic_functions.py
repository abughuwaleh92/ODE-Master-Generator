# -*- coding: utf-8 -*-
"""
BasicFunctions
--------------
A comprehensive, symbolic function catalog in a single variable z for "basic"
functions used across the Master Generators app.

API (backwards compatible):
- BasicFunctions().get_function(name: str) -> sympy.Expr   # returns f(z)
- BasicFunctions().get_function_names() -> List[str]       # sorted function keys

Notes:
- Case-insensitive; tolerant to spaces, hyphens, and aliases.
- Returns symbolic expressions of z (SymPy) – no numerics baked in.
"""

from __future__ import annotations
from typing import Dict, List, Callable, Tuple
import sympy as sp


__all__ = ["BasicFunctions"]


def _norm(name: str) -> str:
    """Normalize keys: lower, strip, replace spaces/hyphens with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class BasicFunctions:
    """
    Rich symbolic function registry (non-special functions) in variable z.

    Example
    -------
    >>> lib = BasicFunctions()
    >>> f = lib.get_function("exp")      # exp(z)
    >>> g = lib.get_function("z^3")      # z**3
    >>> names = lib.get_function_names() # view supported keys
    """

    def __init__(self) -> None:
        # Primary symbol
        self.z = sp.Symbol("z", real=True)

        # Fixed (non-parameterized) map: key -> expression in z
        E = self._expr: Dict[str, sp.Expr] = {}

        # Families (parameterized): key_base -> builder(n) -> expr
        # For "basic" we include power family; specials live in SpecialFunctions.
        self._families: Dict[str, Callable[[int], sp.Expr]] = {}

        # -------------- Scalars / Identity / Polynomials -----------------
        E["zero"] = sp.Integer(0)
        E["one"] = sp.Integer(1)
        E["constant_0"] = sp.Integer(0)
        E["constant_1"] = sp.Integer(1)

        # Identity / linear
        E["z"] = self.z
        E["identity"] = self.z
        E["linear"] = self.z

        # Common powers (also exposed as family power(n))
        E["z^2"] = self.z**2
        E["z^3"] = self.z**3
        E["z^4"] = self.z**4
        E["z^5"] = self.z**5
        E["z^6"] = self.z**6
        self._families["power"] = lambda n: self.z**int(n)

        # -------------- Exponential / Logarithms -------------------------
        E["exp"] = sp.exp(self.z)
        E["exp_neg"] = sp.exp(-self.z)
        E["exponential"] = sp.exp(self.z)
        E["log"] = sp.log(self.z)            # ln(z)
        E["ln"] = sp.log(self.z)
        E["log1p"] = sp.log(1 + self.z)      # ln(1+z)

        # -------------- Roots / Abs / Sign / Piecewise -------------------
        E["sqrt"] = sp.sqrt(self.z)
        E["abs"] = sp.Abs(self.z)
        E["sign"] = sp.sign(self.z)
        # Heaviside default SymPy value at 0 is a symbol; acceptable for symbolic work
        E["heaviside"] = sp.Heaviside(self.z)
        E["relu"] = sp.Max(sp.Integer(0), self.z)
        E["softplus"] = sp.log(1 + sp.exp(self.z))

        # -------------- Trigonometric -----------------------------------
        E["sin"] = sp.sin(self.z)
        E["cos"] = sp.cos(self.z)
        E["tan"] = sp.tan(self.z)
        E["cot"] = sp.cot(self.z)
        E["sec"] = sp.sec(self.z)
        E["csc"] = sp.csc(self.z)

        # Inverse trig
        E["asin"] = sp.asin(self.z)
        E["acos"] = sp.acos(self.z)
        E["atan"] = sp.atan(self.z)

        # -------------- Hyperbolic --------------------------------------
        E["sinh"] = sp.sinh(self.z)
        E["cosh"] = sp.cosh(self.z)
        E["tanh"] = sp.tanh(self.z)
        E["coth"] = sp.coth(self.z)
        E["sech"] = sp.sech(self.z)
        E["csch"] = sp.csch(self.z)

        # -------------- Common analysis functions ------------------------
        # NOTE: SymPy's sp.sinc(x) = sin(pi*x)/(pi*x). Provide both variants.
        E["sinc"] = sp.sinc(self.z)                    # normalized sinc
        E["sinc_unscaled"] = sp.sin(self.z) / self.z   # sin(z)/z
        E["gaussian"] = sp.exp(-(self.z**2))
        E["erf"] = sp.erf(self.z)
        E["erfc"] = sp.erfc(self.z)
        E["gamma"] = sp.gamma(self.z)
        E["loggamma"] = sp.loggamma(self.z)
        E["ei"] = sp.Ei(self.z)   # Exponential integral
        E["si"] = sp.Si(self.z)   # Sine integral
        E["ci"] = sp.Ci(self.z)   # Cosine integral

        # Aliases (common names users may try)
        self._aliases: Dict[str, str] = {
            "const0": "constant_0",
            "const1": "constant_1",
            "identity": "z",
            "quadratic": "z^2",
            "cubic": "z^3",
            "quartic": "z^4",
            "expz": "exp",
            "lnz": "log",
            "absolute": "abs",
            "unit_step": "heaviside",
            "heaviside_step": "heaviside",
            "relu_z": "relu",
            "sigmoid": "1_over_1_plus_exp_minus_z",
        }

        # Provide a canonical sigmoid name (kept out of aliases map as it's an expression)
        E["1_over_1_plus_exp_minus_z"] = sp.Integer(1) / (sp.Integer(1) + sp.exp(-self.z))

        # Build reverse index with normalized keys
        self._index: Dict[str, str] = {}
        for k in list(E.keys()):
            self._index[_norm(k)] = k
        for alias, target in self._aliases.items():
            if target in E:
                self._index[_norm(alias)] = target

    # --------------------------- Public API ------------------------------

    def get_function(self, name: str) -> sp.Expr:
        """
        Return a SymPy expression f(z) for the requested function name.

        Supports:
        - direct keys: "exp", "z^3", "sin", "log1p", ...
        - aliases: "quartic" -> "z^4", ...
        - family syntax for powers: "power(7)" -> z**7, "power7" -> z**7

        Raises:
            KeyError if name not found/parsable.
        """
        if not isinstance(name, str):
            raise KeyError("Function name must be a string.")

        key = _norm(name)

        # 1) Exact / alias match:
        if key in self._index:
            base = self._index[key]
            return self._expr[base]

        # 2) Power family parsing: "power(5)", "power5", or "z^5"
        expr = self._try_parse_power_family(key)
        if expr is not None:
            return expr

        # 3) If user typed raw "z^N" (covered already) or "z**N"
        if key.startswith("z**"):
            try:
                n = int(key[3:])
                return self.z ** n
            except Exception:
                pass

        raise KeyError(f"Unknown basic function '{name}'. Try one of: {', '.join(self.get_function_names()[:20])} ...")

    def get_function_names(self) -> List[str]:
        """
        Return a sorted list of supported (canonical) function keys.
        Includes sample entries for the power family.
        """
        names = sorted(set(self._expr.keys()))
        # add representative power() entries
        names += [f"power({n})" for n in (2, 3, 4, 5, 6, 7)]
        return sorted(names)

    # --------------------------- Helpers --------------------------------

    def _try_parse_power_family(self, key: str) -> sp.Expr | None:
        """
        Parse power family notations:
            - "power(7)"  -> z**7
            - "power7"    -> z**7
            - "z^7"       -> z**7 (already covered in fixed map but keep for safety)
        """
        if key.startswith("power(") and key.endswith(")"):
            inner = key[len("power("):-1]
            try:
                n = int(inner)
                return self.z ** n
            except Exception:
                return None

        if key.startswith("power") and len(key) > len("power"):
            maybe_n = key[len("power"):]
            try:
                n = int(maybe_n)
                return self.z ** n
            except Exception:
                return None

        if key.startswith("z^"):
            try:
                n = int(key[2:])
                return self.z ** n
            except Exception:
                return None

        return None
