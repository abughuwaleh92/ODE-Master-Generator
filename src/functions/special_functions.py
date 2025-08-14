# -*- coding: utf-8 -*-
"""
SpecialFunctions
----------------
A comprehensive, symbolic function catalog in variable z for special functions:
Airy, Bessel (J/Y/I/K), spherical Bessel, Struve, Fresnel, Legendre, Chebyshev,
Hermite, Laguerre, and more.

API (backwards compatible):
- SpecialFunctions().get_function(name: str) -> sympy.Expr   # returns f(z)
- SpecialFunctions().get_function_names() -> List[str]       # sorted function keys

Features:
- Case-insensitive; tolerant to spaces/hyphens.
- Parameterized families supported with two forms:
    "bessel_j(2)"  or  "bessel_j2"
    "legendre_p(5)" or "legendre_p5"
- Common “concrete” names provided for convenience:
    "airy_ai", "airy_bi", "bessel_j0", "bessel_j1", ..., "bessel_y0", ...
"""

from __future__ import annotations
from typing import Dict, List, Callable
import sympy as sp


__all__ = ["SpecialFunctions"]


def _norm(name: str) -> str:
    """Normalize keys: lower, strip, replace spaces/hyphens with underscores."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class SpecialFunctions:
    """
    Rich symbolic special-function registry in variable z.

    Example
    -------
    >>> lib = SpecialFunctions()
    >>> f = lib.get_function("airy_ai")     # Ai(z)
    >>> g = lib.get_function("bessel_j(2)") # J_2(z)
    >>> names = lib.get_function_names()
    """

    def __init__(self) -> None:
        self.z = sp.Symbol("z", real=True)

        # Fixed (non-parameterized) map: key -> expr
        E = self._expr: Dict[str, sp.Expr] = {}

        # Families: key_base -> builder(n) -> expr
        self._families: Dict[str, Callable[[int], sp.Expr]] = {}

        # -------------------- Airy --------------------
        E["airy_ai"] = sp.airyai(self.z)
        E["airy_bi"] = sp.airybi(self.z)

        # -------------------- Bessel (cylindrical) ----
        self._families["bessel_j"] = lambda n: sp.besselj(int(n), self.z)
        self._families["bessel_y"] = lambda n: sp.bessely(int(n), self.z)
        self._families["bessel_i"] = lambda n: sp.besseli(int(n), self.z)
        self._families["bessel_k"] = lambda n: sp.besselk(int(n), self.z)

        # Provide common concrete entries J0..J3, Y0..Y3, I0..I3, K0..K3
        for fam in ("bessel_j", "bessel_y", "bessel_i", "bessel_k"):
            for n in range(0, 4):
                key = f"{fam}{n}"
                E[key] = self._families[fam](n)

        # -------------------- Spherical Bessel --------
        # j_n(z), y_n(z)
        self._families["spherical_bessel_j"] = lambda n: sp.spherical_bessel_j(int(n), self.z)
        self._families["spherical_bessel_y"] = lambda n: sp.spherical_bessel_y(int(n), self.z)
        for fam in ("spherical_bessel_j", "spherical_bessel_y"):
            for n in range(0, 4):
                key = f"{fam}{n}"
                E[key] = self._families[fam](n)

        # -------------------- Struve ------------------
        # H_n(z) and L_n(z)
        self._families["struve_h"] = lambda n: sp.struveh(int(n), self.z)
        self._families["struve_l"] = lambda n: sp.struvel(int(n), self.z)
        for fam in ("struve_h", "struve_l"):
            for n in range(0, 3):
                E[f"{fam}{n}"] = self._families[fam](n)

        # -------------------- Fresnel -----------------
        E["fresnel_s"] = sp.fresnels(self.z)
        E["fresnel_c"] = sp.fresnelc(self.z)

        # -------------------- Legendre / Chebyshev / Orthogonal polys ----
        self._families["legendre_p"] = lambda n: sp.legendre(int(n), self.z)   # P_n(z)
        self._families["chebyshev_t"] = lambda n: sp.chebyshevt(int(n), self.z)
        self._families["chebyshev_u"] = lambda n: sp.chebyshevu(int(n), self.z)
        self._families["hermite"] = lambda n: sp.hermite(int(n), self.z)
        self._families["laguerre"] = lambda n: sp.laguerre(int(n), self.z)

        for fam in ("legendre_p", "chebyshev_t", "chebyshev_u", "hermite", "laguerre"):
            for n in range(0, 6):
                E[f"{fam}{n}"] = self._families[fam](n)

        # -------------------- Other common specials ----------------------
        E["lambert_w"] = sp.LambertW(self.z)
        E["polylog_2"] = sp.polylog(2, self.z)      # Li_2(z)
        E["dilog"] = sp.polylog(2, self.z)          # alias
        E["zeta"] = sp.zeta(self.z)

        # Hypergeometric examples (fixed parameters for convenience)
        # 2F1(a,b;c;z) with (a,b,c) = (1/2,1;3/2)
        E["hyp2f1_demo"] = sp.hyper([sp.Rational(1, 2), 1], [sp.Rational(3, 2)], self.z)
        # 1F1(a;c;z) with (a,c) = (1;2)
        E["hyp1f1_demo"] = sp.hyper([1], [2], self.z)
        # U(a,b,z) with (a,b)=(1,2)
        E["hyperu_demo"] = sp.hyperu(1, 2, self.z)

        # Aliases users may try
        self._aliases: Dict[str, str] = {
            "airy_ai_z": "airy_ai",
            "airy_bi_z": "airy_bi",
            "besselj0": "bessel_j0",
            "besselj1": "bessel_j1",
            "bessely0": "bessel_y0",
            "bessely1": "bessel_y1",
            "besseli0": "bessel_i0",
            "besselk0": "bessel_k0",
            "legendre": "legendre_p1",
            "chebyshev_t": "chebyshev_t1",
            "chebyshev_u": "chebyshev_u1",
            "hermite_h": "hermite1",
            "laguerre_l": "laguerre1",
            "li2": "polylog_2",
        }

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
        Return a SymPy expression f(z) for the requested special function.

        Supports:
        - concrete keys: "airy_ai", "bessel_j0", "legendre_p3", "fresnel_s", ...
        - aliases: "besselj0" -> "bessel_j0", ...
        - family syntax: "bessel_j(2)" or "bessel_j2"
                         "legendre_p(5)" or "legendre_p5"
                         "hermite(3)" etc.

        Raises:
            KeyError if name not found/parsable.
        """
        if not isinstance(name, str):
            raise KeyError("Function name must be a string.")
        key = _norm(name)

        # 1) Exact / alias match
        if key in self._index:
            base = self._index[key]
            return self._expr[base]

        # 2) Family syntaxes: foo(n) / fooN
        fam_expr = self._try_parse_family(key)
        if fam_expr is not None:
            return fam_expr

        raise KeyError(
            f"Unknown special function '{name}'. Try one of: {', '.join(self.get_function_names()[:20])} ..."
        )

    def get_function_names(self) -> List[str]:
        """
        Return a sorted list of supported (canonical) function keys.
        Includes representative entries for family functions.
        """
        names = sorted(set(self._expr.keys()))
        # Advertise common family entries
        demo = [
            "bessel_j(0)", "bessel_j(1)", "bessel_i(0)", "bessel_k(0)",
            "spherical_bessel_j(0)", "spherical_bessel_y(0)",
            "legendre_p(3)", "chebyshev_t(4)", "hermite(5)", "laguerre(2)",
        ]
        names.extend(demo)
        return sorted(names)

    # ---------------------------- Helpers --------------------------------

    def _try_parse_family(self, key: str) -> sp.Expr | None:
        """
        Parse names like:
            'bessel_j(2)'  -> family 'bessel_j', n=2
            'bessel_j2'    -> family 'bessel_j', n=2
        Recognized families: bessel_j/y/i/k, spherical_bessel_j/y,
                             struve_h/l, legendre_p, chebyshev_t/u,
                             hermite, laguerre.
        """
        # (...) form
        if "(" in key and key.endswith(")"):
            fam = key[: key.index("(")]
            inner = key[key.index("(") + 1 : -1]
            if fam in self._families:
                try:
                    n = int(inner)
                    return self._families[fam](n)
                except Exception:
                    return None

        # trailing integer form
        for fam in self._families.keys():
            if key.startswith(fam):
                tail = key[len(fam):]
                if tail != "" and tail.lstrip("-").isdigit():
                    try:
                        n = int(tail)
                        return self._families[fam](n)
                    except Exception:
                        return None

        return None
