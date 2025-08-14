# core_master_generators.py
# Shared core for Master Generators UI/API.
# Implements Theorem 4.2 (general form), generator composition, and ML/DL wrappers.

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import sympy as sp
from sympy import Eq
from sympy.functions.combinatorial.numbers import stirling  # stirling(n,k,kind=2) for Stirling S(n,k)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _sympy_ns() -> Dict[str, Any]:
    """A safe SymPy namespace for sympify, including common functions."""
    ns = {name: getattr(sp, name) for name in dir(sp) if not name.startswith("_")}
    # Short aliases commonly used by users
    ns.update({
        "I": sp.I, "pi": sp.pi, "E": sp.E,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "re": sp.re, "im": sp.im,
    })
    return ns


def safe_eval_f_of_z(f_str: str) -> Callable[[sp.Expr], sp.Expr]:
    """
    Returns a symbolic callable f(arg) from a string f_str using a variable 'z'.
    Example inputs: "sin(z)", "exp(z)", "log(1+z)", "cosh(z**2-3*z)".
    """
    z = sp.Symbol('z')
    try:
        f_expr = sp.sympify(f_str, locals={**_sympy_ns(), "z": z})
    except Exception as e:
        raise ValueError(f"Cannot parse f(z) from string: {f_str!r}. Error: {e}")

    def f(arg: sp.Expr) -> sp.Expr:
        return sp.simplify(f_expr.xreplace({z: arg}))

    return f


def count_symbolic_complexity(expr: sp.Expr) -> Dict[str, int]:
    """A few cheap metrics to help rank/rate complexity."""
    atoms = list(expr.atoms())
    ops = sum(1 for _ in sp.preorder_traversal(expr))
    depth = expr.count_ops(visual=True)
    return {
        "n_atoms": len(atoms),
        "n_preorder_nodes": ops,
        "count_ops_visual": depth,
    }


def ode_to_json(lhs: sp.Expr, rhs: sp.Expr, meta: Optional[Dict[str, Any]] = None) -> str:
    """Serialize an ODE LHS=RHS with metadata (LaTeX and a SymPy string form)."""
    payload = {
        "lhs_latex": sp.latex(lhs),
        "rhs_latex": sp.latex(rhs),
        "lhs_str": str(lhs),
        "rhs_str": str(rhs),
        "lhs_srepr": sp.srepr(lhs),
        "rhs_srepr": sp.srepr(rhs),
        "complexity_lhs": count_symbolic_complexity(lhs),
        "complexity_rhs": count_symbolic_complexity(rhs),
        "meta": meta or {},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def expr_from_srepr(srepr_str: str) -> sp.Expr:
    """
    Reconstruct a SymPy expression from either a normal parseable string or an srepr-like form.
    We first try sympify; if it fails, we fallback to eval with a sanitized locals dictionary.
    IMPORTANT: Provide integer-assumed symbols for s, j, m, n to keep Derivative(..., (alpha, j)) valid.
    """
    # Build locals with integer assumptions for indices
    x = sp.symbols('x', real=True)
    alpha = sp.symbols('alpha', real=True)
    beta = sp.symbols('beta', real=True)
    n = sp.symbols('n', integer=True, positive=True)
    m = sp.symbols('m', integer=True, positive=True)
    j = sp.symbols('j', integer=True, positive=True)
    s = sp.symbols('s', integer=True, positive=True)

    loc = {**_sympy_ns(),
           "x": x, "alpha": alpha, "beta": beta,
           "n": n, "m": m, "j": j, "s": s}

    # Try a normal sympify first
    try:
        return sp.sympify(srepr_str, locals=loc)
    except Exception:
        pass

    # Fallback: eval of srepr (Symbol('x'), Derivative(...))
    # Build a minimal constructor environment for srepr strings.
    ctor = {}
    # Expose SymPy constructors used by srepr
    for name in (
        "Symbol", "Function", "Integer", "Rational", "Float",
        "Add", "Mul", "Pow", "ExpBase", "Pow", "Derivative",
        "sin", "cos", "tan", "sinh", "cosh", "tanh",
        "exp", "log", "Abs", "re", "im",
        "Integral", "Sum", "Product", "Piecewise", "Eq",
    ):
        if hasattr(sp, name):
            ctor[name] = getattr(sp, name)
    ctor.update(loc)

    try:
        return eval(srepr_str, {"__builtins__": {}}, ctor)
    except Exception as e:
        raise ValueError(f"Cannot reconstruct SymPy expression from srepr/str. Error: {e}\nInput:\n{srepr_str}")


# ---------------------------------------------------------------------
# Theorem 4.2 — general compact form
# ---------------------------------------------------------------------

class Theorem42:
    """
    Implements the generalized (compact) form of Theorem 4.2 suitable for
    arbitrary analytic f, supporting symbolic 'n' and symbolic/numeric 'm'.

    Key objects:
        ω_s := (2s−1)π/(2n)
        λ_s := exp(i x cos ω_s − x sin ω_s)
        \bar{λ}_s := exp(−i x cos ω_s − x sin ω_s)
        ψ := f(α + β λ_s),   φ := f(α + β \bar{λ}_s)
        ζ := exp(−x sin ω_s)

    m-th derivative (symbolic form):
        y^{(m)}(x) = − π/(2n) Σ_{s=1..n} Σ_{j=1..m} S(m,j) (β ζ)^j [ λ^m ∂^j_α ψ + \bar{λ}^m ∂^j_α φ ]

    where S(m,j) is the Stirling number of the second kind.

    The base function y(x) is:
        y(x) = π/(2n) Σ_{s=1..n} [ 2 f(α+β) − ψ − φ ].
    """

    def __init__(
        self,
        x: Optional[sp.Symbol] = None,
        alpha: Optional[sp.Symbol] = None,
        beta: Optional[sp.Symbol] = None,
        n: Any = "n",
        m_sym: Optional[sp.Symbol] = None
    ):
        self.x = x or sp.symbols("x", real=True)
        self.alpha = alpha if alpha is not None else sp.symbols("alpha", real=True)
        self.beta = beta if beta is not None else sp.symbols("beta", real=True)

        # n may be a number or a positive integer Symbol
        if isinstance(n, (int, sp.Integer)):
            self.n = int(n)
        else:
            self.n = sp.symbols(str(n), integer=True, positive=True)

        # canonical integer dummy symbols
        self.s = sp.symbols("s", integer=True, positive=True)  # s = 1..n
        self.j = sp.symbols("j", integer=True, positive=True)  # j = 1..m

        # symbolic m (if not passed numerically)
        self.m_sym = m_sym if isinstance(m_sym, sp.Symbol) else sp.symbols("m", integer=True, positive=True)

        # y(x) symbol for LHS use
        self.yfun = sp.Function("y")(self.x)

    # ---- building blocks ----

    def _omega(self) -> sp.Expr:
        return sp.pi * (2*self.s - 1) / (2*self.n)

    def _lam(self) -> sp.Expr:
        w = self._omega()
        return sp.exp(sp.I*self.x*sp.cos(w) - self.x*sp.sin(w))

    def _lam_bar(self) -> sp.Expr:
        w = self._omega()
        return sp.exp(-sp.I*self.x*sp.cos(w) - self.x*sp.sin(w))

    def _zeta(self) -> sp.Expr:
        w = self._omega()
        return sp.exp(-self.x*sp.sin(w))

    def _psi(self, f: Callable[[sp.Expr], sp.Expr]) -> sp.Expr:
        return f(self.alpha + self.beta*self._lam())

    def _phi(self, f: Callable[[sp.Expr], sp.Expr]) -> sp.Expr:
        return f(self.alpha + self.beta*self._lam_bar())

    # ---- outputs ----

    def y_base(self, f: Callable[[sp.Expr], sp.Expr], n_override: Optional[Any] = None) -> sp.Expr:
        nval = self.n if n_override is None else n_override
        s = self.s
        expr = sp.pi/(2*nval) * sp.summation(
            2*f(self.alpha + self.beta) - self._psi(f) - self._phi(f),
            (s, 1, nval)
        )
        return sp.simplify(expr)

    def y_derivative(
        self,
        f: Callable[[sp.Expr], sp.Expr],
        m: Optional[Any] = None,
        n_override: Optional[Any] = None,
        complex_form: bool = True
    ) -> sp.Expr:
        """
        General m-th derivative of y(x). For numeric m we emit a fully explicit sum;
        for symbolic m we return nested symbolic sums with Derivative(..., (alpha, j)),
        where j is an integer Symbol, which is what SymPy requires.
        """
        nval = self.n if n_override is None else n_override
        s, j = self.s, self.j

        lam, lamp = self._lam(), self._lam_bar()
        zeta = self._zeta()

        # Numeric m branch (fast and eager)
        if isinstance(m, (int, sp.Integer)):
            m = int(m)
            if m <= 0:
                return sp.S.Zero

            inner = sp.S.Zero
            for jj in range(1, m+1):
                S = stirling(m, jj, kind=2)  # exact integer
                dpsi = sp.Derivative(self._psi(f), (self.alpha, jj))
                dphi = sp.Derivative(self._phi(f), (self.alpha, jj))
                term = S * (self.beta**jj) * (zeta**jj) * (lam**m)  * dpsi \
                     + S * (self.beta**jj) * (zeta**jj) * (lamp**m) * dphi
                inner += term
            result = -sp.pi/(2*nval) * sp.summation(inner, (s, 1, nval))
            return result

        # Symbolic m branch (safe with integer j)
        m_sym = self.m_sym if m is None else m
        S2 = stirling(m_sym, j, kind=2)
        dpsi = sp.Derivative(self._psi(f), (self.alpha, j))   # j is integer symbol
        dphi = sp.Derivative(self._phi(f), (self.alpha, j))
        summand = S2 * (self.beta**j) * (zeta**j) * (lam**m_sym)  * dpsi \
                + S2 * (self.beta**j) * (zeta**j) * (lamp**m_sym) * dphi
        expr = -sp.pi/(2*nval) * sp.summation(sp.summation(summand, (j, 1, m_sym)), (s, 1, nval))
        return expr if complex_form else sp.re(expr)


# ---------------------------------------------------------------------
# Generator builder
# ---------------------------------------------------------------------

@dataclass
class TemplateConfig:
    alpha: Any = sp.symbols("alpha", real=True)
    beta: Any = sp.symbols("beta", real=True)
    n: Any = sp.symbols("n", integer=True, positive=True)
    m_sym: Any = sp.symbols("m", integer=True, positive=True)


class GeneratorBuilder:
    """
    Builds an ODE from a free-form template (LHS) and computes the RHS via Theorem 4.2.
    Template language:
        - y            → the solution symbol
        - Dy1, Dy2     → derivatives (1st, 2nd, …)
        - Dym          → m‑th derivative if m is symbolic
        - y^(m), y^(3) → aliases for Dym, Dy3
        - Any SymPy function wrapper, e.g. exp(Dy2), sinh(Dym), cos(y), log(1+y)

    Example template:
        "y + exp(Dy2) + sinh(Dym)"
        "y^(m) + y^(2) + sinh(y^(3))"
    """

    def __init__(self, theorem: Theorem42, config: Optional[TemplateConfig] = None):
        self.T = theorem
        self.cfg = config or TemplateConfig()
        self.x = self.T.x
        self.yfun = self.T.yfun
        self.m_val: Optional[Any] = None  # numeric or symbolic m (None -> symbolic)

    # --- parsing helpers ---

    @staticmethod
    def _alias(template: str) -> str:
        """Map y^(m) → Dym, y^(k) → Dyk for integer k."""
        s = re.sub(r"y\^\(\s*m\s*\)", "Dym", template)
        s = re.sub(r"y\^\(\s*(\d+)\s*\)", lambda m: f"Dy{m.group(1)}", s)
        return s

    @staticmethod
    def _orders_in_template(template: str) -> List[int]:
        """Extract derivative orders explicitly referenced as DyK (e.g., Dy1, Dy2, Dy10)."""
        return sorted({int(m) for m in re.findall(r"\bDy(\d+)\b", template)})

    @staticmethod
    def _uses_dym(template: str) -> bool:
        return bool(re.search(r"\bDym\b", template))

    # --- namespaces ---

    def _namespace(self, lhs: bool, orders: List[int], uses_dym: bool) -> Dict[str, Any]:
        """
        Build the evaluation namespace for sympify(template, locals=ns).
        - On LHS: y is y(x), DyK are Derivative(y(x), (x, K)), Dym is a placeholder if m is symbolic.
        - On RHS: y is y_base(expr), DyK are y^(K), Dym is y^(m) (symbolic sums if m is symbolic).
        """
        ns = _sympy_ns()
        ns.update({"x": self.x, "alpha": self.T.alpha, "beta": self.T.beta, "pi": sp.pi, "I": sp.I})
        # expose m to the template
        if self.m_val is None or isinstance(self.m_val, sp.Symbol):
            ns["m"] = self.T.m_sym
        else:
            ns["m"] = int(self.m_val)

        if lhs:
            ns["y"] = self.yfun
            for k in orders:
                ns[f"Dy{k}"] = sp.Derivative(self.yfun, (self.x, k))
            if uses_dym:
                if isinstance(self.m_val, (int, sp.Integer)):
                    ns["Dym"] = sp.Derivative(self.yfun, (self.x, int(self.m_val)))
                else:
                    ns["Dym"] = sp.Symbol("Dym")  # placeholder on LHS if m is symbolic
        else:
            # RHS exact expressions from Theorem 4.2
            # we cannot build these before f is known; the caller will set these keys
            pass
        return ns

    # --- public API ---

    def build(
        self,
        template: str,
        f: Callable[[sp.Expr], sp.Expr],
        m: Optional[Any] = None,
        n_override: Optional[Any] = None,
        complex_form: bool = True
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Build LHS (from template) and RHS (by substituting y, DyK, Dym with Theorem42 outputs).
        """
        self.m_val = m
        template = self._alias(template)
        orders = self._orders_in_template(template)
        uses_dym = self._uses_dym(template)

        # LHS namespace
        ns_lhs = self._namespace(lhs=True, orders=orders, uses_dym=uses_dym)
        try:
            lhs = sp.sympify(template, locals=ns_lhs)
        except Exception as e:
            raise ValueError(f"Cannot parse LHS template: {e}\nTemplate: {template}")

        # Build RHS substitution dictionary
        rhs_map: Dict[str, sp.Expr] = {}
        rhs_map["y"] = self.T.y_base(f, n_override=n_override)
        for k in orders:
            rhs_map[f"Dy{k}"] = self.T.y_derivative(f, m=k, n_override=n_override, complex_form=complex_form)
        if uses_dym:
            rhs_map["Dym"] = self.T.y_derivative(
                f,
                m=(self.m_val if self.m_val is not None else self.T.m_sym),
                n_override=n_override,
                complex_form=complex_form
            )

        # RHS namespace is same as LHS but with symbols bound to expressions
        ns_rhs = self._namespace(lhs=False, orders=orders, uses_dym=uses_dym)
        ns_rhs.update(rhs_map)

        try:
            rhs = sp.sympify(template, locals=ns_rhs)
        except Exception as e:
            raise ValueError(f"Cannot build RHS from Theorem 4.2: {e}\nTemplate: {template}")

        return sp.simplify(lhs), sp.simplify(rhs)


# ---------------------------------------------------------------------
# ML & DL (graceful fallbacks)
# ---------------------------------------------------------------------

class GeneratorPatternLearner:
    """
    Multi-head classifier for:
      - linearity (0/1)
      - stiffness bucket (int)
      - solvability class (int)
    If scikit-learn is unavailable, falls back to simple heuristics.
    """
    def __init__(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            self._TfidfVectorizer = TfidfVectorizer
            self._LogisticRegression = LogisticRegression
            self._sk_ok = True
        except Exception:
            self._sk_ok = False

        self.vec = None
        self.clf_lin = None
        self.clf_stiff = None
        self.clf_solv = None

    @staticmethod
    def _textualize(lhs: sp.Expr, rhs: sp.Expr) -> str:
        return f"{sp.srepr(lhs)} || {sp.srepr(rhs)}"

    def train(self, data: List[Tuple[sp.Expr, sp.Expr, Dict[str, int]]]) -> None:
        X = [self._textualize(l, r) for (l, r, _) in data]
        y_lin = [meta.get("linear", 0) for (_, _, meta) in data]
        y_stiff = [meta.get("stiffness", 0) for (_, _, meta) in data]
        y_solv = [meta.get("solvability", 0) for (_, _, meta) in data]

        if not self._sk_ok:
            # Heuristic fallback: nothing to train
            self.vec = None
            return

        self.vec = self._TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        Xv = self.vec.fit_transform(X)

        self.clf_lin = self._LogisticRegression(max_iter=1000).fit(Xv, y_lin)
        self.clf_stiff = self._LogisticRegression(max_iter=1000).fit(Xv, y_stiff)
        self.clf_solv = self._LogisticRegression(max_iter=1000).fit(Xv, y_solv)

    def predict(self, lhs_rhs_list: List[Tuple[sp.Expr, sp.Expr]]) -> List[Dict[str, int]]:
        out = []
        if not self._sk_ok or self.vec is None:
            # Heuristic: detect "exp" etc. as nonlinear
            for lhs, rhs in lhs_rhs_list:
                txt = f"{lhs} {rhs}"
                linear = 0 if re.search(r"exp|log|sin|cos|sinh|cosh|tanh", txt) else 1
                stiff = 1 if "exp" in txt else 0
                solv = 1 if "Integral" in sp.srepr(rhs) else 0
                out.append({"linear": linear, "stiffness": stiff, "solvability": solv})
            return out

        X = [self._textualize(l, r) for (l, r) in lhs_rhs_list]
        Xv = self.vec.transform(X)
        out.append({
            "linear": int(self.clf_lin.predict(Xv)[0]),
            "stiffness": int(self.clf_stiff.predict(Xv)[0]),
            "solvability": int(self.clf_solv.predict(Xv)[0]),
        })
        if len(lhs_rhs_list) > 1:
            # batch
            out = [{"linear": int(a), "stiffness": int(b), "solvability": int(c)}
                   for a, b, c in zip(self.clf_lin.predict(Xv),
                                      self.clf_stiff.predict(Xv),
                                      self.clf_solv.predict(Xv))]
        return out


class NoveltyDetector:
    """
    Tiny novelty/complexity scorer using a Transformer encoder (PyTorch),
    with a deterministic heuristic fallback if torch is unavailable.
    """
    def __init__(self, dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        self.available = False
        self.model = None
        self.token = None
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            self.available = True

            class TinyTransformer(nn.Module):
                def __init__(self, vocab=4096, dim=dim, n_heads=n_heads, n_layers=n_layers):
                    super().__init__()
                    self.emb = nn.Embedding(vocab, dim)
                    encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
                    self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                    self.head = nn.Linear(dim, 1)

                def forward(self, x):
                    h = self.emb(x)
                    h = self.enc(h)
                    h = h.mean(dim=1)
                    return self.head(h).squeeze(-1)

            self.model = TinyTransformer()
        except Exception:
            self.available = False

    @staticmethod
    def _tokenize(expr: sp.Expr, vocab: int = 4096) -> List[int]:
        # very simple hash tokenizer
        toks = []
        for node in sp.preorder_traversal(expr):
            toks.append((hash(type(node).__name__) ^ hash(str(node.func if hasattr(node, "func") else node))) % vocab)
        return toks[:512] or [1]

    def score(self, lhs: sp.Expr, rhs: sp.Expr) -> float:
        if not self.available:
            # heuristic novelty: (structure) + (rhs depth)
            c = count_symbolic_complexity(rhs)
            return 0.5 * c["n_preorder_nodes"] + 0.5 * c["count_ops_visual"]

        torch = self.torch
        with torch.no_grad():
            lt = self._tokenize(lhs)
            rt = self._tokenize(rhs)
            ids = torch.tensor([lt + [0] + rt], dtype=torch.long)
            s = self.model(ids).item()
            return float(s)

    def train_pairs(self, pairs_with_target: List[Tuple[sp.Expr, sp.Expr, float]], epochs: int = 3, lr: float = 1e-3):
        if not self.available or not pairs_with_target:
            return
        torch = self.torch
        nn = self.nn
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for lhs, rhs, y in pairs_with_target:
                lt = self._tokenize(lhs)
                rt = self._tokenize(rhs)
                ids = torch.tensor([lt + [0] + rt], dtype=torch.long)
                target = torch.tensor([y], dtype=torch.float32)
                pred = self.model(ids)
                loss = loss_fn(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()


# ---------------------------------------------------------------------
# Preset recipes
# ---------------------------------------------------------------------

class GeneratorLibrary:
    """A few ready-made presets for the UI page."""
    @staticmethod
    def preset_pantograph_linear() -> Tuple[str, str]:
        # A simple pantograph-like linear template
        return ("y + Dy2", "z")  # f(z) = z

    @staticmethod
    def preset_nonlinear_wrap() -> Tuple[str, str]:
        return ("exp(Dy2) + y", "sin(z)")

    @staticmethod
    def preset_multiorder_mix() -> Tuple[str, str]:
        return ("y + Dy1 + Dy3 + sinh(Dym)", "exp(z)")


# ---------------------------------------------------------------------
# End
# ---------------------------------------------------------------------
