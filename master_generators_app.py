
# -*- coding: utf-8 -*-
"""
Master Generators App (Streamlit)
=================================

- Implements the *new* compact shape of Theorem 4.2 via Stirling numbers.
- Free-form **Generator Builder**: combine y, Dy1, Dy2, ..., Dym with wrappers
  like exp(.), sinh(.), cosh(.), polynomials, etc. The app computes the RHS
  from the theorem so that the base y(x) is an exact solution.
- Symbolic **n** and both symbolic/explicit **m** are supported.
- ML (scikit-learn) and DL (PyTorch) applied to constructed generators/ODEs.
- Robust import strategy for `src/` and optional deps.

Run:
    streamlit run master_generators_app.py
"""

# ======================================================================
# Standard imports
# ======================================================================
import os
import sys
import json
import re
import math
import cmath
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

# Streamlit UI
try:
    import streamlit as st
except Exception as _e:
    st = types.SimpleNamespace(
        warning=lambda *a, **k: print("[streamlit missing]:", *a),
        error=lambda *a, **k: print("[streamlit missing ERROR]:", *a),
        info=lambda *a, **k: print("[streamlit info]:", *a),
        success=lambda *a, **k: print("[streamlit success]:", *a),
        write=print, latex=print, markdown=print, header=print, subheader=print,
        title=print, columns=lambda n: [types.SimpleNamespace()] * n, sidebar=types.SimpleNamespace(),
        set_page_config=lambda **k: None, radio=lambda *a, **k: "Overview",
        text_input=lambda *a, **k: "sin(z)", number_input=lambda *a, **k: 2,
        checkbox=lambda *a, **k: True, selectbox=lambda *a, **k: "Theorem 4.2",
        button=lambda *a, **k: False, code=print, json=print, write_stream=None
    )

# Numeric / symbolic stack
import numpy as np
import sympy as sp
from sympy import I as i, pi as PI, Eq, symbols, Function
from sympy.functions.combinatorial.numbers import stirling as stirling2
from sympy import Sum, summation, simplify, diff, exp, cos, sin, re, im, lambdify, srepr

# Optional ML / DL
try:
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

# HTTP client for API testing (optional)
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# ======================================================================
# Flexible import resolution for local project structure
# ======================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if os.path.isdir(SRC_DIR) and (SRC_DIR not in sys.path):
    sys.path.insert(0, SRC_DIR)

def _try_import(path: str):
    try:
        __import__(path)
        return sys.modules[path]
    except Exception:
        return None

mod_ui = _try_import("ui.main_app") or _try_import("UI.main_app")
mod_components = _try_import("ui.components") or _try_import("UI.components")
mod_generators = _try_import("generators")
mod_master = _try_import("generators.master_generator")
mod_lin = _try_import("generators.linear_generators")
mod_nonlin = _try_import("generators.nonlinear_generators")
mod_utils = _try_import("utils")

# ======================================================================
# Utilities
# ======================================================================
def _latex(expr: sp.Expr) -> str:
    try:
        return sp.latex(simplify(expr))
    except Exception:
        return sp.latex(expr)

def _safe_eval_function_of_z(expr_str: str) -> sp.Function:
    """
    Parse 'f(z)' given as a Pythonic / SymPy string, returning a callable f(z).
    Supported: z, sin(z), exp(z), cosh(z), z**2 + sin(z), etc.
    """
    z = sp.Symbol('z', complex=True)
    safe_ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
    safe_ns.update({'z': z, 'i': sp.I, 'I': sp.I, 'pi': sp.pi, 'PI': sp.pi, 'E': sp.E})
    try:
        expr = eval(expr_str, {"__builtins__": {}}, safe_ns)
    except Exception as e:
        raise ValueError(f"Could not parse f(z) from '{expr_str}': {e}")
    return sp.Lambda(z, expr)

def _is_integer_value(x: Any) -> bool:
    try:
        return int(x) == x
    except Exception:
        return False

# ======================================================================
# Theorem 4.2 (Stirling-number compact form)
# ======================================================================
@dataclass
class Theorem42:
    """
    Implements the compact form of Theorem 4.2 using Stirling numbers of the second kind.
    Supports numeric and symbolic n and m.
    """
    x: sp.Symbol = sp.Symbol('x', real=True)
    alpha: sp.Symbol = sp.Symbol('alpha', real=True)
    beta: sp.Symbol = sp.Symbol('beta', positive=True)
    n: sp.Symbol = sp.Symbol('n', integer=True, positive=True)
    m_sym: sp.Symbol = sp.Symbol('m', integer=True, positive=True)

    def omega(self, s: sp.Symbol) -> sp.Expr:
        return (2*s - 1)*sp.pi/(2*self.n)

    def zeta(self, s: sp.Symbol) -> sp.Expr:
        w = self.omega(s)
        return sp.exp(-self.x*sp.sin(w)) * sp.exp(sp.I*self.x*sp.cos(w))

    def lambda_phase(self, s: sp.Symbol) -> sp.Expr:
        w = self.omega(s)
        return sp.exp(sp.I*(sp.pi/2 + w))

    def psi(self, f: Callable[[sp.Symbol], sp.Expr], s: sp.Symbol) -> sp.Expr:
        return f(self.alpha + self.beta*self.zeta(s))

    def phi(self, f: Callable[[sp.Symbol], sp.Expr], s: sp.Symbol) -> sp.Expr:
        w = self.omega(s)
        zbar = sp.exp(-self.x*sp.sin(w)) * sp.exp(-sp.I*self.x*sp.cos(w))
        return f(self.alpha + self.beta*zbar)

    def y_base(self, f: Callable, n_override: Optional[Union[int, sp.Symbol]] = None) -> sp.Expr:
        """
        y(x) = (pi/(2n)) * sum_{s=1}^n [ 2 f(alpha+beta) - psi - phi ]
        Symbolic n returns a Sum; numeric n is explicitly summed.
        """
        n_ = self.n if n_override is None else n_override
        s = sp.Symbol('s', integer=True, positive=True)
        expr = (sp.pi/(2*n_)) * sp.summation(
            2*f(self.alpha + self.beta) - self.psi(f, s) - self.phi(f, s),
            (s, 1, n_)
        )
        return simplify(expr)

    def y_derivative(self,
                     f: Callable,
                     m: Optional[Union[int, sp.Symbol]] = None,
                     n_override: Optional[Union[int, sp.Symbol]] = None,
                     complex_form: bool = True) -> sp.Expr:
        """
        y^{(m)}(x) in compact Faà di Bruno / Stirling form.

        m=None => symbolic m. m=int => explicit finite sums.
        complex_form=False => take Re(.) to display real trig combination.
        """
        n_ = self.n if n_override is None else n_override
        s = sp.Symbol('s', integer=True, positive=True)
        if m is None:
            m = self.m_sym

        j = sp.Symbol('j', integer=True, positive=True)
        lam = self.lambda_phase(s)
        z = self.zeta(s)
        psi = self.psi(f, s)
        phi = self.phi(f, s)
        dpsi_j = lambda j: diff(psi, self.alpha, j)
        dphi_j = lambda j: diff(phi, self.alpha, j)
        pref = -(sp.pi/(2*n_))

        # symbolic n/m version
        S_mj = lambda m_, j_: sp.functions.combinatorial.numbers.stirling(m_, j_, kind=2)
        inner = Sum(
            S_mj(m, j) * (
                (self.beta*z)**j * lam**m * dpsi_j(j)
                + (self.beta*sp.conjugate(z))**j * sp.conjugate(lam)**m * dphi_j(j)
            ),
            (j, 1, m)
        )
        total = pref * Sum(inner, (s, 1, n_))

        if isinstance(m, sp.Integer) and isinstance(n_, int):
            # expand numeric sums
            total = simplify(total.doit())

        if not complex_form:
            total = sp.re(total)
        return simplify(total)

# ======================================================================
# Free-form Generator Builder
# ======================================================================
class GeneratorBuilder:
    """
    Builds ODEs from a user-provided expression template in terms of
    y, Dy1, Dy2, Dy3, ..., Dyk, and Dym (if used), combined by arbitrary
    SymPy-available wrappers such as exp(.), sinh(.), cosh(.), log(.), etc.

    LHS operator form: template evaluated on symbolic y(x) and Derivative(y(x), ...).
    RHS explicit form: same template evaluated on the exact y(x) and y^{(m)} from Theorem 4.2.
    """
    def __init__(self, T: Theorem42, f: Callable, n_val: Union[int, sp.Symbol],
                 m_val: Optional[Union[int, sp.Symbol]], complex_form: bool):
        self.T = T
        self.f = f
        self.n_val = n_val
        self.m_val = m_val
        self.complex_form = complex_form
        self.x = T.x

        # cache derivatives to avoid recomputation
        self._d_cache: Dict[Union[int, str], sp.Expr] = {}

        # symbolic y(x)
        self.yfun = sp.Function('y')(self.x)

    def _parse_needed_orders(self, template: str) -> Tuple[List[int], bool]:
        """
        Detects which derivative orders are referenced in the template string.
        Returns (orders_list, uses_Dym:boolean).
        """
        orders = sorted({int(m) for m in re.findall(r'Dy(\d+)', template)})
        uses_dym = bool(re.search(r'\bDym\b', template))
        return orders, uses_dym

    def _build_eval_namespace(self, lhs: bool, d_orders: List[int], uses_dym: bool) -> Dict[str, Any]:
        """
        Build eval namespace for the template.
        lhs=True  => y and derivatives as operator forms (y(x), Derivative(y(x), ...)).
        lhs=False => y and derivatives replaced by exact y(x) and theorem derivatives.
        """
        ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
        ns.update({"x": self.x, "alpha": self.T.alpha, "beta": self.T.beta, "pi": sp.pi, "I": sp.I})
        if lhs:
            ns["y"] = self.yfun
            for k in d_orders:
                ns[f"Dy{k}"] = sp.Derivative(self.yfun, (self.x, k))
            if uses_dym:
                if isinstance(self.m_val, (int, sp.Integer)):
                    ns["Dym"] = sp.Derivative(self.yfun, (self.x, int(self.m_val)))
                else:
                    # Symbolic m can't be a Derivative order; use a placeholder symbol
                    ns["Dym"] = sp.Symbol("Dym")
        else:
            # RHS exact expressions
            ns["y"] = self._y_exact()
            for k in d_orders:
                ns[f"Dy{k}"] = self._y_deriv_exact(k)
            if uses_dym:
                ns["Dym"] = self._y_deriv_exact(self.m_val if self.m_val is not None else self.T.m_sym)
        return ns

    def _y_exact(self) -> sp.Expr:
        key = "y0"
        if key not in self._d_cache:
            self._d_cache[key] = simplify(self.T.y_base(self.f, n_override=self.n_val))
        return self._d_cache[key]

    def _y_deriv_exact(self, m: Union[int, sp.Symbol]) -> sp.Expr:
        key = f"m={m}"
        if key not in self._d_cache:
            self._d_cache[key] = simplify(self.T.y_derivative(self.f, m=m, n_override=self.n_val,
                                                              complex_form=self.complex_form))
        return self._d_cache[key]

    def build(self, template: str) -> Tuple[sp.Expr, sp.Expr]:
        """
        Returns (lhs_operator_form, rhs_explicit_form).
        """
        d_orders, uses_dym = self._parse_needed_orders(template)
        lhs_ns = self._build_eval_namespace(lhs=True, d_orders=d_orders, uses_dym=uses_dym)
        rhs_ns = self._build_eval_namespace(lhs=False, d_orders=d_orders, uses_dym=uses_dym)

        # Evaluate the template into SymPy expressions
        safe_globals = {"__builtins__": {}}
        try:
            lhs_expr = eval(template, safe_globals, lhs_ns)
        except Exception as e:
            raise ValueError(f"Failed to parse LHS template: {e}")
        try:
            rhs_expr = eval(template, safe_globals, rhs_ns)
        except Exception as e:
            raise ValueError(f"Failed to construct RHS from theorem: {e}")

        return (simplify(lhs_expr), simplify(rhs_expr))

# ======================================================================
# Generator utilities (optional presets)
# ======================================================================
class GeneratorLibrary:
    def __init__(self, T: Theorem42) -> None:
        self.T = T
    def linear_1(self, f) -> Tuple[sp.Expr, sp.Expr, sp.Expr, str]:
        x, a, b, n = self.T.x, self.T.alpha, self.T.beta, self.T.n
        y = self.T.y_base(f)
        ode_lhs = diff(y, x, 2) + y
        g = f(a + b*sp.exp(-x))
        rhs = sp.pi*( f(a+b) - f(a + b*sp.exp(-x)) ) \
              - sp.pi*b*sp.exp(-x)*diff(g, a, 1) \
              - sp.pi*b**2*sp.exp(-2*x)*diff(g, a, 2)
        sol = sp.pi*( f(a+b) - f(a + b*sp.exp(-x)) )
        notes = "From Theorem 4.1 with n=1; RHS via ∂_α on f(α+β e^{-x})."
        return (ode_lhs, rhs, sol, notes)

# ======================================================================
# ML: Generator Pattern Learner
# ======================================================================
class GeneratorPatternLearner:
    def __init__(self) -> None:
        self.available = HAVE_SK
        if self.available:
            self.pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=1, max_features=10000)),
                ("clf", LogisticRegression(max_iter=400))
            ])
        else:
            self.pipeline = None

    def _stringify(self, expr: sp.Expr) -> str:
        return sp.srepr(simplify(expr))

    def train(self, dataset: List[Tuple[sp.Expr, sp.Expr, int]]) -> Dict[str, Any]:
        if not self.available or not dataset:
            return {"status":"fallback", "message": "Using rule-based heuristic (scikit-learn unavailable or empty dataset)."}
        X = [ self._stringify(Eq(l,r)) for (l,r,_) in dataset ]
        y = [ label for (_,_,label) in dataset ]
        self.pipeline.fit(X, y)
        return {"status":"ok", "classes": sorted(set(y))}

    def predict(self, pairs: List[Tuple[sp.Expr, sp.Expr]]) -> List[int]:
        if not self.available:
            # Simple heuristic: nonlinear if powers of derivatives appear
            preds = []
            for (l,r) in pairs:
                s = sp.srepr(l) + sp.srepr(r)
                preds.append(1 if ("Pow(Derivative" in s or "sin" in s or "cosh" in s or "exp" in s) else 0))
            return preds
        X = [ self._stringify(Eq(l,r)) for (l,r) in pairs ]
        return list(self.pipeline.predict(X))

# ======================================================================
# DL: Novelty Detector
# ======================================================================
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=512, d_model=128, nhead=4, num_layers=2, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model)*0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)
    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T, :]
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.out(h).squeeze(-1)

class NoveltyDetector:
    def __init__(self, vocab: Optional[Dict[str,int]]=None, max_len: int=256):
        self.available = HAVE_TORCH
        self.max_len = max_len
        if vocab is None:
            chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/^()=,._{}[]| <>eEpiI")
            self.vocab = {ch:i+2 for i,ch in enumerate(chars)}
            self.vocab["<pad>"] = 0
            self.vocab["<unk>"] = 1
        else:
            self.vocab = vocab
        if self.available:
            self.model = TinyTransformer(vocab_size=len(self.vocab), max_len=max_len)
            self.opt = optim.Adam(self.model.parameters(), lr=3e-4)
            self.loss_fn = nn.MSELoss()
        else:
            self.model = None

    def encode(self, s: str) -> np.ndarray:
        ids = [ self.vocab.get(ch, 1) for ch in s[:self.max_len] ]
        if len(ids)<self.max_len:
            ids += [0]*(self.max_len-len(ids))
        return np.array(ids, dtype=np.int64)

    def novelty_score(self, ode_strs: List[str]) -> List[float]:
        if not self.available:
            scores = []
            for s in ode_strs:
                uniq = len(set(s))
                scores.append( min(1.0, 0.1 + 0.9*uniq/max(10, len(s))) )
            return scores
        self.model.eval()
        out = []
        with torch.no_grad():
            for s in ode_strs:
                x = torch.tensor(self.encode(s))[None, :]
                y = torch.sigmoid(self.model(x)).item()
                out.append(float(y))
        return out

    def quick_train(self, pairs: List[Tuple[str, float]], epochs: int=3, batch_size: int=16):
        if not self.available or not pairs:
            return {"status":"skipped"}
        self.model.train()
        xs = torch.stack([ torch.tensor(self.encode(s)) for s,_ in pairs ])
        ys = torch.tensor([ t for _,t in pairs ], dtype=torch.float32)
        N = xs.size(0)
        for ep in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = xs[idx], ys[idx]
                self.opt.zero_grad()
                pred = torch.sigmoid(self.model(xb))
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.opt.step()
        return {"status":"ok", "epochs":epochs}

# ======================================================================
# Streamlit App pages
# ======================================================================
def _get_session_list(name: str) -> List[Any]:
    if not hasattr(st, "session_state"):
        return []
    if name not in st.session_state:
        st.session_state[name] = []
    return st.session_state[name]

def page_overview():
    st.title("Master Generators — Research App")
    st.markdown("""
    This app implements the **Master Generators** framework with the new **Theorem 4.2**
    (Stirling-number compact form). Use the **Generator Builder** to create arbitrary
    generators like `y + exp(Dy2)` or `sinh(Dym)`, and the app will compute the RHS through
    the theorem, producing exact ODEs for your chosen \(f(z)\), \(n\), and \(m\).
    """)

def page_theorem42():
    st.header("Theorem 4.2 — Compact Stirling-Number Form")
    x = sp.Symbol('x', real=True)
    alpha = sp.Symbol('alpha', real=True)
    beta = sp.Symbol('beta', positive=True)
    n_sym = sp.Symbol('n', integer=True, positive=True)
    T = Theorem42(x=x, alpha=alpha, beta=beta, n=n_sym)

    col1, col2 = st.columns(2)
    with col1:
        f_str = st.text_input("Enter f(z):", value="sin(z)", help="Any SymPy expression in variable z.")
        f = _safe_eval_function_of_z(f_str)
        use_symbolic_n = st.checkbox("Use symbolic n", value=True)
        if use_symbolic_n:
            n_val = n_sym
        else:
            n_val = st.number_input("n (integer ≥ 1):", min_value=1, max_value=64, value=2, step=1)
        use_symbolic_m = st.checkbox("Unspecified derivative order (symbolic m)", value=True)
        if use_symbolic_m:
            m_val = None
        else:
            m_val = st.number_input("m (integer ≥ 1):", min_value=1, max_value=12, value=3, step=1)
        complex_form = st.checkbox("Use compact complex form", value=True)
    with col2:
        alpha_val = st.text_input("α (can be symbolic):", value="alpha")
        beta_val = st.text_input("β (>0, can be symbolic):", value="beta")
        safe_ns = {**{k:getattr(sp,k) for k in dir(sp) if not k.startswith('_')}, 'alpha':alpha, 'beta':beta, 'x':x}
        try:
            alpha_eval = eval(alpha_val, {"__builtins__": {}}, safe_ns)
        except Exception:
            alpha_eval = alpha
        try:
            beta_eval = eval(beta_val, {"__builtins__": {}}, safe_ns)
        except Exception:
            beta_eval = beta

    T = Theorem42(x=x, alpha=alpha_eval, beta=beta_eval, n=n_val if use_symbolic_n else sp.Integer(n_val))
    y_expr = T.y_base(f, n_override=(n_val if use_symbolic_n else int(n_val)))
    st.markdown("**Base solution (Theorem 4.1):**")
    st.latex(_latex(y_expr))

    st.markdown("**General derivative \(y^{(m)}(x)\) (Theorem 4.2):**")
    y_m = T.y_derivative(f, m=m_val, n_override=(n_val if use_symbolic_n else int(n_val)), complex_form=complex_form)
    st.latex(_latex(y_m))
    st.caption("Tip: switch off 'complex form' to display a real-valued trig combination (may be slower).")

def page_builder():
    st.header("Generator Builder — Free Construction")
    x = sp.Symbol('x', real=True)
    alpha = sp.Symbol('alpha', real=True)
    beta = sp.Symbol('beta', positive=True)
    n_sym = sp.Symbol('n', integer=True, positive=True)
    m_sym = sp.Symbol('m', integer=True, positive=True)

    colA, colB = st.columns(2)
    with colA:
        f_str = st.text_input("Enter f(z):", value="z", help="e.g. z, sin(z), exp(z), cosh(z), z**2 + sin(z)")
        f = _safe_eval_function_of_z(f_str)
        use_symbolic_n = st.checkbox("Use symbolic n", value=True, key="b_sym_n")
        n_val = n_sym if use_symbolic_n else st.number_input("n (integer ≥ 1):", min_value=1, max_value=64, value=2, step=1, key="b_n")
        use_symbolic_m = st.checkbox("Unspecified derivative order (symbolic m)", value=False, key="b_sym_m")
        m_val = None if use_symbolic_m else st.number_input("m (integer ≥ 1):", min_value=1, max_value=16, value=2, step=1, key="b_m")
        complex_form = st.checkbox("Use compact complex form for derivatives", value=True, key="b_cf")
    with colB:
        alpha_val = st.text_input("α (can be symbolic):", value="alpha", key="b_alpha")
        beta_val = st.text_input("β (>0, can be symbolic):", value="beta", key="b_beta")
        safe_ns = {**{k:getattr(sp,k) for k in dir(sp) if not k.startswith('_')}, 'alpha':alpha, 'beta':beta, 'x':x, 'm':m_sym}
        try:
            alpha_eval = eval(alpha_val, {"__builtins__": {}}, safe_ns)
        except Exception:
            alpha_eval = alpha
        try:
            beta_eval = eval(beta_val, {"__builtins__": {}}, safe_ns)
        except Exception:
            beta_eval = beta

    # Provide template help
    st.markdown("""
**Template syntax** (SymPy):
- Variables: `y`, `Dy1`, `Dy2`, `Dy3`, ..., `Dyk`, and `Dym` (if you want the m-th derivative).
- Use any SymPy wrappers: `exp(.)`, `sinh(.)`, `cosh(.)`, `sqrt(.)`, `log(.)`, `sin(.)`, `cos(.)`, polynomials, sums, products.
- Examples:
  - `y + Dy2`
  - `exp(Dy2) + y`
  - `sinh(Dym) - y`
  - `Dy1**2 + cosh(Dy2) + y`
    """)
    template = st.text_input("Enter generator template:", value="y + exp(Dy2) + sinh(Dym)")

    # Build Theorem object
    T = Theorem42(x=x, alpha=alpha_eval, beta=beta_eval, n=(n_val if use_symbolic_n else sp.Integer(n_val)), m_sym=m_sym)
    builder = GeneratorBuilder(T, f, (n_val if use_symbolic_n else int(n_val)), (m_val if not use_symbolic_m else None), complex_form=complex_form)

    if st.button("Build Generator → ODE"):
        try:
            lhs, rhs = builder.build(template)
            st.subheader("Constructed ODE")
            st.latex(_latex(Eq(lhs, rhs)))
            # Store to session for ML/DL
            dataset = _get_session_list("built_odes")
            dataset.append((lhs, rhs))
            st.success("Added to in-session ODE set (for ML/DL).")
        except Exception as e:
            st.error(str(e))

    # Show current set and export
    odes = _get_session_list("built_odes")
    if odes:
        st.subheader("Current in-session ODEs")
        for idx, (L, R) in enumerate(odes[-5:]):
            st.markdown(f"**ODE #{len(odes)-5+idx+1}**")
            st.latex(_latex(Eq(L, R)))
        if st.button("Export ODEs (JSON)"):
            data = [{"lhs": sp.srepr(L), "rhs": sp.srepr(R)} for (L,R) in odes]
            st.code(json.dumps(data)[:2000] + ("..." if len(json.dumps(data))>2000 else ""))

def page_generators():
    st.header("Preset Generators (Examples)")
    x = sp.Symbol('x', real=True)
    alpha = sp.Symbol('alpha', real=True)
    beta = sp.Symbol('beta', positive=True)
    n_sym = sp.Symbol('n', integer=True, positive=True)
    T = Theorem42(x=x, alpha=alpha, beta=beta, n=n_sym)
    G = GeneratorLibrary(T)

    f_str = st.text_input("Enter f(z):", value="z", help="e.g. z, sin(z), exp(z), cosh(z)")
    f = _safe_eval_function_of_z(f_str)

    gen_choice = st.selectbox("Pick a generator", ["Linear #1 (y''+y)", "More coming..."])
    if gen_choice.startswith("Linear #1"):
        lhs, rhs, sol, notes = G.linear_1(f)
        st.subheader("ODE")
        st.latex(_latex(Eq(lhs, rhs)))
        st.subheader("Exact solution")
        st.latex(_latex(sp.Eq(sp.Function('y')(x), sol)))
        st.caption(notes)

def page_ml():
    st.header("ML — Apply to Built Generators / ODEs")
    learner = GeneratorPatternLearner()
    odes = _get_session_list("built_odes")
    if not odes:
        st.warning("No ODEs built yet. Use the Generator Builder first.")
        return

    # Labeling UI (toy)
    st.markdown("Assign labels to your ODEs (e.g., 0=linear-like, 1=nonlinear-like).")
    labels = []
    for idx, (L,R) in enumerate(odes):
        st.latex(_latex(Eq(L,R)))
        lab = st.number_input(f"Label for ODE #{idx+1}", min_value=0, max_value=9, value=1, step=1, key=f"lab{idx}")
        labels.append(lab)

    train_data = [(L,R,lab) for (L,R),lab in zip(odes, labels)]
    res = learner.train(train_data)
    st.json(res)

    preds = learner.predict(odes)
    st.write("Predictions on current ODEs:", preds)

def page_dl():
    st.header("DL — Novelty on Built ODEs")
    det = NoveltyDetector()
    odes = _get_session_list("built_odes")
    if not odes:
        st.warning("No ODEs built yet.")
        return
    strings = [ sp.srepr(Eq(L,R)) for (L,R) in odes ]
    scores = det.novelty_score(strings)
    st.write("Novelty scores (0–1):")
    for k,(s,score) in enumerate(zip(strings, scores), start=1):
        st.write(f"ODE #{k}: {score:0.3f}")

def page_api_client():
    st.header("API Client (optional)")
    st.caption("Targets the local FastAPI server in api_server.py if it's running.")
    if not HAVE_REQUESTS:
        st.warning("`requests` not installed; API client disabled.")
        return
    base_url = st.text_input("Base URL", value="http://localhost:8000")
    endpoint = st.text_input("Endpoint", value="/health")
    if st.button("GET"):
        try:
            r = requests.get(base_url.rstrip("/") + endpoint)
            st.code(f"Status: {r.status_code}\n\n{r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

def page_export():
    st.header("Export / Save")
    odes = _get_session_list("built_odes")
    if not odes:
        st.warning("No ODEs built yet.")
        return
    export_choice = st.selectbox("Export format", ["LaTeX", "JSON"])
    if export_choice == "LaTeX":
        tex = "\n\n".join([ sp.latex(Eq(L,R)) for (L,R) in odes ])
        st.code(tex)
    else:
        data = [{"lhs": sp.srepr(L), "rhs": sp.srepr(R)} for (L,R) in odes]
        st.code(json.dumps(data, indent=2))

# ======================================================================
# Main
# ======================================================================
def main():
    st.set_page_config(page_title="Master Generators App", layout="wide")
    with st.sidebar:
        st.title("Master Generators")
        page = st.radio("Navigate", [
            "Overview",
            "Theorem 4.2",
            "Generator Builder",
            "Preset Generators",
            "ML",
            "DL",
            "API Client",
            "Export"
        ])
        st.caption("Build free-form generators; RHS is computed from Theorem 4.2.")

    if page == "Overview":
        page_overview()
    elif page == "Theorem 4.2":
        page_theorem42()
    elif page == "Generator Builder":
        page_builder()
    elif page == "Preset Generators":
        page_generators()
    elif page == "ML":
        page_ml()
    elif page == "DL":
        page_dl()
    elif page == "API Client":
        page_api_client()
    elif page == "Export":
        page_export()

if __name__ == "__main__":
    main()
