
# -*- coding: utf-8 -*-
"""
Master Generators App (Streamlit) — Free Generator Builder + Batch ML/DL
========================================================================

Key capabilities:
- New compact Theorem 4.2 via Stirling numbers (Faà di Bruno form).
- Symbolic `n` and both unspecified `m` (symbolic) and explicitly specified `m`.
- **Free-form Generator Builder**: compose ODE generators from y, y', y'', ..., y^(m)
  using wrappers like exp(), sinh(), cosh(), log(), powers, sums, products, etc.
- RHS is computed by **plugging the Theorem-based y and y^{(k)}** into the same generator,
  so the constructed y(x) is **guaranteed** to be a solution.
- **Batch generation** of infinite ODE families by sweeping f(z), n, and m (and expressions).
- **Full ML and DL pipelines** (with graceful fallback if scikit-learn or PyTorch
  are unavailable): classification/clustering & novelty scoring on generated ODEs.
- Robust import strategy across `src/` layouts, plus optional API client.

Run:
    streamlit run master_generators_app.py
"""

# ======================================================================
# Standard imports
# ======================================================================
import os
import sys
import json
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
        error=lambda *a, **k: print("[streamlit ERROR]:", *a),
        info=lambda *a, **k: print("[streamlit info]:", *a),
        success=lambda *a, **k: print("[streamlit success]:", *a),
        write=print, latex=print, markdown=print, code=print, json=print,
        set_page_config=lambda *a, **k: None, columns=lambda n: [types.SimpleNamespace()] * n,
        sidebar=types.SimpleNamespace, title=print, header=print, subheader=print,
        text_input=lambda *a, **k: "sin(z)", checkbox=lambda *a, **k: True,
        number_input=lambda *a, **k: 2, selectbox=lambda *a, **k: "Overview",
        radio=lambda *a, **k: "Overview", button=lambda *a, **k: False, caption=print,
        file_uploader=lambda *a, **k: None, write_stream=print
    )

# Numeric / symbolic stack
import numpy as np
import sympy as sp
from sympy import I as i, pi as PI, Eq, symbols, Function
from sympy.functions.combinatorial.numbers import stirling as stirling2
from sympy import Sum, summation, simplify, diff, exp, cos, sin, re, im, lambdify, srepr
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor, implicit_multiplication_application

# Optional ML / DL
try:
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
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
    Parse 'f(z)' given as a Pythonic / SymPy string, returning a callable 'f' that maps a SymPy symbol to an expression.
    Examples: "exp(z)" , "sin(z)", "cosh(z)", "z", "z**2 + sin(z)"
    """
    z = sp.Symbol('z', complex=True)
    safe_ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
    safe_ns.update({'z': z, 'i': sp.I, 'I': sp.I, 'pi': sp.pi, 'PI': sp.pi, 'E': sp.E})
    try:
        expr = eval(expr_str, {"__builtins__": {}}, safe_ns)
    except Exception as e:
        raise ValueError(f"Could not parse f(z) from '{expr_str}': {e}")
    f = sp.Lambda(z, expr)
    return f

def _stringify_expr(e: sp.Expr) -> str:
    try:
        return sp.srepr(simplify(e))
    except Exception:
        return sp.srepr(e)

def _detect_nonlinearity(expr: sp.Expr) -> bool:
    s = _stringify_expr(expr)
    # crude signal: powers of derivatives, exp/log/sin/cosh on derivatives, etc.
    nonlin_signals = ["Pow(Derivative", "exp(Derivative", "log(Derivative", "sin(Derivative",
                      "cos(Derivative", "sinh(Derivative", "cosh(Derivative"]
    return any(tok in s for tok in nonlin_signals)

# ======================================================================
# New Theorem 4.2 (compact Stirling-number form)
# ======================================================================
@dataclass
class Theorem42:
    """
    Implements the compact form of Theorem 4.2 using Stirling numbers of the second kind.
    Supports both numeric and symbolic n and m.
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
        y(x) from Theorem 4.1:
            y(x) = (pi/(2n)) * sum_{s=1}^n [ 2 f(alpha+beta) - psi - phi ]
        Works for symbolic n (left as Sum) or numeric n (explicit sum).
        """
        n_ = self.n if n_override is None else n_override
        s = sp.Symbol('s', integer=True, positive=True)
        expr = (sp.pi/(2*n_)) * sp.summation(
            2*sp.Function('f')(0).subs(sp.Function('f')(0), f(self.alpha + self.beta)) - self.psi(f, s) - self.phi(f, s),
            (s, 1, n_)
        )
        return simplify(expr)

    def y_derivative(self,
                     f: Callable,
                     m: Optional[Union[int, sp.Symbol]] = None,
                     n_override: Optional[Union[int, sp.Symbol]] = None,
                     complex_form: bool = True) -> sp.Expr:
        """
        General derivative y^{(m)}(x) using the compact Faà di Bruno / Stirling shape.
        - m None  -> symbolic m (returns nested Sum)
        - m int   -> finite sums
        - complex_form True -> compact complex. False -> take real part for readability.
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

        # Symbolic n or m => keep sums
        S_mj = lambda m_, j_: sp.functions.combinatorial.numbers.stirling(m_, j_, kind=2)
        inner_sum = Sum(
            S_mj(m, j) * (
                (self.beta*z)**j * lam**m * dpsi_j(j) +
                (self.beta*sp.conjugate(z))**j * sp.conjugate(lam)**m * dphi_j(j)
            ),
            (j, 1, m)
        )
        total = pref * Sum(inner_sum, (s, 1, n_))
        if isinstance(m, int) or isinstance(m, sp.Integer):
            # If m numeric and n numeric, we could expand explicitly, but symbolic Sum is fine.
            pass
        if not complex_form:
            total = sp.re(total)
        return simplify(total)

# ======================================================================
# Generator Builder
# ======================================================================
class GeneratorBuilder:
    """
    Free-form generator construction from strings involving Y0,Y1,...,Yk, Ym.
    - LHS is built using unknown function y(x) and its derivatives.
    - RHS is built by substituting Theorem-based y(x) and y^{(k)}(x).
    - Supports wrappers: exp, log, sin, cos, tanh, sinh, cosh, sqrt, abs, etc.
    """
    def __init__(self, T: Theorem42) -> None:
        self.T = T
        self.x = T.x
        self.y = sp.Function('y')(self.x)
        # Allowed symbols/functions in user expressions
        self.allowed_funcs = {
            'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
            'sqrt': sp.sqrt, 'Abs': sp.Abs, 'abs': sp.Abs,
            'pi': sp.pi, 'E': sp.E
        }
        # Parsing transformations
        self._transforms = standard_transformations + (convert_xor, implicit_multiplication_application)

    def _parse(self, expr_str: str, local_dict: Dict[str, Any]) -> sp.Expr:
        return parse_expr(expr_str, local_dict=local_dict, transformations=self._transforms, evaluate=True)

    def build_LHS(self, expr_str: str, max_order: int, include_symbolic_m: bool) -> sp.Expr:
        """
        Build LHS expression as a SymPy expression in terms of y(x) and Derivatives(y(x),(x,k)).
        Placeholders:
            Y0 -> y(x), Y1 -> y'(x), ..., Yk -> y^{(k)}(x), Ym -> symbolic Y_m
        """
        local = dict(self.allowed_funcs)
        local['x'] = self.x
        local['Y0'] = self.y
        for k in range(1, max_order+1):
            local[f'Y{k}'] = sp.Derivative(self.y, (self.x, k))
        if include_symbolic_m:
            local['Ym'] = sp.Symbol('Y_m', complex=False)  # formal symbol (unknown-order derivative)
        return self._parse(expr_str, local)

    def build_RHS(self,
                  expr_str: str,
                  f_callable: Callable[[sp.Symbol], sp.Expr],
                  n_val: Union[int, sp.Symbol],
                  orders_needed: List[Union[int, str]],
                  m_for_Ym: Optional[Union[int, sp.Symbol]] = None,
                  complex_form: bool = True) -> sp.Expr:
        """
        Build RHS by substituting Theorem-based y(x) and y^{(k)}(x) into the same expression.
        - orders_needed: list like [0,2,'m'] if Y0, Y2, Ym appear.
        - m_for_Ym: None -> symbolic m, int -> concrete.
        """
        T = self.T
        # y(x) base
        y0 = T.y_base(f_callable, n_override=n_val)
        # Build mapping for Yk placeholders
        subst: Dict[sp.Symbol, sp.Expr] = {}
        # We'll parse using symbolic placeholders and then substitute to avoid re-parsing wrappers.
        Y0 = sp.Symbol('Y0')
        local = dict(self.allowed_funcs)
        local['x'] = self.x
        local['Y0'] = Y0
        # Add Yk symbols
        for k in [o for o in orders_needed if isinstance(o, int) and o>=1]:
            local[f'Y{k}'] = sp.Symbol(f'Y{k}')
        if 'm' in orders_needed:
            local['Ym'] = sp.Symbol('Ym')
        # Parse
        parsed = self._parse(expr_str, local)
        # Build substitutions
        subst[Y0] = y0
        for k in [o for o in orders_needed if isinstance(o, int) and o>=1]:
            yk = T.y_derivative(f_callable, m=k, n_override=n_val, complex_form=complex_form)
            subst[sp.Symbol(f'Y{k}')] = yk
        if 'm' in orders_needed:
            y_m_expr = T.y_derivative(f_callable, m=None if m_for_Ym is None else m_for_Ym,
                                      n_override=n_val, complex_form=complex_form)
            subst[sp.Symbol('Ym')] = y_m_expr
        rhs = simplify(parsed.subs(subst))
        return rhs

    def infer_orders_needed(self, expr_str: str, max_order_hint: int) -> List[Union[int,str]]:
        present: List[Union[int,str]] = []
        for k in range(0, max_order_hint+1):
            if f"Y{k}" in expr_str:
                present.append(k)
        if "Ym" in expr_str:
            present.append('m')
        return present

# ======================================================================
# Generator Library (one example; user uses Builder for free-form)
# ======================================================================
class GeneratorLibrary:
    def __init__(self, T: Theorem42) -> None:
        self.T = T

    def linear_1(self, f) -> Tuple[sp.Expr, sp.Expr, sp.Expr, str]:
        x, a, b, n = self.T.x, self.T.alpha, self.T.beta, self.T.n
        y = self.T.y_base(f)
        ode_lhs = diff(y, x, 2) + y
        g = f(a + b*sp.exp(-x))
        rhs_explicit = sp.pi*( f(a+b) - f(a + b*sp.exp(-x)) ) \
            - sp.pi*b*sp.exp(-x)*diff(g, a, 1) - sp.pi*(b**2)*sp.exp(-2*x)*diff(g, a, 2)
        sol = sp.pi*( f(a+b) - f(a + b*sp.exp(-x)) )
        notes = "From Theorem 4.1 with n=1; RHS uses ∂/∂α on f(α+β e^{-x})."
        return (ode_lhs, rhs_explicit, sol, notes)

# ======================================================================
# ML: Generator Pattern Learner (enhanced)
# ======================================================================
class GeneratorPatternLearner:
    """
    ML pipeline for generator/ODE strings:
    - Supervised classification if labels provided (LogReg).
    - Unsupervised clustering (KMeans) for structure discovery.
    Graceful fallback if scikit-learn isn't available.
    """
    def __init__(self) -> None:
        self.available = HAVE_SK
        if self.available:
            self.clf = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=1, max_features=8000)),
                ("clf", LogisticRegression(max_iter=500))
            ])
            self.vec = TfidfVectorizer(ngram_range=(1,3), min_df=1, max_features=8000)
            self.kmeans = None
        else:
            self.clf = None
            self.vec = None
            self.kmeans = None

    def _s(self, e: sp.Expr) -> str:
        return _stringify_expr(e)

    def train_supervised(self, data: List[Tuple[sp.Expr, sp.Expr, int]]) -> Dict[str, Any]:
        if not self.available:
            return {"status":"fallback", "message":"scikit-learn not installed"}
        X = [self._s(Eq(l,r)) for (l,r,_) in data]
        y = [lab for (_,_,lab) in data]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(set(y))>1 else None)
        self.clf.fit(Xtr, ytr)
        yhat = self.clf.predict(Xte)
        try:
            report = classification_report(yte, yhat, output_dict=True)
        except Exception:
            report = {"acc": float(np.mean(np.array(yhat)==np.array(yte)))}
        return {"status":"ok", "report": report}

    def cluster(self, pairs: List[Tuple[sp.Expr, sp.Expr]], k: int=4) -> Dict[str, Any]:
        if not self.available:
            return {"status":"fallback", "message":"scikit-learn not installed"}
        X = [self._s(Eq(l,r)) for (l,r) in pairs]
        Xv = self.vec.fit_transform(X)
        self.kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = self.kmeans.fit_predict(Xv)
        return {"status":"ok", "labels": labels.tolist()}

    def predict(self, pairs: List[Tuple[sp.Expr, sp.Expr]]) -> List[int]:
        if not self.available:
            # baseline: nonlinearity -> 1, else 0
            out = []
            for l,r in pairs:
                out.append(1 if (_detect_nonlinearity(l) or _detect_nonlinearity(r)) else 0)
            return out
        X = [self._s(Eq(l,r)) for (l,r) in pairs]
        return list(self.clf.predict(X))

# ======================================================================
# DL: Novelty Detector (enhanced tiny Transformer)
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
        h = h.mean(dim=1)  # mean pool
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
        if len(ids) < self.max_len:
            ids += [0]*(self.max_len-len(ids))
        return np.array(ids, dtype=np.int64)

    def novelty_score(self, ode_strs: List[str]) -> List[float]:
        if not self.available:
            # heuristic novelty: normalized unique symbol count
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

    def quick_train(self, dataset: List[Tuple[str, float]], epochs: int=3, batch_size: int=16):
        if not self.available or len(dataset)==0:
            return {"status":"skipped"}
        self.model.train()
        xs = torch.stack([ torch.tensor(self.encode(s)) for s,_ in dataset ])
        ys = torch.tensor([ t for _,t in dataset ], dtype=torch.float32)
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
# Batch Manager
# ======================================================================
class BatchManager:
    """
    Generates ODE families by sweeping:
    - list of f(z) strings
    - list of expressions for generators (in Yk form)
    - n values and m (symbolic or numeric)
    Produces ODE pairs (LHS, RHS) along with metadata.
    """
    def __init__(self, T: Theorem42, builder: GeneratorBuilder) -> None:
        self.T = T
        self.builder = builder

    def build_pairs(self,
                    f_strings: List[str],
                    expr_strings: List[str],
                    n_values: List[Union[int, sp.Symbol]],
                    max_order: int,
                    include_symbolic_m: bool,
                    m_for_Ym: Optional[Union[int, sp.Symbol]]=None,
                    complex_form: bool = True) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for f_str in f_strings:
            try:
                f = _safe_eval_function_of_z(f_str)
            except Exception as e:
                continue
            for expr in expr_strings:
                orders_needed = self.builder.infer_orders_needed(expr, max_order_hint=max_order)
                lhs = self.builder.build_LHS(expr, max_order=max_order, include_symbolic_m=include_symbolic_m)
                for n_val in n_values:
                    rhs = self.builder.build_RHS(expr, f, n_val=n_val,
                                                 orders_needed=orders_needed,
                                                 m_for_Ym=m_for_Ym,
                                                 complex_form=complex_form)
                    results.append({
                        "f": f_str,
                        "expr": expr,
                        "n": n_val,
                        "lhs": simplify(lhs),
                        "rhs": simplify(rhs),
                    })
        return results

# ======================================================================
# Streamlit Pages
# ======================================================================
def page_overview():
    st.title("Master Generators — Free Generator Builder + Batch ML/DL")
    st.markdown("""
This app implements the research program for **Master Generators** with:
- New **Theorem 4.2** in compact Stirling-number form (symbolic `n`, symbolic or numeric `m`).
- A **free-form Generator Builder**: compose expressions with `Y0, Y1, Y2, ..., Ym` and wrappers
  like `exp()`, `sinh()`, `cosh()`, `log()`, polynomials, products, etc.
- The **RHS** is derived by *plugging the Theorem-based* \( y(x) \) and \( y^{(k)}(x) \) into the same expression,
  hence the constructed \( y(x) \) is a solution of the generated ODE.
- **Batch** generation of ODE families by sweeping `f(z)`, `n`, `m`, and expressions.
- **Full ML/DL**: classification/clustering and novelty scoring on generated families.
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

    st.markdown("**General derivative \(y^{(m)}(x)\) using the new Theorem 4.2:**")
    y_m = T.y_derivative(f, m=m_val, n_override=(n_val if use_symbolic_n else int(n_val)), complex_form=complex_form)
    st.latex(_latex(y_m))
    st.caption("Tip: switch off 'complex form' to display a real-valued trig combination (may be slower).")

def page_builder():
    st.header("Free Generator Builder")
    x = sp.Symbol('x', real=True)
    alpha = sp.Symbol('alpha', real=True)
    beta = sp.Symbol('beta', positive=True)
    n_sym = sp.Symbol('n', integer=True, positive=True)
    T = Theorem42(x=x, alpha=alpha, beta=beta, n=n_sym)
    B = GeneratorBuilder(T)

    st.markdown("""
Enter a generator expression using `Y0, Y1, Y2, ...` and optionally `Ym` (unspecified order).
Examples:
- `Y2 + Y0`  (i.e., \(y'' + y\))
- `exp(Y2) + sinh(Ym)`
- `Y1**2 + cosh(Y2) - 3*Y0`
Allowed wrappers: `exp, log, sin, cos, tan, sinh, cosh, tanh, sqrt, Abs`.
""")

    expr_str = st.text_input("Generator expression (in terms of Yk):", value="exp(Y2) + sinh(Ym)")
    max_order = st.number_input("Max explicit derivative order k available (for Yk):", min_value=0, max_value=20, value=3, step=1)
    include_Ym = st.checkbox("Allow symbolic Ym (unspecified derivative order)", value=True)
    use_symbolic_n = st.checkbox("Use symbolic n", value=True)
    n_val = n_sym if use_symbolic_n else st.number_input("n (integer ≥ 1):", min_value=1, max_value=64, value=2, step=1)
    use_symbolic_m = st.checkbox("Use symbolic m for Ym", value=True)
    m_for_Ym = None if use_symbolic_m else st.number_input("Concrete m value for Ym:", min_value=1, max_value=12, value=3, step=1)
    complex_form = st.checkbox("Use compact complex form", value=True)

    f_str = st.text_input("Enter f(z):", value="z", help="Pick any analytic f: z, sin(z), exp(z), cosh(z), ...")
    f = _safe_eval_function_of_z(f_str)

    # Build LHS
    try:
        lhs_expr = B.build_LHS(expr_str, max_order=max_order, include_symbolic_m=include_Ym)
        st.subheader("LHS (formal)")
        st.latex(_latex(lhs_expr))
    except Exception as e:
        st.error(f"LHS parse/build error: {e}")
        return

    # Build RHS by Theorem substitution
    try:
        needed = B.infer_orders_needed(expr_str, max_order_hint=max_order)
        rhs_expr = B.build_RHS(expr_str, f_callable=f, n_val=(n_val if use_symbolic_n else int(n_val)),
                               orders_needed=needed, m_for_Ym=(None if use_symbolic_m else int(m_for_Ym)),
                               complex_form=complex_form)
        st.subheader("RHS (from Theorem 4.1 & 4.2)")
        st.latex(_latex(rhs_expr))
    except Exception as e:
        st.error(f"RHS build error: {e}")
        return

    st.subheader("Generated ODE")
    st.latex(_latex(Eq(lhs_expr, rhs_expr)))

    # Also expose the exact solution y(x) being enforced by construction
    y_sol = T.y_base(f, n_override=(n_val if use_symbolic_n else int(n_val)))
    st.subheader("Exact solution enforced by construction")
    st.latex(_latex(sp.Eq(sp.Function('y')(x), y_sol)))

def page_batch_ml_dl():
    st.header("Batch Generators, ML & DL")
    x = sp.Symbol('x', real=True)
    alpha = sp.Symbol('alpha', real=True)
    beta = sp.Symbol('beta', positive=True)
    n_sym = sp.Symbol('n', integer=True, positive=True)
    T = Theorem42(x=x, alpha=alpha, beta=beta, n=n_sym)
    B = GeneratorBuilder(T)
    M = BatchManager(T, B)

    st.subheader("Batch Inputs")
    f_list_str = st.text_input("Enter f(z) list (comma or newline separated):", value="z, sin(z), exp(z), cosh(z)")
    exprs_str = st.text_input("Enter generator expressions (semicolon or newline separated):",
                              value="Y2 + Y0; exp(Y2) + sinh(Ym); Y1**2 + cosh(Y2) - 3*Y0")
    n_vals_str = st.text_input("n values (comma separated, ints only for batch):", value="1,2,3")
    max_order = st.number_input("Max explicit order k (Yk) to consider:", min_value=0, max_value=20, value=3, step=1)
    include_Ym = st.checkbox("Allow Ym (symbolic) in batch", value=True)
    use_symbolic_m = st.checkbox("Use symbolic m in batch", value=True)
    m_for_Ym = None if use_symbolic_m else st.number_input("Concrete m for Ym in batch:", min_value=1, max_value=12, value=3, step=1)
    complex_form = st.checkbox("Use compact complex form (batch)", value=True)

    # Parse lists
    f_strings = [s.strip() for s in f_list_str.replace("\n", ",").split(",") if s.strip()]
    expr_strings = [s.strip() for s in exprs_str.replace("\n", ";").split(";") if s.strip()]
    try:
        n_values = [int(s.strip()) for s in n_vals_str.split(",") if s.strip()]
    except Exception:
        st.error("Please provide integer n values for batch.")
        return

    # Build pairs
    pairs = M.build_pairs(
        f_strings=f_strings,
        expr_strings=expr_strings,
        n_values=n_values,
        max_order=max_order,
        include_symbolic_m=include_Ym,
        m_for_Ym=None if use_symbolic_m else int(m_for_Ym),
        complex_form=complex_form
    )
    st.success(f"Generated {len(pairs)} ODEs.")
    if len(pairs)==0:
        return

    # Show a preview
    st.subheader("Preview (first 3)")
    for row in pairs[:3]:
        st.markdown(f"**f(z)** = `{row['f']}`, **expr** = `{row['expr']}`, **n** = {row['n']}")
        st.latex(_latex(Eq(row['lhs'], row['rhs'])))

    # ====== ML ======
    st.subheader("ML — Supervised (LogReg) and Unsupervised (KMeans)")
    learner = GeneratorPatternLearner()
    # Build heuristic labels: 1 if nonlinearity, 0 otherwise (for demo)
    data = [(d['lhs'], d['rhs'], 1 if (_detect_nonlinearity(d['lhs']) or _detect_nonlinearity(d['rhs'])) else 0) for d in pairs]
    ml_res = learner.train_supervised(data)
    st.json(ml_res)

    # Cluster
    clus_res = learner.cluster([(d['lhs'], d['rhs']) for d in pairs], k=min(6, max(2, len(pairs)//3)))
    st.json(clus_res if clus_res['status']=="ok" else {"status":"unsupervised-fallback"})

    # ====== DL ======
    st.subheader("DL — Novelty Detector (tiny Transformer)")
    det = NoveltyDetector()
    as_strings = [ _stringify_expr(Eq(d['lhs'], d['rhs'])) for d in pairs[:32] ]  # limit for speed
    scores = det.novelty_score(as_strings)
    st.write({ i: float(scores[i]) for i in range(len(scores)) })

    # Optional quick training (toy targets: higher score for expressions with Ym or exp/cosh)
    toy_targets = []
    for s in as_strings:
        tgt = 0.5
        if "Ym" in s: tgt += 0.25
        if "exp(" in s or "cosh(" in s or "sinh(" in s: tgt += 0.15
        toy_targets.append(min(1.0, tgt))
    tr_res = det.quick_train(list(zip(as_strings, toy_targets)), epochs=2, batch_size=8)
    st.json(tr_res)

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

def main():
    st.set_page_config(page_title="Master Generators App", layout="wide")
    with st.sidebar:
        st.title("Master Generators")
        page = st.radio("Navigate", [
            "Overview",
            "Theorem 4.2",
            "Generator Builder",
            "Batch + ML/DL",
            "API Client"
        ])
        st.caption("Free-form generators, symbolic n and m, batch ODEs, ML & DL.")

    if page == "Overview":
        page_overview()
    elif page == "Theorem 4.2":
        page_theorem42()
    elif page == "Generator Builder":
        page_builder()
    elif page == "Batch + ML/DL":
        page_batch_ml_dl()
    elif page == "API Client":
        page_api_client()

if __name__ == "__main__":
    main()
