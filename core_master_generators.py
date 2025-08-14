
# -*- coding: utf-8 -*-
"""
core_master_generators.py
=========================

Shared core for Master Generators:
- Theorem 4.2 (compact Stirling-number form)
- GeneratorBuilder
- GeneratorLibrary (presets)
- ML (GeneratorPatternLearner) with enriched labels
- DL (NoveltyDetector) for triage

Safe to import from Streamlit app and FastAPI server.
"""

import os, re, json, math, cmath, types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import sympy as sp
import numpy as np

# Optional ML/DL
try:
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
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


# =============================================================================
# Utilities
# =============================================================================
def safe_latex(expr: sp.Expr) -> str:
    try:
        return sp.latex(sp.simplify(expr))
    except Exception:
        return sp.latex(expr)

def safe_eval_f_of_z(expr_str: str) -> sp.Lambda:
    z = sp.Symbol('z', complex=True)
    ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
    ns.update({'z': z, 'i': sp.I, 'I': sp.I, 'pi': sp.pi, 'PI': sp.pi, 'E': sp.E})
    try:
        expr = eval(expr_str, {"__builtins__": {}}, ns)
    except Exception as e:
        raise ValueError(f"Invalid f(z): {e}")
    return sp.Lambda(z, expr)

def stringify_expr(expr: sp.Expr) -> str:
    return sp.srepr(sp.simplify(expr))

def count_symbolic_complexity(expr: sp.Expr) -> Dict[str, Any]:
    """
    Heuristic complexity metrics for triage.
    """
    s = stringify_expr(expr)
    deriv_orders = [int(m) for m in re.findall(r"Derivative\(.*?,\s*\(Symbol\('x'\),\s*(\d+)\)\)", s)]
    max_order = max(deriv_orders) if deriv_orders else 0
    nonlin_tokens = ['Pow(Derivative', 'sin', 'cos', 'sinh', 'cosh', 'exp', 'log']
    uses_nonlin = any(tok in s for tok in nonlin_tokens)
    pantograph_like = 'Function(' in s and 'x' in s and '/Symbol' in s  # crude
    uniq = len(set(s))
    return {
        "max_order": int(max_order),
        "uses_nonlin": bool(uses_nonlin),
        "pantograph_like": bool(pantograph_like),
        "symbol_length": len(s),
        "unique_chars": int(uniq)
    }

# =============================================================================
# Theorem 4.2 (compact form)
# =============================================================================
@dataclass
class Theorem42:
    x: sp.Symbol = sp.Symbol('x', real=True)
    alpha: sp.Symbol = sp.Symbol('alpha', real=True)
    beta: sp.Symbol = sp.Symbol('beta', positive=True)
    n: Union[int, sp.Symbol] = sp.Symbol('n', integer=True, positive=True)
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
        n_ = self.n if n_override is None else n_override
        s = sp.Symbol('s', integer=True, positive=True)
        expr = (sp.pi/(2*n_)) * sp.summation(
            2*f(self.alpha + self.beta) - self.psi(f, s) - self.phi(f, s),
            (s, 1, n_)
        )
        return sp.simplify(expr)

    def y_derivative(self,
                     f: Callable,
                     m: Optional[Union[int, sp.Symbol]] = None,
                     n_override: Optional[Union[int, sp.Symbol]] = None,
                     complex_form: bool = True) -> sp.Expr:
        n_ = self.n if n_override is None else n_override
        s = sp.Symbol('s', integer=True, positive=True)
        if m is None:
            m = self.m_sym

        j = sp.Symbol('j', integer=True, positive=True)
        lam = self.lambda_phase(s)
        z = self.zeta(s)
        psi = self.psi(f, s)
        phi = self.phi(f, s)
        dpsi_j = lambda jj: sp.diff(psi, self.alpha, jj)
        dphi_j = lambda jj: sp.diff(phi, self.alpha, jj)
        pref = -(sp.pi/(2*n_))

        S = sp.functions.combinatorial.numbers.stirling
        inner = sp.Sum(
            S(m, j, kind=2) * (
                (self.beta*z)**j * lam**m * dpsi_j(j)
                + (self.beta*sp.conjugate(z))**j * sp.conjugate(lam)**m * dphi_j(j)
            ),
            (j, 1, m)
        )
        total = pref * sp.Sum(inner, (s, 1, n_))

        # Expand if both are numeric
        if isinstance(m, (int, sp.Integer)) and isinstance(n_, (int, sp.Integer)):
            total = sp.simplify(total.doit())

        if not complex_form:
            total = sp.re(total)
        return sp.simplify(total)


# =============================================================================
# Generator Builder
# =============================================================================
class GeneratorBuilder:
    def __init__(self, T: Theorem42, f: Callable, n_val: Union[int, sp.Symbol],
                 m_val: Optional[Union[int, sp.Symbol]], complex_form: bool):
        self.T = T
        self.f = f
        self.n_val = n_val
        self.m_val = m_val
        self.complex_form = complex_form
        self.x = T.x
        self.yfun = sp.Function('y')(self.x)
        self._cache: Dict[Union[int, str], sp.Expr] = {}

    def _parse_orders(self, template: str) -> Tuple[List[int], bool]:
        orders = sorted({int(m) for m in re.findall(r'Dy(\d+)', template)})
        uses_dym = bool(re.search(r'\bDym\b', template))
        return orders, uses_dym

    def _y_exact(self) -> sp.Expr:
        key = "y"
        if key not in self._cache:
            self._cache[key] = sp.simplify(self.T.y_base(self.f, n_override=self.n_val))
        return self._cache[key]

    def _y_deriv_exact(self, m: Union[int, sp.Symbol]) -> sp.Expr:
        key = f"m={m}"
        if key not in self._cache:
            self._cache[key] = sp.simplify(self.T.y_derivative(self.f, m=m, n_override=self.n_val,
                                                               complex_form=self.complex_form))
        return self._cache[key]

    def _namespace(self, lhs: bool, orders: List[int], uses_dym: bool) -> Dict[str, Any]:
        ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
        ns.update({"x": self.x, "alpha": self.T.alpha, "beta": self.T.beta, "pi": sp.pi, "I": sp.I})
        if lhs:
            ns["y"] = self.yfun
            for k in orders:
                ns[f"Dy{k}"] = sp.Derivative(self.yfun, (self.x, k))
            if uses_dym:
                if isinstance(self.m_val, (int, sp.Integer)):
                    ns["Dym"] = sp.Derivative(self.yfun, (self.x, int(self.m_val)))
                else:
                    ns["Dym"] = sp.Symbol("Dym")
        else:
            ns["y"] = self._y_exact()
            for k in orders:
                ns[f"Dy{k}"] = self._y_deriv_exact(k)
            if uses_dym:
                mv = self.m_val if self.m_val is not None else self.T.m_sym
                ns["Dym"] = self._y_deriv_exact(mv)
        return ns

    def build(self, template: str) -> Tuple[sp.Expr, sp.Expr]:
        # Preprocess aliases: y^(m) -> Dym, y^(k) (integer) -> Dyk
        def _alias(s: str) -> str:
            s = re.sub(r"y\^\(\s*m\s*\)", "Dym", s)
            s = re.sub(r"y\^\(\s*(\d+)\s*\)", lambda m: f"Dy{m.group(1)}", s)
            return s
        template = _alias(template)

        orders, uses_dym = self._parse_orders(template)
        ns_lhs = self._namespace(lhs=True, orders=orders, uses_dym=uses_dym)
        ns_rhs = self._namespace(lhs=False, orders=orders, uses_dym=uses_dym)
        try:
            lhs = eval(template, {"__builtins__": {}}, ns_lhs)
        except Exception as e:
            raise ValueError(f"LHS parse error: {e}")
        try:
            rhs = eval(template, {"__builtins__": {}}, ns_rhs)
        except Exception as e:
            raise ValueError(f"RHS construction error: {e}")
        return sp.simplify(lhs), sp.simplify(rhs)


# =============================================================================
# Preset library (recipes)
# =============================================================================
class GeneratorLibrary:
    def __init__(self, T: Theorem42):
        self.T = T

    def pantograph_linear(self, f: Callable) -> Dict[str, sp.Expr]:
        x, a, b, n = self.T.x, self.T.alpha, self.T.beta, self.T.n
        y = self.T.y_base(f)
        # Example: y''(x) + y(x/a) - y(x)
        lhs = sp.diff(y, x, 2) + sp.Function('y')(x/sp.Symbol('a')) - sp.Function('y')(x)
        g = f(a + b*sp.exp(-x))
        rhs = sp.pi*( f(a+b) - f(a + b*sp.exp(-x)) ) \
              - sp.pi*b*sp.exp(-x)*sp.diff(g, a, 1) \
              - sp.pi*b**2*sp.exp(-2*x)*sp.diff(g, a, 2)
        return {"lhs": sp.simplify(lhs), "rhs": sp.simplify(rhs), "solution": sp.simplify(self.T.y_base(f))}

    def multi_order_mix(self, f: Callable, orders=(1,2,3)) -> Dict[str, sp.Expr]:
        x = self.T.x
        y = self.T.y_base(f)
        lhs = sum(sp.diff(y, x, k) for k in orders) + y
        # naive RHS via substitution of exact y into same functional form:
        rhs = sum(self.T.y_derivative(f, m=k, complex_form=True) for k in orders) + self.T.y_base(f)
        return {"lhs": sp.simplify(lhs), "rhs": sp.simplify(rhs), "solution": sp.simplify(self.T.y_base(f))}

    def nonlinear_wrap(self, f: Callable, wrap: Callable[[sp.Expr], sp.Expr]) -> Dict[str, sp.Expr]:
        x = self.T.x
        y = self.T.y_base(f)
        lhs = wrap(sp.diff(y, x, 2)) + y
        rhs = wrap(self.T.y_derivative(f, m=2, complex_form=True)) + self.T.y_base(f)
        return {"lhs": sp.simplify(lhs), "rhs": sp.simplify(rhs), "solution": sp.simplify(self.T.y_base(f))}


# =============================================================================
# ML with enriched labels
# =============================================================================
class GeneratorPatternLearner:
    """
    Multi-head classifier:
      - linearity: {0,1}
      - stiffness: {0,1,2,...} (user-defined)
      - solvability: {0,1,2,...} (user-defined)
    Each head is a separate pipeline; falls back to heuristics if sklearn missing.
    """
    def __init__(self) -> None:
        self.available = HAVE_SK
        if self.available:
            base = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=1, max_features=20000)),
                             ("clf", LogisticRegression(max_iter=800))])
            self.pipe_linearity = Pipeline(base.steps.copy())
            self.pipe_stiffness = Pipeline(base.steps.copy())
            self.pipe_solvability = Pipeline(base.steps.copy())
        else:
            self.pipe_linearity = self.pipe_stiffness = self.pipe_solvability = None

    def _stringify(self, L: sp.Expr, R: sp.Expr) -> str:
        return stringify_expr(sp.Eq(L, R))

    def train(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        dataset: list of { 'lhs': sympy.Expr, 'rhs': sympy.Expr,
                           'labels': {'linearity': int, 'stiffness': int, 'solvability': int} }
        """
        if not dataset:
            return {"status": "error", "message": "empty dataset"}
        X = [ self._stringify(d['lhs'], d['rhs']) for d in dataset ]
        Y_lin = [ int(d['labels'].get('linearity', 0)) for d in dataset ]
        Y_sti = [ int(d['labels'].get('stiffness', 0)) for d in dataset ]
        Y_sol = [ int(d['labels'].get('solvability', 0)) for d in dataset ]
        if not self.available:
            return {"status": "fallback", "message": "sklearn not available; using heuristics."}
        self.pipe_linearity.fit(X, Y_lin)
        self.pipe_stiffness.fit(X, Y_sti)
        self.pipe_solvability.fit(X, Y_sol)
        return {"status": "ok"}

    def predict(self, pairs: List[Tuple[sp.Expr, sp.Expr]]) -> List[Dict[str, int]]:
        X = [ self._stringify(L, R) for (L,R) in pairs ]
        out = []
        if not self.available:
            # simple heuristics
            for x in X:
                lin = 0 if ("Pow(Derivative" not in x and "sin" not in x and "exp" not in x and "sinh" not in x) else 1
                sti = 1 if "Derivative" in x and "exp" in x else 0
                sol = 1 if "y" in x else 0
                out.append({"linearity": lin, "stiffness": sti, "solvability": sol})
            return out
        out.append({"linearity": int(self.pipe_linearity.predict(X)[0]) if X else 0,
                    "stiffness": int(self.pipe_stiffness.predict(X)[0]) if X else 0,
                    "solvability": int(self.pipe_solvability.predict(X)[0]) if X else 0})
        # For multiple pairs
        if len(X)>1:
            out = [{"linearity": int(a), "stiffness": int(b), "solvability": int(c)}
                   for a,b,c in zip(self.pipe_linearity.predict(X),
                                    self.pipe_stiffness.predict(X),
                                    self.pipe_solvability.predict(X))]
        return out


# =============================================================================
# DL Novelty Detector (Tiny Transformer)
# =============================================================================
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=512, d_model=128, nhead=4, num_layers=2, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model)*0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.embed(x) + self.pos[:, :x.size(1), :]
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
            return [min(1.0, 0.1 + 0.9*len(set(s))/max(10, len(s))) for s in ode_strs]
        self.model.eval()
        outs = []
        with torch.no_grad():
            for s in ode_strs:
                x = torch.tensor(self.encode(s))[None, :]
                y = torch.sigmoid(self.model(x)).item()
                outs.append(float(y))
        return outs

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


# =============================================================================
# Helpers to (de)serialize SymPy expressions for API/DB
# =============================================================================
def expr_from_srepr(s: str) -> sp.Expr:
    return sp.sympify(s, locals={})

def ode_to_json(lhs: sp.Expr, rhs: sp.Expr, meta: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    data = {
        "lhs_srepr": stringify_expr(lhs),
        "rhs_srepr": stringify_expr(rhs),
        "lhs_latex": safe_latex(lhs),
        "rhs_latex": safe_latex(rhs),
    }
    if meta:
        data["meta"] = meta
    return data
