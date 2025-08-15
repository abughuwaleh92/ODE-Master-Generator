
# -*- coding: utf-8 -*-
"""
API Server — Master Generators (FastAPI)
=======================================

Mirrors the Streamlit Builder with programmatic endpoints:
- /health
- /theorem/y (y(x) from Theorem 4.1)
- /theorem/derivative (y^(m)(x) from new Theorem 4.2)
- /generator/build (free-form generator: build LHS, RHS via theorem, and the exact solution)
- /batch/build (build many ODEs at once)
- /ml/supervised/train (optional labels, otherwise heuristic)
- /ml/unsupervised/cluster
- /dl/novelty/score
- /dl/novelty/train

Run:
    uvicorn api_server:app --reload --port 8000
"""

import os
import sys
import json
import math
import cmath
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sympy as sp
from sympy import Eq
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, convert_xor, implicit_multiplication_application
)

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

# =====================================================================================
# Utilities
# =====================================================================================
def _latex(expr: sp.Expr) -> str:
    try:
        return sp.latex(sp.simplify(expr))
    except Exception:
        return sp.latex(expr)

def _stringify_expr(e: sp.Expr) -> str:
    try:
        return sp.srepr(sp.simplify(e))
    except Exception:
        return sp.srepr(e)

def _safe_eval_function_of_z(expr_str: str) -> sp.Function:
    z = sp.Symbol('z', complex=True)
    safe_ns = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
    safe_ns.update({'z': z, 'i': sp.I, 'I': sp.I, 'pi': sp.pi, 'PI': sp.pi, 'E': sp.E})
    try:
        expr = eval(expr_str, {"__builtins__": {}}, safe_ns)
    except Exception as e:
        raise ValueError(f"Could not parse f(z) from '{expr_str}': {e}")
    return sp.Lambda(z, expr)

def _safe_eval_sympy(expr_str: str, extra: Optional[Dict[str,Any]]=None) -> sp.Expr:
    safe = {k: getattr(sp, k) for k in dir(sp) if not k.startswith('_')}
    if extra:
        safe.update(extra)
    try:
        return eval(expr_str, {"__builtins__": {}}, safe)
    except Exception:
        # treat as a symbol
        return sp.Symbol(expr_str)

def _detect_nonlinearity(expr: sp.Expr) -> bool:
    s = _stringify_expr(expr)
    nonlin_signals = [
        "Pow(Derivative", "exp(Derivative", "log(Derivative", "sin(Derivative",
        "cos(Derivative", "sinh(Derivative", "cosh(Derivative"
    ]
    return any(tok in s for tok in nonlin_signals)

# =====================================================================================
# Theorem 4.2 (compact Stirling-number form) + Builder + Batch + ML/DL
# =====================================================================================
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

    def psi(self, f: sp.Function, s: sp.Symbol) -> sp.Expr:
        return f(self.alpha + self.beta*self.zeta(s))

    def phi(self, f: sp.Function, s: sp.Symbol) -> sp.Expr:
        w = self.omega(s)
        zbar = sp.exp(-self.x*sp.sin(w)) * sp.exp(-sp.I*self.x*sp.cos(w))
        return f(self.alpha + self.beta*zbar)

    def y_base(self, f: sp.Function, n_override: Optional[Union[int, sp.Symbol]] = None) -> sp.Expr:
        n_ = self.n if n_override is None else n_override
        s = sp.Symbol('s', integer=True, positive=True)
        expr = (sp.pi/(2*n_)) * sp.summation(
            2*f(self.alpha + self.beta) - self.psi(f, s) - self.phi(f, s),
            (s, 1, n_)
        )
        return sp.simplify(expr)

    def y_derivative(self,
                     f: sp.Function,
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
        dpsi_j = lambda j: sp.diff(psi, self.alpha, j)
        dphi_j = lambda j: sp.diff(phi, self.alpha, j)
        pref = -(sp.pi/(2*n_))

        S_mj = lambda m_, j_: sp.functions.combinatorial.numbers.stirling(m_, j_, kind=2)
        inner = sp.Sum(
            S_mj(m, j) * (
                (self.beta*z)**j * lam**m * dpsi_j(j) +
                (self.beta*sp.conjugate(z))**j * sp.conjugate(lam)**m * dphi_j(j)
            ),
            (j, 1, m)
        )
        total = pref * sp.Sum(inner, (s, 1, n_))
        if not complex_form:
            total = sp.re(total)
        return sp.simplify(total)

class GeneratorBuilder:
    def __init__(self, T: Theorem42) -> None:
        self.T = T
        self.x = T.x
        self.y = sp.Function('y')(self.x)
        self.allowed_funcs = {
            'exp': sp.exp, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
            'sqrt': sp.sqrt, 'Abs': sp.Abs, 'abs': sp.Abs,
            'pi': sp.pi, 'E': sp.E
        }
        self._transforms = standard_transformations + (convert_xor, implicit_multiplication_application)

    def _parse(self, expr_str: str, local_dict: Dict[str, Any]) -> sp.Expr:
        return parse_expr(expr_str, local_dict=local_dict, transformations=self._transforms, evaluate=True)

    def build_LHS(self, expr_str: str, max_order: int, include_symbolic_m: bool) -> sp.Expr:
        local = dict(self.allowed_funcs)
        local['x'] = self.x
        local['Y0'] = self.y
        for k in range(1, max_order+1):
            local[f'Y{k}'] = sp.Derivative(self.y, (self.x, k))
        if include_symbolic_m:
            local['Ym'] = sp.Symbol('Y_m', complex=False)
        return self._parse(expr_str, local)

    def build_RHS(self,
                  expr_str: str,
                  f_callable: sp.Function,
                  n_val: Union[int, sp.Symbol],
                  orders_needed: List[Union[int, str]],
                  m_for_Ym: Optional[Union[int, sp.Symbol]] = None,
                  complex_form: bool = True) -> sp.Expr:
        T = self.T
        y0 = T.y_base(f_callable, n_override=n_val)
        subst: Dict[sp.Symbol, sp.Expr] = {}
        Y0 = sp.Symbol('Y0')
        local = dict(self.allowed_funcs)
        local['x'] = self.x
        local['Y0'] = Y0
        for k in [o for o in orders_needed if isinstance(o, int) and o>=1]:
            local[f'Y{k}'] = sp.Symbol(f'Y{k}')
        if 'm' in orders_needed:
            local['Ym'] = sp.Symbol('Ym')
        parsed = self._parse(expr_str, local)
        subst[Y0] = y0
        for k in [o for o in orders_needed if isinstance(o, int) and o>=1]:
            yk = T.y_derivative(f_callable, m=k, n_override=n_val, complex_form=complex_form)
            subst[sp.Symbol(f'Y{k}')] = yk
        if 'm' in orders_needed:
            y_m_expr = T.y_derivative(f_callable, m=None if m_for_Ym is None else m_for_Ym,
                                      n_override=n_val, complex_form=complex_form)
            subst[sp.Symbol('Ym')] = y_m_expr
        rhs = sp.simplify(parsed.subs(subst))
        return rhs

    def infer_orders_needed(self, expr_str: str, max_order_hint: int) -> List[Union[int,str]]:
        present: List[Union[int,str]] = []
        for k in range(0, max_order_hint+1):
            if f"Y{k}" in expr_str:
                present.append(k)
        if "Ym" in expr_str:
            present.append('m')
        return present

class BatchManager:
    def __init__(self, T: Theorem42, builder: GeneratorBuilder) -> None:
        self.T = T
        self.builder = builder

    def build_pairs(self,
                    f_strings: List[str],
                    expr_strings: List[str],
                    n_values: List[int],
                    max_order: int,
                    include_symbolic_m: bool,
                    m_for_Ym: Optional[Union[int, sp.Symbol]]=None,
                    complex_form: bool = True) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for f_str in f_strings:
            try:
                f = _safe_eval_function_of_z(f_str)
            except Exception:
                continue
            for expr in expr_strings:
                orders_needed = self.builder.infer_orders_needed(expr, max_order_hint=max_order)
                lhs = self.builder.build_LHS(expr, max_order=max_order, include_symbolic_m=include_symbolic_m)
                for n_val in n_values:
                    rhs = self.builder.build_RHS(expr, f_callable=f, n_val=int(n_val),
                                                 orders_needed=orders_needed,
                                                 m_for_Ym=m_for_Ym, complex_form=complex_form)
                    results.append({
                        "f": f_str,
                        "expr": expr,
                        "n": int(n_val),
                        "lhs": sp.simplify(lhs),
                        "rhs": sp.simplify(rhs),
                    })
        return results

# =================== ML / DL ===================
class GeneratorPatternLearner:
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
        strat = y if len(set(y))>1 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)
        self.clf.fit(Xtr, ytr)
        yhat = self.clf.predict(Xte)
        try:
            report = classification_report(yte, yhat, output_dict=True)
        except Exception:
            report = {"acc": float((yhat==yte).mean())}
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
            out = []
            for l,r in pairs:
                out.append(1 if (_detect_nonlinearity(l) or _detect_nonlinearity(r)) else 0)
            return out
        X = [self._s(Eq(l,r)) for (l,r) in pairs]
        return list(self.clf.predict(X))

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

    def encode(self, s: str):
        ids = [ self.vocab.get(ch, 1) for ch in s[:self.max_len] ]
        if len(ids) < self.max_len:
            ids += [0]*(self.max_len-len(ids))
        return ids

    def novelty_score(self, ode_strs: List[str]) -> List[float]:
        if not self.available:
            scores = []
            for s in ode_strs:
                uniq = len(set(s))
                scores.append( min(1.0, 0.1 + 0.9*uniq/max(10, len(s))) )
            return scores
        self.model.eval()
        out = []
        import torch
        with torch.no_grad():
            for s in ode_strs:
                x = torch.tensor(self.encode(s))[None, :]
                y = torch.sigmoid(self.model(x)).item()
                out.append(float(y))
        return out

    def quick_train(self, dataset: List[Tuple[str, float]], epochs: int=3, batch_size: int=16):
        if not self.available or len(dataset)==0:
            return {"status":"skipped"}
        import torch
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

# =====================================================================================
# FastAPI app
# =====================================================================================
app = FastAPI(title="Master Generators API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons
x = sp.Symbol('x', real=True)
alpha = sp.Symbol('alpha', real=True)
beta = sp.Symbol('beta', positive=True)
n_sym = sp.Symbol('n', integer=True, positive=True)

T = Theorem42(x=x, alpha=alpha, beta=beta, n=n_sym)
B = GeneratorBuilder(T)
BM = BatchManager(T, B)
LEARN = GeneratorPatternLearner()
DET = NoveltyDetector()

LAST_BATCH: List[Dict[str, Any]] = []  # store last built ODEs

# =====================================================================================
# Request models
# =====================================================================================
class TheoremYReq(BaseModel):
    f: str = Field(..., description="SymPy expression in z, e.g. 'sin(z)'")
    alpha: str = Field("alpha")
    beta: str = Field("beta")
    n: Union[int, str] = Field("symbolic", description="'symbolic' or integer")

class TheoremDerivReq(BaseModel):
    f: str
    alpha: str = "alpha"
    beta: str = "beta"
    n: Union[int, str] = "symbolic"
    m: Union[int, str] = "symbolic"
    complex_form: bool = True

class GeneratorBuildReq(BaseModel):
    expr: str = Field(..., description="Generator in Y-notation, e.g. 'exp(Y2)+sinh(Ym)'")
    f: str = "z"
    n: Union[int, str] = "symbolic"
    max_order: int = 3
    include_Ym: bool = True
    m_for_Ym: Optional[int] = None
    complex_form: bool = True
    alpha: str = "alpha"
    beta: str = "beta"

class BatchBuildReq(BaseModel):
    f_list: List[str]
    expr_list: List[str]
    n_values: List[int]
    max_order: int = 3
    include_Ym: bool = True
    m_for_Ym: Optional[int] = None
    complex_form: bool = True
    alpha: str = "alpha"
    beta: str = "beta"

class MLSupervisedReq(BaseModel):
    # If empty, will heuristically label LAST_BATCH (nonlinear=1 else 0)
    data: Optional[List[Dict[str, Any]]] = None  # each: {"lhs": str, "rhs": str, "label": int}

class MLClusterReq(BaseModel):
    k: int = 4
    # If pairs omitted, uses LAST_BATCH
    pairs: Optional[List[Dict[str, str]]] = None  # [{"lhs": str, "rhs": str}]

class DLScoreReq(BaseModel):
    # If strings omitted, uses LAST_BATCH
    strings: Optional[List[str]] = None

class DLTrainReq(BaseModel):
    dataset: List[Dict[str, Union[str, float]]]  # [{"s": "...", "t": 0.8}]
    epochs: int = 3
    batch_size: int = 16

# =====================================================================================
# Endpoints
# =====================================================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/theorem/y")
def theorem_y(req: TheoremYReq):
    f = _safe_eval_function_of_z(req.f)
    a = _safe_eval_sympy(req.alpha, {"alpha": alpha, "beta": beta, "x": x})
    b = _safe_eval_sympy(req.beta, {"alpha": alpha, "beta": beta, "x": x})
    n_val = n_sym if str(req.n).lower()=="symbolic" else int(req.n)
    T2 = Theorem42(x=x, alpha=a, beta=b, n=n_val)
    y_expr = T2.y_base(f, n_override=n_val)
    return {"y": {"latex": _latex(y_expr), "srepr": _stringify_expr(y_expr), "str": str(y_expr)}}

@app.post("/theorem/derivative")
def theorem_derivative(req: TheoremDerivReq):
    f = _safe_eval_function_of_z(req.f)
    a = _safe_eval_sympy(req.alpha, {"alpha": alpha, "beta": beta, "x": x})
    b = _safe_eval_sympy(req.beta, {"alpha": alpha, "beta": beta, "x": x})
    n_val = n_sym if str(req.n).lower()=="symbolic" else int(req.n)
    m_val = T.m_sym if str(req.m).lower()=="symbolic" else int(req.m)
    T2 = Theorem42(x=x, alpha=a, beta=b, n=n_val)
    y_m = T2.y_derivative(f, m=m_val if isinstance(m_val, int) else None,
                          n_override=n_val, complex_form=req.complex_form)
    return {"y_m": {"latex": _latex(y_m), "srepr": _stringify_expr(y_m), "str": str(y_m)}}

@app.post("/generator/build")
def generator_build(req: GeneratorBuildReq):
    f = _safe_eval_function_of_z(req.f)
    a = _safe_eval_sympy(req.alpha, {"alpha": alpha, "beta": beta, "x": x})
    b = _safe_eval_sympy(req.beta, {"alpha": alpha, "beta": beta, "x": x})
    n_val = n_sym if str(req.n).lower()=="symbolic" else int(req.n)
    T2 = Theorem42(x=x, alpha=a, beta=b, n=n_val)
    builder = GeneratorBuilder(T2)

    lhs = builder.build_LHS(req.expr, max_order=req.max_order, include_symbolic_m=req.include_Ym)
    needed = builder.infer_orders_needed(req.expr, max_order_hint=req.max_order)
    rhs = builder.build_RHS(req.expr, f_callable=f, n_val=n_val,
                            orders_needed=needed, m_for_Ym=req.m_for_Ym,
                            complex_form=req.complex_form)
    y_sol = T2.y_base(f, n_override=n_val)
    return {
        "lhs": {"latex": _latex(lhs), "srepr": _stringify_expr(lhs), "str": str(lhs)},
        "rhs": {"latex": _latex(rhs), "srepr": _stringify_expr(rhs), "str": str(rhs)},
        "equation_latex": _latex(sp.Eq(lhs, rhs)),
        "solution": {"latex": _latex(sp.Eq(sp.Function('y')(x), y_sol)), "srepr": _stringify_expr(y_sol), "str": str(y_sol)}
    }

@app.post("/batch/build")
def batch_build(req: BatchBuildReq):
    global LAST_BATCH
    a = _safe_eval_sympy(req.alpha, {"alpha": alpha, "beta": beta, "x": x})
    b = _safe_eval_sympy(req.beta, {"alpha": alpha, "beta": beta, "x": x})
    T2 = Theorem42(x=x, alpha=a, beta=b, n=n_sym)  # n is set per-item
    builder = GeneratorBuilder(T2)
    bm = BatchManager(T2, builder)
    pairs = bm.build_pairs(
        f_strings=req.f_list,
        expr_strings=req.expr_list,
        n_values=[int(v) for v in req.n_values],
        max_order=req.max_order,
        include_symbolic_m=req.include_Ym,
        m_for_Ym=req.m_for_Ym,
        complex_form=req.complex_form
    )
    # serialize
    out = []
    for d in pairs:
        out.append({
            "f": d["f"],
            "expr": d["expr"],
            "n": d["n"],
            "lhs": {"latex": _latex(d["lhs"]), "srepr": _stringify_expr(d["lhs"]), "str": str(d["lhs"])},
            "rhs": {"latex": _latex(d["rhs"]), "srepr": _stringify_expr(d["rhs"]), "str": str(d["rhs"])},
            "equation_latex": _latex(Eq(d["lhs"], d["rhs"])),
        })
    LAST_BATCH = pairs
    return {"count": len(out), "items": out}

@app.post("/ml/supervised/train")
def ml_supervised_train(req: MLSupervisedReq):
    if req.data:
        data = []
        for item in req.data:
            lhs = _safe_eval_sympy(item["lhs"], {"x": x})
            rhs = _safe_eval_sympy(item["rhs"], {"x": x})
            lab = int(item["label"])
            data.append((lhs, rhs, lab))
    else:
        if not LAST_BATCH:
            return {"status":"empty", "message":"No LAST_BATCH to train on"}
        data = []
        for d in LAST_BATCH:
            lhs, rhs = d["lhs"], d["rhs"]
            lab = 1 if (_detect_nonlinearity(lhs) or _detect_nonlinearity(rhs)) else 0
            data.append((lhs, rhs, lab))
    learner = LEARN
    res = learner.train_supervised(data)
    return res

@app.post("/ml/unsupervised/cluster")
def ml_cluster(req: MLClusterReq):
    if req.pairs:
        pairs = []
        for item in req.pairs:
            lhs = _safe_eval_sympy(item["lhs"], {"x": x})
            rhs = _safe_eval_sympy(item["rhs"], {"x": x})
            pairs.append((lhs, rhs))
    else:
        if not LAST_BATCH:
            return {"status":"empty", "message":"No LAST_BATCH to cluster"}
        pairs = [(d["lhs"], d["rhs"]) for d in LAST_BATCH]
    res = LEARN.cluster(pairs, k=int(req.k))
    return res

@app.post("/dl/novelty/score")
def dl_novelty_score(req: DLScoreReq):
    if req.strings:
        strings = req.strings
    else:
        if not LAST_BATCH:
            return {"status":"empty", "message":"No LAST_BATCH to score"}
        strings = [_stringify_expr(Eq(d["lhs"], d["rhs"])) for d in LAST_BATCH]
    scores = DET.novelty_score(strings)
    return {"scores": [float(s) for s in scores]}

@app.post("/dl/novelty/train")
def dl_novelty_train(req: DLTrainReq):
    ds = [(item["s"], float(item["t"])) for item in req.dataset]
    res = DET.quick_train(ds, epochs=int(req.epochs), batch_size=int(req.batch_size))
    return res
