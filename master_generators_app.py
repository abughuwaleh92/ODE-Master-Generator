"""
Master Generators for ODEs — Exact Symbolic Edition
---------------------------------------------------
- Theorem 4.1 (exact) and Theorem 4.2 (Stirling-number compact form)
- Free-Form Generator Builder: mix terms like sinh(y'), exp(y'''''''), ln(y''), power(...)
- Arbitrary derivative orders (inner derivative of y, and an optional outer derivative of the whole wrapped term)
- Argument scaling/shift (pantograph-like y(x/a + shift))
- Backward compatible with src/ services (ML/DL, novelty, batch, export, viz)

This file does not change your src/ package; it adapts to it.
"""

from __future__ import annotations

import os
import sys
import io
import json
import time
import logging
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sympy as sp
from sympy.core.function import AppliedUndef

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import zipfile

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# -----------------------------------------------------------------------------
# Ensure src/ on path
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -----------------------------------------------------------------------------
# Probe and import from src/ robustly
# -----------------------------------------------------------------------------
HAVE_SRC = True

def _probe_src():
    out = dict(
        MasterGenerator=None, EnhancedMasterGenerator=None, CompleteMasterGenerator=None,
        LinearGeneratorFactory=None, CompleteLinearGeneratorFactory=None,
        NonlinearGeneratorFactory=None, CompleteNonlinearGeneratorFactory=None,
        GeneratorConstructor=None, GeneratorSpecification=None,
        DerivativeTerm=None, DerivativeType=None, OperatorType=None,
        MasterTheoremSolver=None, MasterTheoremParameters=None, ExtendedMasterTheorem=None,
        ODEClassifier=None, PhysicalApplication=None,
        BasicFunctions=None, SpecialFunctions=None,
        GeneratorPatternLearner=None, GeneratorVAE=None, GeneratorTransformer=None, create_model=None,
        MLTrainer=None, ODEDataset=None, ODEDataGenerator=None,
        GeneratorPattern=None, GeneratorPatternNetwork=None, GeneratorLearningSystem=None,
        ODENoveltyDetector=None, NoveltyAnalysis=None, ODETokenizer=None, ODETransformer=None,
        Settings=None, AppConfig=None, CacheManager=None, cached=None, ParameterValidator=None, UIComponents=None
    )
    candidates = [
        "src.generators.master_generator",
        "src.generators.linear_generators",
        "src.generators.nonlinear_generators",
        "src.generators.generator_constructor",
        "src.generators.master_theorem",
        "src.generators.ode_classifier",
        "src.functions.basic_functions",
        "src.functions.special_functions",
        "src.ml.pattern_learner",
        "src.ml.trainer",
        "src.ml.generator_learner",
        "src.dl.novelty_detector",
        "src.utils.config",
        "src.utils.cache",
        "src.utils.validators",
        "src.ui.components",
    ]
    def _imp(m):
        try:
            return __import__(m, fromlist=["*"])
        except Exception as e:
            logger.debug(f"Skip module {m}: {e}")
            return None

    mods = {m: _imp(m) for m in candidates}

    def _set(name, module_candidates):
        for m in module_candidates:
            mod = mods.get(m)
            if mod is None:
                continue
            if hasattr(mod, name):
                out[name] = getattr(mod, name)
                return

    # factories
    _set("MasterGenerator", ["src.generators.master_generator"])
    _set("EnhancedMasterGenerator", ["src.generators.master_generator"])
    _set("CompleteMasterGenerator", ["src.generators.master_generator"])

    _set("LinearGeneratorFactory", ["src.generators.linear_generators", "src.generators.master_generator"])
    _set("CompleteLinearGeneratorFactory", ["src.generators.master_generator", "src.generators.linear_generators"])
    _set("NonlinearGeneratorFactory", ["src.generators.nonlinear_generators", "src.generators.master_generator"])
    _set("CompleteNonlinearGeneratorFactory", ["src.generators.master_generator", "src.generators.nonlinear_generators"])

    # constructor
    _set("GeneratorConstructor", ["src.generators.generator_constructor"])
    _set("GeneratorSpecification", ["src.generators.generator_constructor"])
    _set("DerivativeTerm", ["src.generators.generator_constructor"])
    _set("DerivativeType", ["src.generators.generator_constructor"])
    _set("OperatorType", ["src.generators.generator_constructor"])

    # theorems
    _set("MasterTheoremSolver", ["src.generators.master_theorem"])
    _set("MasterTheoremParameters", ["src.generators.master_theorem"])
    _set("ExtendedMasterTheorem", ["src.generators.master_theorem"])

    # classifier
    _set("ODEClassifier", ["src.generators.ode_classifier"])
    _set("PhysicalApplication", ["src.generators.ode_classifier"])

    # functions
    _set("BasicFunctions", ["src.functions.basic_functions"])
    _set("SpecialFunctions", ["src.functions.special_functions"])

    # ML/DL
    _set("GeneratorPatternLearner", ["src.ml.pattern_learner"])
    _set("GeneratorVAE", ["src.ml.pattern_learner"])
    _set("GeneratorTransformer", ["src.ml.pattern_learner"])
    _set("create_model", ["src.ml.pattern_learner"])

    _set("MLTrainer", ["src.ml.trainer"])
    _set("ODEDataset", ["src.ml.trainer"])
    _set("ODEDataGenerator", ["src.ml.trainer"])

    _set("GeneratorPattern", ["src.ml.generator_learner"])
    _set("GeneratorPatternNetwork", ["src.ml.generator_learner"])
    _set("GeneratorLearningSystem", ["src.ml.generator_learner"])

    _set("ODENoveltyDetector", ["src.dl.novelty_detector"])
    _set("NoveltyAnalysis", ["src.dl.novelty_detector"])
    _set("ODETokenizer", ["src.dl.novelty_detector"])
    _set("ODETransformer", ["src.dl.novelty_detector"])

    # utils/ui
    _set("Settings", ["src.utils.config"])
    _set("AppConfig", ["src.utils.config"])
    _set("CacheManager", ["src.utils.cache"])
    _set("cached", ["src.utils.cache"])
    _set("ParameterValidator", ["src.utils.validators"])
    _set("UIComponents", ["src.ui.components"])

    return out

try:
    PROBED = _probe_src()
except Exception as e:
    HAVE_SRC = False
    PROBED = {}
    logger.warning(f"Failed to probe src/: {e}")

# Bind (may be None if not present)
MasterGenerator = PROBED.get("MasterGenerator")
EnhancedMasterGenerator = PROBED.get("EnhancedMasterGenerator")
CompleteMasterGenerator = PROBED.get("CompleteMasterGenerator")

LinearGeneratorFactory = PROBED.get("LinearGeneratorFactory")
CompleteLinearGeneratorFactory = PROBED.get("CompleteLinearGeneratorFactory")
NonlinearGeneratorFactory = PROBED.get("NonlinearGeneratorFactory")
CompleteNonlinearGeneratorFactory = PROBED.get("CompleteNonlinearGeneratorFactory")

GeneratorSpecification = PROBED.get("GeneratorSpecification")
DerivativeTerm = PROBED.get("DerivativeTerm")
DerivativeType = PROBED.get("DerivativeType")
OperatorType = PROBED.get("OperatorType")

MasterTheoremSolver = PROBED.get("MasterTheoremSolver")
ExtendedMasterTheorem = PROBED.get("ExtendedMasterTheorem")

ODEClassifier = PROBED.get("ODEClassifier")
PhysicalApplication = PROBED.get("PhysicalApplication")

BasicFunctions = PROBED.get("BasicFunctions")
SpecialFunctions = PROBED.get("SpecialFunctions")

MLTrainer = PROBED.get("MLTrainer")
ODENoveltyDetector = PROBED.get("ODENoveltyDetector")

# -----------------------------------------------------------------------------
# Streamlit UI setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Master Generators ODE System — Symbolic Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        padding:1.5rem;border-radius:14px;margin-bottom:1rem;color:#fff;text-align:center;}
    .metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        color:#fff;padding:0.9rem;border-radius:12px;text-align:center;}
    .info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
        border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:1rem 0;}
    .result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
        border:2px solid #4caf50;padding:1rem;border-radius:12px;margin:1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# LaTeX Exporter
# -----------------------------------------------------------------------------
class LaTeXExporter:
    @staticmethod
    def sympy_to_latex(expr: Any) -> str:
        if expr is None:
            return ""
        try:
            expr = sp.nsimplify(expr, [sp.E, sp.pi, sp.I], rational=True)
        except Exception:
            pass
        try:
            return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
        except Exception:
            return str(expr)

    @staticmethod
    def document_for_ode(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        gen = ode_data.get("generator", "")
        sol = ode_data.get("solution", "")
        rhs = ode_data.get("rhs", "")
        params = ode_data.get("parameters", {})
        cls = ode_data.get("classification", {})
        ics = ode_data.get("initial_conditions", {})

        parts = []
        if include_preamble:
            parts.append(
r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\usepackage{hyperref}
\geometry{margin=1in}
\title{Master Generators ODE System}
\author{Generated by Master Generators App}
\date{\today}
\begin{document}
\maketitle

\section{Generated Ordinary Differential Equation}
"""
            )
        parts.append(r"\subsection{Generator Equation}")
        parts.append(r"\begin{equation}")
        parts.append(f"{LaTeXExporter.sympy_to_latex(gen)} = {LaTeXExporter.sympy_to_latex(rhs)}")
        parts.append(r"\end{equation}")

        parts.append(r"\subsection{Exact Solution}")
        parts.append(r"\begin{equation}")
        parts.append(f"y(x) = {LaTeXExporter.sympy_to_latex(sol)}")
        parts.append(r"\end{equation}")

        parts.append(r"\subsection{Parameters}")
        parts.append(r"\begin{align}")
        parts.append(f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha', ''))} \\\\")
        parts.append(f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta', ''))} \\\\")
        parts.append(f"n       &= {LaTeXExporter.sympy_to_latex(params.get('n', ''))} \\\\")
        parts.append(f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M', ''))}")
        parts.append(r"\end{align}")

        if ics:
            parts.append(r"\subsection{Initial Conditions}")
            parts.append(r"\begin{align}")
            items = list(ics.items())
            for i, (k, v) in enumerate(items):
                sep = r" \\" if i < len(items) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}{sep}")
            parts.append(r"\end{align}")

        if cls:
            parts.append(r"\subsection{Mathematical Classification}")
            parts.append(r"\begin{itemize}")
            parts.append(f"\\item \\textbf{{Type:}} {cls.get('type','Unknown')}")
            parts.append(f"\\item \\textbf{{Order:}} {cls.get('order','Unknown')}")
            if "field" in cls:
                parts.append(f"\\item \\textbf{{Field:}} {cls.get('field')}")
            if "applications" in cls and cls["applications"]:
                parts.append(f"\\item \\textbf{{Applications:}} {', '.join(cls['applications'])}")
            parts.append(r"\end{itemize}")

        parts.append(r"\subsection{Solution Verification}")
        parts.append("Substitute $y(x)$ into the generator operator to verify $L[y] = \\text{RHS}$.")

        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def zip_package(ode_data: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            tex = LaTeXExporter.document_for_ode(ode_data, include_preamble=True)
            z.writestr("ode_document.tex", tex)
            z.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            z.writestr(
                "README.txt",
                "Master Generators ODE Export\n"
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Compile: pdflatex ode_document.tex\n",
            )
        buf.seek(0)
        return buf.getvalue()

# -----------------------------------------------------------------------------
# Symbolic Helpers: Theorem 4.1 & 4.2 (Stirling) + Operator Application
# -----------------------------------------------------------------------------
EPS = sp.Symbol("epsilon", positive=True)

def _to_exact(value: str | float | int) -> sp.Expr:
    try:
        return sp.nsimplify(value, rational=True)
    except Exception:
        return sp.sympify(value)

def get_function_expr(source: str, name: str) -> sp.Expr:
    z = sp.Symbol("z")
    lib = None
    if source == "Basic" and BasicFunctions:
        lib = BasicFunctions()
    elif source == "Special" and SpecialFunctions:
        lib = SpecialFunctions()
    else:
        raise ValueError("Function library not available.")
    obj = None
    if hasattr(lib, "get_function"):
        obj = lib.get_function(name)
    elif hasattr(lib, "functions") and name in getattr(lib, "functions"):
        obj = lib.functions[name]
    else:
        raise ValueError(f"Function '{name}' not found in {source} library.")
    try:
        if callable(obj):
            return sp.sympify(obj(z))
        return sp.sympify(obj)
    except Exception:
        # treat as symbol
        return sp.Symbol(name)

def theorem_4_1_solution_expr(f_expr: sp.Expr, alpha, beta, n: int, M, x: sp.Symbol) -> sp.Expr:
    """
    y(x) = π/(2n) ∑_{s=1}^n [ 2 f(α+β) - ( ψ_s + φ_s ) ] + π M
    with ψ_s = f(α + β e^{i x cos ω_s - x sin ω_s}), φ_s = f(α + β e^{-i x cos ω_s - x sin ω_s}),
         ω_s = (2s-1)π/(2n)
    """
    z = sp.Symbol("z")
    terms = []
    for s in range(1, n+1):
        ω = sp.Rational(2*s-1, 2*n)*sp.pi
        ψ = f_expr.subs(z, alpha + beta*sp.exp(sp.I*x*sp.cos(ω) - x*sp.sin(ω)))
        φ = f_expr.subs(z, alpha + beta*sp.exp(-sp.I*x*sp.cos(ω) - x*sp.sin(ω)))
        terms.append(2*f_expr.subs(z, alpha + beta) - (ψ + φ))
    y = sp.pi/(2*n) * sum(terms) + sp.pi*M
    try:
        y = sp.nsimplify(y, [sp.E, sp.pi, sp.I], rational=True)
    except Exception:
        pass
    return sp.simplify(y)

def theorem_4_2_y_m_expr(f_expr: sp.Expr, alpha_value, beta, n: int, m: int, x: sp.Symbol) -> sp.Expr:
    """
    Theorem 4.2 in compact (complex) form using Stirling numbers of the second kind.
    y^{(m)}(x) = -(π/(2n)) ∑_{s=1}^n { λ_s^m ∑_{j=1}^m S(m,j) (β ζ_s)^j ∂_α^j ψ + conj(λ_s)^m ∑_{j=1}^m S(m,j) (β \bar ζ_s)^j ∂_α^j φ }
    where ψ(α,ω,x)=f(α+β ζ_s(x)), φ uses \bar ζ_s; ζ_s(x)=exp(-x sin ω) exp(i x cos ω), λ_s=e^{i(π/2+ω)}.
    We treat α as a symbol during differentiation and substitute alpha_value at the end.
    """
    z = sp.Symbol("z")
    αsym = sp.Symbol("alpha_sym", real=True)
    total = 0
    for s in range(1, n+1):
        ω = sp.Rational(2*s-1, 2*n)*sp.pi
        λ = sp.exp(sp.I*(sp.pi/2 + ω))
        ζ = sp.exp(-x*sp.sin(ω)) * sp.exp(sp.I*x*sp.cos(ω))
        ζb = sp.exp(-x*sp.sin(ω)) * sp.exp(-sp.I*x*sp.cos(ω))

        ψ = f_expr.subs(z, αsym + beta*ζ)
        φ = f_expr.subs(z, αsym + beta*ζb)

        sum1 = 0
        sum2 = 0
        for j in range(1, m+1):
            S = sp.functions.combinatorial.numbers.stirling(m, j, kind=2)
            sum1 += S * (beta*ζ)**j * sp.diff(ψ, αsym, j)
            sum2 += S * (beta*ζb)**j * sp.diff(φ, αsym, j)
        total += λ**m * sum1 + sp.conjugate(λ)**m * sum2

    y_m = -sp.pi/(2*n) * total
    y_m = y_m.subs(αsym, alpha_value)
    try:
        y_m = sp.nsimplify(y_m, [sp.E, sp.pi, sp.I], rational=True)
    except Exception:
        pass
    return sp.simplify(y_m)

def apply_lhs_to_solution(lhs_expr: sp.Expr, solution_y: sp.Expr, x: sp.Symbol, y_name: str = "y") -> sp.Expr:
    """
    Substitute y(arg) and d^k/dx^k y(arg) in lhs_expr by the corresponding expressions computed from solution_y.
    This supports scaled/shifted arguments: y(x/a + b), and the chain rule is handled by differentiating after substitution.
    """
    subs_map: Dict[sp.Expr, sp.Expr] = {}

    # y(arg)
    for f in lhs_expr.atoms(AppliedUndef):
        if f.func.__name__ == y_name and len(f.args) == 1:
            arg = f.args[0]
            subs_map[f] = sp.simplify(solution_y.subs(x, arg))

    # derivatives of y(arg) with respect to x
    for d in lhs_expr.atoms(sp.Derivative):
        base = d.expr
        if isinstance(base, AppliedUndef) and base.func.__name__ == y_name and len(base.args) == 1:
            arg = base.args[0]
            try:
                order = sum(c for v, c in d.variable_count if v == x)
            except Exception:
                order = sum(1 for v in d.variables if v == x)
            subs_map[d] = sp.diff(solution_y.subs(x, arg), (x, order))

    try:
        rhs = sp.simplify(lhs_expr.xreplace(subs_map))
    except Exception:
        rhs = sp.simplify(lhs_expr.subs(subs_map))
    return rhs

# -----------------------------------------------------------------------------
# Free‑Form Generator Builder (SymPy-based)
# -----------------------------------------------------------------------------
def wrap_expr(u: sp.Expr, wrapper: str, param: sp.Expr | None) -> sp.Expr:
    w = (wrapper or "identity").strip().lower()
    if w in ("identity", "id"):
        return u
    if w in ("exp",):
        return sp.exp(u)
    if w in ("log", "ln"):
        return sp.log(EPS + sp.Abs(u))
    if w in ("sin",):
        return sp.sin(u)
    if w in ("cos",):
        return sp.cos(u)
    if w in ("tan",):
        return sp.tan(u)
    if w in ("sinh",):
        return sp.sinh(u)
    if w in ("cosh",):
        return sp.cosh(u)
    if w in ("tanh",):
        return sp.tanh(u)
    if w in ("power", "pow"):
        p = sp.sympify(param if param is not None else 1)
        return sp.Pow(u, p)
    # fallback: try a SymPy named function (e.g., erf)
    try:
        f = getattr(sp, wrapper)
        return f(u)
    except Exception:
        return u

def build_free_lhs_expr(free_terms: List[Dict[str, Any]], x: sp.Symbol, yfunc: sp.Function) -> sp.Expr:
    """
    free_terms item:
      {
        'coef': "1",             # symbolic ok
        'wrap': "sinh"|"exp"|...|"power"
        'wrap_power': "3"        # only used if wrap == 'power'
        'inner_order': 2,        # derivative order of y
        'outer_order': 0,        # derivative order applied to whole wrapped term
        'op': "standard"|"scaled",
        'a': "2",                # x -> x/a + shift (if scaled)
        'shift': "0"
      }
    """
    total = 0
    for t in free_terms:
        try:
            c = _to_exact(t.get("coef", "1"))
            wrap = t.get("wrap", "identity")
            wpow = _to_exact(t.get("wrap_power", "1"))
            inner = int(t.get("inner_order", 0))
            outer = int(t.get("outer_order", 0))
            op = t.get("op", "standard")
            a = _to_exact(t.get("a", "1"))
            b = _to_exact(t.get("shift", "0"))

            arg = x if op == "standard" else (x/a + b)
            base = yfunc(arg) if inner == 0 else sp.Derivative(yfunc(arg), (x, inner))
            wrapped = wrap_expr(base, wrap, wpow)
            term = c * wrapped
            if outer > 0:
                term = sp.diff(term, (x, outer))
            total += term
        except Exception as e:
            logger.debug(f"Free term skipped: {e}")
    return sp.simplify(total)

# -----------------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------------
def init_state():
    if "generated_odes" not in st.session_state:
        st.session_state.generated_odes = []
    if "structured_terms" not in st.session_state:
        st.session_state.structured_terms = []  # src.DerivativeTerm-friendly
    if "free_terms" not in st.session_state:
        st.session_state.free_terms = []        # free-form dicts
    if "current_generator_spec" not in st.session_state:
        st.session_state.current_generator_spec = None  # src spec (if constructed)
    if "current_generator_lhs_expr" not in st.session_state:
        st.session_state.current_generator_lhs_expr = None  # SymPy LHS from free-form
    if "ml_trainer" not in st.session_state:
        st.session_state.ml_trainer = None
    if "ml_trained" not in st.session_state:
        st.session_state.ml_trained = False
    if "basic_functions" not in st.session_state and BasicFunctions:
        st.session_state.basic_functions = BasicFunctions()
    if "special_functions" not in st.session_state and SpecialFunctions:
        st.session_state.special_functions = SpecialFunctions()
    if "ode_classifier" not in st.session_state and ODEClassifier:
        st.session_state.ode_classifier = ODEClassifier()
    if "novelty_detector" not in st.session_state and ODENoveltyDetector:
        st.session_state.novelty_detector = ODENoveltyDetector()

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def enum_values(E) -> List[str]:
    try:
        if hasattr(E, "__members__"):
            return [m.value if hasattr(m, "value") else str(m) for m in E.__members__.values()]
        return [e.value if hasattr(e, "value") else str(e) for e in list(E)]
    except Exception:
        return []

def guess_order_from_lhs(lhs: Any, x: sp.Symbol) -> int:
    try:
        max_o = 0
        for node in sp.preorder_traversal(lhs):
            if isinstance(node, sp.Derivative):
                try:
                    o = sum(c for v, c in node.variable_count if v == x)
                except Exception:
                    o = sum(1 for v in node.variables if v == x)
                max_o = max(max_o, int(o))
        return max_o
    except Exception:
        return 0

def is_linear_lhs(lhs: Any, x: sp.Symbol) -> bool:
    """Heuristic linearity check."""
    try:
        y = sp.Function("y")
        # check non-1 powers of y(.) or derivatives
        for p in sp.preorder_traversal(lhs):
            if isinstance(p, sp.Pow):
                if p.base.has(y(x)) or any(isinstance(a, sp.Derivative) and a.expr == y(x) for a in p.base.atoms(sp.Derivative)):
                    if p.exp != 1:
                        return False
        return True
    except Exception:
        return False

def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_dashboard():
    st.markdown('<div class="main-header"><h2>🔬 Master Generators for ODEs</h2>'
                '<p>Symbolic Theorems 4.1 & 4.2 + Free‑Form Generator + ML/DL</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h4>Generated ODEs</h4><h2>{len(st.session_state.generated_odes)}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4>Free‑Form Terms</h4><h2>{len(st.session_state.free_terms)}</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4>ML Trained</h4><h2>{"Yes" if st.session_state.ml_trained else "No"}</h2></div>', unsafe_allow_html=True)
    with c4:
        has_lhs = "Yes" if (st.session_state.current_generator_lhs_expr is not None or st.session_state.current_generator_spec is not None) else "No"
        st.markdown(f'<div class="metric-card"><h4>Have LHS</h4><h2>{has_lhs}</h2></div>', unsafe_allow_html=True)

    if st.session_state.generated_odes:
        st.subheader("Recent ODEs")
        cols = ["type","order","function_used","timestamp"]
        df = pd.DataFrame(st.session_state.generated_odes)
        show = [c for c in cols if c in df.columns]
        if show:
            st.dataframe(df[show].tail(8), use_container_width=True)

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    y = sp.Function("y")
    x = sp.Symbol("x", real=True)

    tabs = st.tabs(["Structured (src)", "Free‑Form (advanced)"])

    # ---------------- Structured (src) ----------------
    with tabs[0]:
        if not (GeneratorSpecification and DerivativeTerm):
            st.info("src-based constructor not available; use Free‑Form tab.")
        else:
            st.markdown("<div class='info-box'>Build a spec from your src/ enums and terms.</div>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                deriv_order = st.number_input("Derivative Order", min_value=0, max_value=20, value=0, step=1)
            with col2:
                dt_opts = enum_values(DerivativeType) if DerivativeType else ["identity","sin","cos","exp","log","power"]
                func_type_val = st.selectbox("Function Type", dt_opts)
            with col3:
                coefficient = st.text_input("Coefficient (symbolic ok)", value="1")
            with col4:
                power = st.number_input("Power", min_value=1, max_value=10, value=1, step=1)

            col1b, col2b, col3b = st.columns(3)
            with col1b:
                op_opts = enum_values(OperatorType) if OperatorType else ["standard","delay","advance"]
                operator_val = st.selectbox("Operator", op_opts)
            with col2b:
                scaling = st.text_input("Scaling a (for delay/advance)", value="2")
            with col3b:
                shift = st.text_input("Shift b (for delay/advance)", value="0")

            if st.button("Add Structured Term", use_container_width=True):
                try:
                    ftype = DerivativeType(func_type_val) if DerivativeType else func_type_val
                    otype = OperatorType(operator_val) if OperatorType else operator_val
                    kwargs = dict(
                        derivative_order=int(deriv_order),
                        coefficient=float(coefficient) if coefficient.replace(".","",1).isdigit() else _to_exact(coefficient),
                        power=int(power),
                        function_type=ftype,
                        operator_type=otype,
                    )
                    if "delay" in str(operator_val).lower() or "advance" in str(operator_val).lower():
                        kwargs["scaling"] = float(scaling) if scaling.replace(".","",1).isdigit() else _to_exact(scaling)
                        kwargs["shift"] = float(shift) if shift.replace(".","",1).isdigit() else _to_exact(shift)
                    term = DerivativeTerm(**kwargs)
                    st.session_state.structured_terms.append(term)
                    desc = getattr(term, "get_description", lambda: str(term))()
                    st.success(f"Added structured term: {desc}")
                except Exception as e:
                    st.error(f"Failed to add structured term: {e}")

            if st.session_state.structured_terms:
                st.subheader("Current Structured Terms")
                for i, term in enumerate(st.session_state.structured_terms):
                    c1, c2 = st.columns([8,1])
                    with c1:
                        desc = getattr(term, "get_description", lambda: str(term))()
                        st.info(desc)
                    with c2:
                        if st.button("❌", key=f"del_struct_{i}"):
                            st.session_state.structured_terms.pop(i)
                            st.experimental_rerun()

                if st.button("🔨 Build Generator Specification", type="primary", use_container_width=True):
                    try:
                        spec = GeneratorSpecification(terms=st.session_state.structured_terms,
                                                     name=f"Structured Generator #{len(st.session_state.generated_odes)+1}")
                        st.session_state.current_generator_spec = spec
                        # Try to get LHS from spec
                        lhs = getattr(spec, "lhs", None)
                        if lhs is None and hasattr(spec, "get_lhs"):
                            lhs = spec.get_lhs()
                        if lhs is None and hasattr(spec, "build_lhs"):
                            lhs = spec.build_lhs()
                        if lhs is not None:
                            st.latex(sp.latex(lhs) + " = \\text{RHS}")
                        else:
                            st.info("Specification built, but no .lhs present to display.")
                    except Exception as e:
                        st.error(f"Failed to build specification: {e}")

                if st.button("🗑️ Clear Structured Terms", use_container_width=True):
                    st.session_state.structured_terms = []
                    st.session_state.current_generator_spec = None
                    st.experimental_rerun()

    # ---------------- Free‑Form (advanced) ----------------
    with tabs[1]:
        st.markdown("<div class='info-box'>Compose arbitrary terms like sinh(y'), exp(y'''''''), ln(y''), power(...) and apply an outer derivative too.</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            coef = st.text_input("Coefficient c (symbolic ok)", value="1")
        with col2:
            wrap = st.selectbox("Wrapper", ["identity","sin","cos","tan","sinh","cosh","tanh","exp","log","power"])
        with col3:
            wrap_pow = st.text_input("Wrapper power p (if power)", value="1")
        with col4:
            inner_order = st.number_input("Inner derivative order of y", min_value=0, max_value=20, value=0, step=1)

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            outer_order = st.number_input("Outer derivative of whole term", min_value=0, max_value=20, value=0, step=1)
        with col6:
            op = st.selectbox("Argument mode", ["standard","scaled"])
        with col7:
            a = st.text_input("a (for x/a + b)", value="1")
        with col8:
            b = st.text_input("b (shift in x/a + b)", value="0")

        if st.button("➕ Add Free‑Form Term", use_container_width=True):
            st.session_state.free_terms.append(dict(
                coef=coef, wrap=wrap, wrap_power=wrap_pow,
                inner_order=int(inner_order), outer_order=int(outer_order),
                op=op, a=a, shift=b
            ))
            st.success("Added free‑form term.")

        if st.session_state.free_terms:
            st.subheader("Current Free‑Form Terms")
            for i, t in enumerate(st.session_state.free_terms):
                pretty = f"c={t['coef']}, {t['wrap']}[ D^{t['inner_order']} y(arg) ] (outer D^{t['outer_order']}) ; arg = {'x' if t['op']=='standard' else 'x/'+str(t['a'])+' + '+str(t['shift'])}"
                c1, c2 = st.columns([8,1])
                with c1:
                    st.info(pretty)
                with c2:
                    if st.button("❌", key=f"del_free_{i}"):
                        st.session_state.free_terms.pop(i)
                        st.experimental_rerun()

            if st.button("🔨 Build Free‑Form LHS", type="primary", use_container_width=True):
                try:
                    lhs_expr = build_free_lhs_expr(st.session_state.free_terms, x, y)
                    st.session_state.current_generator_lhs_expr = lhs_expr
                    st.success("Free‑form LHS built.")
                    st.latex(sp.latex(lhs_expr) + " = \\text{RHS}")
                except Exception as e:
                    st.error(f"Failed to build free LHS: {e}")

            if st.button("🗑️ Clear Free‑Form Terms", use_container_width=True):
                st.session_state.free_terms = []
                st.session_state.current_generator_lhs_expr = None
                st.experimental_rerun()

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (4.1 & 4.2)")
    if not (BasicFunctions or SpecialFunctions):
        st.warning("Function libraries not available from src/.")
        return

    colL, colR = st.columns([1,1])
    with colL:
        source_lib = st.selectbox("Function Library", ["Basic","Special"])
        func_names = []
        try:
            lib = st.session_state.basic_functions if source_lib == "Basic" else st.session_state.special_functions
            if hasattr(lib, "get_function_names"):
                func_names = lib.get_function_names()
            elif hasattr(lib, "functions"):
                func_names = list(lib.functions.keys())
        except Exception:
            pass
        func_name = st.selectbox("Choose f(z)", func_names)

    with colR:
        alpha = st.text_input("α", value="1")
        beta  = st.text_input("β", value="1")
        n     = st.number_input("n (integer ≥ 1)", min_value=1, max_value=20, value=1, step=1)
        M     = st.text_input("M", value="0")

    # ---------------- exact/symbolic toggle (requested) ----------------
    use_exact = st.checkbox("Exact (symbolic) parameters", value=True)
    def to_exact(v):
        try:
            return sp.nsimplify(v, rational=True)
        except Exception:
            return sp.sympify(v)

    # ---------------- Theorem 4.1 (Solution y(x)) ----------------
    if st.button("🚀 Generate ODE (Theorem 4.1)", type="primary", use_container_width=True):
        with st.spinner("Computing y(x) (4.1) and constructing RHS = L[y]..."):
            try:
                x = sp.Symbol("x", real=True)

                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)
                𝑀 = to_exact(M)     if use_exact else sp.Float(M)

                f_expr = get_function_expr(source_lib, func_name)
                solution = theorem_4_1_solution_expr(f_expr, α, β, int(n), 𝑀, x)

                # determine LHS (Free‑Form preferred; then src spec; else Symbol)
                lhs = st.session_state.get("current_generator_lhs_expr")
                if lhs is None:
                    spec = st.session_state.get("current_generator_spec")
                    if spec is not None:
                        lhs = getattr(spec, "lhs", None)
                        if lhs is None and hasattr(spec, "get_lhs"):
                            lhs = spec.get_lhs()
                        if lhs is None and hasattr(spec, "build_lhs"):
                            lhs = spec.build_lhs()
                if lhs is None:
                    lhs = sp.Symbol("LHS")

                # Build RHS = L[y]
                rhs = apply_lhs_to_solution(lhs, solution, x, y_name="y")

                # Classification (best-effort)
                classification = {}
                try:
                    if ODEClassifier:
                        classifier = st.session_state.ode_classifier
                        meta = {"ode": lhs, "solution": solution, "rhs": rhs}
                        c_out = classifier.classify_ode(meta)
                        classification = c_out.get("classification", {})
                        classification["order"] = classification.get("order", guess_order_from_lhs(lhs, x))
                        classification["type"]  = classification.get("type", "Linear" if is_linear_lhs(lhs, x) else "Nonlinear")
                        classification["field"] = classification.get("field", "Mathematical Physics")
                        classification["applications"] = classification.get("applications", ["Research Equation"])
                except Exception:
                    classification = {
                        "order": guess_order_from_lhs(lhs, x),
                        "type": "Linear" if is_linear_lhs(lhs, x) else "Nonlinear",
                        "field": "Mathematical Physics",
                        "applications": ["Research Equation"]
                    }

                result = {
                    "generator": lhs,
                    "solution": solution,
                    "rhs": rhs,
                    "parameters": {"alpha": α, "beta": β, "n": int(n), "M": 𝑀},
                    "function_used": func_name,
                    "type": classification.get("type", "Unknown"),
                    "order": classification.get("order", 0),
                    "classification": classification,
                    "initial_conditions": {"y(0)": sp.simplify(solution.subs(x, 0))},
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "generator_number": len(st.session_state.generated_odes) + 1,
                }
                st.session_state.generated_odes.append(result)

                st.success("✅ ODE generated successfully.")
                tabs = st.tabs(["📐 Equation", "💡 Solution", "🏷️ Classification", "📤 Export"])
                with tabs[0]:
                    st.latex(sp.latex(lhs) + " = " + sp.latex(rhs))
                with tabs[1]:
                    st.latex("y(x) = " + sp.latex(solution))
                    st.markdown("**Initial Condition:**")
                    st.latex("y(0) = " + sp.latex(sp.simplify(solution.subs(x, 0))))
                with tabs[2]:
                    st.json(classification, expanded=False)
                with tabs[3]:
                    tex = LaTeXExporter.document_for_ode(result, include_preamble=True)
                    st.download_button("📄 Download LaTeX", tex, file_name="ode_solution.tex", mime="text/x-latex")
                    pkg = LaTeXExporter.zip_package(result)
                    st.download_button("📦 Download Package (ZIP)", pkg, file_name=f"ode_package_{int(time.time())}.zip", mime="application/zip")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.exception(e)

    # ---------------- Theorem 4.2: y^{(m)}(x) ----------------
    with st.expander("Theorem 4.2: Compute general m‑th derivative y^{(m)}(x)", expanded=False):
        m = st.number_input("m (order for y^{(m)})", min_value=1, max_value=50, value=3, step=1)
        if st.button("🧮 Compute y^{(m)}(x) via 4.2", use_container_width=True):
            try:
                x = sp.Symbol("x", real=True)
                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)

                f_expr = get_function_expr(source_lib, func_name)
                y_m = theorem_4_2_y_m_expr(f_expr, α, β, int(n), int(m), x)

                st.success("y^{(m)}(x) computed by Theorem 4.2 (Stirling form).")
                st.latex(f"y^{{({int(m)})}}(x) = " + sp.latex(y_m))
            except Exception as e:
                st.error(f"Failed to compute y^{m}(x): {e}")

def page_ml():
    st.header("🤖 ML Pattern Learning")
    if not MLTrainer:
        st.info("MLTrainer not available in src/.")
        return

    model_type = st.selectbox(
        "Model",
        ["pattern_learner","vae","transformer"],
        format_func=lambda s: {"pattern_learner": "Pattern Learner", "vae": "VAE", "transformer": "Transformer"}[s],
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.slider("Epochs", 10, 500, 100, 5)
        batch_size = st.slider("Batch Size", 8, 128, 32, 8)
    with c2:
        lr = st.select_slider("Learning Rate", options=[1e-4,5e-4,1e-3,5e-3,1e-2], value=1e-3)
        samples = st.slider("Training Samples", 100, 5000, 1000, 100)
    with c3:
        val_split = st.slider("Validation Split", 0.05, 0.3, 0.2, 0.05)
        use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate at least 5 ODEs before training.")
        return

    if st.button("🚀 Train", type="primary"):
        device = "cuda" if (use_gpu and torch_cuda_available()) else "cpu"
        try:
            trainer = MLTrainer(model_type=model_type, learning_rate=lr, device=device)
            st.session_state.ml_trainer = trainer
            prog = st.progress(0)
            info = st.empty()
            def cb(epoch, total_epochs):
                prog.progress(int(100*epoch/total_epochs))
                info.info(f"Epoch {epoch}/{total_epochs}")
            trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                samples=samples,
                validation_split=val_split,
                progress_callback=cb
            )
            st.session_state.ml_trained = True
            st.success("Training completed.")
            hist = getattr(trainer, "history", {})
            if hist.get("train_loss"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist["train_loss"], mode="lines", name="train loss"))
                if hist.get("val_loss"):
                    fig.add_trace(go.Scatter(y=hist["val_loss"], mode="lines", name="val loss"))
                fig.update_layout(height=300, title="Training History")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)

    if st.session_state.ml_trainer and st.session_state.ml_trained:
        st.subheader("Generate Novel ODEs")
        k = st.slider("How many", 1, 10, 1)
        if st.button("🎲 Generate", type="primary"):
            for i in range(k):
                try:
                    res = st.session_state.ml_trainer.generate_new_ode()
                    if res:
                        st.session_state.generated_odes.append(res)
                        with st.expander(f"Generated ODE #{len(st.session_state.generated_odes)}"):
                            if "ode" in res:
                                st.latex(sp.latex(res["ode"]) if not isinstance(res["ode"], str) else res["ode"])
                            st.write({k: v for k, v in res.items() if k not in ["ode","solution"]})
                except Exception as e:
                    st.warning(f"Generation {i+1} failed: {e}")

def page_batch():
    st.header("📊 Batch ODE Generation (via 4.1)")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_odes = st.slider("Number of ODEs", 5, 300, 20)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with col2:
        func_categories = st.multiselect("Function Categories", ["Basic","Special"], default=["Basic"])
        include_solutions = st.checkbox("Include Solutions Preview", True)
    with col3:
        vary = st.checkbox("Vary Parameters", True)
        if vary:
            alpha_min, alpha_max = st.number_input("α min", value=-2.0), st.number_input("α max", value=2.0)
            beta_min, beta_max   = st.number_input("β min", value=0.5), st.number_input("β max", value=2.0)
        else:
            alpha_min = alpha_max = 1.0
            beta_min = beta_max = 1.0

    if st.button("🚀 Generate Batch", type="primary"):
        results = []
        pool = []
        if "Basic" in func_categories and hasattr(st.session_state, "basic_functions"):
            try: pool += st.session_state.basic_functions.get_function_names()
            except Exception: pass
        if "Special" in func_categories and hasattr(st.session_state, "special_functions"):
            try: pool += st.session_state.special_functions.get_function_names()
            except Exception: pass
        pool = pool[:50]

        x = sp.Symbol("x", real=True)
        # choose LHS to apply (Free-form preferred → spec → None)
        lhs = st.session_state.get("current_generator_lhs_expr")
        if lhs is None:
            spec = st.session_state.get("current_generator_spec")
            if spec is not None:
                lhs = getattr(spec, "lhs", None)
                if lhs is None and hasattr(spec, "get_lhs"): lhs = spec.get_lhs()
                if lhs is None and hasattr(spec, "build_lhs"): lhs = spec.build_lhs()
        if lhs is None:
            lhs = sp.Symbol("LHS")

        for i in range(num_odes):
            try:
                α = float(np.random.uniform(alpha_min, alpha_max))
                β = float(np.random.uniform(beta_min, beta_max))
                n = int(np.random.randint(1, 3))
                M = float(np.random.uniform(-1, 1))
                if not pool:
                    break
                f_name = str(np.random.choice(pool))
                source = "Basic" if (np.random.rand() < 0.7) else "Special"

                f_expr = get_function_expr(source, f_name)
                y = theorem_4_1_solution_expr(f_expr, sp.nsimplify(α), sp.nsimplify(β), n, sp.nsimplify(M), x)
                rhs = apply_lhs_to_solution(lhs, y, x, y_name="y")

                rec = {
                    "ID": i+1,
                    "Type": "linear" if is_linear_lhs(lhs, x) else "nonlinear",
                    "Generator": "free-form" if st.session_state.current_generator_lhs_expr is not None else "structured",
                    "Function": f_name,
                    "Order": guess_order_from_lhs(lhs, x),
                    "α": α, "β": β, "n": n, "M": M,
                }
                if include_solutions:
                    rec["Solution"] = sp.sstr(y)[:120] + "..."
                results.append(rec)
            except Exception as e:
                logger.debug(f"Batch item failed: {e}")

        st.success(f"Generated {len(results)} ODE records.")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📊 Download CSV", df.to_csv(index=False), file_name="batch_odes.csv", mime="text/csv")
        with c2:
            st.download_button("📄 Download JSON", json.dumps(results, indent=2), file_name="batch_odes.json", mime="application/json")
        with c3:
            lines = [r"\begin{tabular}{|c|c|c|c|c|}\hline",
                     r"ID & Type & Generator & Function & Order \\ \hline"]
            for r in results[:30]:
                lines.append(f"{r['ID']} & {r['Type']} & {r['Generator']} & {r['Function']} & {r['Order']} \\\\")
            lines.append(r"\hline\end{tabular}")
            st.download_button("📝 Download LaTeX Table", "\n".join(lines), file_name="batch_odes.tex", mime="text/x-latex")

def page_novelty():
    st.header("🔍 Novelty Detection")
    if not ODENoveltyDetector:
        st.info("Novelty detector not available in src/.")
        return
    det = st.session_state.novelty_detector
    mode = st.radio("Input", ["Use Current Generator", "Enter ODE LaTeX/Text", "Select from Generated"], index=0)
    ode_obj = None
    if mode == "Use Current Generator":
        lhs = st.session_state.get("current_generator_lhs_expr")
        if lhs is None:
            spec = st.session_state.get("current_generator_spec")
            if spec is not None:
                lhs = getattr(spec, "lhs", None)
                if lhs is None and hasattr(spec, "get_lhs"): lhs = spec.get_lhs()
                if lhs is None and hasattr(spec, "build_lhs"): lhs = spec.build_lhs()
        if lhs is not None:
            ode_obj = {"ode": lhs, "type": "custom", "order": guess_order_from_lhs(lhs, sp.Symbol("x"))}
        else:
            st.warning("No generator available.")
    elif mode == "Enter ODE LaTeX/Text":
        ode_str = st.text_area("Enter ODE", "")
        if ode_str.strip():
            ode_obj = {"ode": ode_str, "type": "manual", "order": st.number_input("Order", 1, 20, 2)}
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox(
                "Select ODE",
                range(len(st.session_state.generated_odes)),
                format_func=lambda i: f"ODE {i+1} (order {st.session_state.generated_odes[i].get('order','?')})",
            )
            ode_obj = st.session_state.generated_odes[idx]

    if ode_obj and st.button("Analyze", type="primary"):
        try:
            analysis = det.analyze(ode_obj, check_solvability=True, detailed=True)
            st.metric("Novelty Score", f"{analysis.novelty_score:.1f}/100")
            st.metric("Confidence", f"{analysis.confidence:.1%}")
            if analysis.special_characteristics:
                st.write("**Special characteristics:**")
                st.write(analysis.special_characteristics[:10])
            if analysis.recommended_methods:
                st.write("**Recommended methods:**")
                st.write(analysis.recommended_methods[:10])
            if analysis.detailed_report:
                st.download_button("📥 Download Report", analysis.detailed_report, file_name="novelty_report.txt")
        except Exception as e:
            st.error(f"Novelty analysis failed: {e}")

def page_analysis():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs yet. Generate some first.")
        return
    df = pd.DataFrame([
        {"Type": rec.get("type",""),
         "Function": rec.get("function_used",""),
         "Order": rec.get("order",""),
         "Timestamp": rec.get("timestamp","")}
        for rec in st.session_state.generated_odes
    ])
    st.dataframe(df, use_container_width=True)
    if not df.empty:
        fig = px.histogram(df, x="Order", nbins=10, title="Order Distribution")
        st.plotly_chart(fig, use_container_width=True)

def page_visualize():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.info("No ODEs to visualize yet.")
        return
    idx = st.selectbox(
        "Select ODE",
        range(len(st.session_state.generated_odes)),
        format_func=lambda i: f"#{i+1} | {st.session_state.generated_odes[i].get('function_used','?')} | order {st.session_state.generated_odes[i].get('order','?')}"
    )
    ode = st.session_state.generated_odes[idx]
    x = sp.Symbol("x", real=True)
    try:
        y = ode["solution"]
        if st.button("Generate Plot", type="primary"):
            xs = np.linspace(-5, 5, 600)
            yfn = sp.lambdify([x], y, "numpy")
            ys = np.array([yfn(val) for val in xs], dtype=np.complex128)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines", name="Re y(x)"))
            if np.any(np.imag(ys) != 0):
                fig.add_trace(go.Scatter(x=xs, y=np.imag(ys), mode="lines", name="Im y(x)"))
            fig.update_layout(title="Solution Plot", xaxis_title="x", yaxis_title="y(x)")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to visualize: {e}")

def page_export():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.info("No ODEs to export.")
        return
    idx = st.selectbox(
        "Select ODE",
        range(len(st.session_state.generated_odes)),
        format_func=lambda i: f"ODE {i+1} ({st.session_state.generated_odes[i].get('function_used','?')})",
    )
    rec = st.session_state.generated_odes[idx]
    preview = LaTeXExporter.document_for_ode(rec, include_preamble=False)
    st.code(preview, language="latex")
    c1, c2 = st.columns(2)
    with c1:
        tex = LaTeXExporter.document_for_ode(rec, include_preamble=True)
        st.download_button("📄 Download LaTeX", tex, file_name=f"ode_{idx+1}.tex", mime="text/x-latex")
    with c2:
        pkg = LaTeXExporter.zip_package(rec)
        st.download_button("📦 Download Package (ZIP)", pkg, file_name=f"ode_package_{idx+1}.zip", mime="application/zip")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    init_state()
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "🔧 Generator Constructor",
            "🎯 Apply Master Theorem",
            "🤖 ML Pattern Learning",
            "📊 Batch Generation",
            "🔍 Novelty Detection",
            "📈 Analysis & Classification",
            "📐 Visualization",
            "📤 Export & LaTeX",
        ],
        index=0
    )

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🔧 Generator Constructor":
        page_generator_constructor()
    elif page == "🎯 Apply Master Theorem":
        page_apply_master_theorem()
    elif page == "🤖 ML Pattern Learning":
        page_ml()
    elif page == "📊 Batch Generation":
        page_batch()
    elif page == "🔍 Novelty Detection":
        page_novelty()
    elif page == "📈 Analysis & Classification":
        page_analysis()
    elif page == "📐 Visualization":
        page_visualize()
    elif page == "📤 Export & LaTeX":
        page_export()

if __name__ == "__main__":
    main()
