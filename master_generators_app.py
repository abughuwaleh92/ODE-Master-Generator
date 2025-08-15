# -*- coding: utf-8 -*-
"""
Master Generators for ODEs — Complete Edition (with Algebraic Free‑form, Faster n>1, ML/DL integration)

- Theorem 4.1 (exact/symbolic or numeric params)
- Theorem 4.2 via Stirling numbers (Faà di Bruno)
- Free‑form LHS:
  * Term-based (wrappers, delay/advance, inner/outer derivatives)
  * NEW: Algebraic editor (type any SymPy expression using y(x), y1(x), …, y10(x), Dy(k,g), etc.)
- Optimized apply(LHS) with affine-compose derivative shortcut => avoids hangs when n>1
- Per‑ODE export buttons (LaTeX + ZIP)
- Batch -> ML toggle; defaults for q,v,a; robust fallbacks
- Model synthesizer builds new symbolic generators after training
- Initial conditions up to highest derivative order shown with solution
"""

# ============================================================================
# Standard libs
# ============================================================================
import os
import sys
import io
import json
import time
import zipfile
import logging
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import types

# ============================================================================
# Third-party
# ============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp

# Optionally torch
try:
    import torch
except Exception:
    torch = None

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# For SymPy internals / nodes
from sympy import Symbol, Function, Derivative
from sympy.core.function import AppliedUndef
from sympy.functions.combinatorial.numbers import stirling

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# Ensure src/ is on path (works in Railway & local)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

HAVE_SRC = True

# ============================================================================
# Resilient imports from src/
# ============================================================================
MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
LinearGeneratorFactory = CompleteLinearGeneratorFactory = None
NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
GeneratorConstructor = GeneratorSpecification = None
DerivativeTerm = DerivativeType = OperatorType = None
MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None
ODEClassifier = PhysicalApplication = None
BasicFunctions = SpecialFunctions = None
GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = create_model = None
MLTrainer = ODEDataset = ODEDataGenerator = None
GeneratorPattern = GeneratorPatternNetwork = GeneratorLearningSystem = None
ODENoveltyDetector = NoveltyAnalysis = ODETokenizer = ODETransformer = None
Settings = AppConfig = None
CacheManager = None

def _safe_cached_decorator(*args, **kwargs):
    def _wrap(fn): return fn
    return _wrap

cached = _safe_cached_decorator

try:
    from src.generators.master_generator import (
        MasterGenerator,
        EnhancedMasterGenerator,
        CompleteMasterGenerator,
    )
    try:
        from src.generators.master_generator import (
            CompleteLinearGeneratorFactory,
            CompleteNonlinearGeneratorFactory,
        )
    except Exception:
        from src.generators.linear_generators import (
            LinearGeneratorFactory,
            CompleteLinearGeneratorFactory,
        )
        from src.generators.nonlinear_generators import (
            NonlinearGeneratorFactory,
            CompleteNonlinearGeneratorFactory,
        )

    from src.generators.generator_constructor import (
        GeneratorConstructor,
        GeneratorSpecification,
        DerivativeTerm,
        DerivativeType,
        OperatorType,
    )

    from src.generators.master_theorem import (
        MasterTheoremSolver,
        MasterTheoremParameters,
        ExtendedMasterTheorem,
    )

    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication

    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions

    from src.ml.pattern_learner import (
        GeneratorPatternLearner,
        GeneratorVAE,
        GeneratorTransformer,
        create_model,
    )
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
    from src.ml.generator_learner import (
        GeneratorPattern,
        GeneratorPatternNetwork,
        GeneratorLearningSystem,
    )

    from src.dl.novelty_detector import (
        ODENoveltyDetector,
        NoveltyAnalysis,
        ODETokenizer,
        ODETransformer,
    )

    from src.utils.config import Settings, AppConfig
    from src.utils.cache import CacheManager, cached as _cached
    cached = _cached
    from src.utils.validators import ParameterValidator
    from src.ui.components import UIComponents

except Exception as e:
    logger.warning(f"Some imports from src/ failed or are missing: {e}")
    HAVE_SRC = False

# ============================================================================
# Streamlit Page Config
# ============================================================================
st.set_page_config(
    page_title="Master Generators ODE System - Complete Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CSS
# ============================================================================
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.0rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.4rem; }
    .subtitle { font-size: 1.05rem; opacity: 0.95; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 12px; text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3; padding: .9rem; border-radius: 10px; margin: .9rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50; padding: 1rem; border-radius: 10px; margin: 1rem 0;
    }
    .latex-export-box{
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border: 2px solid #9c27b0; padding:.9rem; border-radius: 10px; margin:.9rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# LaTeX Exporter
# ============================================================================
class LaTeXExporter:
    @staticmethod
    def _sx(expr) -> str:
        if expr is None:
            return ""
        try:
            if isinstance(expr, str):
                try:
                    expr = sp.sympify(expr)
                except Exception:
                    return expr
            return sp.latex(expr)
        except Exception:
            return str(expr)

    @staticmethod
    def generate_latex_document(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        gen = ode_data.get("generator", "")
        rhs = ode_data.get("rhs", "")
        sol = ode_data.get("solution", "")
        params = ode_data.get("parameters", {}) or {}
        classification = ode_data.get("classification", {}) or {}
        initial_conditions = ode_data.get("initial_conditions", {}) or {}
        title = ode_data.get("title", "Master Generators ODE System")

        parts = []
        if include_preamble:
            parts.append(r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\usepackage{hyperref}
\geometry{margin=1in}
\title{""" + title + r"""}
\author{Generated by Master Generators App}
\date{\today}
\begin{document}
\maketitle

\section{Generated Ordinary Differential Equation}
""")

        parts += [
            r"\subsection{Generator Equation}",
            r"\begin{equation}",
            f"{LaTeXExporter._sx(gen)} = {LaTeXExporter._sx(rhs)}",
            r"\end{equation}",
            "",
            r"\subsection{Exact Solution}",
            r"\begin{equation}",
            "y(x) = " + LaTeXExporter._sx(sol),
            r"\end{equation}",
            "",
            r"\subsection{Parameters}",
            r"\begin{align}",
            r"\alpha &= " + LaTeXExporter._sx(params.get("alpha", "")) + r" \\",
            r"\beta  &= " + LaTeXExporter._sx(params.get("beta", "")) + r" \\",
            r"n      &= " + LaTeXExporter._sx(params.get("n", "")) + r" \\",
            r"M      &= " + LaTeXExporter._sx(params.get("M", "")),
        ]
        if "q" in params:
            parts.append(r" \\ q &= " + LaTeXExporter._sx(params["q"]))
        if "v" in params:
            parts.append(r" \\ v &= " + LaTeXExporter._sx(params["v"]))
        if "a" in params:
            parts.append(r" \\ a &= " + LaTeXExporter._sx(params["a"]))
        parts.append(r"\end{align}")

        if initial_conditions:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            items = list(initial_conditions.items())
            for i, (k, v) in enumerate(items):
                sep = r" \\" if i < len(items) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter._sx(v)}{sep}")
            parts += [r"\end{align}"]

        if classification:
            parts += [
                r"\subsection{Mathematical Classification}",
                r"\begin{itemize}",
                r"\item \textbf{Type:} " + str(classification.get("type", "Unknown")),
                r"\item \textbf{Order:} " + str(classification.get("order", "Unknown")),
                r"\item \textbf{Linearity:} " + str(classification.get("linearity", "Unknown")),
            ]
            if "field" in classification:
                parts.append(r"\item \textbf{Field:} " + str(classification["field"]))
            if "applications" in classification:
                apps = classification.get("applications", [])
                if isinstance(apps, (list, tuple)):
                    parts.append(r"\item \textbf{Applications:} " + ", ".join(map(str, apps)))
            parts += [r"\end{itemize}"]

        parts += [
            r"\subsection{Solution Verification}",
            r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$.",
        ]

        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            latex_content = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
            zf.writestr("ode_document.tex", latex_content)
            zf.writestr("ode_data.json", json.dumps(ode_data, default=str, indent=2))
            zf.writestr("README.txt", f"Exported {datetime.now().isoformat()}")
            if include_extras:
                zf.writestr("reproduce.py", LaTeXExporter._repro_code(ode_data))
        zbuf.seek(0)
        return zbuf.getvalue()

    @staticmethod
    def _repro_code(ode_data: Dict[str, Any]) -> str:
        params = ode_data.get("parameters", {})
        gen_type = ode_data.get("type", "linear")
        func_name = ode_data.get("function_used", "exp")
        return f'''# Reproduction snippet
import sympy as sp
try:
    from src.generators.master_generator import CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory
except Exception:
    from src.generators.linear_generators import LinearGeneratorFactory as CompleteLinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory as CompleteNonlinearGeneratorFactory

z=sp.Symbol("z")
f_z = sp.exp(z) if "{func_name}"=="exp" else z
params={json.dumps(params, default=str)}

factory = CompleteLinearGeneratorFactory() if "{gen_type}"=="linear" else CompleteNonlinearGeneratorFactory()
# May need defaults for q,v,a depending on generator:
params.setdefault("q", 2); params.setdefault("v", 2); params.setdefault("a", 1)
print("Params:", params)
'''

# ============================================================================
# Session utilities/state
# ============================================================================
def _ensure_ss_key(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

def set_lhs_source(source: str):
    assert source in {"freeform", "constructor"}
    st.session_state["lhs_source"] = source

def get_active_generator_spec():
    _ensure_ss_key("lhs_source", "constructor")
    if st.session_state["lhs_source"] == "freeform":
        spec = st.session_state.get("freeform_gen_spec")
        if spec is not None:
            return spec
    return st.session_state.get("current_generator")

def _infer_type_from_spec(spec) -> str:
    try:
        return "linear" if bool(getattr(spec, "is_linear", False)) else "nonlinear"
    except Exception:
        pass
    desc = getattr(spec, "freeform_descriptor", None)
    if isinstance(desc, dict) and "terms" in desc:
        for t in desc["terms"]:
            if str(t.get("wrapper", "id")).lower() != "id":
                return "nonlinear"
            if int(t.get("power", 1)) != 1:
                return "nonlinear"
        return "linear"
    # If algebraic free‑form exists, we decide by naive heuristic: presence of nonlinearity signs
    algebraic = getattr(spec, "freeform_algebraic", None)
    if algebraic is not None:
        # if it contains pow/sin/log on y(...), treat as nonlinear
        if any(s in str(algebraic) for s in ["sin(", "cos(", "tanh(", "sinh(", "log(", "Abs(", "**2", "**3"]):
            return "nonlinear"
        return "linear"
    return "nonlinear"

def _max_y_derivative_in_expr(expr: sp.Expr, x: sp.Symbol, yname="y") -> int:
    """
    Highest order of Derivative(y(g(x)), (x,k)) encountered (order w.r.t. x).
    If only y(x) appears (no Derivative nodes), returns 0.
    """
    y = Function(yname)
    maxk = 0
    for d in expr.atoms(Derivative):
        try:
            if d.expr.func == y:
                # order w.r.t. x
                try:
                    k = sum(c for v, c in d.variable_count if v == x)
                except Exception:
                    k = sum(1 for v in d.variables if v == x)
                if k > maxk:
                    maxk = k
        except Exception:
            pass
    # also consider bare y(x) as order 0
    for fnode in expr.atoms(sp.Function):
        if fnode.func == y:
            maxk = max(maxk, 0)
    return maxk

def _infer_order_from_spec(spec) -> int:
    # Prefer symbolic inspection of LHS if available
    L = getattr(spec, "lhs", None)
    if L is not None:
        x = sp.Symbol("x", real=True)
        try:
            return _max_y_derivative_in_expr(L, x, "y")
        except Exception:
            pass
    # fallback to meta
    try:
        if hasattr(spec, "order"):
            return int(spec.order)
    except Exception:
        pass
    desc = getattr(spec, "freeform_descriptor", None)
    if isinstance(desc, dict) and "terms" in desc:
        try:
            return max(int(t.get("inner_order", 0)) + int(t.get("outer_order", 0)) for t in desc["terms"])
        except Exception:
            return 0
    return 0

def register_generated_ode(result: dict):
    _ensure_ss_key("generated_odes", [])
    rec = dict(result)
    rec.setdefault("type", "nonlinear")
    rec.setdefault("order", 0)
    rec.setdefault("function_used", "unknown")
    rec.setdefault("parameters", {})
    rec.setdefault("classification", {})
    rec.setdefault("timestamp", datetime.now().isoformat())
    rec["generator_number"] = len(st.session_state.generated_odes) + 1
    cl = rec["classification"]
    cl.setdefault("type", "Linear" if rec["type"] == "linear" else "Nonlinear")
    cl.setdefault("order", rec["order"])
    cl.setdefault("linearity", "Linear" if rec["type"] == "linear" else "Nonlinear")
    cl.setdefault("field", "Mathematical Physics")
    cl.setdefault("applications", ["Research Equation"])
    rec["classification"] = cl
    if "ode" not in rec and all(k in rec for k in ("generator", "rhs")):
        try:
            rec["ode"] = sp.Eq(rec["generator"], rec["rhs"])
        except Exception:
            rec["ode"] = f"{rec.get('generator')} = {rec.get('rhs')}"
    st.session_state.generated_odes.append(rec)

def initialize_session():
    _ensure_ss_key("generator_constructor", GeneratorConstructor() if GeneratorConstructor else None)
    _ensure_ss_key("generator_terms", [])
    _ensure_ss_key("generated_odes", [])
    _ensure_ss_key("generator_patterns", [])
    _ensure_ss_key("ml_trainer", None)
    _ensure_ss_key("ml_trained", False)
    _ensure_ss_key("training_history", {"train_loss": [], "val_loss": []})
    _ensure_ss_key("batch_results", [])
    _ensure_ss_key("batch_records_full", [])
    _ensure_ss_key("analysis_results", [])
    _ensure_ss_key("cache_manager", CacheManager() if CacheManager else None)
    _ensure_ss_key("basic_functions", BasicFunctions() if BasicFunctions else None)
    _ensure_ss_key("special_functions", SpecialFunctions() if SpecialFunctions else None)
    _ensure_ss_key("ode_classifier", ODEClassifier() if ODEClassifier else None)
    _ensure_ss_key("theorem_solver", MasterTheoremSolver() if MasterTheoremSolver else None)
    _ensure_ss_key("extended_theorem", ExtendedMasterTheorem() if ExtendedMasterTheorem else None)
    _ensure_ss_key("freeform_terms", [])
    _ensure_ss_key("freeform_gen_spec", None)
    _ensure_ss_key("lhs_source", "constructor")
    _ensure_ss_key("last_solution", None)
    _ensure_ss_key("last_rhs", None)
    _ensure_ss_key("last_lhs", None)

# ============================================================================
# Theorem 4.1 & 4.2 (symbolic)
# ============================================================================
def simplify_expr(expr: sp.Expr, level: str = "light") -> sp.Expr:
    if level == "none":
        return expr
    try:
        e = sp.together(expr)
        e = sp.cancel(e)
        e = sp.simplify(e)
        if level == "aggressive":
            try:
                e = sp.nsimplify(e, [sp.E, sp.pi, sp.I], rational=True, maxsteps=40)
            except Exception:
                pass
        return e
    except Exception:
        return expr

def to_exact(v):
    try:
        return sp.nsimplify(v, rational=True)
    except Exception:
        return sp.sympify(v)

@sp.cacheit
def omega_list(n: int):
    return tuple(sp.Rational(2*s-1, 2*n)*sp.pi for s in range(1, int(n)+1))

def theorem_4_1_solution_expr(f_of_z, alpha, beta, n: int, M, x: sp.Symbol, simplify_level="light"):
    z = sp.Symbol("z", real=True)
    base = f_of_z.subs(z, alpha + beta)
    parts = []
    for ω in omega_list(int(n)):
        epos = sp.exp(sp.I*x*sp.cos(ω) - x*sp.sin(ω))
        eneg = sp.exp(-sp.I*x*sp.cos(ω) - x*sp.sin(ω))
        psi = f_of_z.subs(z, alpha + beta*epos)
        phi = f_of_z.subs(z, alpha + beta*eneg)
        parts.append(2*base - psi - phi)
    y = sp.pi/(2*n)*sp.Add(*parts) + sp.pi*M
    return simplify_expr(y, level=simplify_level)

def theorem_4_2_mth_derivative_expr(f_of_z, alpha, beta, n: int, m: int, x: sp.Symbol):
    z = sp.Symbol("z")
    res = 0
    for s in range(1, n+1):
        ω = ((2*s - 1) * sp.pi) / (2*n)
        λ = sp.exp(sp.I*(sp.pi/2 + ω))
        ζ  = sp.exp(-x*sp.sin(ω)) * sp.exp(sp.I * x*sp.cos(ω))
        ζc = sp.exp(-x*sp.sin(ω)) * sp.exp(-sp.I * x*sp.cos(ω))
        ψ = f_of_z.subs(z, alpha + beta*ζ)
        φ = f_of_z.subs(z, alpha + beta*ζc)
        term_ψ = λ**m * sum(stirling(m, j, kind=2) * (beta*ζ)**j * sp.diff(ψ, alpha, j) for j in range(1, m+1))
        term_φ = sp.conjugate(λ)**m * sum(stirling(m, j, kind=2) * (beta*ζc)**j * sp.diff(φ, alpha, j) for j in range(1, m+1))
        res += term_ψ + term_φ
    return - (sp.pi/(2*n)) * sp.simplify(res)

# ============================================================================
# Function resolver
# ============================================================================
def get_function_expr(source, name: str) -> sp.Expr:
    z = sp.Symbol("z")
    # Provided library?
    if source == "Basic" and st.session_state.get("basic_functions"):
        try:
            obj = st.session_state.basic_functions.get_function(name)
            return sp.sympify(obj) if isinstance(obj, sp.Expr) else obj(z)
        except Exception:
            pass
    if source == "Special" and st.session_state.get("special_functions"):
        try:
            obj = st.session_state.special_functions.get_function(name)
            return sp.sympify(obj) if isinstance(obj, sp.Expr) else obj(z)
        except Exception:
            pass
    # Try to sympify free text (exp(z), z**2+1, etc.)
    try:
        return sp.sympify(name, locals={"z": z, "E": sp.E, "pi": sp.pi})
    except Exception as e:
        raise ValueError(f"Unknown f(z) '{name}': {e}")

# ============================================================================
# Free‑form builders: term-based (existing) and NEW algebraic editor
# ============================================================================
EPS = sp.Symbol("epsilon", positive=True)

def build_freeform_term(x: sp.Symbol, coef=1, inner_order=0, wrapper="id", power=1,
                        arg_scale=None, arg_shift=None, outer_order=0,
                        y_name="y") -> sp.Expr:
    yfun = sp.Function(y_name)
    arg = x if arg_scale in (None, 0) and arg_shift in (None, 0) else (x/(arg_scale or 1) + (arg_shift or 0))
    base = yfun(arg)
    if inner_order > 0:
        base = sp.diff(base, (x, int(inner_order)))

    if wrapper == "id":
        core = base
    elif wrapper == "exp":
        core = sp.exp(base)
    elif wrapper == "sin":
        core = sp.sin(base)
    elif wrapper == "cos":
        core = sp.cos(base)
    elif wrapper == "sinh":
        core = sp.sinh(base)
    elif wrapper == "cosh":
        core = sp.cosh(base)
    elif wrapper == "tanh":
        core = sp.tanh(base)
    elif wrapper == "log":
        core = sp.log(EPS + sp.Abs(base))
    elif wrapper == "abs":
        core = sp.Abs(base)
    else:
        try:
            fn = getattr(sp, wrapper)
            core = fn(base)
        except Exception:
            core = base

    term = sp.Integer(1)*coef * (core**power)
    if outer_order > 0:
        term = sp.diff(term, (x, int(outer_order)))
    return term

def build_freeform_lhs(x: sp.Symbol, terms: list, y_name="y") -> sp.Expr:
    if not terms:
        return sp.Symbol("LHS")
    return sp.Add(*[build_freeform_term(x, **t, y_name=y_name) for t in terms])

# ---------- NEW: algebraic LHS parser ----------
def _algebraic_locals(x: sp.Symbol, yname="y") -> Dict[str, Any]:
    y = sp.Function(yname)

    def Dy(k, g=None):
        if g is None:
            g = x
        return sp.Derivative(y(g), (x, int(k)))

    # y1..y10 shorthand
    def _y_k(k):
        return lambda g=None: sp.Derivative(y(x if g is None else g), (x, k))

    loc = {
        "x": x, "y": y, "Dy": Dy,
        "epsilon": EPS, "E": sp.E, "pi": sp.pi,
        # common wrappers
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "Abs": sp.Abs, "sqrt": sp.sqrt,
        "sign": sp.sign, "Heaviside": sp.Heaviside,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sech": sp.sech, "csch": sp.csch, "coth": sp.coth,
        "erf": sp.erf, "erfc": sp.erfc,
        "gamma": sp.gamma, "loggamma": sp.loggamma,
        "airyai": sp.airyai, "airybi": sp.airybi,
        "besselj": sp.besselj, "bessely": sp.bessely,
    }
    for k in range(1, 11):
        loc[f"y{k}"] = _y_k(k)
    return loc

def parse_algebraic_lhs(expr_text: str, x: sp.Symbol, yname="y") -> sp.Expr:
    """
    Safe-ish parser for user-supplied SymPy expressions.
    Supports y(x), y1(x)=d/dx y(x), …, y10(x), Dy(k, g), delays via using g(x)=x-τ, etc.
    """
    if not expr_text or not str(expr_text).strip():
        raise ValueError("Empty generator expression.")
    loc = _algebraic_locals(x, yname=yname)
    try:
        expr = sp.sympify(expr_text, locals=loc, convert_xor=True)
        return expr
    except Exception as e:
        raise ValueError(f"Failed to parse algebraic LHS: {e}")

# ============================================================================
# Apply LHS to y(x): optimized for affine g(x)
# ============================================================================
def _deriv_order_wrt_x(d: Derivative, x: sp.Symbol) -> int:
    try:
        return sum(c for v, c in d.variable_count if v == x)
    except Exception:
        return sum(1 for v in d.variables if v == x)

def _is_affine_in_x(arg: sp.Expr, x: sp.Symbol) -> Tuple[bool, sp.Expr]:
    """
    Returns (True, a) if arg = a*x + b (a constant in x), else (False, None).
    """
    try:
        d2 = sp.simplify(sp.diff(arg, (x, 2)))
        if d2 != 0:
            return (False, None)
        a = sp.simplify(sp.diff(arg, x))
        return (True, a)
    except Exception:
        return (False, None)

def apply_lhs_to_solution(lhs_expr: sp.Expr, y_expr: sp.Expr, x: sp.Symbol, y_name="y", simplify_level="light") -> sp.Expr:
    """
    Substitute y(g(x)), Derivative(y(g(x)), (x,k)) with y_expr composed at g(x),
    using affine shortcut: if g''(x)=0 then d^k/dx^k y_expr(g(x)) = (g')^k * y_expr^{(k)}(g(x)).
    """
    y = Function(y_name)
    mapping = {}

    # Precompute derivatives of y_expr at x for reuse
    # (used when arg == x or when using affine shortcut + substitution)
    d_cache = {0: y_expr}
    max_needed = 0

    # Collect needs
    needs = []
    for node in lhs_expr.atoms(AppliedUndef, Derivative):
        if isinstance(node, AppliedUndef) and node.func == y and len(node.args) == 1:
            arg = node.args[0]
            needs.append(("y", node, arg, 0))
        elif isinstance(node, Derivative) and isinstance(node.expr, AppliedUndef) and node.expr.func == y:
            arg = node.expr.args[0]
            k = _deriv_order_wrt_x(node, x)
            needs.append(("dy", node, arg, k))
            if arg == x:
                max_needed = max(max_needed, k)

    for k in range(1, max_needed + 1):
        d_cache[k] = sp.diff(y_expr, (x, k))

    local_cache = {}

    def eval_at(arg_expr: sp.Expr, order: int) -> sp.Expr:
        key = (sp.srepr(arg_expr), order)
        if key in local_cache:
            return local_cache[key]
        if arg_expr == x:
            val = d_cache[order]
        else:
            affine, a = _is_affine_in_x(arg_expr, x)
            if affine:
                # affine shortcut
                # d^k/dx^k [y_expr(arg)] = (a)^k * y_expr^{(k)}(arg)
                val = (a**order) * sp.diff(y_expr, (x, order)).subs(x, arg_expr)
            else:
                # general fallback
                val = sp.diff(y_expr.subs(x, arg_expr), (x, order))
        local_cache[key] = val
        return val

    for typ, node, arg, k in needs:
        if k == 0:
            mapping[node] = eval_at(arg, 0)
        else:
            mapping[node] = eval_at(arg, k)

    try:
        out = lhs_expr.xreplace(mapping)
    except Exception:
        out = lhs_expr.subs(mapping)

    return simplify_expr(out, level=simplify_level)

# ============================================================================
# Timeout helpers
# ============================================================================
import concurrent.futures as _fut

def _worker_t41(f_expr, alpha, beta, n, M, xname, simplify_level):
    x = sp.Symbol(xname, real=True)
    return theorem_4_1_solution_expr(f_expr, alpha, beta, int(n), M, x, simplify_level)

def _worker_apply(lhs_expr, solution_y, xname, yname, simplify_level):
    x = sp.Symbol(xname, real=True)
    return apply_lhs_to_solution(lhs_expr, solution_y, x, y_name=yname, simplify_level=simplify_level)

def _worker_ic(sol, xname, j):
    x = sp.Symbol(xname, real=True)
    return sp.simplify(sp.diff(sol, (x, int(j))).subs(x, 0))

def run_with_timeout(fn, timeout_sec, *args):
    if not timeout_sec or timeout_sec <= 0:
        return fn(*args)
    try:
        with _fut.ProcessPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, *args)
            return fut.result(timeout=timeout_sec)
    except _fut.TimeoutError:
        raise TimeoutError(f"Operation exceeded {timeout_sec} seconds")
    except Exception:
        # fallback inline (best effort)
        return fn(*args)

# ============================================================================
# UI Pages
# ============================================================================
def page_header():
    st.markdown(
        """
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Free‑form algebraic generators • Theorems 4.1 & 4.2 • ML/DL • Exports</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def dashboard_page():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>🧬 ML Patterns</h3><h1>{len(st.session_state.generator_patterns)}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>📊 Batch Results</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c4:
        model_status = "✅ Trained" if st.session_state.get("ml_trained") else "⏳ Not Trained"
        st.markdown(f'<div class="metric-card"><h3>🤖 ML Model</h3><p style="font-size: 1.2rem;">{model_status}</p></div>', unsafe_allow_html=True)

    st.subheader("📊 Recent ODEs")
    if st.session_state.generated_odes:
        df = pd.DataFrame(
            [
                {"No": r.get("generator_number"),
                 "Type": r.get("type"),
                 "Order": r.get("order"),
                 "Func": r.get("function_used"),
                 "Time": r.get("timestamp", "")[:19]}
                for r in st.session_state.generated_odes[-10:]
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No ODEs yet. Try **Apply Master Theorem**.")

def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Use your project Constructor (term list), or use Free‑form on the theorem page.</div>', unsafe_allow_html=True)

    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found in src/. You can still use the Free‑form builder on the theorem page.")
        return

    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5,6])
        with c2:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType])
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)
        c5, c6, c7 = st.columns(3)
        with c5:
            operator_type = st.selectbox("Operator Type", [t.value for t in OperatorType])
        with c6:
            scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if operator_type in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay","advance"] else None

        if st.button("➕ Add Term", type="primary"):
            term = DerivativeTerm(
                derivative_order=int(deriv_order),
                coefficient=float(coefficient),
                power=int(power),
                function_type=DerivativeType(func_type),
                operator_type=OperatorType(operator_type),
                scaling=float(scaling) if scaling is not None else None,
                shift=float(shift) if shift is not None else None,
            )
            st.session_state.generator_terms.append(term)
            st.success("Term added.")

    if st.session_state.generator_terms:
        st.subheader("📝 Current Terms")
        for i, term in enumerate(st.session_state.generator_terms, 1):
            desc = term.get_description() if hasattr(term, "get_description") else str(term)
            st.write(f"{i}. {desc}")

        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(
                    terms=st.session_state.generator_terms,
                    name=f"Custom Generator {len(st.session_state.generated_odes) + 1}",
                )
                st.session_state.current_generator = gen_spec
                set_lhs_source("constructor")
                st.success("Specification built & selected.")
                try:
                    st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to build specification: {e}")

    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None

# -------------------- Apply Master Theorem --------------------
def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem")

    # LHS source
    sel = st.radio(
        "LHS source",
        options=("constructor", "freeform"),
        index=0 if st.session_state.get("lhs_source","constructor") == "constructor" else 1,
        format_func=lambda s: {"constructor":"Constructor LHS", "freeform":"Free‑form LHS"}[s],
        horizontal=True,
    )
    set_lhs_source(sel)

    col1, col2 = st.columns([1,1])
    with col1:
        lib = st.selectbox("Function library", ["Basic", "Special"], index=0)
        names = []
        try:
            if lib == "Basic" and st.session_state.basic_functions:
                names = st.session_state.basic_functions.get_function_names()
            elif lib == "Special" and st.session_state.special_functions:
                names = st.session_state.special_functions.get_function_names()
        except Exception:
            pass
        func_name = st.selectbox("Select f(z)", names) if names else st.text_input("Enter f(z) (name or expression)", "exp(z)")
    with col2:
        alpha = st.text_input("α", "1")
        beta  = st.text_input("β", "1")
        n     = st.number_input("n ≥ 1", 1, 20, 1)
        M     = st.text_input("M", "0")

    cA, cB, cC = st.columns(3)
    with cA:
        use_exact = st.checkbox("Exact (symbolic) parameters", value=True)
    with cB:
        simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with cC:
        timeout_sec = st.slider("Timeout (sec)", 0, 90, 20, help="0 = unlimited")

    # Thm 4.2 (optional)
    with st.expander("Theorem 4.2 (optional)"):
        m = st.number_input("m ≥ 1", 1, 20, 1)
        if st.button("Compute y^{(m)}(x)"):
            try:
                f_expr = get_function_expr(lib, func_name)
                x = sp.Symbol("x", real=True)
                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)
                y_m = theorem_4_2_mth_derivative_expr(f_expr, α, β, int(n), int(m), x)
                st.latex(fr"y^{{({int(m)})}}(x) = " + sp.latex(y_m))
            except Exception as e:
                st.error(f"Thm 4.2 error: {e}")

    # ---------- FREE‑FORM builders ----------
    st.subheader("🧩 Free‑form LHS (optional)")
    with st.expander("A) Term‑based builder (wrappers, delays)", expanded=False):
        if "freeform_terms" not in st.session_state:
            st.session_state.freeform_terms = []
        cols = st.columns([1,1,1,1,1,1,1])
        with cols[0]:
            coef = st.text_input("coef", "1")
        with cols[1]:
            inner_order = st.number_input("inner k", 0, 12, 1)
        with cols[2]:
            wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","sinh","cosh","tanh","log","abs"], index=0)
        with cols[3]:
            power = st.number_input("power", 1, 12, 1)
        with cols[4]:
            outer_order = st.number_input("outer m", 0, 12, 0)
        with cols[5]:
            a = st.text_input("arg scale a", "1")
        with cols[6]:
            b = st.text_input("arg shift b", "0")
        if st.button("➕ Add free‑form term"):
            try:
                st.session_state.freeform_terms.append({
                    "coef": to_exact(coef),
                    "inner_order": int(inner_order),
                    "wrapper": wrapper,
                    "power": int(power),
                    "outer_order": int(outer_order),
                    "arg_scale": to_exact(a),
                    "arg_shift": to_exact(b),
                })
                st.success("Term added.")
            except Exception as e:
                st.error(f"Failed to add term: {e}")

        if st.session_state.freeform_terms:
            st.write("**Current terms:**")
            for i, t in enumerate(st.session_state.freeform_terms, 1):
                st.write(f"{i}. coef={t['coef']} · D^{t['outer_order']}[ {t['wrapper']}((y^{t['inner_order']})(x/{t.get('arg_scale',1)}+{t.get('arg_shift',0)}))^{t['power']} ]")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("🔨 Build & Select this LHS"):
                    x = sp.Symbol("x", real=True)
                    lhs = build_freeform_lhs(x, st.session_state.freeform_terms)
                    spec = types.SimpleNamespace()
                    spec.lhs = lhs
                    spec.freeform_descriptor = {"terms": list(st.session_state.freeform_terms)}
                    try:
                        spec.order = _max_y_derivative_in_expr(lhs, x, "y")
                        spec.is_linear = False
                    except Exception:
                        pass
                    st.session_state.freeform_gen_spec = spec
                    set_lhs_source("freeform")
                    st.success("Stored & selected.")
                    try:
                        st.latex(sp.latex(lhs))
                    except Exception:
                        st.code(str(lhs))
            with c2:
                if st.button("🗑️ Clear terms"):
                    st.session_state.freeform_terms = []
            with c3:
                st.info("Use the Algebraic editor below for full control (products, ratios, etc.).")

    with st.expander("B) ✍️ Free‑form Algebraic LHS (type any SymPy) — Recommended", expanded=True):
        st.markdown(
            """
**Syntax help**  
- Use **y(x)**, **y1(x)=d/dx y(x)**, **y2(x)**, … **y10(x)**; or **Dy(k, g)** for \(d^k/dx^k\,y(g(x))\).  
- You can multiply/divide and compose freely, e.g. ` (y(x) + y1(x)/(1+y(x/2))) - sinh(y5(x)) `.  
- Wrappers include: `sin, cos, tan, sinh, cosh, tanh, exp, log, Abs, sqrt, sign, erf, erfc, gamma, loggamma, airyai, airybi, besselj, bessely`.  
- `epsilon` is available for safe logs: `log(epsilon+Abs(y1(x)))`.
            """
        )
        alg_text = st.text_area(
            "Enter LHS expression",
            value="y(x) + log(epsilon+Abs(y1(x))) - sinh(y5(x))",
            height=130,
        )
        if st.button("🔨 Parse & Select Algebraic LHS"):
            try:
                x = sp.Symbol("x", real=True)
                lhs = parse_algebraic_lhs(alg_text, x, yname="y")
                spec = types.SimpleNamespace()
                spec.lhs = lhs
                spec.freeform_algebraic = alg_text
                spec.order = _max_y_derivative_in_expr(lhs, x, "y")
                spec.is_linear = False
                st.session_state.freeform_gen_spec = spec
                set_lhs_source("freeform")
                st.success("Algebraic LHS stored & selected.")
                try:
                    st.latex(sp.latex(lhs))
                except Exception:
                    st.code(str(lhs))
            except Exception as e:
                st.error(f"Parse error: {e}")

    # ---------- Generate ODE ----------
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        with st.spinner("Applying Theorem 4.1 and constructing RHS…"):
            try:
                f_expr = get_function_expr(lib, func_name)
                x = sp.Symbol("x", real=True)
                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)
                𝑀 = to_exact(M)     if use_exact else sp.Float(M)

                solution = run_with_timeout(_worker_t41, timeout_sec, f_expr, α, β, int(n), 𝑀, "x", simplify_level)

                active_spec = get_active_generator_spec()
                lhs_to_use = None
                if active_spec and hasattr(active_spec, "lhs") and active_spec.lhs is not None:
                    lhs_to_use = active_spec.lhs
                else:
                    lhs_to_use = sp.Symbol("L[y]")

                z = sp.Symbol("z")
                default_rhs = simplify_expr(sp.pi*(f_expr.subs(z, α+β) + 𝑀), level=simplify_level)

                if lhs_to_use != sp.Symbol("L[y]"):
                    try:
                        rhs = run_with_timeout(_worker_apply, timeout_sec, lhs_to_use, solution, "x", "y", simplify_level)
                        generator_lhs = lhs_to_use
                    except Exception as e:
                        logger.warning(f"apply(LHS) failed, fallback RHS. Reason: {e}")
                        rhs = default_rhs
                        generator_lhs = sp.Symbol("L[y]")
                else:
                    rhs = default_rhs
                    generator_lhs = lhs_to_use

                # Compute initial conditions up to highest derivative order in LHS
                max_order = _max_y_derivative_in_expr(generator_lhs if isinstance(generator_lhs, sp.Expr) else sp.Symbol("L[y]"), x, "y")
                ICs = {}
                try:
                    upto = max(0, max_order-1)
                    for j in range(0, upto+1):
                        try:
                            val = run_with_timeout(_worker_ic, min(10, max(3, timeout_sec//2)), solution, "x", j)
                        except TimeoutError:
                            val = sp.diff(solution, (x, j)).subs(x, 0)
                        key = "y(0)" if j == 0 else fr"y^({j})(0)"
                        ICs[key] = sp.simplify(val)
                except Exception:
                    pass

                ode_type = _infer_type_from_spec(active_spec) if active_spec else "nonlinear"
                ode_order = max_order

                result = {
                    "generator": generator_lhs,
                    "rhs": rhs,
                    "solution": solution,
                    "parameters": {"alpha": α, "beta": β, "n": int(n), "M": 𝑀},
                    "function_used": str(func_name),
                    "type": ode_type,
                    "order": ode_order,
                    "classification": {
                        "type": "Linear" if ode_type == "linear" else "Nonlinear",
                        "order": ode_order,
                        "linearity": "Linear" if ode_type == "linear" else "Nonlinear",
                        "field": "Mathematical Physics",
                        "applications": ["Research Equation"],
                    },
                    "initial_conditions": ICs,
                    "timestamp": datetime.now().isoformat(),
                    "lhs_source": st.session_state.get("lhs_source","constructor"),
                    "title": "Master Generators ODE System",
                }
                register_generated_ode(result)

                st.markdown('<div class="result-box"><b>✅ ODE Generated!</b></div>', unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["📐 Equation", "💡 Solution & ICs", "📤 Export"])
                with t1:
                    try:
                        st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                    except Exception:
                        st.code(f"LHS: {generator_lhs}\nRHS: {rhs}")
                    st.caption(f"LHS source: **{st.session_state.get('lhs_source','constructor')}**")
                with t2:
                    try:
                        st.latex("y(x) = " + sp.latex(solution))
                    except Exception:
                        st.code(f"y(x) = {solution}")
                    if result["initial_conditions"]:
                        st.markdown("**Initial conditions (at x=0):**")
                        lines = []
                        for k, v in result["initial_conditions"].items():
                            lines.append(f"${k} = {sp.latex(v)}$")
                        st.write(",  ".join(lines))
                    st.write(f"**Parameters:** α={α}, β={β}, n={int(n)}, M={𝑀}")
                with t3:
                    ode_idx = len(st.session_state.generated_odes)
                    ode_data = dict(result); ode_data["generator_number"] = ode_idx
                    latex_doc = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
                    st.download_button("📄 Download LaTeX", latex_doc, file_name=f"ode_{ode_idx}.tex", mime="text/x-latex")
                    pkg = LaTeXExporter.create_export_package(ode_data, include_extras=True)
                    st.download_button("📦 Download ZIP", pkg, file_name=f"ode_package_{ode_idx}.zip", mime="application/zip")

            except TimeoutError as te:
                st.error(str(te))
            except Exception as e:
                logger.error("Generation error", exc_info=True)
                st.error(f"Error generating ODE: {e}")

# -------------------- ML Pattern Learning --------------------
def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning")
    if not MLTrainer:
        st.info("MLTrainer not available here.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Patterns", len(st.session_state.generator_patterns))
    with c2: st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c3: st.metric("Training Epochs", len(st.session_state.training_history.get("train_loss", [])))
    with c4: st.metric("Model Status", "Trained" if st.session_state.get("ml_trained") else "Not Trained")

    st.markdown("You can import batch outputs into the training corpus:")
    cimp1, cimp2 = st.columns([2,1])
    with cimp1:
        if st.button("📥 Import last-batch into training corpus"):
            if st.session_state.batch_records_full:
                added = 0
                for rec in st.session_state.batch_records_full:
                    try:
                        register_generated_ode(rec)
                        added += 1
                    except Exception:
                        pass
                st.success(f"Imported {added} batch ODEs into corpus.")
            else:
                st.info("No batch records available. Run a batch first.")
    with cimp2:
        st.write("")

    model_type = st.selectbox("Select ML Model", ["pattern_learner", "vae", "transformer"],
                              format_func=lambda x: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[x])

    with st.expander("Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 10, 800, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
            samples = st.slider("Training Samples", 100, 5000, 1000)
        with c3:
            val_split = st.slider("Validation Split", 0.05, 0.3, 0.2)
            use_gpu = st.checkbox("Use GPU if available", value=True)
    cES1, cES2, cES3 = st.columns(3)
    with cES1:
        until_plateau = st.checkbox("Train until plateau (early stopping)", value=False)
    with cES2:
        patience = st.slider("Patience", 3, 20, 7)
    with cES3:
        min_delta = st.select_slider("Min improvement Δ", options=[1e-4,5e-4,1e-3,5e-3,1e-2], value=1e-3)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate/import at least 5 ODEs before training.")
        return

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            try:
                device = "cuda" if (use_gpu and torch and torch.cuda.is_available()) else "cpu"
                trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
                st.session_state.ml_trainer = trainer
                history = {"train_loss": [], "val_loss": []}

                def do_train(eps):
                    trainer.train(epochs=eps, batch_size=batch_size, samples=samples,
                                  validation_split=val_split,
                                  progress_callback=lambda e,t: None)
                    # merge histories
                    for k in ("train_loss", "val_loss"):
                        if k in trainer.history:
                            history[k] += trainer.history[k]

                if not until_plateau:
                    do_train(epochs)
                else:
                    best = float("inf"); bad = 0; total_done = 0
                    while total_done < epochs:
                        step = min(50, epochs-total_done)
                        do_train(step)
                        total_done += step
                        last = history["val_loss"][-1] if history["val_loss"] else history["train_loss"][-1]
                        if last < best - min_delta:
                            best = last; bad = 0
                        else:
                            bad += 1
                            if bad >= patience:
                                break

                st.session_state.ml_trained = True
                st.session_state.training_history = history
                st.success("✅ Training complete.")
                if history["train_loss"]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=history["train_loss"], mode="lines", name="train"))
                    if history["val_loss"]:
                        fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="val"))
                    fig.update_layout(title="Training History", xaxis_title="epoch", yaxis_title="loss", height=360)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.subheader("🎨 Synthesize new generators after training")
    cgen1, cgen2 = st.columns([2,1])
    with cgen1:
        if st.button("🧪 Synthesize Free‑form LHS from model"):
            try:
                # heuristic synthesis from corpus (wrappers + derivative-order histogram)
                x = sp.Symbol("x", real=True)
                # derive order histogram
                orders = []
                for rec in st.session_state.generated_odes:
                    L = rec.get("generator")
                    if isinstance(L, sp.Expr):
                        orders.append(_max_y_derivative_in_expr(L, x, "y"))
                k = int(np.clip(int(np.median(orders)) if orders else 2, 1, 6))
                wrappers = ["id","exp","sin","cos","sinh","tanh","log","abs"]
                chosen = list(np.random.choice(wrappers, size=min(3, k), replace=True))
                terms = []
                for j, w in enumerate(chosen, 1):
                    terms.append({"coef": to_exact(1), "inner_order": j, "wrapper": w, "power": 1, "outer_order": 0, "arg_scale": 1, "arg_shift": 0})
                lhs = build_freeform_lhs(x, terms)
                spec = types.SimpleNamespace()
                spec.lhs = lhs
                spec.freeform_descriptor = {"terms": terms, "synth": True}
                spec.order = _max_y_derivative_in_expr(lhs, x, "y")
                spec.is_linear = False
                st.session_state.freeform_gen_spec = spec
                set_lhs_source("freeform")
                st.success("A synthesized generator has been stored & selected.")
                st.latex(sp.latex(lhs))
            except Exception as e:
                st.error(f"Synthesis failed: {e}")
    with cgen2:
        st.write("")

# -------------------- Batch Generation --------------------
def safe_create_linear(factory, gen_num, f_z, params):
    try:
        return factory.create(gen_num, f_z, **params)
    except Exception as e:
        return {}

def safe_create_nonlinear(factory, gen_num, f_z, params):
    # inject defaults for common requirements
    p = dict(params)
    p.setdefault("q", 2)
    p.setdefault("v", 2)
    p.setdefault("a", 1)
    try:
        return factory.create(gen_num, f_z, **p)
    except Exception:
        # try different defaults
        p["q"] = 3; p["v"] = 3; p["a"] = 2
        try:
            return factory.create(gen_num, f_z, **p)
        except Exception:
            return {}

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs quickly and (optionally) add them to ML training corpus.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 500, 50)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_cats = st.multiselect("Function libraries", ["Basic","Special"], default=["Basic"])
        vary_params = st.checkbox("Vary parameters", True)
    with c3:
        also_add_to_ml = st.checkbox("Also add batch outputs to ML corpus", True)

    with st.expander("Advanced"):
        export_format = st.selectbox("Export format", ["JSON","CSV","LaTeX","All"], index=1)
        include_solutions = st.checkbox("Include solution preview", True)

    if vary_params:
        a_rng = st.slider("α range", -5.0, 5.0, (-2.0, 2.0))
        b_rng = st.slider("β range", 0.1, 5.0, (0.5, 2.0))
        n_rng = st.slider("n range", 1, 5, (1, 3))
    else:
        a_rng = (1.0,1.0); b_rng=(1.0,1.0); n_rng=(1,1)

    if st.button("🚀 Generate Batch", type="primary"):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            results = []
            full_records = []

            all_names = []
            if "Basic" in func_cats and st.session_state.basic_functions:
                all_names += st.session_state.basic_functions.get_function_names()
            if "Special" in func_cats and st.session_state.special_functions:
                all_names += st.session_state.special_functions.get_function_names()[:20]
            if not all_names:
                all_names = ["exp", "sin", "cos", "z", "z**2"]

            prog = st.progress(0.0)
            for i in range(num_odes):
                prog.progress((i+1)/num_odes)
                try:
                    params = {
                        "alpha": float(np.random.uniform(*a_rng)),
                        "beta": float(np.random.uniform(*b_rng)),
                        "n": int(np.random.randint(n_rng[0], n_rng[1]+1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    fname = str(np.random.choice(all_names))
                    # Build f(z)
                    try:
                        f_z = st.session_state.basic_functions.get_function(fname)
                    except Exception:
                        try:
                            f_z = st.session_state.special_functions.get_function(fname)
                        except Exception:
                            z = sp.Symbol("z")
                            f_z = sp.exp(z)

                    gt = str(np.random.choice(gen_types))
                    rec = {}
                    if gt == "linear" and CompleteLinearGeneratorFactory:
                        factory = CompleteLinearGeneratorFactory()
                        gnum = int(np.random.randint(1, 9))
                        if gnum in [4,5]:
                            params["a"] = float(np.random.uniform(1, 3))
                        rec = safe_create_linear(factory, gnum, f_z, params)
                    elif gt == "nonlinear" and CompleteNonlinearGeneratorFactory:
                        factory = CompleteNonlinearGeneratorFactory()
                        gnum = int(np.random.randint(1, 11))
                        rec = safe_create_nonlinear(factory, gnum, f_z, params)
                    else:
                        rec = {}

                    if not rec:
                        # fallback via Thm 4.1 only
                        x = sp.Symbol("x", real=True)
                        yx = theorem_4_1_solution_expr(f_z, to_exact(params["alpha"]), to_exact(params["beta"]),
                                                       int(params["n"]), to_exact(params["M"]), x, "none")
                        lhs = sp.Function("L")(x)
                        z = sp.Symbol("z")
                        rhs = sp.pi*(f_z.subs(z, to_exact(params["alpha"])+to_exact(params["beta"])) + to_exact(params["M"]))
                        rec = {"ode": sp.Eq(lhs, rhs), "solution": yx, "type": gt, "order": 0,
                               "function_used": fname, "parameters": params}

                    # normalize + store simple row
                    row = {
                        "ID": i+1,
                        "Type": rec.get("type","unknown"),
                        "Function": fname,
                        "Order": rec.get("order", 0),
                        "α": round(params["alpha"], 4),
                        "β": round(params["beta"], 4),
                        "n": params["n"],
                    }
                    if include_solutions:
                        s = str(rec.get("solution",""))
                        row["Solution"] = (s[:120]+"...") if len(s)>120 else s
                    results.append(row)

                    # prepare a generated_odes‑compatible record for ML corpus
                    full = {
                        "generator": rec.get("ode").lhs if hasattr(rec.get("ode"), "lhs") else sp.Symbol("L[y]"),
                        "rhs": rec.get("ode").rhs if hasattr(rec.get("ode"), "rhs") else sp.Symbol("RHS"),
                        "solution": rec.get("solution"),
                        "parameters": rec.get("parameters", params),
                        "function_used": fname,
                        "type": rec.get("type","nonlinear"),
                        "order": rec.get("order", 0),
                        "classification": {"type":"Linear" if rec.get("type")=="linear" else "Nonlinear",
                                           "order": rec.get("order",0), "linearity":"Linear" if rec.get("type")=="linear" else "Nonlinear",
                                           "field":"Mathematical Physics","applications":["Research Equation"]},
                        "initial_conditions": {},
                        "timestamp": datetime.now().isoformat(),
                        "lhs_source": "batch",
                    }
                    full_records.append(full)

                except Exception as e:
                    logger.debug(f"Failed batch item {i+1}: {e}")

            st.session_state.batch_results.extend(results)
            st.session_state.batch_records_full = full_records

            if also_add_to_ml:
                added = 0
                for rec in full_records:
                    try:
                        register_generated_ode(rec); added += 1
                    except Exception:
                        pass
                st.success(f"Generated {len(results)} ODEs; added {added} to ML corpus.")
            else:
                st.success(f"Generated {len(results)} ODEs.")

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                csv = df.to_csv(index=False)
                st.download_button("📊 CSV", csv, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with c2:
                j = json.dumps(results, indent=2)
                st.download_button("📄 JSON", j, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            with c3:
                latex = "\\begin{tabular}{|c|c|c|c|c|}\\hline ID & Type & Function & n & Order \\\\ \\hline\n" + \
                        "\n".join([f"{r['ID']} & {r['Type']} & {r['Function']} & {r['n']} & {r['Order']} \\\\" for r in results[:30]]) + \
                        "\n\\hline\\end{tabular}"
                st.download_button("📝 LaTeX table", latex, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")
            with c4:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("batch.csv", df.to_csv(index=False))
                    zf.writestr("batch.json", json.dumps(results, indent=2))
                    zf.writestr("batch.tex", latex)
                    zf.writestr("README.txt", f"Batch generated {datetime.now().isoformat()}")
                zbuf.seek(0)
                st.download_button("📦 ZIP (all)", zbuf.getvalue(), file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

# -------------------- Novelty Detection --------------------
def novelty_detection_page():
    st.header("🔍 Novelty Detection")
    if not st.session_state.get("ode_classifier"):
        st.info("Classifier not available in this environment.")
        return
    st.info("Use your ODEClassifier / NoveltyDetector from src/ here if wired.")

# -------------------- Analysis & Classification --------------------
def analysis_classification_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("Generate ODEs first.")
        return
    df = pd.DataFrame(
        [{"No": r.get("generator_number"), "Type": r.get("type"), "Order": r.get("order"),
          "Function": r.get("function_used"), "Timestamp": r.get("timestamp","")[:19]}
         for r in st.session_state.generated_odes]
    )
    st.dataframe(df, use_container_width=True)
    types = pd.Series([r.get("type","unknown") for r in st.session_state.generated_odes]).value_counts()
    fig = px.pie(values=types.values, names=types.index, title="Type distribution")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Export & LaTeX --------------------
def export_latex_page():
    st.header("📤 Export & LaTeX")
    st.markdown('<div class="latex-export-box"><b>Export ODEs in LaTeX or as ZIP packages.</b></div>', unsafe_allow_html=True)
    if not st.session_state.generated_odes:
        st.info("No ODEs to export.")
        return
    idx = st.selectbox(
        "Select ODE",
        options=list(range(1, len(st.session_state.generated_odes) + 1)),
        format_func=lambda i: f"ODE #{i}: {st.session_state.generated_odes[i-1].get('type')} (order {st.session_state.generated_odes[i-1].get('order')})",
    )
    ode = st.session_state.generated_odes[idx - 1]
    st.subheader("Preview")
    try:
        preview = LaTeXExporter.generate_latex_document(ode, include_preamble=False)
        st.code(preview, language="latex")
    except Exception:
        st.write(ode)
    c1, c2 = st.columns(2)
    with c1:
        full_latex = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
        st.download_button("📄 Download LaTeX", full_latex, file_name=f"ode_{idx}.tex", mime="text/x-latex", use_container_width=True)
    with c2:
        pkg = LaTeXExporter.create_export_package(ode, include_extras=True)
        st.download_button("📦 Download ZIP", pkg, file_name=f"ode_package_{idx}.zip", mime="application/zip", use_container_width=True)

# -------------------- Visualization (placeholder) --------------------
def create_solution_plot(ode: Dict, x_range: Tuple, num_points: int) -> go.Figure:
    x = np.linspace(x_range[0], x_range[1], num_points)
    # This is a placeholder visually; symbolic numeric evaluation varies by f(z)
    y = np.sin(x) * np.exp(-0.1*np.abs(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Solution"))
    fig.update_layout(title="ODE Solution (demo)", xaxis_title="x", yaxis_title="y(x)")
    return fig

def visualization_page():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize.")
        return
    sel = st.selectbox(
        "Select ODE to Visualize",
        range(len(st.session_state.generated_odes)),
        format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')} (Order {st.session_state.generated_odes[i].get('order',0)})"
    )
    ode = st.session_state.generated_odes[sel]
    c1, c2, c3 = st.columns(3)
    with c1:
        plot_type = st.selectbox("Plot Type", ["Solution","Phase Portrait","3D Surface","Direction Field"])
    with c2:
        x_range = st.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    with c3:
        num_points = st.slider("Number of Points", 100, 2000, 500)

    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Creating visualization..."):
            try:
                if plot_type == "Solution":
                    fig = create_solution_plot(ode, x_range, num_points)
                else:
                    fig = go.Figure()  # Placeholder for other types
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization failed: {e}")

# -------------------- Docs & Settings --------------------
def documentation_page():
    st.header("📖 Documentation")
    st.markdown(
        r"""
**Theorem 4.1.**  
\[
y(x)=\frac{\pi}{2n}\sum_{s=1}^n\Big(2f(\alpha+\beta)-\psi_s(x)-\phi_s(x)\Big)+\pi M
\]
with \(\psi_s=f(\alpha+\beta e^{ix\cos\omega_s-x\sin\omega_s})\), \(\phi_s=f(\alpha+\beta e^{-ix\cos\omega_s-x\sin\omega_s})\), \(\omega_s=(2s-1)\pi/(2n)\).

**Theorem 4.2 (Stirling‑number compact form).**  
\[
y^{(m)}(x)=-\frac{\pi}{2n}\sum_{s=1}^n\left\{
\lambda_s^m\sum_{j=1}^m\mathbf{S}(m,j)(\beta\zeta_s)^j\partial_\alpha^{\,j}\psi
+\overline{\lambda_s}^{\,m}\sum_{j=1}^m\mathbf{S}(m,j)(\beta\overline{\zeta_s})^j\partial_\alpha^{\,j}\phi
\right\}.
\]
Here \(\zeta_s = e^{-x\sin\omega_s}e^{ix\cos\omega_s}\), \(\lambda_s=e^{i(\pi/2+\omega_s)}\), \(\omega_s=(2s-1)\pi/(2n)\).

**Free‑form Algebraic LHS**  
Type any SymPy expression for the operator \(L[y]\). Shorthands: `y(x)`, `y1(x)`, … `y10(x)`, `Dy(k,g)` for derivatives of \(y(g(x))\). Delays and affine transforms go in `g`: e.g. `Dy(3, x/2+1)`.
""",
        unsafe_allow_html=False,
    )

def settings_page():
    st.header("⚙️ Settings")
    if st.checkbox("Show session keys"):
        st.write(list(st.session_state.keys()))
    if st.button("Clear cache/state (safe)"):
        for k in ["generated_odes","batch_results","batch_records_full","generator_patterns","training_history"]:
            st.session_state[k] = []
        st.success("Cleared.")

# ============================================================================
# Main
# ============================================================================
def main():
    initialize_session()
    page_header()
    page = st.sidebar.radio(
        "📍 Navigation",
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
            "⚙️ Settings",
            "📖 Documentation",
        ],
    )

    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔧 Generator Constructor":
        generator_constructor_page()
    elif page == "🎯 Apply Master Theorem":
        page_apply_master_theorem()
    elif page == "🤖 ML Pattern Learning":
        ml_pattern_learning_page()
    elif page == "📊 Batch Generation":
        batch_generation_page()
    elif page == "🔍 Novelty Detection":
        novelty_detection_page()
    elif page == "📈 Analysis & Classification":
        analysis_classification_page()
    elif page == "📐 Visualization":
        visualization_page()
    elif page == "📤 Export & LaTeX":
        export_latex_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "📖 Documentation":
        documentation_page()

if __name__ == "__main__":
    main()
