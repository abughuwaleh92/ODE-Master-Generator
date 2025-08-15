# -*- coding: utf-8 -*-
"""
Master Generators for ODEs — Complete App (patched)
- Exact/symbolic parameter handling for Theorem 4.1
- Compact Theorem 4.2 via Stirling numbers (Faà di Bruno)
- Free-form LHS builder (sinh(y'), exp(y^(7)), ln(|y''|)+ε, powers, coefficients)
- LHS source selector (Constructor vs Free‑form)
- Per‑ODE export (LaTeX + ZIP) in Apply Master Theorem
- Immediate registration so ML/DL counters update
- Robust, layered imports for src/ services (factories/solvers/classifiers/etc.)
"""

# =============================================================================
# Imports & setup
# =============================================================================
import os
import sys
import io
import json
import zipfile
import pickle
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import sympy as sp
from sympy import Symbol, Function, Derivative
from sympy.functions.combinatorial.numbers import stirling

# DO NOT import AppliedUndef (not needed, causes ImportError with some sympy builds)

# Torch is optional (ML/DL pages will check availability)
try:
    import torch
except Exception:
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

APP_TITLE = "Master Generators ODE System — Patched Edition"
APP_ICON = "🔬"

# Make sure `src/` is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# =============================================================================
# Layered imports from src/ with safe fallbacks/aliases
# =============================================================================
HAVE_SRC = True
try:
    # Core (master generator + theorem utilities)
    try:
        from src.generators.master_generator import (
            MasterGenerator,
            EnhancedMasterGenerator,
            CompleteMasterGenerator,
        )
    except Exception as e:
        MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
        logger.warning(f"master_generator import warning: {e}")

    # Linear factories
    LinearGeneratorFactory = None
    CompleteLinearGeneratorFactory = None
    try:
        from src.generators.linear_generators import (
            LinearGeneratorFactory as _LGF,
            CompleteLinearGeneratorFactory as _CLGF,
        )
        LinearGeneratorFactory = _LGF
        CompleteLinearGeneratorFactory = _CLGF
    except Exception:
        # Some repos put CompleteLinearGeneratorFactory under master_generator
        try:
            from src.generators.master_generator import (
                LinearGeneratorFactory as _LGF2,
                CompleteLinearGeneratorFactory as _CLGF2,
            )
            LinearGeneratorFactory = _LGF2
            CompleteLinearGeneratorFactory = _CLGF2
        except Exception as e:
            logger.warning(f"linear factories import warning: {e}")

    # Nonlinear factories
    NonlinearGeneratorFactory = None
    CompleteNonlinearGeneratorFactory = None
    try:
        from src.generators.nonlinear_generators import (
            NonlinearGeneratorFactory as _NLF,
            CompleteNonlinearGeneratorFactory as _CNLF,
        )
        NonlinearGeneratorFactory = _NLF
        CompleteNonlinearGeneratorFactory = _CNLF
    except Exception as e:
        logger.warning(f"nonlinear factories import warning: {e}")

    # Constructor & types
    try:
        from src.generators.generator_constructor import (
            GeneratorConstructor,
            GeneratorSpecification,
            DerivativeTerm,
            DerivativeType,
            OperatorType,
        )
    except Exception as e:
        GeneratorConstructor = GeneratorSpecification = None
        DerivativeTerm = DerivativeType = OperatorType = None
        logger.warning(f"generator_constructor import warning: {e}")

    # Master theorem solver from src (we also provide symbolic fallbacks below)
    try:
        from src.generators.master_theorem import (
            MasterTheoremSolver,
            MasterTheoremParameters,
            ExtendedMasterTheorem,
        )
    except Exception as e:
        MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None
        logger.warning(f"master_theorem import warning: {e}")

    # Classifier
    try:
        from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
    except Exception as e:
        ODEClassifier = PhysicalApplication = None
        logger.warning(f"ode_classifier import warning: {e}")

    # Functions libraries
    try:
        from src.functions.basic_functions import BasicFunctions
    except Exception as e:
        BasicFunctions = None
        logger.warning(f"basic_functions import warning: {e}")

    try:
        from src.functions.special_functions import SpecialFunctions
    except Exception as e:
        SpecialFunctions = None
        logger.warning(f"special_functions import warning: {e}")

    # ML/DL
    try:
        from src.ml.pattern_learner import (
            GeneratorPatternLearner,
            GeneratorVAE,
            GeneratorTransformer,
            create_model,
        )
    except Exception as e:
        GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = create_model = None
        logger.warning(f"pattern_learner import warning: {e}")

    try:
        from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
    except Exception as e:
        MLTrainer = ODEDataset = ODEDataGenerator = None
        logger.warning(f"ml.trainer import warning: {e}")

    try:
        from src.ml.generator_learner import (
            GeneratorPattern,
            GeneratorPatternNetwork,
            GeneratorLearningSystem,
        )
    except Exception as e:
        GeneratorPattern = GeneratorPatternNetwork = GeneratorLearningSystem = None
        logger.warning(f"generator_learner import warning: {e}")

    try:
        from src.dl.novelty_detector import (
            ODENoveltyDetector,
            NoveltyAnalysis,
            ODETokenizer,
            ODETransformer,
        )
    except Exception as e:
        ODENoveltyDetector = NoveltyAnalysis = ODETokenizer = ODETransformer = None
        logger.warning(f"novelty_detector import warning: {e}")

    # Utils
    try:
        from src.utils.config import Settings, AppConfig
    except Exception:
        Settings = AppConfig = None

    try:
        from src.utils.cache import CacheManager, cached
    except Exception:
        CacheManager = None
        def cached(*args, **kwargs):  # no-op decorator fallback
            def _wrap(f):
                return f
            return _wrap

    try:
        from src.utils.validators import ParameterValidator
    except Exception:
        ParameterValidator = None

    try:
        from src.ui.components import UIComponents
    except Exception:
        UIComponents = None

except Exception as e:
    HAVE_SRC = False
    logger.error("Failed to import services from src/: %s", e)

# =============================================================================
# Streamlit page config & CSS
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 2.2rem; border-radius: 14px; margin-bottom: 1.4rem;
  color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: .3rem; }
.subtitle { font-size: 1.05rem; opacity: .95; }

.metric-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white; padding: 1rem; border-radius: 12px; text-align:center;
  box-shadow: 0 10px 20px rgba(0,0,0,0.2); transition: transform .2s ease;
}
.metric-card:hover { transform: scale(1.03); }

.generator-term {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  padding: .8rem; border-radius: 9px; margin: .5rem 0;
  border-left: 5px solid #667eea; box-shadow: 0 3px 10px rgba(0,0,0,.08);
}

.result-box {
  background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
  border: 2px solid #4caf50; padding: 1rem; border-radius: 12px; margin: 1rem 0;
  box-shadow: 0 5px 15px rgba(76,175,80,.2);
}
.success-animation { animation: successPulse .45s ease-in-out; }
@keyframes successPulse { 0%{transform:scale(1);}50%{transform:scale(1.04);}100%{transform:scale(1);} }

.info-box {
  background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
  border-left: 5px solid #2196f3; padding: .8rem; border-radius: 9px; margin: .7rem 0;
}
.latex-export-box {
  background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
  border: 2px solid #9c27b0; padding: .8rem; border-radius: 10px; margin: .9rem 0;
  box-shadow: 0 5px 15px rgba(156,39,176,.18);
}
.error-box {
  background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
  border: 2px solid #f44336; padding: .9rem; border-radius: 10px; margin: .7rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# LaTeX exporter
# =============================================================================
class LaTeXExporter:
    """Publication-ready LaTeX export for a single ODE result dict."""

    @staticmethod
    def _sx(expr) -> str:
        if expr is None:
            return ""
        try:
            # Keep expressions exact (avoid auto-eval to floats)
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
            r"\end{align}",
            "",
        ]

        if initial_conditions:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            keys = list(initial_conditions.keys())
            for i, k in enumerate(keys):
                sep = r" \\" if i < len(keys) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter._sx(initial_conditions[k])}{sep}")
            parts += [r"\end{align}", ""]

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
            parts += [r"\end{itemize}", ""]

        parts += [
            r"\subsection{Solution Verification}",
            r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$.",
        ]

        if include_preamble:
            parts.append(r"\end{document}")

        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        """ZIP: LaTeX + JSON (+ reproduction code)."""
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            latex_content = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
            zf.writestr("ode_document.tex", latex_content)
            zf.writestr("ode_data.json", json.dumps(ode_data, default=str, indent=2))
            readme = f"""Master Generator ODE Export
Generated: {datetime.now().isoformat()}

Files:
- ode_document.tex (LaTeX)
- ode_data.json    (raw data)
- reproduce.py     (code, if present)
"""
            zf.writestr("README.txt", readme)

            if include_extras:
                zf.writestr("reproduce.py", LaTeXExporter._repro_code(ode_data))

        zbuf.seek(0)
        return zbuf.getvalue()

    @staticmethod
    def _repro_code(ode_data: Dict[str, Any]) -> str:
        params = ode_data.get("parameters", {})
        gen_type = ode_data.get("type", "linear")
        func_name = ode_data.get("function_used", "unknown")
        gen_num = ode_data.get("generator_number", 1)

        return f'''# Reproduction snippet (requires the same src/ project)
import sympy as sp
try:
    from src.generators.linear_generators import LinearGeneratorFactory
except Exception:
    LinearGeneratorFactory = None
try:
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
except Exception:
    NonlinearGeneratorFactory = None
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions

z = sp.Symbol("z")
basic = BasicFunctions() if "{func_name}" else None
special = SpecialFunctions() if "{func_name}" else None

def get_f(name):
    if basic:
        try:
            return basic.get_function(name)
        except Exception:
            pass
    if special:
        try:
            return special.get_function(name)
        except Exception:
            pass
    # fallback
    return sp.exp(z) if name == "exponential" else z

f_z = get_f("{func_name}")
params = {json.dumps(params, default=str)}
factory = LinearGeneratorFactory() if "{gen_type}"=="linear" and LinearGeneratorFactory else NonlinearGeneratorFactory()
if factory:
    res = factory.create({gen_num}, f_z, **params)
    print("ODE:", res.get("ode"))
    print("Solution:", res.get("solution"))
'''

# =============================================================================
# Session state manager
# =============================================================================
def _ensure_ss_key(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

def set_lhs_source(source: str):
    assert source in {"constructor", "freeform"}
    st.session_state["lhs_source"] = source

def get_active_generator_spec():
    """Return active GeneratorSpecification; precedence: freeform -> constructor."""
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
            if str(t.get("wrap", "id")).lower() != "id":
                return "nonlinear"
            if int(t.get("power", 1)) != 1:
                return "nonlinear"
        return "linear"
    return "nonlinear"

def _infer_order_from_spec(spec) -> int:
    try:
        if hasattr(spec, "order"):
            return int(spec.order)
    except Exception:
        pass
    desc = getattr(spec, "freeform_descriptor", None)
    if isinstance(desc, dict) and "terms" in desc:
        try:
            return max(int(t.get("inner_order", 0)) for t in desc["terms"])
        except Exception:
            return 0
    return 0

def register_generated_ode(result: dict):
    """Normalize & append so ML/DL can see it immediately."""
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

# =============================================================================
# Symbolic math helpers — Theorems 4.1 & 4.2 (compact)
# =============================================================================
def theorem_4_1_solution_expr(f_of_z, alpha, beta, n: int, M, x: sp.Symbol):
    """
    y(x) = (π/(2n)) * Σ_{s=1}^n [ 2f(α+β) - ψ_s(x) - φ_s(x) ] + π M
    with ψ_s = f(α + β e^{ i x cos ω_s - x sin ω_s }), φ_s = f(α + β e^{ -i x cos ω_s - x sin ω_s })
    """
    z = sp.Symbol("z")
    ωs = [((2*s - 1) * sp.pi) / (2*n) for s in range(1, n+1)]

    def ψ(ω):
        return f_of_z.subs(z, alpha + beta * sp.exp(sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
    def φ(ω):
        return f_of_z.subs(z, alpha + beta * sp.exp(-sp.I * x * sp.cos(ω) - x * sp.sin(ω)))

    const = f_of_z.subs(z, alpha + beta)
    S = sum(2*const - ψ(ω) - φ(ω) for ω in ωs)
    return (sp.pi/(2*n)) * sp.simplify(S) + sp.pi * M

def theorem_4_2_mth_derivative_expr(f_of_z, alpha, beta, n: int, m: int, x: sp.Symbol):
    """
    y^(m)(x) in compact Stirling-number form.
    λ_s = exp(i(pi/2 + ω_s)), ζ_s(x) = exp(-x sin ω_s) * exp(i x cos ω_s)
    d^m/dx^m f(α+β ζ_s) = λ_s^m Σ_{j=1..m} S(m,j) (β ζ_s)^j ∂^j/∂α^j f(α+β ζ_s)
    Combine ψ/φ terms and multiply by -π/(2n).
    """
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

# =============================================================================
# Generic operator application L[y] for a symbolic LHS (works for free-form or constructor)
# =============================================================================
def _derivative_order_of(d: Derivative, x: sp.Symbol) -> int:
    """Robustly read order of Derivative(..., (x,k)) or (..., x, x, ...)."""
    k = 0
    for v in d.variables:
        if isinstance(v, sp.Symbol):
            if v == x:
                k += 1
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            if v[0] == x:
                k += int(v[1])
    return k

def max_derivative_order_in_lhs(lhs_expr: sp.Expr, y_func: Function, x: sp.Symbol) -> int:
    orders = []
    for d in lhs_expr.atoms(Derivative):
        try:
            if d.expr == y_func(x):
                orders.append(_derivative_order_of(d, x))
        except Exception:
            pass
    return max(orders) if orders else 0

def apply_lhs_to_solution(lhs_expr: sp.Expr, y_expr: sp.Expr, x: sp.Symbol, y_name="y") -> sp.Expr:
    """
    Apply a symbolic LHS operator (containing y(x) and its derivatives) to a given y_expr.
    Handles:
      - y(x), Derivative(y(x), x, k)
      - function arguments like y(g(x)) by substituting x->g(x) in y_expr
    """
    y = Function(y_name)
    mapping = {}

    # Map plain function calls y(some_arg)
    for fnode in lhs_expr.atoms(sp.Function):
        try:
            if fnode.func == y:
                arg = fnode.args[0] if fnode.args else x
                mapping[fnode] = y_expr.subs(x, arg)
        except Exception:
            pass

    # Map derivatives of y(...)
    for d in lhs_expr.atoms(Derivative):
        try:
            if isinstance(d.expr, sp.AppliedUndef):  # not available in some sympy versions
                pass
        except Exception:
            pass

        try:
            if hasattr(d, "expr") and d.expr.func == y:
                # If derivative is of y(g(x)), apply chain rule by substitution first
                inner_arg = d.expr.args[0] if d.expr.args else x
                k = _derivative_order_of(d, x)
                # Compute derivative w.r.t. x of y_expr(x) composed with inner_arg
                composed = y_expr.subs(x, inner_arg)
                mapping[d] = sp.diff(composed, (x, k))
        except Exception:
            pass

    # Also map base y(x) if not already
    mapping.setdefault(y(x), y_expr)

    try:
        return sp.simplify(lhs_expr.xreplace(mapping))
    except Exception:
        return lhs_expr.subs(mapping)

# =============================================================================
# Function library resolver
# =============================================================================
def get_function_expr(source: str, name: str) -> sp.Expr:
    """
    Resolve f(z) as a SymPy expression given name and source library.
    Falls back to a small builtin palette if src.functions.* is missing.
    """
    z = sp.Symbol("z")

    # Try src libraries first
    try:
        if source == "Basic" and st.session_state.get("basic_functions"):
            obj = st.session_state.basic_functions.get_function(name)
            # obj can be sympy expr in z or a callable; try calling with z if needed
            try:
                return sp.sympify(obj) if isinstance(obj, (sp.Expr,)) else obj(z)
            except Exception:
                pass
        if source == "Special" and st.session_state.get("special_functions"):
            obj = st.session_state.special_functions.get_function(name)
            try:
                return sp.sympify(obj) if isinstance(obj, (sp.Expr,)) else obj(z)
            except Exception:
                pass
    except Exception:
        pass

    # Built-in safe fallback
    palette = {
        "linear": z,
        "quadratic": z**2,
        "cubic": z**3,
        "exponential": sp.exp(z),
        "log": sp.log(z),
        "sin": sp.sin(z),
        "cos": sp.cos(z),
        "tanh": sp.tanh(z),
    }
    return palette.get(name, sp.exp(z))

# =============================================================================
# Free-form LHS builder utilities
# =============================================================================
WRAPPERS = {
    "id": lambda expr: expr,
    "sin": sp.sin,
    "cos": sp.cos,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "logabs": lambda expr: sp.log(sp.Symbol("epsilon", positive=True) + sp.Abs(expr)),
}

def build_freeform_lhs_expr(terms: List[Dict[str, Any]], x: sp.Symbol, yname: str = "y") -> sp.Expr:
    """
    Sum_i coef_i * wrap_i( D^{k_i} y(x) ) ** power_i
    Each term dict: {coef, inner_order, wrap, power}
    """
    y = Function(yname)
    s = 0
    for t in terms:
        coef = sp.nsimplify(t.get("coef", 1), rational=True)
        k = int(t.get("inner_order", 0))
        wrap = str(t.get("wrap", "id")).lower()
        power = int(t.get("power", 1))
        inner = y(x) if k == 0 else Derivative(y(x), (x, k))
        wfun = WRAPPERS.get(wrap, WRAPPERS["id"])
        term = coef * (wfun(inner))**power
        s += term
    return sp.simplify(s)

# =============================================================================
# UI Pages
# =============================================================================
def page_header():
    st.markdown(
        f"""
    <div class="main-header">
      <div class="main-title">{APP_ICON} {APP_TITLE}</div>
      <div class="subtitle">Theorems 4.1 & 4.2 · Free‑form LHS · ML/DL · Exports</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def page_dashboard():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h4>Generated ODEs</h4><h2>{len(st.session_state.generated_odes)}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4>ML Patterns</h4><h2>{len(st.session_state.generator_patterns)}</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4>Batch Results</h4><h2>{len(st.session_state.batch_results)}</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(
            f'<div class="metric-card"><h4>ML Trainer</h4><h3>{"✅ Trained" if st.session_state.ml_trained else "⏳ Not Trained"}</h3></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Recent ODEs")
    if st.session_state.generated_odes:
        df = pd.DataFrame(
            [
                {
                    "No": r.get("generator_number"),
                    "Type": r.get("type"),
                    "Order": r.get("order"),
                    "Func": r.get("function_used"),
                    "Timestamp": r.get("timestamp", "")[:19],
                }
                for r in st.session_state.generated_odes[-10:]
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No ODEs yet. Try the **Generator Constructor** and **Apply Master Theorem** tabs.")

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    st.markdown(
        """<div class="info-box">
        Build generators either with your project's Constructor (terms) or with the Free‑form builder below.
        </div>""",
        unsafe_allow_html=True,
    )

    # 1) Classic constructor from src (if available)
    with st.expander("➕ Term-based Constructor (from src)", expanded=False):
        if not st.session_state.generator_constructor or not DerivativeTerm:
            st.warning("The src-based constructor is not available. Use the Free‑form builder.")
        else:
            cols = st.columns(4)
            with cols[0]:
                deriv_order = st.number_input("Derivative order k", 0, 10, 1)
            with cols[1]:
                coef = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
            with cols[2]:
                power = st.number_input("Power", 1, 10, 1)
            with cols[3]:
                # Provide a small menu for function type/operator if enums exist
                ftype = st.selectbox(
                    "Function Type",
                    [t.value for t in DerivativeType] if DerivativeType else ["standard"],
                )
            cols2 = st.columns(3)
            with cols2[0]:
                op_type = st.selectbox(
                    "Operator",
                    [t.value for t in OperatorType] if OperatorType else ["identity"],
                )
            with cols2[1]:
                scaling = st.number_input("Scaling a (for delay/advance)", 0.1, 10.0, 1.0, 0.1)
            with cols2[2]:
                shift = st.number_input("Shift", -10.0, 10.0, 0.0, 0.1)

            if st.button("Add Term (src constructor)", use_container_width=True):
                try:
                    term = DerivativeTerm(
                        derivative_order=int(deriv_order),
                        coefficient=float(coef),
                        power=int(power),
                        function_type=DerivativeType(ftype) if DerivativeType else ftype,
                        operator_type=OperatorType(op_type) if OperatorType else op_type,
                        scaling=float(scaling),
                        shift=float(shift),
                    )
                    # Some repos had no add_term(); try both patterns
                    ctor = st.session_state.generator_constructor
                    if hasattr(ctor, "add_term"):
                        ctor.add_term(term)
                    elif hasattr(ctor, "terms"):
                        ctor.terms.append(term)
                    else:
                        st.warning("GeneratorConstructor has no add_term/terms. Term not stored.")
                    st.session_state.generator_terms.append(term)
                    st.success("Term added.")
                except Exception as e:
                    st.error(f"Failed to add term: {e}")

            if st.button("🔨 Build Generator Specification (src)", type="primary", use_container_width=True):
                try:
                    gen_spec = GeneratorSpecification(
                        terms=getattr(st.session_state.generator_constructor, "terms", st.session_state.generator_terms),
                        name=f"Custom Generator #{len(st.session_state.generated_odes) + 1}",
                    )
                    st.session_state["current_generator"] = gen_spec
                    st.session_state["lhs_source"] = "constructor"
                    st.success("✅ Generator Specification built and selected (Constructor).")
                    try:
                        st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Failed to build specification: {e}")

    # 2) Free-form builder (powerful + general)
    with st.expander("🧪 Free‑form LHS Builder (advanced)", expanded=True):
        st.markdown(
            "Add terms of the form:  **coef · wrap( y^{(k)}(x) )^{power}**  "
            "with wrappers: `id`, `sin`, `cos`, `sinh`, `cosh`, `tanh`, `exp`, `logabs`.",
        )
        cf = st.columns(5)
        with cf[0]:
            ff_coef = st.text_input("Coefficient", "1")
        with cf[1]:
            ff_k = st.number_input("Derivative order k", 0, 12, 1)
        with cf[2]:
            ff_wrap = st.selectbox("Wrapper", list(WRAPPERS.keys()), index=0)
        with cf[3]:
            ff_pow = st.number_input("Power", 1, 12, 1)
        with cf[4]:
            if st.button("➕ Add free‑form term", use_container_width=True):
                try:
                    t = {
                        "coef": sp.nsimplify(ff_coef, rational=True),
                        "inner_order": int(ff_k),
                        "wrap": ff_wrap,
                        "power": int(ff_pow),
                    }
                    st.session_state.freeform_terms.append(t)
                    st.success(f"Added term: {t}")
                except Exception as e:
                    st.error(f"Failed to add free‑form term: {e}")

        # Current terms
        if st.session_state.freeform_terms:
            st.write("**Current free‑form terms**")
            for i, t in enumerate(st.session_state.freeform_terms, 1):
                st.markdown(
                    f"<div class='generator-term'><b>Term {i}:</b> "
                    f"{t['coef']} · {t['wrap']}( y^{t['inner_order']} )^{t['power']}</div>",
                    unsafe_allow_html=True,
                )
            if st.button("🗑️ Clear free‑form terms", use_container_width=True):
                st.session_state.freeform_terms = []
                st.session_state.freeform_gen_spec = None
                st.info("Cleared.")

        # Build + store Free‑form spec
        if st.button("🔨 Build Free‑form LHS and select it", type="primary", use_container_width=True):
            try:
                x = sp.Symbol("x", real=True)
                lhs_expr = build_freeform_lhs_expr(st.session_state.freeform_terms, x, yname="y")
                # Wrap in a light spec object so the rest of the app can consume it
                if GeneratorSpecification:
                    ff_spec = GeneratorSpecification(terms=[], name=f"Free-form Generator {datetime.now().strftime('%H%M%S')}")
                    ff_spec.lhs = lhs_expr
                    ff_spec.freeform_descriptor = {"terms": st.session_state.freeform_terms, "note": "free-form"}
                else:
                    # Minimal shim
                    class _Spec: ...
                    ff_spec = _Spec()
                    ff_spec.lhs = lhs_expr
                    ff_spec.freeform_descriptor = {"terms": st.session_state.freeform_terms, "note": "free-form"}

                st.session_state["freeform_gen_spec"] = ff_spec
                set_lhs_source("freeform")
                st.success("✅ Free‑form LHS stored and selected. It will be used when generating ODEs.")
                try:
                    st.latex(sp.latex(lhs_expr))
                except Exception:
                    st.write(lhs_expr)
            except Exception as e:
                st.error(f"Failed to build free‑form LHS: {e}")

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem")
    st.markdown(
        "<div class='info-box'>Select the LHS source, function f(z), parameters "
        "and generate an ODE with an exact solution (Thm 4.1). You can also compute "
        "general derivatives via Thm 4.2.</div>",
        unsafe_allow_html=True,
    )

    # LHS source selector (Constructor vs Free‑form)
    source_label = {"constructor": "Constructor LHS", "freeform": "Free‑form LHS"}
    sel = st.radio(
        "Generator LHS source",
        options=("constructor", "freeform"),
        index=0 if st.session_state.get("lhs_source", "constructor") == "constructor" else 1,
        format_func=lambda s: source_label[s],
        horizontal=True,
    )
    set_lhs_source(sel)

    # Function + parameters
    col1, col2 = st.columns([1, 1])
    with col1:
        source_lib = st.selectbox("Function library", ["Basic", "Special"], index=0)
        # Get available function names if possible
        names = []
        try:
            if source_lib == "Basic" and st.session_state.basic_functions:
                names = st.session_state.basic_functions.get_function_names()
            elif source_lib == "Special" and st.session_state.special_functions:
                names = st.session_state.special_functions.get_function_names()
        except Exception:
            pass
        if not names:
            names = ["exponential", "linear", "quadratic", "sin", "cos", "log", "tanh"]
        func_name = st.selectbox("Select f(z)", names, index=0)

        simplify_out = st.checkbox("Simplify outputs", value=True)

    with col2:
        st.markdown("**Theorem parameters**")
        alpha = st.text_input("α", "1")
        beta = st.text_input("β", "1")
        n = st.number_input("n (integer ≥1)", 1, 20, 1)
        M = st.text_input("M", "0")
        use_exact = st.checkbox("Exact (symbolic) parameters", value=True)

    def to_exact(v):
        try:
            return sp.nsimplify(v, rational=True)
        except Exception:
            return sp.sympify(v)

    α = to_exact(alpha) if use_exact else sp.Float(alpha)
    β = to_exact(beta) if use_exact else sp.Float(beta)
    𝑀 = to_exact(M) if use_exact else sp.Float(M)

    x = sp.Symbol("x", real=True)

    # Thm 4.2 controls
    with st.expander("Optional: Theorem 4.2 (m‑th derivative)", expanded=False):
        m = st.number_input("Order m (≥1)", 1, 20, 1)
        if st.button("Compute y^{(m)}(x) (Thm 4.2)"):
            try:
                f_expr = get_function_expr(source_lib, func_name)
                y_m = theorem_4_2_mth_derivative_expr(f_expr, α, β, int(n), int(m), x)
                if simplify_out:
                    y_m = sp.simplify(y_m)
                st.latex("y^{(" + str(int(m)) + ")}(x) = " + sp.latex(y_m))
            except Exception as e:
                st.error(f"Failed to compute m‑th derivative: {e}")

    # GENERATE ODE (Thm 4.1) + Export
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        with st.spinner("Applying Master Theorem and constructing RHS…"):
            try:
                # Resolve f(z)
                f_expr = get_function_expr(source_lib, func_name)

                # Build y(x) via Thm 4.1
                yx = theorem_4_1_solution_expr(f_expr, α, β, int(n), 𝑀, x)
                if simplify_out:
                    yx = sp.simplify(yx)

                # Get active LHS
                active_spec = get_active_generator_spec()
                if active_spec and hasattr(active_spec, "lhs") and active_spec.lhs is not None:
                    try:
                        rhs = apply_lhs_to_solution(active_spec.lhs, yx, x, y_name="y")
                        if simplify_out:
                            rhs = sp.simplify(rhs)
                        generator_lhs = active_spec.lhs
                    except Exception as e:
                        logger.warning(f"Failed to apply LHS to y(x); using canonical RHS. Reason: {e}")
                        z = sp.Symbol("z")
                        rhs = sp.simplify(sp.pi * (f_expr.subs(z, α + β) + 𝑀))
                        generator_lhs = sp.Symbol("L[y]")
                else:
                    z = sp.Symbol("z")
                    rhs = sp.simplify(sp.pi * (f_expr.subs(z, α + β) + 𝑀))
                    generator_lhs = sp.Symbol("L[y]")

                # Infer classification/order
                if active_spec is not None:
                    ode_type = _infer_type_from_spec(active_spec)
                    ode_order = _infer_order_from_spec(active_spec)
                else:
                    ode_type, ode_order = "nonlinear", 0

                result = {
                    "generator": generator_lhs,
                    "rhs": rhs,
                    "solution": yx,
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
                    "initial_conditions": {},
                    "timestamp": datetime.now().isoformat(),
                    "lhs_source": st.session_state.get("lhs_source", "constructor"),
                    "title": "Master Generators ODE System",
                }

                register_generated_ode(result)

                st.markdown(
                    "<div class='result-box success-animation'><h4>✅ ODE Generated Successfully!</h4></div>",
                    unsafe_allow_html=True,
                )
                t1, t2, t3 = st.tabs(["📐 Equation", "💡 Solution", "📤 Export"])

                with t1:
                    try:
                        st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                    except Exception:
                        st.write("LHS =", generator_lhs)
                        st.write("RHS =", rhs)
                    st.caption(f"LHS source: **{st.session_state.get('lhs_source','constructor')}**")

                with t2:
                    st.latex("y(x) = " + sp.latex(yx))
                    st.markdown("**Parameters:**")
                    st.write(f"α = {α}, β = {β}, n = {int(n)}, M = {𝑀}")
                    st.write(f"**f(z):** {f_expr}")

                with t3:
                    try:
                        latex_doc = LaTeXExporter.generate_latex_document(
                            {
                                "generator": generator_lhs,
                                "rhs": rhs,
                                "solution": yx,
                                "parameters": {"alpha": α, "beta": β, "n": int(n), "M": 𝑀},
                                "classification": result["classification"],
                                "initial_conditions": {},
                                "function_used": str(func_name),
                                "generator_number": len(st.session_state.generated_odes),
                                "type": ode_type,
                                "order": ode_order,
                                "title": result.get("title", "Master Generators ODE System"),
                            },
                            include_preamble=True,
                        )
                        st.download_button(
                            "📄 Download LaTeX Document",
                            latex_doc,
                            file_name=f"ode_{len(st.session_state.generated_odes)}.tex",
                            mime="text/x-latex",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"LaTeX export failed: {e}")

                    try:
                        pkg = LaTeXExporter.create_export_package(
                            {
                                "generator": generator_lhs,
                                "rhs": rhs,
                                "solution": yx,
                                "parameters": {"alpha": α, "beta": β, "n": int(n), "M": 𝑀},
                                "classification": result["classification"],
                                "initial_conditions": {},
                                "function_used": str(func_name),
                                "generator_number": len(st.session_state.generated_odes),
                                "type": ode_type,
                                "order": ode_order,
                                "title": result.get("title", "Master Generators ODE System"),
                            },
                            include_extras=True,
                        )
                        st.download_button(
                            "📦 Download Complete Package (ZIP)",
                            pkg,
                            file_name=f"ode_package_{len(st.session_state.generated_odes)}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"ZIP export failed: {e}")

            except Exception as e:
                logger.error("Generation error", exc_info=True)
                st.error(f"Error generating ODE: {e}")

def page_ml_pattern_learning():
    st.header("🤖 ML Pattern Learning")
    if MLTrainer is None:
        st.info("ML trainer not available in this environment.")
        return

    col = st.columns(4)
    with col[0]:
        st.metric("Patterns", len(st.session_state.generator_patterns))
    with col[1]:
        st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with col[2]:
        st.metric("Training Epochs", len(st.session_state.training_history.get("train_loss", [])))
    with col[3]:
        st.metric("Model Status", "Trained" if st.session_state.ml_trained else "Not Trained")

    model_type = st.selectbox(
        "Select ML Model",
        ["pattern_learner", "vae", "transformer"],
        index=0,
        format_func=lambda x: {
            "pattern_learner": "Pattern Learner (Encoder‑Decoder)",
            "vae": "Variational Autoencoder",
            "transformer": "Transformer",
        }[x],
    )

    with st.expander("Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 5, 500, 50)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            samples = st.slider("Training Samples", 100, 5000, 500)
        with c3:
            val_split = st.slider("Validation Split", 0.05, 0.3, 0.2)
            use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate at least 5 ODEs before training.")
        return

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training..."):
            try:
                device = "cuda" if (use_gpu and torch and torch.cuda.is_available()) else "cpu"
                trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
                st.session_state.ml_trainer = trainer

                # simple progress
                progress = st.progress(0.0)
                info = st.empty()

                def cb(ep, total):
                    p = ep / float(total)
                    progress.progress(min(max(p, 0.0), 1.0))
                    info.text(f"Epoch {ep}/{total}")

                trainer.train(
                    epochs=epochs,
                    batch_size=batch_size,
                    samples=samples,
                    validation_split=val_split,
                    progress_callback=cb,
                )
                st.session_state.ml_trained = True
                st.session_state.training_history = trainer.history

                st.success("✅ Training complete.")
                if trainer.history["train_loss"]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=trainer.history["train_loss"], mode="lines", name="train"))
                    if trainer.history["val_loss"]:
                        fig.add_trace(go.Scatter(y=trainer.history["val_loss"], mode="lines", name="val"))
                    fig.update_layout(title="Training History", xaxis_title="epoch", yaxis_title="loss", height=360)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.ml_trained and st.session_state.ml_trainer:
        st.subheader("Generate novel ODEs")
        num = st.slider("How many", 1, 10, 1)
        if st.button("🎲 Generate", use_container_width=True):
            with st.spinner("Generating..."):
                for i in range(num):
                    try:
                        res = st.session_state.ml_trainer.generate_new_ode()
                        if res:
                            register_generated_ode(res)
                            st.success(f"Generated ODE #{res.get('generator_number')}")
                    except Exception as e:
                        st.warning(f"One generation failed: {e}")

def page_batch_generation():
    st.header("📊 Batch ODE Generation")
    st.markdown("<div class='info-box'>Generate many ODEs with randomized params and export the results.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 1000, 25)
        gen_types = st.multiselect("Generator types", ["linear", "nonlinear"], default=["linear", "nonlinear"])
    with c2:
        func_cats = st.multiselect("Function libraries", ["Basic", "Special"], default=["Basic"])
        vary_params = st.checkbox("Vary parameters", True)
    with c3:
        if vary_params:
            a_rng = st.slider("α range", -5.0, 5.0, (-2.0, 2.0))
            b_rng = st.slider("β range", 0.1, 5.0, (0.5, 2.0))
            n_rng = st.slider("n range", 1, 5, (1, 3))
        else:
            a_rng = (1.0, 1.0)
            b_rng = (1.0, 1.0)
            n_rng = (1, 1)

    with st.expander("Advanced", expanded=False):
        export_format = st.selectbox("Export format", ["JSON", "CSV", "LaTeX", "All"], index=1)
        include_solutions = st.checkbox("Include solution preview", True)

    if st.button("🚀 Generate Batch", type="primary", use_container_width=True):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            results = []
            # build function names
            all_names = []
            if "Basic" in func_cats and st.session_state.basic_functions:
                all_names += st.session_state.basic_functions.get_function_names()
            if "Special" in func_cats and st.session_state.special_functions:
                # take a small slice to avoid huge heavy special families
                all_names += st.session_state.special_functions.get_function_names()[:20]
            if not all_names:
                all_names = ["exponential", "sin", "cos", "quadratic", "log", "tanh"]

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
                    gtype = np.random.choice(gen_types)
                    fname = np.random.choice(all_names)
                    f_expr = get_function_expr("Basic" if fname in all_names else "Special", fname)

                    if gtype == "linear" and CompleteLinearGeneratorFactory:
                        factory = CompleteLinearGeneratorFactory()
                        gnum = int(np.random.randint(1, 9))
                        if gnum in [4, 5]:
                            params["a"] = float(np.random.uniform(1, 3))
                        result = factory.create(gnum, f_expr, **params)
                    elif gtype == "nonlinear" and CompleteNonlinearGeneratorFactory:
                        factory = CompleteNonlinearGeneratorFactory()
                        gnum = int(np.random.randint(1, 11))
                        if gnum in [1, 2, 4]:
                            params["q"] = int(np.random.randint(2, 6))
                        if gnum in [2, 3, 5]:
                            params["v"] = int(np.random.randint(2, 6))
                        if gnum in [4, 5, 9, 10]:
                            params["a"] = float(np.random.uniform(1, 3))
                        result = factory.create(gnum, f_expr, **params)
                    else:
                        # Fallback via Thm 4.1
                        x = sp.Symbol("x", real=True)
                        yx = theorem_4_1_solution_expr(f_expr, sp.nsimplify(params["alpha"]), sp.nsimplify(params["beta"]), int(params["n"]), sp.nsimplify(params["M"]), x)
                        lhs = sp.Function("L")(x)  # dummy L
                        rhs = sp.pi * (f_expr.subs(sp.Symbol("z"), params["alpha"] + params["beta"])) + sp.pi * params["M"]
                        result = {
                            "ode": sp.Eq(lhs, rhs),
                            "solution": yx,
                            "type": gtype,
                            "order": 0,
                            "generator_number": i+1,
                            "function_used": fname,
                            "subtype": "fallback",
                        }

                    rec = {
                        "ID": i+1,
                        "Type": result.get("type"),
                        "Generator": result.get("generator_number"),
                        "Function": fname,
                        "Order": result.get("order", 0),
                        "α": round(params["alpha"], 3),
                        "β": round(params["beta"], 3),
                        "n": params["n"],
                    }
                    if include_solutions:
                        rec["Solution"] = str(result.get("solution"))[:120] + "..."
                    results.append(rec)
                except Exception as e:
                    logger.debug(f"Failed to generate ODE {i+1}: {e}")

            st.session_state.batch_results.extend(results)
            st.success(f"Generated {len(results)} ODEs")
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
                latextable = "\\begin{tabular}{|c|c|c|c|c|}\\hline ID & Type & Gen & Function & Order \\\\ \\hline\n" + \
                             "\n".join([f"{r['ID']} & {r['Type']} & {r['Generator']} & {r['Function']} & {r['Order']} \\\\" for r in results[:30]]) + \
                             "\n\\hline\\end{tabular}"
                st.download_button("📝 LaTeX table", latextable, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")
            with c4:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("batch.csv", df.to_csv(index=False))
                    zf.writestr("batch.json", json.dumps(results, indent=2))
                    zf.writestr("batch.tex", latextable)
                    zf.writestr("README.txt", f"Batch generated {datetime.now().isoformat()}")
                zbuf.seek(0)
                st.download_button("📦 ZIP (all)", zbuf.getvalue(), file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

def page_novelty_detection():
    st.header("🔍 Novelty Detection")
    if ODENoveltyDetector is None:
        st.info("Novelty detector not available in this environment.")
        return
    det = ODENoveltyDetector()
    mode = st.radio("Input", ["Use current LHS", "Enter ODE text"], horizontal=True)
    ode_obj = None
    if mode == "Use current LHS":
        spec = get_active_generator_spec()
        if not spec:
            st.warning("No active generator. Build one first.")
            return
        ode_obj = {"ode": spec.lhs, "type": _infer_type_from_spec(spec), "order": _infer_order_from_spec(spec)}
    else:
        txt = st.text_area("Paste an ODE (LaTeX or plain)")
        if txt.strip():
            ode_obj = {"ode": txt, "type": "manual", "order": st.number_input("Order", 1, 10, 2)}

    if ode_obj and st.button("Analyze novelty"):
        try:
            res = det.analyze(ode_obj, check_solvability=True, detailed=True)
            st.metric("Novelty Score", f"{res.novelty_score:.1f}/100")
            st.metric("Confidence", f"{res.confidence:.1%}")
            st.write("**Complexity:**", res.complexity_level)
            if res.special_characteristics:
                st.write("**Special traits:**", ", ".join(res.special_characteristics))
            if res.recommended_methods:
                st.write("**Suggested methods:**", ", ".join(res.recommended_methods[:5]))
        except Exception as e:
            st.error(f"Analysis failed: {e}")

def page_analysis_classification():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("Generate some ODEs first.")
        return
    df = pd.DataFrame(
        [
            {
                "No": r.get("generator_number"),
                "Type": r.get("type"),
                "Order": r.get("order"),
                "Function": r.get("function_used"),
                "Timestamp": r.get("timestamp", "")[:19],
            }
            for r in st.session_state.generated_odes
        ]
    )
    st.dataframe(df, use_container_width=True)

    types = pd.Series([r.get("type", "unknown") for r in st.session_state.generated_odes]).value_counts()
    fig = px.pie(values=types.values, names=types.index, title="Type distribution")
    st.plotly_chart(fig, use_container_width=True)

def page_export_latex():
    st.header("📤 Export & LaTeX")
    st.markdown("<div class='latex-export-box'><b>Export your ODEs in LaTeX or as ZIP packages.</b></div>", unsafe_allow_html=True)
    if not st.session_state.generated_odes:
        st.info("No ODEs to export.")
        return

    idx = st.selectbox(
        "Select ODE",
        options=list(range(1, len(st.session_state.generated_odes) + 1)),
        format_func=lambda i: f"ODE #{i}: {st.session_state.generated_odes[i-1].get('type')} (order {st.session_state.generated_odes[i-1].get('order')})",
    )
    ode = st.session_state.generated_odes[idx - 1]

    # Preview (no preamble)
    st.subheader("Preview")
    try:
        preview = LaTeXExporter.generate_latex_document(ode, include_preamble=False)
        st.code(preview, language="latex")
    except Exception:
        st.write(ode)

    c1, c2 = st.columns(2)
    with c1:
        try:
            full_latex = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
            st.download_button("📄 Download LaTeX", full_latex, file_name=f"ode_{idx}.tex", mime="text/x-latex", use_container_width=True)
        except Exception as e:
            st.warning(f"LaTeX export failed: {e}")
    with c2:
        try:
            pkg = LaTeXExporter.create_export_package(ode, include_extras=True)
            st.download_button("📦 Download ZIP", pkg, file_name=f"ode_package_{idx}.zip", mime="application/zip", use_container_width=True)
        except Exception as e:
            st.warning(f"ZIP export failed: {e}")

def page_examples():
    st.header("📚 Examples Library")
    st.markdown("A few quick examples for inspiration.")
    ex = [
        {"name": "Harmonic oscillator", "lhs": sp.Function("y")(sp.Symbol("x")) + Derivative(sp.Function("y")(sp.Symbol("x")), (sp.Symbol("x"), 2))},
        {"name": "Damped oscillator", "lhs": Derivative(sp.Function("y")(sp.Symbol("x")), (sp.Symbol("x"), 2)) + 2*Derivative(sp.Function("y")(sp.Symbol("x")), sp.Symbol("x")) + sp.Function("y")(sp.Symbol("x"))},
        {"name": "Pantograph-like", "lhs": Derivative(sp.Function("y")(sp.Symbol("x")), (sp.Symbol("x"), 2)) + sp.Function("y")(sp.Symbol("x")/2) - sp.Function("y")(sp.Symbol("x"))},
    ]
    for i, e in enumerate(ex, 1):
        with st.expander(f"Example {i}: {e['name']}"):
            try:
                st.latex(sp.latex(e["lhs"]) + " = RHS")
            except Exception:
                st.write(e["lhs"])
            if st.button(f"Use as Free‑form LHS: {e['name']}", key=f"use_ex_{i}"):
                # Store as free-form spec
                class _Spec: ...
                spec = _Spec()
                spec.lhs = e["lhs"]
                spec.freeform_descriptor = {"terms": [], "note": "example"}
                st.session_state["freeform_gen_spec"] = spec
                set_lhs_source("freeform")
                st.success("Stored into Free‑form and selected.")

def page_settings():
    st.header("⚙️ Settings")
    st.write("General settings and debug tools.")
    if st.checkbox("Show session keys"):
        st.write(list(st.session_state.keys()))
    if st.button("Clear cache (if any)"):
        try:
            cm = st.session_state.get("cache_manager")
            if cm:
                cm.clear()
            st.success("Cache cleared.")
        except Exception as e:
            st.warning(f"Failed to clear cache: {e}")

def page_docs():
    st.header("📖 Documentation")
    st.markdown(
        """
**Theorem 4.1.**  
\\[
y(x)=\\frac{\\pi}{2n}\\sum_{s=1}^n\\Big(2f(\\alpha+\\beta)-\\psi_s(x)-\\phi_s(x)\\Big)+\\pi M
\\]
with \\(\\psi_s=f(\\alpha+\\beta e^{ix\\cos\\omega_s-x\\sin\\omega_s})\\), \\(\\phi_s=f(\\alpha+\\beta e^{-ix\\cos\\omega_s-x\\sin\\omega_s})\\).

**Theorem 4.2 (compact).**  
Using Stirling numbers \\(\\mathbf{S}(m,j)\\) (Faà di Bruno collapse for exponential inner map), the m‑th derivative is:
\\[
y^{(m)}(x)=-\\frac{\\pi}{2n}\\sum_{s=1}^n\\left\\{
\\lambda_s^m\\sum_{j=1}^m\\mathbf{S}(m,j)(\\beta\\zeta_s)^j\\partial_\\alpha^{\,j}\\psi
+\\overline{\\lambda_s}^{\,m}\\sum_{j=1}^m\\mathbf{S}(m,j)(\\beta\\overline{\\zeta_s})^j\\partial_\\alpha^{\,j}\\phi
\\right\\}.
\\]
Here \\(\\zeta_s = e^{-x\\sin\\omega_s}e^{ix\\cos\\omega_s}\\), \\(\\lambda_s=e^{i(\\pi/2+\\omega_s)}\\), \\(\\omega_s=(2s-1)\\pi/(2n)\\).
""",
        unsafe_allow_html=False,
    )

# =============================================================================
# Main
# =============================================================================
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
            "📤 Export & LaTeX",
            "📚 Examples",
            "⚙️ Settings",
            "📖 Documentation",
        ],
    )

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🔧 Generator Constructor":
        page_generator_constructor()
    elif page == "🎯 Apply Master Theorem":
        page_apply_master_theorem()
    elif page == "🤖 ML Pattern Learning":
        page_ml_pattern_learning()
    elif page == "📊 Batch Generation":
        page_batch_generation()
    elif page == "🔍 Novelty Detection":
        page_novelty_detection()
    elif page == "📈 Analysis & Classification":
        page_analysis_classification()
    elif page == "📤 Export & LaTeX":
        page_export_latex()
    elif page == "📚 Examples":
        page_examples()
    elif page == "⚙️ Settings":
        page_settings()
    elif page == "📖 Documentation":
        page_docs()

if __name__ == "__main__":
    main()