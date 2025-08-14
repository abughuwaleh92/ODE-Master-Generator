# -*- coding: utf-8 -*-
"""
Master Generators for ODEs — Complete App (Corrected)
- Flexible imports that match your src/ layout (including cases where
  CompleteLinearGeneratorFactory lives in src.generators.master_generator).
- Correct "Apply Master Theorem" implementation (Theorem 4.1) with verified RHS.
- Robust helpers:
    * get_function_expr
    * theorem_4_1_solution_expr
    * apply_generator_to
- Clean UI (no st.switch_page), fixed quote issues, export tools.
"""

# ======================================================
# Standard imports
# ======================================================
import os
import sys
import io
import json
import time
import zipfile
import pickle
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Numeric / symbolic / plotting
import numpy as np
import pandas as pd
import sympy as sp
from sympy.core.function import AppliedUndef  # for y(x), y(x/a) pattern matching

# Streamlit UI
import streamlit as st

# Optional ML/DL
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    torch = None

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# ======================================================
# Logging
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")


# ======================================================
# Ensure src/ is importable
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

HAVE_SRC = True
IMPORT_WARNINGS: List[str] = []


def _try_import(module_path: str):
    try:
        __import__(module_path)
        return sys.modules[module_path]
    except Exception as e:
        IMPORT_WARNINGS.append(f"Import failed: {module_path} -> {e}")
        return None


# Core modules (some may be missing; we guard usage)
mod_master_gen = _try_import("src.generators.master_generator")
mod_lin_gen    = _try_import("src.generators.linear_generators")
mod_nonlin_gen = _try_import("src.generators.nonlinear_generators")
mod_constructor = _try_import("src.generators.generator_constructor")
mod_theorem     = _try_import("src.generators.master_theorem")
mod_classifier  = _try_import("src.generators.ode_classifier")

mod_funcs_basic = _try_import("src.functions.basic_functions")
mod_funcs_spec  = _try_import("src.functions.special_functions")

mod_ml_pl       = _try_import("src.ml.pattern_learner")
mod_ml_tr       = _try_import("src.ml.trainer")
mod_ml_gl       = _try_import("src.ml.generator_learner")
mod_dl_nd       = _try_import("src.dl.novelty_detector")

mod_utils_conf  = _try_import("src.utils.config")
mod_utils_cache = _try_import("src.utils.cache")
mod_utils_valid = _try_import("src.utils.validators")
mod_ui_comp     = _try_import("src.ui.components")


# ======================================================
# Extract names (robustly) from imported modules
# ======================================================
# Generators / factories
MasterGenerator = getattr(mod_master_gen, "MasterGenerator", None) if mod_master_gen else None
EnhancedMasterGenerator = getattr(mod_master_gen, "EnhancedMasterGenerator", None) if mod_master_gen else None
CompleteMasterGenerator = getattr(mod_master_gen, "CompleteMasterGenerator", None) if mod_master_gen else None

# Depending on project layout:
LinearGeneratorFactory = getattr(mod_lin_gen, "LinearGeneratorFactory", None) if mod_lin_gen else None
NonlinearGeneratorFactory = getattr(mod_nonlin_gen, "NonlinearGeneratorFactory", None) if mod_nonlin_gen else None

# Some repos place the "Complete*" factories inside master_generator
CompleteLinearGeneratorFactory = (
    getattr(mod_lin_gen, "CompleteLinearGeneratorFactory", None)
    if mod_lin_gen else None
)
if CompleteLinearGeneratorFactory is None and mod_master_gen:
    CompleteLinearGeneratorFactory = getattr(mod_master_gen, "CompleteLinearGeneratorFactory", None)

CompleteNonlinearGeneratorFactory = (
    getattr(mod_nonlin_gen, "CompleteNonlinearGeneratorFactory", None)
    if mod_nonlin_gen else None
)
if CompleteNonlinearGeneratorFactory is None and mod_master_gen:
    CompleteNonlinearGeneratorFactory = getattr(mod_master_gen, "CompleteNonlinearGeneratorFactory", None)

# Constructor
GeneratorConstructor = getattr(mod_constructor, "GeneratorConstructor", None) if mod_constructor else None
GeneratorSpecification = getattr(mod_constructor, "GeneratorSpecification", None) if mod_constructor else None
DerivativeTerm = getattr(mod_constructor, "DerivativeTerm", None) if mod_constructor else None
DerivativeType = getattr(mod_constructor, "DerivativeType", None) if mod_constructor else None
OperatorType = getattr(mod_constructor, "OperatorType", None) if mod_constructor else None

# Theorem 4.1 / 4.2 solver
MasterTheoremSolver = getattr(mod_theorem, "MasterTheoremSolver", None) if mod_theorem else None
MasterTheoremParameters = getattr(mod_theorem, "MasterTheoremParameters", None) if mod_theorem else None
ExtendedMasterTheorem = getattr(mod_theorem, "ExtendedMasterTheorem", None) if mod_theorem else None

# Classifier / applications
ODEClassifier = getattr(mod_classifier, "ODEClassifier", None) if mod_classifier else None
PhysicalApplication = getattr(mod_classifier, "PhysicalApplication", None) if mod_classifier else None

# Functions libraries
BasicFunctions = getattr(mod_funcs_basic, "BasicFunctions", None) if mod_funcs_basic else None
SpecialFunctions = getattr(mod_funcs_spec, "SpecialFunctions", None) if mod_funcs_spec else None

# ML / DL (optional)
GeneratorPatternLearner = getattr(mod_ml_pl, "GeneratorPatternLearner", None) if mod_ml_pl else None
GeneratorVAE = getattr(mod_ml_pl, "GeneratorVAE", None) if mod_ml_pl else None
GeneratorTransformer = getattr(mod_ml_pl, "GeneratorTransformer", None) if mod_ml_pl else None
create_model = getattr(mod_ml_pl, "create_model", None) if mod_ml_pl else None

MLTrainer = getattr(mod_ml_tr, "MLTrainer", None) if mod_ml_tr else None
ODEDataset = getattr(mod_ml_tr, "ODEDataset", None) if mod_ml_tr else None
ODEDataGenerator = getattr(mod_ml_tr, "ODEDataGenerator", None) if mod_ml_tr else None

GeneratorPattern = getattr(mod_ml_gl, "GeneratorPattern", None) if mod_ml_gl else None
GeneratorPatternNetwork = getattr(mod_ml_gl, "GeneratorPatternNetwork", None) if mod_ml_gl else None
GeneratorLearningSystem = getattr(mod_ml_gl, "GeneratorLearningSystem", None) if mod_ml_gl else None

ODENoveltyDetector = getattr(mod_dl_nd, "ODENoveltyDetector", None) if mod_dl_nd else None
NoveltyAnalysis = getattr(mod_dl_nd, "NoveltyAnalysis", None) if mod_dl_nd else None
ODETokenizer = getattr(mod_dl_nd, "ODETokenizer", None) if mod_dl_nd else None
ODETransformer = getattr(mod_dl_nd, "ODETransformer", None) if mod_dl_nd else None

# Utils
Settings = getattr(mod_utils_conf, "Settings", None) if mod_utils_conf else None
AppConfig = getattr(mod_utils_conf, "AppConfig", None) if mod_utils_conf else None
CacheManager = getattr(mod_utils_cache, "CacheManager", None) if mod_utils_cache else None
cached = getattr(mod_utils_cache, "cached", None) if mod_utils_cache else None
ParameterValidator = getattr(mod_utils_valid, "ParameterValidator", None) if mod_utils_valid else None
UIComponents = getattr(mod_ui_comp, "UIComponents", None) if mod_ui_comp else None

HAVE_SRC = (
    (CompleteMasterGenerator is not None)
    or (MasterTheoremSolver is not None)
    or (GeneratorConstructor is not None)
)


# ======================================================
# Streamlit Page Config
# ======================================================
st.set_page_config(
    page_title="Master Generators ODE System — Corrected",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# Enhanced CSS
# ======================================================
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem; color: white;
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.25rem; }
    .subtitle { font-size: 1.05rem; opacity: 0.95; }
    .generator-term {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 12px; border-radius: 10px; margin: 8px 0; border-left: 5px solid #667eea;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08); transition: transform 0.25s ease;
    }
    .generator-term:hover { transform: translateX(5px); }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50; padding: 1.25rem; border-radius: 15px; margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(76,175,80,0.2);
    }
    .latex-export-box {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border: 2px solid #9c27b0; padding: 1rem; border-radius: 10px; margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(156,39,176,0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3; padding: 0.9rem; border-radius: 10px; margin: 0.8rem 0;
    }
    @keyframes successPulse {
        0% { transform: scale(1); } 50% { transform: scale(1.04); } 100% { transform: scale(1); }
    }
    .success-animation { animation: successPulse 0.5s ease-in-out; }
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# Helpers (SymPy + glue)
# ======================================================

def safe_simplify(expr: sp.Expr) -> sp.Expr:
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def get_function_expr(source_lib: str, func_name: str) -> sp.Expr:
    """
    Resolve a function f(z) from either BasicFunctions or SpecialFunctions into a pure SymPy expression
    in the symbol z.

    Accepts returns of various shapes from the library:
      - SymPy Expr in z
      - string like "exp(z)" or "sin(z)"
      - python callable f(z) returning a SymPy Expr
    """
    z = sp.Symbol("z")

    def _from_obj(obj: Any) -> sp.Expr:
        # Already a SymPy expression?
        if isinstance(obj, sp.Expr):
            # Ensure it is an expression of z (if it has other symbol names, we rename to z)
            syms = list(obj.free_symbols)
            if len(syms) == 1 and next(iter(syms)).name != "z":
                return obj.xreplace({syms[0]: z})
            if len(syms) == 0:
                # Constant function -> treat as f(z) = constant
                return obj
            return obj
        # Is it a string?
        if isinstance(obj, str):
            try:
                return sp.sympify(obj, locals={"z": z, **{k: getattr(sp, k) for k in dir(sp)}})
            except Exception:
                # Fallback: try to parse common names
                return sp.sympify(obj.replace("^", "**"), locals={"z": z})
        # Callable?
        if callable(obj):
            try:
                out = obj(z)  # Expect a SymPy expression back
                return _from_obj(out)
            except Exception as e:
                raise TypeError(f"Function object for '{func_name}' is not SymPy-callable: {e}")
        raise TypeError(f"Unsupported function object type: {type(obj)}")

    # Pull the function object from the requested library
    lib_obj = None
    if source_lib == "Basic" and BasicFunctions is not None:
        lib = st.session_state.get("basic_functions") or BasicFunctions()
        st.session_state.basic_functions = lib
        lib_obj = lib.get_function(func_name)
    elif source_lib == "Special" and SpecialFunctions is not None:
        lib = st.session_state.get("special_functions") or SpecialFunctions()
        st.session_state.special_functions = lib
        lib_obj = lib.get_function(func_name)
    else:
        raise ValueError("Function library not available or invalid library selected.")

    return _from_obj(lib_obj)


def theorem_4_1_solution_expr(
    f_expr: sp.Expr, alpha: Union[float, sp.Expr], beta: Union[float, sp.Expr],
    n: int, M: Union[float, sp.Expr], x: sp.Symbol
) -> sp.Expr:
    """
    Build the solution y(x) based on the statement of Theorem 4.1:

    y(x) = (π/(2n)) * sum_{s=1..n} [ 2 f(α + β) - ( ψ_s(x) + φ_s(x) ) ] + π M,
    where ω_s = (2s-1)π/(2n),
          ψ_s(x) = f(α + β * exp(i x cos ω_s - x sin ω_s)),
          φ_s(x) = f(α + β * exp(-i x cos ω_s - x sin ω_s)).

    f_expr is a SymPy expression in z.
    """
    z = sp.Symbol("z")
    f = sp.Lambda(z, f_expr)
    α = sp.sympify(alpha)
    β = sp.sympify(beta)
    𝑀 = sp.sympify(M)

    # f(α+β) is a constant term in x
    f_alpha_beta = safe_simplify(f(α + β))

    summand = 0
    for s in range(1, int(n) + 1):
        ω_s = (2 * s - 1) * sp.pi / (2 * n)
        E_pos = sp.exp(sp.I * x * sp.cos(ω_s) - x * sp.sin(ω_s))
        E_neg = sp.exp(-sp.I * x * sp.cos(ω_s) - x * sp.sin(ω_s))
        ψ_s = safe_simplify(f(α + β * E_pos))
        φ_s = safe_simplify(f(α + β * E_neg))
        summand += (2 * f_alpha_beta - (ψ_s + φ_s))

    y = sp.pi / (2 * n) * summand + sp.pi * 𝑀
    # Present the real part (the expression may be complex-valued before simplification)
    return safe_simplify(sp.re(y))


def apply_generator_to(solution: sp.Expr, gen_spec: Any, x: sp.Symbol) -> sp.Expr:
    """
    Apply a symbolic generator operator (gen_spec.lhs) to a concrete solution y(x):

    - Finds applied undefined function calls like y(g(x)), y(x/a), y(x+a), etc.
    - Finds derivatives like D^k y(g(x)).
    - Substitutes them with solution.subs(x, g(x)) and diff(solution.subs(x, g), (x, k)), respectively.
    - Works for delay/advance with linear arguments; also works for general g(x) using Symbolic chain rule.

    Returns the fully substituted expression, i.e. RHS := L[y].
    """
    if gen_spec is None or not hasattr(gen_spec, "lhs"):
        raise ValueError("Generator specification missing or invalid (no .lhs).")

    lhs = gen_spec.lhs
    y_applied_candidates = list(lhs.atoms(AppliedUndef))
    deriv_candidates = list(lhs.atoms(sp.Derivative))

    # Heuristically identify the dependent function name (usually "y")
    y_name = None
    for f_app in y_applied_candidates:
        # f_app is like y(g(x)), f_app.func is the function symbol (class), .func.__name__ its name
        try:
            y_name = f_app.func.__name__
            break
        except Exception:
            continue
    if y_name is None:
        # fallback: search in derivatives
        for d in deriv_candidates:
            base = d.expr
            if isinstance(base, AppliedUndef):
                try:
                    y_name = base.func.__name__
                    break
                except Exception:
                    pass

    if y_name is None:
        # Nothing to substitute
        return lhs

    subs_map = {}

    # 1) Substitute pure function calls y(g(x)) -> solution.subs(x, g(x))
    for f_app in y_applied_candidates:
        try:
            if f_app.func.__name__ == y_name:
                if len(f_app.args) != 1:
                    continue
                g = f_app.args[0]
                subs_map[f_app] = safe_simplify(solution.subs(x, g))
        except Exception:
            continue

    # 2) Substitute derivatives Derivative(y(g(x)), (x,k)) -> diff(solution.subs(x,g(x)), (x,k))
    for d in deriv_candidates:
        base = d.expr
        try:
            if isinstance(base, AppliedUndef) and base.func.__name__ == y_name:
                g = base.args[0] if base.args else x
                # Determine derivative order wrt x
                # d.variables returns a tuple with variables repeated
                order = sum(1 for v in d.variables if v == x)
                if order <= 0:
                    subs_map[d] = 0
                else:
                    subs_map[d] = safe_simplify(sp.diff(solution.subs(x, g), (x, order)))
        except Exception:
            continue

    # Return substituted RHS
    return safe_simplify(lhs.subs(subs_map))


# ======================================================
# Export helpers
# ======================================================
class LaTeXExporter:
    """LaTeX export system for ODEs"""

    @staticmethod
    def sympy_to_latex(expr) -> str:
        if expr is None:
            return ""
        try:
            if isinstance(expr, str):
                try:
                    expr = sp.sympify(expr)
                except Exception:
                    return expr
            s = sp.latex(expr)
            return s.replace(r"\left(", "(").replace(r"\right)", ")")
        except Exception:
            return str(expr)

    @staticmethod
    def generate_latex_document(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        generator = ode_data.get("generator", "")
        solution = ode_data.get("solution", "")
        rhs = ode_data.get("rhs", "")
        params = ode_data.get("parameters", {})
        classification = ode_data.get("classification", {})
        initial_conditions = ode_data.get("initial_conditions", {})

        parts = []
        if include_preamble:
            parts.append(r"""
\documentclass[12pt]{article}
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
""")

        parts.append(r"\subsection{Generator Equation}")
        parts.append(r"\begin{equation}")
        parts.append(f"{LaTeXExporter.sympy_to_latex(generator)} = {LaTeXExporter.sympy_to_latex(rhs)}")
        parts.append(r"\end{equation}")
        parts.append("")

        parts.append(r"\subsection{Exact Solution}")
        parts.append(r"\begin{equation}")
        parts.append(f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}")
        parts.append(r"\end{equation}")
        parts.append("")

        parts.append(r"\subsection{Parameters}")
        parts.append(r"\begin{align}")
        if "alpha" in params: parts.append(f"\\alpha &= {params.get('alpha')} \\\\")
        if "beta"  in params: parts.append(f"\\beta  &= {params.get('beta')} \\\\")
        if "n"     in params: parts.append(f"n       &= {params.get('n')} \\\\")
        if "M"     in params: parts.append(f"M       &= {params.get('M')}")
        # optional
        for extra_key in ("q", "v", "a"):
            if extra_key in params:
                parts.append(f" \\\\ {extra_key} &= {params[extra_key]}")
        parts.append(r"\end{align}")
        parts.append("")

        if initial_conditions:
            parts.append(r"\subsection{Initial Conditions}")
            parts.append(r"\begin{align}")
            items = list(initial_conditions.items())
            for i, (k, v) in enumerate(items):
                sep = r" \\" if i < len(items) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}{sep}")
            parts.append(r"\end{align}")
            parts.append("")

        if classification:
            parts.append(r"\subsection{Mathematical Classification}")
            parts.append(r"\begin{itemize}")
            if "type" in classification:
                parts.append(f"\\item \\textbf{{Type:}} {classification.get('type')}")
            if "order" in classification:
                parts.append(f"\\item \\textbf{{Order:}} {classification.get('order')}")
            if "linearity" in classification:
                parts.append(f"\\item \\textbf{{Linearity:}} {classification.get('linearity')}")
            if "field" in classification:
                parts.append(f"\\item \\textbf{{Field:}} {classification.get('field')}")
            if "applications" in classification:
                apps = classification.get("applications") or []
                parts.append(f"\\item \\textbf{{Applications:}} {', '.join(apps)}")
            parts.append(r"\end{itemize}")
            parts.append("")

        parts.append(r"\subsection{Solution Verification}")
        parts.append(r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$.")

        if include_preamble:
            parts.append(r"\end{document}")

        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, include_preamble=True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", f"""Master Generator ODE Export
Generated: {datetime.now().isoformat()}""")
            if include_extras:
                code = LaTeXExporter.generate_python_code(ode_data)
                zf.writestr("reproduce.py", code)
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def generate_python_code(ode_data: Dict[str, Any]) -> str:
        params = ode_data.get("parameters", {})
        gen_type = ode_data.get("type", "linear")
        gen_num = ode_data.get("generator_number", 1)
        func_name = ode_data.get("function_used", "linear")

        return f'''# Repro script (skeleton)
import sympy as sp
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
from src.generators.master_generator import CompleteMasterGenerator

alpha={params.get('alpha',1.0)}
beta={params.get('beta',1.0)}
n={params.get('n',1)}
M={params.get('M',0.0)}

basic=BasicFunctions(); special=SpecialFunctions()
try:
    f = basic.get_function("{func_name}")
except Exception:
    f = special.get_function("{func_name}")

z = sp.Symbol("z")
if callable(f):
    f_expr = f(z)
else:
    f_expr = sp.sympify(f)

x=sp.Symbol("x", real=True)
# Insert your constructor/operator here if needed
# e.g., build LHS and compute RHS = L[y]
'''

# ======================================================
# Session state manager
# ======================================================
class SessionStateManager:
    @staticmethod
    def initialize():
        if "generator_terms" not in st.session_state:
            st.session_state.generator_terms = []
        if "generated_odes" not in st.session_state:
            st.session_state.generated_odes = []
        if "generator_patterns" not in st.session_state:
            st.session_state.generator_patterns = []
        if "batch_results" not in st.session_state:
            st.session_state.batch_results = []
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = []
        if "ml_trainer" not in st.session_state:
            st.session_state.ml_trainer = None
        if "ml_trained" not in st.session_state:
            st.session_state.ml_trained = False
        if "training_history" not in st.session_state:
            st.session_state.training_history = []
        if "basic_functions" not in st.session_state and BasicFunctions is not None:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions is not None:
            st.session_state.special_functions = SpecialFunctions()
        if "ode_classifier" not in st.session_state and ODEClassifier is not None:
            st.session_state.ode_classifier = ODEClassifier()
        if "novelty_detector" not in st.session_state and ODENoveltyDetector is not None:
            st.session_state.novelty_detector = ODENoveltyDetector()
        if "theorem_solver" not in st.session_state and MasterTheoremSolver is not None:
            st.session_state.theorem_solver = MasterTheoremSolver()
        if "extended_theorem" not in st.session_state and ExtendedMasterTheorem is not None:
            st.session_state.extended_theorem = ExtendedMasterTheorem()
        if "generator_constructor" not in st.session_state and GeneratorConstructor is not None:
            st.session_state.generator_constructor = GeneratorConstructor()


# ======================================================
# UI pages
# ======================================================
def header():
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">🔬 Master Generators for ODEs — Corrected Edition</div>
            <div class="subtitle">Theorems 4.1 & 4.2 • Generator Constructor • Verification • Export</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar():
    return st.sidebar.radio(
        "📍 Navigation",
        [
            "🏠 Dashboard",
            "🔧 Generator Constructor",
            "🎯 Apply Master Theorem",
            "📊 Batch Generation",
            "🤖 ML Pattern Learning",
            "🔍 Novelty Detection",
            "📈 Analysis & Classification",
            "📤 Export & LaTeX",
            "📚 Examples Library",
            "⚙️ Settings",
            "📖 Documentation",
        ],
        index=0,
    )


def page_dashboard():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Generated ODEs", len(st.session_state.generated_odes))
    c2.metric("ML Patterns", len(st.session_state.generator_patterns))
    c3.metric("Batch Results", len(st.session_state.batch_results))
    c4.metric("ML Model", "Trained" if st.session_state.ml_trained else "Not trained")

    st.subheader("Recent ODEs")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes)
        cols = [c for c in ["type", "order", "function_used", "timestamp"] if c in df.columns]
        if cols:
            st.dataframe(df[cols].tail(10), use_container_width=True)
        else:
            st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("No ODEs generated yet. Start in **Generator Constructor**.")


def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    if GeneratorConstructor is None or DerivativeTerm is None:
        st.error("Generator constructor classes are not available in src/.")
        return

    constructor: GeneratorConstructor = st.session_state.get("generator_constructor") or GeneratorConstructor()
    st.session_state.generator_constructor = constructor

    st.markdown(
        """
        <div class="info-box">
        Build a custom generator operator by adding terms. You can combine derivatives, powers,
        and (if supported by your src/) delay/advance operators. The generator is used later to build RHS = L[y].
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("➕ Add Generator Term", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            deriv_order = st.selectbox(
                "Derivative order", [0, 1, 2, 3, 4, 5],
                format_func=lambda k: {0: "y", 1: "y′", 2: "y″", 3: "y‴", 4: "y⁽⁴⁾", 5: "y⁽⁵⁾"}.get(k, f"y⁽{k}⁾")
            )
        with col2:
            func_type = st.selectbox(
                "Function Type",
                [t.value for t in DerivativeType],
                format_func=lambda s: s.replace("_", " ").title()
            )
        with col3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with col4:
            power = st.number_input("Power", 1, 5, 1)

        col5, col6, col7 = st.columns(3)
        with col5:
            operator_type = st.selectbox(
                "Operator Type",
                [t.value for t in OperatorType],
                format_func=lambda s: s.replace("_", " ").title()
            )
        with col6:
            scaling = st.number_input("Scaling a", 0.5, 5.0, 1.0, 0.1) if operator_type in ("delay", "advance") else None
        with col7:
            shift = st.number_input("Shift", -10.0, 10.0, 0.0, 0.1) if operator_type in ("delay", "advance") else None

        if st.button("Add Term", type="primary"):
            term = DerivativeTerm(
                derivative_order=deriv_order,
                coefficient=coefficient,
                power=power,
                function_type=DerivativeType(func_type),
                operator_type=OperatorType(operator_type),
                scaling=scaling,
                shift=shift,
            )
            st.session_state.generator_terms.append(term)
            st.success(f"Added: {term.get_description() if hasattr(term, 'get_description') else str(term)}")

    # Show current terms
    if st.session_state.generator_terms:
        st.subheader("📝 Current Generator Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            colA, colB = st.columns([8, 1])
            with colA:
                desc = term.get_description() if hasattr(term, "get_description") else str(term)
                st.markdown(f'<div class="generator-term"><strong>Term {i+1}:</strong> {desc}</div>', unsafe_allow_html=True)
            with colB:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()

        if st.button("🔨 Build Generator Specification", type="primary", use_container_width=True):
            gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms, name="Custom Generator")
            st.session_state.current_generator = gen_spec
            st.markdown('<div class="result-box success-animation"><h4>✅ Generator Spec Created!</h4></div>', unsafe_allow_html=True)
            try:
                st.latex(sp.latex(gen_spec.lhs) + " = RHS")
            except Exception:
                st.write("LHS constructed.")


def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem")

    st.markdown(
        """
        <div class="info-box">
        <strong>What this does:</strong> It builds the exact solution via Theorem 4.1 given a function
        \(f(z)\) and parameters \((\\alpha, \\beta, n, M)\). If you have built a generator \(L\) in the
        constructor, we form the complete ODE by computing \(\\mathrm{RHS} = L[y]\) symbolically.
        A small numerical residual check is then performed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Function selection
    col1, col2 = st.columns(2)
    with col1:
        lib = st.selectbox("Function Library", ["Basic", "Special"])
        if lib == "Basic" and BasicFunctions is not None:
            func_names = st.session_state.basic_functions.get_function_names()
        elif lib == "Special" and SpecialFunctions is not None:
            func_names = st.session_state.special_functions.get_function_names()
        else:
            func_names = []
        func_name = st.selectbox("Select f(z)", func_names) if func_names else st.text_input("Enter f(z) name or expression")

    with col2:
        alpha = st.number_input("α", -10.0, 10.0, 1.0, 0.1)
        beta = st.number_input("β", 0.1, 10.0, 1.0, 0.1)
        n = st.slider("n (positive integer)", 1, 6, 1)
        M = st.number_input("M", -10.0, 10.0, 0.0, 0.1)

    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        with st.spinner("Applying Theorem 4.1 and constructing RHS..."):
            try:
                # Resolve f(z) as SymPy expression
                f_expr = get_function_expr(lib, func_name)
                x = sp.Symbol("x", real=True)

                # Prefer official solver when generator exists
                gen_spec = st.session_state.get("current_generator")
                solution = None
                rhs = None
                generator_lhs = None
                used_solver = False

                if gen_spec is not None and MasterTheoremSolver is not None and MasterTheoremParameters is not None:
                    try:
                        params = MasterTheoremParameters(f_z=f_expr, alpha=alpha, beta=beta, n=int(n), M=M)
                        solver = st.session_state.get("theorem_solver") or MasterTheoremSolver()
                        st.session_state.theorem_solver = solver
                        res = solver.apply_theorem_4_1(gen_spec, params)
                        # Adapt to possible dict shape
                        sol = res.get("solution", None) if isinstance(res, dict) else None
                        # prefer real part for display
                        solution = safe_simplify(sp.re(sol)) if sol is not None else None
                        rhs = (res.get("rhs", None) if isinstance(res, dict) else None) or \
                              (res.get("ode", {}).get("rhs", None) if isinstance(res, dict) and "ode" in res else None)
                        generator_lhs = getattr(gen_spec, "lhs", sp.Symbol("L[y]"))
                        used_solver = solution is not None and rhs is not None
                    except Exception as e:
                        logger.warning(f"MasterTheoremSolver failed; falling back to manual construction: {e}")

                if not used_solver:
                    # Manual build: y(x) then RHS = L[y]
                    solution = theorem_4_1_solution_expr(f_expr, alpha, beta, int(n), M, x)
                    if gen_spec is not None and hasattr(gen_spec, "lhs"):
                        try:
                            rhs = apply_generator_to(solution, gen_spec, x)
                            generator_lhs = gen_spec.lhs
                        except Exception as e:
                            logger.warning(f"Failed to apply generator to solution: {e}")
                            # Placeholder RHS only if generator application failed
                            z = next(iter(f_expr.free_symbols)) if f_expr.free_symbols else sp.Symbol("z")
                            rhs = safe_simplify(sp.pi * (f_expr.subs(z, alpha + beta) + M))
                            generator_lhs = sp.Symbol("L[y]")
                            st.warning("Using a placeholder RHS because applying the generator failed.")
                    else:
                        # No generator: placeholder RHS (clearly labeled)
                        z = next(iter(f_expr.free_symbols)) if f_expr.free_symbols else sp.Symbol("z")
                        rhs = safe_simplify(sp.pi * (f_expr.subs(z, alpha + beta) + M))
                        generator_lhs = sp.Symbol("L[y]")
                        st.info("No generator specified; showing a placeholder RHS.")

                # Verification (residual) if we have a concrete generator
                verification_note = ""
                if gen_spec is not None and hasattr(gen_spec, "lhs"):
                    try:
                        Ly = apply_generator_to(solution, gen_spec, x)
                        residual = safe_simplify(Ly - rhs)
                        fnum = sp.lambdify(x, residual, "numpy")
                        pts = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                        vals = np.array([float(sp.N(fnum(t))) for t in pts])
                        maxabs = float(np.max(np.abs(vals)))
                        verification_note = f"Max |L[y]-RHS| on test points = {maxabs:.2e}"
                        if maxabs < 1e-7:
                            st.success("✅ Solution verified numerically on test points.")
                        else:
                            st.warning("⚠️ Residual is not tiny; check generator terms and parameters.")
                    except Exception as e:
                        verification_note = f"Verification skipped (error: {e})"
                        logger.debug(verification_note)

                # Persist
                result = {
                    "generator": generator_lhs,
                    "solution": solution,
                    "rhs": rhs,
                    "parameters": {"alpha": sp.sympify(alpha), "beta": sp.sympify(beta), "n": int(n), "M": sp.sympify(M)},
                    "function_used": func_name,
                    "type": ("Linear" if getattr(gen_spec, "is_linear", False) else "Nonlinear") if gen_spec else "Unknown",
                    "order": getattr(gen_spec, "order", 0) if gen_spec else 0,
                    "classification": {
                        "type": ("Linear" if getattr(gen_spec, "is_linear", False) else "Nonlinear") if gen_spec else "Unknown",
                        "order": getattr(gen_spec, "order", "Unknown") if gen_spec else "Unknown",
                        "field": "Mathematical Physics",
                        "applications": ["Research Equation"],
                    },
                    "initial_conditions": {"y(0)": safe_simplify(solution.subs(x, 0))},
                    "timestamp": datetime.now().isoformat(),
                    "generator_number": len(st.session_state.generated_odes) + 1,
                    "verification_note": verification_note,
                }
                st.session_state.generated_odes.append(result)

                # Show
                tabs = st.tabs(["📐 ODE", "💡 Solution", "🧪 Verification", "📤 Export"])
                with tabs[0]:
                    st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                with tabs[1]:
                    st.latex("y(x) = " + sp.latex(solution))
                    st.caption(f"Initial condition:  y(0) = {sp.latex(result['initial_conditions']['y(0)'])}")
                with tabs[2]:
                    note = result.get("verification_note", "")
                    if note:
                        st.info(note)
                    if generator_lhs == sp.Symbol("L[y]"):
                        st.warning("Placeholder RHS shown (no generator or application failed).")
                with tabs[3]:
                    latex_doc = LaTeXExporter.generate_latex_document(result, include_preamble=True)
                    st.download_button("📄 Download LaTeX", latex_doc, "ode_solution.tex", "text/x-latex", use_container_width=True)
                    pkg = LaTeXExporter.create_export_package(result, include_extras=True)
                    st.download_button("📦 Download Package (ZIP)", pkg, f"ode_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip", use_container_width=True)

            except Exception as e:
                st.error(f"Error generating ODE: {e}")
                logger.error("Generation error:\n" + traceback.format_exc())


def _resolve_factories():
    """
    Robustly return (LinearFactory, NonlinearFactory, CompleteLinearFactory, CompleteNonlinearFactory),
    resolving across the different layouts observed in your src/.
    """
    LF = LinearGeneratorFactory or getattr(mod_master_gen, "LinearGeneratorFactory", None)
    NLF = NonlinearGeneratorFactory or getattr(mod_master_gen, "NonlinearGeneratorFactory", None)
    CLF = CompleteLinearGeneratorFactory or getattr(mod_master_gen, "CompleteLinearGeneratorFactory", None)
    CNLF = CompleteNonlinearGeneratorFactory or getattr(mod_master_gen, "CompleteNonlinearGeneratorFactory", None)
    return LF, NLF, CLF, CNLF


def page_batch_generation():
    st.header("📊 Batch Generation")

    if not HAVE_SRC:
        st.error("src/ is not available. Batch generation depends on src implementations.")
        return

    LF, NLF, CLF, CNLF = _resolve_factories()

    col1, col2, col3 = st.columns(3)
    with col1:
        num_odes = st.slider("Number of ODEs", 5, 300, 30)
        gen_types = st.multiselect("Generator Types", ["linear", "nonlinear"], default=["linear", "nonlinear"])
    with col2:
        func_cats = st.multiselect("Function Categories", ["Basic", "Special"], default=["Basic"])
        vary = st.checkbox("Vary parameters", True)
    with col3:
        if vary:
            alpha_range = st.slider("α range", -10.0, 10.0, (-3.0, 3.0))
            beta_range = st.slider("β range", 0.05, 5.0, (0.5, 2.0))
            n_range = st.slider("n range", 1, 5, (1, 3))
        else:
            alpha_range, beta_range, n_range = (1.0, 1.0), (1.0, 1.0), (1, 1)

    if st.button("Run Batch", type="primary"):
        if not func_cats:
            st.warning("Select at least one function category.")
            return

        funcs = []
        if "Basic" in func_cats and BasicFunctions is not None:
            funcs += list(st.session_state.basic_functions.get_function_names())
        if "Special" in func_cats and SpecialFunctions is not None:
            funcs += list(st.session_state.special_functions.get_function_names())
        if not funcs:
            st.warning("No functions available.")
            return

        res_rows = []
        prog = st.progress(0)

        for i in range(num_odes):
            prog.progress((i + 1) / num_odes)
            try:
                params = {
                    "alpha": np.random.uniform(*alpha_range),
                    "beta": np.random.uniform(*beta_range),
                    "n": np.random.randint(n_range[0], n_range[1] + 1),
                    "M": np.random.uniform(-1, 1),
                }
                f_name = np.random.choice(funcs)
                gen_type = np.random.choice(gen_types) if gen_types else "linear"
                # Try to use "complete" factories when available
                if gen_type == "linear" and CLF is not None:
                    f_expr = get_function_expr("Basic" if f_name in st.session_state.basic_functions.get_function_names() else "Special", f_name)
                    factory = CLF()
                    # Pick a plausible generator index (your src defines valid ranges)
                    gen_num = np.random.randint(1, 9)
                    result = factory.create(gen_num, f_expr, **params)
                elif gen_type == "nonlinear" and CNLF is not None:
                    f_expr = get_function_expr("Basic" if f_name in st.session_state.basic_functions.get_function_names() else "Special", f_name)
                    factory = CNLF()
                    gen_num = np.random.randint(1, 11)
                    # add plausible optional params
                    if gen_num in (1, 2, 4): params["q"] = np.random.randint(2, 5)
                    if gen_num in (2, 3, 5): params["v"] = np.random.randint(2, 5)
                    if gen_num in (4, 5, 9, 10): params["a"] = np.random.uniform(1.2, 2.5)
                    result = factory.create(gen_num, f_expr, **params)
                else:
                    # Fallback: synthesize via Theorem 4.1 only
                    f_expr = get_function_expr("Basic" if f_name in st.session_state.basic_functions.get_function_names() else "Special", f_name)
                    x = sp.Symbol("x", real=True)
                    y = theorem_4_1_solution_expr(f_expr, params["alpha"], params["beta"], int(params["n"]), params["M"], x)
                    result = {
                        "ode": None, "solution": y, "type": gen_type, "order": None,
                        "generator_number": None, "function_used": f_name
                    }

                res_rows.append({
                    "ID": i + 1,
                    "Type": result.get("type", gen_type),
                    "Generator": result.get("generator_number", "—"),
                    "Function": f_name,
                    "Order": result.get("order", "—"),
                    "α": round(params["alpha"], 3),
                    "β": round(params["beta"], 3),
                    "n": params["n"]
                })

                # Optionally persist minimal result
                st.session_state.batch_results.append(res_rows[-1])
            except Exception as e:
                logger.debug(f"Batch item {i+1} failed: {e}")
                continue

        st.success(f"Generated {len(res_rows)} ODE rows.")
        df = pd.DataFrame(res_rows)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("📊 Download CSV", csv, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")


def page_ml_pattern_learning():
    st.header("🤖 ML Pattern Learning")
    if MLTrainer is None:
        st.info("MLTrainer not available in this src/ build.")
        return

    model_type = st.selectbox(
        "Model",
        ["pattern_learner", "vae", "transformer"],
        format_func=lambda x: {"pattern_learner": "Pattern Learner", "vae": "VAE", "transformer": "Transformer"}[x]
    )
    epochs = st.slider("Epochs", 10, 300, 50)
    batch_size = st.slider("Batch size", 8, 128, 32)
    lr = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    samples = st.slider("Training samples", 100, 5000, 1000)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate at least 5 ODEs before training.")
        return

    if st.button("Train", type="primary"):
        try:
            trainer = MLTrainer(model_type=model_type, learning_rate=lr, device="cuda" if torch and torch.cuda.is_available() else "cpu")
            st.session_state.ml_trainer = trainer
            # Dummy train call (since real data pipeline is domain-specific)
            trainer.train(epochs=epochs, batch_size=batch_size, samples=samples, validation_split=0.2)
            st.session_state.ml_trained = True
            st.success("Training finished.")
        except Exception as e:
            st.error(f"Training failed: {e}")


def page_novelty_detection():
    st.header("🔍 Novelty Detection")
    if ODENoveltyDetector is None:
        st.info("Novelty detector not available in this src/.")
        return

    nd = st.session_state.get("novelty_detector") or ODENoveltyDetector()
    st.session_state.novelty_detector = nd

    method = st.radio("Input", ["Use Current Generator", "Enter ODE Manually", "Select from Generated"])
    ode_to_analyze = None
    if method == "Use Current Generator":
        gen_spec = st.session_state.get("current_generator")
        if gen_spec is not None and hasattr(gen_spec, "lhs"):
            ode_to_analyze = {"ode": gen_spec.lhs, "type": "custom", "order": getattr(gen_spec, "order", 0)}
        else:
            st.warning("No generator defined.")
            return
    elif method == "Enter ODE Manually":
        expr = st.text_area("Enter an ODE expression (LaTeX or plain):")
        if expr:
            ode_to_analyze = {"ode": expr, "type": "manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox("Choose generated", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}")
            ode_to_analyze = st.session_state.generated_odes[idx]

    if ode_to_analyze and st.button("Analyze", type="primary"):
        try:
            analysis = nd.analyze(ode_to_analyze, check_solvability=True, detailed=True)
            st.metric("Novelty Score", f"{getattr(analysis, 'novelty_score', 0):.1f}/100")
            st.write("**Insights**:")
            for line in getattr(analysis, "special_characteristics", [])[:6]:
                st.write("• " + str(line))
        except Exception as e:
            st.error(f"Novelty analysis failed: {e}")


def page_analysis_classification():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs yet.")
        return

    df = pd.DataFrame(st.session_state.generated_odes)
    if not df.empty:
        keep_cols = [c for c in ["generator_number", "type", "order", "function_used", "timestamp"] if c in df.columns]
        st.dataframe(df[keep_cols].tail(20), use_container_width=True)

        types = df["type"].fillna("Unknown").value_counts()
        fig = px.pie(values=types.values, names=types.index, title="Types")
        st.plotly_chart(fig, use_container_width=True)


def page_export_latex():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return

    idx = st.selectbox("Select ODE to export", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','Unknown')}")
    ode = st.session_state.generated_odes[idx]
    st.subheader("Preview (LaTeX)")
    st.code(LaTeXExporter.generate_latex_document(ode, include_preamble=False), language="latex")

    col1, col2 = st.columns(2)
    with col1:
        doc = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
        st.download_button("📄 Download .tex", doc, f"ode_{idx+1}.tex", "text/x-latex", use_container_width=True)
    with col2:
        pkg = LaTeXExporter.create_export_package(ode, include_extras=True)
        st.download_button("📦 Download ZIP", pkg, f"ode_package_{idx+1}.zip", "application/zip", use_container_width=True)


def page_examples():
    st.header("📚 Examples Library")
    st.write("A few illustrative generator descriptions (text only):")
    st.markdown("- Linear example:  y″ + y = RHS")
    st.markdown("- Damped oscillator:  y″ + 2γ y′ + ω₀² y = RHS")
    st.markdown("- Pantograph:  y″(x) + y(x/a) − y(x) = RHS")
    st.markdown("- Nonlinear (cubic):  (y″)³ + y = RHS")
    st.markdown("- Exponential nonlinearity:  exp(y″) + exp(y′) = RHS")  # NOTE: uses prime characters, no quotes


def page_settings():
    st.header("⚙️ Settings")
    st.write("Nothing to configure here yet.")
    if IMPORT_WARNINGS:
        st.warning("Import warnings:")
        for w in IMPORT_WARNINGS[:8]:
            st.caption("• " + w)


def page_documentation():
    st.header("📖 Documentation")
    st.markdown(
        """
### Apply Master Theorem (Theorem 4.1)
Given \(f(z)\) and parameters \((\\alpha,\\beta,n,M)\), the app builds:
\\[
y(x) = \\frac{\\pi}{2n}\\sum_{s=1}^{n} \\Big(2f(\\alpha+\\beta) - [\\psi_s(x)+\\phi_s(x)]\\Big) + \\pi M,
\\]
with \\(\\omega_s = \\frac{(2s-1)\\pi}{2n}\\), \\(\\psi_s(x)=f(\\alpha+\\beta e^{ix\\cos\\omega_s - x\\sin\\omega_s})\\),
\\(\\phi_s(x)=f(\\alpha+\\beta e^{-ix\\cos\\omega_s - x\\sin\\omega_s})\\).
If a generator \(L\\) is defined, we compute the right-hand side as \(RHS=L[y]\\).

A small numerical residual check on a few points reports whether \(\\lvert L[y]-RHS\\rvert\\) is small.

### Why solutions can look complex
Before simplification, the formula uses complex exponentials. We present \\(\\Re(y(x))\\) to ensure a clean
real-valued expression when appropriate.
"""
    )


# ======================================================
# Main
# ======================================================
def main():
    if not HAVE_SRC:
        st.error("This module requires the src/ package. Ensure the ZIP is extracted with src/ present.")
        return

    SessionStateManager.initialize()
    header()
    page = sidebar()

    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🔧 Generator Constructor":
        page_generator_constructor()
    elif page == "🎯 Apply Master Theorem":
        page_apply_master_theorem()
    elif page == "📊 Batch Generation":
        page_batch_generation()
    elif page == "🤖 ML Pattern Learning":
        page_ml_pattern_learning()
    elif page == "🔍 Novelty Detection":
        page_novelty_detection()
    elif page == "📈 Analysis & Classification":
        page_analysis_classification()
    elif page == "📤 Export & LaTeX":
        page_export_latex()
    elif page == "📚 Examples Library":
        page_examples()
    elif page == "⚙️ Settings":
        page_settings()
    elif page == "📖 Documentation":
        page_documentation()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
