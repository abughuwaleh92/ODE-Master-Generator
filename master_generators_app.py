"""
Master Generators for ODEs - Complete Streamlit App
- Compatible with the provided src/ package structure.
- Includes: Generator Constructor (up to 15th + symbolic k-th), Master Theorem application,
  ML & DL (trainer + novelty detection), Batch generation, Analysis, Visualizations,
  Physical applications, LaTeX export, Examples, Settings, Docs.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard libs
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import io
import json
import time
import math
import pickle
import zipfile
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Third-party libs
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import sympy as sp

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ML/DL deps (optional)
try:
    import torch
except Exception:
    torch = None

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# ──────────────────────────────────────────────────────────────────────────────
# Ensure src/ is importable
# ──────────────────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

HAVE_SRC = True
IMPORT_ERRORS: List[str] = []

# ──────────────────────────────────────────────────────────────────────────────
# Imports from src/ (robust to structure)
# ──────────────────────────────────────────────────────────────────────────────
try:
    # Generators
    from src.generators.master_generator import (
        MasterGenerator,
        EnhancedMasterGenerator,
        CompleteMasterGenerator,
        CompleteLinearGeneratorFactory,         # <- lives here (not in linear_generators.py)
        CompleteNonlinearGeneratorFactory,      # <- lives here
    )
except Exception as e:
    HAVE_SRC = False
    IMPORT_ERRORS.append(f"generators.master_generator: {e}")

try:
    from src.generators.linear_generators import LinearGeneratorFactory
except Exception as e:
    IMPORT_ERRORS.append(f"generators.linear_generators: {e}")
    LinearGeneratorFactory = None

try:
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
except Exception as e:
    IMPORT_ERRORS.append(f"generators.nonlinear_generators: {e}")
    NonlinearGeneratorFactory = None

try:
    from src.generators.generator_constructor import (
        GeneratorConstructor,
        GeneratorSpecification,
        DerivativeTerm,
        DerivativeType,
        OperatorType,
    )
except Exception as e:
    HAVE_SRC = False
    IMPORT_ERRORS.append(f"generators.generator_constructor: {e}")
    GeneratorConstructor = None
    GeneratorSpecification = None
    DerivativeTerm = None
    DerivativeType = None
    OperatorType = None

try:
    from src.generators.master_theorem import (
        MasterTheoremSolver,
        MasterTheoremParameters,
        ExtendedMasterTheorem,
    )
except Exception as e:
    IMPORT_ERRORS.append(f"generators.master_theorem: {e}")
    MasterTheoremSolver = None
    MasterTheoremParameters = None
    ExtendedMasterTheorem = None

try:
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
except Exception as e:
    IMPORT_ERRORS.append(f"generators.ode_classifier: {e}")
    ODEClassifier = None
    PhysicalApplication = None

# Functions
try:
    from src.functions.basic_functions import BasicFunctions
except Exception as e:
    IMPORT_ERRORS.append(f"functions.basic_functions: {e}")
    BasicFunctions = None

try:
    from src.functions.special_functions import SpecialFunctions
except Exception as e:
    IMPORT_ERRORS.append(f"functions.special_functions: {e}")
    SpecialFunctions = None

# ML / DL
try:
    from src.ml.pattern_learner import (
        GeneratorPatternLearner,
        GeneratorVAE,
        GeneratorTransformer,
        create_model,
    )
except Exception as e:
    IMPORT_ERRORS.append(f"ml.pattern_learner: {e}")
    GeneratorPatternLearner = None
    GeneratorVAE = None
    GeneratorTransformer = None
    create_model = None

try:
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
except Exception as e:
    IMPORT_ERRORS.append(f"ml.trainer: {e}")
    MLTrainer = None
    ODEDataset = None
    ODEDataGenerator = None

try:
    from src.ml.generator_learner import (
        GeneratorPattern,
        GeneratorPatternNetwork,
        GeneratorLearningSystem,
    )
except Exception as e:
    IMPORT_ERRORS.append(f"ml.generator_learner: {e}")
    GeneratorPattern = None
    GeneratorPatternNetwork = None
    GeneratorLearningSystem = None

try:
    from src.dl.novelty_detector import (
        ODENoveltyDetector,
        NoveltyAnalysis,
        ODETokenizer,
        ODETransformer,
    )
except Exception as e:
    IMPORT_ERRORS.append(f"dl.novelty_detector: {e}")
    ODENoveltyDetector = None
    NoveltyAnalysis = None
    ODETokenizer = None
    ODETransformer = None

# Utils
try:
    from src.utils.config import Settings, AppConfig
except Exception as e:
    IMPORT_ERRORS.append(f"utils.config: {e}")
    Settings = None
    AppConfig = None

try:
    from src.utils.cache import CacheManager, cached
except Exception as e:
    IMPORT_ERRORS.append(f"utils.cache: {e}")
    CacheManager = None
    def cached(*a, **k):
        def deco(fn): return fn
        return deco

try:
    from src.utils.validators import ParameterValidator
except Exception as e:
    IMPORT_ERRORS.append(f"utils.validators: {e}")
    ParameterValidator = None

if not HAVE_SRC:
    logger.warning("Some critical src modules failed to import. The app will run with fallbacks.")
    for err in IMPORT_ERRORS:
        logger.warning(f"Import issue: {err}")

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config + CSS
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Master Generators ODE System - Complete Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.0rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: .2rem; }
.subtitle { font-size: 1.05rem; opacity: 0.95; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 1.2rem; border-radius: 12px;
    text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.generator-term {
    background: linear-gradient(135deg, #f5f7fa 0%, #dee6f5 100%);
    padding: 12px; border-radius: 10px; margin: 6px 0;
    border-left: 5px solid #667eea; box-shadow: 0 3px 10px rgba(0,0,0,0.06);
}
.result-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border: 2px solid #4caf50; padding: 1.2rem; border-radius: 12px; margin: .8rem 0;
}
.error-box {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border: 2px solid #e53935; padding: 1.0rem; border-radius: 10px; margin: .8rem 0;
}
.ml-box {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border: 2px solid #ff9800; padding: 1.0rem; border-radius: 10px; margin: .8rem 0;
}
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #2196f3; padding: .9rem; border-radius: 9px; margin: .8rem 0;
}
.latex-export-box {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border: 2px solid #9c27b0; padding: 1.0rem; border-radius: 10px; margin: 1.0rem 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def nsafe(x: float) -> sp.Expr:
    """Convert float to a SymPy Rational if close; otherwise to sympy Float."""
    try:
        return sp.nsimplify(x, rational=True, maxsteps=10)
    except Exception:
        return sp.Float(x)

def ensure_symbolic_params(alpha: float, beta: float, n: int, M: float) -> Tuple[sp.Symbol, sp.Symbol, sp.Integer, sp.Symbol]:
    return nsafe(alpha), nsafe(beta), sp.Integer(int(n)), nsafe(M)

def get_function_expr(source_obj: Any, name: str) -> sp.Expr:
    """
    Ask BasicFunctions/SpecialFunctions for a SymPy expression in symbol z.
    Falls back to simple defaults if library is missing.
    """
    z = sp.Symbol("z", real=True)
    try:
        if source_obj is None or not hasattr(source_obj, "get_function"):
            # Fallback set
            funcs = {
                "exponential": sp.exp(z),
                "sin": sp.sin(z),
                "cos": sp.cos(z),
                "polynomial": z**2 + 1,
                "log1p": sp.log(1 + z),
            }
            return funcs.get(name, sp.exp(z))
        expr = source_obj.get_function(name)
        # ensure z symbol usage
        if isinstance(expr, (int, float)):
            return sp.Integer(expr)
        return sp.sympify(expr).subs({sp.Symbol("z"): z})
    except Exception as e:
        logger.warning(f"get_function_expr fallback for '{name}': {e}")
        return sp.exp(z)

def theorem_4_1_solution_expr(fz_expr: sp.Expr, alpha: sp.Expr, beta: sp.Expr, n: sp.Integer, M: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """
    Safe symbolic fallback for Theorem 4.1 used when src solver isn't available.
    Reference structure:
      ω_s = (2s-1)π/(2n)
      y(x) = π/(2n) * Σ_{s=1..n} [ 2 f(α+β)
                - f(α + β * exp(i x cos ω_s - x sin ω_s))
                - f(α + β * exp(-i x cos ω_s - x sin ω_s)) ] + π M
    """
    s = sp.symbols("s", integer=True, positive=True)
    w_s = (2*s - 1) * sp.pi / (2*n)
    z = sp.Symbol("z", real=True)
    term_pos = fz_expr.subs(z, alpha + beta * sp.exp(sp.I * x * sp.cos(w_s) - x * sp.sin(w_s)))
    term_neg = fz_expr.subs(z, alpha + beta * sp.exp(-sp.I * x * sp.cos(w_s) - x * sp.sin(w_s)))
    series = 2 * fz_expr.subs(z, alpha + beta) - term_pos - term_neg
    y = (sp.pi / (2*n)) * sp.summation(series, (s, 1, n)) + sp.pi * M
    return sp.simplify(y)

def apply_term_on_function(
    y_expr: sp.Expr,
    x: sp.Symbol,
    derivative_order: Union[int, str],
    coefficient: sp.Expr,
    power: int,
    d_type: Optional[DerivativeType],
    op_type: Optional[OperatorType],
    scaling: Optional[sp.Expr],
    shift: Optional[sp.Expr],
    eps: sp.Expr = sp.Symbol("epsilon", positive=True)
) -> sp.Expr:
    """
    Apply a single generator term to y(x) to produce the corresponding L[y]-component.
    - Supports derivative order up to 15.
    - 'k' (symbolic) -> treated as a *formal* display term (does not act on y).
    - OperatorType: STANDARD, DELAY (y(x/a + shift)), ADVANCE (y(a x + shift)).
    """
    # Handle formal symbolic k-th derivative: generate a symbolic placeholder term.
    if isinstance(derivative_order, str) and derivative_order.strip().lower() == "k":
        # Formal symbol: D^k y(x)
        Dk = sp.Symbol("D^k y(x)")
        base = Dk
    else:
        m = int(derivative_order)
        if m < 0 or m > 15:
            m = max(0, min(15, m))
        base = sp.diff(y_expr, (x, m)) if m > 0 else y_expr

    # OperatorType effects (scaling/shift)
    # Using basic pantograph-style chain rule for d/dx[y(a x + b)]  -> a^m y^(m)(a x + b)
    if op_type is not None:
        op_name = str(op_type.value if hasattr(op_type, "value") else op_type)
    else:
        op_name = "standard"

    a = scaling if scaling not in (None, "") else 1
    b = shift if shift not in (None, "") else 0
    a = nsafe(float(a)) if isinstance(a, (int, float)) else a
    b = nsafe(float(b)) if isinstance(b, (int, float)) else b

    if op_name.lower() == "delay":
        # y(x/a + b) => factor (1/a)^m when m-th derivative
        if isinstance(derivative_order, str) and derivative_order.strip().lower() == "k":
            factor = sp.Symbol("(1/a)^k")  # formal
        else:
            m = int(derivative_order) if not isinstance(derivative_order, str) else 0
            factor = (1/a)**m
        base = base.subs(x, x/a + b)
        base = factor * base
    elif op_name.lower() == "advance":
        # y(a x + b) => factor a^m
        if isinstance(derivative_order, str) and derivative_order.strip().lower() == "k":
            factor = sp.Symbol("a^k")  # formal
        else:
            m = int(derivative_order) if not isinstance(derivative_order, str) else 0
            factor = (a**m)
        base = base.subs(x, a*x + b)
        base = factor * base
    # (Other operator types can be added as needed.)

    # DerivativeType transforms
    if d_type is None:
        tname = "linear"
    else:
        tname = str(d_type.value if hasattr(d_type, "value") else d_type).lower()

    if tname == "linear":
        term_expr = base
    elif tname == "power":
        term_expr = base**power
    elif tname == "exponential":
        term_expr = sp.exp(base)
    elif tname == "trigonometric":
        # default to sin; power means sin(base)**power
        term_expr = sp.sin(base)**power
    elif tname == "logarithmic":
        term_expr = sp.log(eps + sp.Abs(base))**power
    elif tname == "hyperbolic":
        term_expr = sp.sinh(base)**power
    elif tname == "algebraic":
        # treat as |base|^(power)^(1/power) if power>1? Keep simple:
        term_expr = sp.sign(base) * sp.Abs(base)**sp.Rational(1, max(1, power))
    else:
        # SPECIAL / COMPOSITE -> keep linear if unknown
        term_expr = base

    return sp.simplify(coefficient * term_expr)

def apply_generator_to(y_expr: sp.Expr, gen_spec_or_terms: Any, x: sp.Symbol) -> sp.Expr:
    """
    Compute RHS as L[y] for the current generator.
    Accepts:
      - GeneratorSpecification with .terms list
      - Or raw list[DerivativeTerm-like]
    """
    try:
        terms = getattr(gen_spec_or_terms, "terms", gen_spec_or_terms)
    except Exception:
        terms = []

    rhs_parts = []
    for t in terms:
        try:
            d_order = getattr(t, "derivative_order", 0)
            coeff = getattr(t, "coefficient", 1.0)
            power = getattr(t, "power", 1)
            ftype = getattr(t, "function_type", None)
            otype = getattr(t, "operator_type", None)
            scaling = getattr(t, "scaling", None)
            shift = getattr(t, "shift", None)

            coeff = nsafe(float(coeff)) if isinstance(coeff, (int, float)) else coeff
            power = int(power)

            part = apply_term_on_function(
                y_expr=y_expr, x=x,
                derivative_order=d_order, coefficient=coeff, power=power,
                d_type=ftype, op_type=otype, scaling=scaling, shift=shift
            )
            rhs_parts.append(part)
        except Exception as e:
            logger.warning(f"Skipping term due to error: {e}")
    return sp.simplify(sp.Add(*rhs_parts) if rhs_parts else sp.Integer(0))

def latex_cleanup(expr: Any) -> str:
    try:
        return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
    except Exception:
        return str(expr)

# ──────────────────────────────────────────────────────────────────────────────
# LaTeX Exporter
# ──────────────────────────────────────────────────────────────────────────────
class LaTeXExporter:
    @staticmethod
    def generate_document(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        generator = ode_data.get("generator", "")
        solution = ode_data.get("solution", "")
        rhs = ode_data.get("rhs", "")
        params = ode_data.get("parameters", {})
        classification = ode_data.get("classification", {})
        initial_conditions = ode_data.get("initial_conditions", {})

        parts = []
        if include_preamble:
            parts.append(r"""\documentclass[12pt]{article}
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
        parts.append(f"{latex_cleanup(generator)} = {latex_cleanup(rhs)}")
        parts.append(r"\end{equation}\n")

        parts.append(r"\subsection{Exact Solution}")
        parts.append(r"\begin{equation}")
        parts.append(f"y(x) = {latex_cleanup(solution)}")
        parts.append(r"\end{equation}\n")

        parts.append(r"\subsection{Parameters}")
        parts.append(r"\begin{align}")
        parts.append(rf"\alpha &= {latex_cleanup(params.get('alpha',''))} \\")
        parts.append(rf"\beta  &= {latex_cleanup(params.get('beta',''))} \\")
        parts.append(rf"n      &= {latex_cleanup(params.get('n',''))} \\")
        parts.append(rf"M      &= {latex_cleanup(params.get('M',''))}")
        # optional extras:
        for extra in ("q", "v", "a"):
            if extra in params:
                parts.append(rf" \\ {extra} &= {latex_cleanup(params[extra])}")
        parts.append(r"\end{align}\n")

        if initial_conditions:
            parts.append(r"\subsection{Initial Conditions}")
            parts.append(r"\begin{align}")
            N = len(initial_conditions)
            for i, (k, v) in enumerate(initial_conditions.items()):
                sep = r" \\" if i < N - 1 else ""
                parts.append(f"{k} &= {latex_cleanup(v)}{sep}")
            parts.append(r"\end{align}\n")

        if classification:
            parts.append(r"\subsection{Mathematical Classification}")
            parts.append(r"\begin{itemize}")
            if "type" in classification:
                parts.append(rf"\item \textbf{{Type:}} {classification['type']}")
            if "order" in classification:
                parts.append(rf"\item \textbf{{Order:}} {classification['order']}")
            if "linearity" in classification:
                parts.append(rf"\item \textbf{{Linearity:}} {classification['linearity']}")
            if "field" in classification:
                parts.append(rf"\item \textbf{{Field:}} {classification['field']}")
            if "applications" in classification:
                apps = ", ".join(classification["applications"])
                parts.append(rf"\item \textbf{{Applications:}} {apps}")
            parts.append(r"\end{itemize}\n")

        parts.append(r"\subsection{Solution Verification}")
        parts.append("Substitute $y(x)$ into the generator operator to verify $L[y] = \\text{RHS}$.")
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_package(ode_data: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ode_document.tex", LaTeXExporter.generate_document(ode_data, include_preamble=True))
            z.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            readme = f"""Master Generator ODE Export
Generated: {datetime.now().isoformat()}

Files:
- ode_document.tex
- ode_data.json
- reproduce.py

To compile: pdflatex ode_document.tex
"""
            z.writestr("README.txt", readme)
            z.writestr("reproduce.py", LaTeXExporter.generate_reproducer(ode_data))
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def generate_reproducer(ode_data: Dict[str, Any]) -> str:
        params = ode_data.get("parameters", {})
        gtype = ode_data.get("type", "linear")
        gnum = ode_data.get("generator_number", 1)
        func = ode_data.get("function_used", "exponential")
        return f'''"""
Reproduce ODE from Master Generators System
"""

import sympy as sp
from src.generators.master_generator import CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions

params = {{
    "alpha": {params.get("alpha", 1.0)},
    "beta": {params.get("beta", 1.0)},
    "n": {params.get("n", 1)},
    "M": {params.get("M", 0.0)}
}}

# optional
for k in ("q","v","a"):
    params.setdefault(k, None)

bf = BasicFunctions()
sf = SpecialFunctions()
try:
    fz = bf.get_function("{func}")
except Exception:
    fz = sf.get_function("{func}")

factory = CompleteLinearGeneratorFactory() if "{gtype}".lower()=="linear" else CompleteNonlinearGeneratorFactory()
res = factory.create({gnum}, fz, **params)

print("ODE:", res.get("ode"))
print("Solution:", res.get("solution"))
print("Type:", res.get("type"))
print("Order:", res.get("order"))
'''

# ──────────────────────────────────────────────────────────────────────────────
# Session State Manager
# ──────────────────────────────────────────────────────────────────────────────
class SessionStateManager:
    @staticmethod
    def init():
        if "generator_constructor" not in st.session_state:
            st.session_state.generator_constructor = GeneratorConstructor() if GeneratorConstructor else None
        if "generator_terms" not in st.session_state:
            st.session_state.generator_terms: List[Any] = []
        if "current_generator" not in st.session_state:
            st.session_state.current_generator = None

        if "generated_odes" not in st.session_state:
            st.session_state.generated_odes: List[Dict[str, Any]] = []
        if "generator_patterns" not in st.session_state:
            st.session_state.generator_patterns: List[Any] = []
        if "batch_results" not in st.session_state:
            st.session_state.batch_results: List[Dict[str, Any]] = []
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results: List[Dict[str, Any]] = []

        if "basic_functions" not in st.session_state:
            st.session_state.basic_functions = BasicFunctions() if BasicFunctions else None
        if "special_functions" not in st.session_state:
            st.session_state.special_functions = SpecialFunctions() if SpecialFunctions else None

        if "theorem_solver" not in st.session_state:
            st.session_state.theorem_solver = MasterTheoremSolver() if MasterTheoremSolver else None

        if "novelty_detector" not in st.session_state:
            st.session_state.novelty_detector = ODENoveltyDetector() if ODENoveltyDetector else None
        if "ode_classifier" not in st.session_state:
            st.session_state.ode_classifier = ODEClassifier() if ODEClassifier else None

        if "ml_trainer" not in st.session_state:
            st.session_state.ml_trainer = None
        if "ml_trained" not in st.session_state:
            st.session_state.ml_trained = False
        if "training_history" not in st.session_state:
            st.session_state.training_history = {}

        if "cache_manager" not in st.session_state:
            st.session_state.cache_manager = CacheManager() if CacheManager else None

# ──────────────────────────────────────────────────────────────────────────────
# Main app routing
# ──────────────────────────────────────────────────────────────────────────────
def main():
    SessionStateManager.init()

    st.markdown("""
    <div class="main-header">
        <div class="main-title">🔬 Master Generators for ODEs</div>
        <div class="subtitle">Theorems 4.1 & 4.2 • Generator Constructor (≤15th + k-th) • ML/DL • LaTeX • Batch • Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    if not HAVE_SRC:
        st.error("⚠️ This app requires the `src/` package. It appears some critical modules could not be imported.")
        with st.expander("Import diagnostics"):
            for err in IMPORT_ERRORS:
                st.write("•", err)

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
            "🔬 Physical Applications",
            "📐 Visualization",
            "📤 Export & LaTeX",
            "📚 Examples Library",
            "⚙️ Settings",
            "📖 Documentation",
        ],
    )

    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔧 Generator Constructor":
        generator_constructor_page()
    elif page == "🎯 Apply Master Theorem":
        master_theorem_page()
    elif page == "🤖 ML Pattern Learning":
        ml_pattern_learning_page()
    elif page == "📊 Batch Generation":
        batch_generation_page()
    elif page == "🔍 Novelty Detection":
        novelty_detection_page()
    elif page == "📈 Analysis & Classification":
        analysis_classification_page()
    elif page == "🔬 Physical Applications":
        physical_applications_page()
    elif page == "📐 Visualization":
        visualization_page()
    elif page == "📤 Export & LaTeX":
        export_latex_page()
    elif page == "📚 Examples Library":
        examples_library_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "📖 Documentation":
        documentation_page()

# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

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
        status = "✅ Trained" if st.session_state.ml_trained else "⏳ Not Trained"
        st.markdown(f'<div class="metric-card"><h3>🤖 ML Model</h3><p style="font-size:1.1rem">{status}</p></div>', unsafe_allow_html=True)

    st.subheader("📊 Recent Activity")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes)[["type","order","generator_number","timestamp"]].tail(8)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No ODEs generated yet. Start with **Generator Constructor**.")

def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    st.markdown("""
    <div class="info-box">Build custom generators by combining derivatives and transforms. 
    This page creates a generator specification and, if desired, applies Theorem 4.1 to produce an exact solution and RHS.</div>
    """, unsafe_allow_html=True)

    if not GeneratorConstructor or not DerivativeTerm:
        st.error("Generator constructor is unavailable because src.generators.generator_constructor could not be imported.")
        return

    # Term builder
    with st.expander("➕ Add Generator Term", expanded=True):
        cols = st.columns(5)
        with cols[0]:
            deriv_choice = st.selectbox(
                "Derivative Order",
                ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","k (symbolic)"],
                index=0
            )
        with cols[1]:
            d_types = [t.value for t in DerivativeType]
            func_type_val = st.selectbox("Function Type", d_types, index=0)
        with cols[2]:
            coefficient = st.number_input("Coefficient", value=1.0, step=0.1)
        with cols[3]:
            power = st.number_input("Power (integer)", value=1, min_value=1, max_value=12, step=1)
        with cols[4]:
            op_types = [t.value for t in OperatorType]
            op_type_val = st.selectbox("Operator Type", op_types, index=0)

        cols2 = st.columns(3)
        with cols2[0]:
            scaling = st.text_input("Scaling a (for delay/advance)", value="")
        with cols2[1]:
            shift = st.text_input("Shift b (for delay/advance)", value="")
        with cols2[2]:
            add_clicked = st.button("➕ Add Term", type="primary", use_container_width=True)

        if add_clicked:
            try:
                d_order = "k" if "k" in deriv_choice else int(deriv_choice)
                term = DerivativeTerm(
                    derivative_order=d_order,              # int or 'k'
                    coefficient=float(coefficient),
                    power=int(power),
                    function_type=DerivativeType(func_type_val),
                    operator_type=OperatorType(op_type_val),
                    scaling=float(scaling) if scaling.strip() else None,
                    shift=float(shift) if shift.strip() else None,
                )
                st.session_state.generator_terms.append(term)
                st.success(f"Added term: {getattr(term,'get_description', lambda:'term')()}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to create term: {e}")

    # Show terms
    if st.session_state.generator_terms:
        st.subheader("📝 Current Terms")
        for i, t in enumerate(st.session_state.generator_terms, 1):
            c1, c2 = st.columns([8,1])
            with c1:
                desc = getattr(t, "get_description", lambda: f"{t}")()
                st.markdown(f'<div class="generator-term"><strong>Term {i}:</strong> {desc}</div>', unsafe_allow_html=True)
            with c2:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.generator_terms.pop(i-1)
                    st.experimental_rerun()

        # Build specification
        if st.button("🔨 Build Generator Specification", type="primary", use_container_width=True):
            try:
                spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Generator {len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = spec
                st.markdown('<div class="result-box"><b>✅ Generator specification created!</b></div>', unsafe_allow_html=True)
                # Try to display a symbolic LHS
                try:
                    y = sp.Function("y")
                    x = sp.Symbol("x", real=True)
                    lhs_expr = apply_generator_to(y(x), spec, x)
                    st.latex(latex_cleanup(lhs_expr))
                except Exception:
                    st.info("Built specification, but could not render symbolic LHS.")
            except Exception as e:
                st.error(f"Specification failed: {e}")

        # Generate ODE from Master Theorem + RHS = L[y]
        st.subheader("🎯 Generate ODE with Exact Solution (Theorem 4.1)")
        left, right = st.columns(2)

        with left:
            lib_choice = st.selectbox("Function Library", ["Basic", "Special"])
            if lib_choice == "Basic" and st.session_state.basic_functions:
                f_names = st.session_state.basic_functions.get_function_names()
                source_lib = st.session_state.basic_functions
            elif lib_choice == "Special" and st.session_state.special_functions:
                f_names = st.session_state.special_functions.get_function_names()
                source_lib = st.session_state.special_functions
            else:
                f_names = ["exponential", "sin", "cos", "polynomial", "log1p"]
                source_lib = None
            func_name = st.selectbox("Choose f(z)", f_names, index=0)

            st.markdown("**Master Theorem Parameters**")
            alpha = st.number_input("α", value=1.0, step=0.1)
            beta  = st.number_input("β", value=1.0, step=0.1, min_value=0.0)
            n     = st.number_input("n (positive integer)", value=1, min_value=1, step=1)
            M     = st.number_input("M", value=0.0, step=0.1)

            numeric_preview = st.checkbox("Numeric preview mode (evaluate constants)", value=False)

        with right:
            if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
                with st.spinner("Applying Master Theorem 4.1 and constructing RHS..."):
                    try:
                        # Resolve f(z)
                        f_expr = get_function_expr(source_lib, func_name)
                        x = sp.Symbol("x", real=True)
                        α, β, n_sym, 𝑀 = ensure_symbolic_params(alpha, beta, int(n), M)

                        # Preferred: use src MasterTheoremSolver if available
                        solution = None
                        if st.session_state.theorem_solver and MasterTheoremParameters:
                            try:
                                params = MasterTheoremParameters(f_z=f_expr, alpha=α, beta=β, n=n_sym, M=𝑀)
                                mt_res = st.session_state.theorem_solver.apply_theorem_4_1(
                                    st.session_state.current_generator, params
                                )
                                solution = mt_res.get("solution", None)
                            except Exception as e:
                                logger.warning(f"apply_theorem_4_1 failed, using fallback: {e}")

                        if solution is None:
                            solution = theorem_4_1_solution_expr(f_expr, α, β, n_sym, 𝑀, x)

                        # RHS = L[y]
                        gen_spec = st.session_state.get("current_generator")
                        if gen_spec:
                            try:
                                rhs = apply_generator_to(solution, gen_spec, x)
                                generator_lhs = apply_generator_to(sp.Function("y")(x), gen_spec, x)  # formal LHS form
                            except Exception as e:
                                logger.warning(f"Failed to apply generator to solution, fallback simple RHS: {e}")
                                generator_lhs = sp.Symbol("L[y]")
                                rhs = sp.Integer(0)
                        else:
                            generator_lhs = sp.Symbol("L[y]")
                            rhs = sp.Integer(0)

                        # Optional numeric preview
                        if numeric_preview:
                            solution = sp.N(solution, 10)
                            rhs = sp.N(rhs, 10)
                            generator_lhs = sp.N(generator_lhs, 10)

                        # Classification
                        classification = {}
                        if st.session_state.current_generator:
                            classification = {
                                "type": "Linear" if getattr(st.session_state.current_generator, "is_linear", False) else "Nonlinear",
                                "order": getattr(st.session_state.current_generator, "order", None),
                                "field": "Mathematical Physics",
                                "applications": ["Research Equation"]
                            }

                        # initial conditions example: y(0)
                        try:
                            y0 = sp.simplify(solution.subs(x, 0))
                        except Exception:
                            y0 = None

                        result = {
                            "generator": generator_lhs,
                            "solution": solution,
                            "rhs": rhs,
                            "parameters": {"alpha": α, "beta": β, "n": n_sym, "M": 𝑀},
                            "function_used": func_name,
                            "type": classification.get("type", "Unknown"),
                            "order": classification.get("order", 0),
                            "classification": classification,
                            "initial_conditions": {"y(0)": y0} if y0 is not None else {},
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "generator_number": len(st.session_state.generated_odes) + 1,
                        }

                        st.session_state.generated_odes.append(result)

                        st.markdown('<div class="result-box"><b>✅ ODE Generated Successfully!</b></div>', unsafe_allow_html=True)

                        tabs = st.tabs(["📐 Equation", "💡 Solution", "🏷️ Classification", "📤 Export"])
                        with tabs[0]:
                            st.markdown("### ODE")
                            st.latex(f"{latex_cleanup(generator_lhs)} = {latex_cleanup(rhs)}")
                        with tabs[1]:
                            st.markdown("### Exact Solution")
                            st.latex(f"y(x) = {latex_cleanup(solution)}")
                            if y0 is not None:
                                st.markdown("**Initial condition sample:**")
                                st.latex(f"y(0) = {latex_cleanup(y0)}")
                        with tabs[2]:
                            st.json(classification)
                        with tabs[3]:
                            latex_doc = LaTeXExporter.generate_document(result, include_preamble=True)
                            st.download_button(
                                "📄 Download LaTeX Document",
                                latex_doc,
                                file_name="ode_solution.tex",
                                mime="text/x-latex",
                                use_container_width=True
                            )
                            pkg = LaTeXExporter.create_package(result)
                            st.download_button(
                                "📦 Download Complete Package (ZIP)",
                                pkg,
                                file_name=f"ode_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Error generating ODE: {e}")
                        logger.error(traceback.format_exc())

    # Clear
    if st.button("🗑️ Clear All Terms", use_container_width=True):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None
        st.success("Cleared.")
        st.experimental_rerun()

def master_theorem_page():
    st.header("🎯 Apply Master Theorem")
    if not st.session_state.generator_terms:
        st.warning("Please construct a generator first in **Generator Constructor**.")
        return

    theorem_choice = st.selectbox("Theorem Version", ["Standard (4.1)", "Extended (4.2)"])

    c1, c2 = st.columns(2)
    with c1:
        lib_choice = st.selectbox("Function Library", ["Basic", "Special"])
        if lib_choice == "Basic" and st.session_state.basic_functions:
            f_names = st.session_state.basic_functions.get_function_names()
            source_lib = st.session_state.basic_functions
        elif lib_choice == "Special" and st.session_state.special_functions:
            f_names = st.session_state.special_functions.get_function_names()
            source_lib = st.session_state.special_functions
        else:
            f_names = ["exponential", "sin", "cos", "polynomial", "log1p"]
            source_lib = None
        func_name = st.selectbox("f(z)", f_names)

    with c2:
        alpha = st.number_input("α", value=1.0, step=0.1)
        beta  = st.number_input("β", value=1.0, step=0.1, min_value=0.0)
        n     = st.number_input("n (positive integer)", value=1, min_value=1, step=1)
        M     = st.number_input("M", value=0.0, step=0.1)

    if st.button(f"Apply Theorem {theorem_choice}", type="primary", use_container_width=True):
        try:
            f_expr = get_function_expr(source_lib, func_name)
            α, β, n_sym, 𝑀 = ensure_symbolic_params(alpha, beta, int(n), M)
            x = sp.Symbol("x", real=True)

            solver = st.session_state.theorem_solver
            gen_spec = st.session_state.get("current_generator")

            if theorem_choice.startswith("Standard") and solver and MasterTheoremParameters:
                try:
                    params = MasterTheoremParameters(f_z=f_expr, alpha=α, beta=β, n=n_sym, M=𝑀)
                    res = solver.apply_theorem_4_1(gen_spec, params)
                    y_expr = res.get("solution", None)
                    if y_expr is None:
                        y_expr = theorem_4_1_solution_expr(f_expr, α, β, n_sym, 𝑀, x)
                except Exception:
                    y_expr = theorem_4_1_solution_expr(f_expr, α, β, n_sym, 𝑀, x)
            elif theorem_choice.startswith("Extended") and solver and hasattr(solver, "apply_theorem_4_2"):
                try:
                    params = MasterTheoremParameters(f_z=f_expr, alpha=α, beta=β, n=n_sym, M=𝑀)
                    res = solver.apply_theorem_4_2(gen_spec, params)
                    y_expr = res.get("solution", None)
                    if y_expr is None:
                        # fallback: differentiate Theorem 4.1 result if needed
                        y_expr = theorem_4_1_solution_expr(f_expr, α, β, n_sym, 𝑀, x)
                except Exception:
                    y_expr = theorem_4_1_solution_expr(f_expr, α, β, n_sym, 𝑀, x)
            else:
                y_expr = theorem_4_1_solution_expr(f_expr, α, β, n_sym, 𝑀, x)

            st.success("✅ Theorem applied.")
            st.latex(f"y(x) = {latex_cleanup(y_expr)}")

            # Try verification: compute L[y]
            try:
                rhs = apply_generator_to(y_expr, gen_spec, x)
                st.markdown("**Verification (formal):**")
                st.latex(f"{latex_cleanup(apply_generator_to(sp.Function('y')(x), gen_spec, x))} = {latex_cleanup(rhs)}")
            except Exception as e:
                st.info(f"Verification skipped (could not apply generator): {e}")
        except Exception as e:
            st.error(f"Error: {e}")

def safe_create_trainer(model_type: str, learning_rate: float, device: str):
    """
    Instantiate MLTrainer with flexible signature.
    """
    if MLTrainer is None:
        return None
    try:
        return MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
    except TypeError:
        try:
            return MLTrainer(model_type=model_type)
        except TypeError:
            try:
                return MLTrainer()
            except Exception:
                return None

def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning")
    st.markdown('<div class="ml-box">The ML system learns generator patterns to synthesize new ODEs.</div>', unsafe_allow_html=True)

    c = st.columns(4)
    with c[0]:
        st.metric("Patterns", len(st.session_state.generator_patterns))
    with c[1]:
        st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c[2]:
        st.metric("Training Epochs", len(st.session_state.training_history.get("train_loss", [])))
    with c[3]:
        st.metric("Model", "Trained" if st.session_state.ml_trained else "Not Trained")

    model_type = st.selectbox("Model", ["pattern_learner", "vae", "transformer"], index=0)

    with st.expander("🎯 Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 10, 500, 100, step=10)
            batch_size = st.slider("Batch Size", 8, 256, 32, step=8)
        with c2:
            learning_rate = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            samples = st.slider("Training Samples", 100, 5000, 1000, step=100)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, step=0.05)
            use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate at least 5 ODEs first for a meaningful dataset.")
        return

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        device = "cuda" if (use_gpu and torch is not None and torch.cuda.is_available()) else "cpu"
        trainer = safe_create_trainer(model_type, learning_rate, device)
        if trainer is None:
            st.error("Failed to create MLTrainer (check src/ml/trainer.py).")
            return

        st.session_state.ml_trainer = trainer
        progress = st.progress(0)
        status = st.empty()

        def cb(epoch, total):
            p = int(100 * epoch / max(1, total))
            progress.progress(min(100, p))
            status.info(f"Epoch {epoch}/{total}")

        try:
            trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                samples=samples,
                validation_split=validation_split,
                progress_callback=cb
            )
            st.session_state.ml_trained = True
            st.session_state.training_history = getattr(trainer, "history", {})
            st.success("✅ Training complete.")
            # Plot history
            hist = st.session_state.training_history
            if hist.get("train_loss", None):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1, len(hist["train_loss"])+1)), y=hist["train_loss"], mode="lines", name="train_loss"))
                if hist.get("val_loss", None):
                    fig.add_trace(go.Scatter(x=list(range(1, len(hist["val_loss"])+1)), y=hist["val_loss"], mode="lines", name="val_loss"))
                fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss", height=380)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())

    if st.session_state.ml_trained and st.session_state.ml_trainer:
        st.subheader("🎨 Generate Novel ODEs")
        n_new = st.slider("Number to generate", 1, 10, 1)
        if st.button("🎲 Generate", type="primary", use_container_width=True):
            for i in range(n_new):
                try:
                    res = st.session_state.ml_trainer.generate_new_ode()
                    if res:
                        st.session_state.generated_odes.append(res)
                        with st.expander(f"Generated ODE {len(st.session_state.generated_odes)}"):
                            st.write(res.get("description", ""))
                            ode = res.get("ode", None)
                            if ode is not None:
                                try:
                                    st.latex(latex_cleanup(ode))
                                except Exception:
                                    st.code(str(ode))
                            st.json({k:res.get(k) for k in ("type","order","function_used")})
                    else:
                        st.info("No ODE returned by the generator.")
                except Exception as e:
                    st.warning(f"Generation {i+1} failed: {e}")

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs from factory families (linear/nonlinear).</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 1000, 50, step=5)
        gen_types = st.multiselect("Generator Families", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_cats = st.multiselect("Function categories", ["Basic","Special"], default=["Basic"])
        vary_params = st.checkbox("Vary α, β, n, M", value=True)
    with c3:
        if vary_params:
            a_rng = st.slider("α range", -10.0, 10.0, (-2.0, 2.0))
            b_rng = st.slider("β range", 0.1, 10.0, (0.5, 2.0))
            n_rng = st.slider("n range", 1, 5, (1, 3))
        else:
            a_rng = (1.0, 1.0)
            b_rng = (1.0, 1.0)
            n_rng = (1, 1)

    with st.expander("⚙️ Advanced"):
        export_format = st.selectbox("Default export format", ["JSON","CSV","LaTeX","All"], index=0)
        include_solutions = st.checkbox("Include solutions in the table (truncated)", value=True)
        include_classification = st.checkbox("Include classification fields", value=True)

    if st.button("🚀 Generate Batch", type="primary", use_container_width=True):
        if not (CompleteLinearGeneratorFactory and CompleteNonlinearGeneratorFactory):
            st.error("Complete*GeneratorFactory not available (check src/generators/master_generator.py).")
            return

        all_fn_names = []
        if "Basic" in func_cats and st.session_state.basic_functions:
            all_fn_names += list(st.session_state.basic_functions.get_function_names())
        if "Special" in func_cats and st.session_state.special_functions:
            all_fn_names += list(st.session_state.special_functions.get_function_names())[:16]
        if not all_fn_names:
            all_fn_names = ["exponential","sin","cos","polynomial"]

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i in range(num_odes):
            progress.progress(int((i+1)*100/num_odes))
            status.info(f"Generating {i+1}/{num_odes}")

            try:
                params = {
                    "alpha": float(np.random.uniform(*a_rng)),
                    "beta": float(np.random.uniform(*b_rng)),
                    "n": int(np.random.randint(n_rng[0], n_rng[1]+1)),
                    "M": float(np.random.uniform(-1, 1)),
                }
                fn = np.random.choice(all_fn_names)
                # we won't call f itself—factories use f internally by name or expr
                try:
                    f_expr = get_function_expr(st.session_state.basic_functions, fn)
                except Exception:
                    f_expr = get_function_expr(st.session_state.special_functions, fn)

                if "linear" in gen_types:
                    factory = CompleteLinearGeneratorFactory()
                    gen_num = int(np.random.randint(1, 9))
                else:
                    factory = CompleteNonlinearGeneratorFactory()
                    gen_num = int(np.random.randint(1, 11))

                # optional params per type (best-effort)
                if gen_num in (4,5):        # pantograph-like
                    params["a"] = float(np.random.uniform(1.2, 3.0))
                if gen_num in (1,2,4):      # q
                    params["q"] = int(np.random.randint(2,6))
                if gen_num in (2,3,5):      # v
                    params["v"] = int(np.random.randint(2,6))

                res = factory.create(gen_num, f_expr, **params)
                row = {
                    "ID": i+1,
                    "Type": res.get("type",""),
                    "Generator": res.get("generator_number", gen_num),
                    "Function": fn,
                    "Order": res.get("order",""),
                    "α": round(params["alpha"],3),
                    "β": round(params["beta"],3),
                    "n": params["n"]
                }
                if include_solutions:
                    sol = str(res.get("solution",""))
                    row["Solution"] = (sol[:120] + "…") if len(sol)>120 else sol
                if include_classification:
                    row["Subtype"] = res.get("subtype", "standard")
                results.append(row)
            except Exception as e:
                logger.debug(f"Batch item failed: {e}")

        st.session_state.batch_results.extend(results)
        st.success(f"✅ Generated {len(results)} ODEs.")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        st.subheader("📤 Export")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("CSV", csv, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        with c2:
            j = json.dumps(results, indent=2).encode("utf-8")
            st.download_button("JSON", j, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
        with c3:
            tex = generate_batch_latex(results)
            st.download_button("LaTeX", tex, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")
        with c4:
            if export_format == "All":
                pkg = create_batch_package(results, df)
                st.download_button("ZIP", pkg, file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

def novelty_detection_page():
    st.header("🔍 Novelty Detection")
    st.markdown('<div class="info-box">Transformer-based novelty detection for ODE structures.</div>', unsafe_allow_html=True)

    if not st.session_state.novelty_detector:
        st.error("ODENoveltyDetector is unavailable (check src/dl/novelty_detector.py).")
        return

    method = st.radio("Input", ["Use Current Generator", "Enter ODE Manually", "Select from Generated"])
    ode_to_analyze = None

    if method == "Use Current Generator":
        if st.session_state.current_generator:
            y = sp.Function("y")
            x = sp.Symbol("x", real=True)
            try:
                ode_to_analyze = {
                    "ode": str(apply_generator_to(y(x), st.session_state.current_generator, x)),
                    "type": "custom",
                    "order": getattr(st.session_state.current_generator, "order", 0)
                }
            except Exception:
                st.warning("Could not build ODE from current generator.")
        else:
            st.warning("No generator built yet.")
    elif method == "Enter ODE Manually":
        txt = st.text_area("Enter ODE (as text or LaTeX)", height=120)
        if txt.strip():
            ode_to_analyze = {"ode": txt, "type":"manual", "order": st.number_input("Order",1,15,2)}
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)), format_func=lambda i: f"ODE {i+1}")
            ode_to_analyze = st.session_state.generated_odes[idx]

    if ode_to_analyze and st.button("Analyze", type="primary"):
        try:
            analysis = st.session_state.novelty_detector.analyze(ode_to_analyze, check_solvability=True, detailed=True)
            st.metric("Novelty Score", f"{analysis.novelty_score:.1f}")
            st.metric("Confidence", f"{analysis.confidence:.1%}")
            st.write("**Status:**", "NOVEL" if analysis.is_novel else "STANDARD")
            with st.expander("Details", expanded=True):
                st.write("Complexity:", analysis.complexity_level)
                st.write("Solvable by standard methods:", "Yes" if analysis.solvable_by_standard_methods else "No")
                if getattr(analysis, "special_characteristics", None):
                    st.write("Special characteristics:")
                    for c in analysis.special_characteristics:
                        st.write("•", c)
                if getattr(analysis, "recommended_methods", None):
                    st.write("Recommended methods:")
                    for m in analysis.recommended_methods[:6]:
                        st.write("•", m)
                if getattr(analysis, "similar_known_equations", None):
                    st.write("Similar known equations:")
                    for s in analysis.similar_known_equations[:4]:
                        st.write("•", s)
            if getattr(analysis, "detailed_report", None):
                st.download_button(
                    "📥 Download Report",
                    analysis.detailed_report,
                    file_name=f"novelty_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"Analysis failed: {e}")

def analysis_classification_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    classifier = st.session_state.ode_classifier
    st.subheader("Overview")
    data = []
    for i, r in enumerate(st.session_state.generated_odes[-50:], 1):
        data.append({
            "ID": i, "Type": r.get("type",""), "Order": r.get("order",""),
            "Generator": r.get("generator_number",""), "Function": r.get("function_used",""),
            "Timestamp": r.get("timestamp","")[:19]
        })
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    c = st.columns(4)
    with c[0]:
        lin = sum(1 for r in st.session_state.generated_odes if str(r.get("type","")).lower()=="linear")
        st.metric("Linear", lin)
    with c[1]:
        nonlin = sum(1 for r in st.session_state.generated_odes if str(r.get("type","")).lower()=="nonlinear")
        st.metric("Nonlinear", nonlin)
    with c[2]:
        avg_order = np.mean([r.get("order",0) or 0 for r in st.session_state.generated_odes]) if st.session_state.generated_odes else 0
        st.metric("Avg order", f"{avg_order:.1f}")
    with c[3]:
        uniq_f = len({r.get("function_used","") for r in st.session_state.generated_odes})
        st.metric("Unique functions", uniq_f)

    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    with c1:
        orders = [r.get("order",0) or 0 for r in st.session_state.generated_odes]
        fig = px.histogram(orders, nbins=10, title="Order distribution")
        fig.update_layout(xaxis_title="Order", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        types = [r.get("type","Unknown") for r in st.session_state.generated_odes]
        vc = pd.Series(types).value_counts()
        fig = px.pie(values=vc.values, names=vc.index, title="Type distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏷️ Auto-classify (by field)")
    if classifier and st.button("Classify all", type="primary"):
        fields = []
        for ode in st.session_state.generated_odes:
            try:
                res = classifier.classify_ode(ode) or {}
                fields.append(res.get("classification", {}).get("field", "Unknown"))
            except Exception:
                fields.append("Unknown")
        vc = pd.Series(fields).value_counts()
        fig = px.bar(x=vc.index, y=vc.values, title="Classification by field")
        fig.update_layout(xaxis_title="Field", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    elif not classifier:
        st.info("ODEClassifier is unavailable.")

def physical_applications_page():
    st.header("🔬 Physical Applications")
    st.markdown('<div class="info-box">Explore how the generated ODEs relate to physics and engineering.</div>', unsafe_allow_html=True)

    cat = st.selectbox("Field", ["Mechanics","Quantum","Thermodynamics","Electromagnetism","Biology","Economics","Engineering"])
    apps = {
        "Mechanics": [
            {"name":"Harmonic oscillator", "equation":"y'' + ω^2 y = 0", "desc":"Spring-mass systems, pendulums"},
            {"name":"Damped oscillator", "equation":"y'' + 2γ y' + ω_0^2 y = 0", "desc":"Frictional oscillators"},
            {"name":"Forced oscillator", "equation":"y'' + 2γ y' + ω_0^2 y = F cos(ω t)", "desc":"Driven systems"},
        ],
        "Quantum": [
            {"name":"Time-independent Schrödinger", "equation":"-ℏ^2/(2m) y'' + V(x) y = E y", "desc":"Quantum stationary states"},
            {"name":"Particle in a box", "equation":"y'' + k^2 y = 0", "desc":"Confined particle"},
        ],
        "Thermodynamics": [
            {"name":"Newton cooling", "equation":"dT/dt = -k (T - T_env)", "desc":"Cooling processes"},
        ],
        "Electromagnetism": [
            {"name":"LC circuit", "equation":"L y'' + (1/C) y = 0", "desc":"Undriven LC oscillator"},
        ],
        "Biology": [
            {"name":"Logistic growth (ODE)", "equation":"y' = r y (1 - y/K)", "desc":"Population growth"},
        ],
        "Economics": [
            {"name":"Cobweb model (ODE form)", "equation":"y' = a y - b y(t-τ)", "desc":"Lagged supply-demand"},
        ],
        "Engineering": [
            {"name":"Pantograph-type", "equation":"y''(x) + y(x/a) - y(x) = RHS", "desc":"Delay/scale feedback"},
        ],
    }
    for item in apps.get(cat, []):
        with st.expander(f"📚 {item['name']}"):
            st.latex(item["equation"])
            st.write(item["desc"])

def visualization_page():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize yet.")
        return
    idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)), format_func=lambda i: f"ODE {i+1}")
    ode = st.session_state.generated_odes[idx]

    plot_type = st.selectbox("Plot Type", ["Solution", "Phase Portrait (1st-order systems only)", "Direction Field (1st-order)"])
    x_min, x_max = st.slider("x-range", -10.0, 10.0, (-5.0, 5.0))
    npts = st.slider("Points", 100, 2000, 500, step=50)

    if st.button("Generate Plot", type="primary"):
        try:
            x = sp.Symbol("x", real=True)
            y_expr = ode.get("solution", None)
            if y_expr is None:
                st.info("No solution stored for this ODE.")
                return

            xs = np.linspace(x_min, x_max, npts)
            if plot_type == "Solution":
                y_num = sp.lambdify(x, y_expr, "numpy")
                ys = y_num(xs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="y(x)"))
                fig.update_layout(title="y(x)", xaxis_title="x", yaxis_title="y")
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type.startswith("Phase"):
                # simple (y, y') portrait if possible
                yprime = sp.diff(y_expr, x)
                y_num = sp.lambdify(x, y_expr, "numpy")
                yp_num = sp.lambdify(x, yprime, "numpy")
                ys = y_num(xs)
                yps = yp_num(xs)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ys, y=yps, mode="lines", name="phase"))
                fig.update_layout(title="Phase portrait", xaxis_title="y", yaxis_title="y'")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Direction field: y' = f(x,y) unknown; we can approximate using y'(x) on solution
                yprime = sp.diff(y_expr, x)
                yp_num = sp.lambdify(x, yprime, "numpy")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=yp_num(xs), mode="lines", name="y'(x)"))
                fig.update_layout(title="Direction field proxy (y'(x))", xaxis_title="x", yaxis_title="y'")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization failed: {e}")

def export_latex_page():
    st.header("📤 Export & LaTeX")
    st.markdown('<div class="latex-export-box"><b>Publication-ready LaTeX export</b></div>', unsafe_allow_html=True)
    if not st.session_state.generated_odes:
        st.warning("No ODEs to export.")
        return
    mode = st.radio("Export mode", ["Single ODE", "Multiple ODEs", "Complete Report"])
    if mode == "Single ODE":
        idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)), format_func=lambda i: f"ODE {i+1}")
        ode = st.session_state.generated_odes[idx]
        preview = LaTeXExporter.generate_document(ode, include_preamble=False)
        st.code(preview, language="latex")
        doc = LaTeXExporter.generate_document(ode, include_preamble=True)
        st.download_button("📄 LaTeX", doc, file_name=f"ode_{idx+1}.tex", mime="text/x-latex")
        pkg = LaTeXExporter.create_package(ode)
        st.download_button("📦 Package (ZIP)", pkg, file_name=f"ode_package_{idx+1}.zip", mime="application/zip")
    elif mode == "Multiple ODEs":
        idxs = st.multiselect("Select ODEs", range(len(st.session_state.generated_odes)), format_func=lambda i: f"ODE {i+1}")
        if idxs and st.button("Generate document"):
            parts = [r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{geometry}\geometry{margin=1in}
\title{Collection of Generated ODEs}\author{Master Generators}\date{\today}
\begin{document}\maketitle
"""]
            for j, i in enumerate(idxs, 1):
                parts.append(f"\\section*{{ODE {j}}}")
                parts.append(LaTeXExporter.generate_document(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            doc = "\n".join(parts)
            st.download_button("📄 LaTeX (multi)", doc, file_name=f"multiple_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")
    else:
        if st.button("Generate complete report"):
            doc = generate_complete_report()
            st.download_button("📄 LaTeX", doc, file_name=f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")

def examples_library_page():
    st.header("📚 Examples Library")
    st.markdown("A few quick starters (strings fixed to avoid syntax errors).")

    examples = {
        "Linear Generators": [
            {
                "name": "Simple Harmonic Oscillator",
                "generator": "y'' + y = RHS",
                "parameters": {"alpha": 1, "beta": 1, "n": 1, "M": 0},
                "function": "sin",
                "desc": "Classic SHO"
            },
            {
                "name": "Pantograph Equation",
                "generator": "y''(x) + y(x/2) - y(x) = RHS",
                "parameters": {"alpha": 1, "beta": 1, "n": 1, "M": 0, "a": 2},
                "function": "polynomial",
                "desc": "Scale-delay"
            },
        ],
        "Nonlinear Generators": [
            {
                "name": "Power Nonlinearity",
                "generator": "(y'')^3 + y = RHS",
                "parameters": {"alpha": 1, "beta": 1, "n": 1, "M": 0, "q": 3},
                "function": "polynomial",
                "desc": "Cubic"
            },
            {
                "name": "Exponential Nonlinearity",
                # ⬇⬇ FIXED: avoid unmatched quotes like: exp(y'") + ...
                "generator": "exp(y'') + exp(y') = RHS",
                "parameters": {"alpha": 0, "beta": 1, "n": 1, "M": 0},
                "function": "exponential",
                "desc": "e^(y'') + e^(y')"
            },
        ],
        "Special Functions": [
            {
                "name": "Airy-type",
                "generator": "y'' - x y = RHS",
                "parameters": {"alpha": 0, "beta": 1, "n": 1, "M": 0},
                "function": "airy_ai",
                "desc": "Airy"
            },
            {
                "name": "Bessel-type",
                "generator": "x^2 y'' + x y' + (x^2 - n^2) y = RHS",
                "parameters": {"alpha": 0, "beta": 1, "n": 2, "M": 0},
                "function": "bessel_j0",
                "desc": "Bessel"
            },
        ],
    }

    cat = st.selectbox("Category", list(examples.keys()))
    for ex in examples[cat]:
        with st.expander(f"📖 {ex['name']}"):
            st.latex(ex["generator"])
            st.write(ex["desc"])
            st.write("**Parameters:**"); st.json(ex["parameters"])
            st.write(f"**Function:** f(z) = {ex['function']}")
            if st.button(f"Load {ex['name']}", key=ex["name"]):
                st.info("Loading example: add appropriate terms in the constructor as per this template.")

def settings_page():
    st.header("⚙️ Settings")
    with st.expander("General"):
        st.checkbox("Dark mode (use Streamlit theme)", value=False, disabled=True)
        st.checkbox("Auto-save generated ODEs (not implemented)", value=False, disabled=True)
    with st.expander("ML Defaults"):
        st.write("Set defaults in ML page directly.")
    with st.expander("Export"):
        st.write("LaTeX export options are on the Export page.")
    with st.expander("Diagnostics"):
        st.write("src availability:", HAVE_SRC)
        if IMPORT_ERRORS:
            st.write("Import issues:")
            for e in IMPORT_ERRORS:
                st.write("•", e)

def documentation_page():
    st.header("📖 Documentation")
    tabs = st.tabs(["Quick Start", "Mathematical Theory", "API Reference", "FAQ"])
    with tabs[0]:
        st.markdown("""
1. Go to **Generator Constructor** → add terms → build specification.  
2. Pick f(z), set (α, β, n, M) → **Generate ODE** (uses Theorem 4.1).  
3. Explore **ML Pattern Learning**, **Batch**, **Novelty Detection**.  
4. Export with **LaTeX**.
        """)
    with tabs[1]:
        st.markdown(r"""
**Theorem 4.1 (structure used here)**  
Let \( \omega_s = \frac{(2s-1)\pi}{2n} \). Then a solution form is
\[
y(x)=\frac{\pi}{2n}\sum_{s=1}^n \Big[ 2f(\alpha+\beta) - f\big(\alpha+\beta e^{i x\cos\omega_s - x \sin\omega_s}\big) - f\big(\alpha+\beta e^{-i x\cos\omega_s - x \sin\omega_s}\big) \Big] + \pi M.
\]
When your `src` implements `MasterTheoremSolver.apply_theorem_4_1`, the app uses it first; otherwise it falls back to the symbolic expression above.
        """)
    with tabs[2]:
        st.markdown("""
- `CompleteLinearGeneratorFactory`, `CompleteNonlinearGeneratorFactory` from `src.generators.master_generator`.  
- `LinearGeneratorFactory` / `NonlinearGeneratorFactory` (if needed).  
- `GeneratorConstructor`, `GeneratorSpecification`, `DerivativeTerm`, `DerivativeType`, `OperatorType`.  
- `MasterTheoremSolver`, `MasterTheoremParameters`.  
- `BasicFunctions`, `SpecialFunctions`.  
- `MLTrainer`, `ODENoveltyDetector`, `ODEClassifier`.  
        """)
    with tabs[3]:
        st.markdown("""
**Q:** Is the solution symbolic or numeric?  
**A:** Symbolic by default. Check “Numeric preview mode” for evaluated constants.

**Q:** Why might my earlier solution show numbers like 7.389…?  
**A:** That means the expression was numerically evaluated at some constants. The rewritten app avoids unintended evaluation.

**Q:** Can I use k-th derivative?  
**A:** Yes, as a *formal* term in the constructor. It's included in the display and classification but excluded from actual RHS application (since k is symbolic).
        """)

# ──────────────────────────────────────────────────────────────────────────────
# Batch export helpers
# ──────────────────────────────────────────────────────────────────────────────
def generate_batch_latex(rows: List[Dict[str, Any]]) -> str:
    parts = [r"\begin{tabular}{|c|c|c|c|c|}"]
    parts += [r"\hline", "ID & Type & Generator & Function & Order \\\\", r"\hline"]
    for r in rows[:40]:
        parts.append(f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\")
    parts += [r"\hline", r"\end{tabular}"]
    return "\n".join(parts)

def create_batch_package(results: List[Dict[str, Any]], df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("batch_results.csv", df.to_csv(index=False))
        z.writestr("batch_results.json", json.dumps(results, indent=2))
        z.writestr("batch_results.tex", generate_batch_latex(results))
        z.writestr("README.txt", f"Generated: {datetime.now().isoformat()}\nTotal: {len(results)}")
    buf.seek(0)
    return buf.getvalue()

def generate_complete_report() -> str:
    parts = [r"""\documentclass[12pt]{report}
\usepackage{amsmath,amssymb}
\usepackage{geometry}\geometry{margin=1in}
\title{Master Generators System - Complete Report}\author{Generated}\date{\today}
\begin{document}\maketitle\tableofcontents
\chapter{Summary}This report contains all ODEs generated.
\chapter{Generated ODEs}
"""]
    for i, ode in enumerate(st.session_state.generated_odes, 1):
        parts.append(f"\\section*{{ODE {i}}}")
        parts.append(LaTeXExporter.generate_document(ode, include_preamble=False))
    parts.append(r"\end{document}")
    return "\n".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
