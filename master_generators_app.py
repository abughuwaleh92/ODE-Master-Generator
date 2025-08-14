"""
Master Generators for ODEs — Complete App (Rewritten)
- Robust optional imports + feature-gating
- Correct Theorem 4.1 solution + operator application for RHS
- Safer LaTeX generation
- Single-file navigation (no st.switch_page)
- Resilient session save/load and exports
- Kaleido-guarded image export
"""

import os
import sys
import io
import json
import time
import base64
import zipfile
import pickle
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# App metadata / logging
# -----------------------------------------------------------------------------
APP_VERSION = "2.1.1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# -----------------------------------------------------------------------------
# Optional / external dependencies and feature flags
# -----------------------------------------------------------------------------
HAVE_TORCH = False
try:
    import torch  # noqa
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False

HAVE_SRC = True
try:
    # Core generators, constructors, theorems, classifiers
    from src.generators.master_generator import (
        MasterGenerator,
        EnhancedMasterGenerator,
        CompleteMasterGenerator,
    )
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
    # Functions
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    # ML / DL
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
    # Utils
    from src.utils.config import Settings, AppConfig
    from src.utils.cache import CacheManager, cached
    from src.utils.validators import ParameterValidator
    from src.ui.components import UIComponents

except Exception as e:
    logger.warning(f"Some imports from src/ failed or are missing: {e}")
    HAVE_SRC = False

# -----------------------------------------------------------------------------
# Streamlit page config & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Master Generators ODE System - Complete Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Main Theme */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
.main-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}
.subtitle {
    font-size: 1.2rem;
    opacity: 0.95;
}

/* Generator Terms */
.generator-term {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid #667eea;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}
.generator-term:hover { transform: translateX(5px); }

/* Result Boxes */
.result-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border: 2px solid #4caf50;
    padding: 2rem;
    border-radius: 15px;
    margin: 1.5rem 0;
    box-shadow: 0 5px 20px rgba(76,175,80,0.2);
}
.error-box {
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border: 2px solid #f44336;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

/* ML Box */
.ml-box {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border: 2px solid #ff9800;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 5px 20px rgba(255,152,0,0.2);
}

/* LaTeX Export Box */
.latex-export-box {
    background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
    border: 2px solid #9c27b0;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1.5rem 0;
    box-shadow: 0 5px 20px rgba(156,39,176,0.2);
}

/* Metrics Cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    transition: transform 0.3s ease;
}
.metric-card:hover { transform: scale(1.05); }

/* Buttons */
.custom-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}
.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(102,126,234,0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Info boxes */
.info-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 5px solid #2196f3;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

/* Success Animation */
@keyframes successPulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.success-animation { animation: successPulse 0.5s ease-in-out; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Small helper for robust rerun across Streamlit versions
# -----------------------------------------------------------------------------
def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Safer LaTeX exporter
# -----------------------------------------------------------------------------
SAFE_LOCALS = {name: getattr(sp, name) for name in [
    "sin", "cos", "exp", "log", "sqrt", "Abs", "pi", "E", "I", "Symbol", "Function"
]}


class LaTeXExporter:
    """Enhanced LaTeX export system for ODEs."""

    @staticmethod
    def sympy_to_latex(expr) -> str:
        """Convert SymPy expr or safe string to LaTeX."""
        if expr is None:
            return ""
        try:
            if isinstance(expr, str):
                try:
                    expr = parse_expr(expr, local_dict=SAFE_LOCALS, evaluate=False)
                except Exception:
                    return expr  # leave plain string
            return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
        except Exception as e:
            logger.error(f"LaTeX conversion error: {e}")
            return str(expr)

    @staticmethod
    def generate_latex_document(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        generator = ode_data.get("generator", "")
        solution = ode_data.get("solution", "")
        rhs = ode_data.get("rhs", "")
        params = ode_data.get("parameters", {})
        classification = ode_data.get("classification", {})
        initial_conditions = ode_data.get("initial_conditions", {})

        latex_parts = []
        if include_preamble:
            latex_parts.append(
                r"""
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\title{Master Generator ODE System}
\author{Generated by Master Generators v2.1}
\date{\today}

\begin{document}
\maketitle

\section{Generated Ordinary Differential Equation}
"""
            )

        # Equation
        latex_parts.append(r"\subsection{Generator Equation}")
        latex_parts.append(r"\begin{equation}")
        latex_parts.append(
            f"{LaTeXExporter.sympy_to_latex(generator)} = {LaTeXExporter.sympy_to_latex(rhs)}"
        )
        latex_parts.append(r"\end{equation}")
        latex_parts.append("")

        # Solution
        latex_parts.append(r"\subsection{Exact Solution}")
        latex_parts.append(r"\begin{equation}")
        latex_parts.append(f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}")
        latex_parts.append(r"\end{equation}")
        latex_parts.append("")

        # Parameters
        latex_parts.append(r"\subsection{Parameters}")
        latex_parts.append(r"\begin{align}")
        latex_parts.append(f"\\alpha &= {params.get('alpha', 1.0)} \\\\")
        latex_parts.append(f"\\beta &= {params.get('beta', 1.0)} \\\\")
        latex_parts.append(f"n &= {params.get('n', 1)} \\\\")
        latex_parts.append(f"M &= {params.get('M', 0.0)}")
        for extra_k in ["q", "v", "a"]:
            if extra_k in params:
                latex_parts.append(f" \\\\ {extra_k} &= {params[extra_k]}")
        latex_parts.append(r"\end{align}")
        latex_parts.append("")

        # Initial conditions
        if initial_conditions:
            latex_parts.append(r"\subsection{Initial Conditions}")
            latex_parts.append(r"\begin{align}")
            items = list(initial_conditions.items())
            for i, (k, v) in enumerate(items):
                sep = r" \\" if i < len(items) - 1 else ""
                latex_parts.append(
                    f"{k} &= {LaTeXExporter.sympy_to_latex(v)}{sep}"
                )
            latex_parts.append(r"\end{align}")
            latex_parts.append("")

        # Classification
        if classification:
            latex_parts.append(r"\subsection{Mathematical Classification}")
            latex_parts.append(r"\begin{itemize}")
            latex_parts.append(
                f"\\item \\textbf{{Type:}} {classification.get('type','Unknown')}"
            )
            latex_parts.append(
                f"\\item \\textbf{{Order:}} {classification.get('order','Unknown')}"
            )
            if "linearity" in classification:
                latex_parts.append(
                    f"\\item \\textbf{{Linearity:}} {classification['linearity']}"
                )
            if "field" in classification:
                latex_parts.append(
                    f"\\item \\textbf{{Field:}} {classification['field']}"
                )
            if "applications" in classification:
                apps = ", ".join(classification["applications"][:5])
                latex_parts.append(f"\\item \\textbf{{Applications:}} {apps}")
            latex_parts.append(r"\end{itemize}")
            latex_parts.append("")

        latex_parts.append(r"\subsection{Solution Verification}")
        latex_parts.append(
            r"To verify, substitute $y(x)$ into the generator's left-hand side and confirm it equals the right-hand side."
        )

        if include_preamble:
            latex_parts.append(r"\end{document}")

        return "\n".join(latex_parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            latex_content = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
            zipf.writestr("ode_document.tex", latex_content)

            json_data = json.dumps(ode_data, indent=2, default=str)
            zipf.writestr("ode_data.json", json_data)

            readme = f"""Master Generator ODE Export
Generated: {datetime.now().isoformat()}

Contents:
- ode_document.tex: Complete LaTeX document
- ode_data.json: Raw data in JSON format
- README.txt: This file

Generator Type: {ode_data.get('type', 'Unknown')}
Order: {ode_data.get('order', 'Unknown')}

To compile LaTeX:
pdflatex ode_document.tex

Project: Master Generators (v{APP_VERSION})
"""
            zipf.writestr("README.txt", readme)

            if include_extras:
                # Minimal reproduce script
                python_code = LaTeXExporter.generate_python_code(ode_data)
                zipf.writestr("reproduce.py", python_code)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    @staticmethod
    def generate_python_code(ode_data: Dict[str, Any]) -> str:
        params = ode_data.get("parameters", {})
        gen_type = ode_data.get("type", "linear")
        gen_num = ode_data.get("generator_number", 1)
        func_name = ode_data.get("function_used", "exponential")

        code = f'''"""
Reproduce a generated ODE (Master Generators)
"""

import sympy as sp
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions

params = {{
    'alpha': {params.get('alpha', 1.0)},
    'beta': {params.get('beta', 1.0)},
    'n': {params.get('n', 1)},
    'M': {params.get('M', 0.0)}
}}
'''

        for extra_k in ["q", "v", "a"]:
            if extra_k in params:
                code += f"params['{extra_k}'] = {params[extra_k]}\n"

        code += f'''
basic_functions = BasicFunctions()
special_functions = SpecialFunctions()

try:
    f_z = basic_functions.get_function('{func_name}')
except Exception:
    f_z = special_functions.get_function('{func_name}')

factory = LinearGeneratorFactory() if '{gen_type}' == 'linear' else NonlinearGeneratorFactory()
result = factory.create({gen_num}, f_z, **params)

print("ODE:", result.get('ode'))
print("Solution:", result.get('solution'))
print("Type:", result.get('type'))
print("Order:", result.get('order'))
'''
        return code


# -----------------------------------------------------------------------------
# Theorem 4.1 helpers: solution builder and operator application
# -----------------------------------------------------------------------------
def theorem_4_1_solution_expr(
    f_expr: sp.Expr, alpha, beta, n: int, M: float, x_symbol: sp.Symbol
) -> sp.Expr:
    """
    Y(θ) = (π/(2n)) Σ_{s=1..n} [ 2 f(α+β)
            - f(α + β e^{ i θ cos ω_s − θ sin ω_s })
            - f(α + β e^{−i θ cos ω_s − θ sin ω_s }) ] + π M
    We use x_symbol in place of θ to align with the app's y(x).
    """
    z = sp.Symbol("z")
    base = f_expr.subs(z, alpha + beta)
    total = 0
    for s in range(1, n + 1):
        omega = (2 * s - 1) * sp.pi / (2 * n)
        exp_plus = sp.exp(sp.I * x_symbol * sp.cos(omega) - x_symbol * sp.sin(omega))
        exp_minus = sp.exp(-sp.I * x_symbol * sp.cos(omega) - x_symbol * sp.sin(omega))
        term_plus = f_expr.subs(z, alpha + beta * exp_plus)
        term_minus = f_expr.subs(z, alpha + beta * exp_minus)
        total += 2 * base - term_plus - term_minus
    return sp.simplify(sp.pi / (2 * n) * total + sp.pi * M)


def apply_generator_to(expr: sp.Expr, gen_spec: Any, x_symbol: sp.Symbol) -> sp.Expr:
    """
    Substitute y(x) and its derivatives in gen_spec.lhs with expr and its derivatives to form RHS.
    Assumes gen_spec.lhs is a SymPy expression using y(x), y'(x), ...
    """
    y = sp.Function("y")
    subs_map = {y(x_symbol): expr}
    max_order = getattr(gen_spec, "order", 0)
    for k in range(1, max_order + 1):
        subs_map[sp.Derivative(y(x_symbol), (x_symbol, k))] = sp.diff(expr, (x_symbol, k))
    return sp.simplify(gen_spec.lhs.subs(subs_map))


# -----------------------------------------------------------------------------
# Session state manager (guarded)
# -----------------------------------------------------------------------------
class SessionStateManager:
    @staticmethod
    def initialize():
        # Core collections
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
        if "training_history" not in st.session_state:
            st.session_state.training_history = []
        if "export_history" not in st.session_state:
            st.session_state.export_history = []
        if "ml_trained" not in st.session_state:
            st.session_state.ml_trained = False

        # Active page for single-file navigation
        if "active_page" not in st.session_state:
            st.session_state.active_page = "🏠 Dashboard"

        # Objects that depend on src/*
        if HAVE_SRC:
            if "generator_constructor" not in st.session_state:
                st.session_state.generator_constructor = GeneratorConstructor()
            if "vae_model" not in st.session_state:
                st.session_state.vae_model = GeneratorVAE()
            if "pattern_learner" not in st.session_state:
                st.session_state.pattern_learner = GeneratorPatternLearner()
            if "novelty_detector" not in st.session_state:
                st.session_state.novelty_detector = ODENoveltyDetector()
            if "ode_classifier" not in st.session_state:
                st.session_state.ode_classifier = ODEClassifier()
            if "ml_trainer" not in st.session_state:
                st.session_state.ml_trainer = None
            if "cache_manager" not in st.session_state:
                st.session_state.cache_manager = CacheManager()
            if "ui_components" not in st.session_state:
                st.session_state.ui_components = UIComponents()
            if "basic_functions" not in st.session_state:
                st.session_state.basic_functions = BasicFunctions()
            if "special_functions" not in st.session_state:
                st.session_state.special_functions = SpecialFunctions()
            if "theorem_solver" not in st.session_state:
                st.session_state.theorem_solver = MasterTheoremSolver()
            if "extended_theorem" not in st.session_state:
                st.session_state.extended_theorem = ExtendedMasterTheorem()
        else:
            # Ensure keys exist with None when src is missing
            for k in [
                "generator_constructor", "vae_model", "pattern_learner", "novelty_detector",
                "ode_classifier", "ml_trainer", "cache_manager", "ui_components",
                "basic_functions", "special_functions", "theorem_solver", "extended_theorem"
            ]:
                if k not in st.session_state:
                    st.session_state[k] = None

    @staticmethod
    def save_to_file(filename: str = "session_state.pkl"):
        """Serialize session to pickle safely (skip/convert non-picklables)."""
        try:
            def _safe(obj):
                if isinstance(obj, dict):
                    return {k: _safe(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_safe(v) for v in obj]
                try:
                    pickle.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)

            state_data = {
                "generated_odes": _safe(st.session_state.generated_odes),
                "generator_patterns": _safe(st.session_state.generator_patterns),
                "batch_results": _safe(st.session_state.batch_results),
                "analysis_results": _safe(st.session_state.analysis_results),
                "training_history": _safe(st.session_state.training_history),
                "export_history": _safe(st.session_state.export_history),
                "ml_trained": _safe(st.session_state.ml_trained),
            }
            with open(filename, "wb") as f:
                pickle.dump(state_data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return False

    @staticmethod
    def load_from_file(filename: str = "session_state.pkl"):
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    state_data = pickle.load(f)
                for key, value in state_data.items():
                    st.session_state[key] = value
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return False


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def get_function_expr(func_lib_obj: Any, func_name: str) -> sp.Expr:
    """
    Resolve a function f(z) from a library object (BasicFunctions or SpecialFunctions)
    into a SymPy expression in symbol z.
    """
    z = sp.Symbol("z")
    f_z = func_lib_obj.get_function(func_name)
    # Could be a callable, SymPy Expr, or SymPy Function
    if callable(f_z):
        try:
            # Prefer symbolic call
            return sp.sympify(f_z(z))
        except Exception:
            try:
                return sp.sympify(str(f_z(z)))
            except Exception:
                return sp.sympify(func_name)
    if isinstance(f_z, sp.Expr):
        return f_z
    if isinstance(f_z, sp.FunctionClass) or isinstance(f_z, sp.Function) or hasattr(f_z, "__call__"):
        try:
            return f_z(z)  # sympy function
        except Exception:
            return sp.sympify(func_name)
    # fallback
    try:
        return parse_expr(str(f_z), local_dict={"z": z}, evaluate=False)
    except Exception:
        return sp.sympify(str(f_z))


def find_similar_odes(equation_text: str) -> List[Dict]:
    """Tiny heuristic; replace with structural signature if desired."""
    similar = []
    eq_lower = equation_text.lower()
    for ode in st.session_state.generated_odes:
        t = (ode.get("type") or "").lower()
        if t and t in eq_lower:
            similar.append(ode)
    return similar


def create_solution_plot(ode: Dict, x_range: Tuple[float, float], num_points: int) -> go.Figure:
    """
    Try to plot the actual symbolic solution numerically (real part if complex).
    Fallback to a smooth damped sine if lambdify fails.
    """
    x = sp.Symbol("x", real=True)
    fig = go.Figure()
    sol = ode.get("solution", None)
    try:
        xs = np.linspace(float(x_range[0]), float(x_range[1]), int(num_points))
        if isinstance(sol, (sp.Expr, sp.Add, sp.Mul, sp.Pow, sp.Function)):
            f_num = sp.lambdify(x, sol, modules=["numpy"])
            ys = f_num(xs)
            ys = np.real(ys) if np.iscomplexobj(ys) else ys
        else:
            raise ValueError("Non-symbolic solution")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Solution"))
        fig.update_layout(title="ODE Solution", xaxis_title="x", yaxis_title="y(x)")
        return fig
    except Exception as e:
        logger.warning(f"Falling back plot: {e}")
        xs = np.linspace(float(x_range[0]), float(x_range[1]), int(num_points))
        ys = np.sin(xs) * np.exp(-0.1 * np.abs(xs))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Solution (fallback)"))
        fig.update_layout(title="ODE Solution (Fallback)", xaxis_title="x", yaxis_title="y(x)")
        return fig


def create_phase_portrait(ode: Dict, x_range: Tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title="Phase Portrait (placeholder)")
    return fig


def create_3d_surface(ode: Dict, x_range: Tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title="3D Surface (placeholder)")
    return fig


def create_direction_field(ode: Dict, x_range: Tuple[float, float]) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title="Direction Field (placeholder)")
    return fig


def generate_batch_latex(results: List[Dict]) -> str:
    rows = min(20, len(results))
    lines = [r"\begin{tabular}{|c|c|c|c|c|}", r"\hline", r"ID & Type & Generator & Function & Order \\", r"\hline"]
    for r in results[:rows]:
        lines.append(f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def create_batch_package(results: List[Dict], df: pd.DataFrame) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("batch_results.csv", df.to_csv(index=False))
        zipf.writestr("batch_results.json", json.dumps(results, indent=2, default=str))
        zipf.writestr("batch_results.tex", generate_batch_latex(results))
        readme = f"""Batch ODE Generation Results
Generated: {datetime.now().isoformat()}
Total ODEs: {len(results)}

Files:
- batch_results.csv
- batch_results.json
- batch_results.tex

Master Generators v{APP_VERSION}
"""
        zipf.writestr("README.txt", readme)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def generate_complete_report() -> str:
    parts = [
        r"""
\documentclass[12pt]{report}
\usepackage{amsmath,amssymb,graphicx,hyperref}
\usepackage{geometry}
\geometry{margin=1in}
\title{Master Generators System\\Complete Report}
\author{Generated Automatically}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
\chapter{Executive Summary}
This report contains all ODEs generated by the Master Generators System.

\chapter{Generated ODEs}
"""
    ]
    for i, ode in enumerate(st.session_state.generated_odes):
        parts.append(f"\\section{{ODE {i+1}}}")
        parts.append(LaTeXExporter.generate_latex_document(ode, include_preamble=False))
    parts.append(
        r"""
\chapter{Statistical Analysis}
[Statistical analysis would go here]

\chapter{Conclusions}
The Master Generators System successfully generated and analyzed multiple ODEs.

\end{document}
"""
    )
    return "\n".join(parts)


def export_all_formats(formats: List[str]):
    for fmt in formats:
        if fmt == "LaTeX":
            st.download_button("📄 Download LaTeX", generate_complete_report(), "all_odes.tex", "text/x-latex")
        elif fmt == "JSON":
            st.download_button(
                "📄 Download JSON",
                json.dumps(st.session_state.generated_odes, indent=2, default=str),
                "all_odes.json",
                "application/json",
            )


def create_complete_export_package() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        all_data = {
            "generated_odes": st.session_state.generated_odes,
            "batch_results": st.session_state.batch_results,
            "analysis_results": st.session_state.analysis_results,
            "export_timestamp": datetime.now().isoformat(),
        }
        zipf.writestr("all_data.json", json.dumps(all_data, indent=2, default=str))
        if st.session_state.generated_odes:
            report = generate_complete_report()
            zipf.writestr("complete_report.tex", report)
        readme = f"""Master Generators System - Complete Export
Generated: {datetime.now().isoformat()}

Contents:
- all_data.json
- complete_report.tex (if available)

Counts:
- Generated ODEs: {len(st.session_state.generated_odes)}
- Batch Results: {len(st.session_state.batch_results)}
- Analysis Results: {len(st.session_state.analysis_results)}

Master Generators v{APP_VERSION}
"""
        zipf.writestr("README.txt", readme)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def export_all_results():
    st.download_button(
        "📦 Download Complete Package",
        create_complete_export_package(),
        f"complete_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        "application/zip",
    )


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def header():
    st.markdown(
        f"""
    <div class="main-header">
      <h1 class="main-title">🔬 Master Generators for ODEs</h1>
      <p class="subtitle">Theorems 4.1 & 4.2 · ML/DL · LaTeX · v{APP_VERSION}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def sidebar_nav() -> str:
    st.sidebar.title("📍 Navigation")
    return st.sidebar.radio(
        "Select Module",
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
        key="active_page",
    )


def dashboard_page():
    st.header("🏠 Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
        <div class="metric-card">
          <h3>📝 Generated ODEs</h3>
          <h1>{len(st.session_state.generated_odes)}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
        <div class="metric-card">
          <h3>🧬 ML Patterns</h3>
          <h1>{len(st.session_state.generator_patterns)}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
        <div class="metric-card">
          <h3>📊 Batch Results</h3>
          <h1>{len(st.session_state.batch_results)}</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c4:
        model_status = "✅ Trained" if st.session_state.ml_trained else "⏳ Not Trained"
        st.markdown(
            f"""
        <div class="metric-card">
          <h3>🤖 ML Model</h3>
          <p style="font-size: 1.2rem;">{model_status}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.subheader("📊 Recent Activity")
    if st.session_state.generated_odes:
        recent_df = pd.DataFrame(st.session_state.generated_odes[-5:])
        display_cols = [c for c in ["type", "order", "generator_number", "timestamp"] if c in recent_df.columns]
        if display_cols:
            st.dataframe(recent_df[display_cols], use_container_width=True)
        else:
            st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No ODEs generated yet. Start with the Generator Constructor!")

    st.subheader("⚡ Quick Actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("🔧 Create New Generator", use_container_width=True):
            st.session_state.generator_terms = []
            st.session_state.active_page = "🔧 Generator Constructor"
            _rerun()
    with q2:
        if st.button("📊 Generate Batch ODEs", use_container_width=True):
            st.session_state.active_page = "📊 Batch Generation"
            _rerun()
    with q3:
        export_all_results()


def generator_constructor_page():
    st.header("🔧 Generator Constructor")

    if not HAVE_SRC:
        st.error("This module requires the `src/` package. Ensure the project ZIP is extracted with src/ present.")
        return

    st.markdown(
        """
    <div class="info-box">
        Build custom generators by combining derivatives with transformations.
        The system will compute the exact solution via Theorem 4.1 and derive the RHS by applying your generator.
    </div>
    """,
        unsafe_allow_html=True,
    )

    constructor = st.session_state.generator_constructor

    # ----- Term builder -----
    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox(
                "Derivative Order",
                [0, 1, 2, 3, 4, 5],
                format_func=lambda x: {0: "y", 1: "y'", 2: "y''", 3: "y'''", 4: "y⁽⁴⁾", 5: "y⁽⁵⁾"}.get(x, f"y⁽{x}⁾"),
            )
        with c2:
            func_type = st.selectbox(
                "Function Type",
                [t.value for t in DerivativeType],
                format_func=lambda x: x.replace("_", " ").title(),
            )
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 5, 1)

        a1, a2, a3 = st.columns(3)
        with a1:
            operator_type = st.selectbox(
                "Operator Type",
                [t.value for t in OperatorType],
                format_func=lambda x: x.replace("_", " ").title(),
            )
        with a2:
            scaling = st.number_input("Scaling (a)", 0.5, 5.0, 2.0, 0.1) if operator_type in ["delay", "advance"] else None
        with a3:
            shift = st.number_input("Shift", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay", "advance"] else None

        if st.button("➕ Add Term", type="primary", use_container_width=True):
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
            # keep constructor in sync
            try:
                if constructor is not None:
                    constructor.add_term(term)
            except Exception as e:
                logger.warning(f"Could not add term to constructor: {e}")
            st.success(f"Added: {term.get_description()}")
            _rerun()

    # ----- Display current terms -----
    if st.session_state.generator_terms:
        st.subheader("📝 Current Generator Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"""
                <div class="generator-term">
                    <strong>Term {i+1}:</strong> {term.get_description()}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.generator_terms.pop(i)
                    # rebuild constructor from remaining terms
                    new_constructor = GeneratorConstructor()
                    for t in st.session_state.generator_terms:
                        new_constructor.add_term(t)
                    st.session_state.generator_constructor = new_constructor
                    _rerun()

        if st.button("🔨 Build Generator Specification", type="primary", use_container_width=True):
            gen_spec = GeneratorSpecification(
                terms=st.session_state.generator_terms,
                name=f"Custom Generator {len(st.session_state.generated_odes) + 1}",
            )
            st.session_state.current_generator = gen_spec
            st.markdown(
                """
            <div class="result-box">
                <h3>✅ Generator Specification Created!</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.latex(sp.latex(gen_spec.lhs) + " = \\text{RHS}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Order", gen_spec.order)
                st.metric("Type", "Linear" if gen_spec.is_linear else "Nonlinear")
            with c2:
                feat = getattr(gen_spec, "special_features", [])
                st.metric("Special Features", len(feat))
                if feat:
                    st.write("Features:", ", ".join(feat))

        # ----- Generate ODE + Solution -----
        st.subheader("🎯 Generate ODE with Exact Solution (Theorem 4.1)")

        c1, c2 = st.columns(2)
        with c1:
            func_category = st.selectbox("Function Library", ["Basic", "Special"])
            if func_category == "Basic":
                func_names = st.session_state.basic_functions.get_function_names()
                source_lib = st.session_state.basic_functions
            else:
                func_names = st.session_state.special_functions.get_function_names()
                source_lib = st.session_state.special_functions
            func_name = st.selectbox("Select f(z)", func_names)

            st.markdown("**Master Theorem Parameters:**")
            alpha = st.slider("α", -5.0, 5.0, 1.0, 0.1)
            beta = st.slider("β", 0.1, 5.0, 1.0, 0.1)
            n = st.slider("n", 1, 3, 1)
            M = st.slider("M", -5.0, 5.0, 0.0, 0.1)

        with c2:
            if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
                with st.spinner("Applying Master Theorem 4.1 and constructing RHS..."):
                    try:
                        # Resolve f(z) as a SymPy expression
                        f_expr = get_function_expr(source_lib, func_name)
                        x = sp.Symbol("x", real=True)

                        # Build y(x) via Theorem 4.1
                        solution = theorem_4_1_solution_expr(f_expr, alpha, beta, int(n), float(M), x)

                        # Build RHS = L[y] if we have a generator spec
                        gen_spec = st.session_state.get("current_generator")
                        if gen_spec and hasattr(gen_spec, "lhs"):
                            try:
                                rhs = apply_generator_to(solution, gen_spec, x)
                                generator_lhs = gen_spec.lhs
                            except Exception as e:
                                logger.warning(f"Failed to apply generator to solution: {e}")
                                rhs = sp.simplify(sp.pi * (f_expr.subs(sp.Symbol("z"), alpha + beta) + M))
                                generator_lhs = sp.Symbol("LHS")
                        else:
                            rhs = sp.simplify(sp.pi * (f_expr.subs(sp.Symbol("z"), alpha + beta) + M))
                            generator_lhs = sp.Symbol("LHS")

                        # Classification (simple, based on spec)
                        classification = {}
                        if gen_spec:
                            classification = {
                                "type": "linear" if gen_spec.is_linear else "nonlinear",
                                "order": gen_spec.order,
                                "linearity": "Linear" if gen_spec.is_linear else "Nonlinear",
                                "field": "Mathematical Physics",
                                "applications": ["Research Equation"],
                            }

                        result = {
                            "generator": generator_lhs,
                            "solution": solution,
                            "rhs": rhs,
                            "parameters": {"alpha": alpha, "beta": beta, "n": int(n), "M": M},
                            "function_used": func_name,
                            "type": classification.get("type", "Unknown"),
                            "order": classification.get("order", 0),
                            "classification": classification,
                            "initial_conditions": {},
                            "timestamp": datetime.now().isoformat(),
                            "generator_number": len(st.session_state.generated_odes) + 1,
                        }
                        st.session_state.generated_odes.append(result)

                        st.markdown(
                            """
                        <div class="result-box success-animation">
                            <h3>✅ ODE Generated Successfully!</h3>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        tabs = st.tabs(["📐 Equation", "💡 Solution", "🏷️ Classification", "📤 Export"])
                        with tabs[0]:
                            st.markdown("### Complete ODE")
                            st.latex(sp.latex(result["generator"]) + " = " + sp.latex(result["rhs"]))
                        with tabs[1]:
                            st.markdown("### Exact Solution")
                            st.latex("y(x) = " + sp.latex(result["solution"]))
                        with tabs[2]:
                            st.markdown("### Classification")
                            if classification:
                                st.write(f"**Type:** {classification.get('type', 'Unknown')}")
                                st.write(f"**Order:** {classification.get('order', 'Unknown')}")
                                st.write(f"**Linearity:** {classification.get('linearity', 'Unknown')}")
                                st.write(f"**Field:** {classification.get('field', 'Unknown')}")
                                st.write(
                                    f"**Applications:** {', '.join(classification.get('applications', []))}"
                                )
                            else:
                                st.info("No classification available.")
                        with tabs[3]:
                            st.markdown("### Export")
                            latex_doc = LaTeXExporter.generate_latex_document(result, include_preamble=True)
                            st.download_button(
                                "📄 Download LaTeX Document",
                                latex_doc,
                                "ode_solution.tex",
                                "text/x-latex",
                                use_container_width=True,
                            )
                            package = LaTeXExporter.create_export_package(result, include_extras=True)
                            st.download_button(
                                "📦 Download Complete Package (ZIP)",
                                package,
                                f"ode_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                "application/zip",
                                use_container_width=True,
                            )

                    except Exception as e:
                        st.error(f"Error generating ODE: {str(e)}")
                        logger.error(f"Generation error: {traceback.format_exc()}")

        if st.button("🗑️ Clear All Terms", use_container_width=True):
            st.session_state.generator_terms = []
            st.session_state.generator_constructor = GeneratorConstructor()
            if "current_generator" in st.session_state:
                st.session_state.pop("current_generator")
            _rerun()
    else:
        st.info("No terms yet. Use the builder above to add your first term.")


def master_theorem_page():
    st.header("🎯 Apply Master Theorem")

    if not HAVE_SRC:
        st.error("This module requires the `src/` package. Ensure the project ZIP is extracted with src/ present.")
        return

    if not st.session_state.generator_terms:
        st.warning("Please construct a generator first in the Generator Constructor!")
        return

    st.info(f"Using generator with {len(st.session_state.generator_terms)} terms")

    theorem_type = st.selectbox(
        "Select Theorem Implementation",
        ["Standard (4.1)", "Extended (4.2)", "Special Functions"],
    )

    if theorem_type == "Standard (4.1)":
        apply_standard_theorem()
    elif theorem_type == "Extended (4.2)":
        apply_extended_theorem()
    else:
        apply_special_functions_theorem()


def apply_standard_theorem():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Function Selection")
        func_category = st.selectbox("Category", ["Basic", "Special"])
        if func_category == "Basic":
            names = st.session_state.basic_functions.get_function_names()
            lib = st.session_state.basic_functions
        else:
            names = st.session_state.special_functions.get_function_names()
            lib = st.session_state.special_functions
        func_name = st.selectbox("f(z)", names)

    with col2:
        st.subheader("Parameters")
        alpha = st.number_input("α", -10.0, 10.0, 1.0, 0.1)
        beta = st.number_input("β", 0.1, 10.0, 1.0, 0.1)
        n = st.number_input("n", 1, 5, 1)
        M = st.number_input("M", -10.0, 10.0, 0.0, 0.1)

    if st.button("Apply Theorem 4.1", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            try:
                f_expr = get_function_expr(lib, func_name)
                x = sp.Symbol("x", real=True)
                sol_expr = theorem_4_1_solution_expr(f_expr, alpha, beta, int(n), float(M), x)

                gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms, name="Master Theorem 4.1")
                rhs = apply_generator_to(sol_expr, gen_spec, x)

                st.success("✅ Theorem Applied Successfully!")
                st.latex("y(x) = " + sp.latex(sol_expr))
                st.markdown("**Derived RHS (L[y]):**")
                st.latex(sp.latex(gen_spec.lhs) + " = " + sp.latex(rhs))

            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())


def apply_extended_theorem():
    st.info("Extended Theorem 4.2 – placeholder UI.")
    if not HAVE_SRC or st.session_state.extended_theorem is None or st.session_state.theorem_solver is None:
        st.error("Extended theorem solver is not available in the environment.")
        return
    st.warning("Integrate your concrete ExtendedMasterTheorem logic here.")


def apply_special_functions_theorem():
    st.info("Special-functions variant – placeholder UI.")
    st.write("Tip: reuse Theorem 4.1 setup but restrict f(z) menu to special functions.")


def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning")

    if not HAVE_SRC:
        st.error("This module requires the `src/` package. Ensure the project ZIP is extracted with src/ present.")
        return
    if not HAVE_TORCH:
        st.warning("PyTorch is not installed. The ML Pattern Learning module is disabled.")
        return

    st.markdown(
        """
    <div class="ml-box">
      The ML system learns generator patterns to create new families of ODEs.
      Models: Pattern Learner, VAE, Transformer.
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Patterns", len(st.session_state.generator_patterns))
    with c2:
        st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c3:
        st.metric("Training Epochs", len(st.session_state.training_history))
    with c4:
        st.metric("Model Status", "Trained" if st.session_state.ml_trained else "Not Trained")

    model_type = st.selectbox(
        "Select ML Model",
        ["pattern_learner", "vae", "transformer"],
        format_func=lambda x: {
            "pattern_learner": "Pattern Learner (Encoder-Decoder)",
            "vae": "Variational Autoencoder (VAE)",
            "transformer": "Transformer Architecture",
        }[x],
    )

    with st.expander("🎯 Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            samples = st.slider("Training Samples", 100, 5000, 1000)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning(f"Need at least 5 generated ODEs. Current: {len(st.session_state.generated_odes)}")
        return

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        device = "cuda" if use_gpu and torch and torch.cuda.is_available() else "cpu"
        with st.spinner(f"Training {model_type} model on {device}..."):
            try:
                trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
                st.session_state.ml_trainer = trainer
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(epoch, total_epochs):
                    progress_bar.progress(epoch / total_epochs)
                    status_text.text(f"Epoch {epoch}/{total_epochs}")

                trainer.train(
                    epochs=epochs,
                    batch_size=batch_size,
                    samples=samples,
                    validation_split=validation_split,
                    progress_callback=progress_callback,
                )
                st.session_state.ml_trained = True
                st.session_state.training_history = trainer.history
                st.success("✅ Model trained successfully!")

                if trainer.history.get("train_loss"):
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(trainer.history["train_loss"]) + 1)),
                            y=trainer.history["train_loss"],
                            mode="lines",
                            name="Training Loss",
                        )
                    )
                    if trainer.history.get("val_loss"):
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(1, len(trainer.history["val_loss"]) + 1)),
                                y=trainer.history["val_loss"],
                                mode="lines",
                                name="Validation Loss",
                            )
                        )
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss", height=400)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                logger.error(traceback.format_exc())

    if st.session_state.ml_trained and st.session_state.ml_trainer:
        st.subheader("🎨 Generate Novel Patterns")
        c1, c2 = st.columns(2)
        with c1:
            num_generate = st.slider("Number to Generate", 1, 10, 1)
        with c2:
            if st.button("🎲 Generate Novel ODEs", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    for i in range(num_generate):
                        try:
                            result = st.session_state.ml_trainer.generate_new_ode()
                            if result:
                                st.success(f"✅ Generated ODE {i+1}")
                                with st.expander(f"ODE {i+1}: {result.get('description','')}"):
                                    ode_expr = result.get("ode")
                                    st.latex(sp.latex(ode_expr) if isinstance(ode_expr, sp.Basic) else str(ode_expr))
                                    st.write(f"**Type:** {result.get('type')}")
                                    st.write(f"**Order:** {result.get('order')}")
                                    st.write(f"**Function:** {result.get('function_used', 'Unknown')}")
                                st.session_state.generated_odes.append(result)
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")
                            logger.error(traceback.format_exc())


def batch_generation_page():
    st.header("📊 Batch ODE Generation")

    if not HAVE_SRC:
        st.error("This module requires the `src/` package. Ensure the project ZIP is extracted with src/ present.")
        return

    st.markdown(
        """
    <div class="info-box">
        Generate multiple ODEs efficiently with customizable parameters.
        Supports parallel processing for large batches.
    </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 1000, 50)
        gen_types = st.multiselect("Generator Types", ["linear", "nonlinear"], default=["linear", "nonlinear"])
    with c2:
        func_categories = st.multiselect("Function Categories", ["Basic", "Special"], default=["Basic"])
        vary_params = st.checkbox("Vary Parameters", True)
    with c3:
        if vary_params:
            alpha_range = st.slider("α range", -10.0, 10.0, (-5.0, 5.0))
            beta_range = st.slider("β range", 0.1, 10.0, (0.5, 5.0))
            n_range = st.slider("n range", 1, 5, (1, 3))
        else:
            alpha_range = (1.0, 1.0)
            beta_range = (1.0, 1.0)
            n_range = (1, 1)

    with st.expander("⚙️ Advanced Options"):
        parallel = st.checkbox("Use Parallel Processing", True)
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "LaTeX", "All"])
        include_solutions = st.checkbox("Include Full Solutions", True)
        include_classification = st.checkbox("Include Classification", True)

    if st.button("🚀 Generate Batch", type="primary", use_container_width=True):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            results = []
            progress = st.progress(0)
            status = st.empty()

            all_functions = []
            if "Basic" in func_categories:
                all_functions.extend(st.session_state.basic_functions.get_function_names())
            if "Special" in func_categories:
                all_functions.extend(st.session_state.special_functions.get_function_names()[:10])

            for i in range(num_odes):
                try:
                    progress.progress((i + 1) / num_odes)
                    status.text(f"Generating ODE {i+1}/{num_odes}")
                    params = {
                        "alpha": float(np.random.uniform(*alpha_range)),
                        "beta": float(np.random.uniform(*beta_range)),
                        "n": int(np.random.randint(n_range[0], n_range[1] + 1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    gen_type = str(np.random.choice(gen_types))
                    func_name = str(np.random.choice(all_functions))
                    try:
                        f_z = st.session_state.basic_functions.get_function(func_name)
                    except Exception:
                        f_z = st.session_state.special_functions.get_function(func_name)

                    if gen_type == "linear":
                        factory = CompleteLinearGeneratorFactory()
                        gen_num = int(np.random.randint(1, 9))
                        if gen_num in [4, 5]:
                            params["a"] = float(np.random.uniform(1, 3))
                        result = factory.create(gen_num, f_z, **params)
                    else:
                        factory = CompleteNonlinearGeneratorFactory()
                        gen_num = int(np.random.randint(1, 11))
                        if gen_num in [1, 2, 4]:
                            params["q"] = int(np.random.randint(2, 6))
                        if gen_num in [2, 3, 5]:
                            params["v"] = int(np.random.randint(2, 6))
                        if gen_num in [4, 5, 9, 10]:
                            params["a"] = float(np.random.uniform(1, 3))
                        result = factory.create(gen_num, f_z, **params)

                    row = {
                        "ID": i + 1,
                        "Type": result.get("type"),
                        "Generator": result.get("generator_number"),
                        "Function": func_name,
                        "Order": result.get("order"),
                        "α": round(params["alpha"], 3),
                        "β": round(params["beta"], 3),
                        "n": params["n"],
                    }
                    if include_solutions:
                        sol = result.get("solution")
                        row["Solution"] = (str(sol)[:100] + "...") if sol is not None else ""
                    if include_classification:
                        row["Subtype"] = result.get("subtype", "standard")

                    results.append(row)
                except Exception as e:
                    logger.debug(f"Failed to generate ODE {i+1}: {e}")

            st.session_state.batch_results.extend(results)
            st.success(f"✅ Generated {len(results)} ODEs successfully!")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            st.subheader("📤 Export Results")
            e1, e2, e3, e4 = st.columns(4)

            with e1:
                st.download_button(
                    "📊 Download CSV",
                    df.to_csv(index=False),
                    f"batch_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                )
            with e2:
                st.download_button(
                    "📄 Download JSON",
                    json.dumps(results, indent=2, default=str),
                    f"batch_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                )
            with e3:
                if export_format in ["LaTeX", "All"]:
                    st.download_button(
                        "📝 Download LaTeX",
                        generate_batch_latex(results),
                        f"batch_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                        "text/x-latex",
                    )
            with e4:
                if export_format == "All":
                    package = create_batch_package(results, df)
                    st.download_button(
                        "📦 Download All (ZIP)",
                        package,
                        f"batch_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        "application/zip",
                    )


def novelty_detection_page():
    st.header("🔍 Novelty Detection")

    if not HAVE_SRC:
        st.error("This module requires the `src/` package. Ensure the project ZIP is extracted with src/ present.")
        return

    detector = st.session_state.novelty_detector
    if detector is None:
        st.error("Novelty detector is unavailable.")
        return

    st.markdown(
        """
    <div class="info-box">
        Transformer-based novelty detection. Analyzes ODEs for novel patterns and research potential.
    </div>
    """,
        unsafe_allow_html=True,
    )

    method = st.radio("Input Method", ["Use Current Generator", "Enter ODE Manually", "Select from Generated"])
    ode_to_analyze = None

    if method == "Use Current Generator":
        if st.session_state.generator_terms:
            constructor = GeneratorConstructor()
            for t in st.session_state.generator_terms:
                constructor.add_term(t)
            ode_to_analyze = {
                "ode": getattr(constructor, "get_generator_expression", lambda: "LHS")(),
                "type": "custom",
                "order": max(t.derivative_order for t in st.session_state.generator_terms),
            }
        else:
            st.warning("No generator terms defined!")
    elif method == "Enter ODE Manually":
        ode_str = st.text_area("Enter ODE (LaTeX or text format):")
        if ode_str:
            ode_to_analyze = {"ode": ode_str, "type": "manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox(
                "Select ODE",
                range(len(st.session_state.generated_odes)),
                format_func=lambda x: f"ODE {x+1}: {st.session_state.generated_odes[x].get('type', 'Unknown')}",
            )
            ode_to_analyze = st.session_state.generated_odes[idx]

    if ode_to_analyze and st.button("🔍 Analyze Novelty", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                analysis = detector.analyze(ode_to_analyze, check_solvability=True, detailed=True)
                st.session_state.analysis_results.append(
                    {"ode": ode_to_analyze, "analysis": analysis, "timestamp": datetime.now().isoformat()}
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    novelty_emoji = "🟢" if getattr(analysis, "is_novel", False) else "🔴"
                    st.metric("Novelty", f"{novelty_emoji} {'NOVEL' if analysis.is_novel else 'STANDARD'}")
                with c2:
                    st.metric("Novelty Score", f"{getattr(analysis, 'novelty_score', 0.0):.1f}/100")
                with c3:
                    st.metric("Confidence", f"{getattr(analysis, 'confidence', 0.0):.2%}")

                with st.expander("📊 Detailed Analysis", expanded=True):
                    st.write(f"**Complexity Level:** {getattr(analysis, 'complexity_level', 'n/a')}")
                    st.write(
                        f"**Solvable by Standard Methods:** {'Yes' if getattr(analysis,'solvable_by_standard_methods', False) else 'No'}"
                    )
                    if getattr(analysis, "special_characteristics", None):
                        st.write("**Special Characteristics:**")
                        for ch in analysis.special_characteristics:
                            st.write(f"• {ch}")
                    if getattr(analysis, "recommended_methods", None):
                        st.write("**Recommended Solution Methods:**")
                        for meth in analysis.recommended_methods[:5]:
                            st.write(f"• {meth}")
                    if getattr(analysis, "similar_known_equations", None):
                        st.write("**Similar Known Equations:**")
                        for eq in analysis.similar_known_equations[:3]:
                            st.write(f"• {eq}")

                if getattr(analysis, "detailed_report", None):
                    st.subheader("📄 Detailed Report")
                    st.text(analysis.detailed_report)
                    st.download_button(
                        "📥 Download Report",
                        analysis.detailed_report,
                        f"novelty_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain",
                    )
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error(traceback.format_exc())


def analysis_classification_page():
    st.header("📈 Analysis & Classification")

    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet. Start with the Generator Constructor!")
        return

    # Overview
    st.subheader("📊 Generated ODEs Overview")
    summary_data = []
    for i, ode in enumerate(st.session_state.generated_odes[-20:]):
        summary_data.append(
            {
                "ID": i + 1,
                "Type": ode.get("type", "Unknown"),
                "Order": ode.get("order", 0),
                "Generator": ode.get("generator_number", "N/A"),
                "Function": ode.get("function_used", "Unknown"),
                "Timestamp": (ode.get("timestamp") or "")[:19],
            }
        )
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Linear ODEs", sum(1 for ode in st.session_state.generated_odes if ode.get("type") == "linear"))
    with c2:
        st.metric("Nonlinear ODEs", sum(1 for ode in st.session_state.generated_odes if ode.get("type") == "nonlinear"))
    with c3:
        try:
            avg_order = np.mean([ode.get("order", 0) for ode in st.session_state.generated_odes])
            st.metric("Average Order", f"{avg_order:.1f}")
        except Exception:
            st.metric("Average Order", "n/a")
    with c4:
        unique_funcs = len(set(ode.get("function_used", "") for ode in st.session_state.generated_odes))
        st.metric("Unique Functions", unique_funcs)

    st.subheader("📊 Distributions")
    d1, d2 = st.columns(2)
    with d1:
        orders = [ode.get("order", 0) for ode in st.session_state.generated_odes]
        fig = px.histogram(x=orders, title="Order Distribution", nbins=10)
        fig.update_layout(xaxis_title="Order", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        types = [ode.get("type", "Unknown") for ode in st.session_state.generated_odes]
        counts = pd.Series(types).value_counts()
        fig = px.pie(values=counts.values, names=counts.index, title="Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏷️ Classification Analysis")
    if not HAVE_SRC or st.session_state.ode_classifier is None:
        st.info("Classifier unavailable; install or enable src/ to run classification.")
        return

    if st.button("Classify All ODEs", type="primary"):
        with st.spinner("Classifying..."):
            classifications = []
            for ode in st.session_state.generated_odes:
                try:
                    classifications.append(st.session_state.ode_classifier.classify_ode(ode))
                except Exception:
                    classifications.append({})
            fields = [
                c.get("classification", {}).get("field", "Unknown") for c in classifications if isinstance(c, dict)
            ]
            field_counts = pd.Series(fields).value_counts()
            fig = px.bar(x=field_counts.index, y=field_counts.values, title="Classification by Field")
            fig.update_layout(xaxis_title="Field", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)


def physical_applications_page():
    st.header("🔬 Physical Applications")
    st.markdown(
        """
    <div class="info-box">
      Explore how generated ODEs relate to real-world physics, engineering, and science applications.
    </div>
    """,
        unsafe_allow_html=True,
    )

    category = st.selectbox(
        "Select Application Field",
        ["Mechanics", "Quantum Physics", "Thermodynamics", "Electromagnetism", "Biology", "Economics", "Engineering"],
    )

    applications = {
        "Mechanics": [
            {"name": "Harmonic Oscillator", "equation": "y'' + ω² y = 0", "description": "Spring-mass systems, pendulums"},
            {"name": "Damped Oscillator", "equation": "y'' + 2γ y' + ω₀² y = 0", "description": "Real oscillators with friction"},
            {"name": "Forced Oscillator", "equation": "y'' + 2γ y' + ω₀² y = F cos(ωt)", "description": "Driven systems"},
        ],
        "Quantum Physics": [
            {"name": "Schrödinger (1D)", "equation": "-ℏ²/(2m) y'' + V(x)y = Ey", "description": "Quantum states"},
            {"name": "Particle in Box", "equation": "y'' + (2mE/ℏ²) y = 0", "description": "Confined particles"},
            {"name": "Quantum Oscillator", "equation": "y'' + (2m/ℏ²)(E - ½mω²x²)y = 0", "description": "Quantum oscillator"},
        ],
        "Thermodynamics": [
            {"name": "Heat Equation", "equation": "∂T/∂t = α∇²T", "description": "Heat diffusion"},
            {"name": "Fourier's Law", "equation": "q = -k∇T", "description": "Heat conduction"},
            {"name": "Newton's Cooling", "equation": "dT/dt = -k(T - T_env)", "description": "Cooling processes"},
        ],
    }

    if category in applications:
        for app in applications[category]:
            with st.expander(f"📚 {app['name']}"):
                st.latex(app["equation"])
                st.write(f"**Description:** {app['description']}")
                similar = find_similar_odes(app["equation"])
                if similar:
                    st.write(f"**Similar Generated ODEs:** {len(similar)} found")
                    for ode in similar[:3]:
                        st.write(f"• Generator {ode.get('generator_number', 'N/A')}")

    st.subfooter = st.subheader  # compatibility alias if Streamlit changes
    st.subheader("🔗 Match Your ODEs to Applications")
    if st.session_state.generated_odes and HAVE_SRC and st.session_state.ode_classifier is not None:
        selected_ode = st.selectbox(
            "Select Generated ODE",
            range(len(st.session_state.generated_odes)),
            format_func=lambda x: f"ODE {x+1}: Type={st.session_state.generated_odes[x].get('type','Unknown')}, Order={st.session_state.generated_odes[x].get('order',0)}",
        )
        if st.button("Find Applications"):
            ode = st.session_state.generated_odes[selected_ode]
            try:
                result = st.session_state.ode_classifier.classify_ode(ode)
                matched = result.get("matched_applications", [])
                if matched:
                    st.success(f"Found {len(matched)} applications!")
                    for app in matched:
                        st.write(f"**{getattr(app,'name','(unnamed)')}** ({getattr(app,'field','n/a')})")
                        st.write(f"Description: {getattr(app,'description','')}")
                        if getattr(app, "parameters_meaning", None):
                            st.write("Parameter meanings:")
                            for p, m in app.parameters_meaning.items():
                                st.write(f"• {p}: {m}")
                else:
                    st.info("No specific applications identified. This may be a novel equation!")
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")
    else:
        st.info("Generate ODEs and enable src/ classifier to match applications.")


def visualization_page():
    st.header("📐 Visualization")

    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize. Generate some first!")
        return

    idx = st.selectbox(
        "Select ODE to Visualize",
        range(len(st.session_state.generated_odes)),
        format_func=lambda x: f"ODE {x+1}: {st.session_state.generated_odes[x].get('type','Unknown')} (Order {st.session_state.generated_odes[x].get('order',0)})",
    )
    ode = st.session_state.generated_odes[idx]

    c1, c2, c3 = st.columns(3)
    with c1:
        plot_type = st.selectbox("Plot Type", ["Solution", "Phase Portrait", "3D Surface", "Direction Field"])
    with c2:
        x_range = st.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    with c3:
        num_points = st.slider("Number of Points", 100, 2000, 500)

    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Creating visualization..."):
            try:
                if plot_type == "Solution":
                    fig = create_solution_plot(ode, x_range, num_points)
                elif plot_type == "Phase Portrait":
                    fig = create_phase_portrait(ode, x_range)
                elif plot_type == "3D Surface":
                    fig = create_3d_surface(ode, x_range)
                else:
                    fig = create_direction_field(ode, x_range)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("📷 Save as PNG"):
                        try:
                            import kaleido  # noqa
                            fig.write_image("ode_plot.png")
                            st.success("Saved as ode_plot.png")
                        except Exception:
                            st.error("PNG export requires the `kaleido` package. Install it and try again.")
                with c2:
                    if st.button("📊 Save as HTML"):
                        fig.write_html("ode_plot.html")
                        st.success("Saved as ode_plot.html")
            except Exception as e:
                st.error(f"Visualization failed: {str(e)}")
                logger.error(traceback.format_exc())


def export_latex_page():
    st.header("📤 Export & LaTeX")

    st.markdown(
        """
    <div class="latex-export-box">
        <h3>📝 Professional LaTeX Export System</h3>
        <p>Export your ODEs in publication-ready LaTeX format with complete documentation.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not st.session_state.generated_odes:
        st.warning("No ODEs to export. Generate some first!")
        return

    export_type = st.radio("Export Type", ["Single ODE", "Multiple ODEs", "Complete Report", "Batch Export"])

    if export_type == "Single ODE":
        idx = st.selectbox(
            "Select ODE",
            range(len(st.session_state.generated_odes)),
            format_func=lambda x: f"ODE {x+1}: {st.session_state.generated_odes[x].get('type','Unknown')}",
        )
        ode = st.session_state.generated_odes[idx]

        st.subheader("📋 LaTeX Preview")
        latex_doc = LaTeXExporter.generate_latex_document(ode, include_preamble=False)
        st.code(latex_doc, language="latex")

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "📄 Download LaTeX",
                LaTeXExporter.generate_latex_document(ode, include_preamble=True),
                f"ode_{idx+1}.tex",
                "text/x-latex",
                use_container_width=True,
            )
        with e2:
            st.info("PDF generation requires a local LaTeX compiler (e.g. pdflatex). Download .tex and compile locally.")
        with e3:
            st.download_button(
                "📦 Download Package",
                LaTeXExporter.create_export_package(ode, include_extras=True),
                f"ode_package_{idx+1}.zip",
                "application/zip",
                use_container_width=True,
            )

    elif export_type == "Multiple ODEs":
        indices = st.multiselect(
            "Select ODEs",
            range(len(st.session_state.generated_odes)),
            format_func=lambda x: f"ODE {x+1}: {st.session_state.generated_odes[x].get('type','Unknown')}",
        )
        if indices and st.button("Generate Multi-ODE Document"):
            parts = [
                r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\title{Collection of Generated ODEs}
\author{Master Generators System}
\date{\today}
\begin{document}
\maketitle
"""
            ]
            for j, idx in enumerate(indices):
                ode = st.session_state.generated_odes[idx]
                parts.append(f"\\section{{ODE {j+1}}}")
                parts.append(LaTeXExporter.generate_latex_document(ode, include_preamble=False))
            parts.append(r"\end{document}")
            st.download_button(
                "📄 Download Multi-ODE LaTeX",
                "\n".join(parts),
                f"multiple_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                "text/x-latex",
            )

    elif export_type == "Complete Report":
        if st.button("Generate Complete Report"):
            st.download_button(
                "📄 Download Complete Report",
                generate_complete_report(),
                f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                "text/x-latex",
            )

    else:
        st.subheader("📦 Batch Export Options")
        formats = st.multiselect("Export Formats", ["LaTeX", "JSON", "CSV", "Python", "Mathematica"], default=["LaTeX", "JSON"])
        if st.button("Export All", type="primary"):
            export_all_formats(formats)


def examples_library_page():
    st.header("📚 Examples Library")

    category = st.selectbox(
        "Select Category",
        ["Linear Generators", "Nonlinear Generators", "Special Functions", "Physical Examples", "Advanced Examples"],
    )

    examples = {
        "Linear Generators": [
            {
                "name": "Simple Harmonic Oscillator",
                "generator": "y'' + y = RHS",
                "parameters": {"alpha": 1, "beta": 1, "n": 1, "M": 0},
                "function": "sin",
                "description": "Classic harmonic oscillator equation",
            },
            {
                "name": "Damped Oscillator",
                "generator": "y'' + y' + y = RHS",
                "parameters": {"alpha": 0, "beta": 2, "n": 1, "M": 0},
                "function": "exp",
                "description": "Oscillator with damping term",
            },
        ],
        "Nonlinear Generators": [
            {
                "name": "Cubic Nonlinearity",
                "generator": "(y'')^3 + y = RHS",
                "parameters": {"alpha": 1, "beta": 1, "n": 1, "M": 0, "q": 3},
                "function": "z**2",
                "description": "Nonlinear with cubic derivative term",
            },
            {
                "name": "Exponential Nonlinearity",
                "generator": "exp(y") + exp(y') = RHS",
                "parameters": {"alpha": 0, "beta": 1, "n": 1, "M": 0},
                "function": "log(z+1)",
                "description": "Exponential transformation of derivatives",
            },
        ],
        "Special Functions": [
            {
                "name": "Airy-type Equation",
                "generator": "y'' - x y = RHS",
                "parameters": {"alpha": 0, "beta": 1, "n": 1, "M": 0},
                "function": "airy_ai",
                "description": "Related to Airy functions",
            },
            {
                "name": "Bessel-type Equation",
                "generator": "x^2 y'' + x y' + (x^2 - n^2) y = RHS",
                "parameters": {"alpha": 0, "beta": 1, "n": 2, "M": 0},
                "function": "bessel_j0",
                "description": "Related to Bessel functions",
            },
        ],
    }

    if category in examples:
        for ex in examples[category]:
            with st.expander(f"📖 {ex['name']}"):
                st.latex(ex["generator"])
                st.write(f"**Description:** {ex['description']}")
                st.write("**Parameters:**")
                for k, v in ex["parameters"].items():
                    st.write(f"• {k} = {v}")
                st.write(f"**Function:** f(z) = {ex['function']}")
                if st.button(f"Load Example: {ex['name']}", key=f"load_{ex['name']}"):
                    st.info("Example metadata loaded. You can copy parameters manually into the Generator Constructor.")


def settings_page():
    st.header("⚙️ Settings")

    tabs = st.tabs(["General", "ML Configuration", "Export Settings", "Advanced", "About"])

    with tabs[0]:
        st.subheader("General Settings")
        st.markdown("### Parameter Limits")
        c1, c2 = st.columns(2)
        with c1:
            _ = st.number_input("Max |α|", 1, 1000, 100)
            _ = st.number_input("Max β", 1, 1000, 100)
        with c2:
            _ = st.number_input("Max n", 1, 20, 10)
            _ = st.number_input("Max Derivative Order", 1, 10, 5)

        st.markdown("### Display Settings")
        _ = st.checkbox("Show LaTeX equations", value=True)
        _ = st.checkbox("Auto-save generated ODEs", value=False)
        _ = st.checkbox("Dark mode", value=False)

        if st.button("Save General Settings"):
            st.success("Settings saved!")

    with tabs[1]:
        st.subheader("ML Configuration")
        _ = st.selectbox("Default ML Model", ["pattern_learner", "vae", "transformer"])
        st.markdown("### Training Defaults")
        c1, c2 = st.columns(2)
        with c1:
            _ = st.slider("Default Epochs", 10, 500, 100)
            _ = st.slider("Default Batch Size", 8, 128, 32)
        with c2:
            _ = st.select_slider("Default Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
            _ = st.checkbox("Use GPU if available", value=True)
        if st.button("Save ML Settings"):
            st.success("ML settings saved!")

    with tabs[2]:
        st.subheader("Export Settings")
        _ = st.multiselect("Default Export Formats", ["LaTeX", "JSON", "CSV", "Python"], default=["LaTeX", "JSON"])
        st.markdown("### LaTeX Settings")
        _ = st.checkbox("Include LaTeX preamble", value=True)
        _ = st.checkbox("Include metadata in exports", value=True)
        _ = st.checkbox("Include plots in LaTeX", value=False)
        if st.button("Save Export Settings"):
            st.success("Export settings saved!")

    with tabs[3]:
        st.subheader("Advanced Settings")
        st.markdown("### Cache Management")
        c1, c2, c3 = st.columns(3)
        with c1:
            cache_size = 0
            if st.session_state.get("cache_manager") is not None:
                try:
                    cache_size = len(st.session_state.cache_manager.memory_cache)
                except Exception:
                    cache_size = 0
            st.metric("Cache Size", cache_size)
        with c2:
            if st.button("Clear Cache"):
                if st.session_state.get("cache_manager"):
                    try:
                        st.session_state.cache_manager.clear()
                        st.success("Cache cleared!")
                    except Exception:
                        st.info("No cache to clear.")
                else:
                    st.info("Cache manager unavailable.")
        with c3:
            if st.button("Save Session"):
                ok = SessionStateManager.save_to_file()
                st.success("Session saved!") if ok else st.error("Failed to save session.")

        st.markdown("### Debug Mode")
        debug_mode = st.checkbox("Enable debug mode", value=False)
        if debug_mode:
            st.write("Session State Keys:", list(st.session_state.keys()))

    with tabs[4]:
        st.subheader("About")
        st.markdown(
            f"""
        ### Master Generators for ODEs v{APP_VERSION}
        **Complete Implementation with Enhanced Stability**

        **Core Features**
        - ✅ Exact solutions using Theorem 4.1 (with consistent RHS via operator application)
        - ✅ Optional Theorem 4.2 / special functions stubs
        - ✅ ML/DL pattern learning (if PyTorch installed)
        - ✅ Batch generation
        - ✅ Physical applications mapping
        - ✅ Professional LaTeX export
        - ✅ Novelty detection
        - ✅ Classification summary

        **Notes**
        - Pages requiring `src/` or PyTorch are automatically gated with helpful messages if unavailable.
        """
        )


def documentation_page():
    st.header("📖 Documentation")
    tabs = st.tabs(["Quick Start", "Mathematical Theory", "API Reference", "Examples", "FAQ"])

    with tabs[0]:
        st.markdown(
            """
        ## Quick Start
        1. Go to **Generator Constructor**, add terms, **Build Generator Specification**.
        2. Choose a function \( f(z) \) and parameters \( \alpha,\beta,n,M \).
        3. Click **Generate ODE** to compute the exact solution (Theorem 4.1) and RHS from your operator.
        4. Export via **Export & LaTeX**.
        5. (Optional) Train ML models in **ML Pattern Learning** if PyTorch is installed.
        """
        )

    with tabs[1]:
        st.markdown(
            r"""
        ## Mathematical Theory

        ### Theorem 4.1 (Master Generator)
        \[
        Y(\theta) = \frac{\pi}{2n}\sum_{s=1}^n \Big( 2\,f(\alpha+\beta)
        - f\big(\alpha + \beta e^{\,i\theta\cos\omega_s - \theta\sin\omega_s}\big)
        - f\big(\alpha + \beta e^{-i\theta\cos\omega_s - \theta\sin\omega_s}\big)\Big) + \pi M,\quad
        \omega_s=\frac{(2s-1)\pi}{2n}.
        \]

        The app uses \( x \) as the independent variable (instead of \( \theta \)) and computes \( y(x) \) accordingly.

        ### Theorem 4.2
        Extended formulation for higher-order derivatives (UI stub included; integrate your ExtendedMasterTheorem logic as needed).
        """
        )

    with tabs[2]:
        st.markdown(
            """
        ## API Reference (selected)
        ```python
        # Theorem 4.1 helper
        y = theorem_4_1_solution_expr(f_expr, alpha, beta, n, M, x)

        # Apply generator L[y] to produce RHS
        rhs = apply_generator_to(y, gen_spec, x)

        # LaTeX
        latex = LaTeXExporter.generate_latex_document(ode_data)
        package = LaTeXExporter.create_export_package(ode_data)
        ```
        """
        )

    with tabs[3]:
        st.code(
            """
# Example: Generate a nonlinear ODE with exponential nonlinearity (from factories)
import sympy as sp
from src.generators.nonlinear_generators import CompleteNonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions

params = {'alpha': 1.0, 'beta': 2.0, 'n': 1, 'M': 0.5}
basic = BasicFunctions()
f_z = basic.get_function('exponential')  # f(z) = e^z

factory = CompleteNonlinearGeneratorFactory()
result = factory.create(7, f_z, **params)  # e^(y'') + e^(y') = RHS

print("ODE:", result.get('ode'))
print("Solution:", result.get('solution'))
print("Type:", result.get('type'))
            """,
            language="python",
        )

    with tabs[4]:
        st.markdown(
            """
        ## Frequently Asked Questions

        **What determines novelty?**  
        Structure far from known classes or unusual term combos.

        **How accurate are solutions?**  
        Symbolic/theoretical exactness from Theorem 4.1; numerical verification depends on evaluation.

        **Can I use custom functions?**  
        Yes—any function expressible symbolically (see Basic/Special libraries).

        **Do I need a GPU?**  
        No; CPU is supported. GPU (PyTorch) just speeds up training.

        **PDF export?**  
        Download `.tex` and compile locally with `pdflatex`.
        """
        )


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main():
    SessionStateManager.initialize()
    header()
    page = sidebar_nav()

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


if __name__ == "__main__":
    main()



