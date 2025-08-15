"""
Master Generators for ODEs — Complete App (Rewritten)

What’s new in this version
==========================
• Apply Master Theorem page rewritten:
  - Exact (symbolic) parameters toggle (nsimplify) to keep LaTeX symbolic
  - Theorem 4.1 (symbolic); robust, cached ω_s; controlled simplify
  - Theorem 4.2 (compact Stirling-number form); exact and fast
  - Free-form LHS builder: terms like sinh(y'), exp(y''''''), ln(y''), y(x/a+b), power, outer m-derivative
  - RHS builder: R = L[y] by substituting y, y', y'', … into free-form or current LHS
  - Timeouts: runs heavy steps in a separate process; prevents “stuck” on Railway

• Imports are resilient: try multiple src paths so it works with your ZIP layout
• No reliance on `GeneratorConstructor.add_term()` anywhere (fixes earlier error)
• All other services (ML/DL, Batch, Novelty, Export/LaTeX, etc.) preserved

Run:
  streamlit run master_generators_app.py
"""

# ============================================================================
# Standard libs
# ============================================================================
import os
import sys
import io
import json
import time
import math
import base64
import zipfile
import logging
import traceback
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# ============================================================================
# Third-party
# ============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp

# Optional ML/DL
try:
    import torch
    from torch.utils.data import Dataset, DataLoader  # noqa: F401
except Exception:
    torch = None

# Plotly
import plotly.graph_objects as go
import plotly.express as px

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
# Resilient imports from src/ (try several locations)
# ============================================================================

# Generators & factories
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
CacheManager = cached = None
ParameterValidator = None
UIComponents = None

try:
    # try canonical locations in your ZIP
    from src.generators.master_generator import (
        MasterGenerator,
        EnhancedMasterGenerator,
        CompleteMasterGenerator,
    )
    # Some projects place factories here:
    try:
        from src.generators.master_generator import (
            CompleteLinearGeneratorFactory,
            CompleteNonlinearGeneratorFactory,
        )
    except Exception:
        # or in their own modules:
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
    from src.utils.cache import CacheManager, cached
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
# Custom CSS
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
        border-left: 5px solid #2196f3; padding: 1rem; border-radius: 10px; margin: 1rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50; padding: 1rem; border-radius: 10px; margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336; padding: 1rem; border-radius: 10px; margin: 1rem 0;
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
    def sympy_to_latex(expr) -> str:
        if expr is None:
            return ""
        try:
            if isinstance(expr, str):
                try:
                    expr = sp.sympify(expr)
                except Exception:
                    return expr
            latex_str = sp.latex(expr)
            return latex_str.replace(r"\left(", "(").replace(r"\right)", ")")
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

        parts = []
        if include_preamble:
            parts.append(
                r"""
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
"""
            )
        parts.append(r"\subsection{Generator Equation}")
        parts.append(r"\begin{equation}")
        parts.append(
            f"{LaTeXExporter.sympy_to_latex(generator)} = {LaTeXExporter.sympy_to_latex(rhs)}"
        )
        parts.append(r"\end{equation}")

        parts.append(r"\subsection{Exact Solution}")
        parts.append(r"\begin{equation}")
        parts.append(f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}")
        parts.append(r"\end{equation}")

        parts.append(r"\subsection{Parameters}")
        parts.append(r"\begin{align}")
        parts.append(f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha', 1))} \\\\")
        parts.append(f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta', 1))} \\\\")
        parts.append(f"n       &= {params.get('n', 1)} \\\\")
        parts.append(f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M', 0))}")
        if "q" in params:
            parts.append(f" \\\\ q &= {LaTeXExporter.sympy_to_latex(params['q'])}")
        if "v" in params:
            parts.append(f" \\\\ v &= {LaTeXExporter.sympy_to_latex(params['v'])}")
        if "a" in params:
            parts.append(f" \\\\ a &= {LaTeXExporter.sympy_to_latex(params['a'])}")
        parts.append(r"\end{align}")

        if initial_conditions:
            parts.append(r"\subsection{Initial Conditions}")
            parts.append(r"\begin{align}")
            ic_items = list(initial_conditions.items())
            for idx, (k, v) in enumerate(ic_items):
                sep = r" \\" if idx < len(ic_items) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}{sep}")
            parts.append(r"\end{align}")

        if classification:
            parts.append(r"\subsection{Mathematical Classification}")
            parts.append(r"\begin{itemize}")
            parts.append(f"\\item \\textbf{{Type:}} {classification.get('type','Unknown')}")
            parts.append(f"\\item \\textbf{{Order:}} {classification.get('order','Unknown')}")
            parts.append(f"\\item \\textbf{{Linearity:}} {classification.get('linearity','Unknown')}")
            if "field" in classification:
                parts.append(f"\\item \\textbf{{Field:}} {classification['field']}")
            if "applications" in classification:
                apps = ", ".join(classification["applications"][:5])
                parts.append(f"\\item \\textbf{{Applications:}} {apps}")
            parts.append(r"\end{itemize}")

        parts.append(r"\subsection{Solution Verification}")
        parts.append(
            r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$."
        )

        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            latex_content = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
            zf.writestr("ode_document.tex", latex_content)
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            readme = f"""Master Generator ODE Export
Generated: {datetime.now().isoformat()}

Contents:
- ode_document.tex
- ode_data.json
- README.txt

To compile: pdflatex ode_document.tex
"""
            zf.writestr("README.txt", readme)
            if include_extras:
                zf.writestr("reproduce.py", LaTeXExporter.generate_python_code(ode_data))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    @staticmethod
    def generate_python_code(ode_data: Dict[str, Any]) -> str:
        params = ode_data.get("parameters", {})
        gen_type = ode_data.get("type", "linear")
        gen_num = ode_data.get("generator_number", 1)
        func_name = ode_data.get("function_used", "exp")
        code = f'''"""
Reproduce a generated ODE (skeleton)
"""
import sympy as sp
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
try:
    from src.generators.master_generator import CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory
except Exception:
    from src.generators.linear_generators import LinearGeneratorFactory as CompleteLinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory as CompleteNonlinearGeneratorFactory

params = {{
    'alpha': {repr(params.get('alpha', 1))},
    'beta': {repr(params.get('beta', 1))},
    'n': {repr(params.get('n', 1))},
    'M': {repr(params.get('M', 0))}
}}
basic = BasicFunctions()
special = SpecialFunctions()
try:
    f_z = basic.get_function('{func_name}')
except Exception:
    f_z = special.get_function('{func_name}')

if '{gen_type}' == 'linear':
    factory = CompleteLinearGeneratorFactory()
else:
    factory = CompleteNonlinearGeneratorFactory()

result = factory.create({gen_num}, f_z, **params)
print("ODE:", result.get('ode'))
print("Solution:", result.get('solution'))
'''
        return code


# ============================================================================
# Session State Manager
# ============================================================================
class SessionStateManager:
    @staticmethod
    def initialize():
        if "generator_constructor" not in st.session_state and GeneratorConstructor:
            st.session_state.generator_constructor = GeneratorConstructor()
        if "generator_terms" not in st.session_state:
            st.session_state.generator_terms = []
        if "generated_odes" not in st.session_state:
            st.session_state.generated_odes = []
        if "generator_patterns" not in st.session_state:
            st.session_state.generator_patterns = []
        if "vae_model" not in st.session_state and GeneratorVAE:
            st.session_state.vae_model = GeneratorVAE()
        if "pattern_learner" not in st.session_state and GeneratorPatternLearner:
            st.session_state.pattern_learner = GeneratorPatternLearner()
        if "novelty_detector" not in st.session_state and ODENoveltyDetector:
            try:
                st.session_state.novelty_detector = ODENoveltyDetector()
            except Exception:
                st.session_state.novelty_detector = None
        if "ode_classifier" not in st.session_state and ODEClassifier:
            try:
                st.session_state.ode_classifier = ODEClassifier()
            except Exception:
                st.session_state.ode_classifier = None
        if "ml_trainer" not in st.session_state:
            st.session_state.ml_trainer = None
        if "ml_trained" not in st.session_state:
            st.session_state.ml_trained = False
        if "training_history" not in st.session_state:
            st.session_state.training_history = []
        if "batch_results" not in st.session_state:
            st.session_state.batch_results = []
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = []
        if "cache_manager" not in st.session_state and CacheManager:
            st.session_state.cache_manager = CacheManager()
        if "ui_components" not in st.session_state and UIComponents:
            st.session_state.ui_components = UIComponents()
        if "basic_functions" not in st.session_state and BasicFunctions:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions:
            st.session_state.special_functions = SpecialFunctions()
        if "theorem_solver" not in st.session_state and MasterTheoremSolver:
            st.session_state.theorem_solver = MasterTheoremSolver()
        if "extended_theorem" not in st.session_state and ExtendedMasterTheorem:
            st.session_state.extended_theorem = ExtendedMasterTheorem()
        if "export_history" not in st.session_state:
            st.session_state.export_history = []

    @staticmethod
    def save_to_file(filename: str = "session_state.pkl") -> bool:
        try:
            state_data = {
                "generated_odes": st.session_state.generated_odes,
                "generator_patterns": st.session_state.generator_patterns,
                "batch_results": st.session_state.batch_results,
                "analysis_results": st.session_state.analysis_results,
                "training_history": st.session_state.training_history,
                "export_history": st.session_state.export_history,
            }
            with open(filename, "wb") as f:
                pickle.dump(state_data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
            return False

    @staticmethod
    def load_from_file(filename: str = "session_state.pkl") -> bool:
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    state_data = pickle.load(f)
                for key, value in state_data.items():
                    if key in st.session_state:
                        st.session_state[key] = value
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return False


# ============================================================================
# -------------------- PATCH: Apply Master Theorem subsystem -----------------
# (exact params, T4.1 + T4.2 (Stirling), free-form LHS, fast RHS, timeouts)
# ============================================================================

from functools import lru_cache
import concurrent.futures as _futures
from sympy import Derivative, Function, Symbol, AppliedUndef

def simplify_expr(expr: sp.Expr, level: str = "light") -> sp.Expr:
    if level == "none":
        return expr
    try:
        e = sp.together(expr)
        e = sp.cancel(e)
        e = sp.simplify(e)
        if level == "aggressive":
            try:
                e = sp.nsimplify(e, [sp.E, sp.pi, sp.I], rational=True, maxsteps=50)
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

@lru_cache(maxsize=128)
def omega_list(n: int):
    return tuple(sp.Rational(2*s-1, 2*n)*sp.pi for s in range(1, int(n)+1))

@lru_cache(maxsize=128)
def stirling_row(m: int):
    S = sp.functions.combinatorial.numbers.stirling
    return tuple(S(int(m), j, kind=2) for j in range(1, int(m)+1))

def get_function_expr(source_lib, func_name: str) -> sp.Expr:
    z = sp.Symbol("z", real=True)
    f_obj = None
    if source_lib is not None:
        try:
            f_obj = source_lib.get_function(func_name)
        except Exception:
            f_obj = None
    if f_obj is None:
        try:
            return sp.sympify(func_name, locals={"z": z, "E": sp.E, "pi": sp.pi})
        except Exception as e:
            raise ValueError(f"Unknown function '{func_name}' and cannot sympify: {e}")

    if isinstance(f_obj, sp.Expr):
        frees = list(f_obj.free_symbols)
        if len(frees) == 0:
            return sp.sympify(f_obj)
        g = f_obj
        for s in frees:
            if s != z:
                g = g.subs(s, z)
        return g

    if callable(f_obj):
        try:
            v = f_obj(z)
            if isinstance(v, sp.Expr):
                return v
        except Exception:
            pass

    if isinstance(f_obj, str):
        return sp.sympify(f_obj, locals={"z": z})

    raise TypeError(f"Unsupported function object for '{func_name}'")

def theorem_4_1_solution_expr(f_expr: sp.Expr, alpha, beta, n: int, M, x: sp.Symbol,
                              simplify_level="light") -> sp.Expr:
    z = sp.Symbol("z", real=True)
    base = f_expr.subs(z, alpha + beta)
    terms = []
    for ω in omega_list(int(n)):
        exp_pos = sp.exp(sp.I*x*sp.cos(ω) - x*sp.sin(ω))
        exp_neg = sp.exp(-sp.I*x*sp.cos(ω) - x*sp.sin(ω))
        psi = f_expr.subs(z, alpha + beta*exp_pos)
        phi = f_expr.subs(z, alpha + beta*exp_neg)
        terms.append(2*base - (psi + phi))
    y = sp.pi/(2*n) * sp.Add(*terms) + sp.pi*M
    return simplify_expr(y, level=simplify_level)

def theorem_4_2_y_m_expr(f_expr: sp.Expr, alpha_value, beta, n: int, m: int, x: sp.Symbol,
                         simplify_level="light") -> sp.Expr:
    z = sp.Symbol("z", real=True)
    αsym = sp.Symbol("alpha_sym", real=True)
    total = 0
    Srow = stirling_row(int(m))

    for ω in omega_list(int(n)):
        lam = sp.exp(sp.I*(sp.pi/2 + ω))
        zeta = sp.exp(-x*sp.sin(ω)) * sp.exp(sp.I*x*sp.cos(ω))
        zetab = sp.conjugate(zeta)
        psi = f_expr.subs(z, αsym + beta*zeta)
        phi = f_expr.subs(z, αsym + beta*zetab)

        s1 = 0
        s2 = 0
        for j, S in enumerate(Srow, start=1):
            s1 += S * (beta*zeta)**j  * sp.diff(psi, αsym, j)
            s2 += S * (beta*zetab)**j * sp.diff(phi, αsym, j)

        total += lam**m * s1 + sp.conjugate(lam)**m * s2

    y_m = -sp.pi/(2*n) * total
    y_m = y_m.subs(αsym, alpha_value)
    return simplify_expr(y_m, level=simplify_level)

def apply_lhs_to_solution(lhs_expr: sp.Expr, solution_y: sp.Expr, x: sp.Symbol,
                          y_name: str = "y", simplify_level="light",
                          max_unique_derivs: int = 60) -> sp.Expr:
    subs_map = {}
    yfun = sp.Function(y_name)

    needs = []
    seen = set()
    for node in lhs_expr.atoms(AppliedUndef, sp.Derivative):
        if isinstance(node, AppliedUndef) and node.func == yfun and len(node.args) == 1:
            arg = node.args[0]
            key = ("y", sp.srepr(arg), 0)
            if key not in seen:
                needs.append((node, arg, 0, False))
                seen.add(key)
        elif isinstance(node, sp.Derivative):
            base = node.expr
            if isinstance(base, AppliedUndef) and base.func == yfun and len(base.args) == 1:
                arg = base.args[0]
                try:
                    order = sum(c for v, c in node.variable_count if v == x)
                except Exception:
                    order = sum(1 for v in node.variables if v == x)
                key = ("dy", sp.srepr(arg), int(order))
                if key not in seen:
                    needs.append((node, arg, int(order), True))
                    seen.add(key)

    if len(needs) > max_unique_derivs:
        raise RuntimeError(f"Too many distinct derivatives requested ({len(needs)} > {max_unique_derivs}).")

    max_order_x = max((o for _, a, o, isd in needs if isd and a == x), default=0)
    d_cache = {0: solution_y}
    for k in range(1, max_order_x+1):
        d_cache[k] = sp.diff(solution_y, (x, k))

    local_cache = {}

    def diff_after_sub(arg_expr: sp.Expr, order: int) -> sp.Expr:
        key = (sp.srepr(arg_expr), order)
        if key in local_cache:
            return local_cache[key]
        val = solution_y.subs(x, arg_expr)
        if order > 0:
            val = sp.diff(val, (x, order))
        local_cache[key] = val
        return val

    for node, arg, order, is_deriv in needs:
        if arg == x:
            subs_map[node] = d_cache[order]
        else:
            subs_map[node] = diff_after_sub(arg, order)

    try:
        out = lhs_expr.xreplace(subs_map)
    except Exception:
        out = lhs_expr.subs(subs_map)

    return simplify_expr(out, level=simplify_level)

def build_freeform_term(x: sp.Symbol, coef=1, inner_order=0, wrapper="id", power=1,
                        arg_scale=None, arg_shift=None, outer_order=0,
                        y_name="y", ln_eps=sp.Symbol("epsilon", positive=True)) -> sp.Expr:
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
        core = sp.log(ln_eps + sp.Abs(base))
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

def _worker_theorem_4_1(f_expr, alpha, beta, n, M, x_name, simplify_level):
    x = sp.Symbol(x_name, real=True)
    return theorem_4_1_solution_expr(f_expr, alpha, beta, int(n), M, x, simplify_level)

def _worker_theorem_4_2(f_expr, alpha, beta, n, m, x_name, simplify_level):
    x = sp.Symbol(x_name, real=True)
    return theorem_4_2_y_m_expr(f_expr, alpha, beta, int(n), int(m), x, simplify_level)

def _worker_apply_lhs(lhs_expr, solution_y, x_name, y_name, simplify_level):
    x = sp.Symbol(x_name, real=True)
    return apply_lhs_to_solution(lhs_expr, solution_y, x, y_name=y_name, simplify_level=simplify_level)

def run_with_timeout(func, timeout_sec: int, *args):
    if timeout_sec is None or timeout_sec <= 0:
        return func(*args)
    try:
        with _futures.ProcessPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(func, *args)
            return fut.result(timeout=timeout_sec)
    except _futures.TimeoutError:
        raise TimeoutError(f"Operation exceeded {timeout_sec} seconds")
    except Exception:
        # fallback direct call
        return func(*args)

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (Exact, Fast, with Timeouts)")

    # Function source
    colA, colB = st.columns([1, 1])
    with colA:
        func_source = st.selectbox("Function library", ["Basic", "Special"], index=0)
    with colB:
        basic_lib = st.session_state.get("basic_functions")
        special_lib = st.session_state.get("special_functions")
        if func_source == "Basic" and basic_lib:
            func_names = basic_lib.get_function_names()
            source_lib = basic_lib
        elif func_source == "Special" and special_lib:
            func_names = special_lib.get_function_names()
            source_lib = special_lib
        else:
            func_names = []
            source_lib = None
        func_name = st.selectbox("Choose f(z)", func_names) if func_names else st.text_input("Enter f(z) name or expression", "exp(z)")

    # Parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with col2:
        beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with col3:
        n     = st.number_input("n (positive integer)", value=1, min_value=1, max_value=12, step=1)
    with col4:
        M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    # Controls
    colC, colD, colE = st.columns([1, 1, 1])
    with colC:
        use_exact = st.checkbox("Exact (symbolic) parameters", value=True)
    with colD:
        simplify_level = st.selectbox("Simplify level", ["light", "none", "aggressive"], index=0)
    with colE:
        timeout_sec = st.slider("Timeout (seconds)", min_value=0, max_value=90, value=20,
                                help="Hard cap per heavy operation. 0 = no timeout.")

    apply_generator = st.checkbox("Apply generator (build RHS = L[y])", value=False,
                                  help="Applies the current generator LHS (or free‑form LHS) to y.")

    # Theorem 4.2 option
    colm1, colm2 = st.columns([1, 1])
    with colm1:
        compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", value=False)
    with colm2:
        m_order = st.number_input("m (order for Theorem 4.2)", value=1, min_value=1, max_value=12, step=1)

    # Build f(z)
    try:
        f_expr = get_function_expr(source_lib, func_name)
    except Exception as e:
        st.error(f"Cannot build f(z): {e}")
        return

    # Exact params
    α = to_exact(alpha) if use_exact else sp.Float(alpha)
    β = to_exact(beta)  if use_exact else sp.Float(beta)
    𝑀 = to_exact(M)     if use_exact else sp.Float(M)
    x = sp.Symbol("x", real=True)

    # Resolve current generator LHS (if any)
    generator_lhs = None
    gen_spec = st.session_state.get("current_generator", None)
    if gen_spec is not None and hasattr(gen_spec, "lhs"):
        generator_lhs = gen_spec.lhs
    else:
        constructor = st.session_state.get("generator_constructor", None)
        if constructor is not None and hasattr(constructor, "get_generator_expression"):
            try:
                generator_lhs = constructor.get_generator_expression()
            except Exception:
                generator_lhs = None
    if generator_lhs is None:
        generator_lhs = sp.Symbol("LHS")

    # Free-form builder
    st.subheader("🧩 Optional: Free‑form Generator Builder (LHS)")
    with st.expander("Build custom LHS terms", expanded=False):
        if "free_terms" not in st.session_state:
            st.session_state.free_terms = []
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]:
            coef = st.number_input("coef", value=1.0, step=0.5)
        with cols[1]:
            inner_order = st.number_input("inner k (y^(k))", value=0, min_value=0, max_value=12, step=1)
        with cols[2]:
            wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","sinh","cosh","tanh","log","abs"], index=0)
        with cols[3]:
            power = st.number_input("power", value=1, min_value=1, max_value=6, step=1)
        with cols[4]:
            outer_order = st.number_input("outer m (D^m)", value=0, min_value=0, max_value=12, step=1)
        with cols[5]:
            scale = st.number_input("arg scale (a)", value=1.0, step=0.1, format="%.4f")
        with cols[6]:
            shift = st.number_input("arg shift (b)", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": coef,
                    "inner_order": int(inner_order),
                    "wrapper": wrapper,
                    "power": int(power),
                    "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write("**Current terms:**")
            for i, t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. coef={t['coef']} · D^{t['outer_order']}[ {t['wrapper']}( (y^{t['inner_order']})(x/{t.get('arg_scale',1)}+{t.get('arg_shift',0)}) )^{t['power']} ]")
            colc1, colc2 = st.columns([1,1])
            with colc1:
                if st.button("🧮 Use free‑form LHS"):
                    try:
                        generator_lhs = build_freeform_lhs(x, st.session_state.free_terms)
                        st.success("Free‑form LHS built and will be used as generator LHS.")
                    except Exception as e:
                        st.error(f"Failed to build free‑form LHS: {e}")
            with colc2:
                if st.button("🗑️ Clear terms"):
                    st.session_state.free_terms = []

    # Actions
    if st.button("🚀 Build y(x) via Theorem 4.1", type="primary", use_container_width=True):
        with st.spinner("Applying Theorem 4.1..."):
            try:
                solution = run_with_timeout(_worker_theorem_4_1, timeout_sec,
                                            f_expr, α, β, int(n), 𝑀, "x", simplify_level)
                if apply_generator:
                    rhs = run_with_timeout(_worker_apply_lhs, timeout_sec,
                                           generator_lhs, solution, "x", "y", simplify_level)
                else:
                    rhs = sp.Symbol("RHS")

                st.session_state["last_solution"] = solution
                st.session_state["last_rhs"] = rhs
                st.session_state["last_lhs"] = generator_lhs

                st.markdown("### 📐 Equation")
                st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                st.markdown("### 💡 Solution")
                st.latex("y(x) = " + sp.latex(solution))

                try:
                    y0 = simplify_expr(solution.subs(x, 0), level=simplify_level)
                    st.markdown("### ⚙️ Initial value")
                    st.latex("y(0) = " + sp.latex(y0))
                except Exception:
                    pass

            except TimeoutError as te:
                st.error(str(te))
            except Exception as e:
                st.error(f"Failed to compute solution/RHS: {e}")

    if compute_mth and st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2", use_container_width=True):
        with st.spinner("Applying Theorem 4.2 (Stirling compact form)..."):
            try:
                y_m = run_with_timeout(_worker_theorem_4_2, timeout_sec,
                                       f_expr, α, β, int(n), int(m_order), "x", simplify_level)
                st.markdown("### 🔢 Derivative")
                st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
            except TimeoutError as te:
                st.error(str(te))
            except Exception as e:
                st.error(f"Failed to compute y^{m_order}(x): {e}")


# ============================================================================
# Other Pages (kept; small cleanups so they don't call add_term)
# ============================================================================

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

    st.subheader("📊 Recent Activity")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type", "order", "generator_number", "timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs generated yet. Head to **Apply Master Theorem** or **Generator Constructor**.")

def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    st.markdown(
        '<div class="info-box">Build custom generators by combining derivatives with transformations. '
        'Use this builder or jump to <b>Apply Master Theorem</b> to get y(x) and (optionally) build RHS=L[y].</div>',
        unsafe_allow_html=True,
    )

    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Generator constructor classes were not found in src/. You can still use the Free‑form builder on the theorem page.")
        return

    # Add terms UI
    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox(
                "Derivative Order",
                [0, 1, 2, 3, 4, 5],
                format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x,f"y⁽{x}⁾"),
            )
        with c2:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType], format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)

        c5, c6, c7 = st.columns(3)
        with c5:
            operator_type = st.selectbox("Operator Type", [t.value for t in OperatorType], format_func=lambda s: s.replace("_"," ").title())
        with c6:
            scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if operator_type in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay","advance"] else None

        if st.button("➕ Add Term", type="primary"):
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
            st.success("Term added.")

    # Current terms
    if st.session_state.generator_terms:
        st.subheader("📝 Current Generator Terms")
        for idx, term in enumerate(st.session_state.generator_terms):
            c1, c2 = st.columns([6, 1])
            with c1:
                desc = term.get_description() if hasattr(term, "get_description") else str(term)
                st.write(f"• {desc}")
            with c2:
                if st.button("❌", key=f"rm_{idx}"):
                    st.session_state.generator_terms.pop(idx)
                    st.experimental_rerun()

        # Build specification
        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(
                    terms=st.session_state.generator_terms,
                    name=f"Custom Generator {len(st.session_state.generated_odes) + 1}",
                )
                st.session_state.current_generator = gen_spec
                st.success("Generator specification created.")
                try:
                    st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to build specification: {e}")

    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None

def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning")
    st.markdown(
        '<div class="info-box">Learn generator patterns to create new families of ODEs. '
        'Your existing Trainer, VAE, Transformer code remains compatible.</div>',
        unsafe_allow_html=True,
    )

    if not MLTrainer:
        st.warning("MLTrainer not found in src/. Skipping ML features.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Patterns", len(st.session_state.generator_patterns))
    with c2: st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c3: st.metric("Training Epochs", len(st.session_state.training_history))
    with c4: st.metric("Model Status", "Trained" if st.session_state.get("ml_trained") else "Not Trained")

    model_type = st.selectbox(
        "Select ML Model",
        ["pattern_learner", "vae", "transformer"],
        format_func=lambda x: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[x],
    )

    with st.expander("🎯 Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", options=[0.0001,0.0005,0.001,0.005,0.01], value=0.001)
            samples = st.slider("Training Samples", 100, 5000, 1000)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning(f"Need at least 5 generated ODEs. Current: {len(st.session_state.generated_odes)}")
    else:
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training..."):
                try:
                    device = "cuda" if use_gpu and (torch and torch.cuda.is_available()) else "cpu"
                    trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
                    st.session_state.ml_trainer = trainer

                    prog = st.progress(0)
                    status = st.empty()

                    def progress_callback(epoch, total_epochs):
                        p = epoch/total_epochs
                        prog.progress(min(1.0, p))
                        status.text(f"Epoch {epoch}/{total_epochs}")

                    trainer.train(
                        epochs=epochs,
                        batch_size=batch_size,
                        samples=samples,
                        validation_split=validation_split,
                        progress_callback=progress_callback,
                    )
                    st.session_state.ml_trained = True
                    st.session_state.training_history = trainer.history
                    st.success("Model trained successfully.")

                    if trainer.history.get("train_loss"):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(trainer.history["train_loss"])+1)),
                            y=trainer.history["train_loss"], mode="lines", name="Training Loss"
                        ))
                        if trainer.history.get("val_loss"):
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(trainer.history["val_loss"])+1)),
                                y=trainer.history["val_loss"], mode="lines", name="Validation Loss"
                            ))
                        fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.session_state.get("ml_trained") and st.session_state.get("ml_trainer"):
        st.subheader("🎨 Generate Novel Patterns")
        c1, c2 = st.columns(2)
        with c1:
            num_generate = st.slider("Number to Generate", 1, 10, 1)
        with c2:
            if st.button("🎲 Generate Novel ODEs", type="primary"):
                with st.spinner("Generating..."):
                    try:
                        for i in range(num_generate):
                            res = st.session_state.ml_trainer.generate_new_ode()
                            if res:
                                st.success(f"Generated ODE {i+1}")
                                with st.expander(f"ODE {i+1}"):
                                    if "ode" in res:
                                        try:
                                            st.latex(sp.latex(res["ode"]))
                                        except Exception:
                                            st.code(str(res["ode"]))
                                    for k in ["type","order","function_used","description"]:
                                        if k in res:
                                            st.write(f"**{k}:** {res[k]}")
                                st.session_state.generated_odes.append(res)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    st.markdown(
        '<div class="info-box">Generate many ODEs quickly using your factories. '
        'If a particular factory class is not found, that subset will be skipped.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 500, 50)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_categories = st.multiselect("Function Categories", ["Basic","Special"], default=["Basic"])
        vary_params = st.checkbox("Vary Parameters", True)
    with c3:
        if vary_params:
            alpha_range = st.slider("α range", -10.0, 10.0, (-2.0, 2.0))
            beta_range  = st.slider("β range", 0.1, 10.0, (0.5, 2.0))
            n_range     = st.slider("n range", 1, 5, (1, 3))
        else:
            alpha_range = (1.0, 1.0)
            beta_range  = (1.0, 1.0)
            n_range     = (1, 1)

    with st.expander("⚙️ Advanced Options"):
        parallel = st.checkbox("Use Parallel Processing (factory-dependent)", True)
        export_format = st.selectbox("Export Format", ["JSON","CSV","LaTeX","All"])
        include_solutions = st.checkbox("Include Full Solutions", True)
        include_classification = st.checkbox("Include Classification", True)

    if st.button("🚀 Generate Batch", type="primary"):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            batch_results = []
            prog = st.progress(0)
            status = st.empty()

            all_functions = []
            if "Basic" in func_categories and st.session_state.get("basic_functions"):
                all_functions += st.session_state.basic_functions.get_function_names()
            if "Special" in func_categories and st.session_state.get("special_functions"):
                all_functions += st.session_state.special_functions.get_function_names()[:20]

            for i in range(num_odes):
                try:
                    prog.progress((i+1)/num_odes)
                    status.text(f"Generating ODE {i+1}/{num_odes}")

                    params = {
                        "alpha": np.random.uniform(*alpha_range),
                        "beta": np.random.uniform(*beta_range),
                        "n": int(np.random.randint(n_range[0], n_range[1]+1)),
                        "M": np.random.uniform(-1, 1),
                    }
                    if not all_functions:
                        st.warning("No function names available from libraries.")
                        break
                    func_name = np.random.choice(all_functions)

                    try:
                        f_z = st.session_state.basic_functions.get_function(func_name)
                    except Exception:
                        f_z = st.session_state.special_functions.get_function(func_name)

                    # Choose type/factory
                    gt = np.random.choice(gen_types)
                    res = {}
                    if gt == "linear":
                        if CompleteLinearGeneratorFactory:
                            factory = CompleteLinearGeneratorFactory()
                            gen_num = np.random.randint(1, 9)
                            if gen_num in [4, 5]:
                                params["a"] = np.random.uniform(1, 3)
                            res = factory.create(gen_num, f_z, **params)
                        elif LinearGeneratorFactory:
                            factory = LinearGeneratorFactory()
                            res = factory.create(1, f_z, **params)
                    else:
                        if CompleteNonlinearGeneratorFactory:
                            factory = CompleteNonlinearGeneratorFactory()
                            gen_num = np.random.randint(1, 11)
                            if gen_num in [1, 2, 4]:
                                params["q"] = int(np.random.randint(2, 6))
                            if gen_num in [2, 3, 5]:
                                params["v"] = int(np.random.randint(2, 6))
                            if gen_num in [4, 5, 9, 10]:
                                params["a"] = np.random.uniform(1, 3)
                            res = factory.create(gen_num, f_z, **params)
                        elif NonlinearGeneratorFactory:
                            factory = NonlinearGeneratorFactory()
                            res = factory.create(1, f_z, **params)

                    if not res:
                        continue

                    row = {
                        "ID": i+1,
                        "Type": res.get("type","unknown"),
                        "Generator": res.get("generator_number","?"),
                        "Function": func_name,
                        "Order": res.get("order", 0),
                        "α": round(params["alpha"], 4),
                        "β": round(params["beta"], 4),
                        "n": params["n"],
                    }
                    if include_solutions:
                        s = str(res.get("solution",""))
                        row["Solution"] = (s[:120] + "...") if len(s) > 120 else s
                    if include_classification:
                        row["Subtype"] = res.get("subtype", "standard")
                    batch_results.append(row)

                except Exception as e:
                    logger.debug(f"Failed to generate ODE {i+1}: {e}")

            st.session_state.batch_results.extend(batch_results)
            st.success(f"Generated {len(batch_results)} ODEs.")
            df = pd.DataFrame(batch_results)
            st.dataframe(df, use_container_width=True)

            st.subheader("📤 Export Results")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📊 Download CSV", csv, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            with c2:
                js = json.dumps(batch_results, indent=2, default=str).encode("utf-8")
                st.download_button("📄 Download JSON", js, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")
            with c3:
                if export_format in ["LaTeX","All"]:
                    latex = generate_batch_latex(batch_results)
                    st.download_button("📝 Download LaTeX", latex, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")
            with c4:
                if export_format == "All":
                    package = create_batch_package(batch_results, df)
                    st.download_button("📦 Download All (ZIP)", package, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip")

def novelty_detection_page():
    st.header("🔍 Novelty Detection")
    if not st.session_state.get("novelty_detector"):
        st.warning("Novelty detector not found. Skipping.")
        return

    input_method = st.radio("Input Method", ["Use Current Generator LHS", "Enter ODE Manually", "Select from Generated"])
    ode_to_analyze = None

    if input_method == "Use Current Generator LHS":
        gen_spec = st.session_state.get("current_generator")
        if gen_spec is not None and hasattr(gen_spec, "lhs"):
            ode_to_analyze = {"ode": gen_spec.lhs, "type":"custom", "order": getattr(gen_spec, "order", 2)}
        else:
            st.warning("No generator spec yet. Build one in Generator Constructor, or use free‑form in Apply Master Theorem.")

    elif input_method == "Enter ODE Manually":
        ode_str = st.text_area("Enter ODE (LaTeX or text):")
        if ode_str:
            ode_to_analyze = {"ode": ode_str, "type":"manual", "order": st.number_input("Order", 1, 10, 2)}

    else:
        if st.session_state.generated_odes:
            sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','Unknown')}")
            ode_to_analyze = st.session_state.generated_odes[sel]

    if ode_to_analyze and st.button("🔍 Analyze Novelty", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                analysis = st.session_state.novelty_detector.analyze(
                    ode_to_analyze, check_solvability=True, detailed=True
                )
                st.session_state.analysis_results.append({
                    "ode": ode_to_analyze, "analysis": analysis, "timestamp": datetime.now().isoformat()
                })
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Novelty", "🟢 NOVEL" if analysis.is_novel else "🔴 STANDARD")
                with c2: st.metric("Score", f"{analysis.novelty_score:.1f}/100")
                with c3: st.metric("Confidence", f"{analysis.confidence:.1%}")
                with st.expander("📊 Details", expanded=True):
                    st.write(f"Complexity: {analysis.complexity_level}")
                    st.write(f"Solvable by standard methods: {'Yes' if analysis.solvable_by_standard_methods else 'No'}")
                    if analysis.special_characteristics:
                        st.write("Special characteristics:")
                        for t in analysis.special_characteristics: st.write("•", t)
                    if analysis.recommended_methods:
                        st.write("Recommended methods:")
                        for t in analysis.recommended_methods[:5]: st.write("•", t)
                    if analysis.similar_known_equations:
                        st.write("Similar known equations:")
                        for t in analysis.similar_known_equations[:3]: st.write("•", t)
                if analysis.detailed_report:
                    st.download_button("📥 Download Report",
                                       analysis.detailed_report,
                                       f"novelty_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                       "text/plain")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

def analysis_classification_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    if not st.session_state.get("ode_classifier"):
        st.warning("ODEClassifier not found.")
        return

    st.subheader("📊 Generated ODEs Overview")
    summary = []
    for i, ode in enumerate(st.session_state.generated_odes[-50:]):
        summary.append({
            "ID": i+1,
            "Type": ode.get("type","Unknown"),
            "Order": ode.get("order",0),
            "Generator": ode.get("generator_number","N/A"),
            "Function": ode.get("function_used","Unknown"),
            "Timestamp": ode.get("timestamp","")[:19]
        })
    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Linear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="linear"))
    with c2:
        st.metric("Nonlinear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="nonlinear"))
    with c3:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        avg_order = np.mean(orders) if orders else 0
        st.metric("Average Order", f"{avg_order:.1f}")
    with c4:
        unique = len(set(o.get("function_used","") for o in st.session_state.generated_odes))
        st.metric("Unique Functions", unique)

    st.subheader("📊 Distributions")
    c1, c2 = st.columns(2)
    with c1:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        fig = px.histogram(orders, title="Order Distribution", nbins=10)
        fig.update_layout(xaxis_title="Order", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        types = [o.get("type","Unknown") for o in st.session_state.generated_odes]
        vc = pd.Series(types).value_counts()
        fig = px.pie(values=vc.values, names=vc.index, title="Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏷️ Classification Analysis")
    if st.button("Classify All ODEs", type="primary"):
        with st.spinner("Classifying..."):
            try:
                classifications = []
                for ode in st.session_state.generated_odes:
                    try:
                        result = st.session_state.ode_classifier.classify_ode(ode)
                        classifications.append(result)
                    except Exception:
                        classifications.append({})
                fields = [c.get("classification",{}).get("field","Unknown") for c in classifications if c]
                vc = pd.Series(fields).value_counts()
                fig = px.bar(x=vc.index, y=vc.values, title="Classification by Field")
                fig.update_layout(xaxis_title="Field", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Classification failed: {e}")

def physical_applications_page():
    st.header("🔬 Physical Applications")
    st.markdown(
        '<div class="info-box">Explore how generated ODEs relate to physics/engineering applications.</div>',
        unsafe_allow_html=True,
    )

    applications = {
        "Mechanics": [
            {"name":"Harmonic Oscillator","equation":"y'' + ω^2 y = 0","description":"Spring-mass systems"},
            {"name":"Damped Oscillator","equation":"y'' + 2γ y' + ω₀² y = 0","description":"Oscillators with friction"},
            {"name":"Forced Oscillator","equation":"y'' + 2γ y' + ω₀² y = F cos(ωt)","description":"Driven systems"},
        ],
        "Quantum Physics": [
            {"name":"Schrödinger (1D)","equation":"-ℏ²/(2m) y'' + V(x)y = Ey","description":"Quantum bound states"},
            {"name":"Particle in Box","equation":"y'' + (2mE/ℏ²) y = 0","description":"Infinite potential well"},
        ],
        "Thermodynamics": [
            {"name":"Heat Equation","equation":"∂T/∂t = α∇²T","description":"Heat diffusion"},
            {"name":"Newton Cooling","equation":"dT/dt = -k (T - T_env)","description":"Cooling processes"},
        ],
    }
    category = st.selectbox("Select Application Field", list(applications.keys()))
    for app in applications.get(category, []):
        with st.expander(f"📚 {app['name']}"):
            try:
                st.latex(app["equation"])
            except Exception:
                st.write(app["equation"])
            st.write("Description:", app["description"])

    st.subheader("🔗 Match Your ODEs to Applications")
    if st.session_state.generated_odes and st.session_state.get("ode_classifier"):
        sel = st.selectbox(
            "Select Generated ODE",
            range(len(st.session_state.generated_odes)),
            format_func=lambda i: f"ODE {i+1}: Type={st.session_state.generated_odes[i].get('type','?')}, Order={st.session_state.generated_odes[i].get('order',0)}"
        )
        if st.button("Find Applications"):
            ode = st.session_state.generated_odes[sel]
            try:
                result = st.session_state.ode_classifier.classify_ode(ode)
                apps = result.get("matched_applications", [])
                if apps:
                    st.success(f"Found {len(apps)} applications:")
                    for a in apps:
                        st.write(f"**{getattr(a,'name','?')}** ({getattr(a,'field','?')})")
                        if getattr(a, "description", None):
                            st.write("•", a.description)
                else:
                    st.info("No specific applications identified.")
            except Exception as e:
                st.error(f"Classification failed: {e}")
    else:
        st.info("Generate ODEs and ensure classifier is available.")

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
                elif plot_type == "Phase Portrait":
                    fig = go.Figure()  # Placeholder
                elif plot_type == "3D Surface":
                    fig = go.Figure()  # Placeholder
                else:
                    fig = go.Figure()  # Placeholder
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization failed: {e}")

def export_latex_page():
    st.header("📤 Export & LaTeX")
    st.markdown(
        '<div class="info-box">Export ODEs in publication‑ready LaTeX with exact symbolic expressions.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.generated_odes:
        st.warning("No ODEs to export.")
        return

    export_type = st.radio("Export Type", ["Single ODE","Multiple ODEs","Complete Report","Batch Export"])
    if export_type == "Single ODE":
        idx = st.selectbox(
            "Select ODE",
            range(len(st.session_state.generated_odes)),
            format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}"
        )
        ode = st.session_state.generated_odes[idx]
        st.subheader("📋 LaTeX Preview")
        latex_doc = LaTeXExporter.generate_latex_document(ode, include_preamble=False)
        st.code(latex_doc, language="latex")
        c1, c2, c3 = st.columns(3)
        with c1:
            full_latex = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
            st.download_button("📄 Download LaTeX", full_latex, f"ode_{idx+1}.tex", "text/x-latex")
        with c2:
            st.info("To get PDF, compile the .tex locally with pdflatex/xelatex.")
        with c3:
            package = LaTeXExporter.create_export_package(ode, include_extras=True)
            st.download_button("📦 Download Package", package, f"ode_package_{idx+1}.zip", "application/zip")

    elif export_type == "Multiple ODEs":
        sel = st.multiselect(
            "Select ODEs",
            range(len(st.session_state.generated_odes)),
            format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}"
        )
        if sel and st.button("Generate Multi-ODE Document"):
            parts = [r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\title{Collection of Generated ODEs}
\author{Master Generators System}
\date{\today}
\begin{document}
\maketitle
"""]
            for count, i in enumerate(sel, 1):
                parts.append(f"\\section{{ODE {count}}}")
                parts.append(LaTeXExporter.generate_latex_document(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            doc = "\n".join(parts)
            st.download_button("📄 Download Multi-ODE LaTeX", doc, f"multiple_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")

    elif export_type == "Complete Report":
        if st.button("Generate Complete Report"):
            report = generate_complete_report()
            st.download_button("📄 Download Complete Report", report, f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")

    else:
        st.subheader("📦 Batch Export Options")
        formats = st.multiselect("Export Formats", ["LaTeX","JSON","CSV","Python"], default=["LaTeX","JSON"])
        if st.button("Export All", type="primary"):
            export_all_formats(formats)

def examples_library_page():
    st.header("📚 Examples Library")
    st.markdown("Curated examples of linear/nonlinear/special-function generators.")
    # You can hook into your src examples if present; minimal static demo:
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")

def settings_page():
    st.header("⚙️ Settings")
    tabs = st.tabs(["General","Export","Advanced","About"])
    with tabs[0]:
        st.checkbox("Dark mode", False)
        if st.button("Save General Settings"): st.success("Saved.")
    with tabs[1]:
        include_preamble = st.checkbox("Include LaTeX preamble by default", value=True)
        if st.button("Save Export Settings"): st.success("Saved.")
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            cm = st.session_state.get("cache_manager")
            st.metric("Cache Size", len(getattr(cm,"memory_cache",{})) if cm else 0)
        with c2:
            if st.button("Clear Cache"):
                try:
                    st.session_state.cache_manager.clear()
                    st.success("Cache cleared.")
                except Exception:
                    st.info("No cache manager.")
        with c3:
            if st.button("Save Session"):
                ok = SessionStateManager.save_to_file()
                st.success("Session saved.") if ok else st.error("Failed to save.")
    with tabs[3]:
        st.markdown(
            "**Master Generators for ODEs** — complete system with Theorems 4.1 & 4.2, ML/DL, LaTeX, and novelty detection."
        )

def documentation_page():
    st.header("📖 Documentation")
    st.markdown(
        """
**Quick Start**
1. Go to **Apply Master Theorem**.
2. Pick f(z) from Basic/Special.
3. Set parameters (α,β,n,M) and choose **Exact (symbolic)**.
4. Click **Build y(x) via Theorem 4.1**.
5. (Optional) Build RHS by checking **Apply generator**.
6. (Optional) Compute **y^(m)(x)** via **Theorem 4.2**.
7. Export to LaTeX in **Export & LaTeX**.

**Free‑form LHS builder**
Compose terms like `sinh(y')`, `exp(y''''''')`, `ln(y'')`, `y(x/a+b)`, add powers, and apply outer derivatives.
"""
    )

# ============================================================================
# Helper utilities for batch/exports (unchanged structure)
# ============================================================================
def batch_generation_page():
    """
    Batch ODE Generation (patched):
    - Mode A: Apply Theorem 4.1 using the *active LHS* (Constructor or Free-form)
    - Mode B: Factories (fast), kept for backward compatibility
    - Symbolic/Exact toggle, seed, progress, exports, and ML/DL registration
    """
    st.header("📊 Batch ODE Generation")

    # ----------------------------
    # Controls
    # ----------------------------
    mode = st.radio(
        "Batch mode",
        ["Apply Theorem (use active LHS)", "Factories (fast)"],
        index=0,
        help="Apply Theorem: uses your active LHS (Constructor or Free‑form). Factories: generate via linear/nonlinear factories."
    )

    col_top = st.columns(4)
    with col_top[0]:
        num_odes = st.slider("Number of ODEs", 1, 500, 25)
    with col_top[1]:
        seed_val = st.number_input("Random seed (reproducible)", value=123, step=1)
    with col_top[2]:
        add_to_global = st.checkbox("Add to global ODEs (counts for ML/DL)", value=True)
    with col_top[3]:
        include_latex_files = st.checkbox("Include per‑ODE LaTeX in ZIP (first 50)", value=False)

    st.markdown("---")

    # Shared: function palette
    func_col = st.columns(3)
    with func_col[0]:
        func_libs = st.multiselect(
            "Function libraries",
            ["Basic", "Special"],
            default=["Basic"],
            help="Functions will be sampled uniformly from the selected libraries."
        )
    with func_col[1]:
        # Try to collect names from src libs; otherwise fallback to a safe palette
        names = []
        try:
            if "Basic" in func_libs and st.session_state.get("basic_functions"):
                names += st.session_state.basic_functions.get_function_names()
        except Exception:
            pass
        try:
            if "Special" in func_libs and st.session_state.get("special_functions"):
                names += st.session_state.special_functions.get_function_names()
        except Exception:
            pass
        if not names:
            names = ["exponential", "linear", "quadratic", "sin", "cos", "log", "tanh"]
        st.write(f"Found **{len(names)}** candidate f(z) functions.")
    with func_col[2]:
        st.write("")

    # ----------------------------
    # Mode A: Apply Theorem
    # ----------------------------
    if mode.startswith("Apply Theorem"):
        st.subheader("Parameters for Theorem 4.1 (batch sampling)")
        cA = st.columns(4)
        with cA[0]:
            alpha_range = st.text_input("α range [min,max]", "-2,2")
        with cA[1]:
            beta_range = st.text_input("β range [min,max] (β>0 recommended)", "0.5,2")
        with cA[2]:
            n_range = st.text_input("n range [min,max] (integers ≥1)", "1,3")
        with cA[3]:
            M_range = st.text_input("M range [min,max]", "-1,1")

        cA2 = st.columns(3)
        with cA2[0]:
            use_exact = st.checkbox("Exact (symbolic) parameters", value=False,
                                    help="Symbolic can be much slower in batch. Use sparingly.")
        with cA2[1]:
            simplify_out = st.checkbox("Simplify expressions", value=False,
                                       help="Turning this off can speed up large batches.")
        with cA2[2]:
            lhs_source = st.selectbox(
                "LHS source",
                ["Auto (active)", "Constructor", "Free‑form"],
                index=0,
                help="Auto uses whichever LHS is currently active. You can force Constructor or Free‑form here."
            )

        # Helper to parse ranges safely
        def _parse_range(txt, cast=float, default=(0, 1)):
            try:
                lo, hi = [cast(s.strip()) for s in txt.split(",")]
                if lo > hi:
                    lo, hi = hi, lo
                return lo, hi
            except Exception:
                return default

        a_rng = _parse_range(alpha_range, float, (-2.0, 2.0))
        b_rng = _parse_range(beta_range, float, (0.5, 2.0))
        n_rng = _parse_range(n_range, int, (1, 3))
        M_rng = _parse_range(M_range, float, (-1.0, 1.0))

        # Exactify helper
        def _to_exact(v):
            if use_exact:
                try:
                    return sp.nsimplify(v, rational=True)
                except Exception:
                    return sp.sympify(v)
            return sp.Float(v)

        # Resolve LHS spec
        def _get_active_spec_for_batch():
            # Respect override
            if lhs_source.startswith("Constructor"):
                return st.session_state.get("current_generator")
            if lhs_source.startswith("Free"):
                return st.session_state.get("freeform_gen_spec")
            # Auto (active)
            if callable(globals().get("get_active_generator_spec")):
                return get_active_generator_spec()
            # Fallback precedence: freeform -> constructor
            return st.session_state.get("freeform_gen_spec") or st.session_state.get("current_generator")

        # Apply LHS to y(x)
        def _apply_lhs(lhs_expr, y_expr, x):
            fn = globals().get("apply_lhs_to_solution")
            if callable(fn):
                return fn(lhs_expr, y_expr, x, y_name="y")
            # Minimal fallback (y(x) & its derivatives only)
            y = sp.Function("y")
            mapping = {y(x): y_expr}
            for d in lhs_expr.atoms(sp.Derivative):
                try:
                    if hasattr(d, "expr") and d.expr.func == y:
                        # order
                        k = 0
                        for v in d.variables:
                            if isinstance(v, sp.Symbol) and v == x:
                                k += 1
                            elif isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == x:
                                k += int(v[1])
                        mapping[d] = sp.diff(y_expr, (x, k))
                except Exception:
                    pass
            return lhs_expr.xreplace(mapping)

        # Register to global list
        def _register(rec: dict):
            fn = globals().get("register_generated_ode")
            if callable(fn):
                register_generated_ode(rec)
            else:
                st.session_state.setdefault("generated_odes", [])
                rec = dict(rec)
                rec.setdefault("type", "nonlinear")
                rec.setdefault("order", 0)
                rec.setdefault("timestamp", datetime.now().isoformat())
                rec["generator_number"] = len(st.session_state.generated_odes) + 1
                st.session_state.generated_odes.append(rec)

        # Launch
        if st.button("🚀 Run batch (Apply Theorem)", type="primary", use_container_width=True):
            if not names:
                st.error("No candidate function names found.")
                return

            spec = _get_active_spec_for_batch()
            if not (spec and hasattr(spec, "lhs") and spec.lhs is not None):
                st.error("No active LHS found. Please build/select a Constructor or Free‑form LHS before running this batch.")
                return

            np.random.seed(int(seed_val))
            x = sp.Symbol("x", real=True)
            batch_rows = []
            latex_files = []  # (filename, content)

            progress = st.progress(0.0)
            status = st.empty()

            for i in range(1, num_odes + 1):
                progress.progress(i / num_odes)
                status.text(f"Generating ODE {i}/{num_odes} …")
                try:
                    # Sample params
                    alpha = np.random.uniform(*a_rng)
                    beta = np.random.uniform(*b_rng)
                    n = int(np.random.randint(n_rng[0], n_rng[1] + 1))
                    M = np.random.uniform(*M_rng)

                    α, β, n_int, 𝑀 = _to_exact(alpha), _to_exact(beta), int(n), _to_exact(M)

                    # Sample function
                    fname = str(np.random.choice(names))
                    f_expr = get_function_expr("Basic" if "Basic" in func_libs else "Special", fname)

                    # Build solution and RHS
                    yx = theorem_4_1_solution_expr(f_expr, α, β, n_int, 𝑀, x)
                    if simplify_out:
                        yx = sp.simplify(yx)

                    lhs = spec.lhs
                    rhs = _apply_lhs(lhs, yx, x)
                    if simplify_out:
                        rhs = sp.simplify(rhs)

                    # Type/order inference (if helpers exist)
                    ode_type = "nonlinear"
                    ode_order = 0
                    if hasattr(spec, "is_linear"):
                        ode_type = "linear" if bool(getattr(spec, "is_linear")) else "nonlinear"
                    # If you have a helper like _infer_order_from_spec(spec), use it; otherwise quick scan:
                    try:
                        orders = []
                        for d in lhs.atoms(sp.Derivative):
                            # count occurrences of x in derivative variables
                            k = 0
                            for v in d.variables:
                                if isinstance(v, sp.Symbol) and v == x:
                                    k += 1
                                elif isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == x:
                                    k += int(v[1])
                            orders.append(k)
                        ode_order = max(orders) if orders else 0
                    except Exception:
                        pass

                    # Build record
                    rec = {
                        "generator": lhs,
                        "rhs": rhs,
                        "solution": yx,
                        "parameters": {"alpha": α, "beta": β, "n": n_int, "M": 𝑀},
                        "function_used": fname,
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

                    # Add to global set?
                    if add_to_global:
                        _register(rec)

                    # Row for table
                    batch_rows.append({
                        "ID": i,
                        "Function": fname,
                        "Order": ode_order,
                        "Type": ode_type,
                        "α": str(α), "β": str(β), "n": n_int, "M": str(𝑀),
                    })

                    # Optional per‑ODE LaTeX (limited to first 50)
                    if include_latex_files and i <= 50:
                        try:
                            from_latex = LaTeXExporter.generate_latex_document(rec, include_preamble=True)
                            latex_files.append((f"ode_{i:03d}.tex", from_latex))
                        except Exception as e:
                            logging.debug(f"latex build failed for {i}: {e}")

                except Exception as e:
                    logging.debug(f"Batch item {i} failed: {e}")

            # Present results
            st.success(f"✅ Completed {len(batch_rows)} ODEs (Apply Theorem).")
            df = pd.DataFrame(batch_rows)
            st.dataframe(df, use_container_width=True)

            # Exports
            exp = st.columns(4)
            with exp[0]:
                csv = df.to_csv(index=False)
                st.download_button("📊 CSV", csv, file_name=f"batch_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with exp[1]:
                j = json.dumps(batch_rows, indent=2, default=str)
                st.download_button("📄 JSON", j, file_name=f"batch_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            with exp[2]:
                latex_table = "\\begin{tabular}{|c|c|c|c|c|c|c|}\\hline ID & Function & Type & Order & $\\alpha$ & $\\beta$ & $n$\\\\\\hline\n" + \
                              "\n".join([f"{r['ID']} & {r['Function']} & {r['Type']} & {r['Order']} & {r['α']} & {r['β']} & {r['n']} \\\\" for r in batch_rows[:40]]) + \
                              "\n\\hline\\end{tabular}"
                st.download_button("📝 LaTeX table", latex_table, file_name=f"batch_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")
            with exp[3]:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("batch.csv", csv)
                    zf.writestr("batch.json", j)
                    zf.writestr("batch_table.tex", latex_table)
                    zf.writestr("README.txt", f"Batch (Apply Theorem) generated on {datetime.now().isoformat()}")
                    if include_latex_files and latex_files:
                        for fname, content in latex_files:
                            zf.writestr(f"odes/{fname}", content)
                zbuf.seek(0)
                st.download_button(
                    "📦 ZIP (all)",
                    zbuf.getvalue(),
                    file_name=f"batch_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

    # ----------------------------
    # Mode B: Factories (fast) — unchanged behavior
    # ----------------------------
    else:
        st.subheader("Factories mode (fast)")

        cc = st.columns(3)
        with cc[0]:
            gen_types = st.multiselect("Generator types", ["linear", "nonlinear"], default=["linear", "nonlinear"])
        with cc[1]:
            a_rng = st.slider("α range", -5.0, 5.0, (-2.0, 2.0))
            b_rng = st.slider("β range", 0.1, 5.0, (0.5, 2.0))
        with cc[2]:
            n_rng = st.slider("n range", 1, 5, (1, 3))

        with st.expander("Advanced", expanded=False):
            include_solutions = st.checkbox("Include solution preview", True)

        # layered factories (imported elsewhere in your app)
        CLF = globals().get("CompleteLinearGeneratorFactory")
        CNLF = globals().get("CompleteNonlinearGeneratorFactory")

        if st.button("🚀 Run batch (Factories)", type="primary", use_container_width=True):
            if not names:
                st.error("No candidate function names found.")
                return
            np.random.seed(int(seed_val))

            all_rows = []
            progress = st.progress(0.0)
            status = st.empty()

            for i in range(1, num_odes+1):
                progress.progress(i / num_odes)
                status.text(f"Generating ODE {i}/{num_odes} …")
                try:
                    params = {
                        "alpha": float(np.random.uniform(*a_rng)),
                        "beta": float(np.random.uniform(*b_rng)),
                        "n": int(np.random.randint(n_rng[0], n_rng[1] + 1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    gtype = str(np.random.choice(gen_types))
                    fname = str(np.random.choice(names))
                    f_expr = get_function_expr("Basic" if "Basic" in func_libs else "Special", fname)

                    if gtype == "linear" and CLF:
                        factory = CLF()
                        gnum = int(np.random.randint(1, 9))
                        if gnum in [4, 5]:
                            params["a"] = float(np.random.uniform(1, 3))
                        res = factory.create(gnum, f_expr, **params)
                    elif gtype == "nonlinear" and CNLF:
                        factory = CNLF()
                        gnum = int(np.random.randint(1, 11))
                        if gnum in [1, 2, 4]:
                            params["q"] = int(np.random.randint(2, 6))
                        if gnum in [2, 3, 5]:
                            params["v"] = int(np.random.randint(2, 6))
                        if gnum in [4, 5, 9, 10]:
                            params["a"] = float(np.random.uniform(1, 3))
                        res = factory.create(gnum, f_expr, **params)
                    else:
                        # fallback minimal record
                        res = {
                            "ode": None,
                            "solution": None,
                            "type": gtype,
                            "order": 0,
                            "generator_number": i,
                            "function_used": fname,
                            "subtype": "fallback",
                        }

                    # Register to global list if requested and result looks valid
                    if add_to_global and isinstance(res, dict):
                        rec = {
                            "generator": res.get("ode", sp.Symbol("LHS")),
                            "rhs": None,
                            "solution": res.get("solution"),
                            "parameters": {k: params[k] for k in ["alpha", "beta", "n", "M"] if k in params},
                            "function_used": fname,
                            "type": res.get("type", gtype),
                            "order": res.get("order", 0),
                            "classification": res.get("classification", {}),
                            "initial_conditions": {},
                            "timestamp": datetime.now().isoformat(),
                            "title": "Master Generators ODE System",
                        }
                        register_generated_ode(rec)

                    row = {
                        "ID": i,
                        "Type": res.get("type", gtype),
                        "Generator": res.get("generator_number"),
                        "Function": fname,
                        "Order": res.get("order", 0),
                        "α": round(params["alpha"], 3),
                        "β": round(params["beta"], 3),
                        "n": params["n"],
                    }
                    if include_solutions:
                        row["Solution"] = str(res.get("solution"))[:120] + "..."
                    all_rows.append(row)

                except Exception as e:
                    logging.debug(f"Factories batch item {i} failed: {e}")

            st.success(f"✅ Completed {len(all_rows)} ODEs (Factories).")
            df = pd.DataFrame(all_rows)
            st.dataframe(df, use_container_width=True)

            exp = st.columns(4)
            with exp[0]:
                csv = df.to_csv(index=False)
                st.download_button("📊 CSV", csv, file_name=f"batch_fact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            with exp[1]:
                j = json.dumps(all_rows, indent=2, default=str)
                st.download_button("📄 JSON", j, file_name=f"batch_fact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            with exp[2]:
                latex_table = "\\begin{tabular}{|c|c|c|c|c|}\\hline ID & Type & Gen & Function & Order \\\\ \\hline\n" + \
                              "\n".join([f"{r['ID']} & {r['Type']} & {r['Generator']} & {r['Function']} & {r['Order']} \\\\" for r in all_rows[:40]]) + \
                              "\n\\hline\\end{tabular}"
                st.download_button("📝 LaTeX table", latex_table, file_name=f"batch_fact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", mime="text/x-latex")
            with exp[3]:
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("batch.csv", csv)
                    zf.writestr("batch.json", j)
                    zf.writestr("batch_table.tex", latex_table)
                    zf.writestr("README.txt", f"Batch (Factories) generated on {datetime.now().isoformat()}")
                zbuf.seek(0)
                st.download_button(
                    "📦 ZIP (all)",
                    zbuf.getvalue(),
                    file_name=f"batch_fact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )


# ============================================================================
# Main App
# ============================================================================
def main():
    SessionStateManager.initialize()

    st.markdown(
        """
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">By Mohammad Abu-Ghuwaleh• Free‑form generators • ML/DL • Export • Novelty</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

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
        page_apply_master_theorem()
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
