"""
Master Generators for ODEs — Complete App (Symbolic-Exact Mode + All Services Intact)

- Robust import discovery for factories (linear/nonlinear) no matter which src module they live in.
- Theorem 4.1 (Master Theorem) implemented symbolically with SymPy.
- Exact-mode toggle to keep parameters and output fully symbolic (no long decimals for e, pi).
- RHS construction: apply LHS operator L[y] by substituting symbolic y(x) (supports derivatives and shifts).
- ML/DL, Novelty detection, Export/LaTeX, Batch generation, Visualization pages included.
"""

from __future__ import annotations

import os
import sys
import io
import json
import time
import math
import base64
import pickle
import zipfile
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sympy as sp

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------- Logging ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# ----------------------------- Ensure src on path ----------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ----------------------------- Imports from src ------------------------------
HAVE_SRC = True

# Gracious import probing because your project may place factories in different modules.
def _probe_factories():
    """
    Probe multiple candidate modules for the same factories.
    Returns a dict with classes (or None if not found).
    """
    out = {
        "MasterGenerator": None,
        "EnhancedMasterGenerator": None,
        "CompleteMasterGenerator": None,
        "LinearGeneratorFactory": None,
        "CompleteLinearGeneratorFactory": None,
        "NonlinearGeneratorFactory": None,
        "CompleteNonlinearGeneratorFactory": None,
        "GeneratorConstructor": None,
        "GeneratorSpecification": None,
        "DerivativeTerm": None,
        "DerivativeType": None,
        "OperatorType": None,
        "MasterTheoremSolver": None,
        "MasterTheoremParameters": None,
        "ExtendedMasterTheorem": None,
        "ODEClassifier": None,
        "PhysicalApplication": None,
        "BasicFunctions": None,
        "SpecialFunctions": None,
        "GeneratorPatternLearner": None,
        "GeneratorVAE": None,
        "GeneratorTransformer": None,
        "create_model": None,
        "MLTrainer": None,
        "ODEDataset": None,
        "ODEDataGenerator": None,
        "GeneratorPattern": None,
        "GeneratorPatternNetwork": None,
        "GeneratorLearningSystem": None,
        "ODENoveltyDetector": None,
        "NoveltyAnalysis": None,
        "ODETokenizer": None,
        "ODETransformer": None,
        "Settings": None,
        "AppConfig": None,
        "CacheManager": None,
        "cached": None,
        "ParameterValidator": None,
        "UIComponents": None,
    }

    # core generators / theorem likely in these
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

    # lazy import helper
    def _imp(module: str):
        try:
            return __import__(module, fromlist=["*"])
        except Exception as e:
            logger.debug(f"Module not found or failed: {module}: {e}")
            return None

    mods = {m: _imp(m) for m in candidates}

    def _set(name, module_name_list):
        for m in module_name_list:
            mod = mods.get(m)
            if mod is None:
                continue
            if hasattr(mod, name):
                out[name] = getattr(mod, name)
                return

    # Try to map names to modules (in priority order)
    _set("MasterGenerator", ["src.generators.master_generator"])
    _set("EnhancedMasterGenerator", ["src.generators.master_generator"])
    _set("CompleteMasterGenerator", ["src.generators.master_generator"])
    _set("LinearGeneratorFactory", ["src.generators.linear_generators", "src.generators.master_generator"])
    _set("CompleteLinearGeneratorFactory", ["src.generators.master_generator", "src.generators.linear_generators"])
    _set("NonlinearGeneratorFactory", ["src.generators.nonlinear_generators", "src.generators.master_generator"])
    _set("CompleteNonlinearGeneratorFactory", ["src.generators.master_generator", "src.generators.nonlinear_generators"])

    _set("GeneratorConstructor", ["src.generators.generator_constructor"])
    _set("GeneratorSpecification", ["src.generators.generator_constructor"])
    _set("DerivativeTerm", ["src.generators.generator_constructor"])
    _set("DerivativeType", ["src.generators.generator_constructor"])
    _set("OperatorType", ["src.generators.generator_constructor"])

    _set("MasterTheoremSolver", ["src.generators.master_theorem"])
    _set("MasterTheoremParameters", ["src.generators.master_theorem"])
    _set("ExtendedMasterTheorem", ["src.generators.master_theorem"])

    _set("ODEClassifier", ["src.generators.ode_classifier"])
    _set("PhysicalApplication", ["src.generators.ode_classifier"])

    _set("BasicFunctions", ["src.functions.basic_functions"])
    _set("SpecialFunctions", ["src.functions.special_functions"])

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

    _set("Settings", ["src.utils.config"])
    _set("AppConfig", ["src.utils.config"])

    _set("CacheManager", ["src.utils.cache"])
    _set("cached", ["src.utils.cache"])

    _set("ParameterValidator", ["src.utils.validators"])
    _set("UIComponents", ["src.ui.components"])

    return out

try:
    PROBED = _probe_factories()
except Exception as e:
    HAVE_SRC = False
    PROBED = {}
    logger.warning(f"Failed to probe src modules: {e}")

# convenient aliases (may be None if not found)
MasterGenerator = PROBED.get("MasterGenerator")
EnhancedMasterGenerator = PROBED.get("EnhancedMasterGenerator")
CompleteMasterGenerator = PROBED.get("CompleteMasterGenerator")

LinearGeneratorFactory = PROBED.get("LinearGeneratorFactory")
CompleteLinearGeneratorFactory = PROBED.get("CompleteLinearGeneratorFactory")
NonlinearGeneratorFactory = PROBED.get("NonlinearGeneratorFactory")
CompleteNonlinearGeneratorFactory = PROBED.get("CompleteNonlinearGeneratorFactory")

GeneratorConstructor = PROBED.get("GeneratorConstructor")
GeneratorSpecification = PROBED.get("GeneratorSpecification")
DerivativeTerm = PROBED.get("DerivativeTerm")
DerivativeType = PROBED.get("DerivativeType")
OperatorType = PROBED.get("OperatorType")

MasterTheoremSolver = PROBED.get("MasterTheoremSolver")
MasterTheoremParameters = PROBED.get("MasterTheoremParameters")
ExtendedMasterTheorem = PROBED.get("ExtendedMasterTheorem")

ODEClassifier = PROBED.get("ODEClassifier")
PhysicalApplication = PROBED.get("PhysicalApplication")

BasicFunctions = PROBED.get("BasicFunctions")
SpecialFunctions = PROBED.get("SpecialFunctions")

GeneratorPatternLearner = PROBED.get("GeneratorPatternLearner")
GeneratorVAE = PROBED.get("GeneratorVAE")
GeneratorTransformer = PROBED.get("GeneratorTransformer")
create_model = PROBED.get("create_model")

MLTrainer = PROBED.get("MLTrainer")
ODEDataset = PROBED.get("ODEDataset")
ODEDataGenerator = PROBED.get("ODEDataGenerator")

GeneratorPattern = PROBED.get("GeneratorPattern")
GeneratorPatternNetwork = PROBED.get("GeneratorPatternNetwork")
GeneratorLearningSystem = PROBED.get("GeneratorLearningSystem")

ODENoveltyDetector = PROBED.get("ODENoveltyDetector")
NoveltyAnalysis = PROBED.get("NoveltyAnalysis")
ODETokenizer = PROBED.get("ODETokenizer")
ODETransformer = PROBED.get("ODETransformer")

Settings = PROBED.get("Settings")
AppConfig = PROBED.get("AppConfig")

CacheManager = PROBED.get("CacheManager")
cached = PROBED.get("cached")

ParameterValidator = PROBED.get("ParameterValidator")
UIComponents = PROBED.get("UIComponents")

if not HAVE_SRC:
    st.warning("This module requires the src/ package. Ensure the project ZIP is extracted with src/ present.")

# ----------------------------- Streamlit Page --------------------------------
st.set_page_config(
    page_title="Master Generators ODE System - Exact Symbolic Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Custom CSS (short) ----------------------------
st.markdown(
    """
    <style>
    .main-header {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        padding:2rem;border-radius:16px;margin-bottom:1rem;color:#fff;text-align:center;}
    .metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        color:#fff;padding:1rem;border-radius:12px;text-align:center;}
    .info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
        border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:1rem 0;}
    .result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
        border:2px solid #4caf50;padding:1rem;border-radius:12px;margin:1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# LATEX EXPORTER (symbolic beautify)
# =============================================================================
class LaTeXExporter:
    @staticmethod
    def sympy_to_latex(expr: Any) -> str:
        if expr is None:
            return ""
        try:
            # Try to bring back E, pi etc.
            expr = sp.nsimplify(expr, [sp.E, sp.pi, sp.I], rational=True)
        except Exception:
            pass
        try:
            latex_str = sp.latex(expr)
            return latex_str.replace(r"\left(", "(").replace(r"\right)", ")")
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
            ks = list(ics.items())
            for i, (k, v) in enumerate(ks):
                sep = r" \\" if i < len(ics) - 1 else ""
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
                "Compile with: pdflatex ode_document.tex\n",
            )
        buf.seek(0)
        return buf.getvalue()

# =============================================================================
# Symbolic helpers (Master Theorem 4.1 + operator application)
# =============================================================================
from sympy.core.function import AppliedUndef  # for y(x) pattern

def get_function_expr(source: str, name: str) -> sp.Expr:
    """
    Obtain f(z) as a SymPy expression from BasicFunctions / SpecialFunctions.
    The src libraries differ in return types across projects; we normalize.
    """
    z = sp.Symbol("z")
    lib = None
    if source == "Basic" and BasicFunctions:
        lib = BasicFunctions()
    elif source == "Special" and SpecialFunctions:
        lib = SpecialFunctions()
    else:
        raise ValueError("Function library not available.")

    f_obj = None
    if hasattr(lib, "get_function"):
        f_obj = lib.get_function(name)
    elif hasattr(lib, "functions") and name in getattr(lib, "functions"):
        f_obj = lib.functions[name]
    else:
        raise ValueError(f"Function '{name}' not found.")

    # Try to convert to f(z) expression
    try:
        if callable(f_obj):
            test = f_obj(z)
            return sp.sympify(test)
        else:
            return sp.sympify(f_obj)
    except Exception:
        # last attempt: treat the name itself symbolically
        return sp.Symbol(name)

def theorem_4_1_solution_expr(f_expr: sp.Expr, alpha, beta, n: int, M, x: sp.Symbol) -> sp.Expr:
    """
    Symbolic solution according to the Master Theorem 4.1 used in your app content.
    y(x) = π/(2n) * sum_{s=1}^{n} [ 2 f(α + β) - ( ψ_s + φ_s ) ] + π M
    where
      ω_s = (2s-1)π/(2n),
      ψ_s = f(α + β*exp(i x cos ω_s - x sin ω_s)),
      φ_s = f(α + β*exp(-i x cos ω_s - x sin ω_s))
    """
    z = sp.Symbol("z")
    terms = []
    for s in range(1, n + 1):
        ω = sp.Rational(2 * s - 1, 2 * n) * sp.pi
        ψ_s = f_expr.subs(z, alpha + beta * sp.exp(sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
        φ_s = f_expr.subs(z, alpha + beta * sp.exp(-sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
        terms.append(2 * f_expr.subs(z, alpha + beta) - (ψ_s + φ_s))
    y = sp.pi / (2 * n) * sum(terms) + sp.pi * M
    # Try to keep exact/compact form
    try:
        y = sp.nsimplify(y, [sp.E, sp.pi, sp.I], rational=True)
    except Exception:
        pass
    return sp.simplify(y)

def apply_lhs_to_solution(lhs_expr: sp.Expr, solution_y: sp.Expr, x: sp.Symbol, y_name: str = "y") -> sp.Expr:
    """
    Compute RHS = LHS[y] by substituting y(x) and its derivatives into lhs_expr.
    Supports y(x), y'(x), y''(x), ..., y(x/a) (a could be any subexpr).
    """
    subs_map: Dict[sp.Expr, sp.Expr] = {}

    # 1) Substitute any y(arg) -> solution_y.subs(x, arg)
    for f in lhs_expr.atoms(AppliedUndef):
        if f.func.__name__ == y_name and len(f.args) == 1:
            arg = f.args[0]
            subs_map[f] = sp.simplify(solution_y.subs(x, arg))

    # 2) Substitute derivatives Derivative(y(arg), x, k) -> d^k/dx^k [ solution_y(x)->x=arg ]
    for d in lhs_expr.atoms(sp.Derivative):
        base = d.expr
        if isinstance(base, AppliedUndef) and base.func.__name__ == y_name and len(base.args) == 1:
            arg = base.args[0]
            # order of derivative with respect to x
            try:
                # SymPy newer versions
                order = sum(c for v, c in d.variable_count if v == x)
            except Exception:
                # Fallback for older
                order = sum(1 for v in d.variables if v == x)
            subs_map[d] = sp.diff(solution_y.subs(x, arg), (x, order))

    try:
        rhs = sp.simplify(lhs_expr.xreplace(subs_map))
    except Exception:
        rhs = sp.simplify(lhs_expr.subs(subs_map))
    return rhs

# =============================================================================
# Session State Initialization
# =============================================================================
def init_state():
    if "generated_odes" not in st.session_state:
        st.session_state.generated_odes = []
    if "generator_terms" not in st.session_state:
        st.session_state.generator_terms = []   # DerivativeTerm list (from src)
    if "current_generator" not in st.session_state:
        st.session_state.current_generator = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = []
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
    if "theorem_solver" not in st.session_state and MasterTheoremSolver:
        st.session_state.theorem_solver = MasterTheoremSolver()
    if "extended_theorem" not in st.session_state and ExtendedMasterTheorem:
        st.session_state.extended_theorem = ExtendedMasterTheorem()

# =============================================================================
# UI Pages
# =============================================================================
def page_dashboard():
    st.markdown('<div class="main-header"><h2>🔬 Master Generators for ODEs</h2>'
                '<p>Exact Symbolic Mode + All Services</p></div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Generated ODEs</h4><h2>{len(st.session_state.generated_odes)}</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Batch Results</h4><h2>{len(st.session_state.batch_results)}</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4>ML Trained</h4><h2>{"Yes" if st.session_state.ml_trained else "No"}</h2></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h4>Analyses</h4><h2>{len(st.session_state.analysis_results)}</h2></div>', unsafe_allow_html=True)
    if st.session_state.generated_odes:
        st.subheader("Recent ODEs")
        df = pd.DataFrame(st.session_state.generated_odes)[["type","order","function_used","timestamp"]].tail(6)
        st.dataframe(df, use_container_width=True)

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    if not GeneratorConstructor or not DerivativeTerm:
        st.warning("GeneratorConstructor / DerivativeTerm not available from src/.")
        return
    constructor = GeneratorConstructor()
    with st.expander("➕ Add Generator Term", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], index=0)
        with col2:
            func_type = st.selectbox(
                "Function Type",
                [t.value if hasattr(t, "value") else str(t) for t in (DerivativeType if hasattr(DerivativeType, "__iter__") else list(DerivativeType))] 
                if DerivativeType else ["identity","sin","cos","exp","log","power"]
            )
        with col3:
            coefficient = st.number_input("Coefficient", value=1.0, step=0.1)
        with col4:
            power = st.number_input("Power", min_value=1, max_value=5, value=1)

        col1b, col2b, col3b = st.columns(3)
        with col1b:
            operator_type = st.selectbox(
                "Operator Type",
                [t.value if hasattr(t, "value") else str(t) for t in (OperatorType if hasattr(OperatorType, "__iter__") else list(OperatorType))]
                if OperatorType else ["standard","delay","advance"]
            )
        with col2b:
            scaling = st.number_input("Scaling (a) [for delay/advance]", value=2.0, step=0.1)
        with col3b:
            shift = st.number_input("Shift [for delay/advance]", value=0.0, step=0.1)

        if st.button("Add Term", type="primary", use_container_width=True):
            try:
                term = DerivativeTerm(
                    derivative_order=deriv_order,
                    coefficient=coefficient,
                    power=power,
                    function_type=DerivativeType(func_type) if DerivativeType else func_type,
                    operator_type=OperatorType(operator_type) if OperatorType else operator_type,
                    scaling=scaling if operator_type in ("delay","advance") else None,
                    shift=shift if operator_type in ("delay","advance") else None,
                )
                st.session_state.generator_terms.append(term)
                st.success("Term added.")
            except Exception as e:
                st.error(f"Failed to add term: {e}")

    if st.session_state.generator_terms:
        st.subheader("Current Terms")
        for idx, term in enumerate(st.session_state.generator_terms):
            cols = st.columns([8,1])
            with cols[0]:
                desc = getattr(term, "get_description", lambda: str(term))()
                st.info(desc)
            with cols[1]:
                if st.button("❌", key=f"del_{idx}"):
                    st.session_state.generator_terms.pop(idx)
                    st.experimental_rerun()

        if st.button("Build Generator Spec", type="primary", use_container_width=True):
            try:
                for t in st.session_state.generator_terms:
                    constructor.add_term(t)
                spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Generator #{len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = spec
                st.success("Generator Specification created.")
                if hasattr(spec, "lhs"):
                    st.latex(sp.latex(spec.lhs) + " = \\text{RHS}")
            except Exception as e:
                st.error(f"Failed to build specification: {e}")

        if st.button("Clear All Terms", use_container_width=True):
            st.session_state.generator_terms = []
            st.session_state.current_generator = None
            st.experimental_rerun()

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem 4.1")
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
        alpha = st.text_input("α", value="1")      # text => exact by default
        beta  = st.text_input("β", value="1")
        n     = st.number_input("n (integer ≥ 1)", min_value=1, max_value=10, value=1, step=1)
        M     = st.text_input("M", value="0")

    # -------------------- EXACT MODE BLOCK (as requested) --------------------
    use_exact = st.checkbox("Exact (symbolic) parameters", value=True)

    def to_exact(v):
        try:
            # Convert floats like 1.0 -> 1, 0.5 -> 1/2; keep symbols exact
            return sp.nsimplify(v, rational=True)
        except Exception:
            return sp.sympify(v)

    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        with st.spinner("Applying Master Theorem 4.1 and constructing RHS..."):
            try:
                x = sp.Symbol("x", real=True)

                # Parse user parameters with exact toggle
                alpha_in = alpha.strip()
                beta_in  = beta.strip()
                M_in     = M.strip()

                if use_exact:
                    α = to_exact(alpha_in)
                    β = to_exact(beta_in)
                    𝑀 = to_exact(M_in)
                else:
                    α = sp.Float(alpha_in)
                    β = sp.Float(beta_in)
                    𝑀 = sp.Float(M_in)

                # Resolve f(z) symbolically
                f_expr = get_function_expr(source_lib, func_name)

                # Build y(x) via Theorem 4.1
                solution = theorem_4_1_solution_expr(f_expr, α, β, int(n), 𝑀, x)

                # Build RHS = LHS[y] using current generator if available
                gen_spec = st.session_state.get("current_generator")
                if gen_spec is not None and hasattr(gen_spec, "lhs") and gen_spec.lhs is not None:
                    try:
                        rhs = apply_lhs_to_solution(gen_spec.lhs, solution, x, y_name="y")
                        generator_lhs = gen_spec.lhs
                    except Exception as e:
                        logger.warning(f"Failed to apply generator LHS to solution: {e}")
                        # fallback RHS (still symbolic)
                        z = sp.Symbol("z")
                        rhs = sp.pi * (f_expr.subs(z, α + β) - f_expr.subs(z, α + β*sp.exp(-x))) + sp.pi * 𝑀
                        generator_lhs = sp.Symbol("LHS")
                else:
                    # fallback: if no LHS available, present a constructed RHS from the s=1 pattern (n=1)
                    z = sp.Symbol("z")
                    rhs = sp.pi * (f_expr.subs(z, α + β) - f_expr.subs(z, α + β*sp.exp(-x))) + sp.pi * 𝑀
                    generator_lhs = sp.Symbol("LHS")

                # Classification (best effort using src classifier)
                classification = {}
                if ODEClassifier:
                    try:
                        classifier = st.session_state.ode_classifier
                        # Provide a small dict for the classifier
                        c_in = {"ode": generator_lhs, "solution": solution, "rhs": rhs}
                        c_out = classifier.classify_ode(c_in)
                        classification = c_out.get("classification", {})
                        if "field" not in classification:
                            classification["field"] = c_out.get("field", "Mathematical Physics")
                        if "applications" not in classification:
                            classification["applications"] = c_out.get("applications", ["Research Equation"])
                        classification["order"] = classification.get("order", _guess_order_from_lhs(generator_lhs, x))
                        classification["type"] = classification.get("type", "Linear" if _is_linear_lhs(generator_lhs, x) else "Nonlinear")
                    except Exception:
                        classification = {
                            "field": "Mathematical Physics",
                            "applications": ["Research Equation"],
                            "order": _guess_order_from_lhs(generator_lhs, x),
                            "type": "Linear" if _is_linear_lhs(generator_lhs, x) else "Nonlinear",
                        }
                else:
                    classification = {
                        "field": "Mathematical Physics",
                        "applications": ["Research Equation"],
                        "order": _guess_order_from_lhs(generator_lhs, x),
                        "type": "Linear" if _is_linear_lhs(generator_lhs, x) else "Nonlinear",
                    }

                # Prepare save bundle
                result = {
                    "generator": generator_lhs,
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

                # Display
                st.success("✅ ODE generated successfully.")
                tabs = st.tabs(["📐 Equation", "💡 Solution", "🏷️ Classification", "📤 Export"])
                with tabs[0]:
                    st.markdown("### Full ODE")
                    st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                with tabs[1]:
                    st.markdown("### Exact Solution")
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

def _guess_order_from_lhs(lhs: Any, x: sp.Symbol) -> int:
    try:
        max_order = 0
        for d in sp.preorder_traversal(lhs):
            if isinstance(d, sp.Derivative):
                try:
                    ord_ = sum(c for v, c in d.variable_count if v == x)
                except Exception:
                    ord_ = sum(1 for v in d.variables if v == x)
                if ord_ > max_order:
                    max_order = ord_
        return int(max_order)
    except Exception:
        return 0

def _is_linear_lhs(lhs: Any, x: sp.Symbol) -> bool:
    # Best-effort quick test: if y and its derivatives only appear linearly.
    # This is conservative (might mark some nonlinear as linear if wrapped oddly).
    try:
        y = sp.Function("y")
        # If any powers of y or derivatives appear (>1), likely nonlinear
        powers = [p for p in sp.preorder_traversal(lhs) if isinstance(p, sp.Pow)]
        for p in powers:
            base = p.base
            if base.has(y(x)) or any(isinstance(a, sp.Derivative) and a.expr == y(x) for a in base.atoms(sp.Derivative)):
                if p.exp != 1:
                    return False
        # if sin(y'), exp(y''), log(y), etc. => nonlinear
        for fun in lhs.atoms(AppliedUndef):
            if fun.func.__name__ != "y" and fun.args and any(arg.has(y(x)) for arg in fun.args):
                return False
        return True
    except Exception:
        return False

def page_ml():
    st.header("🤖 ML Pattern Learning")
    if not MLTrainer:
        st.info("MLTrainer not found in src/. Skipping ML page.")
        return

    model_type = st.selectbox(
        "Model",
        ["pattern_learner", "vae", "transformer"],
        format_func=lambda s: {"pattern_learner": "Pattern Learner", "vae": "VAE", "transformer": "Transformer"}[s],
    )
    colA, colB, colC = st.columns(3)
    with colA:
        epochs = st.slider("Epochs", 10, 500, 100, 5)
        batch_size = st.slider("Batch Size", 8, 128, 32, 8)
    with colB:
        lr = st.select_slider("Learning Rate", options=[1e-4,5e-4,1e-3,5e-3,1e-2], value=1e-3)
        samples = st.slider("Training Samples", 100, 5000, 1000, 100)
    with colC:
        val_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
        use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate at least 5 ODEs to train ML.")
        return

    if st.button("🚀 Train", type="primary"):
        device = "cuda" if (use_gpu and _torch_cuda_available()) else "cpu"
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
            st.success("Training finished.")
            # Plot history if present
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
                        with st.expander(f"Generated ODE #{len(st.session_state.generated_odes)}", expanded=False):
                            if "ode" in res:
                                st.latex(sp.latex(res["ode"]) if not isinstance(res["ode"], str) else res["ode"])
                            st.write({k: v for k, v in res.items() if k not in ["ode","solution"]})
                except Exception as e:
                    st.warning(f"Failed to generate #{i+1}: {e}")

def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def page_batch():
    st.header("📊 Batch ODE Generation")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_odes = st.slider("Number of ODEs", 5, 300, 20)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with col2:
        func_categories = st.multiselect("Function Categories", ["Basic","Special"], default=["Basic"])
        include_solutions = st.checkbox("Include Solutions", True)
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
        func_pool = []
        if "Basic" in func_categories and hasattr(st.session_state, "basic_functions"):
            try:
                func_pool += st.session_state.basic_functions.get_function_names()
            except Exception:
                pass
        if "Special" in func_categories and hasattr(st.session_state, "special_functions"):
            try:
                func_pool += st.session_state.special_functions.get_function_names()
            except Exception:
                pass
        func_pool = func_pool[:50]  # limit

        for i in range(num_odes):
            try:
                α = float(np.random.uniform(alpha_min, alpha_max))
                β = float(np.random.uniform(beta_min, beta_max))
                n = int(np.random.randint(1, 3))
                M = float(np.random.uniform(-1, 1))
                source = "Basic" if (np.random.rand() < 0.7) else "Special"
                if not func_pool:
                    break
                f_name = str(np.random.choice(func_pool))

                x = sp.Symbol("x", real=True)
                f_expr = get_function_expr(source, f_name)
                y = theorem_4_1_solution_expr(f_expr, sp.nsimplify(α), sp.nsimplify(β), n, sp.nsimplify(M), x)
                lhs = st.session_state.current_generator.lhs if st.session_state.get("current_generator") else sp.Symbol("LHS")
                rhs = apply_lhs_to_solution(lhs, y, x, y_name="y") if lhs != sp.Symbol("LHS") else sp.Symbol("RHS")

                rec = {
                    "ID": i+1,
                    "Type": "linear" if _is_linear_lhs(lhs, x) else "nonlinear",
                    "Generator": st.session_state.get("current_generator", {"name":"-" }).get("name","-") if hasattr(st.session_state.get("current_generator"), "name") else "-",
                    "Function": f_name,
                    "Order": _guess_order_from_lhs(lhs, x),
                    "α": α, "β": β, "n": n, "M": M,
                }
                if include_solutions:
                    rec["Solution"] = sp.sstr(y)[:120] + "..."
                results.append(rec)
            except Exception as e:
                logger.debug(f"Batch item failed: {e}")
                continue

        st.session_state.batch_results.extend(results)
        st.success(f"Generated {len(results)} records.")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        colA, colB, colC = st.columns(3)
        with colA:
            st.download_button("📊 Download CSV", df.to_csv(index=False), file_name="batch_odes.csv", mime="text/csv")
        with colB:
            st.download_button("📄 Download JSON", json.dumps(results, indent=2), file_name="batch_odes.json", mime="application/json")
        with colC:
            latex_tbl = _latex_table(results)
            st.download_button("📝 Download LaTeX Table", latex_tbl, file_name="batch_odes.tex", mime="text/x-latex")

def _latex_table(rows: List[Dict[str, Any]]) -> str:
    lines = [r"\begin{tabular}{|c|c|c|c|c|}\hline",
             r"ID & Type & Generator & Function & Order \\ \hline"]
    for r in rows[:30]:
        lines.append(f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\")
    lines.append(r"\hline\end{tabular}")
    return "\n".join(lines)

def page_novelty():
    st.header("🔍 Novelty Detection")
    if not ODENoveltyDetector:
        st.info("Novelty detector not available in src/.")
        return
    det = st.session_state.novelty_detector
    mode = st.radio("Input", ["Use Current Generator", "Enter ODE LaTeX/Text", "Select from Generated"], index=0)
    ode_obj = None
    if mode == "Use Current Generator":
        spec = st.session_state.get("current_generator")
        if spec and hasattr(spec, "lhs"):
            ode_obj = {"ode": spec.lhs, "type": "custom", "order": _guess_order_from_lhs(spec.lhs, sp.Symbol("x"))}
        else:
            st.warning("No generator available. Build one in the constructor.")
    elif mode == "Enter ODE LaTeX/Text":
        ode_str = st.text_area("Enter ODE", "")
        if ode_str.strip():
            ode_obj = {"ode": ode_str, "type": "manual", "order": st.number_input("Order", 1, 10, 2)}
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
            st.write("**Special characteristics:**", analysis.special_characteristics[:10])
            st.write("**Recommended methods:**", analysis.recommended_methods[:10])
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
        {
            "Type": rec.get("type",""),
            "Function": rec.get("function_used",""),
            "Order": rec.get("order",""),
            "Timestamp": rec.get("timestamp","")
        } for rec in st.session_state.generated_odes
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
        plot_type = st.selectbox("Plot", ["Solution"], index=0)
        if st.button("Generate Plot", type="primary"):
            xs = np.linspace(-5, 5, 600)
            # Sympy lambdify with safe modules
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
    col1, col2 = st.columns(2)
    with col1:
        tex = LaTeXExporter.document_for_ode(rec, include_preamble=True)
        st.download_button("📄 Download LaTeX", tex, file_name=f"ode_{idx+1}.tex", mime="text/x-latex")
    with col2:
        pkg = LaTeXExporter.zip_package(rec)
        st.download_button("📦 Download Package (ZIP)", pkg, file_name=f"ode_package_{idx+1}.zip", mime="application/zip")

# =============================================================================
# Main
# =============================================================================
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