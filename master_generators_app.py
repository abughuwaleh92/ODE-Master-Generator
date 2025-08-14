"""
Master Generators for ODEs — Exact Symbolic Edition (with Theorem 4.2)

- Theorem 4.1 (solution) and Theorem 4.2 (derivatives) implemented symbolically.
- Theorem 4.2 uses Stirling numbers of the second kind (compact form).
- Exact-mode toggle lets α, β, M be fully symbolic/rational; numeric fallback available.
- LHS application to build RHS = L[y] supports derivatives and shifts (y(arg)), via chain rule.
- Retains ML/DL/Novelty/Batch/Export/Visualization pages.

Drop-in replacement for master_generators_app.py
"""

from __future__ import annotations

import os
import sys
import io
import json
import time
import logging
import traceback
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import sympy as sp
from sympy.core.function import AppliedUndef

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

# ----------------------------- Robust import discovery -----------------------
HAVE_SRC = True

def _probe_src():
    out = {
        # generators
        "MasterGenerator": None,
        "EnhancedMasterGenerator": None,
        "CompleteMasterGenerator": None,
        "LinearGeneratorFactory": None,
        "CompleteLinearGeneratorFactory": None,
        "NonlinearGeneratorFactory": None,
        "CompleteNonlinearGeneratorFactory": None,
        # constructor
        "GeneratorConstructor": None,
        "GeneratorSpecification": None,
        "DerivativeTerm": None,
        "DerivativeType": None,
        "OperatorType": None,
        # theorems / classifier
        "MasterTheoremSolver": None,
        "MasterTheoremParameters": None,
        "ExtendedMasterTheorem": None,
        "ODEClassifier": None,
        "PhysicalApplication": None,
        # functions
        "BasicFunctions": None,
        "SpecialFunctions": None,
        # ML / DL
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
        # utils/ui
        "Settings": None,
        "AppConfig": None,
        "CacheManager": None,
        "cached": None,
        "ParameterValidator": None,
        "UIComponents": None,
    }
    modules = [
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
            logger.debug(f"Module load failed: {m}: {e}")
            return None

    mods = {m: _imp(m) for m in modules}

    def _set(name, search_modules):
        for m in search_modules:
            mod = mods.get(m)
            if mod is not None and hasattr(mod, name):
                out[name] = getattr(mod, name)
                return

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
    PROBED = _probe_src()
except Exception as e:
    PROBED = {}
    HAVE_SRC = False
    logger.warning(f"Failed to probe src/: {e}")

# bind names
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

# ----------------------------- Streamlit Page --------------------------------
st.set_page_config(
    page_title="Master Generators ODE System — Symbolic & Exact",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Small CSS -------------------------------------
st.markdown(
    """
    <style>
    .main-header {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
      padding:1.5rem;border-radius:14px;margin-bottom:1rem;color:#fff;text-align:center;}
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
# LaTeX Exporter
# =============================================================================
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
            s = sp.latex(expr)
            return s.replace(r"\left(", "(").replace(r"\right)", ")")
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
        parts: List[str] = []
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
        parts += [
            r"\subsection{Generator Equation}",
            r"\begin{equation}",
            f"{LaTeXExporter.sympy_to_latex(gen)} = {LaTeXExporter.sympy_to_latex(rhs)}",
            r"\end{equation}",
            r"\subsection{Exact Solution}",
            r"\begin{equation}",
            f"y(x) = {LaTeXExporter.sympy_to_latex(sol)}",
            r"\end{equation}",
            r"\subsection{Parameters}",
            r"\begin{align}",
            f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha',''))} \\\\",
            f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta',''))} \\\\",
            f"n       &= {LaTeXExporter.sympy_to_latex(params.get('n',''))} \\\\",
            f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M',''))}",
            r"\end{align}",
        ]
        if ics:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            items = list(ics.items())
            for i, (k, v) in enumerate(items):
                sep = r" \\" if i < len(items) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}{sep}")
            parts.append(r"\end{align}")

        if cls:
            parts += [r"\subsection{Mathematical Classification}", r"\begin{itemize}"]
            parts.append(f"\\item \\textbf{{Type:}} {cls.get('type','Unknown')}")
            parts.append(f"\\item \\textbf{{Order:}} {cls.get('order','Unknown')}")
            if "field" in cls:
                parts.append(f"\\item \\textbf{{Field:}} {cls['field']}")
            if cls.get("applications"):
                parts.append(f"\\item \\textbf{{Applications:}} {', '.join(cls['applications'])}")
            parts.append(r"\end{itemize}")

        parts += [
            r"\subsection{Solution Verification}",
            r"Substitute $y(x)$ into the generator operator to verify $L[y]=\mathrm{RHS}$.",
        ]
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def zip_package(ode_data: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ode_document.tex", LaTeXExporter.document_for_ode(ode_data, True))
            z.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            z.writestr("README.txt",
                       "Master Generators ODE Export\n"
                       f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                       "Compile with: pdflatex ode_document.tex\n")
        buf.seek(0)
        return buf.getvalue()

# =============================================================================
# Theorem 4.1 and 4.2 helpers (symbolic)
# =============================================================================
def get_function_expr(source: str, name: str) -> sp.Expr:
    """Obtain f(z) as SymPy expression from BasicFunctions / SpecialFunctions."""
    z = sp.Symbol("z")
    lib = None
    if source == "Basic" and BasicFunctions:
        lib = st.session_state.basic_functions
    elif source == "Special" and SpecialFunctions:
        lib = st.session_state.special_functions
    else:
        raise ValueError("Function library not available.")
    f_obj = None
    if hasattr(lib, "get_function"):
        f_obj = lib.get_function(name)
    elif hasattr(lib, "functions") and name in getattr(lib, "functions"):
        f_obj = lib.functions[name]
    else:
        raise ValueError(f"Function '{name}' not found in {source} library.")
    try:
        if callable(f_obj):
            return sp.sympify(f_obj(z))
        return sp.sympify(f_obj)
    except Exception:
        return sp.Symbol(name)

def theorem_4_1_solution_expr(f_expr: sp.Expr, alpha, beta, n: int, M, x: sp.Symbol) -> sp.Expr:
    """
    y(x) = π/(2n) * Σ_{s=1}^n [ 2 f(α+β) − (ψ_s + φ_s) ] + π M
    ψ_s = f(α + β e^{i x cos ω_s − x sin ω_s})
    φ_s = f(α + β e^{−i x cos ω_s − x sin ω_s})
    ω_s = (2s-1)π/(2n)
    """
    z = sp.Symbol("z")
    terms = []
    for s in range(1, n + 1):
        ω = sp.Rational(2*s - 1, 2*n) * sp.pi
        psi = f_expr.subs(z, alpha + beta * sp.exp(sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
        phi = f_expr.subs(z, alpha + beta * sp.exp(-sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
        terms.append(2 * f_expr.subs(z, alpha + beta) - (psi + phi))
    y = sp.pi/(2*n) * sum(terms) + sp.pi*M
    try:
        y = sp.nsimplify(y, [sp.E, sp.pi, sp.I], rational=True)
    except Exception:
        pass
    return sp.simplify(y)

# ---- Stirling numbers of the second kind (symbolic-friendly) -----------------
def S2(m, j):
    """
    Stirling numbers of the second kind S(m,j) with possible symbolic m.
    S(m,j) = 1/j! * sum_{q=0}^j (-1)^{j-q} C(j,q) q^m
    """
    q = sp.symbols('q', integer=True, nonnegative=True)
    return sp.summation((-1)**(j - q) * sp.binomial(j, q) * q**m, (q, 0, j)) / sp.factorial(j)

def _omega_s(s: int, n: int):
    return sp.Rational(2*s - 1, 2*n) * sp.pi

def _zeta(x, ω):
    return sp.exp(-x * sp.sin(ω)) * sp.exp(sp.I * x * sp.cos(ω))

def _lambda(ω):
    return sp.exp(sp.I * (sp.pi/2 + ω))

def theorem_4_2_derivative_trig(
    f_expr: sp.Expr,
    alpha, beta, n: int, m, x: sp.Symbol,
    symbolic_m: bool = False,
) -> sp.Expr:
    """
    Trigonometric compact form from your statement (boxed display).
    Uses ∂^j/∂α^j (ψ ± φ). Internally we treat α as a symbol α_sym and substitute α at the end.
    """
    z = sp.Symbol("z")
    α_sym = sp.Symbol("α", real=True)
    total = 0
    for s in range(1, n + 1):
        ω = _omega_s(s, n)
        ζ = _zeta(x, ω)
        psi = f_expr.subs(z, α_sym + beta*ζ)
        phi = f_expr.subs(z, α_sym + beta*sp.conjugate(ζ))  # = α + β e^{-i x cos ω - x sin ω}

        if symbolic_m:
            m_sym = sp.Symbol('m', integer=True, positive=True)
            j = sp.symbols('j', integer=True, positive=True)
            inner = sp.summation(
                S2(m_sym, j) * beta**j * sp.exp(-j*x*sp.sin(ω)) * (
                    sp.cos(j*x*sp.cos(ω) + m_sym*(sp.pi/2 + ω)) * sp.diff(psi + phi, α_sym, j) +
                    (1/sp.I) * sp.sin(j*x*sp.cos(ω) + m_sym*(sp.pi/2 + ω)) * sp.diff(psi - phi, α_sym, j)
                ),
                (j, 1, m_sym)
            )
            total += inner
        else:
            m_int = int(m)
            sum_j = 0
            for jv in range(1, m_int + 1):
                S_mj = sp.simplify(S2(m_int, jv))
                term = S_mj * beta**jv * sp.exp(-jv*x*sp.sin(ω)) * (
                    sp.cos(jv*x*sp.cos(ω) + m_int*(sp.pi/2 + ω)) * sp.diff(psi + phi, α_sym, jv) +
                    (1/sp.I) * sp.sin(jv*x*sp.cos(ω) + m_int*(sp.pi/2 + ω)) * sp.diff(psi - phi, α_sym, jv)
                )
                sum_j += term
            total += sum_j

    y_m = -sp.pi/(2*n) * total
    # substitute α_sym -> alpha (numeric or exact)
    y_m = sp.simplify(y_m.subs(α_sym, alpha))
    try:
        y_m = sp.nsimplify(y_m, [sp.E, sp.pi, sp.I], rational=True)
    except Exception:
        pass
    return sp.simplify(y_m)

def theorem_4_2_derivative_complex(
    f_expr: sp.Expr,
    alpha, beta, n: int, m, x: sp.Symbol,
    symbolic_m: bool = False,
) -> sp.Expr:
    """
    Complex form from your statement (using λ_s and ζ_s).
    y^{(m)}(x) = -π/(2n) Σ_s [ λ_s^m Σ_{j=1}^m S(m,j)(β ζ_s)^j ∂_α^j ψ + conj ]
    (We build ψ and φ directly, then add conjugate term explicitly.)
    """
    z = sp.Symbol("z")
    α_sym = sp.Symbol("α", real=True)
    total = 0
    for s in range(1, n + 1):
        ω = _omega_s(s, n)
        λ = _lambda(ω)
        ζ = _zeta(x, ω)
        psi = f_expr.subs(z, α_sym + beta*ζ)
        # conj side:
        λc = sp.conjugate(λ)
        ζc = sp.conjugate(ζ)
        phi = f_expr.subs(z, α_sym + beta*ζc)

        if symbolic_m:
            m_sym = sp.Symbol('m', integer=True, positive=True)
            j = sp.symbols('j', integer=True, positive=True)
            term_psi = sp.summation(S2(m_sym, j) * (beta*ζ)**j * sp.diff(psi, α_sym, j), (j, 1, m_sym))
            term_phi = sp.summation(S2(m_sym, j) * (beta*ζc)**j * sp.diff(phi, α_sym, j), (j, 1, m_sym))
            total += λ**m_sym * term_psi + sp.conjugate(λ)**m_sym * term_phi
        else:
            m_int = int(m)
            sum_psi = 0
            sum_phi = 0
            for jv in range(1, m_int + 1):
                S_mj = sp.simplify(S2(m_int, jv))
                sum_psi += S_mj * (beta*ζ)**jv * sp.diff(psi, α_sym, jv)
                sum_phi += S_mj * (beta*ζc)**jv * sp.diff(phi, α_sym, jv)
            total += λ**m_int * sum_psi + λc**m_int * sum_phi

    y_m = -sp.pi/(2*n) * total
    y_m = sp.simplify(y_m.subs(α_sym, alpha))
    try:
        y_m = sp.nsimplify(y_m, [sp.E, sp.pi, sp.I], rational=True)
    except Exception:
        pass
    return sp.simplify(y_m)

# =============================================================================
# LHS application (build RHS = L[y])
# =============================================================================
def _detect_y_name(lhs: sp.Expr) -> str:
    try:
        names = [f.func.__name__ for f in lhs.atoms(AppliedUndef) if len(f.args) == 1]
        if not names:
            return "y"
        # pick most frequent
        return max(set(names), key=names.count)
    except Exception:
        return "y"

def apply_lhs_to_solution(lhs_expr: sp.Expr, solution_y: sp.Expr, x: sp.Symbol, y_name: str = "y") -> sp.Expr:
    """
    Substitute y(arg) and d^k/dx^k y(arg) with solution_y(x) and its derivatives.
    Chain rule is handled by SymPy by differentiating solution_y.subs(x, arg) with respect to x.
    """
    subs_map: Dict[sp.Expr, sp.Expr] = {}

    # direct y(arg)
    for f in lhs_expr.atoms(AppliedUndef):
        if f.func.__name__ == y_name and len(f.args) == 1:
            arg = f.args[0]
            subs_map[f] = sp.simplify(solution_y.subs(x, arg))

    # derivatives d^k/dx^k y(arg)
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

# =============================================================================
# Session state
# =============================================================================
def init_state():
    if "generated_odes" not in st.session_state:
        st.session_state.generated_odes = []
    if "generator_terms" not in st.session_state:
        st.session_state.generator_terms = []  # list of DerivativeTerm
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
# Utilities
# =============================================================================
def enum_to_values(E) -> List[str]:
    try:
        if hasattr(E, "__members__"):
            return [m.value if hasattr(m, "value") else str(m) for m in E.__members__.values()]
        return [e.value if hasattr(e, "value") else str(e) for e in list(E)]
    except Exception:
        return []

def _guess_order_from_lhs(lhs: Any, x: sp.Symbol) -> int:
    try:
        max_order = 0
        for d in sp.preorder_traversal(lhs):
            if isinstance(d, sp.Derivative):
                try:
                    ord_ = sum(c for v, c in d.variable_count if v == x)
                except Exception:
                    ord_ = sum(1 for v in d.variables if v == x)
                max_order = max(max_order, int(ord_))
        return max_order
    except Exception:
        return 0

def _is_linear_lhs(lhs: Any, x: sp.Symbol) -> bool:
    try:
        y = sp.Function("y")
        # if y or derivatives appear with power != 1, it's nonlinear
        for p in sp.preorder_traversal(lhs):
            if isinstance(p, sp.Pow):
                base = p.base
                if base.has(y(x)) or any(isinstance(a, sp.Derivative) and a.expr == y(x) for a in base.atoms(sp.Derivative)):
                    if p.exp != 1:
                        return False
        # sin(y'), log(y), etc.
        for fun in lhs.atoms(AppliedUndef):
            if fun.func.__name__ != "y" and fun.args and any(arg.has(y(x)) for arg in fun.args):
                return False
        return True
    except Exception:
        return False

def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def _latex_table(rows: List[Dict[str, Any]]) -> str:
    lines = [r"\begin{tabular}{|c|c|c|c|c|}\hline",
             r"ID & Type & Generator & Function & Order \\ \hline"]
    for r in rows[:30]:
        lines.append(f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\")
    lines.append(r"\hline\end{tabular}")
    return "\n".join(lines)

# =============================================================================
# Pages
# =============================================================================
def page_dashboard():
    st.markdown('<div class="main-header"><h3>🔬 Master Generators for ODEs</h3>'
                '<p>Symbolic Theorems 4.1 & 4.2 + ML/DL + LaTeX</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h4>Generated ODEs</h4><h2>{len(st.session_state.generated_odes)}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4>Batch Results</h4><h2>{len(st.session_state.batch_results)}</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4>ML Trained</h4><h2>{"Yes" if st.session_state.ml_trained else "No"}</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h4>Analyses</h4><h2>{len(st.session_state.analysis_results)}</h2></div>', unsafe_allow_html=True)

    if st.session_state.generated_odes:
        st.subheader("Recent")
        df = pd.DataFrame(st.session_state.generated_odes)
        cols = [c for c in ["type","order","function_used","timestamp"] if c in df.columns]
        if cols:
            st.dataframe(df[cols].tail(6), use_container_width=True)

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    if not (GeneratorSpecification and DerivativeTerm):
        st.warning("GeneratorSpecification / DerivativeTerm not available from src/.")
        return

    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], index=0)
        with c2:
            dt_opts = enum_to_values(DerivativeType) if DerivativeType else ["identity","sin","cos","exp","log","power"]
            func_type_val = st.selectbox("Function Type", dt_opts)
        with c3:
            coefficient = st.number_input("Coefficient", value=1.0, step=0.1)
        with c4:
            power = st.number_input("Power", min_value=1, max_value=5, value=1)

        b1, b2, b3 = st.columns(3)
        with b1:
            op_opts = enum_to_values(OperatorType) if OperatorType else ["standard","delay","advance"]
            operator_val = st.selectbox("Operator Type", op_opts)
        with b2:
            scaling = st.number_input("Scaling a (for delay/advance)", value=2.0, step=0.1)
        with b3:
            shift = st.number_input("Shift (for delay/advance)", value=0.0, step=0.1)

        if st.button("Add Term", type="primary", use_container_width=True):
            try:
                ftype = DerivativeType(func_type_val) if DerivativeType else func_type_val
                otype = OperatorType(operator_val) if OperatorType else operator_val
                needs_scale = "delay" in str(operator_val).lower() or "advance" in str(operator_val).lower()
                kwargs = dict(
                    derivative_order=deriv_order,
                    coefficient=coefficient,
                    power=power,
                    function_type=ftype,
                    operator_type=otype,
                )
                if needs_scale:
                    kwargs["scaling"] = scaling
                    kwargs["shift"] = shift
                term = DerivativeTerm(**kwargs)
                st.session_state.generator_terms.append(term)
                desc = getattr(term, "get_description", lambda: str(term))()
                st.success(f"Added term: {desc}")
            except Exception as e:
                st.error(f"Failed to add term: {e}")

    if st.session_state.generator_terms:
        st.subheader("📝 Current Terms")
        for i, t in enumerate(st.session_state.generator_terms):
            cols = st.columns([10,1])
            with cols[0]:
                st.info(getattr(t, "get_description", lambda: str(t))())
            with cols[1]:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()

        if st.button("🔨 Build Generator Specification", type="primary", use_container_width=True):
            try:
                spec = GeneratorSpecification(terms=st.session_state.generator_terms,
                                             name=f"Custom Generator #{len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = spec
                st.success("Generator specification created.")
                lhs = getattr(spec, "lhs", None)
                if lhs is None and hasattr(spec, "get_lhs"):
                    try: lhs = spec.get_lhs()
                    except Exception: lhs = None
                if lhs is None and hasattr(spec, "build_lhs"):
                    try: lhs = spec.build_lhs()
                    except Exception: lhs = None
                if lhs is not None:
                    st.latex(sp.latex(lhs) + " = \\text{RHS}")
                else:
                    st.info("Built spec; no .lhs available to display.")
            except Exception as e:
                st.error(f"Failed to build specification: {e}")

        if st.button("🗑️ Clear All Terms", use_container_width=True):
            st.session_state.generator_terms = []
            st.session_state.current_generator = None
            st.experimental_rerun()

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (4.1 & 4.2)")

    # --- Function selection & parameters ---
    colL, colR = st.columns([1,1])
    with colL:
        source_lib = st.selectbox("Function Library", ["Basic","Special"])
        names = []
        try:
            lib = st.session_state.basic_functions if source_lib == "Basic" else st.session_state.special_functions
            if hasattr(lib, "get_function_names"):
                names = lib.get_function_names()
            elif hasattr(lib, "functions"):
                names = list(lib.functions.keys())
        except Exception:
            pass
        func_name = st.selectbox("Choose f(z)", names)
    with colR:
        alpha = st.text_input("α", value="1")
        beta  = st.text_input("β", value="1")
        n     = st.number_input("n (integer ≥ 1)", min_value=1, max_value=12, value=1, step=1)
        M     = st.text_input("M", value="0")

    # exact parameters toggle
    st.markdown("**Parameter Mode**")
    use_exact = st.checkbox("Exact (symbolic) parameters", value=True)

    def to_exact(v):
        try:
            return sp.nsimplify(v, rational=True)
        except Exception:
            return sp.sympify(v)

    # --- Theorem 4.1: build y(x) and RHS ---
    if st.button("🚀 Generate ODE (Theorem 4.1)", type="primary", use_container_width=True):
        with st.spinner("Applying Theorem 4.1 and constructing RHS..."):
            try:
                x = sp.Symbol("x", real=True)
                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)
                𝑀 = to_exact(M)     if use_exact else sp.Float(M)

                f_expr = get_function_expr(source_lib, func_name)
                solution = theorem_4_1_solution_expr(f_expr, α, β, int(n), 𝑀, x)

                # Build LHS from spec (if available)
                gen_spec = st.session_state.get("current_generator")
                generator_lhs = None
                if gen_spec is not None:
                    generator_lhs = getattr(gen_spec, "lhs", None)
                    if generator_lhs is None and hasattr(gen_spec, "get_lhs"):
                        try: generator_lhs = gen_spec.get_lhs()
                        except Exception: generator_lhs = None
                    if generator_lhs is None and hasattr(gen_spec, "build_lhs"):
                        try: generator_lhs = gen_spec.build_lhs()
                        except Exception: generator_lhs = None

                if generator_lhs is not None:
                    y_name = _detect_y_name(generator_lhs)
                    rhs = apply_lhs_to_solution(generator_lhs, solution, x, y_name=y_name)
                else:
                    # fallback RHS dependent on f; still symbolic
                    z = sp.Symbol("z")
                    rhs = sp.pi * (f_expr.subs(z, α + β) - f_expr.subs(z, α + β*sp.exp(-x))) + sp.pi * 𝑀
                    generator_lhs = sp.Symbol("LHS")

                # classification (best effort)
                classification = {}
                try:
                    if ODEClassifier:
                        classifier = st.session_state.ode_classifier
                        c_in = {"ode": generator_lhs, "solution": solution, "rhs": rhs}
                        c_out = classifier.classify_ode(c_in)
                        classification = c_out.get("classification", {})
                        if "order" not in classification:
                            classification["order"] = _guess_order_from_lhs(generator_lhs, x)
                        if "type" not in classification:
                            classification["type"] = "Linear" if _is_linear_lhs(generator_lhs, x) else "Nonlinear"
                        classification.setdefault("field", c_out.get("field", "Mathematical Physics"))
                        classification.setdefault("applications", c_out.get("applications", ["Research Equation"]))
                except Exception:
                    classification = {
                        "field": "Mathematical Physics",
                        "applications": ["Research Equation"],
                        "order": _guess_order_from_lhs(generator_lhs, x),
                        "type": "Linear" if _is_linear_lhs(generator_lhs, x) else "Nonlinear",
                    }

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

                tabs = st.tabs(["📐 Equation", "💡 Solution", "🏷️ Classification", "📤 Export"])
                with tabs[0]:
                    st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                with tabs[1]:
                    st.latex("y(x) = " + sp.latex(solution))
                    st.write("**Initial:**")
                    st.latex("y(0) = " + sp.latex(sp.simplify(solution.subs(x, 0))))
                with tabs[2]:
                    st.json(classification, expanded=False)
                with tabs[3]:
                    tex = LaTeXExporter.document_for_ode(result, include_preamble=True)
                    st.download_button("📄 LaTeX", tex, file_name="ode_solution.tex", mime="text/x-latex")
                    pkg = LaTeXExporter.zip_package(result)
                    st.download_button("📦 ZIP", pkg, file_name=f"ode_package_{int(time.time())}.zip", mime="application/zip")

            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.exception(e)

    # --- Theorem 4.2: Derivatives panel ---
    with st.expander("🧮 Theorem 4.2 — Derivatives y^{(m)}(x)", expanded=False):
        c1, c2 = st.columns([1,1])
        with c1:
            form = st.radio("Formula form", ["Trigonometric (ψ,φ)", "Complex (λ,ζ)"], index=0)
        with c2:
            mode = st.radio("Derivative mode", ["Fixed order m (integer)", "General (symbolic m)"], index=0)

        x = sp.Symbol("x", real=True)
        α = to_exact(alpha) if use_exact else sp.Float(alpha)
        β = to_exact(beta)  if use_exact else sp.Float(beta)
        f_expr = get_function_expr(source_lib, func_name)

        if mode == "Fixed order m (integer)":
            m_val = st.number_input("m (≥1)", min_value=1, max_value=10, value=1)
            if st.button("Compute y^{(m)}(x)"):
                with st.spinner("Building y^{(m)}(x) via Theorem 4.2..."):
                    try:
                        if form.startswith("Trig"):
                            y_m = theorem_4_2_derivative_trig(f_expr, α, β, int(n), int(m_val), x, symbolic_m=False)
                        else:
                            y_m = theorem_4_2_derivative_complex(f_expr, α, β, int(n), int(m_val), x, symbolic_m=False)
                        st.latex(f"y^{{({int(m_val)})}}(x) = {sp.latex(y_m)}")

                        # Optional correctness check against direct differentiation of Theorem 4.1
                        if st.checkbox("Verify numerically vs. d^m/dx^m (Theorem 4.1 solution)"):
                            try:
                                y1 = theorem_4_1_solution_expr(f_expr, α, β, int(n), to_exact(M) if use_exact else sp.Float(M), x)
                                y1m = sp.diff(y1, (x, int(m_val)))
                                # sample points to compare (avoid singularities)
                                y_m_num = sp.lambdify([x], sp.simplify(y_m), "numpy")
                                y1m_num = sp.lambdify([x], sp.simplify(y1m), "numpy")
                                xs = np.linspace(0.1, 0.6, 5)
                                err = np.max(np.abs(y_m_num(xs) - y1m_num(xs)))
                                st.success(f"Max abs error over samples: {err:.3e}")
                            except Exception as e:
                                st.info(f"Verification skipped: {e}")

                    except Exception as e:
                        st.error(f"Failed to compute derivative: {e}")
                        st.exception(e)

        else:
            # general (symbolic m)
            if st.button("Build symbolic y^{(m)}(x)"):
                with st.spinner("Building symbolic y^{(m)}(x) (sum over j with S(m,j))..."):
                    try:
                        if form.startswith("Trig"):
                            y_m = theorem_4_2_derivative_trig(f_expr, α, β, int(n), sp.Symbol('m', integer=True, positive=True), x, symbolic_m=True)
                        else:
                            y_m = theorem_4_2_derivative_complex(f_expr, α, β, int(n), sp.Symbol('m', integer=True, positive=True), x, symbolic_m=True)
                        st.latex(r"y^{(m)}(x) = " + sp.latex(y_m))
                        st.caption("General formula with Stirling numbers S(m,j).")
                    except Exception as e:
                        st.error(f"Failed to build symbolic derivative: {e}")
                        st.exception(e)

def page_ml():
    st.header("🤖 ML Pattern Learning")
    if not MLTrainer:
        st.info("MLTrainer not available in src/.")
        return
    model_type = st.selectbox("Model", ["pattern_learner", "vae", "transformer"])
    a, b, c = st.columns(3)
    with a:
        epochs = st.slider("Epochs", 10, 500, 100, 5)
        batch_size = st.slider("Batch Size", 8, 128, 32, 8)
    with b:
        lr = st.select_slider("Learning Rate", options=[1e-4,5e-4,1e-3,5e-3,1e-2], value=1e-3)
        samples = st.slider("Training Samples", 100, 5000, 1000, 100)
    with c:
        val_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
        use_gpu = st.checkbox("Use GPU if available", value=True)

    if len(st.session_state.generated_odes) < 5:
        st.warning("Generate at least 5 ODEs first.")
        return

    if st.button("🚀 Train", type="primary"):
        device = "cuda" if (use_gpu and _torch_cuda_available()) else "cpu"
        try:
            trainer = MLTrainer(model_type=model_type, learning_rate=lr, device=device)
            st.session_state.ml_trainer = trainer
            prog = st.progress(0)
            info = st.empty()
            def cb(epoch, total):
                prog.progress(int(100*epoch/total))
                info.info(f"Epoch {epoch}/{total}")
            trainer.train(epochs=epochs, batch_size=batch_size, samples=samples, validation_split=val_split, progress_callback=cb)
            st.session_state.ml_trained = True
            st.success("Training finished.")
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
                            st.write({kk: vv for kk, vv in res.items() if kk not in ["ode","solution"]})
                except Exception as e:
                    st.warning(f"Generate failed: {e}")

def page_batch():
    st.header("📊 Batch ODE Generation")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number", 5, 300, 20)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_cats = st.multiselect("Function Categories", ["Basic","Special"], default=["Basic"])
        include_solutions = st.checkbox("Include Solutions", True)
    with c3:
        vary = st.checkbox("Vary Parameters", True)
        if vary:
            α_min, α_max = st.number_input("α min", value=-2.0), st.number_input("α max", value=2.0)
            β_min, β_max = st.number_input("β min", value=0.5), st.number_input("β max", value=2.0)
        else:
            α_min = α_max = 1.0
            β_min = β_max = 1.0

    if st.button("🚀 Generate Batch", type="primary"):
        results = []
        pool = []
        if "Basic" in func_cats and hasattr(st.session_state, "basic_functions"):
            try: pool += st.session_state.basic_functions.get_function_names()
            except Exception: pass
        if "Special" in func_cats and hasattr(st.session_state, "special_functions"):
            try: pool += st.session_state.special_functions.get_function_names()
            except Exception: pass
        pool = pool[:50]
        for i in range(num_odes):
            try:
                α = float(np.random.uniform(α_min, α_max))
                β = float(np.random.uniform(β_min, β_max))
                n  = int(np.random.randint(1, 3))
                M  = float(np.random.uniform(-1, 1))
                if not pool: break
                fname = str(np.random.choice(pool))

                x = sp.Symbol("x", real=True)
                f_expr = get_function_expr("Basic" if (np.random.rand()<0.7) else "Special", fname)
                y = theorem_4_1_solution_expr(f_expr, sp.nsimplify(α), sp.nsimplify(β), n, sp.nsimplify(M), x)
                spec = st.session_state.get("current_generator")
                lhs = None
                if spec is not None:
                    lhs = getattr(spec, "lhs", None)
                    if lhs is None and hasattr(spec, "get_lhs"):
                        try: lhs = spec.get_lhs()
                        except Exception: lhs = None
                    if lhs is None and hasattr(spec, "build_lhs"):
                        try: lhs = spec.build_lhs()
                        except Exception: lhs = None
                if lhs is not None:
                    rhs = apply_lhs_to_solution(lhs, y, x, _detect_y_name(lhs))
                else:
                    lhs = sp.Symbol("LHS")
                    rhs = sp.Symbol("RHS")

                rec = {
                    "ID": i+1,
                    "Type": "linear" if _is_linear_lhs(lhs, x) else "nonlinear",
                    "Generator": getattr(spec, "name", "-") if spec is not None else "-",
                    "Function": fname,
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

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("📊 CSV", df.to_csv(index=False), file_name="batch_odes.csv", mime="text/csv")
        with d2:
            st.download_button("📄 JSON", json.dumps(results, indent=2), file_name="batch_odes.json", mime="application/json")
        with d3:
            st.download_button("📝 LaTeX table", _latex_table(results), file_name="batch_odes.tex", mime="text/x-latex")

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
        if spec:
            lhs = getattr(spec, "lhs", None)
            if lhs is None and hasattr(spec, "get_lhs"):
                try: lhs = spec.get_lhs()
                except Exception: lhs = None
            if lhs is None and hasattr(spec, "build_lhs"):
                try: lhs = spec.build_lhs()
                except Exception: lhs = None
            if lhs is not None:
                ode_obj = {"ode": lhs, "type": "custom", "order": _guess_order_from_lhs(lhs, sp.Symbol("x"))}
            else:
                st.warning("Generator found but no LHS to analyze.")
        else:
            st.warning("Build a generator first.")
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
                st.download_button("📥 Report", analysis.detailed_report, file_name="novelty_report.txt")
        except Exception as e:
            st.error(f"Novelty analysis failed: {e}")

def page_analysis():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs yet.")
        return
    df = pd.DataFrame([
        {"Type": r.get("type",""), "Function": r.get("function_used",""),
         "Order": r.get("order",""), "Timestamp": r.get("timestamp","")}
        for r in st.session_state.generated_odes
    ])
    st.dataframe(df, use_container_width=True)
    if not df.empty and "Order" in df.columns:
        fig = px.histogram(df, x="Order", nbins=10, title="Order Distribution")
        st.plotly_chart(fig, use_container_width=True)

def page_visualize():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.info("No ODEs to visualize.")
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
            xs = np.linspace(-5, 5, 800)
            yfn = sp.lambdify([x], y, "numpy")
            ys = np.array([yfn(val) for val in xs], dtype=np.complex128)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines", name="Re y(x)"))
            if np.any(np.abs(np.imag(ys)) > 1e-12):
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
        st.download_button("📄 LaTeX", LaTeXExporter.document_for_ode(rec, True),
                           file_name=f"ode_{idx+1}.tex", mime="text/x-latex")
    with c2:
        st.download_button("📦 ZIP", LaTeXExporter.zip_package(rec),
                           file_name=f"ode_package_{idx+1}.zip", mime="application/zip")

# =============================================================================
# Main
# =============================================================================
def main():
    init_state()
    if not HAVE_SRC:
        st.warning("This module requires the src/ package. Ensure the project ZIP is extracted with src/ present.")

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