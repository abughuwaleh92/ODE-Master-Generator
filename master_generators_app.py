# -*- coding: utf-8 -*-
"""
Master Generators for ODEs — Power Edition
- Theorem 4.1 (exact/symbolic or numeric params)
- Theorem 4.2 via Stirling numbers (Faà di Bruno)
- EXTENDED Free‑form LHS builder:
  * Many wrappers + composite chains with parameters
  * Inner derivative (on y(g(x))) and outer derivative of whole term
  * Delay/advance and general argument transforms g(x)
- Per‑ODE export buttons right after generation
- ML/DL & everything else kept intact
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

try:
    import torch
except Exception:
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

APP_TITLE = "Master Generators ODE System — Power Edition"
APP_ICON = "🔬"

# Ensure src/ is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# =============================================================================
# Layered imports from src/ with safe fallbacks/aliases
# =============================================================================
HAVE_SRC = True
try:
    # master generator (optional)
    try:
        from src.generators.master_generator import (
            MasterGenerator,
            EnhancedMasterGenerator,
            CompleteMasterGenerator,
        )
    except Exception as e:
        MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
        logger.warning(f"master_generator import warning: {e}")

    # linear factories (some repos place them in master_generator)
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
        try:
            from src.generators.master_generator import (
                LinearGeneratorFactory as _LGF2,
                CompleteLinearGeneratorFactory as _CLGF2,
            )
            LinearGeneratorFactory = _LGF2
            CompleteLinearGeneratorFactory = _CLGF2
        except Exception as e:
            logger.warning(f"linear factories import warning: {e}")

    # nonlinear factories
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

    # constructor/types
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

    # master theorem solver (optional; we also provide symbolic fallback)
    try:
        from src.generators.master_theorem import (
            MasterTheoremSolver,
            MasterTheoremParameters,
            ExtendedMasterTheorem,
        )
    except Exception as e:
        MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None
        logger.warning(f"master_theorem import warning: {e}")

    # classifier
    try:
        from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
    except Exception as e:
        ODEClassifier = PhysicalApplication = None
        logger.warning(f"ode_classifier import warning: {e}")

    # function libraries
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

    # utils
    try:
        from src.utils.config import Settings, AppConfig
    except Exception:
        Settings = AppConfig = None
    try:
        from src.utils.cache import CacheManager, cached
    except Exception:
        CacheManager = None

        def cached(*args, **kwargs):
            def _w(f): return f
            return _w
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
    logger.error("Failed to import from src/: %s", e)

# =============================================================================
# Streamlit page config & CSS
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded"
)
st.markdown(
    """
<style>
.main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:2.0rem;border-radius:14px;margin-bottom:1rem;color:#fff;text-align:center;box-shadow:0 10px 30px rgba(0,0,0,.2)}
.main-title{font-size:2.1rem;font-weight:700;margin-bottom:.2rem}
.subtitle{font-size:1rem;opacity:.95}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:1rem;border-radius:12px;text-align:center;box-shadow:0 10px 20px rgba(0,0,0,.2)}
.generator-term{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);padding:.7rem;border-radius:9px;margin:.4rem 0;border-left:5px solid #667eea;box-shadow:0 3px 10px rgba(0,0,0,.08)}
.result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);border:2px solid #4caf50;padding:1rem;border-radius:12px;margin:1rem 0;box-shadow:0 5px 15px rgba(76,175,80,.2)}
.info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);border-left:5px solid #2196f3;padding:.8rem;border-radius:9px;margin:.7rem 0}
.latex-export-box{background:linear-gradient(135deg,#f3e5f5 0%,#e1bee7 100%);border:2px solid #9c27b0;padding:.8rem;border-radius:10px;margin:.9rem 0;box-shadow:0 5px 15px rgba(156,39,176,.18)}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# LaTeX exporter
# =============================================================================
class LaTeXExporter:
    @staticmethod
    def _sx(expr) -> str:
        if expr is None:
            return ""
        try:
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
        func_name = ode_data.get("function_used", "unknown")
        gen_num = ode_data.get("generator_number", 1)
        return f'''# Reproduction snippet
import sympy as sp
try:
    from src.generators.linear_generators import LinearGeneratorFactory
except Exception:
    LinearGeneratorFactory=None
try:
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
except Exception:
    NonlinearGeneratorFactory=None
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions

z=sp.Symbol("z")
basic=BasicFunctions() if "{func_name}" else None
special=SpecialFunctions() if "{func_name}" else None
def get_f(name):
    if basic:
        try: return basic.get_function(name)
        except: pass
    if special:
        try: return special.get_function(name)
        except: pass
    return sp.exp(z) if name=="exponential" else z

f_z=get_f("{func_name}")
params={json.dumps(params, default=str)}
factory=LinearGeneratorFactory() if "{gen_type}"=="linear" and LinearGeneratorFactory else NonlinearGeneratorFactory()
if factory:
    res=factory.create({gen_num}, f_z, **params)
    print("ODE:",res.get("ode")); print("Solution:",res.get("solution"))
'''

# =============================================================================
# Session utilities
# =============================================================================
def _ensure_ss_key(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

def set_lhs_source(src: str):
    assert src in {"constructor", "freeform"}
    st.session_state["lhs_source"] = src

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
            # conservative upper bound
            return max(int(t.get("dy_order", t.get("inner_order", 0))) + int(t.get("outer_diff", 0)) for t in desc["terms"])
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
# Theorem 4.1 & 4.2 (symbolic)
# =============================================================================
def theorem_4_1_solution_expr(f_of_z, alpha, beta, n: int, M, x: sp.Symbol):
    z = sp.Symbol("z")
    ωs = [((2*s - 1) * sp.pi) / (2*n) for s in range(1, n+1)]
    def ψ(ω): return f_of_z.subs(z, alpha + beta * sp.exp(sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
    def φ(ω): return f_of_z.subs(z, alpha + beta * sp.exp(-sp.I * x * sp.cos(ω) - x * sp.sin(ω)))
    const = f_of_z.subs(z, alpha + beta)
    S = sum(2*const - ψ(ω) - φ(ω) for ω in ωs)
    return (sp.pi/(2*n)) * sp.simplify(S) + sp.pi * M

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

# =============================================================================
# Function resolver
# =============================================================================
def get_function_expr(source: str, name: str) -> sp.Expr:
    z = sp.Symbol("z")
    try:
        if source == "Basic" and st.session_state.get("basic_functions"):
            obj = st.session_state.basic_functions.get_function(name)
            try:
                return sp.sympify(obj) if isinstance(obj, sp.Expr) else obj(z)
            except Exception:
                pass
        if source == "Special" and st.session_state.get("special_functions"):
            obj = st.session_state.special_functions.get_function(name)
            try:
                return sp.sympify(obj) if isinstance(obj, sp.Expr) else obj(z)
            except Exception:
                pass
    except Exception:
        pass
    palette = {
        "linear": z, "quadratic": z**2, "cubic": z**3, "exponential": sp.exp(z),
        "log": sp.log(z), "sin": sp.sin(z), "cos": sp.cos(z), "tanh": sp.tanh(z),
    }
    return palette.get(name, sp.exp(z))

# =============================================================================
# EXTENDED Free‑form LHS builder
# =============================================================================
EPS = sp.Symbol("epsilon", positive=True)

# Simple wrappers (no parameters)
SIMPLE_WRAPPERS = {
    "id": lambda e: e,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log,
    "logabs": lambda e: sp.log(EPS + sp.Abs(e)),
    "abs": sp.Abs, "sign": sp.sign, "sqrt": sp.sqrt,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sech": sp.sech, "csch": sp.csch, "coth": sp.coth,
    "erf": sp.erf, "erfc": sp.erfc,
    "gamma": sp.gamma, "loggamma": sp.loggamma,
    "airy_ai": sp.airyai, "airy_bi": sp.airybi,
    # Bessel shortcut names
    "besselj0": lambda e: sp.besselj(0, e),
    "besselj1": lambda e: sp.besselj(1, e),
    "bessely0": lambda e: sp.bessely(0, e),
    "bessely1": lambda e: sp.bessely(1, e),
    # Sigmoid / Softplus
    "sigmoid": lambda e: 1/(1+sp.exp(-e)),
    "softplus": lambda e: sp.log(1+sp.exp(e)),
}

def _param_wrapper(name: str, param_vals: List[sp.Expr]):
    """Parametric wrappers: pow(p), scale(c), shift(b), affine(a,b)."""
    name = name.lower()
    if name == "pow":
        p = param_vals[0]
        return lambda e: e ** p
    if name == "scale":
        c = param_vals[0]
        return lambda e: c * e
    if name == "shift":
        b = param_vals[0]
        return lambda e: e + b
    if name == "affine":
        a = param_vals[0]
        b = param_vals[1] if len(param_vals) > 1 else sp.Integer(0)
        return lambda e: a*e + b
    # Fallback: identity
    return lambda e: e

def _parse_chain(chain_str: str) -> List:
    """
    Parse "exp -> sinh -> pow(3/2) -> shift(-1/3)"
    into a list of callables to be composed in order.
    """
    chain = []
    if not chain_str or str(chain_str).strip() in {"", "id", "ID"}:
        return [SIMPLE_WRAPPERS["id"]]
    tokens = [t.strip() for t in chain_str.split("->") if t.strip()]
    for tok in tokens:
        if "(" in tok and tok.endswith(")"):
            name = tok.split("(", 1)[0].strip().lower()
            raw = tok[len(name)+1:-1]
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            try:
                params = [sp.nsimplify(p, rational=True) for p in parts]
            except Exception:
                params = [sp.sympify(p) for p in parts]
            chain.append(_param_wrapper(name, params))
        else:
            name = tok.lower()
            if name in SIMPLE_WRAPPERS:
                chain.append(SIMPLE_WRAPPERS[name])
            else:
                # attempt SymPy attr
                fn = getattr(sp, name, None)
                chain.append(fn if callable(fn) else SIMPLE_WRAPPERS["id"])
    if not chain:
        chain = [SIMPLE_WRAPPERS["id"]]
    return chain

def _compose(expr: sp.Expr, fns: List) -> sp.Expr:
    out = expr
    for f in fns:
        try:
            out = f(out)
        except Exception:
            pass
    return out

def _safe_sympify(expr_str: str, x: sp.Symbol) -> sp.Expr:
    # Restrict to SymPy safe namespace
    locals_d = {k: getattr(sp, k) for k in [
        "sin","cos","tan","sinh","cosh","tanh","exp","log","sqrt","Abs","sign","Heaviside","pi"
    ]}
    locals_d["x"] = x
    locals_d["epsilon"] = EPS
    try:
        return sp.sympify(expr_str, locals=locals_d, convert_xor=True)
    except Exception:
        return x

def _build_arg_expr(t: Dict[str, Any], x: sp.Symbol) -> sp.Expr:
    mode = str(t.get("arg_mode", "custom")).lower()
    if "arg_expr" in t and str(t["arg_expr"]).strip():
        return _safe_sympify(str(t["arg_expr"]), x)

    a = sp.nsimplify(t.get("a", 1), rational=True) if "a" in t else sp.Integer(1)
    b = sp.nsimplify(t.get("b", 0), rational=True) if "b" in t else sp.Integer(0)
    tau = sp.nsimplify(t.get("tau", 0), rational=True) if "tau" in t else sp.Integer(0)

    if mode in {"x"}:
        return x
    if mode in {"affine", "ax+b"}:
        return a*x + b
    if mode in {"pantograph_div", "x/a+b"}:
        return x/a + b
    if mode in {"scale"}:
        return a*x
    if mode in {"shift"}:
        return x + b
    if mode in {"delay"}:
        return x - tau
    if mode in {"advance"}:
        return x + tau
    # default custom
    return x

def build_freeform_lhs_expr(terms: List[Dict[str, Any]], x: sp.Symbol, yname: str = "y") -> sp.Expr:
    """
    Term dictionary (advanced):
      coef: str/number
      dy_order: int (derivatives on y(g(x)) w.r.t x)          [= inner derivative]
      wrap_chain: "exp -> sinh -> pow(2)"                     [applied before power]
      power: int                                              [default 1]
      outer_chain: "logabs -> sqrt"                           [applied after power]
      outer_diff: int                                         [differentiate whole term]
      arg_mode: one of {"x","affine","x/a+b","pantograph_div","scale","shift","delay","advance","custom"}
      a,b,tau: numbers for arg_mode presets
      arg_expr: custom g(x) string (overrides arg_mode presets)

    Backward compat:
      inner_order: used if dy_order is missing
      wrap: single wrapper name if wrap_chain missing
    """
    y = Function(yname)
    total = 0
    for t in terms:
        coef = t.get("coef", 1)
        try:
            coef = sp.nsimplify(coef, rational=True)
        except Exception:
            coef = sp.sympify(coef)

        dy_order = int(t.get("dy_order", t.get("inner_order", 0)))
        power = int(t.get("power", 1))
        outer_diff = int(t.get("outer_diff", 0))

        wrap_chain = str(t.get("wrap_chain", t.get("wrap", "id")))
        outer_chain = str(t.get("outer_chain", "id"))

        # argument g(x)
        arg = _build_arg_expr(t, x)

        # base: y(g(x)) with inner derivative w.r.t x
        base = y(arg)
        if dy_order > 0:
            base = Derivative(base, (x, dy_order))

        # inner composition then power
        inner_wrapped = _compose(base, _parse_chain(wrap_chain))
        raised = inner_wrapped**power

        # outer composition
        out_wrapped = _compose(raised, _parse_chain(outer_chain))

        # outer derivative (chain rule handled by sympy)
        term_expr = sp.diff(out_wrapped, (x, outer_diff)) if outer_diff > 0 else out_wrapped

        total += coef * term_expr

    return sp.simplify(total)

# =============================================================================
# Apply symbolic LHS to y(x)
# =============================================================================
def _derivative_order_of(d: Derivative, x: sp.Symbol) -> int:
    k = 0
    for v in d.variables:
        if isinstance(v, sp.Symbol):
            if v == x:
                k += 1
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            if v[0] == x:
                k += int(v[1])
    return k

def apply_lhs_to_solution(lhs_expr: sp.Expr, y_expr: sp.Expr, x: sp.Symbol, y_name="y") -> sp.Expr:
    """
    Substitute y(g(x)) and Derivative(y(g(x)), (x,k)) with
    y_expr(x) composed at g(x), then differentiate w.r.t x (chain rule).
    """
    y = Function(y_name)
    mapping = {}

    # map plain y(g(x))
    for fnode in lhs_expr.atoms(sp.Function):
        try:
            if fnode.func == y:
                arg = fnode.args[0] if fnode.args else x
                mapping[fnode] = y_expr.subs(x, arg)
        except Exception:
            pass

    # map Derivative(y(g(x)), (x,k))
    for d in lhs_expr.atoms(Derivative):
        try:
            if hasattr(d, "expr") and d.expr.func == y:
                inner_arg = d.expr.args[0] if d.expr.args else x
                k = _derivative_order_of(d, x)
                composed = y_expr.subs(x, inner_arg)
                mapping[d] = sp.diff(composed, (x, k))
        except Exception:
            pass

    mapping.setdefault(y(x), y_expr)

    try:
        return sp.simplify(lhs_expr.xreplace(mapping))
    except Exception:
        return lhs_expr.subs(mapping)

# =============================================================================
# UI Pages
# =============================================================================
def page_header():
    st.markdown(
        f"""
    <div class="main-header">
      <div class="main-title">{APP_ICON} {APP_TITLE}</div>
      <div class="subtitle">Thm 4.1 · Thm 4.2 · Free‑form Composites/Delays · ML/DL · Export</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def page_dashboard():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h4>Generated ODEs</h4><h2>{len(st.session_state.generated_odes)}</h2></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h4>ML Patterns</h4><h2>{len(st.session_state.generator_patterns)}</h2></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h4>Batch Results</h4><h2>{len(st.session_state.batch_results)}</h2></div>',
            unsafe_allow_html=True,
        )
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
        st.info("No ODEs yet. Try **Generator Constructor** then **Apply Master Theorem**.")

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    st.markdown(
        """<div class="info-box">
        Build generators either with your project's Constructor (terms) or with the
        <b>Extended Free‑form</b> builder (composites, delays, chain rule, etc.).
        </div>""",
        unsafe_allow_html=True,
    )

    # 1) src constructor (if available)
    with st.expander("➕ Term‑based Constructor (from src)", expanded=False):
        if not st.session_state.generator_constructor or not DerivativeTerm:
            st.warning("The src-based constructor is unavailable. Use the Free‑form builder.")
        else:
            cols = st.columns(4)
            with cols[0]:
                deriv_order = st.number_input("Derivative order k", 0, 10, 1)
            with cols[1]:
                coef = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
            with cols[2]:
                power = st.number_input("Power", 1, 10, 1)
            with cols[3]:
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
                scaling = st.number_input("Scaling a (delay/advance)", 0.1, 10.0, 1.0, 0.1)
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
                    ctor = st.session_state.generator_constructor
                    if hasattr(ctor, "add_term"):
                        ctor.add_term(term)
                    elif hasattr(ctor, "terms"):
                        ctor.terms.append(term)
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
                    set_lhs_source("constructor")
                    st.success("✅ Specification built & selected (Constructor).")
                    try:
                        st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                    except Exception:
                        st.write(gen_spec.lhs)
                except Exception as e:
                    st.error(f"Failed to build specification: {e}")

    # 2) Extended Free‑form
    with st.expander("🧪 Extended Free‑form LHS Builder (composites · chain rule · delay)", expanded=True):
        st.markdown(
            """
**A term** = `coef · outer_chain( (inner_chain(D^{dy_order} y(g(x))) )^power )` then **outer_diff** derivatives.  
**Examples** of chains (*in order, use `->`*):
- `exp -> sinh -> logabs`
- `pow(3/2) -> scale(2) -> shift(-1/3)`
- `logabs` or `besselj0 -> pow(2)`

**g(x)** presets: `x`, `affine (a*x+b)`, `x/a+b`, `scale (a*x)`, `shift (x+b)`, `delay (x-τ)`, `advance (x+τ)`, or **custom**.
            """
        )

        # Quick add (simple)
        cquick = st.columns(5)
        with cquick[0]:
            ff_coef = st.text_input("Coefficient", "1")
        with cquick[1]:
            ff_dy = st.number_input("dy_order (inner)", 0, 12, 1)
        with cquick[2]:
            ff_wrap_chain = st.text_input("Inner chain (e.g. exp->sinh)", "id")
        with cquick[3]:
            ff_power = st.number_input("Power", 1, 12, 1)
        with cquick[4]:
            ff_outer_chain = st.text_input("Outer chain (e.g. logabs)", "id")

        cmore = st.columns(5)
        with cmore[0]:
            ff_outer_diff = st.number_input("outer_diff", 0, 12, 0)
        with cmore[1]:
            ff_arg_mode = st.selectbox(
                "g(x) preset",
                ["custom", "x", "affine", "x/a+b", "scale", "shift", "delay", "advance"],
                index=0,
            )
        with cmore[2]:
            ff_a = st.text_input("a (for affine/scale/x/a+b)", "1")
        with cmore[3]:
            ff_b = st.text_input("b (affine/shift/x/a+b)", "0")
        with cmore[4]:
            ff_tau = st.text_input("τ (delay/advance)", "0")

        g_custom = st.text_input("Custom g(x) (overrides preset)", "")

        if st.button("➕ Add free‑form term", use_container_width=True):
            try:
                term = {
                    "coef": sp.nsimplify(ff_coef, rational=True),
                    "dy_order": int(ff_dy),
                    "wrap_chain": ff_wrap_chain.strip(),
                    "power": int(ff_power),
                    "outer_chain": ff_outer_chain.strip(),
                    "outer_diff": int(ff_outer_diff),
                    "arg_mode": ff_arg_mode,
                    "a": sp.nsimplify(ff_a, rational=True),
                    "b": sp.nsimplify(ff_b, rational=True),
                    "tau": sp.nsimplify(ff_tau, rational=True),
                }
                if g_custom.strip():
                    term["arg_expr"] = g_custom.strip()
                st.session_state.freeform_terms.append(term)
                st.success(f"Added term: {term}")
            except Exception as e:
                st.error(f"Failed to add free‑form term: {e}")

        # Show current terms
        if st.session_state.freeform_terms:
            st.write("**Current free‑form terms**")
            for i, t in enumerate(st.session_state.freeform_terms, 1):
                desc = (
                    f"coef={t.get('coef')} · inner=({t.get('wrap_chain','id')}) "
                    f"dy_order={t.get('dy_order',0)} pow={t.get('power',1)} "
                    f"outer=({t.get('outer_chain','id')}) outer_diff={t.get('outer_diff',0)} "
                    f"g(x)={t.get('arg_expr', t.get('arg_mode','x'))}"
                )
                st.markdown(f"<div class='generator-term'><b>Term {i}:</b> {desc}</div>", unsafe_allow_html=True)
            cbtn = st.columns(2)
            with cbtn[0]:
                if st.button("🗑️ Clear terms"):
                    st.session_state.freeform_terms = []
                    st.session_state.freeform_gen_spec = None
                    st.info("Cleared.")
            with cbtn[1]:
                if st.button("🔨 Build Free‑form LHS and select"):
                    try:
                        x = sp.Symbol("x", real=True)
                        lhs_expr = build_freeform_lhs_expr(st.session_state.freeform_terms, x, yname="y")
                        # wrap as spec
                        if GeneratorSpecification:
                            ff_spec = GeneratorSpecification(terms=[], name=f"Free-form {datetime.now().strftime('%H%M%S')}")
                            ff_spec.lhs = lhs_expr
                            ff_spec.freeform_descriptor = {"terms": st.session_state.freeform_terms, "note": "free-form"}
                        else:
                            class _Spec: ...
                            ff_spec = _Spec()
                            ff_spec.lhs = lhs_expr
                            ff_spec.freeform_descriptor = {"terms": st.session_state.freeform_terms, "note": "free-form"}
                        st.session_state["freeform_gen_spec"] = ff_spec
                        set_lhs_source("freeform")
                        st.success("✅ Free‑form LHS stored & selected.")
                        try:
                            st.latex(sp.latex(lhs_expr))
                        except Exception:
                            st.write(lhs_expr)
                    except Exception as e:
                        st.error(f"Failed to build free‑form LHS: {e}")

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem")
    st.markdown(
        "<div class='info-box'>Choose LHS source (Constructor or Free‑form), pick f(z), parameters, and generate an ODE. Thm 4.2 is available for general m‑th derivatives.</div>",
        unsafe_allow_html=True,
    )

    # Select LHS source
    sel = st.radio(
        "LHS source",
        options=("constructor", "freeform"),
        index=0 if st.session_state.get("lhs_source", "constructor") == "constructor" else 1,
        format_func=lambda s: {"constructor": "Constructor LHS", "freeform": "Free‑form LHS"}[s],
        horizontal=True,
    )
    set_lhs_source(sel)

    # Function & params
    col1, col2 = st.columns([1, 1])
    with col1:
        source_lib = st.selectbox("Function library", ["Basic", "Special"], index=0)
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
        beta  = st.text_input("β", "1")
        n     = st.number_input("n (integer ≥1)", 1, 20, 1)
        M     = st.text_input("M", "0")
        use_exact = st.checkbox("Exact (symbolic) parameters", value=True)

    def to_exact(v):
        try: return sp.nsimplify(v, rational=True)
        except Exception: return sp.sympify(v)

    α = to_exact(alpha) if use_exact else sp.Float(alpha)
    β = to_exact(beta)  if use_exact else sp.Float(beta)
    𝑀 = to_exact(M)     if use_exact else sp.Float(M)
    x = sp.Symbol("x", real=True)

    # Thm 4.2
    with st.expander("Optional: Theorem 4.2 (m‑th derivative)", expanded=False):
        m = st.number_input("m ≥ 1", 1, 20, 1)
        if st.button("Compute y^{(m)}(x) (Thm 4.2)"):
            try:
                f_expr = get_function_expr(source_lib, func_name)
                y_m = theorem_4_2_mth_derivative_expr(f_expr, α, β, int(n), int(m), x)
                if simplify_out:
                    y_m = sp.simplify(y_m)
                st.latex("y^{(" + str(int(m)) + ")}(x) = " + sp.latex(y_m))
            except Exception as e:
                st.error(f"Failed to compute y^{m}(x): {e}")

    # Generate ODE
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        with st.spinner("Applying Theorem 4.1 and constructing RHS…"):
            try:
                f_expr = get_function_expr(source_lib, func_name)
                yx = theorem_4_1_solution_expr(f_expr, α, β, int(n), 𝑀, x)
                if simplify_out:
                    yx = sp.simplify(yx)

                spec = get_active_generator_spec()
                if spec and hasattr(spec, "lhs") and spec.lhs is not None:
                    try:
                        rhs = apply_lhs_to_solution(spec.lhs, yx, x, y_name="y")
                        if simplify_out:
                            rhs = sp.simplify(rhs)
                        generator_lhs = spec.lhs
                    except Exception as e:
                        logger.warning(f"LHS application failed: {e}")
                        z = sp.Symbol("z")
                        rhs = sp.simplify(sp.pi * (f_expr.subs(z, α + β) + 𝑀))
                        generator_lhs = sp.Symbol("L[y]")
                else:
                    z = sp.Symbol("z")
                    rhs = sp.simplify(sp.pi * (f_expr.subs(z, α + β) + 𝑀))
                    generator_lhs = sp.Symbol("L[y]")

                ode_type = _infer_type_from_spec(spec) if spec else "nonlinear"
                ode_order = _infer_order_from_spec(spec) if spec else 0

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

                st.markdown("<div class='result-box'><b>✅ ODE Generated!</b></div>", unsafe_allow_html=True)
                t1, t2, t3 = st.tabs(["📐 Equation", "💡 Solution", "📤 Export"])
                with t1:
                    st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                    st.caption(f"LHS source: **{st.session_state.get('lhs_source','constructor')}**")
                with t2:
                    st.latex("y(x) = " + sp.latex(yx))
                    st.write(f"**Parameters:** α={α}, β={β}, n={int(n)}, M={𝑀}")
                    st.write(f"**f(z):** {f_expr}")
                with t3:
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
                        "📄 Download LaTeX", latex_doc, file_name=f"ode_{len(st.session_state.generated_odes)}.tex", mime="text/x-latex"
                    )
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
                        "📦 Download ZIP", pkg, file_name=f"ode_package_{len(st.session_state.generated_odes)}.zip", mime="application/zip"
                    )
            except Exception as e:
                logger.error("Generation error", exc_info=True)
                st.error(f"Error generating ODE: {e}")

def page_ml_pattern_learning():
    st.header("🤖 ML Pattern Learning")
    if MLTrainer is None:
        st.info("ML trainer not available here.")
        return
    col = st.columns(4)
    with col[0]: st.metric("Patterns", len(st.session_state.generator_patterns))
    with col[1]: st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with col[2]: st.metric("Training Epochs", len(st.session_state.training_history.get("train_loss", [])))
    with col[3]: st.metric("Model Status", "Trained" if st.session_state.ml_trained else "Not Trained")

    model_type = st.selectbox(
        "Select ML Model",
        ["pattern_learner", "vae", "transformer"],
        index=0,
        format_func=lambda x: {"pattern_learner": "Pattern Learner", "vae": "VAE", "transformer": "Transformer"}[x],
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
                progress = st.progress(0.0); info = st.empty()
                def cb(ep, total): progress.progress(min(1.0, ep/float(total))); info.text(f"Epoch {ep}/{total}")
                trainer.train(epochs=epochs, batch_size=batch_size, samples=samples, validation_split=val_split, progress_callback=cb)
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
                for _ in range(num):
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
            a_rng = (1.0, 1.0); b_rng = (1.0, 1.0); n_rng = (1, 1)

    with st.expander("Advanced", expanded=False):
        export_format = st.selectbox("Export format", ["JSON", "CSV", "LaTeX", "All"], index=1)
        include_solutions = st.checkbox("Include solution preview", True)

    if st.button("🚀 Generate Batch", type="primary", use_container_width=True):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            results = []
            # function names
            all_names = []
            if "Basic" in func_cats and st.session_state.basic_functions:
                all_names += st.session_state.basic_functions.get_function_names()
            if "Special" in func_cats and st.session_state.special_functions:
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
                        # fallback via Thm 4.1
                        x = sp.Symbol("x", real=True)
                        yx = theorem_4_1_solution_expr(f_expr, sp.nsimplify(params["alpha"]), sp.nsimplify(params["beta"]), int(params["n"]), sp.nsimplify(params["M"]), x)
                        lhs = sp.Function("L")(x)
                        rhs = sp.pi * (f_expr.subs(sp.Symbol("z"), params["alpha"] + params["beta"])) + sp.pi * params["M"]
                        result = {"ode": sp.Eq(lhs, rhs), "solution": yx, "type": gtype, "order": 0, "generator_number": i+1, "function_used": fname, "subtype": "fallback"}

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

def page_examples():
    st.header("📚 Examples")
    x = sp.Symbol("x")
    y = sp.Function("y")
    ex = [
        {"name": "Harmonic oscillator", "lhs": y(x) + Derivative(y(x), (x, 2))},
        {"name": "Damped oscillator", "lhs": Derivative(y(x), (x, 2)) + 2*Derivative(y(x), x) + y(x)},
        {"name": "Pantograph-like", "lhs": Derivative(y(x), (x, 2)) + y(x/2) - y(x)},
    ]
    for i, e in enumerate(ex, 1):
        with st.expander(f"Example {i}: {e['name']}"):
            st.latex(sp.latex(e["lhs"]) + " = RHS")
            if st.button(f"Use as Free‑form LHS: {e['name']}", key=f"use_ex_{i}"):
                class _Spec: ...
                spec = _Spec()
                spec.lhs = e["lhs"]
                spec.freeform_descriptor = {"terms": [], "note": "example"}
                st.session_state["freeform_gen_spec"] = spec
                set_lhs_source("freeform")
                st.success("Stored into Free‑form and selected.")

def page_settings():
    st.header("⚙️ Settings")
    if st.checkbox("Show session keys"):
        st.write(list(st.session_state.keys()))
    if st.button("Clear cache"):
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
        r"""
**Theorem 4.1.**  
\[
y(x)=\frac{\pi}{2n}\sum_{s=1}^n\Big(2f(\alpha+\beta)-\psi_s(x)-\phi_s(x)\Big)+\pi M
\]
with \(\psi_s=f(\alpha+\beta e^{ix\cos\omega_s-x\sin\omega_s})\), \(\phi_s=f(\alpha+\beta e^{-ix\cos\omega_s-x\sin\omega_s})\).

**Theorem 4.2 (compact Stirling‑number form).**  
\[
y^{(m)}(x)=-\frac{\pi}{2n}\sum_{s=1}^n\left\{
\lambda_s^m\sum_{j=1}^m\mathbf{S}(m,j)(\beta\zeta_s)^j\partial_\alpha^{\,j}\psi
+\overline{\lambda_s}^{\,m}\sum_{j=1}^m\mathbf{S}(m,j)(\beta\overline{\zeta_s})^j\partial_\alpha^{\,j}\phi
\right\}.
\]
Here \(\zeta_s = e^{-x\sin\omega_s}e^{ix\cos\omega_s}\), \(\lambda_s=e^{i(\pi/2+\omega_s)}\), \(\omega_s=(2s-1)\pi/(2n)\).
""",
        unsafe_allow_html=False,
    )

# =============================================================================
# Main
# =============================================================================
def main():
    initialize_session()
    # Header
    st.markdown(
        f"""
    <div class="main-header">
      <div class="main-title">{APP_ICON} {APP_TITLE}</div>
      <div class="subtitle">Theorems 4.1 & 4.2 · Extended Free‑form LHS · ML/DL · Exports</div>
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
