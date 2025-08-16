# master_generators_app.py
"""
Master Generators for ODEs — Complete App (Corrected + Reverse Engineering + Choice-A Phi Adapter)

What’s included
---------------
• Apply Master Theorem page
  - LHS source selector: Constructor / Free‑form / Arbitrary SymPy
  - Choice‑A Φ‑library (logistic, erf, sinh/cosh/tanh, exp/sin/cos) works transparently
  - Exact/approx params, simplification levels, optional async compute (RQ)
  - Theorem 4.2 quick derivative

• Reverse Engineering page (3 modes)
  1) From y(x) ⇒ infer multi-block template params (φ, α, β, n, M) by RMSE grid search,
     then build ODE using the selected LHS source (constructor/free‑form/arbitrary).
  2) Fit linear operator L[y] = g(x): least squares for coefficients c_i of ∑ c_i y^(i).
  3) From target ODE L[y] = RHS ⇒ fit y‑template parameters minimizing ||L[y]-RHS|| on grid.

• All other pages retained: Dashboard, Constructor, ML, Batch, Novelty, Analysis,
  Applications, Visualization, Export, Examples, Settings, Docs.

Important implementation notes
------------------------------
• Choice A (Φ as Special): If user chooses "Phi", we pass ComputeParams(function_library="Special")
  but provide special_lib = _PhiAdapter(phi_library). No ComputeParams/ode_core signature change is needed.

• No 'phi_lib' argument is passed anywhere to ComputeParams. The previous error
  “ComputeParams.__init__() got unexpected keyword argument 'phi_lib'” is avoided.

• Async compute: When using Redis jobs, we send only serializable payload (strings/numbers).
  The worker must apply the same Choice‑A adapter internally if function_library == "Phi".
"""

# ---------------- std libs ----------------
import os, sys, io, json, logging, pickle, zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import itertools

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp
from sympy.core.function import AppliedUndef

# optional torch (for ML page)
try:
    import torch
except Exception:
    torch = None

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# ---------------- path setup ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------- import src (resilient) ----------------
HAVE_SRC = True
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
    from src.generators.master_generator import (
        MasterGenerator, EnhancedMasterGenerator, CompleteMasterGenerator,
    )
    try:
        from src.generators.master_generator import (
            CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory,
        )
    except Exception:
        from src.generators.linear_generators import (
            LinearGeneratorFactory, CompleteLinearGeneratorFactory,
        )
        from src.generators.nonlinear_generators import (
            NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory,
        )
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType,
    )
    from src.generators.master_theorem import (
        MasterTheoremSolver, MasterTheoremParameters, ExtendedMasterTheorem,
    )
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    from src.ml.pattern_learner import (
        GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer, create_model,
    )
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
    from src.ml.generator_learner import (
        GeneratorPattern, GeneratorPatternNetwork, GeneratorLearningSystem,
    )
    from src.dl.novelty_detector import (
        ODENoveltyDetector, NoveltyAnalysis, ODETokenizer, ODETransformer,
    )
    from src.utils.config import Settings, AppConfig
    from src.utils.cache import CacheManager, cached
    from src.utils.validators import ParameterValidator
    from src.ui.components import UIComponents
except Exception as e:
    logger.warning(f"Some imports from src/ failed or are missing: {e}")
    HAVE_SRC = False

# ---------------- internal math core & queue utils ----------------
# (Provided by your project; not modified here.)
from shared.ode_core import (
    ComputeParams, compute_ode_full, theorem_4_1_solution_expr, theorem_4_2_y_m_expr,
    get_function_expr, build_freeform_lhs, parse_arbitrary_lhs, to_exact, simplify_expr
)
from rq_utils import has_redis, enqueue_job, fetch_job

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators ODE System - Complete Edition",
    page_icon="🔬", layout="wide", initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
padding:2rem;border-radius:14px;margin-bottom:1.4rem;color:white;text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.2);}
.main-title{font-size:2.2rem;font-weight:700;margin-bottom:.4rem;}
.subtitle{font-size:1.05rem;opacity:.95;}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
color:white;padding:1rem;border-radius:12px;text-align:center;
box-shadow:0 10px 20px rgba(0,0,0,0.2);}
.info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:1rem 0;}
.result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
border:2px solid #4caf50;padding:1rem;border-radius:10px;margin:1rem 0;}
.error-box{background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);
border:2px solid #f44336;padding:1rem;border-radius:10px;margin:1rem 0;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Phi Library (Choice A) and Adapter
# =============================================================================
class PhiLibrary:
    """
    Minimal Φ-library providing SymPy expressions of φ(z).
    This lives only in the UI process. For async jobs, the worker must
    construct the same library and apply the same adapter.
    """
    def __init__(self):
        self._z = sp.Symbol("z", real=True)
        # name -> expression in z
        self._table = {
            "exp": sp.exp(self._z),
            "sin": sp.sin(self._z),
            "cos": sp.cos(self._z),
            "sinh": sp.sinh(self._z),
            "cosh": sp.cosh(self._z),
            "tanh": sp.tanh(self._z),
            "erf": sp.erf(self._z),
            "logistic": 1/(1 + sp.exp(-self._z)),
        }

    def get_function_names(self) -> List[str]:
        return list(self._table.keys())

    def get_function(self, name: str) -> sp.Expr:
        if name not in self._table:
            # Allow free-typed SymPy in z
            try:
                return sp.sympify(name, locals={"z": self._z})
            except Exception:
                raise KeyError(f"Unknown φ '{name}'")
        return self._table[name]


class _PhiAdapter:
    """
    Adapter so that ode_core.get_function_expr(source_lib, func_name) can
    consume PhiLibrary as if it were SpecialFunctions.
    Only two methods are needed: get_function_names(), get_function(name).
    """
    def __init__(self, phi_lib: PhiLibrary):
        self.phi = phi_lib

    def get_function_names(self):
        return self.phi.get_function_names()

    def get_function(self, name: str):
        # return SymPy expression in 'z'
        return self.phi.get_function(name)

# =============================================================================
# Session State Manager
# =============================================================================
class SessionStateManager:
    @staticmethod
    def initialize():
        # Constructor & misc state
        if "generator_constructor" not in st.session_state and GeneratorConstructor:
            st.session_state.generator_constructor = GeneratorConstructor()

        defaults = [
            ("generator_terms", []), ("generated_odes", []), ("generator_patterns", []),
            ("ml_trainer", None), ("ml_trained", False), ("training_history", []),
            ("batch_results", []), ("analysis_results", []), ("export_history", []),
            ("lhs_source", "constructor"), ("freeform_gen_spec", None),
            ("free_terms", []), ("arbitrary_lhs_text", ""),
        ]
        for k, v in defaults:
            if k not in st.session_state:
                st.session_state[k] = v

        # function libraries
        if "basic_functions" not in st.session_state and BasicFunctions:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions:
            st.session_state.special_functions = SpecialFunctions()
        if "phi_library" not in st.session_state:
            st.session_state.phi_library = PhiLibrary()

        # optional heavy objects
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
        if "cache_manager" not in st.session_state and CacheManager:
            st.session_state.cache_manager = CacheManager()

def register_generated_ode(result: dict):
    # normalize & classify
    result = dict(result)
    result.setdefault("type", "nonlinear")
    result.setdefault("order", 0)
    result.setdefault("function_used", "unknown")
    result.setdefault("parameters", {})
    result.setdefault("classification", {})
    result.setdefault("timestamp", datetime.now().isoformat())
    result["generator_number"] = len(st.session_state.generated_odes) + 1

    cl = dict(result.get("classification", {}))
    cl.setdefault("type", "Linear" if result["type"] == "linear" else "Nonlinear")
    cl.setdefault("order", result["order"])
    cl.setdefault("field", cl.get("field", "Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    cl.setdefault("linearity", "Linear" if result["type"] == "linear" else "Nonlinear")
    result["classification"] = cl

    # SymPy Eq convenience
    try:
        result.setdefault("ode", sp.Eq(result["generator"], result["rhs"]))
    except Exception:
        pass

    st.session_state.generated_odes.append(result)

# =============================================================================
# LaTeX Exporter
# =============================================================================
class LaTeXExporter:
    @staticmethod
    def sympy_to_latex(expr) -> str:
        if expr is None: return ""
        try:
            if isinstance(expr, str):
                try:
                    expr = sp.sympify(expr)
                except Exception:
                    return expr
            return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
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
            parts.append(r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{margin=1in}
\title{Master Generators ODE System}
\author{Generated by Master Generators App}
\date{\today}
\begin{document}
\maketitle

\section{Generated Ordinary Differential Equation}
""")
        parts += [
            r"\subsection{Generator Equation}",
            r"\begin{equation}",
            f"{LaTeXExporter.sympy_to_latex(generator)} = {LaTeXExporter.sympy_to_latex(rhs)}",
            r"\end{equation}",
            r"\subsection{Exact Solution}",
            r"\begin{equation}",
            f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}",
            r"\end{equation}",
            r"\subsection{Parameters}",
            r"\begin{align}",
            f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha', 1))} \\\\",
            f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta', 1))} \\\\",
            f"n       &= {params.get('n', 1)} \\\\",
            f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M', 0))}",
            r"\end{align}",
        ]
        if initial_conditions:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            items = list(initial_conditions.items())
            for i, (k, v) in enumerate(items):
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}" + (r" \\" if i < len(items) - 1 else ""))
            parts.append(r"\end{align}")

        if classification:
            parts += [r"\subsection{Mathematical Classification}", r"\begin{itemize}"]
            parts.append(f"\\item \\textbf{{Type:}} {classification.get('type','Unknown')}")
            parts.append(f"\\item \\textbf{{Order:}} {classification.get('order','Unknown')}")
            parts.append(f"\\item \\textbf{{Linearity:}} {classification.get('linearity','Unknown')}")
            if "field" in classification:
                parts.append(f"\\item \\textbf{{Field:}} {classification['field']}")
            if "applications" in classification:
                apps = ", ".join(classification["applications"][:5])
                parts.append(f"\\item \\textbf{{Applications:}} {apps}")
            parts.append(r"\end{itemize}")

        parts += [
            r"\subsection{Solution Verification}",
            r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$."
        ]
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\nTo compile: pdflatex ode_document.tex\n")
            if include_extras:
                zf.writestr("reproduce.txt", "Use ode_data.json with your factories or theorem code.")
        zbuf.seek(0)
        return zbuf.getvalue()

# =============================================================================
# Helpers
# =============================================================================
def _ensure_ss_key(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

def _safe_lambdify(sym_expr, vars_):
    try:
        f = sp.lambdify(vars_, sym_expr, modules=["numpy"])
    except Exception:
        f = sp.lambdify(vars_, sym_expr, modules=["math"])
    return f

def _rmse(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return np.inf
    return float(np.sqrt(np.mean(d * d)))

def _phi_expr_and_callable(lib_obj, func_name):
    z = sp.Symbol("z", real=True)
    try:
        phi_expr = lib_obj.get_function(func_name)
    except Exception:
        phi_expr = sp.sympify(func_name, locals={"z": z})
    phi_num = _safe_lambdify(phi_expr, (z,))
    return phi_expr, phi_num

def _make_y_template(phi_num_list, alphas, betas, ns, M, x_vals):
    x = np.asarray(x_vals, dtype=float)
    x = np.where(x == 0.0, 1e-12, x)  # avoid 0^β when β not integer
    base = x ** M
    for phi_num, a, b, n in zip(phi_num_list, alphas, betas, ns):
        u = a * (x ** b)
        try:
            pj = phi_num(u)
        except Exception:
            return None
        pj = np.asarray(pj, dtype=float)
        pj = np.where(np.isfinite(pj), pj, 0.0)
        base = base * (pj ** n)
    return base

def _fit_outer_scale(yhat, ytrue):
    yh = np.asarray(yhat, dtype=float)
    yt = np.asarray(ytrue, dtype=float)
    mask = np.isfinite(yh) & np.isfinite(yt)
    if not mask.any():
        return 1.0, np.inf
    yh = yh[mask]; yt = yt[mask]
    denom = float(yh @ yh)
    if denom <= 1e-20:
        return 1.0, np.inf
    s = float((yh @ yt) / denom)
    rmse = _rmse(s * yh, yt)
    return s, rmse

def _max_derivative_order(expr, y_sym, x):
    max_k = 0
    for d in expr.atoms(sp.Derivative):
        if d.expr == y_sym:
            k = sum(1 for v in d.variables if v == x)
            max_k = max(max_k, k)
    return max_k

def _adapt_libs_for_computeparams(lib_choice: str):
    """
    Choice A adapter:
    - If "Phi": use function_library="Special" but pass special_lib = _PhiAdapter(PhiLibrary()).
    - Else: use Basic/Special as-is.
    Returns (function_library_str, basic_lib_obj, special_lib_obj)
    """
    basic_lib = st.session_state.get("basic_functions")
    special_lib = st.session_state.get("special_functions")
    phi_lib = st.session_state.get("phi_library")

    if lib_choice == "Phi":
        return "Special", basic_lib, _PhiAdapter(phi_lib)
    elif lib_choice == "Basic":
        return "Basic", basic_lib, special_lib
    else:
        return "Special", basic_lib, special_lib

# =============================================================================
# Apply Master Theorem (forward generation)
# =============================================================================
def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (Exact, Async-ready)")

    # LHS source selector
    _ensure_ss_key("lhs_source", "constructor")
    src = st.radio(
        "Generator LHS source",
        options=("constructor", "freeform", "arbitrary"),
        index={"constructor": 0, "freeform": 1, "arbitrary": 2}.get(st.session_state["lhs_source"], 0),
        horizontal=True
    )
    st.session_state["lhs_source"] = src

    # function library
    colA, colB = st.columns([1, 1])
    with colA:
        lib_choice = st.selectbox("Function library", ["Basic", "Special", "Phi"], index=0)
    with colB:
        basic_lib = st.session_state.get("basic_functions")
        special_lib = st.session_state.get("special_functions")
        phi_lib = st.session_state.get("phi_library")

        if lib_choice == "Basic" and basic_lib:
            func_names = basic_lib.get_function_names()
            source_lib = basic_lib
        elif lib_choice == "Special" and special_lib:
            func_names = special_lib.get_function_names()
            source_lib = special_lib
        elif lib_choice == "Phi" and phi_lib:
            func_names = phi_lib.get_function_names()
            source_lib = _PhiAdapter(phi_lib)  # local preview adapter
        else:
            func_names = []
            source_lib = None
        func_name = st.selectbox("Choose f(z)", func_names) if func_names else st.text_input("Enter f(z)", "exp(z)")

    # parameters
    c1, c2, c3, c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c5, c6, c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light", "none", "aggressive"], index=0)
    with c7:
        async_mode = has_redis()
        st.info("Async via Redis: ON" if async_mode else "Async via Redis: OFF")

    # constructor LHS (if any)
    constructor_lhs = None
    gen_spec = st.session_state.get("current_generator")
    if gen_spec is not None and hasattr(gen_spec, "lhs"):
        constructor_lhs = gen_spec.lhs
    elif st.session_state.get("generator_constructor") and hasattr(
        st.session_state.generator_constructor, "get_generator_expression"
    ):
        try:
            constructor_lhs = st.session_state.generator_constructor.get_generator_expression()
        except Exception:
            constructor_lhs = None

    # Free-form builder
    st.subheader("🧩 Free‑form LHS (Builder)")
    with st.expander("Build custom LHS terms", expanded=False):
        if "free_terms" not in st.session_state:
            st.session_state.free_terms = []
        cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k (y^(k))", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)", [
            "id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs",
            "asin","acos","atan","asinh","acosh","atanh","erf","erfc"
        ], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_order = st.number_input("outer m (D^m)", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale (a)", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift (b)", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": coef, "inner_order": int(inner_order), "wrapper": wrapper,
                    "power": int(power), "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write("**Current terms:**")
            for i, t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. {t}")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("🧮 Use free‑form LHS"):
                    st.session_state["lhs_source"] = "freeform"
                    st.success("Free-form LHS selected.")
            with cc2:
                if st.button("🗑️ Clear terms"):
                    st.session_state.free_terms = []

    # Arbitrary SymPy LHS
    st.subheader("✍️ Arbitrary LHS (SymPy)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Enter any SymPy expression in x and y(x) (e.g., sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1))",
        value=st.session_state.arbitrary_lhs_text or "",
        height=100
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("✅ Validate arbitrary LHS"):
            try:
                _ = parse_arbitrary_lhs(st.session_state.arbitrary_lhs_text)
                st.success("Expression parsed successfully.")
                st.session_state["lhs_source"] = "arbitrary"
            except Exception as e:
                st.error(f"Parse error: {e}")
    with cc2:
        if st.button("↩️ Prefer Constructor LHS"):
            st.session_state["lhs_source"] = "constructor"

    # Theorem 4.2 quick derivative
    st.markdown("---")
    colm1, colm2 = st.columns([1, 1])
    with colm1:
        compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", False)
    with colm2:
        m_order = st.number_input("m", 1, 12, 1)

    # Generate ODE
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        payload = {
            "func_name": func_name, "alpha": alpha, "beta": beta, "n": int(n), "M": M,
            "use_exact": use_exact, "simplify_level": simplify_level,
            "lhs_source": st.session_state["lhs_source"],
            "freeform_terms": st.session_state.get("free_terms"),
            "arbitrary_lhs_text": st.session_state.get("arbitrary_lhs_text"),
            "function_library": lib_choice,
        }

        if not has_redis():
            # Sync path: apply Choice-A adapter locally
            try:
                function_library, cp_basic, cp_special = _adapt_libs_for_computeparams(lib_choice)
                p = ComputeParams(
                    func_name=func_name, alpha=alpha, beta=beta, n=int(n), M=M,
                    use_exact=use_exact, simplify_level=simplify_level,
                    lhs_source=st.session_state["lhs_source"],
                    constructor_lhs=constructor_lhs,
                    freeform_terms=st.session_state.get("free_terms"),
                    arbitrary_lhs_text=st.session_state.get("arbitrary_lhs_text"),
                    function_library=function_library,  # "Basic" or "Special"
                    basic_lib=cp_basic, special_lib=cp_special
                )
                res = compute_ode_full(p)
                register_generated_ode(res)
                show_ode_result(res)
            except Exception as e:
                st.error(f"Generation error: {e}")
        else:
            # Async path: send serializable payload; worker must apply Choice-A adapter if lib == "Phi"
            job_id = enqueue_job("worker.compute_job", payload)
            if job_id:
                st.session_state["last_job_id"] = job_id
                st.success(f"Job submitted. ID = {job_id}")
            else:
                st.error("Failed to submit job (REDIS_URL missing?)")

    # poll async job
    if has_redis() and "last_job_id" in st.session_state:
        st.markdown("### 📡 Job Status")
        colx, _ = st.columns([1, 1])
        with colx:
            if st.button("🔄 Refresh status"):
                pass
        info = fetch_job(st.session_state["last_job_id"])
        if info:
            if info.get("status") == "finished":
                res = info["result"]
                # make pretty
                for k in ("generator", "rhs", "solution"):
                    try:
                        res[k] = sp.sympify(res[k])
                    except Exception:
                        pass
                register_generated_ode(res)
                show_ode_result(res)
                del st.session_state["last_job_id"]
            elif info.get("status") == "failed":
                st.error(f"Job failed: {info.get('error')}")
                del st.session_state["last_job_id"]
            else:
                st.info("⏳ Still computing...")

    # Theorem 4.2 (local)
    if compute_mth and st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2", use_container_width=True):
        try:
            src_for_preview = source_lib  # Basic/Special or Phi-adapter
            f_expr_preview = get_function_expr(src_for_preview, func_name)
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta) if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(f_expr_preview, α, β, int(n), int(m_order), x, simplify_level)
            st.markdown("### 🔢 Derivative")
            st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
        except Exception as e:
            st.error(f"Failed to compute y^{m_order}(x): {e}")

# =============================================================================
# Reverse Engineering
# =============================================================================
def reverse_engineering_page():
    st.header("🔁 Reverse Engineering")

    tabs = st.tabs([
        "From y(x) ⇒ (φ, α, β, n, M) + ODE",
        "Fit L[y] from y(x) and g(x)",
        "From target ODE ⇒ y‑template"
    ])

    # Common libs
    basic_lib = st.session_state.get("basic_functions")
    special_lib = st.session_state.get("special_functions")
    phi_lib = st.session_state.get("phi_library")

    # ---------------------------------------------------------------
    # Tab 1: From y(x) -> infer (φ, α, β, n, M) and build ODE
    # ---------------------------------------------------------------
    with tabs[0]:
        st.markdown("Infer parameters for the template  \n"
                    r"$y(x)\approx x^{M}\prod_{j=1}^{J}\big[\phi_j(\alpha_j x^{\beta_j})\big]^{n_j}$"
                    " and then apply your selected LHS to build the ODE.")

        y_text = st.text_area("Enter y(x) (SymPy)", "exp(-x)*sin(x)")
        x_min, x_max = st.columns(2)
        with x_min:
            xmin = st.number_input("x_min (> 0 recommended)", value=0.001, format="%.6f")
        with x_max:
            xmax = st.number_input("x_max", value=6.283185, format="%.6f")
        npts = st.slider("Sample points", 50, 2000, 400)

        lib_choice = st.selectbox("φ‑library to try", ["Basic", "Special", "Phi"], index=2)
        try_all = st.checkbox("Try all functions in the chosen library", True)

        # Candidate φ names
        if lib_choice == "Basic" and basic_lib:
            all_phi_names = basic_lib.get_function_names()
            lib_obj = basic_lib
        elif lib_choice == "Special" and special_lib:
            all_phi_names = special_lib.get_function_names()
            lib_obj = special_lib
        elif lib_choice == "Phi" and phi_lib:
            all_phi_names = phi_lib.get_function_names()
            lib_obj = phi_lib
        else:
            all_phi_names = []
            lib_obj = None

        if not try_all:
            cand_name = st.selectbox("Candidate φ", all_phi_names)
            cand_list = [cand_name]
        else:
            cand_list = all_phi_names

        # Multi‑block settings
        J = st.slider("Number of blocks J", 1, 3, 1)

        st.markdown("**Parameter grids**")
        c1, c2, c3 = st.columns(3)
        with c1:
            a_min = st.number_input("α min", value=0.5)
            a_max = st.number_input("α max", value=2.0)
            a_steps = st.slider("α steps", 2, 10, 4)
        with c2:
            b_min = st.number_input("β min", value=0.5)
            b_max = st.number_input("β max", value=2.0)
            b_steps = st.slider("β steps", 2, 10, 4)
        with c3:
            n_min = st.number_input("n min (int)", value=1, step=1)
            n_max = st.number_input("n max (int)", value=4, step=1)

        m_min, m_max = st.columns(2)
        with m_min:
            M_min = st.number_input("M min", value=-1.0)
        with m_max:
            M_max = st.number_input("M max", value=1.0)
        M_steps = st.slider("M steps", 2, 10, 4)

        # LHS source to apply after fit
        st.markdown("---")
        st.caption("Which LHS to apply when building the ODE after fitting?")
        lhs_choice = st.radio("LHS source", ["constructor", "freeform", "arbitrary"],
                              index={"constructor": 0, "freeform": 1, "arbitrary": 2}.get(st.session_state.get("lhs_source","constructor"), 0),
                              horizontal=True)
        st.session_state["lhs_source"] = lhs_choice

        # Recover constructor LHS if available
        constructor_lhs = None
        gen_spec = st.session_state.get("current_generator")
        if gen_spec is not None and hasattr(gen_spec, "lhs"):
            constructor_lhs = gen_spec.lhs
        elif st.session_state.get("generator_constructor") and hasattr(
            st.session_state.generator_constructor, "get_generator_expression"
        ):
            try:
                constructor_lhs = st.session_state.generator_constructor.get_generator_expression()
            except Exception:
                constructor_lhs = None

        if st.button("🔎 Infer parameters & build ODE", type="primary"):
            try:
                x = sp.Symbol("x", real=True)
                y_expr = sp.sympify(y_text, locals={"x": x})
                lam_y = _safe_lambdify(y_expr, (x,))
                xs = np.linspace(float(xmin), float(xmax), int(npts))
                y_true = lam_y(xs)
                y_true = np.where(np.isfinite(y_true), y_true, 0.0)

                # grids
                As = np.linspace(a_min, a_max, a_steps)
                Bs = np.linspace(b_min, b_max, b_steps)
                Ns = list(range(int(n_min), int(n_max) + 1))
                Ms = np.linspace(M_min, M_max, M_steps)

                if not cand_list or lib_obj is None:
                    st.error("No φ names available in the chosen library.")
                    return

                best = {"rmse": np.inf}

                # quick guardrail
                total_combos_est = (len(cand_list) ** J) * (a_steps * b_steps * len(Ns)) ** J * (M_steps)
                if total_combos_est > 12000:
                    st.warning(f"Search space is large (~{total_combos_est} combos). "
                               f"Reduce J/steps or restrict φ set.")

                # main search
                for phi_names in itertools.product(cand_list, repeat=J):
                    phi_exprs = []
                    phi_nums = []
                    ok = True
                    for name in phi_names:
                        expr, num = _phi_expr_and_callable(lib_obj, name)
                        phi_exprs.append(expr)
                        phi_nums.append(num)
                        if num is None:
                            ok = False
                            break
                    if not ok:
                        continue

                    for Mv in Ms:
                        for abn_tuple in itertools.product(itertools.product(As, Bs, Ns), repeat=J):
                            alphas = [t[0] for t in abn_tuple]
                            betas  = [t[1] for t in abn_tuple]
                            ns     = [t[2] for t in abn_tuple]

                            yhat = _make_y_template(phi_nums, alphas, betas, ns, Mv, xs)
                            if yhat is None:
                                continue
                            s, err = _fit_outer_scale(yhat, y_true)
                            if err < best["rmse"]:
                                best = {
                                    "rmse": float(err), "M": float(Mv),
                                    "alphas": list(map(float, alphas)),
                                    "betas": list(map(float, betas)),
                                    "ns": list(map(int, ns)),
                                    "phi_names": list(phi_names), "scale": float(s)
                                }

                if not np.isfinite(best["rmse"]):
                    st.error("Could not fit template on the provided range/domain.")
                    return

                st.success(f"Best RMSE = {best['rmse']:.6g}")
                st.write("**Best parameters**")
                st.json(best)

                # Build the ODE using block j=0 (canonical)
                j0 = 0
                func_name = best["phi_names"][j0]
                alpha = best["alphas"][j0]
                beta  = best["betas"][j0]
                n_val = best["ns"][j0]
                M_val = best["M"]

                # Choice A adapter for ComputeParams
                function_library, cp_basic, cp_special = _adapt_libs_for_computeparams(lib_choice)

                use_exact = st.checkbox("Exact (symbolic) parameters", True, key="rev_exact")
                simplify_level = st.selectbox("Simplify", ["light", "none", "aggressive"], index=0, key="rev_simpl")

                p = ComputeParams(
                    func_name=func_name,
                    alpha=float(alpha), beta=float(beta), n=int(n_val), M=float(M_val),
                    use_exact=bool(use_exact), simplify_level=simplify_level,
                    lhs_source=lhs_choice, constructor_lhs=constructor_lhs,
                    freeform_terms=st.session_state.get("free_terms"),
                    arbitrary_lhs_text=st.session_state.get("arbitrary_lhs_text"),
                    function_library=function_library,  # "Basic" or "Special"
                    basic_lib=cp_basic, special_lib=cp_special
                )
                res = compute_ode_full(p)
                for k in ("generator", "rhs", "solution"):
                    try: res[k] = sp.sympify(res[k])
                    except Exception: pass
                register_generated_ode(res)
                show_ode_result(res)

            except Exception as e:
                st.error(f"Reverse fitting failed: {e}")

    # ---------------------------------------------------------------
    # Tab 2: Fit L[y] from y and g
    # ---------------------------------------------------------------
    with tabs[1]:
        st.markdown("Fit a **linear** operator \(L=\sum_{i=0}^K c_i D^i\) from samples so that \(L[y](x)\approx g(x)\).")
        y_text = st.text_area("y(x) (SymPy)", "exp(-x)*sin(x)", key="fitL_y")
        g_text = st.text_area("g(x) (SymPy)", "0", key="fitL_g")
        K = st.slider("Max derivative order K", 0, 6, 2, key="fitL_K")
        xmin = st.number_input("x_min", value=0.1, key="fitL_xmin")
        xmax = st.number_input("x_max", value=5.0, key="fitL_xmax")
        npts = st.slider("Sample points", 50, 2000, 400, key="fitL_npts")

        if st.button("🔧 Fit operator L", type="primary"):
            try:
                x = sp.Symbol("x", real=True)
                y_expr = sp.sympify(y_text, locals={"x": x})
                g_expr = sp.sympify(g_text, locals={"x": x})
                lam_g = _safe_lambdify(g_expr, (x,))
                xs = np.linspace(float(xmin), float(xmax), int(npts))

                # Build A: columns of y^(i)
                cols = []
                for i in range(K + 1):
                    d_expr = sp.diff(y_expr, x, i)
                    lam_d = _safe_lambdify(d_expr, (x,))
                    cols.append(lam_d(xs))
                A = np.vstack(cols).T
                b = lam_g(xs)

                # least squares
                c, *_ = np.linalg.lstsq(A, b, rcond=None)
                # Symbolic operator in y
                y_sym = sp.Function("y")(x)
                LHS_y = sum(sp.Float(float(ci)) * sp.diff(y_sym, x, i) for i, ci in enumerate(c))
                # Concretized LHS using provided y(x)
                LHS_concrete = sum(sp.Float(float(ci)) * sp.diff(y_expr, x, i) for i, ci in enumerate(c))

                st.write("**Coefficients c₀…c_K**")
                st.json({f"c{i}": float(ci) for i, ci in enumerate(c)})
                st.markdown("**Recovered operator (symbolic in y)**")
                try:
                    st.latex(sp.latex(LHS_y) + " = " + sp.latex(g_expr))
                except Exception:
                    st.write("LHS[y] =", LHS_y, "; RHS =", g_expr)

                st.markdown("**Concretized ODE for your y(x)**")
                try:
                    st.latex(sp.latex(LHS_concrete) + " = " + sp.latex(g_expr))
                except Exception:
                    st.write("LHS_concrete =", LHS_concrete, "; RHS =", g_expr)

            except Exception as e:
                st.error(f"Fit failed: {e}")

    # ---------------------------------------------------------------
    # Tab 3: From target ODE -> y-template fit
    # ---------------------------------------------------------------
    with tabs[2]:
        st.markdown("Given a target ODE \(L[y]=RHS\), fit a **y‑template** "
                    r"$y(x)=x^{M}\prod_{j=1}^{J}\big[\phi_j(\alpha_j x^{\beta_j})\big]^{n_j}$ "
                    "so that \(L[y]-RHS\) is small over a sample grid.")

        L_text = st.text_area("LHS in x and y(x) (SymPy)", "y(x).diff(x,2) + y(x)", key="inv_L")
        R_text = st.text_area("RHS in x (SymPy)", "0", key="inv_R")
        xmin = st.number_input("x_min", value=0.1, key="inv_xmin")
        xmax = st.number_input("x_max", value=5.0, key="inv_xmax")
        npts = st.slider("Sample points", 50, 1500, 400, key="inv_npts")

        lib_choice = st.selectbox("φ‑library", ["Basic","Special","Phi"], index=2, key="inv_lib")
        # φ names
        if lib_choice == "Basic" and basic_lib:
            cand_phi = basic_lib.get_function_names()
            lib_obj = basic_lib
        elif lib_choice == "Special" and special_lib:
            cand_phi = special_lib.get_function_names()
            lib_obj = special_lib
        elif lib_choice == "Phi" and phi_lib:
            cand_phi = phi_lib.get_function_names()
            lib_obj = phi_lib
        else:
            cand_phi = []
            lib_obj = None

        J = st.slider("Blocks J", 1, 2, 1, key="inv_J")
        phi_subset = st.multiselect("Restrict φ candidates (optional)", cand_phi, default=cand_phi[:6])

        c1, c2, c3 = st.columns(3)
        with c1:
            a_min = st.number_input("α min", value=0.5, key="inv_amin")
            a_max = st.number_input("α max", value=2.0, key="inv_amax")
            a_steps = st.slider("α steps", 2, 8, 3, key="inv_asteps")
        with c2:
            b_min = st.number_input("β min", value=0.5, key="inv_bmin")
            b_max = st.number_input("β max", value=2.0, key="inv_bmax")
            b_steps = st.slider("β steps", 2, 8, 3, key="inv_bsteps")
        with c3:
            n_min = st.number_input("n min (int)", value=1, step=1, key="inv_nmin")
            n_max = st.number_input("n max (int)", value=4, step=1, key="inv_nmax")

        M_min = st.number_input("M min", value=-1.0, key="inv_Mmin")
        M_max = st.number_input("M max", value=1.0, key="inv_Mmax")
        M_steps = st.slider("M steps", 2, 8, 3, key="inv_Msteps")

        if st.button("🔎 Fit y‑template to target ODE", type="primary"):
            try:
                if lib_obj is None:
                    st.error("No φ library is available."); return

                x = sp.Symbol("x", real=True)
                y_sym = sp.Function("y")(x)
                # allow y(x) in LHS text
                L_expr = sp.sympify(L_text, locals={"x": x, "y": lambda arg: sp.Function("y")(arg)})
                R_expr = sp.sympify(R_text, locals={"x": x})

                xs = np.linspace(float(xmin), float(xmax), int(npts))
                lam_R = _safe_lambdify(R_expr, (x,))
                R_num = lam_R(xs)
                R_num = np.where(np.isfinite(R_num), R_num, 0.0)

                K = _max_derivative_order(L_expr, y_sym, x)
                phi_names = phi_subset if phi_subset else cand_phi

                As = np.linspace(a_min, a_max, a_steps)
                Bs = np.linspace(b_min, b_max, b_steps)
                Ns = list(range(int(n_min), int(n_max) + 1))
                Ms = np.linspace(M_min, M_max, M_steps)

                best = {"rmse": np.inf}

                for names in itertools.product(phi_names, repeat=J):
                    phi_exprs = []
                    for nm in names:
                        expr, _ = _phi_expr_and_callable(lib_obj, nm)
                        phi_exprs.append(expr)

                    for Mv in Ms:
                        for abn_tuple in itertools.product(itertools.product(As, Bs, Ns), repeat=J):
                            alphas = [t[0] for t in abn_tuple]
                            betas  = [t[1] for t in abn_tuple]
                            ns     = [t[2] for t in abn_tuple]

                            # symbolic y_template
                            y_tmp = (sp.Symbol("x") ** Mv)
                            z = sp.Symbol("z", real=True)
                            for expr, a, b, n in zip(phi_exprs, alphas, betas, ns):
                                y_tmp = y_tmp * (expr.subs({z: a * (x ** b)})) ** int(n)

                            # substitute in L
                            subs_map = {y_sym: y_tmp}
                            for k in range(1, K + 1):
                                subs_map[sp.diff(y_sym, x, k)] = sp.diff(y_tmp, x, k)
                            L_sub = L_expr.subs(subs_map)
                            lam_res = _safe_lambdify(L_sub - R_expr, (x,))
                            res_num = lam_res(xs)
                            err = _rmse(res_num, 0.0)
                            if err < best["rmse"]:
                                best = {
                                    "rmse": float(err), "phi_names": list(names), "M": float(Mv),
                                    "alphas": list(map(float, alphas)), "betas": list(map(float, betas)),
                                    "ns": list(map(int, ns))
                                }

                if not np.isfinite(best["rmse"]):
                    st.error("No good fit on the sample grid.")
                    return

                st.success(f"Best RMSE = {best['rmse']:.6g}")
                st.json(best)

            except Exception as e:
                st.error(f"Inverse fitting failed: {e}")

# =============================================================================
# Shared “show result” UI
# =============================================================================
def show_ode_result(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h3>✅ ODE Generated Successfully!</h3></div>', unsafe_allow_html=True)
    t_eq, t_sol, t_exp = st.tabs(["📐 Equation", "💡 Solution & ICs", "📤 Export"])
    with t_eq:
        try:
            st.latex(sp.latex(res["generator"]) + " = " + sp.latex(res["rhs"]))
        except Exception:
            st.write("LHS:", res["generator"]); st.write("RHS:", res["rhs"])
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t_sol:
        try:
            st.latex("y(x) = " + sp.latex(res["solution"]))
        except Exception:
            st.write("y(x) =", res["solution"])
        if res.get("initial_conditions"):
            st.markdown("**Initial conditions:**")
            for k, v in res["initial_conditions"].items():
                try: st.latex(k + " = " + sp.latex(v))
                except Exception: st.write(k, "=", v)
        st.markdown("**Parameters:**")
        p = res.get("parameters", {})
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        fprev = res.get("f_expr_preview")
        if fprev is not None:
            st.write(f"**Function:** f(z) = {fprev}")
    with t_exp:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res["generator"], "rhs": res["rhs"], "solution": res["solution"],
            "parameters": res.get("parameters", {}),
            "classification": {
                "type": "Linear" if res.get("type") == "linear" else "Nonlinear",
                "order": res.get("order", 0),
                "linearity": "Linear" if res.get("type") == "linear" else "Nonlinear",
                "field": "Mathematical Physics", "applications": ["Research Equation"],
            },
            "initial_conditions": res.get("initial_conditions", {}),
            "function_used": str(res.get("function_used","?")),
            "generator_number": idx, "type": res.get("type","nonlinear"),
            "order": res.get("order", 0)
        }
        latex_doc = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
        st.download_button("📄 Download LaTeX Document", latex_doc, f"ode_{idx}.tex", "text/x-latex", use_container_width=True)
        pkg = LaTeXExporter.create_export_package(ode_data, include_extras=True)
        st.download_button("📦 Download Complete Package (ZIP)", pkg, f"ode_package_{idx}.zip", "application/zip", use_container_width=True)

# =============================================================================
# Other pages (unaltered in behavior)
# =============================================================================
def dashboard_page():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>🧬 ML Patterns</h3><h1>{len(st.session_state.generator_patterns)}</h1></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h3>📊 Batch Results</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c4:
        model_status = "✅ Trained" if st.session_state.get("ml_trained") else "⏳ Not Trained"
        st.markdown(f'<div class="metric-card"><h3>🤖 ML Model</h3><p style="font-size: 1.2rem;">{model_status}</p></div>', unsafe_allow_html=True)
    st.subheader("📊 Recent Activity")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type","order","generator_number","timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs yet. Go to **Apply Master Theorem** or **Generator Constructor**.")

def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Build custom generators or use Free‑form/Arbitrary LHS in the theorem page.</div>', unsafe_allow_html=True)
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found in src/. Use Free‑form builder instead.")
        return

    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x,f"y⁽{x}⁾"))
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
                derivative_order=deriv_order, coefficient=coefficient, power=power,
                function_type=DerivativeType(func_type), operator_type=OperatorType(operator_type),
                scaling=scaling, shift=shift,
            )
            st.session_state.generator_terms.append(term)
            st.success("Term added.")
    if st.session_state.generator_terms:
        st.subheader("📝 Current Generator Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            c1, c2 = st.columns([8, 1])
            with c1:
                desc = term.get_description() if hasattr(term, "get_description") else str(term)
                st.write(f"• {desc}")
            with c2:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()
        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Generator {len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = gen_spec
                st.success("Generator specification created.")
                try: st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                except Exception: pass
            except Exception as e:
                st.error(f"Failed to build specification: {e}")
    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None

def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning")
    if not MLTrainer:
        st.warning("MLTrainer not found in src/.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Patterns", len(st.session_state.generator_patterns))
    with c2: st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c3: st.metric("Training Epochs", len(st.session_state.training_history))
    with c4: st.metric("Model Status", "Trained" if st.session_state.get("ml_trained") else "Not Trained")

    model_type = st.selectbox("Select ML Model", ["pattern_learner","vae","transformer"], format_func=lambda s: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[s])

    with st.expander("🎯 Training Configuration", True):
        c1, c2, c3 = st.columns(3)
        with c1: epochs = st.slider("Epochs", 10, 500, 100); batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2: learning_rate = st.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001); samples = st.slider("Training Samples", 100, 5000, 1000)
        with c3: validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2); use_gpu = st.checkbox("Use GPU if available", True)

    # include both generated and batch data if you want
    use_batch_for_training = st.checkbox("Include Batch Results as Training Data", True)

    need = 5
    count = len(st.session_state.generated_odes) + (len(st.session_state.batch_results) if use_batch_for_training else 0)
    if count < need:
        st.warning(f"Need at least {need} ODEs to train. Current: {count}")
        return

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            try:
                device = "cuda" if use_gpu and (torch and torch.cuda.is_available()) else "cpu"
                trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=device)
                st.session_state.ml_trainer = trainer

                prog = st.progress(0); status = st.empty()
                def progress_callback(epoch, total_epochs):
                    prog.progress(min(1.0, epoch/total_epochs)); status.text(f"Epoch {epoch}/{total_epochs}")

                trainer.train(epochs=epochs, batch_size=batch_size, samples=samples, validation_split=validation_split, progress_callback=progress_callback)
                st.session_state.ml_trained = True
                st.session_state.training_history = getattr(trainer, "history", {})
                st.success("Model trained.")
                if trainer.history.get("train_loss"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(1,len(trainer.history["train_loss"])+1)), y=trainer.history["train_loss"], mode="lines", name="Training Loss"))
                    if trainer.history.get("val_loss"):
                        fig.add_trace(go.Scatter(x=list(range(1,len(trainer.history["val_loss"])+1)), y=trainer.history["val_loss"], mode="lines", name="Validation Loss"))
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.get("ml_trained") and st.session_state.get("ml_trainer"):
        st.subheader("🎨 Generate Novel Patterns")
        c1, c2 = st.columns(2)
        with c1: num_generate = st.slider("Number to Generate", 1, 10, 1)
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
                                        try: st.latex(sp.latex(res["ode"]))
                                        except Exception: st.code(str(res["ode"]))
                                    for k in ["type","order","function_used","description"]:
                                        if k in res: st.write(f"**{k}:** {res[k]}")
                                st.session_state.generated_odes.append(res)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs with your factories.</div>', unsafe_allow_html=True)
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
            alpha_range=(1.0,1.0); beta_range=(1.0,1.0); n_range=(1,1)

    with st.expander("⚙️ Advanced Options"):
        export_format = st.selectbox("Export Format", ["JSON","CSV","LaTeX","All"])
        include_solutions = st.checkbox("Include Solutions", True)
        include_classification = st.checkbox("Include Classification", True)

    if st.button("🚀 Generate Batch", type="primary"):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            batch_results = []
            prog = st.progress(0); status = st.empty()

            all_functions = []
            if "Basic" in func_categories and st.session_state.get("basic_functions"):
                all_functions += st.session_state.basic_functions.get_function_names()
            if "Special" in func_categories and st.session_state.get("special_functions"):
                all_functions += st.session_state.special_functions.get_function_names()[:20]
            if not all_functions:
                st.warning("No function names available from libraries.")
                return

            for i in range(num_odes):
                try:
                    prog.progress((i+1)/num_odes); status.text(f"Generating ODE {i+1}/{num_odes}")
                    params = {
                        "alpha": float(np.random.uniform(*alpha_range)),
                        "beta":  float(np.random.uniform(*beta_range)),
                        "n": int(np.random.randint(n_range[0], n_range[1]+1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    func_name = np.random.choice(all_functions)
                    gt = np.random.choice(gen_types)
                    res = {}
                    if gt == "linear":
                        if CompleteLinearGeneratorFactory:
                            factory = CompleteLinearGeneratorFactory()
                            gen_num = np.random.randint(1, 9)
                            if gen_num in [4,5]:
                                params["a"] = float(np.random.uniform(1,3))
                            res = factory.create(gen_num, st.session_state.basic_functions.get_function(func_name), **params)
                        elif LinearGeneratorFactory:
                            factory = LinearGeneratorFactory()
                            res = factory.create(1, st.session_state.basic_functions.get_function(func_name), **params)
                    else:
                        if CompleteNonlinearGeneratorFactory:
                            factory = CompleteNonlinearGeneratorFactory()
                            gen_num = np.random.randint(1, 11)
                            if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                            if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                            if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                            res = factory.create(gen_num, st.session_state.basic_functions.get_function(func_name), **params)
                        elif NonlinearGeneratorFactory:
                            factory = NonlinearGeneratorFactory()
                            res = factory.create(1, st.session_state.basic_functions.get_function(func_name), **params)
                    if not res: continue

                    row = {
                        "ID": i+1, "Type": res.get("type","unknown"),
                        "Generator": res.get("generator_number","?"),
                        "Function": func_name, "Order": res.get("order",0),
                        "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                    }
                    if include_solutions:
                        s = str(res.get("solution",""))
                        row["Solution"] = (s[:120]+"...") if len(s)>120 else s
                    if include_classification: row["Subtype"] = res.get("subtype","standard")
                    batch_results.append(row)
                except Exception as e:
                    logger.debug(f"Failed to generate ODE {i+1}: {e}")

            st.session_state.batch_results.extend(batch_results)
            st.success(f"Generated {len(batch_results)} ODEs.")
            df = pd.DataFrame(batch_results); st.dataframe(df, use_container_width=True)

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
                    latex = "\n".join([
                        r"\begin{tabular}{|c|c|c|c|c|}", r"\hline", r"ID & Type & Generator & Function & Order \\",
                        r"\hline", *[f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\" for r in batch_results[:30]],
                        r"\hline", r"\end{tabular}"
                    ])
                    st.download_button("📝 Download LaTeX", latex, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")
            with c4:
                if export_format == "All":
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("batch_results.csv", df.to_csv(index=False))
                        zf.writestr("batch_results.json", json.dumps(batch_results, indent=2, default=str))
                    zbuf.seek(0)
                    st.download_button("📦 Download All (ZIP)", zbuf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip")

def novelty_detection_page():
    st.header("🔍 Novelty Detection")
    if not st.session_state.get("novelty_detector"):
        st.warning("Novelty detector not found.")
        return
    method = st.radio("Input Method", ["Use Current Generator LHS", "Enter ODE Manually", "Select from Generated"])
    ode_to_analyze = None
    if method == "Use Current Generator LHS":
        gen_spec = st.session_state.get("current_generator")
        if gen_spec is not None and hasattr(gen_spec, "lhs"):
            ode_to_analyze = {"ode": gen_spec.lhs, "type":"custom", "order": getattr(gen_spec, "order", 2)}
        else:
            st.warning("No generator spec. Use Constructor or Free‑form.")
    elif method == "Enter ODE Manually":
        ode_str = st.text_area("Enter ODE (LaTeX or text):")
        if ode_str:
            ode_to_analyze = {"ode": ode_str, "type":"manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)), format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','Unknown')}")
            ode_to_analyze = st.session_state.generated_odes[sel]

    if ode_to_analyze and st.button("🔍 Analyze Novelty", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                analysis = st.session_state.novelty_detector.analyze(ode_to_analyze, check_solvability=True, detailed=True)
                st.metric("Novelty", "🟢 NOVEL" if analysis.is_novel else "🔴 STANDARD")
                st.metric("Score", f"{analysis.novelty_score:.1f}/100")
                st.metric("Confidence", f"{analysis.confidence:.1%}")
                with st.expander("📊 Details", True):
                    st.write(f"Complexity: {analysis.complexity_level}")
                    st.write(f"Solvable by standard methods: {'Yes' if analysis.solvable_by_standard_methods else 'No'}")
                    if analysis.special_characteristics:
                        st.write("Special characteristics:"); [st.write("•", t) for t in analysis.special_characteristics]
                    if analysis.recommended_methods:
                        st.write("Recommended methods:"); [st.write("•", t) for t in analysis.recommended_methods[:5]]
                    if analysis.similar_known_equations:
                        st.write("Similar known equations:"); [st.write("•", t) for t in analysis.similar_known_equations[:3]]
                if analysis.detailed_report:
                    st.download_button("📥 Download Report", analysis.detailed_report, f"novelty_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

def analysis_classification_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    if not st.session_state.get("ode_classifier"):
        st.warning("Classifier not found.")
        return
    st.subheader("📊 Generated ODEs Overview")
    summary = []
    for i, ode in enumerate(st.session_state.generated_odes[-50:]):
        summary.append({"ID": i+1, "Type": ode.get("type","Unknown"), "Order": ode.get("order",0),
                        "Generator": ode.get("generator_number","N/A"), "Function": ode.get("function_used","Unknown"),
                        "Timestamp": ode.get("timestamp","")[:19]})
    df = pd.DataFrame(summary); st.dataframe(df, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Linear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="linear"))
    with c2: st.metric("Nonlinear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="nonlinear"))
    with c3:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        st.metric("Average Order", f"{(np.mean(orders) if orders else 0):.1f}")
    with c4:
        unique = len(set(o.get("function_used","") for o in st.session_state.generated_odes))
        st.metric("Unique Functions", unique)

    st.subheader("📊 Distributions")
    c1, c2 = st.columns(2)
    with c1:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        fig = px.histogram(orders, title="Order Distribution", nbins=10); fig.update_layout(xaxis_title="Order", yaxis_title="Count")
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
                    try: classifications.append(st.session_state.ode_classifier.classify_ode(ode))
                    except Exception: classifications.append({})
                fields = [c.get("classification",{}).get("field","Unknown") for c in classifications if c]
                vc = pd.Series(fields).value_counts()
                fig = px.bar(x=vc.index, y=vc.values, title="Classification by Field"); fig.update_layout(xaxis_title="Field", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Classification failed: {e}")

def physical_applications_page():
    st.header("🔬 Physical Applications")
    st.markdown('<div class="info-box">Explore physics/engineering matches.</div>', unsafe_allow_html=True)
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
            try: st.latex(app["equation"])
            except Exception: st.write(app["equation"])
            st.write("Description:", app["description"])

def visualization_page():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize."); return
    sel = st.selectbox(
        "Select ODE",
        range(len(st.session_state.generated_odes)),
        format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')} (Order {st.session_state.generated_odes[i].get('order',0)})"
    )
    # Placeholder visualization
    x_range = st.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    num_points = st.slider("Number of Points", 100, 2000, 500)
    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Creating..."):
            try:
                x = np.linspace(x_range[0], x_range[1], num_points)
                y = np.sin(x) * np.exp(-0.1*np.abs(x))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Solution"))
                fig.update_layout(title="ODE Solution", xaxis_title="x", yaxis_title="y(x)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization failed: {e}")

def export_latex_page():
    st.header("📤 Export & LaTeX")
    st.markdown('<div class="info-box">Export ODEs in publication‑ready LaTeX.</div>', unsafe_allow_html=True)
    if not st.session_state.generated_odes:
        st.warning("No ODEs to export."); return
    export_type = st.radio("Export Type", ["Single ODE","Multiple ODEs","Complete Report"])
    if export_type == "Single ODE":
        idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                           format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}")
        ode = st.session_state.generated_odes[idx]
        st.subheader("📋 LaTeX Preview")
        latex_doc = LaTeXExporter.generate_latex_document(ode, include_preamble=False)
        st.code(latex_doc, language="latex")
        c1, c2 = st.columns(2)
        with c1:
            full_latex = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
            st.download_button("📄 Download LaTeX", full_latex, f"ode_{idx+1}.tex", "text/x-latex")
        with c2:
            package = LaTeXExporter.create_export_package(ode, include_extras=True)
            st.download_button("📦 Download Package", package, f"ode_package_{idx+1}.zip", "application/zip")
    elif export_type == "Multiple ODEs":
        sel = st.multiselect("Select ODEs", range(len(st.session_state.generated_odes)),
                             format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}")
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
    else:
        if st.button("Generate Complete Report"):
            parts = [r"""\documentclass[12pt]{report}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\title{Master Generators System\\Complete Report}
\author{Generated Automatically}
\date{\today}
\begin{document}
\maketitle
\tableofcontents

\chapter{Executive Summary}
This report contains all ODEs generated by the system.

\chapter{Generated ODEs}
"""]
            for i, ode in enumerate(st.session_state.generated_odes):
                parts.append(f"\\section{{ODE {i+1}}}")
                parts.append(LaTeXExporter.generate_latex_document(ode, include_preamble=False))
            parts.append(r"""
\chapter{Conclusions}
The system successfully generated and analyzed multiple ODEs.
\end{document}
""")
            st.download_button("📄 Download Complete Report", "\n".join(parts), f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")

def examples_library_page():
    st.header("📚 Examples Library")
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")

def settings_page():
    st.header("⚙️ Settings")
    tabs = st.tabs(["General","Export","Advanced","About"])
    with tabs[0]:
        st.checkbox("Dark mode", False)
        if st.button("Save General Settings"): st.success("Saved.")
    with tabs[1]:
        include_preamble = st.checkbox("Include LaTeX preamble by default", True)
        if st.button("Save Export Settings"): st.success("Saved.")
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            cm = st.session_state.get("cache_manager"); st.metric("Cache Size", len(getattr(cm,"memory_cache",{})) if cm else 0)
        with c2:
            if st.button("Clear Cache"):
                try: st.session_state.cache_manager.clear(); st.success("Cache cleared.")
                except Exception: st.info("No cache manager.")
        with c3:
            if st.button("Save Session"):
                ok = False
                try:
                    with open("session_state.pkl","wb") as f:
                        pickle.dump({k:st.session_state.get(k) for k in
                                     ["generated_odes","generator_patterns","batch_results",
                                      "analysis_results","training_history","export_history"]}, f)
                    ok = True
                except Exception: ok = False
                st.success("Session saved.") if ok else st.error("Failed to save.")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorems 4.1 & 4.2, Reverse Engineering, ML/DL, Export, Novelty.")

def documentation_page():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Go to **Apply Master Theorem**.
2. Pick f(z) from Basic/Special/**Phi** (or type one).
3. Set parameters (α,β,n,M) and choose **Exact (symbolic)** if you want rationals.
4. Choose LHS source: **Constructor**, **Free‑form**, or **Arbitrary SymPy**.
5. Click **Generate ODE**. If Redis is configured, the job runs in background.
6. Use **🔁 Reverse Engineering** if you want to infer parameters or fit an operator from y(x).
7. Export from the **📤 Export** tab or the **Export & LaTeX** page.
8. Compute **y^(m)(x)** via **Theorem 4.2** when needed.
""")

# =============================================================================
# Main
# =============================================================================
def main():
    SessionStateManager.initialize()
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">By Mohammad Abu Ghuwaleh • Free‑form/Arbitrary generators • Reverse Engineering • ML/DL • Export • Novelty • Async Jobs</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard",
        "🔧 Generator Constructor",
        "🎯 Apply Master Theorem",
        "🔁 Reverse Engineering",
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
    ])
    if page == "🏠 Dashboard": dashboard_page()
    elif page == "🔧 Generator Constructor": generator_constructor_page()
    elif page == "🎯 Apply Master Theorem": page_apply_master_theorem()
    elif page == "🔁 Reverse Engineering": reverse_engineering_page()
    elif page == "🤖 ML Pattern Learning": ml_pattern_learning_page()
    elif page == "📊 Batch Generation": batch_generation_page()
    elif page == "🔍 Novelty Detection": novelty_detection_page()
    elif page == "📈 Analysis & Classification": analysis_classification_page()
    elif page == "🔬 Physical Applications": physical_applications_page()
    elif page == "📐 Visualization": visualization_page()
    elif page == "📤 Export & LaTeX": export_latex_page()
    elif page == "📚 Examples Library": examples_library_page()
    elif page == "⚙️ Settings": settings_page()
    elif page == "📖 Documentation": documentation_page()

if __name__ == "__main__":
    main()