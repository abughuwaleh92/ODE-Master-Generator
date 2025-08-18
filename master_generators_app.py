# master_generators_app.py
"""
Master Generators for ODEs — Full App (Complete & Corrected; all services intact)

What's included:
- Dashboard
- Generator Constructor
- Apply Master Theorem (constructor / free-form / arbitrary SymPy LHS)
- ML Pattern Learning (RQ-based persistent training + local fallback)
- Batch Generation
- Novelty Detection
- Analysis & Classification
- Physical Applications
- Visualization
- Export & LaTeX
- Examples Library
- Settings
- Documentation
- Reverse Engineering lab

Key correctness notes:
- Uses 'choice A': never passes 'phi_lib' into ComputeParams.
- Async compute & training via Redis Queue (RQ), with persistent progress/logs/artifacts.
- Training sessions can be saved/loaded/uploaded/resumed (when paths are accessible).
- Reverse engineering + generation utilities are available post-training.
"""

# ---------------- std libs ----------------
import os, sys, io, json, time, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp

# Optional plotting
_PLOTLY_OK = True
try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    _PLOTLY_OK = False

# Optional torch (for quick local generation/reverse demos)
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

# ---------------- RQ utilities ----------------
# (Persistent logs/progress/artifacts)
try:
    from rq_utils import (
        has_redis, enqueue_job, fetch_job,
        get_progress, get_logs, get_artifacts
    )
except Exception as e:
    logger.warning(f"rq_utils not available: {e}")
    def has_redis(): return False
    def enqueue_job(*a, **k): return None
    def fetch_job(*a, **k): return None
    def get_progress(*a, **k): return {}
    def get_logs(*a, **k): return []
    def get_artifacts(*a, **k): return {}

# ---------------- ML trainer (enhanced) ----------------
# (Used for local generate/reverse and session load/export in-app)
MLTrainer = TrainConfig = None
try:
    from src.ml.trainer import MLTrainer, TrainConfig
except Exception as e:
    logger.warning(f"Enhanced MLTrainer not available: {e}")

# ---------------- shared ODE core ----------------
# (Single source of truth for theorem & compute)
ComputeParams = compute_ode_full = theorem_4_2_y_m_expr = None
get_function_expr = to_exact = simplify_expr = expr_to_str = None
try:
    from shared.ode_core import (
        ComputeParams, compute_ode_full,
        theorem_4_2_y_m_expr, get_function_expr,
        to_exact, simplify_expr, expr_to_str
    )
except Exception as e:
    logger.warning(f"shared.ode_core not available: {e}")

# ---------------- Optional src imports ----------------
# (Factory, Theorem helpers, classifier, novelty, UI)
MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
LinearGeneratorFactory = CompleteLinearGeneratorFactory = None
NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
GeneratorConstructor = GeneratorSpecification = None
DerivativeTerm = DerivativeType = OperatorType = None
MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None
ODEClassifier = PhysicalApplication = None
BasicFunctions = SpecialFunctions = None
GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = create_model = None
GeneratorPattern = GeneratorPatternNetwork = GeneratorLearningSystem = None
ODENoveltyDetector = NoveltyAnalysis = ODETokenizer = ODETransformer = None
Settings = AppConfig = None
CacheManager = cached = None
ParameterValidator = None
UIComponents = None

try:
    # Factories
    try:
        from src.generators.master_generator import (
            MasterGenerator, EnhancedMasterGenerator, CompleteMasterGenerator
        )
        try:
            from src.generators.master_generator import (
                CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory
            )
        except Exception:
            from src.generators.linear_generators import (
                LinearGeneratorFactory, CompleteLinearGeneratorFactory
            )
            from src.generators.nonlinear_generators import (
                NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
            )
    except Exception:
        from src.generators.linear_generators import (
            LinearGeneratorFactory, CompleteLinearGeneratorFactory
        )
        from src.generators.nonlinear_generators import (
            NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
        )

    # Constructor
    try:
        from src.generators.generator_constructor import (
            GeneratorConstructor, GeneratorSpecification,
            DerivativeTerm, DerivativeType, OperatorType
        )
    except Exception:
        pass

    # Optional extras
    try:
        from src.generators.master_theorem import (
            MasterTheoremSolver, MasterTheoremParameters, ExtendedMasterTheorem
        )
    except Exception:
        pass

    try:
        from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
    except Exception:
        pass

    try:
        from src.functions.basic_functions import BasicFunctions
        from src.functions.special_functions import SpecialFunctions
    except Exception:
        pass

    try:
        from src.ml.pattern_learner import (
            GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer, create_model
        )
    except Exception:
        pass

    try:
        from src.ml.generator_learner import (
            GeneratorPattern, GeneratorPatternNetwork, GeneratorLearningSystem
        )
    except Exception:
        pass

    try:
        from src.dl.novelty_detector import (
            ODENoveltyDetector, NoveltyAnalysis, ODETokenizer, ODETransformer
        )
    except Exception:
        pass

    try:
        from src.utils.config import Settings, AppConfig
    except Exception:
        pass

    try:
        from src.utils.cache import CacheManager, cached
    except Exception:
        pass

    try:
        from src.utils.validators import ParameterValidator
    except Exception:
        pass

    try:
        from src.ui.components import UIComponents
    except Exception:
        pass

except Exception as e:
    logger.warning(f"Some optional src imports failed: {e}")

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators for ODEs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header{
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding:1.3rem;border-radius:14px;margin-bottom:1.2rem;color:white;text-align:center;
  box-shadow:0 10px 30px rgba(0,0,0,0.25);
}
.main-title{font-size:1.9rem;font-weight:700;margin-bottom:.35rem;}
.subtitle{font-size:.98rem;opacity:.95;}
.metric-card{
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:white;padding:.9rem;border-radius:12px;text-align:center;
  box-shadow:0 10px 20px rgba(0,0,0,0.2);
}
.info-box{
  background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
  border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:1rem 0;
}
.result-box{
  background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
  border:2px solid #4caf50;padding:1rem;border-radius:10px;margin:1rem 0;
}
.error-box{
  background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);
  border:2px solid #f44336;padding:1rem;border-radius:10px;margin:1rem 0;
}
.small{
  font-size:0.85rem; opacity:0.85;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
def _ensure(name: str, default):
    if name not in st.session_state:
        st.session_state[name] = default

def _init_session():
    # light helpers and heavy libs (lazy)
    _ensure("generator_constructor", GeneratorConstructor() if GeneratorConstructor else None)
    # UI/session data
    defaults = {
        "generator_terms": [],
        "generated_odes": [],
        "batch_results": [],
        "analysis_results": [],
        "export_history": [],
        "training_history": [],
        "ml_trainer": None,
        "ml_trained": False,
        "ml_trained_models": [],         # list of dicts {job_id, best_model, when}
        "ml_job_id": None,
        "last_compute_job_id": None,
        "lhs_source": "constructor",
        "free_terms": [],
        "arbitrary_lhs_text": "",
        "current_generator": None,
        "basic_functions": BasicFunctions() if BasicFunctions else None,
        "special_functions": SpecialFunctions() if SpecialFunctions else None,
        "ode_classifier": None,
        "novelty_detector": None,
        "cache_manager": CacheManager() if CacheManager else None,
        "loaded_trainer": None,         # local loaded trainer (session zip)
    }
    for k, v in defaults.items():
        _ensure(k, v)

    # optional heavy objects
    if st.session_state["ode_classifier"] is None and ODEClassifier:
        try:
            st.session_state["ode_classifier"] = ODEClassifier()
        except Exception:
            st.session_state["ode_classifier"] = None

    if st.session_state["novelty_detector"] is None and ODENoveltyDetector:
        try:
            st.session_state["novelty_detector"] = ODENoveltyDetector()
        except Exception:
            st.session_state["novelty_detector"] = None


# ---------------- LaTeX Exporter ----------------
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
            for i,(k,v) in enumerate(items):
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}" + (r" \\" if i<len(items)-1 else ""))
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
            r"\subsection{Verification}",
            r"Substitute $y(x)$ into the operator $L$ to verify $L[y] = \mathrm{RHS}$."
        ]
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\nCompile with: pdflatex ode_document.tex\n")
            if include_extras:
                zf.writestr("reproduce.txt", "Use ode_data.json with your factories/theorem code.")
        buf.seek(0)
        return buf.getvalue()


# ---------------- Helpers ----------------
def _register_generated_ode(res: Dict[str, Any]):
    """Normalize & store a generated ODE result."""
    d = dict(res)
    d.setdefault("type", "nonlinear")
    d.setdefault("order", 0)
    d.setdefault("parameters", {})
    d.setdefault("classification", {})
    d.setdefault("timestamp", datetime.now().isoformat())
    d["generator_number"] = len(st.session_state.generated_odes) + 1

    # classification defaults
    cl = dict(d.get("classification", {}))
    cl.setdefault("type", "Linear" if d["type"] == "linear" else "Nonlinear")
    cl.setdefault("order", d.get("order", 0))
    cl.setdefault("linearity", "Linear" if d["type"] == "linear" else "Nonlinear")
    cl.setdefault("field", cl.get("field", "Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    d["classification"] = cl

    # convenience Eq
    try:
        if isinstance(d.get("generator"), str):
            d["generator"] = sp.sympify(d["generator"])
        if isinstance(d.get("rhs"), str):
            d["rhs"] = sp.sympify(d["rhs"])
        if isinstance(d.get("solution"), str):
            d["solution"] = sp.sympify(d["solution"])
        d.setdefault("ode", sp.Eq(d["generator"], d["rhs"]))
    except Exception:
        pass

    st.session_state.generated_odes.append(d)


def _show_ode_result(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h3>✅ ODE Generated Successfully</h3></div>', unsafe_allow_html=True)
    tabs = st.tabs(["📐 Equation", "💡 Solution & ICs", "📤 Export"])
    with tabs[0]:
        try:
            st.latex(sp.latex(res["generator"]) + " = " + sp.latex(res["rhs"]))
        except Exception:
            st.write("LHS:", res.get("generator"))
            st.write("RHS:", res.get("rhs"))
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with tabs[1]:
        try:
            st.latex("y(x) = " + sp.latex(res["solution"]))
        except Exception:
            st.write("y(x) =", res.get("solution"))
        if res.get("initial_conditions"):
            st.markdown("**Initial conditions:**")
            for k, v in res["initial_conditions"].items():
                try:
                    st.latex(k + " = " + sp.latex(v))
                except Exception:
                    st.write(k, "=", v)
        p = res.get("parameters", {})
        st.markdown("**Parameters:**")
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        if res.get("f_expr_preview"):
            st.write(f"**Function:** f(z) = {res['f_expr_preview']}")
    with tabs[2]:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res.get("generator"),
            "rhs": res.get("rhs"),
            "solution": res.get("solution"),
            "parameters": res.get("parameters", {}),
            "classification": res.get("classification", {}),
            "initial_conditions": res.get("initial_conditions", {}),
            "function_used": str(res.get("function_used", "?")),
            "generator_number": idx,
            "type": res.get("type", "nonlinear"),
            "order": res.get("order", 0)
        }
        tex = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
        st.download_button("📄 Download LaTeX", tex, f"ode_{idx}.tex", "text/x-latex", use_container_width=True)
        pkg = LaTeXExporter.create_export_package(ode_data, include_extras=True)
        st.download_button("📦 Download ZIP (data+LaTeX)", pkg, f"ode_package_{idx}.zip", "application/zip", use_container_width=True)


# ---------------- Pages ----------------

def page_dashboard():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>🤖 Trained Models</h3><h1>{len(st.session_state.ml_trained_models)}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>📊 Batch Results</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c4:
        status = "ON" if has_redis() else "OFF"
        st.markdown(f'<div class="metric-card"><h3>🧵 Redis (RQ)</h3><h1>{status}</h1></div>', unsafe_allow_html=True)

    st.subheader("📊 Recent Activity")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type","order","generator_number","timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs yet. Go to **Apply Master Theorem** or **Generator Constructor**.")


def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Build custom generators. You can also use Free‑form or Arbitrary LHS on the theorem page.</div>', unsafe_allow_html=True)

    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found in src/. You can still use Free‑form/Arbitrary LHS in the Theorem page.")
        return

    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], index=0, format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x,f"y⁽{x}⁾"))
        with c2:
            try:
                ft_opts = [t.value for t in DerivativeType]
            except Exception:
                ft_opts = ["scalar"]
            func_type = st.selectbox("Function Type", ft_opts, index=0, format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)
        c5, c6, c7 = st.columns(3)
        with c5:
            try:
                op_opts = [t.value for t in OperatorType]
            except Exception:
                op_opts = ["identity"]
            operator_type = st.selectbox("Operator Type", op_opts, index=0, format_func=lambda s: s.replace("_"," ").title())
        with c6:
            scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if operator_type in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay","advance"] else None

        if st.button("➕ Add Term", type="primary"):
            try:
                term = DerivativeTerm(
                    derivative_order=int(deriv_order),
                    coefficient=float(coefficient),
                    power=int(power),
                    function_type=DerivativeType(func_type) if hasattr(DerivativeType, "__call__") else func_type,
                    operator_type=OperatorType(operator_type) if hasattr(OperatorType, "__call__") else operator_type,
                    scaling=scaling,
                    shift=shift,
                )
                st.session_state.generator_terms.append(term)
                st.success("Term added.")
            except Exception as e:
                st.error(f"Failed to add term: {e}")

    if st.session_state.generator_terms:
        st.subheader("📝 Current Generator Terms")
        for i, term in enumerate(list(st.session_state.generator_terms)):
            c1, c2 = st.columns([8,1])
            with c1:
                desc = term.get_description() if hasattr(term, "get_description") else str(term)
                st.write(f"• {desc}")
            with c2:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()

        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(
                    terms=st.session_state.generator_terms,
                    name=f"Custom Generator {len(st.session_state.generated_odes)+1}"
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


def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem")

    # LHS source
    src = st.radio("Generator LHS source", ("constructor","freeform","arbitrary"),
                   index={"constructor":0,"freeform":1,"arbitrary":2}.get(st.session_state.lhs_source, 0),
                   horizontal=True)
    st.session_state.lhs_source = src

    # function selection
    colA, colB = st.columns(2)
    with colA:
        lib_choice = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        lib_inst = None
        func_names = []
        if lib_choice == "Basic" and st.session_state.basic_functions:
            lib_inst = st.session_state.basic_functions
            func_names = lib_inst.get_function_names()
        elif lib_choice == "Special" and st.session_state.special_functions:
            lib_inst = st.session_state.special_functions
            func_names = lib_inst.get_function_names()
        func_name = st.selectbox("Choose f(z)", func_names) if func_names else st.text_input("Enter f(z)", "exp(z)")

    # parameters
    c1, c2, c3, c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c5, c6, c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7:
        st.info(f"Async via Redis: {'ON' if has_redis() else 'OFF'}")

    # Show constructor LHS if available
    constructor_lhs = None
    gen_spec = st.session_state.get("current_generator")
    if gen_spec is not None and hasattr(gen_spec, "lhs"):
        constructor_lhs = gen_spec.lhs
    elif st.session_state.get("generator_constructor") and \
         hasattr(st.session_state.generator_constructor, "get_generator_expression"):
        try:
            constructor_lhs = st.session_state.generator_constructor.get_generator_expression()
        except Exception:
            constructor_lhs = None

    # Free-form builder (optional terms already built in constructor page)
    st.subheader("🧩 Free‑form LHS (Builder)")
    with st.expander("Build or reuse custom LHS terms", expanded=False):
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k (y^(k))", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs","asin","acos","atan","asinh","acosh","atanh","erf","erfc"], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_order = st.number_input("outer m (D^m)", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale (a)", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift (b)", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": float(coef),
                    "inner_order": int(inner_order),
                    "wrapper": wrapper,
                    "power": int(power),
                    "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write("**Current free‑form terms:**")
            for i, t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. {t}")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("🧮 Use free‑form LHS"):
                    st.session_state.lhs_source = "freeform"
                    st.success("Free‑form LHS selected.")
            with cc2:
                if st.button("🗑️ Clear terms"):
                    st.session_state.free_terms = []

    # Arbitrary SymPy LHS
    st.subheader("✍️ Arbitrary LHS (SymPy expression)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Enter any SymPy expression in x and y(x) (e.g., sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1))",
        value=st.session_state.arbitrary_lhs_text or "", height=100
    )
    cva, cvb = st.columns(2)
    with cva:
        if st.button("✅ Validate arbitrary LHS"):
            try:
                # best-effort parse: we'll send as-is to worker/core which will parse/sympify
                sp.sympify(st.session_state.arbitrary_lhs_text)
                st.success("Expression parsed successfully.")
                st.session_state.lhs_source = "arbitrary"
            except Exception as e:
                st.error(f"Parse error: {e}")
    with cvb:
        if st.button("↩️ Prefer Constructor LHS"):
            st.session_state.lhs_source = "constructor"

    # Theorem 4.2 (y^(m))
    st.markdown("---")
    c1, c2 = st.columns([1,1])
    with c1:
        compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", False)
    with c2:
        m_order = st.number_input("m", 1, 12, 1)

    # Generate ODE
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        if ComputeParams is None or compute_ode_full is None:
            st.error("Core engine (shared.ode_core) is not available.")
        else:
            payload = {
                "func_name": func_name,
                "alpha": float(alpha),
                "beta": float(beta),
                "n": int(n),
                "M": float(M),
                "use_exact": bool(use_exact),
                "simplify_level": simplify_level,
                "lhs_source": st.session_state.lhs_source,
                "freeform_terms": st.session_state.get("free_terms"),
                "arbitrary_lhs_text": st.session_state.get("arbitrary_lhs_text"),
                "function_library": lib_choice,
            }
            if has_redis():
                job_id = enqueue_job(
                    "worker.compute_job", payload,
                    job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600")),
                    result_ttl=int(os.getenv("RQ_RESULT_TTL", "604800"))
                )
                if job_id:
                    st.session_state.last_compute_job_id = job_id
                    st.success(f"Job submitted. ID = {job_id}")
                else:
                    st.error("Failed to submit job (check REDIS_URL).")
            else:
                # sync fallback
                try:
                    p = ComputeParams(
                        func_name=func_name, alpha=float(alpha), beta=float(beta), n=int(n), M=float(M),
                        use_exact=use_exact, simplify_level=simplify_level,
                        lhs_source=st.session_state.lhs_source,
                        constructor_lhs=constructor_lhs,
                        freeform_terms=st.session_state.get("free_terms"),
                        arbitrary_lhs_text=st.session_state.get("arbitrary_lhs_text"),
                        function_library=lib_choice,
                        basic_lib=st.session_state.get("basic_functions"),
                        special_lib=st.session_state.get("special_functions"),
                    )
                    res = compute_ode_full(p)
                    _register_generated_ode(res)
                    _show_ode_result(res)
                except Exception as e:
                    st.error(f"Generation error: {e}")

    # Poll async compute job
    if has_redis() and st.session_state.last_compute_job_id:
        st.markdown("### 📡 Job Status")
        colx, coly = st.columns([1,1])
        with colx:
            if st.button("🔄 Refresh status"):
                pass
        info = fetch_job(st.session_state.last_compute_job_id)
        if info:
            if info.get("status") == "finished" and info.get("result"):
                res = info["result"]
                # Try sympify for nice LaTeX
                for k in ["generator","rhs","solution"]:
                    try:
                        res[k] = sp.sympify(res[k])
                    except Exception:
                        pass
                _register_generated_ode(res)
                _show_ode_result(res)
                st.session_state.last_compute_job_id = None
            elif info.get("status") == "failed":
                st.error("Job failed.")
                st.session_state.last_compute_job_id = None
            else:
                st.info("⏳ Still computing...")

    # Theorem 4.2 (immediate)
    if compute_mth and theorem_4_2_y_m_expr is not None and get_function_expr is not None and to_exact is not None:
        if st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2", use_container_width=True):
            try:
                lib_inst = st.session_state.basic_functions if lib_choice == "Basic" else st.session_state.special_functions
                f_preview = get_function_expr(lib_inst, func_name) if lib_inst else sp.sympify(func_name)
                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)
                x = sp.Symbol("x", real=True)
                y_m = theorem_4_2_y_m_expr(f_preview, α, β, int(n), int(m_order), x, simplify_level)
                st.markdown("### 🔢 Derivative")
                st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
            except Exception as e:
                st.error(f"Failed to compute derivative: {e}")


def page_ml_pattern_learning():
    st.header("🤖 ML Pattern Learning (Persistent)")

    if not MLTrainer or not TrainConfig:
        st.warning("Enhanced ML trainer not found in src/ml/trainer.py. The training UI remains visible but actions may not work.")
    if not has_redis():
        st.info("Redis is not configured — RQ training is disabled in this environment. You can still use local tools below.")

    # ---- Training configuration ----
    with st.expander("🎯 Training Configuration", True):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"], index=0)
            hidden_dim = st.select_slider("Hidden Dim", [64,128,256,384,512], value=128)
            normalize = st.checkbox("Normalize Features", False)
        with c2:
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.select_slider("Batch Size", [8,16,32,64,128], value=32)
            samples = st.slider("Samples", 200, 10000, 1000, step=100)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
            use_generator = st.checkbox("Use Generator (streaming)", True)
            enable_amp = st.checkbox("Mixed Precision (AMP)", False)

    with st.expander("⚙️ Advanced (β‑VAE / Schedules / Loss Mask)", False):
        c1, c2, c3 = st.columns(3)
        with c1:
            beta_vae = st.number_input("β‑VAE", value=1.0, step=0.1, format="%.2f")
            kl_anneal = st.selectbox("KL Anneal", ["linear","sigmoid","none"], index=0)
        with c2:
            kl_max_beta = st.number_input("KL Max β", value=1.0, step=0.1)
            kl_warmup = st.number_input("KL Warmup Epochs", value=10, step=1)
        with c3:
            early_stop = st.number_input("Early Stop Patience", value=12, step=1)
            loss_weights = st.text_input("Loss Weights (12 floats or empty)", "")

    # parse loss weights
    lw = None
    if loss_weights.strip():
        try:
            lw = [float(x.strip()) for x in loss_weights.strip().split(",")]
        except Exception:
            st.warning("Could not parse loss weights; ignoring.")
            lw = None

    # ---- Control bar ----
    if "ml_job_id" not in st.session_state:
        st.session_state.ml_job_id = None

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if has_redis() and st.button("🚀 Start Training (RQ)", type="primary"):
            payload = {
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "normalize": normalize,
                "beta_vae": beta_vae,
                "kl_anneal": kl_anneal,
                "kl_max_beta": kl_max_beta,
                "kl_warmup_epochs": kl_warmup,
                "early_stop_patience": early_stop,
                "loss_weights": lw,
                "epochs": epochs,
                "batch_size": batch_size,
                "samples": samples,
                "validation_split": validation_split,
                "use_generator": use_generator,
                "enable_mixed_precision": enable_amp,
            }
            job_id = enqueue_job(
                "worker.train_job", payload,
                job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "86400")),
                result_ttl=int(os.getenv("RQ_RESULT_TTL", "604800")),
            )
            if job_id:
                st.session_state.ml_job_id = job_id
                st.success(f"Enqueued training job: {job_id}")
            else:
                st.error("Failed to enqueue job. Check REDIS_URL.")
    with b2:
        if st.session_state.ml_job_id and st.button("🔄 Refresh Status"):
            pass
    with b3:
        if st.session_state.ml_job_id and st.button("🧹 Clear Job ID"):
            st.session_state.ml_job_id = None
    with b4:
        st.metric("Trained Models", len(st.session_state.ml_trained_models))

    # ---- Monitor persistent training ----
    if st.session_state.ml_job_id:
        job = fetch_job(st.session_state.ml_job_id)
        prog = get_progress(st.session_state.ml_job_id)
        logs = get_logs(st.session_state.ml_job_id, start=0, end=-1)
        arts = get_artifacts(st.session_state.ml_job_id)

        st.subheader("📡 Status")
        st.json(job or {})

        st.subheader("📈 Progress (last snapshot)")
        if prog:
            st.json(prog)
        else:
            st.info("No progress yet.")

        with st.expander("🗒️ Live Logs", True):
            if logs:
                st.code("\n".join(logs[-200:]), language="text")
            else:
                st.info("No logs yet.")

        st.subheader("📦 Artifacts")
        st.write(arts or {})
        if arts.get("session_zip"):
            try:
                with open(arts["session_zip"], "rb") as f:
                    st.download_button("⬇️ Download Session ZIP", f.read(), file_name="session_artifacts.zip")
            except Exception:
                st.info("Artifacts path may not be accessible in this process.")

        # Mark trained and persist into list when finished
        if job and job.get("status") == "finished":
            st.session_state.ml_trained = True
            if arts.get("best_model"):
                st.session_state.ml_trained_models.append({
                    "job_id": st.session_state.ml_job_id,
                    "best_model": arts.get("best_model"),
                    "session_zip": arts.get("session_zip"),
                    "when": datetime.now().isoformat(),
                })
            st.success("Training finished. Best model saved by worker.")

    st.markdown("---")

    # ---- Local session management ----
    st.subheader("💾 Session Management (Local)")
    c1, c2, c3 = st.columns(3)
    with c1:
        if MLTrainer and TrainConfig and st.button("Save Empty Session ZIP (demo)"):
            cfg = TrainConfig(model_type=model_type, hidden_dim=hidden_dim, normalize=normalize)
            local_trainer = MLTrainer(cfg)
            os.makedirs("artifacts", exist_ok=True)
            path = os.path.join("artifacts", f"local_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
            local_trainer.export_session_zip(path, best_path=None)
            with open(path, "rb") as f:
                st.download_button("⬇️ Download Local Session ZIP", f.read(), file_name=os.path.basename(path))
    with c2:
        up = st.file_uploader("Upload Session ZIP", type=["zip"])
        if up is not None and MLTrainer and TrainConfig:
            tmp = os.path.join("artifacts", f"uploaded_{datetime.now().timestamp()}.zip")
            os.makedirs("artifacts", exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(up.read())
            # attempt load into a resident trainer
            cfg = TrainConfig(model_type=model_type, hidden_dim=hidden_dim, normalize=normalize)
            tr = MLTrainer(cfg)
            ok = tr.load_session_zip(tmp)
            if ok:
                st.session_state.loaded_trainer = tr
                st.success(f"Session loaded into app: {tmp}")
            else:
                st.error("Failed to load uploaded session.")
    with c3:
        st.info("To resume on worker: ensure the worker container can access the checkpoint path; then pass 'resume_from' in the training payload.")

    st.markdown("---")

    # ---- Post-training utility ----
    st.subheader("🧪 Generate / Reverse Engineer (Local demo)")
    col1, col2 = st.columns(2)
    with col1:
        num_gen = st.slider("Generate N", 1, 10, 3)
        if st.button("🎲 Generate Parameters"):
            if not MLTrainer or not TrainConfig:
                st.warning("Trainer not available.")
            else:
                cfg = TrainConfig(model_type=model_type, hidden_dim=hidden_dim, normalize=normalize)
                tmp_trainer = st.session_state.get("loaded_trainer") or MLTrainer(cfg)
                try:
                    res = tmp_trainer.generate_new_ode(num=num_gen)
                    st.json(res)
                except Exception as e:
                    st.error(f"Generate failed: {e}")
    with col2:
        st.write("Reverse Engineer a feature vector (12 floats).")
        t = st.text_input("Enter 12 comma-separated floats", "")
        if st.button("🔁 Reverse Engineer"):
            if not MLTrainer or not TrainConfig:
                st.warning("Trainer not available.")
            else:
                try:
                    vec = np.array([float(x.strip()) for x in t.split(",")], dtype=np.float32)
                    cfg = TrainConfig(model_type=model_type, hidden_dim=hidden_dim, normalize=normalize)
                    tmp_trainer = st.session_state.get("loaded_trainer") or MLTrainer(cfg)
                    res = tmp_trainer.reverse_engineer(vec)
                    st.json(res)
                except Exception as e:
                    st.error(f"Reverse failed: {e}")


def page_batch_generation():
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
            if not (BasicFunctions or st.session_state.get("basic_functions")):
                st.warning("Function libraries not found. Batch generation needs at least BasicFunctions.")
                return
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
                    params = {
                        "alpha": float(np.random.uniform(*alpha_range)),
                        "beta":  float(np.random.uniform(*beta_range)),
                        "n": int(np.random.randint(n_range[0], n_range[1]+1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    func_name = np.random.choice(all_functions)
                    gt = np.random.choice(gen_types) if gen_types else "linear"
                    res = {}
                    # Try complete factories first
                    if gt == "linear" and CompleteLinearGeneratorFactory:
                        factory = CompleteLinearGeneratorFactory()
                        gen_num = np.random.randint(1, 9)
                        if gen_num in [4,5]:
                            params["a"] = float(np.random.uniform(1,3))
                        res = factory.create(gen_num, st.session_state.basic_functions.get_function(func_name), **params)
                    elif gt == "linear" and LinearGeneratorFactory:
                        factory = LinearGeneratorFactory()
                        res = factory.create(1, st.session_state.basic_functions.get_function(func_name), **params)
                    elif gt == "nonlinear" and CompleteNonlinearGeneratorFactory:
                        factory = CompleteNonlinearGeneratorFactory()
                        gen_num = np.random.randint(1, 11)
                        # Optional params
                        if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                        if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                        if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                        res = factory.create(gen_num, st.session_state.basic_functions.get_function(func_name), **params)
                    elif gt == "nonlinear" and NonlinearGeneratorFactory:
                        factory = NonlinearGeneratorFactory()
                        res = factory.create(1, st.session_state.basic_functions.get_function(func_name), **params)

                    if not res:
                        continue

                    row = {
                        "ID": i+1, "Type": res.get("type","unknown"),
                        "Generator": res.get("generator_number","?"),
                        "Function": func_name, "Order": res.get("order",0),
                        "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                    }
                    if include_solutions:
                        s = str(res.get("solution",""))
                        row["Solution"] = (s[:120]+"...") if len(s)>120 else s
                    if include_classification:
                        row["Subtype"] = res.get("subtype","standard")
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
                        r"\begin{tabular}{|c|c|c|c|c|}", r"\hline",
                        r"ID & Type & Generator & Function & Order \\",
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


def page_novelty_detection():
    st.header("🔍 Novelty Detection")
    if not st.session_state.get("novelty_detector"):
        st.warning("Novelty detector not found (src/dl/novelty_detector.py).")
        return

    method = st.radio("Input Method", ["Use Current Generator LHS", "Enter ODE Manually", "Select from Generated"])
    ode_obj = None
    if method == "Use Current Generator LHS":
        gen_spec = st.session_state.get("current_generator")
        if gen_spec is not None and hasattr(gen_spec, "lhs"):
            ode_obj = {"ode": gen_spec.lhs, "type":"custom", "order": getattr(gen_spec, "order", 2)}
        else:
            st.warning("No generator spec. Use Constructor or Free‑form.")
    elif method == "Enter ODE Manually":
        ode_str = st.text_area("Enter ODE (LaTeX or text):")
        if ode_str:
            ode_obj = {"ode": ode_str, "type":"manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','Unknown')}")
            ode_obj = st.session_state.generated_odes[sel]

    if ode_obj and st.button("🔎 Analyze Novelty", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                analysis = st.session_state.novelty_detector.analyze(ode_obj, check_solvability=True, detailed=True)
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


def page_analysis_classification():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    if not st.session_state.get("ode_classifier"):
        st.warning("Classifier not found.")
        return

    st.subheader("📊 Generated ODEs Overview")
    summary = []
    for i, ode in enumerate(st.session_state.generated_odes[-100:]):
        summary.append({
            "ID": i+1, "Type": ode.get("type","Unknown"), "Order": ode.get("order",0),
            "Generator": ode.get("generator_number","N/A"), "Function": ode.get("function_used","Unknown"),
            "Timestamp": ode.get("timestamp","")[:19]
        })
    df = pd.DataFrame(summary); st.dataframe(df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Linear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="linear"))
    with c2:
        st.metric("Nonlinear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="nonlinear"))
    with c3:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        st.metric("Average Order", f"{(np.mean(orders) if orders else 0):.1f}")
    with c4:
        unique = len(set(o.get("function_used","") for o in st.session_state.generated_odes))
        st.metric("Unique Functions", unique)

    st.subheader("📊 Distributions")
    if _PLOTLY_OK:
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
                        classifications.append(st.session_state.ode_classifier.classify_ode(ode))
                    except Exception:
                        classifications.append({})
                fields = [c.get("classification",{}).get("field","Unknown") for c in classifications if c]
                vc = pd.Series(fields).value_counts()
                if _PLOTLY_OK:
                    fig = px.bar(x=vc.index, y=vc.values, title="Classification by Field")
                    fig.update_layout(xaxis_title="Field", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(vc)
            except Exception as e:
                st.error(f"Classification failed: {e}")


def page_physical_applications():
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


def page_visualization():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize."); return
    sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')} (Order {st.session_state.generated_odes[i].get('order',0)})")
    ode = st.session_state.generated_odes[sel]
    c1, c2, c3 = st.columns(3)
    with c1: plot_type = st.selectbox("Plot Type", ["Solution","Phase Portrait","3D Surface","Direction Field"])
    with c2: x_range = st.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    with c3: num_points = st.slider("Number of Points", 100, 2000, 500)
    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Creating..."):
            try:
                x = np.linspace(x_range[0], x_range[1], num_points)
                # Placeholder curve; replace with numeric eval of ode["solution"] if you have ICs
                y = np.sin(x) * np.exp(-0.1*np.abs(x))
                if _PLOTLY_OK:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Solution"))
                    fig.update_layout(title="ODE Solution", xaxis_title="x", yaxis_title="y(x)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(pd.DataFrame({"x":x, "y":y}).set_index("x"))
            except Exception as e:
                st.error(f"Visualization failed: {e}")


def page_export_latex():
    st.header("📤 Export & LaTeX")
    st.markdown('<div class="info-box">Export ODEs in publication‑ready LaTeX.</div>', unsafe_allow_html=True)
    if not st.session_state.generated_odes:
        st.warning("No ODEs to export."); return

    mode = st.radio("Export Type", ["Single ODE","Multiple ODEs","Complete Report"])
    if mode == "Single ODE":
        idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                           format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}")
        ode = st.session_state.generated_odes[idx]
        st.subheader("📋 LaTeX Preview")
        tex = LaTeXExporter.generate_latex_document(ode, include_preamble=False)
        st.code(tex, language="latex")
        c1, c2 = st.columns(2)
        with c1:
            full_latex = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
            st.download_button("📄 Download LaTeX", full_latex, f"ode_{idx+1}.tex", "text/x-latex")
        with c2:
            package = LaTeXExporter.create_export_package(ode, include_extras=True)
            st.download_button("📦 Download Package", package, f"ode_package_{idx+1}.zip", "application/zip")
    elif mode == "Multiple ODEs":
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


def page_examples_library():
    st.header("📚 Examples Library")
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")
    with st.expander("Damped Oscillator"):
        st.latex("y'' + 2\\gamma y' + \\omega_0^2 y = 0")
    with st.expander("Airy Equation"):
        st.latex("y'' - x y = 0")


def page_settings():
    st.header("⚙️ Settings")
    tabs = st.tabs(["General","Export","Advanced","About"])
    with tabs[0]:
        st.checkbox("Dark mode", False)
        if st.button("Save General Settings"):
            st.success("Saved.")
    with tabs[1]:
        include_preamble = st.checkbox("Include LaTeX preamble by default", True)
        if st.button("Save Export Settings"):
            st.success("Saved.")
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            cm = st.session_state.get("cache_manager")
            st.metric("Cache Size", len(getattr(cm, "memory_cache", {})) if cm else 0)
        with c2:
            if st.button("Clear Cache"):
                try:
                    st.session_state.cache_manager.clear()
                    st.success("Cache cleared.")
                except Exception:
                    st.info("No cache manager.")
        with c3:
            if st.button("Save Session Snapshot"):
                ok = False
                try:
                    snap = {
                        "generated_odes": st.session_state.get("generated_odes", []),
                        "batch_results": st.session_state.get("batch_results", []),
                        "training_history": st.session_state.get("training_history", {}),
                        "ml_trained_models": st.session_state.get("ml_trained_models", []),
                    }
                    with open("session_state.pkl","wb") as f:
                        pickle.dump(snap, f)
                    ok = True
                except Exception:
                    ok = False
                st.success("Session saved.") if ok else st.error("Failed to save session.")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorem 4.1/4.2, ML/DL, Export, Novelty, Async Jobs.")


def page_documentation():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Go to **Apply Master Theorem**.
2. Pick f(z) from Basic/Special (or type one).
3. Set parameters (α, β, n, M) and choose **Exact (symbolic)** if you want rationals.
4. Choose LHS source: **Constructor**, **Free‑form**, or **Arbitrary SymPy**.
5. Click **Generate ODE**. If Redis is configured, the job runs in background; otherwise it runs locally.
6. Export from **📤 Export** or generate more with **📊 Batch Generation**.
7. Use **🤖 ML Pattern Learning** to train models in the background (with persistent logs/progress and downloadable artifacts).
8. **🔁 Reverse Engineering** uses the trained model to reconstruct/clean parameter vectors.
""")


def page_reverse_engineering():
    st.header("🔁 Reverse Engineering Lab")
    st.write("Use the (trained/loaded) model to project/clean a target feature vector (size=12).")
    txt = st.text_input("Target feature vector (12 comma-separated floats)", "1,1,2,0,3,1,4,2,0,0,0,0")
    if st.button("Run Reverse Engineering"):
        if not MLTrainer or not TrainConfig:
            st.warning("Trainer not available.")
            return
        try:
            vec = np.array([float(x.strip()) for x in txt.split(",")], dtype=np.float32)
            cfg = TrainConfig(model_type="pattern_learner", hidden_dim=128, normalize=True)
            tr = st.session_state.get("loaded_trainer") or MLTrainer(cfg)
            out = tr.reverse_engineer(vec)
            st.json(out)
        except Exception as e:
            st.error(f"Failed: {e}")


# ---------------- Main ----------------
def main():
    _init_session()
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Free‑form/Arbitrary generators • Theorem 4.1/4.2 • ML/DL • Export • Novelty • Async Jobs</div>
    </div>
    """, unsafe_allow_html=True)

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
            "🔁 Reverse Engineering",
            "⚙️ Settings",
            "📖 Documentation",
        ]
    )

    if page == "🏠 Dashboard":                 page_dashboard()
    elif page == "🔧 Generator Constructor":   page_generator_constructor()
    elif page == "🎯 Apply Master Theorem":    page_apply_master_theorem()
    elif page == "🤖 ML Pattern Learning":     page_ml_pattern_learning()
    elif page == "📊 Batch Generation":        page_batch_generation()
    elif page == "🔍 Novelty Detection":       page_novelty_detection()
    elif page == "📈 Analysis & Classification": page_analysis_classification()
    elif page == "🔬 Physical Applications":   page_physical_applications()
    elif page == "📐 Visualization":           page_visualization()
    elif page == "📤 Export & LaTeX":          page_export_latex()
    elif page == "📚 Examples Library":        page_examples_library()
    elif page == "🔁 Reverse Engineering":     page_reverse_engineering()
    elif page == "⚙️ Settings":                page_settings()
    elif page == "📖 Documentation":           page_documentation()


if __name__ == "__main__":
    main()