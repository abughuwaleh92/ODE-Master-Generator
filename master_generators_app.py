# master_generators_app.py
"""
Master Generators for ODEs — Complete App (Corrected + Async-ready, RQ‑persistent)

What’s new vs previous version:
• RQ queue is consistent (default: ode_jobs) across web + worker (env RQ_QUEUE can override)
• Persistent job meta for ODE compute + training ⇒ status never “disappears”
• Jobs & Workers panel shows queues/workers via rq_inspect()
• Training can run in background (RQ) with live epoch/loss; model marked as trained on finish
• Save/Load/Upload model files (upload solves separate-filesystem issue between web and worker)
• All pages/services preserved and untouched behaviorally unless noted as robustness improvements
"""

# ---------------- std libs ----------------
import os, sys, io, json, time, base64, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import types

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp
from sympy.core.function import AppliedUndef

# optional torch
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

# ---------------- RQ helpers (fixed) ----------------
from rq_utils import has_redis, enqueue_job, fetch_job, rq_inspect
RQ_QUEUE = os.environ.get("RQ_QUEUE", "ode_jobs")  # must match worker

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
from shared.ode_core import (
    ComputeParams, compute_ode_full, theorem_4_1_solution_expr, theorem_4_2_y_m_expr,
    get_function_expr, build_freeform_lhs, parse_arbitrary_lhs, to_exact, simplify_expr
)

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
.small-muted{opacity:.7;font-size:.9rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.5rem;background:#eee;margin-left:.25rem}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
class SessionStateManager:
    @staticmethod
    def initialize():
        if "generator_constructor" not in st.session_state and GeneratorConstructor:
            st.session_state.generator_constructor = GeneratorConstructor()
        for key, default in [
            ("generator_terms", []), ("generated_odes", []), ("generator_patterns", []),
            ("ml_trainer", None), ("ml_trained", False), ("training_history", {}),
            ("batch_results", []), ("analysis_results", []), ("export_history", []),
            ("lhs_source", "constructor"), ("freeform_gen_spec", None),
            ("free_terms", []), ("arbitrary_lhs_text", ""),
            ("last_job_id", None), ("last_train_job_id", None),
            ("trained_model_path", None),
        ]:
            if key not in st.session_state: st.session_state[key] = default
        # optional heavy objects
        if "basic_functions" not in st.session_state and BasicFunctions:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions:
            st.session_state.special_functions = SpecialFunctions()
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
    result = dict(result)
    result.setdefault("type", "nonlinear")
    result.setdefault("order", 0)
    result.setdefault("function_used", "unknown")
    result.setdefault("parameters", {})
    result.setdefault("classification", {})
    result.setdefault("timestamp", datetime.now().isoformat())
    result["generator_number"] = len(st.session_state.generated_odes) + 1
    cl = dict(result.get("classification", {}))
    cl.setdefault("type", "Linear" if result["type"]=="linear" else "Nonlinear")
    cl.setdefault("order", result["order"])
    cl.setdefault("field", cl.get("field", "Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    cl.setdefault("linearity", "Linear" if result["type"]=="linear" else "Nonlinear")
    result["classification"] = cl
    # build Eq object for convenience
    try:
        result.setdefault("ode", sp.Eq(result["generator"], result["rhs"]))
    except Exception:
        pass
    st.session_state.generated_odes.append(result)

# ---------------- LaTeX Exporter ----------------
class LaTeXExporter:
    @staticmethod
    def sympy_to_latex(expr) -> str:
        if expr is None: return ""
        try:
            if isinstance(expr, str):
                try: expr = sp.sympify(expr)
                except Exception: return expr
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
            r"\subsection{Solution Verification}",
            r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$."
        ]
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\nTo compile: pdflatex ode_document.tex\n")
            if include_extras:
                zf.writestr("reproduce.txt", "Use ode_data.json with your factories or theorem code.")
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

# ---------------- helpers ----------------
def _ensure_ss_key(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

def jobs_workers_panel():
    """Small panel that shows queues/workers; confirms queue in use."""
    st.subheader("🛰️ Jobs & Workers")
    st.caption(f"Using queue: **{RQ_QUEUE}**")
    if not has_redis():
        st.info("Redis not configured (REDIS_URL). Background jobs disabled.")
        return
    info = rq_inspect()
    if "error" in info:
        st.error(f"RQ inspect error: {info['error']}")
        return
    qdf = pd.DataFrame(info.get("queues", []))
    wdf = pd.DataFrame(info.get("workers", []))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Queues**")
        if not qdf.empty: st.dataframe(qdf, use_container_width=True)
        else: st.write("No queues discovered.")
    with c2:
        st.markdown("**Workers**")
        if not wdf.empty: st.dataframe(wdf, use_container_width=True)
        else: st.write("No workers discovered.")

# ---------------- Apply Master Theorem (fixed) ----------------
def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (Exact, Async-ready)")
    jobs_workers_panel()

    # choose source of LHS
    _ensure_ss_key("lhs_source", "constructor")
    src = st.radio(
        "Generator LHS source",
        options=("constructor","freeform","arbitrary"),
        index={"constructor":0,"freeform":1,"arbitrary":2}[st.session_state["lhs_source"]] \
              if st.session_state["lhs_source"] in ("constructor","freeform","arbitrary") else 0,
        horizontal=True
    )
    st.session_state["lhs_source"] = src

    # function source
    colA, colB = st.columns([1,1])
    with colA:
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        basic_lib = st.session_state.get("basic_functions")
        special_lib = st.session_state.get("special_functions")
        if lib == "Basic" and basic_lib:
            func_names = basic_lib.get_function_names()
            source_lib = basic_lib
        elif lib == "Special" and special_lib:
            func_names = special_lib.get_function_names()
            source_lib = special_lib
        else:
            func_names = []
            source_lib = None
        func_name = st.selectbox("Choose f(z)", func_names) if func_names else st.text_input("Enter f(z)", "exp(z)")

    # parameters
    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c5,c6,c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7:
        async_mode = has_redis()
        st.info("Async via Redis: ON" if async_mode else "Async via Redis: OFF")

    # show constructor LHS if present
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

    # Free-form builder
    st.subheader("🧩 Free‑form LHS (Builder)")
    with st.expander("Build custom LHS terms", expanded=False):
        if "free_terms" not in st.session_state: st.session_state.free_terms = []
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k (y^(k))", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)",
                          ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs",
                           "asin","acos","atan","asinh","acosh","atanh","erf","erfc"], index=0)
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
            cc1,cc2 = st.columns(2)
            with cc1:
                if st.button("🧮 Use free‑form LHS"):
                    st.session_state["lhs_source"] = "freeform"
                    st.success("Free-form LHS selected.")
            with cc2:
                if st.button("🗑️ Clear terms"):
                    st.session_state.free_terms = []

    # Arbitrary SymPy expression editor
    st.subheader("✍️ Arbitrary LHS (SymPy expression)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Enter any SymPy expression in x and y(x) (e.g., sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1))",
        value=st.session_state.arbitrary_lhs_text or "",
        height=100
    )
    cc1,cc2 = st.columns(2)
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

    # Theorem 4.2
    st.markdown("---")
    colm1, colm2 = st.columns([1,1])
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
            "function_library": lib,
        }
        # constructor LHS is session-local; for sync path only
        if not has_redis():
            try:
                basic_lib = st.session_state.get("basic_functions")
                special_lib = st.session_state.get("special_functions")
                p = ComputeParams(
                    func_name=func_name, alpha=alpha, beta=beta, n=int(n), M=M,
                    use_exact=use_exact, simplify_level=simplify_level,
                    lhs_source=st.session_state["lhs_source"],
                    constructor_lhs=constructor_lhs,
                    freeform_terms=st.session_state.get("free_terms"),
                    arbitrary_lhs_text=st.session_state.get("arbitrary_lhs_text"),
                    function_library=lib, basic_lib=basic_lib, special_lib=special_lib
                )
                res = compute_ode_full(p)
                register_generated_ode(res)
                show_ode_result(res)
            except Exception as e:
                st.error(f"Generation error: {e}")
        else:
            # enqueue background job on the SAME queue as worker
            job_id = enqueue_job(
                "worker.compute_job",
                payload,
                queue=RQ_QUEUE,
                description="ODE compute",
                job_timeout=600,
                result_ttl=86400,
                failure_ttl=86400,
            )
            if job_id:
                st.session_state["last_job_id"] = job_id
                st.success(f"Job submitted. ID = {job_id}")
            else:
                st.error("Failed to submit job (check REDIS_URL/RQ_QUEUE).")

    # poll result (if async)
    if has_redis() and st.session_state.get("last_job_id"):
        st.markdown("### 📡 ODE Job Status")
        snap = fetch_job(st.session_state["last_job_id"])
        if not snap:
            st.info("Job not found. It may have expired.")
        else:
            st.write(f"Status: **{snap['status']}** | Desc: **{snap.get('description','')}** | Origin queue: `{snap.get('origin')}`")
            meta = snap.get("meta") or {}
            stage = meta.get("stage", "")
            if stage: st.write(f"Stage: **{stage}**")
            if snap["status"] == "finished" and snap.get("result"):
                res = snap["result"]
                # re-cast SymPy (optional)
                try:
                    res["generator"] = sp.sympify(res["generator"])
                    res["rhs"]       = sp.sympify(res["rhs"])
                    res["solution"]  = sp.sympify(res["solution"])
                except Exception:
                    pass
                register_generated_ode(res)
                show_ode_result(res)
                st.session_state["last_job_id"] = None
            elif snap["status"] == "failed":
                st.error("Job failed.")
                if snap.get("exc_info"): st.code(snap["exc_info"])
                st.session_state["last_job_id"] = None

    # Theorem 4.2 (immediate; can be slow → recommend async if heavy)
    if compute_mth and st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2", use_container_width=True):
        try:
            basic_lib = st.session_state.get("basic_functions")
            special_lib = st.session_state.get("special_functions")
            f_expr_preview = get_function_expr(basic_lib if lib=="Basic" else special_lib, func_name)
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta)  if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(f_expr_preview, α, β, int(n), int(m_order), x, simplify_level)
            st.markdown("### 🔢 Derivative")
            st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
        except Exception as e:
            st.error(f"Failed to compute y^{m_order}(x): {e}")

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
            for k,v in res["initial_conditions"].items():
                try:
                    st.latex(k + " = " + sp.latex(v))
                except Exception:
                    st.write(k, "=", v)
        st.markdown("**Parameters:**")
        p = res.get("parameters", {})
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        st.write(f"**Function:** f(z) = {res.get('f_expr_preview')}")
    with t_exp:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res["generator"], "rhs": res["rhs"], "solution": res["solution"],
            "parameters": res.get("parameters", {}),
            "classification": {
                "type": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "order": res.get("order", 0),
                "linearity": "Linear" if res.get("type")=="linear" else "Nonlinear",
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

# ---------------- Other pages (unchanged behaviors, minor robustness) ----------------
def dashboard_page():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>🧬 ML Patterns</h3><h1>{len(st.session_state.generator_patterns)}</h1></div>', unsafe_allow_html=True)
    with c3:
        # treat 'trained' as 1 if either trainer exists or an uploaded model is present
        trained_flag = bool(st.session_state.get("ml_trained")) or bool(st.session_state.get("trained_model_path"))
        st.markdown(f'<div class="metric-card"><h3>🤖 ML Model</h3><p style="font-size: 1.6rem;">{"✅ Trained" if trained_flag else "⏳ Not Trained"}</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>📊 Batch Results</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)

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
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x,f"y⁽{x}⁾"))
        with c2:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType], format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)
        c5,c6,c7 = st.columns(3)
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
            c1,c2 = st.columns([8,1])
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

    # status summary
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Patterns", len(st.session_state.generator_patterns))
    with c2: st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c3: st.metric("Trained?", "Yes" if st.session_state.get("ml_trained") else "No")
    with c4: st.metric("History epochs", st.session_state.get("training_history", {}).get("epochs", 0))

    # config
    model_type = st.selectbox("Select ML Model", ["pattern_learner","vae","transformer"], format_func=lambda s: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[s])

    with st.expander("🎯 Training Configuration", True):
        c1,c2,c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
            samples = st.slider("Training Samples", 100, 5000, 1000)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            use_gpu = st.checkbox("Use GPU if available", True)
        c4,c5 = st.columns(2)
        with c4:
            use_generator = st.checkbox("Use synthetic generator (memory-efficient)", True)
        with c5:
            run_in_background = st.checkbox("Run training in background (RQ)", value=has_redis())

    # Use both individually generated ODEs and batch results as training data (optional)
    st.checkbox("Include Batch Results as Training Data", True, key="use_batch_for_training")

    # Controls
    train_col, gen_col = st.columns([1,1])
    with train_col:
        if st.button("🚀 Start Training", type="primary"):
            try:
                device = "cuda" if use_gpu and (torch and torch.cuda.is_available()) else "cpu"
                payload = {
                    "model_type": model_type,
                    "learning_rate": float(learning_rate),
                    "device": device,
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "samples": int(samples),
                    "validation_split": float(validation_split),
                    "use_generator": bool(use_generator),
                }
                if run_in_background and has_redis():
                    job_id = enqueue_job(
                        "worker.train_job",
                        payload,
                        queue=RQ_QUEUE,
                        description="training",
                        job_timeout=86400,   # up to 24h
                        result_ttl=604800,   # keep a week
                        failure_ttl=604800,
                    )
                    st.session_state["last_train_job_id"] = job_id
                    st.success(f"Training job submitted. ID = {job_id}")
                else:
                    # sync training
                    trainer = MLTrainer(model_type=model_type, learning_rate=float(learning_rate), device=device)
                    st.session_state.ml_trainer = trainer
                    prog = st.progress(0); status = st.empty()

                    def progress_callback(epoch, total_epochs):
                        prog.progress(min(1.0, epoch/total_epochs))
                        status.text(f"Epoch {epoch}/{total_epochs}")
                    trainer.train(
                        epochs=epochs, batch_size=batch_size, samples=samples,
                        validation_split=validation_split, use_generator=use_generator,
                        progress_callback=progress_callback, save_best=True
                    )
                    st.session_state.ml_trained = True
                    st.session_state.training_history = getattr(trainer, "history", {})
                    st.success("Model trained (sync).")
            except Exception as e:
                st.error(f"Training failed: {e}")

    with gen_col:
        if st.button("🎨 Generate with Trained Model"):
            try:
                # prefer using a trainer already in memory
                trainer = st.session_state.get("ml_trainer")
                if not trainer:
                    # attempt to load from uploaded/saved path
                    path = st.session_state.get("trained_model_path")
                    if path:
                        trainer = MLTrainer(model_type=model_type, device="cpu")
                        if trainer.load_model(path):
                            st.session_state.ml_trainer = trainer
                        else:
                            trainer = None
                if not trainer:
                    st.warning("No trained model loaded yet. Train or upload a model.")
                else:
                    res = trainer.generate_new_ode()
                    if res:
                        st.success("Generated ODE from model.")
                        st.session_state.generated_odes.append(res)
                        with st.expander("Preview"):
                            if "ode" in res:
                                try: st.latex(sp.latex(res["ode"]))
                                except Exception: st.code(str(res["ode"]))
                            for k in ["type","order","function_used","description"]:
                                if k in res: st.write(f"**{k}:** {res[k]}")
            except Exception as e:
                st.error(f"Generation failed: {e}")

    # Training job status (background)
    if has_redis() and st.session_state.get("last_train_job_id"):
        st.markdown("### 📡 Training Job Status")
        t = fetch_job(st.session_state["last_train_job_id"])
        if t:
            meta = t.get("meta") or {}
            st.write(f"Status: **{t['status']}** | Stage: **{meta.get('stage','-')}** | Queue: `{t.get('origin')}`")
            ep, tot = meta.get("epoch","-"), meta.get("total_epochs","-")
            st.write(f"Epoch **{ep} / {tot}**")
            st.write(f"Train Loss: **{meta.get('train_loss','-')}**")
            st.write(f"Val Loss: **{meta.get('val_loss','-')}**")
            if t["status"] == "finished" and t.get("result"):
                res = t["result"]
                st.session_state["ml_trained"] = bool(res.get("trained"))
                st.session_state["training_history"] = res.get("history", {})
                # NOTE: model_path is on worker FS; upload to use in web
                st.session_state["trained_model_path_info"] = res.get("model_path")
                st.success(f"Training finished. (Worker saved: {res.get('model_path')})")
                st.session_state["last_train_job_id"] = None
            elif t["status"] == "failed":
                st.error("Training job failed.")
                if t.get("exc_info"): st.code(t["exc_info"])
                st.session_state["last_train_job_id"] = None

    # Save / Load / Upload model section
    st.markdown("---")
    st.subheader("💾 Model Management")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Save Current Model to checkpoints/"):
            trainer = st.session_state.get("ml_trainer")
            if not trainer:
                st.warning("No trainer in memory.")
            else:
                os.makedirs("checkpoints", exist_ok=True)
                path = os.path.join("checkpoints", f"{trainer.model_type}_manual_save_{int(time.time())}.pth")
                trainer.save_model(path)
                st.session_state["trained_model_path"] = path
                st.success(f"Saved to {path}")
    with c2:
        path_in = st.text_input("Load model from path (web container)", st.session_state.get("trained_model_path") or "")
        if st.button("Load Model from Path"):
            if not path_in or not os.path.exists(path_in):
                st.error("Path does not exist in web container.")
            else:
                trainer = MLTrainer(model_type=model_type, device="cpu")
                if trainer.load_model(path_in):
                    st.session_state.ml_trainer = trainer
                    st.session_state.ml_trained = True
                    st.session_state.trained_model_path = path_in
                    st.success(f"Loaded model from {path_in}")
                else:
                    st.error("Failed to load model.")
    with c3:
        up = st.file_uploader("Upload .pth model", type=["pth","pt"])
        if up is not None:
            os.makedirs("checkpoints/uploads", exist_ok=True)
            save_path = os.path.join("checkpoints/uploads", up.name)
            with open(save_path, "wb") as f:
                f.write(up.read())
            trainer = MLTrainer(model_type=model_type, device="cpu")
            ok = trainer.load_model(save_path)
            if ok:
                st.session_state.ml_trainer = trainer
                st.session_state.ml_trained = True
                st.session_state.trained_model_path = save_path
                st.success(f"Uploaded & loaded model: {save_path}")
            else:
                st.error("Uploaded file could not be loaded as a model.")

    # Plot training history if available
    hist = st.session_state.get("training_history", {})
    if hist.get("train_loss"):
        st.subheader("📈 Training History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1,len(hist["train_loss"])+1)), y=hist["train_loss"], mode="lines", name="Training Loss"))
        if hist.get("val_loss"):
            fig.add_trace(go.Scatter(x=list(range(1,len(hist["val_loss"])+1)), y=hist["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs with your factories.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
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
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📊 Download CSV", csv, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            with c2:
                js = json.dumps(batch_results, indent=2, default=str).encode("utf-8")
                st.download_button("📄 Download JSON", js, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")
            with c3:
                if export_format in ["LaTeX","All"]:
                    latex = "\\n".join([
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
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Linear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="linear"))
    with c2: st.metric("Nonlinear ODEs", sum(1 for o in st.session_state.generated_odes if o.get("type")=="nonlinear"))
    with c3:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        st.metric("Average Order", f"{(np.mean(orders) if orders else 0):.1f}")
    with c4:
        unique = len(set(o.get("function_used","") for o in st.session_state.generated_odes))
        st.metric("Unique Functions", unique)

    st.subheader("📊 Distributions")
    c1,c2 = st.columns(2)
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
    ode = st.session_state.generated_odes[sel]
    c1,c2,c3 = st.columns(3)
    with c1: plot_type = st.selectbox("Plot Type", ["Solution","Phase Portrait","3D Surface","Direction Field"])
    with c2: x_range = st.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    with c3: num_points = st.slider("Number of Points", 100, 2000, 500)
    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Creating..."):
            try:
                x = np.linspace(x_range[0], x_range[1], num_points)
                # placeholder; you can plug numeric eval of res["solution"] if desired
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
        c1,c2 = st.columns(2)
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
            for count,i in enumerate(sel,1):
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
        c1,c2,c3 = st.columns(3)
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
                                      "analysis_results","training_history","export_history",
                                      "trained_model_path","ml_trained"]}, f)
                    ok = True
                except Exception: ok = False
                st.success("Session saved.") if ok else st.error("Failed to save.")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorems 4.1 & 4.2, ML/DL, Export, Novelty. "
                    f"RQ Queue in use: **{RQ_QUEUE}**. Make sure worker listens to the same queue.")

def documentation_page():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Go to **Apply Master Theorem**.
2. Pick f(z) from Basic/Special (or type one).
3. Set parameters (α,β,n,M) and choose **Exact (symbolic)** if you want rationals.
4. Choose LHS source: **Constructor**, **Free‑form**, or **Arbitrary SymPy**.
5. Click **Generate ODE**. If Redis is configured, the job runs in background.
6. Export from the **📤 Export** tab or the **Export & LaTeX** page.
7. Compute **y^(m)(x)** via **Theorem 4.2** when needed.

**Training in Background (RQ)**
• Ensure both web and worker services use the same queue name (env `RQ_QUEUE`, default `ode_jobs`).  
• Web enqueues to that queue; worker must listen to it: `rq worker -u $REDIS_URL $RQ_QUEUE`.  
• Status is persisted via `job.meta`: stage, epoch, train/val loss — visible until TTL expiry.
""")

# ---------------- Main ----------------
def main():
    SessionStateManager.initialize()
    st.markdown(f"""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">By Mohammad Abu Ghuwaleh • Free‑form/Arbitrary generators • ML/DL • Export • Novelty • Async Jobs • Queue: <span class="badge">{RQ_QUEUE}</span></div>
    </div>
    """, unsafe_allow_html=True)
    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard", "🔧 Generator Constructor", "🎯 Apply Master Theorem", "🤖 ML Pattern Learning",
        "📊 Batch Generation", "🔍 Novelty Detection", "📈 Analysis & Classification",
        "🔬 Physical Applications", "📐 Visualization", "📤 Export & LaTeX", "📚 Examples Library",
        "⚙️ Settings", "📖 Documentation",
    ])
    if page == "🏠 Dashboard": dashboard_page()
    elif page == "🔧 Generator Constructor": generator_constructor_page()
    elif page == "🎯 Apply Master Theorem": page_apply_master_theorem()
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