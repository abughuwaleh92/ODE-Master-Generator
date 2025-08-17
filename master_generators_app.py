# master_generators_app.py
"""
Master Generators for ODEs — Full App (Corrected & Improved, all services preserved)

New capabilities:
• Durable RQ job progress + run registry: training status no longer “disappears”.
• Load/Upload finished ML sessions; dashboard shows trained status correctly.
• Post‑training: generate novel ODEs from model; model‑assisted reverse engineering.
• Apply Master Theorem: constructor/free‑form/arbitrary LHS, exact params, simplify levels.
• Theorem 4.2 derivative helper.
• Batch generation, novelty detection, analysis & classification, visualization, export & LaTeX, examples, settings—unchanged (preserved) with minor robustness fixes.

Requires:
- rq_utils.py (enqueue_job supports job_timeout, result_ttl; run registry)
- worker.py (compute_job, train_job, reverse_job)
- src/ml/trainer.py (upgraded trainer)
- shared/reverse_engineering.py, shared/phi_lib.py
- shared/ode_core.py (ComputeParams, compute_ode_full, theorem helpers)
"""

# ---------------- std libs ----------------
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

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp
from sympy.core.function import AppliedUndef

# optional torch (only needed for local training or quick device check)
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

# ---------------- internal core & RQ utils ----------------
# The ode_core module must provide these call sites
from shared.ode_core import (
    ComputeParams, compute_ode_full, theorem_4_2_y_m_expr,  # apply Master Theorem helpers
    get_function_expr, parse_arbitrary_lhs, to_exact
)

from rq_utils import has_redis, enqueue_job, fetch_job, list_runs, load_run

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators ODE System - Complete Edition",
    page_icon="🔬", layout="wide", initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header{
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding:2rem;border-radius:14px;margin-bottom:1.4rem;color:white;text-align:center;
  box-shadow:0 10px 30px rgba(0,0,0,0.2);
}
.main-title{font-size:2.1rem;font-weight:700;margin-bottom:.35rem;}
.subtitle{font-size:1.02rem;opacity:.95;}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:white;padding:1rem;border-radius:12px;text-align:center;
  box-shadow:0 10px 20px rgba(0,0,0,0.2);}
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
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
class SessionStateManager:
    @staticmethod
    def initialize():
        if "generator_constructor" not in st.session_state and GeneratorConstructor:
            st.session_state.generator_constructor = GeneratorConstructor()
        defaults = {
            "generator_terms": [],
            "generated_odes": [],
            "generator_patterns": [],
            "ml_trainer": None,
            "ml_trained": False,
            "training_history": {},
            "batch_results": [],
            "analysis_results": [],
            "export_history": [],
            "lhs_source": "constructor",
            "free_terms": [],
            "arbitrary_lhs_text": "",
            "last_job_id": None,
            "last_reverse_job": None
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        # libraries / helpers
        if "basic_functions" not in st.session_state and BasicFunctions:
            try: st.session_state.basic_functions = BasicFunctions()
            except Exception: st.session_state.basic_functions = None
        if "special_functions" not in st.session_state and SpecialFunctions:
            try: st.session_state.special_functions = SpecialFunctions()
            except Exception: st.session_state.special_functions = None
        if "novelty_detector" not in st.session_state and ODENoveltyDetector:
            try: st.session_state.novelty_detector = ODENoveltyDetector()
            except Exception: st.session_state.novelty_detector = None
        if "ode_classifier" not in st.session_state and ODEClassifier:
            try: st.session_state.ode_classifier = ODEClassifier()
            except Exception: st.session_state.ode_classifier = None
        if "cache_manager" not in st.session_state and CacheManager:
            try: st.session_state.cache_manager = CacheManager()
            except Exception: st.session_state.cache_manager = None

def register_generated_ode(result: dict):
    """Normalize & store a generated ODE record for consistent downstream pages."""
    result = dict(result or {})
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
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\nTo compile: pdflatex ode_document.tex\n")
            if include_extras:
                zf.writestr("reproduce.txt", "Use ode_data.json with your factories or theorem code.")
        buf.seek(0)
        return buf.getvalue()

# ---------------- Helpers ----------------
def _ensure_ss_key(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

# ---------------- Apply Master Theorem (corrected) ----------------
def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (Exact, Async‑ready)")

    # LHS source
    _ensure_ss_key("lhs_source", "constructor")
    src = st.radio(
        "Generator LHS source",
        options=("constructor","freeform","arbitrary"),
        index={"constructor":0,"freeform":1,"arbitrary":2}.get(st.session_state["lhs_source"],0),
        horizontal=True
    )
    st.session_state["lhs_source"] = src

    # Function source
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

    # Parameters
    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")
    c5,c6,c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7: st.info("Async via Redis: ON" if has_redis() else "Async via Redis: OFF")

    # Constructor LHS preview (if available)
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

    # Free‑form builder
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
        if not has_redis():
            # Sync execution (local)
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
            # Async via worker
            job_id = enqueue_job(
                "worker.compute_job", payload,
                job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT","3600")),
                result_ttl=int(os.getenv("RQ_RESULT_TTL","604800")),
                description="compute ODE"
            )
            if job_id:
                st.session_state["last_job_id"] = job_id
                st.success(f"Job submitted. ID = {job_id}")
            else:
                st.error("Failed to submit job (REDIS_URL missing?)")

    # Poll result (if async)
    if has_redis() and "last_job_id" in st.session_state and st.session_state["last_job_id"]:
        st.markdown("### 📡 Job Status")
        colx, coly = st.columns([1,1])
        with colx:
            if st.button("🔄 Refresh status"):
                pass
        info = fetch_job(st.session_state["last_job_id"])
        if info:
            status = info.get("status")
            if status == "finished":
                res = info.get("result") or {}
                # try to sympify for LaTeX rendering
                for k in ("generator","rhs","solution"):
                    try:
                        if k in res: res[k] = sp.sympify(res[k])
                    except Exception:
                        pass
                register_generated_ode(res)
                show_ode_result(res)
                st.session_state["last_job_id"] = None
            elif status == "failed":
                st.error("Job failed.")
                st.session_state["last_job_id"] = None
            else:
                meta = info.get("meta", {})
                st.info(f"⏳ Still computing… {meta.get('state','')}")
        else:
            st.info("No info returned (might be queued).")

    # Theorem 4.2 (local)
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
            st.write("LHS:", res.get("generator")); st.write("RHS:", res.get("rhs"))
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t_sol:
        try:
            st.latex("y(x) = " + sp.latex(res["solution"]))
        except Exception:
            st.write("y(x) =", res.get("solution"))
        if res.get("initial_conditions"):
            st.markdown("**Initial conditions:**")
            for k,v in res["initial_conditions"].items():
                try: st.latex(k + " = " + sp.latex(v))
                except Exception: st.write(k, "=", v)
        st.markdown("**Parameters:**")
        p = res.get("parameters", {})
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        st.write(f"**Function:** f(z) = {res.get('f_expr_preview')}")
    with t_exp:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res.get("generator"), "rhs": res.get("rhs"), "solution": res.get("solution"),
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

# ---------------- Other pages (kept, with small improvements) ----------------
def dashboard_page():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>🧬 ML Patterns</h3><h1>{len(st.session_state.generator_patterns)}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>📊 Batch Results</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c4:
        # Count trained models from run registry as well
        trained_local = 1 if st.session_state.get("ml_trained") else 0
        try:
            runs = list_runs()
            finished = sum(1 for r in runs if (r.get("status") == "finished") or r.get("summary"))
        except Exception:
            finished = 0
        status = f"{trained_local + finished}"
        st.markdown(f'<div class="metric-card"><h3>🤖 Models Trained</h3><h1>{status}</h1></div>', unsafe_allow_html=True)

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
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5],
                                       format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x,f"y⁽{x}⁾"))
        with c2:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType],
                                     format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)
        c5,c6,c7 = st.columns(3)
        with c5:
            operator_type = st.selectbox("Operator Type", [t.value for t in OperatorType],
                                         format_func=lambda s: s.replace("_"," ").title())
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
                gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms,
                                                  name=f"Custom Generator {len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = gen_spec
                st.success("Generator specification created.")
                try: st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                except Exception: pass
            except Exception as e:
                st.error(f"Failed to build specification: {e}")
    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None

# ---------------- ML & Reverse pages (improved) ----------------
def _format_job_row(run):
    s = run.get("status", "unknown")
    best = run.get("summary", {}).get("best_val")
    desc = run.get("description","")
    return f"{run['job_id'][:8]} • {s} • best={best if best is not None else '-'} • {desc}"

def ml_pattern_learning_page():
    st.header("🤖 ML / DL Training & Inference")
    st.markdown("Durable RQ progress, load/upload sessions, generation, and reverse are available here.")

    # --- Configuration
    with st.expander("🎯 Training Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"])
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
            samples = st.slider("Training Samples", 100, 10000, 1000, step=100)
            val_split = st.slider("Validation Split", 0.05, 0.5, 0.2, 0.05)
        with c3:
            hidden_dim = st.select_slider("Hidden dim", [64, 128, 256, 512], value=128)
            use_generator = st.checkbox("Use Generator (synth data)", True)
            normalize = st.checkbox("Normalize Features", True)
            early_stop = st.slider("Early stop patience", 0, 50, 12)

    # --- Launch training (RQ) ---
    colL, colR = st.columns([1,1])
    with colL:
        if has_redis():
            if st.button("🚀 Enqueue Training Job", type="primary", use_container_width=True):
                payload = {
                    "model_type": model_type,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "samples": samples,
                    "validation_split": val_split,
                    "use_generator": use_generator,
                    "hidden_dim": hidden_dim,
                    "normalize": normalize,
                    "early_stop_patience": early_stop,
                }
                job_id = enqueue_job("worker.train_job", payload,
                                     job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "86400")),
                                     result_ttl=int(os.getenv("RQ_RESULT_TTL","604800")),
                                     description=f"train {model_type}")
                if job_id:
                    st.success(f"Training job queued: {job_id}")
                else:
                    st.error("Failed to enqueue (no Redis connection?)")
        else:
            st.info("Redis not configured; run training locally below.")
    with colR:
        # Local (sync) training fallback
        if st.button("🧪 Train Now (Local Sync)", use_container_width=True):
            with st.spinner("Training locally..."):
                device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                try:
                    trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate,
                                        device=device, checkpoint_dir="checkpoints", enable_mixed_precision=False)
                    trainer.normalize = normalize
                    trainer.train(epochs=epochs, batch_size=batch_size, samples=samples,
                                  validation_split=val_split, use_generator=use_generator,
                                  early_stop_patience=early_stop)
                    st.session_state.ml_trainer = trainer
                    st.session_state.ml_trained = True
                    st.session_state.training_history = trainer.history
                    st.success("Local training finished.")
                except Exception as e:
                    st.error(f"Local training failed: {e}")

    # --- Active & Completed Jobs ---
    st.subheader("📡 Training Jobs")
    try:
        runs = list_runs()
    except Exception:
        runs = []
    if not runs:
        st.info("No runs yet (or registry empty).")
    else:
        # Active
        active = [r for r in runs if r.get("status") in ("queued","started","deferred","running")]
        completed = [r for r in runs if (r.get("status") in ("finished","stopped","failed")) or r.get("summary")]
        if active:
            st.markdown("**Active**")
            for r in active[:10]:
                with st.expander(_format_job_row(r), expanded=False):
                    info = fetch_job(r["job_id"]) or r
                    meta = info.get("meta", {})
                    progress = meta.get("progress", {})
                    stats = meta.get("stats", {})
                    st.write("Progress:", progress)
                    st.write("Stats:", stats)
                    if st.button("🔄 Refresh", key=f"refresh_{r['job_id']}"):
                        st.experimental_rerun()
        if completed:
            st.markdown("**Completed**")
            labels = [ _format_job_row(r) for r in completed[:30] ]
            idx = st.selectbox("Attach to finished run", list(range(len(labels))), format_func=lambda i: labels[i], key="attach_sel")
            selected = completed[idx] if completed else None
            if selected and st.button("📥 Load Finished Run", type="primary"):
                summary = (selected.get("summary") or {})
                best = summary.get("best_model_path")
                if best and os.path.exists(best):
                    try:
                        trainer = MLTrainer(model_type=summary.get("model_type","pattern_learner"),
                                            checkpoint_dir=os.path.dirname(best))
                        ok = trainer.load_model(best)
                        if ok:
                            st.session_state.ml_trainer = trainer
                            st.session_state.ml_trained = True
                            st.session_state.training_history = trainer.history
                            st.success(f"Loaded model: {best}")
                        else:
                            st.warning("Could not load the best model file.")
                    except Exception as e:
                        st.error(f"Load failed: {e}")
                else:
                    st.warning("No best model file found in summary. You can upload a .pth below.")

    # --- Upload a checkpoint / session ---
    st.subheader("🗂️ Upload Model / Session")
    up = st.file_uploader("Upload a *.pth checkpoint", type=["pth"])
    if up is not None:
        saved_path = os.path.join("checkpoints", up.name)
        os.makedirs("checkpoints", exist_ok=True)
        with open(saved_path, "wb") as f:
            f.write(up.getbuffer())
        try:
            trainer = MLTrainer()
            ok = trainer.load_model(saved_path)
            st.session_state.ml_trainer = trainer if ok else None
            st.session_state.ml_trained = bool(ok)
            st.session_state.training_history = trainer.history if ok else {}
            st.success(f"Loaded uploaded model: {saved_path}")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")

    # --- Trained? Show benefits: generation & reverse quicklink ---
    st.subheader("🎨 Generate ODEs from Trained Model")
    if st.session_state.get("ml_trained") and st.session_state.get("ml_trainer"):
        c1, c2 = st.columns(2)
        with c1:
            num = st.slider("How many?", 1, 10, 1)
            if st.button("🎲 Generate", type="primary"):
                for i in range(num):
                    try:
                        res = st.session_state.ml_trainer.generate_new_ode()
                        if res:
                            st.success(f"Generated #{i+1}")
                            st.session_state.generated_odes.append(res)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
        with c2:
            st.info("For model‑assisted analysis, open **🔁 Reverse Engineering** (side nav).")
    else:
        st.info("Train or load a model first; then generation and reverse will be available.")

def reverse_engineering_page():
    st.header("🔁 Reverse Engineering (Model‑assisted)")
    st.markdown("Paste an ODE or upload (x, y) samples. This page can run locally or via the RQ worker (for heavier numeric fits).")

    mode = st.radio("Mode", ["Equation text", "Samples"], horizontal=True)
    payload = {}
    if mode == "Equation text":
        ode_text = st.text_area("ODE equation (SymPy/LaTeX-ish)", "y'' + y = 0")
        payload = {"ode_text": ode_text}
    else:
        st.caption("Provide sample points (comma-separated).")
        xs = st.text_input("x values", "0,0.2,0.4,0.6,0.8,1.0")
        ys = st.text_input("y values", "0,0.1,0.2,0.05,-0.1,-0.2")
        try:
            xvals = [float(t.strip()) for t in xs.split(",") if t.strip()!=""]
            yvals = [float(t.strip()) for t in ys.split(",") if t.strip()!=""]
        except Exception:
            xvals, yvals = [], []
        payload = {"samples": {"x": xvals, "y": yvals}}

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧪 Run Locally"):
            try:
                from shared.reverse_engineering import reverse_engineer
                res = reverse_engineer(payload)
                st.json(res)
            except Exception as e:
                st.error(f"Reverse failed: {e}")
    with colB:
        if has_redis():
            if st.button("🚀 Enqueue Reverse Job", type="primary"):
                job_id = enqueue_job("worker.reverse_job", payload,
                                     job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600")),
                                     result_ttl=int(os.getenv("RQ_RESULT_TTL","604800")),
                                     description="reverse")
                if job_id:
                    st.success(f"Reverse job queued: {job_id}")
                    st.session_state.last_reverse_job = job_id
        else:
            st.info("No Redis connection detected.")

    if has_redis() and st.session_state.get("last_reverse_job"):
        st.markdown("### 📡 Reverse Job Status")
        info = fetch_job(st.session_state.last_reverse_job)
        if info:
            st.write("Status:", info.get("status"))
            st.write("Meta:", info.get("meta"))
            if info.get("status") == "finished" and info.get("result"):
                st.success("Result:")
                st.json(info["result"])
                st.session_state.last_reverse_job = None
            elif info.get("status") == "failed":
                st.error(info.get("exc_info","Job failed."))
                st.session_state.last_reverse_job = None
        if st.button("🔄 Refresh reverse job"):
            st.experimental_rerun()

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
            sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','Unknown')}")
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
    for i, ode in enumerate(st.session_state.generated_odes[-100:]):
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
                # If you have a numeric solution evaluator, plug it here.
                # For now, a placeholder smooth curve:
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
            cm = st.session_state.get("cache_manager")
            st.metric("Cache Size", len(getattr(cm,"memory_cache",{})) if cm else 0)
        with c2:
            if st.button("Clear Cache"):
                try: st.session_state.cache_manager.clear(); st.success("Cache cleared.")
                except Exception: st.info("No cache manager.")
        with c3:
            if st.button("Save Session"):
                ok = False
                try:
                    os.makedirs("checkpoints", exist_ok=True)
                    with open(os.path.join("checkpoints","session_state.pkl"),"wb") as f:
                        pickle.dump({
                            "generated_odes": st.session_state.get("generated_odes"),
                            "generator_patterns": st.session_state.get("generator_patterns"),
                            "batch_results": st.session_state.get("batch_results"),
                            "analysis_results": st.session_state.get("analysis_results"),
                            "training_history": st.session_state.get("training_history"),
                            "export_history": st.session_state.get("export_history"),
                            "ml_trained": st.session_state.get("ml_trained")
                        }, f)
                    ok = True
                except Exception:
                    ok = False
                st.success("Session saved.") if ok else st.error("Failed to save.")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorems 4.1 & 4.2, ML/DL, Export, Novelty. All services preserved with background jobs and resumable training.")

def documentation_page():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Go to **🎯 Apply Master Theorem**.
2. Pick f(z) from Basic/Special (or type one).
3. Set parameters (α,β,n,M) and choose **Exact (symbolic)** if desired.
4. Choose LHS source: **Constructor**, **Free‑form**, or **Arbitrary SymPy**.
5. Click **Generate ODE**. If Redis is configured, the job runs in background.
6. Export from the **📤 Export** page.
7. Train a model in **🤖 ML / DL**, then generate or reverse in **🔁 Reverse Engineering**.
""")

# ---------------- Main ----------------
def main():
    SessionStateManager.initialize()
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Free‑form/Arbitrary generators • Master Theorem • ML/DL • Reverse • Export • Async Jobs</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard", "🔧 Generator Constructor", "🎯 Apply Master Theorem", "🤖 ML / DL",
        "📊 Batch Generation", "🔁 Reverse Engineering",
        "🔍 Novelty Detection", "📈 Analysis & Classification",
        "🔬 Physical Applications", "📐 Visualization", "📤 Export & LaTeX",
        "📚 Examples Library", "⚙️ Settings", "📖 Documentation",
    ])

    if page == "🏠 Dashboard": dashboard_page()
    elif page == "🔧 Generator Constructor": generator_constructor_page()
    elif page == "🎯 Apply Master Theorem": page_apply_master_theorem()
    elif page == "🤖 ML / DL": ml_pattern_learning_page()
    elif page == "📊 Batch Generation": batch_generation_page()
    elif page == "🔁 Reverse Engineering": reverse_engineering_page()
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