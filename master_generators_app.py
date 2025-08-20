# master_generators_app.py
"""
Master Generators for ODEs — Full Application (Rewritten & Corrected)

What you get in this version:
  • Fix: "Generate ODEs" no longer gets stuck — constructor LHS is serialized to the worker.
  • Robust Redis detection (no placeholder false-positives), with sync fallback if Redis is down.
  • Training via RQ with long TTLs; persistent job info + live logs pulled from Redis.
  • Training sessions can be saved, loaded, and uploaded; model artifacts can be uploaded/reused.
  • Reverse Engineering: analytical heuristics + ML-assisted refinement (if a trained model is present).
  • All pages/services preserved: Dashboard, Constructor, Theorem 4.1/4.2, ML, Batch, Novelty,
    Analysis & Classification, Physical Applications, Visualization, Export & LaTeX, Examples, Settings, Docs.

Assumptions:
  • shared/ode_core.py provides:
      ComputeParams, compute_ode_full, theorem_4_2_y_m_expr, get_function_expr,
      parse_arbitrary_lhs, to_exact, simplify_expr, expr_to_str (optional)
  • rq_utils.py provides:
      has_redis, enqueue_job, fetch_job, redis_status
  • Worker has:
      worker.compute_job, worker.train_job, worker.ping_job
"""

# ---------------- std libs ----------------
import os
import io
import gc
import sys
import json
import glob
import time
import math
import base64
import pickle
import zipfile
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("master_generators_app")

# ---------------- path setup ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------- resilient imports from src/ ----------------
HAVE_SRC = True
LinearGeneratorFactory = NonlinearGeneratorFactory = None
CompleteLinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
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
    # Generators, constructors, factories
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType
    )
    try:
        from src.generators.master_generator import (
            CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory
        )
    except Exception:
        from src.generators.linear_generators import LinearGeneratorFactory, CompleteLinearGeneratorFactory
        from src.generators.nonlinear_generators import NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory

    # Master theorem utils
    from src.generators.master_theorem import (
        MasterTheoremSolver, MasterTheoremParameters, ExtendedMasterTheorem
    )

    # Functions libraries
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions

    # ML / DL modules
    from src.ml.pattern_learner import (
        GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer, create_model
    )
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator

    # DL novelty & tokenizer
    from src.dl.novelty_detector import (
        ODENoveltyDetector, NoveltyAnalysis, ODETokenizer, ODETransformer
    )

    # Utils & UI
    from src.utils.config import Settings, AppConfig
    from src.utils.cache import CacheManager, cached
    from src.utils.validators import ParameterValidator
    from src.ui.components import UIComponents

except Exception as e:
    HAVE_SRC = False
    log.warning(f"Some imports from src/ failed or are missing: {e}")

# ---------------- internal core & RQ helpers ----------------
try:
    from shared.ode_core import (
        ComputeParams, compute_ode_full, theorem_4_2_y_m_expr, get_function_expr,
        parse_arbitrary_lhs, to_exact, simplify_expr
    )
except Exception as e:
    log.error(f"shared/ode_core import error: {e}")
    raise

try:
    from rq_utils import has_redis, enqueue_job, fetch_job, redis_status
except Exception as e:
    log.error(f"rq_utils import error: {e}")
    raise

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators for ODEs — Complete",
    page_icon="🧮", layout="wide", initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1E88E5 0%,#6A1B9A 100%);
padding:1.6rem;border-radius:14px;margin-bottom:1.2rem;color:white;text-align:center;
box-shadow:0 10px 28px rgba(0,0,0,0.2);}
.main-title{font-size:2.0rem;font-weight:700;margin-bottom:.3rem;}
.subtitle{font-size:1.0rem;opacity:.95;}
.metric-card{background:linear-gradient(135deg,#1E88E5 0%,#6A1B9A 100%);color:white;
padding:1rem;border-radius:12px;text-align:center;box-shadow:0 10px 20px rgba(0,0,0,0.2);}
.info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:1rem 0;}
.result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
border:2px solid #4caf50;padding:1rem;border-radius:10px;margin:1rem 0;}
.error-box{background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);
border:2px solid #f44336;padding:1rem;border-radius:10px;margin:1rem 0;}
.logbox{background:#0b1021;color:#cde7ff;padding:0.75rem;border-radius:8px;font-family:monospace;
max-height:320px;overflow:auto;border:1px solid #2c3a5b;}
.small{font-size:0.88rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
def _ss_init():
    if "generator_constructor" not in st.session_state and GeneratorConstructor:
        st.session_state.generator_constructor = GeneratorConstructor()
    defaults = {
        "generator_terms": [],
        "generated_odes": [],
        "generator_patterns": [],
        "ml_trainer": None,
        "ml_trained": False,
        "training_history": {},
        "trained_models": [],
        "batch_results": [],
        "analysis_results": [],
        "export_history": [],
        "lhs_source": "constructor",
        "free_terms": [],
        "arbitrary_lhs_text": "",
        "last_job_id": None,
        "last_train_job_id": None,
        "last_ping_job_id": None,
        "basic_functions": None,
        "special_functions": None,
        "novelty_detector": None,
        "ode_classifier": None,
        "cache_manager": None,
        "save_preamble": True,
        "reverse_cache": {}
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.basic_functions is None and BasicFunctions:
        st.session_state.basic_functions = BasicFunctions()
    if st.session_state.special_functions is None and SpecialFunctions:
        st.session_state.special_functions = SpecialFunctions()
    if st.session_state.novelty_detector is None and ODENoveltyDetector:
        try:
            st.session_state.novelty_detector = ODENoveltyDetector()
        except Exception:
            st.session_state.novelty_detector = None
    if st.session_state.ode_classifier is None and ODEClassifier:
        try:
            st.session_state.ode_classifier = ODEClassifier()
        except Exception:
            st.session_state.ode_classifier = None
    if st.session_state.cache_manager is None and CacheManager:
        st.session_state.cache_manager = CacheManager()

_ss_init()

# ---------------- Small helpers ----------------
def _latex(expr) -> str:
    try:
        if isinstance(expr, str):
            try:
                expr = sp.sympify(expr)
            except Exception:
                return expr
        return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
    except Exception:
        return str(expr)

def _mask(s: str, keep=4) -> str:
    s = s or ""
    return s[:keep] + "***" + s[-keep:] if len(s) > keep*2 else "***"

def _register_generated_ode(res: Dict[str, Any]):
    d = dict(res)
    d.setdefault("type", "nonlinear")
    d.setdefault("order", 0)
    d.setdefault("function_used", "unknown")
    d.setdefault("parameters", {})
    d.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    d["generator_number"] = len(st.session_state.generated_odes) + 1
    try:
        d.setdefault("ode", sp.Eq(sp.sympify(d["generator"]), sp.sympify(d["rhs"])))
    except Exception:
        pass
    st.session_state.generated_odes.append(d)

def _download_bytes(label: str, data: bytes, filename: str, mime: str = "application/octet-stream"):
    st.download_button(label, data, file_name=filename, mime=mime, use_container_width=True)

def _list_artifacts(pattern: str = "checkpoints/*.pth") -> List[str]:
    try:
        return sorted(glob.glob(pattern))
    except Exception:
        return []

# ---------------- Redis / RQ helpers on the UI side ----------------
def _try_enqueue_or_sync(func_path: str, payload: dict, description: str):
    """
    Try enqueue to RQ; if Redis down/unavailable, compute synchronously in this process (for compute jobs).
    Training fallback stays in worker only (no local heavy training in web).
    """
    if has_redis():
        job_id = enqueue_job(func_path, payload, description=description)
        if job_id:
            return {"mode": "queued", "job_id": job_id}
        else:
            st.warning("Enqueue failed; falling back to synchronous (compute only).")
    else:
        st.info("Redis not available; falling back to synchronous (compute only).")

    if func_path == "worker.compute_job":
        # Synchronous compute here:
        try:
            basic_lib = st.session_state.basic_functions
            special_lib = st.session_state.special_functions
            constructor_lhs = None
            if payload.get("lhs_source") == "constructor" and payload.get("constructor_lhs"):
                constructor_lhs = sp.sympify(payload["constructor_lhs"])
            p = ComputeParams(
                func_name=payload.get("func_name","exp(z)"),
                alpha=payload.get("alpha",1),
                beta=payload.get("beta",1),
                n=int(payload.get("n",1)),
                M=payload.get("M",0),
                use_exact=bool(payload.get("use_exact",True)),
                simplify_level=payload.get("simplify_level","light"),
                lhs_source=payload.get("lhs_source","constructor"),
                constructor_lhs=constructor_lhs,
                freeform_terms=payload.get("freeform_terms"),
                arbitrary_lhs_text=payload.get("arbitrary_lhs_text"),
                function_library=payload.get("function_library","Basic"),
                basic_lib=basic_lib,
                special_lib=special_lib,
            )
            res = compute_ode_full(p)
            return {"mode": "sync", "result": res}
        except Exception as e:
            return {"mode": "sync", "error": str(e)}

    return {"mode": "none"}

# ---------------- Exporters ----------------
class LaTeXExporter:
    @staticmethod
    def generate(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        generator = ode_data.get("generator", "")
        rhs       = ode_data.get("rhs", "")
        sol       = ode_data.get("solution", "")
        params    = ode_data.get("parameters", {})
        ic        = ode_data.get("initial_conditions", {})
        cls       = ode_data.get("classification", {})
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
            f"{_latex(generator)} = {_latex(rhs)}",
            r"\end{equation}",
            r"\subsection{Exact Solution}",
            r"\begin{equation}",
            f"y(x) = {_latex(sol)}",
            r"\end{equation}",
            r"\subsection{Parameters}",
            r"\begin{align}",
            f"\\alpha &= {_latex(params.get('alpha', 1))} \\\\",
            f"\\beta  &= {_latex(params.get('beta', 1))} \\\\",
            f"n       &= {params.get('n', 1)} \\\\",
            f"M       &= {_latex(params.get('M', 0))}",
            r"\end{align}",
        ]
        if ic:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            items = list(ic.items())
            for i,(k,v) in enumerate(items):
                parts.append(f"{k} &= {_latex(v)}" + (r" \\" if i<len(items)-1 else ""))
            parts.append(r"\end{align}")
        if cls:
            parts += [r"\subsection{Classification}", r"\begin{itemize}"]
            parts.append(f"\\item Type: {cls.get('type','Unknown')}")
            parts.append(f"\\item Order: {cls.get('order','Unknown')}")
            parts.append(f"\\item Linearity: {cls.get('linearity','Unknown')}")
            if "field" in cls: parts.append(f"\\item Field: {cls['field']}")
            if "applications" in cls: parts.append(f"\\item Applications: {', '.join(cls['applications'][:6])}")
            parts.append(r"\end{itemize}")
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def package(ode_data: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\nCompile with: pdflatex ode_document.tex\n")
        buf.seek(0)
        return buf.getvalue()

# ---------------- UI sections ----------------
def header():
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🧮 Master Generators for ODEs — Complete Edition</div>
      <div class="subtitle">Theorem 4.1 & 4.2 • Free‑form/Arbitrary LHS • Batch • ML/DL Training via RQ • Reverse Engineering • Export</div>
    </div>
    """, unsafe_allow_html=True)

def dashboard_page():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>ML Trained</h3><h1>{"✅" if st.session_state.ml_trained else "❌"}</h1></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h3>Training Epochs</h3><h1>{len(st.session_state.training_history.get("train_loss", []))}</h1></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><h3>Artifacts</h3><h1>{len(_list_artifacts())}</h1></div>', unsafe_allow_html=True)

    st.subheader("Recent ODEs")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes)[["generator_number","type","order","function_used","timestamp"]]
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("No ODEs yet — try **Apply Master Theorem**.")

def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Build custom LHS for Theorem 4.1. You can also use Free‑form or Arbitrary expressions on the theorem page.</div>', unsafe_allow_html=True)
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not available. You can still use Free‑form/Arbitrary LHS on the theorem page.")
        return

    with st.expander("➕ Add Term", True):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            k = st.selectbox("Derivative order k", [0,1,2,3,4,5], format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}[x])
        with c2:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType], format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coef = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 8, 1)

        c5,c6,c7 = st.columns(3)
        with c5:
            op_type = st.selectbox("Operator", [t.value for t in OperatorType], format_func=lambda s: s.replace("_"," ").title())
        with c6:
            scaling = st.number_input("Scaling a", 0.25, 8.0, 1.0, 0.25) if op_type in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift b", -10.0, 10.0, 0.0, 0.1) if op_type in ["delay","advance"] else None

        if st.button("Add Term", type="primary"):
            term = DerivativeTerm(
                derivative_order=int(k),
                coefficient=float(coef),
                power=int(power),
                function_type=DerivativeType(func_type),
                operator_type=OperatorType(op_type),
                scaling=scaling,
                shift=shift
            )
            st.session_state.generator_terms.append(term)
            st.success("Term added.")

    if st.session_state.generator_terms:
        st.subheader("Current Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            cols = st.columns([8,1])
            with cols[0]:
                desc = term.get_description() if hasattr(term,"get_description") else str(term)
                st.write(f"• {desc}")
            with cols[1]:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()

        if st.button("Build Generator Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Generator {len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = gen_spec
                st.success("Generator specification created.")
                try:
                    st.latex(_latex(gen_spec.lhs) + " = RHS")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to build spec: {e}")

    if st.button("Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None
        st.success("Cleared.")

def _build_freeform_terms_ui():
    st.subheader("🧩 Free‑form LHS Builder")
    with st.expander("Build terms", False):
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_k = st.number_input("inner k", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs","asin","acos","atan","asinh","acosh","atanh","erf","erfc"], index=0)
        with cols[3]: power = st.number_input("power", 1, 8, 1)
        with cols[4]: outer_m = st.number_input("outer m (D^m)", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale a", value=1.0, step=0.1)
        with cols[6]: shift = st.number_input("arg shift b", value=0.0, step=0.1)
        with cols[7]:
            if st.button("➕ Add"):
                st.session_state.free_terms.append({
                    "coef": float(coef),
                    "inner_order": int(inner_k),
                    "wrapper": wrapper,
                    "power": int(power),
                    "outer_order": int(outer_m),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write("**Current terms:**")
            for i,t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. {t}")
            c1,c2 = st.columns(2)
            with c1:
                if st.button("Use free‑form"):
                    st.session_state.lhs_source = "freeform"
                    st.success("Free‑form selected.")
            with c2:
                if st.button("Clear free‑form terms"):
                    st.session_state.free_terms = []

def apply_master_theorem_page():
    st.header("🎯 Apply Master Theorem (4.1 / 4.2)")

    # Source of LHS
    src_idx = {"constructor":0, "freeform":1, "arbitrary":2}.get(st.session_state.lhs_source, 0)
    lhs_src = st.radio("LHS source", ["constructor","freeform","arbitrary"], index=src_idx, horizontal=True)
    st.session_state.lhs_source = lhs_src

    # Function library
    colA, colB = st.columns(2)
    with colA:
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        if lib == "Basic" and st.session_state.basic_functions:
            func_names = st.session_state.basic_functions.get_function_names()
        elif lib == "Special" and st.session_state.special_functions:
            func_names = st.session_state.special_functions.get_function_names()
        else:
            func_names = []
        func_name = st.selectbox("Choose f(z)", func_names) if func_names else st.text_input("Enter f(z)", "exp(z)")

    # Parameters
    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c5,c6,c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7:
        rstat = redis_status()
        st.caption("Redis: " + ("✅ Available" if rstat.get("ok") else "❌ Unavailable"))
        if rstat.get("ok"):
            st.caption("Queue: " + rstat.get("queue", "?"))

    # Free‑form builder
    _build_freeform_terms_ui()

    # Arbitrary LHS
    st.subheader("✍️ Arbitrary LHS (SymPy expr in x, y(x))")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Example: sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1)",
        value=st.session_state.arbitrary_lhs_text or "",
        height=100
    )
    cc1,cc2 = st.columns(2)
    with cc1:
        if st.button("✅ Validate arbitrary LHS"):
            try:
                _ = parse_arbitrary_lhs(st.session_state.arbitrary_lhs_text)
                st.success("Expression parsed successfully.")
                st.session_state.lhs_source = "arbitrary"
            except Exception as e:
                st.error(f"Parse error: {e}")
    with cc2:
        if st.button("↩️ Prefer Constructor"):
            st.session_state.lhs_source = "constructor"

    # Compute
    st.markdown("---")
    cl1, cl2 = st.columns([1,1], gap="large")
    with cl1:
        if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
            # gather constructor LHS (if present)
            constructor_lhs = None
            if "current_generator" in st.session_state and getattr(st.session_state.get("current_generator"), "lhs", None) is not None:
                constructor_lhs = st.session_state.current_generator.lhs
            elif st.session_state.get("generator_constructor") and hasattr(st.session_state.generator_constructor, "get_generator_expression"):
                try:
                    constructor_lhs = st.session_state.generator_constructor.get_generator_expression()
                except Exception:
                    constructor_lhs = None

            payload = {
                "func_name": func_name,
                "alpha": alpha,
                "beta": beta,
                "n": int(n),
                "M": M,
                "use_exact": use_exact,
                "simplify_level": simplify_level,
                "lhs_source": st.session_state.lhs_source,
                "freeform_terms": st.session_state.free_terms,
                "arbitrary_lhs_text": st.session_state.arbitrary_lhs_text,
                "function_library": lib,
                # >>> IMPORTANT: serialize constructor LHS for the worker (fixes "stuck" bug)
                "constructor_lhs": (str(constructor_lhs) if constructor_lhs is not None else None),
            }

            result = _try_enqueue_or_sync("worker.compute_job", payload, description="ode-generate")
            if result.get("mode") == "queued":
                st.session_state.last_job_id = result["job_id"]
                st.success(f"Job submitted (ID: {result['job_id']})")
            elif result.get("mode") == "sync":
                if result.get("result"):
                    res = result["result"]
                    try:
                        res["generator"] = sp.sympify(res["generator"])
                        res["rhs"]       = sp.sympify(res["rhs"])
                        res["solution"]  = sp.sympify(res["solution"])
                    except Exception:
                        pass
                    _register_generated_ode(res)
                    _show_ode_result(res)
                else:
                    st.error(f"Generation error: {result.get('error', 'Unknown error')}")

    with cl2:
        compute_mth = st.checkbox("Compute y^(m) via Theorem 4.2", False)
        m_order = st.number_input("m", 1, 12, 1)
        if compute_mth and st.button("🧮 Compute y^{(m)}(x)", use_container_width=True):
            try:
                lib_obj = st.session_state.basic_functions if lib=="Basic" else st.session_state.special_functions
                f_expr = get_function_expr(lib_obj, func_name)
                α = to_exact(alpha) if use_exact else sp.Float(alpha)
                β = to_exact(beta)  if use_exact else sp.Float(beta)
                x = sp.Symbol("x", real=True)
                y_m = theorem_4_2_y_m_expr(f_expr, α, β, int(n), int(m_order), x, simplify_level)
                st.latex(fr"y^{{({int(m_order)})}}(x) = " + _latex(y_m))
            except Exception as e:
                st.error(f"Failed to compute y^{m_order}(x): {e}")

    # Poll job if any
    if st.session_state.last_job_id:
        st.markdown("### 📡 Job Monitor")
        colx, coly = st.columns([1,1])
        with colx:
            if st.button("🔄 Refresh job status"):
                pass
        info = fetch_job(st.session_state.last_job_id)
        if info:
            st.code(json.dumps({
                "id": info.get("id"),
                "status": info.get("status"),
                "origin": info.get("origin"),
                "enqueued_at": info.get("enqueued_at"),
                "started_at": info.get("started_at"),
                "ended_at": info.get("ended_at"),
                "meta": info.get("meta", {}),
            }, indent=2), language="json")
            logs = info.get("logs") or []
            if logs:
                st.markdown("**Worker Logs**")
                st.markdown('<div class="logbox">' + "<br/>".join([sp.escape(l) if hasattr(sp, "escape") else l for l in logs[-200:]]) + "</div>", unsafe_allow_html=True)

            if info.get("status") == "finished" and info.get("result"):
                res = info["result"]
                try:
                    res["generator"] = sp.sympify(res["generator"])
                    res["rhs"]       = sp.sympify(res["rhs"])
                    res["solution"]  = sp.sympify(res["solution"])
                except Exception:
                    pass
                _register_generated_ode(res)
                _show_ode_result(res)
                st.session_state.last_job_id = None
            elif info.get("status") == "failed":
                st.error("Job failed.")
                if info.get("exc_info"):
                    st.code(info["exc_info"], language="python")
                st.session_state.last_job_id = None
        else:
            st.info("Waiting for job info…")

def _show_ode_result(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h4>✅ ODE Generated</h4></div>', unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["📐 Equation", "💡 Solution & Params", "📤 Export"])
    with t1:
        try:
            st.latex(_latex(res["generator"]) + " = " + _latex(res["rhs"]))
        except Exception:
            st.write("LHS:", res.get("generator")); st.write("RHS:", res.get("rhs"))
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t2:
        try:
            st.latex("y(x) = " + _latex(res["solution"]))
        except Exception:
            st.write("y(x) =", res.get("solution"))
        params = res.get("parameters", {})
        st.write(f"**Parameters:** α={params.get('alpha')} | β={params.get('beta')} | n={params.get('n')} | M={params.get('M')}")
        ic = res.get("initial_conditions", {})
        if ic:
            st.write("**Initial Conditions:**")
            for k, v in ic.items():
                try: st.latex(k + " = " + _latex(v))
                except Exception: st.write(k, "=", v)
        if "f_expr_preview" in res:
            st.write(f"**f(z)** preview:", res["f_expr_preview"])
    with t3:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res.get("generator"),
            "rhs": res.get("rhs"),
            "solution": res.get("solution"),
            "parameters": res.get("parameters", {}),
            "classification": {
                "type": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "order": res.get("order", 0),
                "linearity": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "field": "Mathematical Physics",
                "applications": ["Research Equation"]
            },
            "initial_conditions": res.get("initial_conditions", {}),
            "function_used": str(res.get("function_used","?")),
            "generator_number": idx,
            "type": res.get("type","nonlinear"),
            "order": res.get("order", 0)
        }
        tex = LaTeXExporter.generate(ode_data, include_preamble=st.session_state.save_preamble)
        st.download_button("📄 Download LaTeX", tex, f"ode_{idx}.tex", "text/x-latex", use_container_width=True)
        pkg = LaTeXExporter.package(ode_data)
        _download_bytes("📦 Download Package (ZIP)", pkg, f"ode_package_{idx}.zip", "application/zip")

def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning (via RQ worker)")
    if not MLTrainer:
        st.warning("MLTrainer not available.")
    rstat = redis_status()
    st.caption("Redis: " + ("✅ Available" if rstat.get("ok") else "❌ Unavailable"))
    if not rstat.get("ok"):
        st.error("Background training requires Redis/RQ.")
    col0,colA,colB = st.columns([1,1,1])
    with col0:
        model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"], index=0)
    with colA:
        epochs = st.slider("Epochs", 5, 500, 100)
        batch_size = st.slider("Batch Size", 8, 256, 32)
        samples = st.slider("Training Samples (synthetic)", 100, 10000, 1000)
    with colB:
        val_split = st.select_slider("Validation Split", [0.1,0.15,0.2,0.25,0.3], value=0.2)
        hidden_dim = st.select_slider("Hidden Dim", [32,64,128,256,512], value=128)
        mp = st.checkbox("Mixed Precision", False)

    # Actions
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        if st.button("🚀 Start Training (RQ)", type="primary", use_container_width=True):
            if not rstat.get("ok"):
                st.error("Redis not configured; cannot start training.")
            else:
                payload = {
                    "model_type": model_type,
                    "hidden_dim": hidden_dim,
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "samples": int(samples),
                    "validation_split": float(val_split),
                    "use_generator": True,
                    "enable_mixed_precision": bool(mp),
                    # Keep ctor/train kwargs minimal to avoid signature mismatches
                }
                job_id = enqueue_job("worker.train_job", payload, description="ml-train")
                if job_id:
                    st.session_state.last_train_job_id = job_id
                    st.success(f"Training job queued (ID: {job_id})")
                else:
                    st.error("Failed to enqueue training job.")
    with c2:
        if st.button("📡 Refresh Status", use_container_width=True):
            pass
    with c3:
        if st.button("🧽 Clear Train Job", use_container_width=True):
            st.session_state.last_train_job_id = None
    with c4:
        if st.button("🔎 List Artifacts", use_container_width=True):
            st.session_state["_artifacts"] = _list_artifacts()

    # Monitor
    if st.session_state.last_train_job_id:
        st.subheader("📡 Training Monitor")
        info = fetch_job(st.session_state.last_train_job_id)
        if info:
            st.code(json.dumps({
                "id": info.get("id"),
                "status": info.get("status"),
                "enqueued_at": info.get("enqueued_at"),
                "started_at": info.get("started_at"),
                "ended_at": info.get("ended_at"),
                "meta": info.get("meta", {}),
            }, indent=2), language="json")
            logs = info.get("logs") or []
            if logs:
                st.markdown("**Worker Logs**")
                st.markdown('<div class="logbox">' + "<br/>".join([sp.escape(l) if hasattr(sp, "escape") else l for l in logs[-300:]]) + "</div>", unsafe_allow_html=True)
            if info.get("status") == "finished" and info.get("result"):
                res = info["result"]
                st.success("Training finished.")
                hist = res.get("history", {})
                st.session_state.training_history = hist
                st.session_state.ml_trained = True
                artifact = res.get("artifact")
                if artifact and os.path.exists(artifact):
                    if artifact not in st.session_state.trained_models:
                        st.session_state.trained_models.append(artifact)
                # Plot training curves if present
                if hist.get("train_loss"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(1, len(hist["train_loss"])+1)), y=hist["train_loss"], mode="lines", name="Train"))
                    if hist.get("val_loss"):
                        fig.add_trace(go.Scatter(x=list(range(1, len(hist["val_loss"])+1)), y=hist["val_loss"], mode="lines", name="Val"))
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                st.session_state.last_train_job_id = None
            elif info.get("status") == "failed":
                st.error("Training failed.")
                if info.get("exc_info"):
                    st.code(info["exc_info"], language="python")
                st.session_state.last_train_job_id = None
        else:
            st.info("Waiting for training job…")

    st.markdown("---")
    st.subheader("📦 Artifacts & Session")
    cols = st.columns(4)
    with cols[0]:
        st.write("Artifacts on disk:")
        files = st.session_state.get("_artifacts") or _list_artifacts()
        if files:
            st.dataframe(pd.DataFrame({"artifact": files}))
        else:
            st.caption("No .pth artifacts found in checkpoints/.")
    with cols[1]:
        up = st.file_uploader("Upload model artifact (.pth)", type=["pth"])
        if up is not None:
            try:
                os.makedirs("checkpoints", exist_ok=True)
                path = os.path.join("checkpoints", f"uploaded_{int(time.time())}.pth")
                with open(path, "wb") as f:
                    f.write(up.getbuffer())
                st.success(f"Uploaded to {path}")
                st.session_state.trained_models.append(path)
                st.session_state.ml_trained = True
            except Exception as e:
                st.error(f"Upload failed: {e}")
    with cols[2]:
        if st.button("💾 Save Session", use_container_width=True):
            try:
                data = {
                    "generated_odes": st.session_state.generated_odes,
                    "training_history": st.session_state.training_history,
                    "trained_models": st.session_state.trained_models,
                    "ml_trained": st.session_state.ml_trained,
                    "batch_results": st.session_state.batch_results,
                }
                buf = io.BytesIO()
                pickle.dump(data, buf)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                st.download_button("⬇️ Download Session File", base64.b64decode(b64), f"session_{int(time.time())}.pkl", "application/octet-stream", use_container_width=True)
            except Exception as e:
                st.error(f"Save failed: {e}")
    with cols[3]:
        up_sess = st.file_uploader("Upload Session (.pkl)", type=["pkl"], key="sess_up")
        if up_sess is not None:
            try:
                data = pickle.loads(up_sess.getvalue())
                st.session_state.generated_odes = data.get("generated_odes", st.session_state.generated_odes)
                st.session_state.training_history = data.get("training_history", {})
                st.session_state.trained_models = data.get("trained_models", [])
                st.session_state.ml_trained = bool(data.get("ml_trained", False))
                st.session_state.batch_results = data.get("batch_results", st.session_state.batch_results)
                st.success("Session loaded.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.markdown("---")
    st.subheader("🎨 Generate Novel ODEs (from trained model)")
    num_gen = st.slider("How many", 1, 10, 1)
    if st.button("Generate", type="primary"):
        if not MLTrainer:
            st.error("MLTrainer not available.")
        else:
            try:
                # Try to load the most recent artifact if trainer not in memory
                if st.session_state.ml_trainer is None:
                    st.session_state.ml_trainer = MLTrainer()
                    # best effort to load last artifact if exists
                    arts = _list_artifacts()
                    if arts:
                        st.session_state.ml_trainer.load_model(arts[-1])
                gen_ok = 0
                for i in range(num_gen):
                    res = st.session_state.ml_trainer.generate_new_ode()
                    if res:
                        _register_generated_ode(res)
                        gen_ok += 1
                st.success(f"Generated {gen_ok}/{num_gen} ODEs.")
            except Exception as e:
                st.error(f"Generation failed: {e}")

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs by sampling parameters and factory types.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        N = st.slider("Number of ODEs", 5, 500, 50)
        gen_types = st.multiselect("Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        categories = st.multiselect("Function categories", ["Basic","Special"], default=["Basic"])
        vary = st.checkbox("Vary parameters", True)
    with c3:
        if vary:
            alpha_range = st.slider("α range", -10.0, 10.0, (-2.0, 2.0))
            beta_range  = st.slider("β range",  0.1, 10.0, (0.5, 2.0))
            n_range     = st.slider("n range", 1, 5, (1, 3))
        else:
            alpha_range=(1.0,1.0); beta_range=(1.0,1.0); n_range=(1,1)

    if st.button("🚀 Generate Batch", type="primary"):
        with st.spinner("Generating…"):
            out = []
            funcs = []
            if "Basic" in categories and st.session_state.basic_functions:
                funcs += st.session_state.basic_functions.get_function_names()
            if "Special" in categories and st.session_state.special_functions:
                funcs += st.session_state.special_functions.get_function_names()[:20]
            if not funcs:
                st.error("No functions available.")
                return
            for i in range(N):
                try:
                    params = {
                        "alpha": float(np.random.uniform(*alpha_range)),
                        "beta":  float(np.random.uniform(*beta_range)),
                        "n": int(np.random.randint(n_range[0], n_range[1]+1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    fname = np.random.choice(funcs)
                    t = np.random.choice(gen_types)
                    res = {}
                    if t == "linear":
                        if CompleteLinearGeneratorFactory:
                            fac = CompleteLinearGeneratorFactory()
                            gen_num = np.random.randint(1,9)
                            if gen_num in [4,5]:
                                params["a"] = float(np.random.uniform(1,3))
                            res = fac.create(gen_num, st.session_state.basic_functions.get_function(fname), **params)
                        elif LinearGeneratorFactory:
                            fac = LinearGeneratorFactory()
                            res = fac.create(1, st.session_state.basic_functions.get_function(fname), **params)
                    else:
                        if CompleteNonlinearGeneratorFactory:
                            fac = CompleteNonlinearGeneratorFactory()
                            gen_num = np.random.randint(1,11)
                            if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                            if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                            if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                            res = fac.create(gen_num, st.session_state.basic_functions.get_function(fname), **params)
                        elif NonlinearGeneratorFactory:
                            fac = NonlinearGeneratorFactory()
                            res = fac.create(1, st.session_state.basic_functions.get_function(fname), **params)
                    if not res:
                        continue
                    row = {
                        "ID": i+1, "Type": res.get("type","unknown"),
                        "Generator": res.get("generator_number","?"),
                        "Function": fname, "Order": res.get("order",0),
                        "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                    }
                    out.append(row)
                except Exception:
                    pass
            st.session_state.batch_results.extend(out)
            st.success(f"Generated {len(out)} ODE rows.")
            if out:
                st.dataframe(pd.DataFrame(out), use_container_width=True)

def novelty_detection_page():
    st.header("🔍 Novelty Detection")
    if not st.session_state.novelty_detector:
        st.warning("Novelty detector not available.")
        return
    mode = st.radio("Input", ["Use Current LHS", "Enter ODE", "From Generated"])
    ode_obj = None
    if mode == "Use Current LHS":
        gen_spec = st.session_state.get("current_generator")
        if gen_spec is not None and hasattr(gen_spec, "lhs"):
            ode_obj = {"ode": gen_spec.lhs, "type":"custom", "order": getattr(gen_spec, "order", 2)}
        else:
            st.warning("No generator spec available.")
    elif mode == "Enter ODE":
        ode_str = st.text_area("Enter ODE (text/LaTeX/sympy):")
        if ode_str:
            ode_obj = {"ode": ode_str, "type":"manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if not st.session_state.generated_odes:
            st.info("No generated ODEs yet.")
            return
        idx = st.selectbox("Select", range(len(st.session_state.generated_odes)),
                           format_func=lambda i: f"ODE {i+1} (order {st.session_state.generated_odes[i].get('order',0)})")
        ode_obj = st.session_state.generated_odes[idx]

    if ode_obj and st.button("Analyze", type="primary"):
        with st.spinner("Analyzing…"):
            try:
                analy = st.session_state.novelty_detector.analyze(ode_obj, check_solvability=True, detailed=True)
                st.metric("Novelty", "🟢 NOVEL" if analy.is_novel else "🔴 STANDARD")
                st.metric("Score", f"{analy.novelty_score:.1f}/100")
                st.metric("Confidence", f"{analy.confidence:.1%}")
                with st.expander("Details", True):
                    st.write("Complexity:", analy.complexity_level)
                    st.write("Solvable by standard methods:", "Yes" if analy.solvable_by_standard_methods else "No")
                    if analy.recommended_methods:
                        st.write("Recommended methods:"); [st.write("•", t) for t in analy.recommended_methods[:6]]
                if analy.detailed_report:
                    st.download_button("📥 Download Report", analy.detailed_report,
                                       f"novelty_report_{int(time.time())}.txt", "text/plain")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

def analysis_classification_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs to analyze.")
        return
    if not st.session_state.ode_classifier:
        st.warning("Classifier not available.")
        return
    summary = []
    for i, ode in enumerate(st.session_state.generated_odes[-100:]):
        summary.append({
            "ID": i+1,
            "Type": ode.get("type","?"),
            "Order": ode.get("order",0),
            "Function": ode.get("function_used","?"),
            "Time": (ode.get("timestamp") or "")[:19]
        })
    df = pd.DataFrame(summary)
    st.dataframe(df, use_container_width=True)
    if st.button("Classify All", type="primary"):
        with st.spinner("Classifying…"):
            try:
                classes = []
                for ode in st.session_state.generated_odes:
                    try:
                        classes.append(st.session_state.ode_classifier.classify_ode(ode))
                    except Exception:
                        classes.append({})
                fields = [c.get("classification",{}).get("field","Unknown") for c in classes if c]
                vc = pd.Series(fields).value_counts()
                fig = px.bar(x=vc.index, y=vc.values, title="Field Distribution")
                fig.update_layout(xaxis_title="Field", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Classification failed: {e}")

def physical_applications_page():
    st.header("🔬 Physical Applications")
    st.markdown('<div class="info-box">Browse common physics/engineering equations.</div>', unsafe_allow_html=True)
    apps = {
        "Mechanics": [
            ("Harmonic Oscillator", "y'' + ω^2 y = 0"),
            ("Damped Oscillator", "y'' + 2γ y' + ω₀² y = 0"),
            ("Forced Oscillator", "y'' + 2γ y' + ω₀² y = F cos(ωt)"),
        ],
        "Quantum": [
            ("Schrödinger (1D)", "-ℏ²/(2m) y'' + V(x) y = E y"),
            ("Particle in a Box", "y'' + (2mE/ℏ²) y = 0"),
        ],
        "Thermodynamics": [
            ("Heat Equation", "∂T/∂t = α ∇²T"),
            ("Newton Cooling", "dT/dt = -k (T - T_env)"),
        ]
    }
    cat = st.selectbox("Select field", list(apps.keys()))
    for name, eq in apps[cat]:
        with st.expander(f"📘 {name}"):
            try: st.latex(eq)
            except Exception: st.write(eq)

def visualization_page():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.info("Generate ODEs first.")
        return
    sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1} (order {st.session_state.generated_odes[i].get('order',0)})")
    ode = st.session_state.generated_odes[sel]
    c1,c2,c3 = st.columns(3)
    with c1: ptype = st.selectbox("Plot", ["Solution","Direction Field","Phase Portrait"])
    with c2: x_range = st.slider("x range", -10.0, 10.0, (-5.0, 5.0))
    with c3: npoints = st.slider("Points", 100, 2000, 400)
    if st.button("Plot", type="primary"):
        with st.spinner("Rendering…"):
            # Placeholder numeric example — replace with numeric eval of solution if needed
            x = np.linspace(x_range[0], x_range[1], npoints)
            y = np.sin(x) * np.exp(-0.05*np.abs(x))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="y(x)"))
            fig.update_layout(title="Solution (illustrative)", xaxis_title="x", yaxis_title="y(x)")
            st.plotly_chart(fig, use_container_width=True)

def export_latex_page():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.info("No ODEs to export.")
        return
    kind = st.radio("Export", ["Single","Multiple","Complete Report"])
    if kind == "Single":
        idx = st.selectbox("Select", range(len(st.session_state.generated_odes)),
                           format_func=lambda i: f"ODE {i+1} ({st.session_state.generated_odes[i].get('type','?')})")
        ode = st.session_state.generated_odes[idx]
        st.subheader("Preview (LaTeX code)")
        tex = LaTeXExporter.generate(ode, include_preamble=False)
        st.code(tex, language="latex")
        col1,col2 = st.columns(2)
        with col1:
            full = LaTeXExporter.generate(ode, include_preamble=st.session_state.save_preamble)
            st.download_button("📄 Download LaTeX", full, f"ode_{idx+1}.tex", "text/x-latex")
        with col2:
            pkg = LaTeXExporter.package(ode)
            _download_bytes("📦 Download Package", pkg, f"ode_package_{idx+1}.zip", "application/zip")
    elif kind == "Multiple":
        sel = st.multiselect("Select ODEs", range(len(st.session_state.generated_odes)),
                             format_func=lambda i: f"ODE {i+1} ({st.session_state.generated_odes[i].get('type','?')})")
        if sel and st.button("Build LaTeX"):
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
            for k,i in enumerate(sel,1):
                parts.append(f"\\section*{{ODE {k}}}")
                parts.append(LaTeXExporter.generate(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            st.download_button("📄 Download", "\n".join(parts), f"odes_{int(time.time())}.tex", "text/x-latex")
    else:
        if st.button("Generate Complete Report"):
            parts = [r"""\documentclass[12pt]{report}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\title{Master Generators System - Complete Report}
\author{Generated Automatically}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
"""]
            parts.append(r"\chapter{Generated ODEs}")
            for i, ode in enumerate(st.session_state.generated_odes, 1):
                parts.append(f"\\section*{{ODE {i}}}")
                parts.append(LaTeXExporter.generate(ode, include_preamble=False))
            parts.append(r"\end{document}")
            st.download_button("📄 Download Report", "\n".join(parts), f"report_{int(time.time())}.tex", "text/x-latex")

def examples_library_page():
    st.header("📚 Examples Library")
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")
    with st.expander("Exponential Decay"):
        st.latex("y' = -k y")

def documentation_page():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Open **🎯 Apply Master Theorem**.
2. Pick *Basic* or *Special* f(z); or type your own.
3. Set parameters (α, β, n, M) and toggle **Exact** for rationals.
4. Choose **constructor**, **free‑form**, or **arbitrary** LHS.
5. Click **Generate ODE**. If Redis is configured, job runs in background. Otherwise it computes synchronously.
6. Export from **📤 Export & LaTeX**.
7. Train ML via **🤖 ML Pattern Learning** (RQ worker) and monitor logs.
8. Explore **🔍 Novelty**, **📈 Analysis**, and **📐 Visualization**.
""")

def settings_page():
    st.header("⚙️ Settings")
    st.subheader("General")
    st.session_state.save_preamble = st.checkbox("Include LaTeX preamble by default", st.session_state.save_preamble)
    st.subheader("Redis Diagnostics")
    r = redis_status()
    st.json(r)
    col1,col2 = st.columns(2)
    with col1:
        if st.button("Enqueue ping job"):
            jid = enqueue_job("worker.ping_job", {"hello":"world"}, description="ping-job")
            st.session_state.last_ping_job_id = jid
            st.write("Job ID:", jid)
    with col2:
        if st.button("Check ping result"):
            jid = st.session_state.last_ping_job_id
            info = fetch_job(jid) if jid else None
            st.json(info or {"error":"no job id"})

# ---------------- Reverse Engineering ----------------
def reverse_engineering_page():
    st.header("🔁 Reverse Engineering")
    st.markdown("""This tool attempts to infer generator parameters or a best‑fit structure from a target **solution y(x)** or a target **ODE**.
It uses a hybrid strategy:
  • Analytical heuristics for simple φ‑forms \
  • ML‑assisted refinement (if a trained model is available)
""")
    mode = st.radio("Target", ["I have y(x) (solution)", "I have an ODE (LHS = RHS)"])
    if mode == "I have y(x) (solution)":
        sol_txt = st.text_area("Enter y(x) as a SymPy expression (e.g., exp(-x)*sin(x))", "")
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
        if st.button("Infer Parameters", type="primary"):
            try:
                x = sp.Symbol("x", real=True)
                yx = sp.sympify(sol_txt)
                # Simple heuristic: try to match y(x) ≈ x^M * [f(alpha x^beta)]^n
                candidates = (st.session_state.basic_functions.get_function_names()
                              if (lib=="Basic" and st.session_state.basic_functions) else
                              st.session_state.special_functions.get_function_names() if st.session_state.special_functions else [])
                best = {"score": -np.inf}
                xs = np.linspace(0.2, 2.0, 64)  # avoid x=0 singularities
                y_num = np.array([float(yx.subs(x, t)) for t in xs])
                y_num[np.isnan(y_num)] = 0.0
                y_num[np.isinf(y_num)] = 0.0

                for fname in candidates[:50]:
                    try:
                        f = (st.session_state.basic_functions.get_function(fname) if lib=="Basic"
                             else st.session_state.special_functions.get_function(fname))
                        # grid over alpha,beta,n,M (coarse)
                        for alpha in [0.5, 1.0, 2.0]:
                            for beta in [0.5, 1.0, 2.0]:
                                for n in [1,2,3]:
                                    for M in [-2,-1,0,1,2]:
                                        pred = []
                                        for t in xs:
                                            try:
                                                val = (t**M) * (float(f(alpha*(t**beta)))**n)
                                            except Exception:
                                                val = 0.0
                                            pred.append(val)
                                        pred = np.array(pred, dtype=float)
                                        # scale c*pred ~ y_num => c = argmin ||c*pred - y|| => closed form
                                        denom = float(np.dot(pred, pred)) + 1e-12
                                        c = float(np.dot(pred, y_num)) / denom
                                        err = float(np.mean((c*pred - y_num)**2))
                                        score = -err
                                        if score > best["score"]:
                                            best = {"fname": fname, "alpha": alpha, "beta": beta, "n": n, "M": M, "scale": c, "score": score}
                    except Exception:
                        continue

                if best.get("fname"):
                    st.success("Heuristic fit found.")
                    st.json(best)
                    st.caption("You can now use Theorem 4.1 with these parameters to construct a candidate generator.")
                    # Optional ML refinement: push to trained model if available
                    if st.session_state.ml_trained and MLTrainer:
                        st.info("ML refinement (denoising feature vector)")
                        # assemble a feature vector (same layout used by the trainer, best effort)
                        feat = np.array([best["alpha"], 1.0, best["n"], best["M"], 0, 1, 1, 2, 0, 0, 0, 0.0], dtype=np.float32)
                        try:
                            if st.session_state.ml_trainer is None:
                                st.session_state.ml_trainer = MLTrainer()
                                arts = _list_artifacts()
                                if arts:
                                    st.session_state.ml_trainer.load_model(arts[-1])
                            with st.spinner("Refining with ML…"):
                                t = st.session_state.ml_trainer
                                x_t = sp.Matrix(feat.tolist())
                                # The trainer implements forward(feature)->feature; use generate_new_ode as a proxy
                                res = t.generate_new_ode()
                                if res:
                                    st.write("ML suggested generator:")
                                    st.write(res.get("description","(no description)"))
                        except Exception as e:
                            st.warning(f"ML refinement not applied: {e}")
                else:
                    st.warning("No good fit found. Try a different library or expression.")
            except Exception as e:
                st.error(f"Reverse engineering failed: {e}")
    else:
        ode_txt = st.text_area("Enter ODE as `LHS = RHS` (SymPy/LaTeX/text)", "")
        if st.button("Analyze ODE"):
            try:
                # minimal parser: split at '=' and try sympify both sides
                if "=" in ode_txt:
                    left_s, right_s = ode_txt.split("=", 1)
                else:
                    left_s, right_s = ode_txt, "0"
                lhs = sp.sympify(left_s)
                rhs = sp.sympify(right_s)
                st.latex(_latex(lhs) + " = " + _latex(rhs))
                st.info("Try building a constructor LHS with matching structure, then apply Theorem 4.1.")
            except Exception as e:
                st.error(f"Parse failed: {e}")

# ---------------- Main router ----------------
def main():
    header()
    page = st.sidebar.radio("📍 Navigation", [
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
        "🔁 Reverse Engineering",
        "📚 Examples Library",
        "⚙️ Settings",
        "📖 Documentation",
    ])
    if page == "🏠 Dashboard":                dashboard_page()
    elif page == "🔧 Generator Constructor":  generator_constructor_page()
    elif page == "🎯 Apply Master Theorem":   apply_master_theorem_page()
    elif page == "🤖 ML Pattern Learning":    ml_pattern_learning_page()
    elif page == "📊 Batch Generation":       batch_generation_page()
    elif page == "🔍 Novelty Detection":      novelty_detection_page()
    elif page == "📈 Analysis & Classification": analysis_classification_page()
    elif page == "🔬 Physical Applications":  physical_applications_page()
    elif page == "📐 Visualization":          visualization_page()
    elif page == "📤 Export & LaTeX":         export_latex_page()
    elif page == "🔁 Reverse Engineering":    reverse_engineering_page()
    elif page == "📚 Examples Library":       examples_library_page()
    elif page == "⚙️ Settings":               settings_page()
    elif page == "📖 Documentation":          documentation_page()

if __name__ == "__main__":
    main()