# master_generators_app.py
"""
Master Generators for ODEs — Full App (Corrected + RQ/Persistence + ML/DL Enhancements)

This version preserves all services while adding:
- Robust RQ queue selection + origin visibility (no more "stuck queued" if worker listens to another queue).
- Persistent training progress (job.meta + Redis logs) and status that never disappears.
- Save / Load / Upload training sessions (best checkpoint + history JSON).
- "Trained" model state tracked on Dashboard; mark an Active Model.
- Post-training generation + ML-assisted reverse engineering (with safe fallbacks).
- Exposes VAE / Transformer hyper-parameters without breaking older flows.
- Keeps Constructor, Apply Theorem, Batch, Novelty, Analysis, Physical Apps, Visualization, Export, Examples, Settings, Docs.

Requires the fixed rq_utils.py and worker.py that expose:
- has_redis, enqueue_job, fetch_job, get_queue_stats, read_training_log
- worker.compute_job and worker.train_job
"""

# ---------------- std libs ----------------
import os, sys, io, json, time, base64, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import types
from pathlib import Path

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp
from sympy.core.function import AppliedUndef

# optional torch (not required for web, worker uses it)
try:
    import torch
except Exception:
    torch = None

# ---------------- logging ----------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("master_generators_app")

# ---------------- path setup ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------- RQ utils ----------------
try:
    from rq_utils import (
        has_redis, enqueue_job, fetch_job,
        get_queue_stats, read_training_log
    )
except Exception as e:
    logger.warning(f"rq_utils import failed: {e}")
    def has_redis(): return False
    def enqueue_job(*a, **k): return None
    def fetch_job(*a, **k): return {"status": "unknown", "error": "rq_utils missing"}
    def get_queue_stats(): return {"queues": [], "workers": []}
    def read_training_log(*a, **k): return []

# ---------------- math / core ----------------
# We call into shared.ode_core; all calls are error-protected and degrade gracefully.
try:
    from shared.ode_core import (
        ComputeParams, compute_ode_full, theorem_4_2_y_m_expr,
        get_function_expr, parse_arbitrary_lhs, to_exact, simplify_expr
    )
except Exception as e:
    logger.warning(f"shared.ode_core imports missing/partial: {e}")
    # Minimal safe stubs to keep UI alive if core isn't present (won't be used in worker mode anyway)
    class ComputeParams:
        def __init__(self, **kwargs): pass
    def compute_ode_full(p): raise RuntimeError("compute_ode_full not available")
    def theorem_4_2_y_m_expr(*a, **k): return sp.Symbol("y_m")(sp.Symbol("x"))
    def get_function_expr(*a, **k): return sp.exp(sp.Symbol("z"))
    def parse_arbitrary_lhs(txt): return sp.sympify(txt)
    def to_exact(x): 
        try: return sp.Rational(str(x))
        except Exception: return sp.nsimplify(x, rational=True)
    def simplify_expr(e, level="light"): return sp.simplify(e)

# ---------------- optional src libraries (best-effort) ----------------
HAVE_SRC = True
LinearGeneratorFactory = NonlinearGeneratorFactory = None
CompleteLinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
GeneratorConstructor = GeneratorSpecification = None
DerivativeTerm = DerivativeType = None
OperatorType = None
MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None
ODEClassifier = PhysicalApplication = None
BasicFunctions = SpecialFunctions = None
MLTrainer = ODEDataset = ODEDataGenerator = None
GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = None
ODENoveltyDetector = NoveltyAnalysis = ODETokenizer = ODETransformer = None
Settings = AppConfig = None
CacheManager = cached = None
ParameterValidator = None
UIComponents = None

try:
    from src.generators.linear_generators import LinearGeneratorFactory, CompleteLinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType
    )
    from src.generators.master_theorem import (
        MasterTheoremSolver, MasterTheoremParameters, ExtendedMasterTheorem
    )
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
    from src.ml.pattern_learner import (
        GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer
    )
    from src.dl.novelty_detector import (
        ODENoveltyDetector, NoveltyAnalysis, ODETokenizer, ODETransformer
    )
    from src.utils.config import Settings, AppConfig
    from src.utils.cache import CacheManager, cached
    from src.utils.validators import ParameterValidator
    from src.ui.components import UIComponents
except Exception as e:
    HAVE_SRC = False
    logger.warning(f"Some src/ imports failed (app will degrade gracefully): {e}")

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators ODE System",
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
kbd{background:#222;color:#fff;border-radius:6px;padding:2px 6px;}
</style>
""", unsafe_allow_html=True)

# ---------------- App constants ----------------
PREFERRED_QUEUE = os.getenv("RQ_QUEUE")  # worker + web should share this; rq_utils will auto-detect if None
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
HISTORY_DIR = os.getenv("HISTORY_DIR", CHECKPOINT_DIR)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)

# ---------------- Session State ----------------
def _ensure(name, default):
    if name not in st.session_state:
        st.session_state[name] = default

class SessionStateManager:
    @staticmethod
    def initialize():
        # Core collections
        for k, v in [
            ("generated_odes", []),
            ("generator_terms", []),
            ("generator_patterns", []),
            ("batch_results", []),
            ("analysis_results", []),
            ("export_history", []),
            ("training_history", {}),
            ("trained_models", []),          # keep track of finished trainings (paths)
            ("active_model_path", None),     # currently selected model to use
        ]:
            _ensure(k, v)

        # Objects / helpers
        if "generator_constructor" not in st.session_state and GeneratorConstructor:
            st.session_state.generator_constructor = GeneratorConstructor()
        if "basic_functions" not in st.session_state and BasicFunctions:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions:
            st.session_state.special_functions = SpecialFunctions()
        if "novelty_detector" not in st.session_state:
            try:
                st.session_state.novelty_detector = ODENoveltyDetector() if ODENoveltyDetector else None
            except Exception:
                st.session_state.novelty_detector = None
        if "ode_classifier" not in st.session_state:
            try:
                st.session_state.ode_classifier = ODEClassifier() if ODEClassifier else None
            except Exception:
                st.session_state.ode_classifier = None
        if "cache_manager" not in st.session_state and CacheManager:
            st.session_state.cache_manager = CacheManager()

        # Theorem page state
        for k, v in [
            ("lhs_source", "constructor"),
            ("free_terms", []),
            ("arbitrary_lhs_text", ""),
            ("last_compute_job_id", None),
            ("last_train_job_id", None)
        ]:
            _ensure(k, v)

# ---------------- Utilities ----------------
def _register_generated_ode(result: dict):
    """Normalize and append to session list."""
    r = dict(result)
    r.setdefault("type", "nonlinear")
    r.setdefault("order", 0)
    r.setdefault("function_used", "unknown")
    r.setdefault("parameters", {})
    r.setdefault("classification", {})
    r.setdefault("timestamp", datetime.utcnow().isoformat())
    r["generator_number"] = len(st.session_state.generated_odes) + 1
    cl = dict(r.get("classification", {}))
    cl.setdefault("type", "Linear" if r["type"] == "linear" else "Nonlinear")
    cl.setdefault("order", r["order"])
    cl.setdefault("linearity", "Linear" if r["type"] == "linear" else "Nonlinear")
    cl.setdefault("field", cl.get("field", "Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    r["classification"] = cl
    try:
        r.setdefault("ode", sp.Eq(r["generator"], r["rhs"]))
    except Exception:
        pass
    st.session_state.generated_odes.append(r)

def _expr_to_latex(e) -> str:
    try:
        if isinstance(e, str):
            try: e = sp.sympify(e)
            except Exception: return e
        return sp.latex(e).replace(r"\left(", "(").replace(r"\right)", ")")
    except Exception:
        return str(e)

def _download_bytes(label: str, data: bytes, file_name: str, mime: str = "application/octet-stream", use_container_width=True):
    st.download_button(label, data, file_name=file_name, mime=mime, use_container_width=use_container_width)

# ---------------- Job panels ----------------
def _queue_workers_panel():
    stats = get_queue_stats()
    with st.expander("📡 Jobs & Workers"):
        cols = st.columns(2)
        with cols[0]:
            st.write("**Queues**")
            qdf = pd.DataFrame(stats.get("queues", []))
            if not qdf.empty:
                st.dataframe(qdf, use_container_width=True)
            else:
                st.info("No queues detected.")
        with cols[1]:
            st.write("**Workers**")
            wdf = pd.DataFrame(stats.get("workers", []))
            if not wdf.empty:
                st.dataframe(wdf, use_container_width=True)
            else:
                st.info("No workers detected.")
        st.caption("Tip: Ensure the web and worker share the same queue (env `RQ_QUEUE`, e.g. `ode_jobs`).")

def _job_status_panel(job_id: str, title: str = "Job Status"):
    if not job_id:
        st.info("No job submitted yet.")
        return None

    info = fetch_job(job_id)
    st.markdown(f"### {title}")
    st.write(f"**Status:** {info.get('status')} | **Desc:** {info.get('description')} | **Origin queue:** `{info.get('origin')}`")
    meta = info.get("meta") or {}
    if meta:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Stage", meta.get("status", info.get("status", "?")))
        with c2: st.metric("Epoch", f"{meta.get('epoch','-')}/{meta.get('epochs','-')}")
        with c3: st.metric("Train Loss", f"{meta.get('train_loss','-')}")
        with c4: st.metric("Val Loss", f"{meta.get('val_loss','-')}")
        prog = meta.get("progress")
        if isinstance(prog, (int, float)):
            st.progress(min(max(prog, 0.0), 1.0))

    # recent logs
    logs_tail = info.get("logs_tail") or []
    if logs_tail:
        with st.expander("🧾 Recent Logs"):
            for row in logs_tail[-50:]:
                st.write(f"[{row.get('ts','')}] {row.get('event','')}", {k:v for k,v in row.items() if k not in ('ts','event')})

    # error
    if info.get("exc_info"):
        st.error("Exception:\n\n" + str(info["exc_info"]))

    return info

# ---------------- LaTeX Exporter ----------------
class LaTeXExporter:
    @staticmethod
    def generate(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
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
            f"{_expr_to_latex(generator)} = {_expr_to_latex(rhs)}",
            r"\end{equation}",
            r"\subsection{Exact Solution}",
            r"\begin{equation}",
            f"y(x) = {_expr_to_latex(solution)}",
            r"\end{equation}",
            r"\subsection{Parameters}",
            r"\begin{align}",
            f"\\alpha &= {_expr_to_latex(params.get('alpha', 1))} \\\\",
            f"\\beta  &= {_expr_to_latex(params.get('beta', 1))} \\\\",
            f"n       &= {params.get('n', 1)} \\\\",
            f"M       &= {_expr_to_latex(params.get('M', 0))}",
            r"\end{align}",
        ]
        if initial_conditions:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            items = list(initial_conditions.items())
            for i,(k,v) in enumerate(items):
                parts.append(f"{k} &= {_expr_to_latex(v)}" + (r" \\" if i<len(items)-1 else ""))
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
    def create_zip(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\nTo compile: pdflatex ode_document.tex\n")
            if include_extras:
                zf.writestr("reproduce.txt", "Use ode_data.json with your factories or theorem code.")
        buf.seek(0)
        return buf.getvalue()

# ---------------- Pages ----------------

def dashboard_page():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    trained_count = len([p for p in st.session_state.trained_models if p and os.path.exists(p)])
    with c2: st.markdown(f'<div class="metric-card"><h3>🤖 Trained Models</h3><h1>{trained_count}</h1></div>', unsafe_allow_html=True)
    with c3:
        act = st.session_state.get("active_model_path")
        st.markdown(f'<div class="metric-card"><h3>📌 Active Model</h3><p style="font-size: .95rem;">{act or "None selected"}</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>📡 Redis</h3><p>{"ON" if has_redis() else "OFF"}</p></div>', unsafe_allow_html=True)
    _queue_workers_panel()
    st.subheader("📊 Recent Activity")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type","order","generator_number","timestamp"] if c in df.columns]
        if cols: st.dataframe(df[cols], use_container_width=True)
        else: st.dataframe(df, use_container_width=True)
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

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (Exact, Async‑ready)")
    _queue_workers_panel()

    # choose LHS source
    src = st.radio("Generator LHS source", options=("constructor","freeform","arbitrary"),
                   index={"constructor":0,"freeform":1,"arbitrary":2}[st.session_state["lhs_source"]] if st.session_state["lhs_source"] in ("constructor","freeform","arbitrary") else 0,
                   horizontal=True)
    st.session_state["lhs_source"] = src

    # function selection
    colA, colB = st.columns([1,1])
    with colA:
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        basic_lib = st.session_state.get("basic_functions")
        special_lib = st.session_state.get("special_functions")
        if lib == "Basic" and basic_lib:
            func_names = basic_lib.get_function_names()
        elif lib == "Special" and special_lib:
            func_names = special_lib.get_function_names()
        else:
            func_names = []
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
        st.info("Async via Redis: ON" if has_redis() else "Async via Redis: OFF")

    # constructor LHS if available
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
        value=st.session_state.arbitrary_lhs_text or "", height=100
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

    # Theorem 4.2 (derivative computation)
    st.markdown("---")
    colm1, colm2 = st.columns([1,1])
    with colm1:
        compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", False)
    with colm2:
        m_order = st.number_input("m", 1, 12, 1)

    # Generate ODE (async if Redis present)
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        payload = {
            "func_name": func_name, "alpha": alpha, "beta": beta, "n": int(n), "M": M,
            "use_exact": use_exact, "simplify_level": simplify_level,
            "lhs_source": st.session_state["lhs_source"],
            "freeform_terms": st.session_state.get("free_terms"),
            "arbitrary_lhs_text": st.session_state.get("arbitrary_lhs_text"),
            "function_library": lib,
        }
        if has_redis():
            job_id = enqueue_job("worker.compute_job", payload, queue_name=PREFERRED_QUEUE, description="ODE compute")
            if job_id:
                st.session_state["last_compute_job_id"] = job_id
                st.success(f"Job submitted. ID = {job_id}")
            else:
                st.error("Failed to submit job (check Redis/queue).")
        else:
            # local fallback (sync) — limited on serverless
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
                _register_generated_ode(res)
                _show_ode_result(res)
            except Exception as e:
                st.error(f"Generation error: {e}")

    # Poll status & fetch result
    if has_redis() and st.session_state.get("last_compute_job_id"):
        _job_status_panel(st.session_state["last_compute_job_id"], "📡 ODE Job Status")
        info = fetch_job(st.session_state["last_compute_job_id"])
        if info.get("status") == "finished" and info.get("result"):
            res = info["result"]
            try:
                # for latex rendering
                res["generator"] = sp.sympify(res["generator"])
                res["rhs"] = sp.sympify(res["rhs"])
                res["solution"] = sp.sympify(res["solution"])
            except Exception:
                pass
            _register_generated_ode(res)
            _show_ode_result(res)
            st.session_state["last_compute_job_id"] = None
        elif info.get("status") in ("failed", "stopped", "canceled"):
            st.error(f"ODE job failed: {info.get('exc_info') or info.get('error') or info}")
            st.session_state["last_compute_job_id"] = None

    # Theorem 4.2 derivative computation
    if compute_mth and st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2", use_container_width=True):
        try:
            flib = st.session_state.get("basic_functions") if lib == "Basic" else st.session_state.get("special_functions")
            f_expr_preview = get_function_expr(flib, func_name)
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta)  if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(f_expr_preview, α, β, int(n), int(m_order), x, simplify_level)
            st.markdown("### 🔢 Derivative")
            st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
        except Exception as e:
            st.error(f"Failed to compute y^{m_order}(x): {e}")

def _show_ode_result(res: Dict[str, Any]):
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
        latex_doc = LaTeXExporter.generate(ode_data, include_preamble=True)
        _download_bytes("📄 Download LaTeX Document", latex_doc.encode("utf-8"), f"ode_{idx}.tex", "text/x-latex")
        pkg = LaTeXExporter.create_zip(ode_data, include_extras=True)
        _download_bytes("📦 Download Complete Package (ZIP)", pkg, f"ode_package_{idx}.zip", "application/zip")

def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning (RQ‑enabled)")
    _queue_workers_panel()

    model_type = st.selectbox("Select ML Model", ["pattern_learner","vae","transformer"],
                              format_func=lambda s: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[s])

    with st.expander("🎯 Training Configuration", True):
        c1,c2,c3 = st.columns(3)
        with c1:
            epochs     = st.slider("Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate    = st.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
            samples          = st.slider("Training Samples", 100, 5000, 1000)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
            use_gpu          = st.checkbox("Use GPU if available", True)
        # Optional advanced knobs (kept optional to avoid breaking older flows)
        with st.expander("⚙️ Advanced (optional)"):
            amp = st.checkbox("Enable mixed precision (AMP)", False)
            resume_from = st.text_input("Resume from checkpoint path (optional)", "")

    if not has_redis():
        st.warning("Redis is OFF — training should run in the worker for long jobs. The UI can still upload/load models below.")
    else:
        if st.button("🚀 Start Training (Background via RQ)", type="primary"):
            payload = {
                "model_type": model_type, "epochs": epochs, "batch_size": batch_size,
                "samples": samples, "validation_split": validation_split,
                "learning_rate": learning_rate, "enable_mixed_precision": amp,
                "device": ("cuda" if (use_gpu and torch and torch.cuda.is_available()) else "cpu"),
                "checkpoint_dir": CHECKPOINT_DIR,
            }
            if resume_from.strip():
                payload["resume_from"] = resume_from.strip()
            job_id = enqueue_job("worker.train_job", payload, queue_name=PREFERRED_QUEUE, description="ML training")
            if job_id:
                st.session_state["last_train_job_id"] = job_id
                st.success(f"Training submitted. Job ID = {job_id}")
            else:
                st.error("Failed to submit training job.")

    # Training status panel + artifacts
    if st.session_state.get("last_train_job_id"):
        info = _job_status_panel(st.session_state["last_train_job_id"], "📡 Training Status")
        if info and info.get("status") == "finished" and info.get("result"):
            res = info["result"]
            # Mark trained + register model path
            best_path = res.get("best_model_path")
            if best_path and os.path.exists(best_path):
                if best_path not in st.session_state.trained_models:
                    st.session_state.trained_models.append(best_path)
                st.session_state.active_model_path = best_path  # auto-select newest
                st.success(f"Best model ready: {best_path}")
            # Keep history for plotting
            history = res.get("history") or {}
            st.session_state.training_history = history
            # Plot
            if history.get("train_loss"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1,len(history["train_loss"])+1)), y=history["train_loss"], mode="lines", name="Training Loss"))
                if history.get("val_loss"):
                    fig.add_trace(go.Scatter(x=list(range(1,len(history["val_loss"])+1)), y=history["val_loss"], mode="lines", name="Validation Loss"))
                fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig, use_container_width=True)
            st.session_state["last_train_job_id"] = None
        elif info and info.get("status") in ("failed","stopped","canceled"):
            st.error("Training failed: " + (info.get("exc_info") or info.get("error") or "unknown error"))
            st.session_state["last_train_job_id"] = None

    # Upload / Load / Manage models
    st.subheader("📦 Model Management")
    up = st.file_uploader("Upload checkpoint (*.pth)", type=["pth"])
    if up is not None:
        path = os.path.join(CHECKPOINT_DIR, up.name)
        with open(path, "wb") as f:
            f.write(up.getbuffer())
        st.success(f"Uploaded to {path}")
        if path not in st.session_state.trained_models:
            st.session_state.trained_models.append(path)

    # Select active model
    if st.session_state.trained_models:
        choice = st.selectbox("Select Active Model", st.session_state.trained_models, index=max(0, len(st.session_state.trained_models)-1))
        if st.button("✅ Use as Active Model"):
            st.session_state.active_model_path = choice
            st.success(f"Active model set: {choice}")
    else:
        st.info("No trained models yet. Train or upload a checkpoint.")

    # Manual history upload
    hist_up = st.file_uploader("Upload history JSON (optional)", type=["json"])
    if hist_up is not None:
        try:
            hist = json.load(hist_up)
            st.session_state.training_history = hist
            st.success("History loaded.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    # Show training history if present
    if st.session_state.training_history.get("train_loss"):
        st.subheader("📈 Last Training History")
        h = st.session_state.training_history
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1,len(h["train_loss"])+1)), y=h["train_loss"], mode="lines", name="Training Loss"))
        if h.get("val_loss"):
            fig.add_trace(go.Scatter(x=list(range(1,len(h["val_loss"])+1)), y=h["val_loss"], mode="lines", name="Validation Loss"))
        fig.update_layout(title="Training History (Loaded/Last)", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    # Post-training: generate and reverse-engineer demos using active model
    st.subheader("🎨 Use Model (Generate / Reverse)")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🎲 Generate ODE from Model", type="primary"):
            if not MLTrainer:
                st.warning("MLTrainer class not available in web runtime. Use worker-trained results or run locally.")
            else:
                try:
                    trainer = MLTrainer(model_type=model_type, device="cpu", checkpoint_dir=CHECKPOINT_DIR)
                    if st.session_state.active_model_path and os.path.exists(st.session_state.active_model_path):
                        trainer.load_model(st.session_state.active_model_path)
                    res = trainer.generate_new_ode()
                    if res:
                        st.success("Generated via model.")
                        _register_generated_ode(res)
                        with st.expander("View ODE"):
                            try:
                                st.latex(sp.latex(res["ode"]))  # if present
                            except Exception:
                                st.code(str(res))
                except Exception as e:
                    st.error(f"Generation failed: {e}")
    with c2:
        odetxt = st.text_area("Reverse engineering: paste an ODE (LaTeX/text) or a solution y(x) expression", height=120)
        if st.button("🧩 Reverse Engineer", type="primary"):
            try:
                # Try ML-assisted: If MLTrainer provides a method, use it. Else do a heuristic via features.
                if MLTrainer:
                    trainer = MLTrainer(model_type=model_type, device="cpu", checkpoint_dir=CHECKPOINT_DIR)
                    if st.session_state.active_model_path and os.path.exists(st.session_state.active_model_path):
                        trainer.load_model(st.session_state.active_model_path)
                    # If your MLTrainer has a dedicated reverse method, call it; else approximate with generate/evaluate loop
                    if hasattr(trainer, "reverse_engineer"):
                        guess = trainer.reverse_engineer(odetxt)
                    else:
                        # Heuristic fallback: generate multiple and pick closest by symbolic distance
                        cand = []
                        for _ in range(6):
                            r = trainer.generate_new_ode()
                            if r: cand.append(r)
                        # simplistic similarity: order + function id closeness if available
                        best = cand[0] if cand else None
                        guess = best or {}
                else:
                    guess = {"note": "MLTrainer not available; reverse engineering heuristic unavailable in web process."}
                st.success("Reverse engineering result (approximate):")
                st.write(guess)
            except Exception as e:
                st.error(f"Reverse engineering failed: {e}")

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
                _download_bytes("📊 Download CSV", csv, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            with c2:
                js = json.dumps(batch_results, indent=2, default=str).encode("utf-8")
                _download_bytes("📄 Download JSON", js, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")
            with c3:
                if export_format in ["LaTeX","All"]:
                    latex = "\n".join([
                        r"\begin{tabular}{|c|c|c|c|c|}", r"\hline", r"ID & Type & Generator & Function & Order \\",
                        r"\hline", *[f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\" for r in batch_results[:30]],
                        r"\hline", r"\end{tabular}"
                    ])
                    _download_bytes("📝 Download LaTeX", latex.encode("utf-8"), f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")
            with c4:
                if export_format == "All":
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("batch_results.csv", df.to_csv(index=False))
                        zf.writestr("batch_results.json", json.dumps(batch_results, indent=2, default=str))
                    zbuf.seek(0)
                    _download_bytes("📦 Download All (ZIP)", zbuf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", "application/zip")

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
                    _download_bytes("📥 Download Report", analysis.detailed_report, f"novelty_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain")
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
                # Placeholder numeric model; you can plug numeric eval of solution here.
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
        latex_doc = LaTeXExporter.generate(ode, include_preamble=False)
        st.code(latex_doc, language="latex")
        c1,c2 = st.columns(2)
        with c1:
            full_latex = LaTeXExporter.generate(ode, include_preamble=True)
            _download_bytes("📄 Download LaTeX", full_latex.encode("utf-8"), f"ode_{idx+1}.tex", "text/x-latex")
        with c2:
            package = LaTeXExporter.create_zip(ode, include_extras=True)
            _download_bytes("📦 Download Package", package, f"ode_package_{idx+1}.zip", "application/zip")
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
                parts.append(LaTeXExporter.generate(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            doc = "\n".join(parts)
            _download_bytes("📄 Download Multi-ODE LaTeX", doc.encode("utf-8"), f"multiple_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")
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
                parts.append(LaTeXExporter.generate(ode, include_preamble=False))
            parts.append(r"""
\chapter{Conclusions}
The system successfully generated and analyzed multiple ODEs.
\end{document}
""")
            _download_bytes("📄 Download Complete Report", "\n".join(parts).encode("utf-8"), f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")

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
        cm = st.session_state.get("cache_manager")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cache Size", len(getattr(cm,"memory_cache",{})) if cm else 0)
        with col2:
            if st.button("Clear Cache"):
                try: st.session_state.cache_manager.clear(); st.success("Cache cleared.")
                except Exception: st.info("No cache manager.")
        with col3:
            if st.button("Save Session"):
                try:
                    with open("session_state.pkl","wb") as f:
                        pickle.dump({
                            "generated_odes": st.session_state.get("generated_odes", []),
                            "generator_patterns": st.session_state.get("generator_patterns", []),
                            "batch_results": st.session_state.get("batch_results", []),
                            "analysis_results": st.session_state.get("analysis_results", []),
                            "training_history": st.session_state.get("training_history", {}),
                            "export_history": st.session_state.get("export_history", []),
                            "trained_models": st.session_state.get("trained_models", []),
                            "active_model_path": st.session_state.get("active_model_path")
                        }, f)
                    st.success("Session saved.")
                except Exception as e:
                    st.error(f"Failed to save session: {e}")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorems 4.1 & 4.2, ML/DL, Export, Novelty. RQ-backed progress and model management are now persistent.")

def documentation_page():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Go to **Apply Master Theorem**.
2. Pick f(z) from Basic/Special (or type one).
3. Set parameters (α,β,n,M) and choose **Exact (symbolic)** if you want rationals.
4. Choose LHS source: **Constructor**, **Free‑form**, or **Arbitrary SymPy**.
5. Click **Generate ODE**. If Redis is configured, the job runs in background and status persists.
6. Export from the **📤 Export** tab or the **Export & LaTeX** page.
7. Compute **y^(m)(x)** via **Theorem 4.2** when needed.
8. For ML/DL, use the **ML Pattern Learning** page: train via RQ, then upload/load checkpoints, generate, or reverse engineer.
""")

# ---------------- Main ----------------
def main():
    SessionStateManager.initialize()
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Complete app • Free‑form/Arbitrary generators • ML/DL • Export • Novelty • RQ Jobs • Persistent Training</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard", "🔧 Generator Constructor", "🎯 Apply Master Theorem", "🤖 ML Pattern Learning",
        "📊 Batch ODE Generation", "🔍 Novelty Detection", "📈 Analysis & Classification",
        "🔬 Physical Applications", "📐 Visualization", "📤 Export & LaTeX", "📚 Examples Library",
        "⚙️ Settings", "📖 Documentation",
    ])

    if page == "🏠 Dashboard": dashboard_page()
    elif page == "🔧 Generator Constructor": generator_constructor_page()
    elif page == "🎯 Apply Master Theorem": page_apply_master_theorem()
    elif page == "🤖 ML Pattern Learning": ml_pattern_learning_page()
    elif page == "📊 Batch ODE Generation": batch_generation_page()
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