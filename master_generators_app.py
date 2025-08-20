# master_generators_app.py
# ============================================================
# Master Generators for ODEs — Refactored App (RQ-stable)
# ============================================================
# Key improvements:
# - Uses rq_utils.enqueue_job(..., timeout=...) (no job_timeout) — compatible with older RQ
# - Async→Sync fallback: never stuck if job stays queued (local compute button + automatic local fallback on enqueue failure)
# - Persistent Training Monitor: shows job meta/logs/progress until completion; survives UI refresh
# - Training session management: save/load session, upload trained models, resume training
# - Trained flag fixed: Dashboard reflects trained model count; artifacts/best-path tracked
# - All original services preserved: Constructor, Apply Master Theorem (constructor/freeform/arbitrary),
#   Batch Generation, ML Pattern Learning (RQ & local), Novelty, Analysis & Classification,
#   Physical Applications, Visualization, Export (LaTeX/ZIP), Settings, Documentation.
#
# Externals expected:
# - shared.ode_core: ComputeParams, compute_ode_full, theorem_4_2_y_m_expr, get_function_expr, to_exact
# - rq_utils: has_redis, enqueue_job, fetch_job, redis_status
# - src/* modules as in your project (optional; guarded with try/except)
#
# NOTE: This app does not modify your worker. It only calls "worker.compute_job" and "worker.train_job".
#       Ensure your worker implements those and updates job.meta["progress"] and meta["logs"] for visibility.
# ============================================================

import os, sys, io, json, time, base64, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import importlib

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# ---------------- path setup ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------- import src libs (guarded) ----------------
HAVE_SRC = True
try:
    from src.generators.master_generator import (
        MasterGenerator, EnhancedMasterGenerator, CompleteMasterGenerator,
    )
except Exception:
    MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None

try:
    from src.generators.linear_generators import (
        LinearGeneratorFactory, CompleteLinearGeneratorFactory
    )
except Exception:
    LinearGeneratorFactory = CompleteLinearGeneratorFactory = None

try:
    from src.generators.nonlinear_generators import (
        NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
    )
except Exception:
    NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None

try:
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType
    )
except Exception:
    GeneratorConstructor = GeneratorSpecification = None
    DerivativeTerm = DerivativeType = OperatorType = None

try:
    from src.generators.master_theorem import (
        MasterTheoremSolver, MasterTheoremParameters, ExtendedMasterTheorem
    )
except Exception:
    MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None

try:
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
except Exception:
    ODEClassifier = PhysicalApplication = None

try:
    from src.functions.basic_functions import BasicFunctions
except Exception:
    BasicFunctions = None

try:
    from src.functions.special_functions import SpecialFunctions
except Exception:
    SpecialFunctions = None

# ML (optional on web app; typically done in worker)
try:
    from src.ml.pattern_learner import (
        GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer, create_model
    )
except Exception:
    GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = create_model = None

try:
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
except Exception:
    MLTrainer = ODEDataset = ODEDataGenerator = None

try:
    from src.ml.generator_learner import (
        GeneratorPattern, GeneratorPatternNetwork, GeneratorLearningSystem
    )
except Exception:
    GeneratorPattern = GeneratorPatternNetwork = GeneratorLearningSystem = None

try:
    from src.dl.novelty_detector import (
        ODENoveltyDetector, NoveltyAnalysis, ODETokenizer, ODETransformer
    )
except Exception:
    ODENoveltyDetector = NoveltyAnalysis = ODETokenizer = ODETransformer = None

try:
    from src.utils.config import Settings, AppConfig
except Exception:
    Settings = AppConfig = None

try:
    from src.utils.cache import CacheManager, cached
except Exception:
    CacheManager = cached = None

try:
    from src.utils.validators import ParameterValidator
except Exception:
    ParameterValidator = None

try:
    from src.ui.components import UIComponents
except Exception:
    UIComponents = None

# ---------------- core & RQ utils ----------------
from shared.ode_core import (
    ComputeParams, compute_ode_full, theorem_4_2_y_m_expr,
    get_function_expr, to_exact
)
from rq_utils import has_redis, enqueue_job, fetch_job, redis_status

# ---------------- optional torch just for device availability ----------------
try:
    import torch
except Exception:
    torch = None

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators ODE System",
    page_icon="🔬", layout="wide", initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
padding:1.4rem;border-radius:14px;margin-bottom:1rem;color:white;text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.2);}
.main-title{font-size:2rem;font-weight:700;margin-bottom:.25rem;}
.subtitle{font-size:0.95rem;opacity:.95;}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
color:white;padding:1rem;border-radius:12px;text-align:center;
box-shadow:0 10px 20px rgba(0,0,0,0.2);}
.info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:.75rem 0;}
.result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
border:2px solid #4caf50;padding:1rem;border-radius:10px;margin:.75rem 0;}
.error-box{background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);
border:2px solid #f44336;padding:1rem;border-radius:10px;margin:.75rem 0;}
.codebox{font-family:monospace;background:#111;color:#eee;padding:.75rem;border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ---------------- helpers: session ----------------
def _ss_init():
    defaults = dict(
        generator_constructor=None,
        generator_terms=[],
        current_generator=None,
        generated_odes=[],
        generator_patterns=[],
        ml_trainer=None,
        ml_trained=False,
        training_history={},
        batch_results=[],
        analysis_results=[],
        export_history=[],
        lhs_source="constructor",
        free_terms=[],
        arbitrary_lhs_text="",
        # job tracking
        last_job_id=None,
        _last_payload=None,
        train_job_id=None,
        train_last_info=None,
        # libs
        basic_functions=None,
        special_functions=None,
        novelty_detector=None,
        ode_classifier=None,
        cache_manager=None,
        # model artifacts persisted in web service
        model_artifact_b64=None,   # bytes of .pth (optional)
        model_loaded=False,
        model_meta={},
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # heavy optional objects
    if st.session_state.basic_functions is None and BasicFunctions:
        try: st.session_state.basic_functions = BasicFunctions()
        except Exception: pass
    if st.session_state.special_functions is None and SpecialFunctions:
        try: st.session_state.special_functions = SpecialFunctions()
        except Exception: pass
    if st.session_state.novelty_detector is None and ODENoveltyDetector:
        try: st.session_state.novelty_detector = ODENoveltyDetector()
        except Exception: pass
    if st.session_state.ode_classifier is None and ODEClassifier:
        try: st.session_state.ode_classifier = ODEClassifier()
        except Exception: pass
    if st.session_state.cache_manager is None and CacheManager:
        try: st.session_state.cache_manager = CacheManager()
        except Exception: pass
    if st.session_state.generator_constructor is None and GeneratorConstructor:
        try: st.session_state.generator_constructor = GeneratorConstructor()
        except Exception: pass

_ss_init()

# ---------------- helpers: LaTeX Export ----------------
class LaTeXExporter:
    @staticmethod
    def _to_latex(expr) -> str:
        if expr is None:
            return ""
        try:
            if isinstance(expr, str):
                try: expr = sp.sympify(expr)
                except Exception: return expr
            return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
        except Exception:
            return str(expr)

    @staticmethod
    def document(ode: Dict[str, Any], include_preamble=True) -> str:
        gen = ode.get("generator","")
        rhs = ode.get("rhs","")
        sol = ode.get("solution","")
        params = ode.get("parameters", {})
        cls = ode.get("classification", {})
        ics = ode.get("initial_conditions", {})

        parts = []
        if include_preamble:
            parts.append(r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{margin=1in}
\title{Master Generators ODE}
\author{Auto-generated}
\date{\today}
\begin{document}
\maketitle
""")
        parts += [
            r"\section{Equation}",
            r"\begin{equation}",
            f"{LaTeXExporter._to_latex(gen)} = {LaTeXExporter._to_latex(rhs)}",
            r"\end{equation}",
            r"\section{Solution}",
            r"\begin{equation}",
            f"y(x) = {LaTeXExporter._to_latex(sol)}",
            r"\end{equation}",
            r"\section{Parameters}",
            r"\begin{align}",
            f"\alpha &= {LaTeXExporter._to_latex(params.get('alpha',1))} \\\\",
            f"\beta  &= {LaTeXExporter._to_latex(params.get('beta',1))} \\\\",
            f"n      &= {params.get('n',1)} \\\\",
            f"M      &= {LaTeXExporter._to_latex(params.get('M',0))}",
            r"\end{align}",
        ]
        if ics:
            parts += [r"\section{Initial Conditions}", r"\begin{align}"]
            items = list(ics.items())
            for i,(k,v) in enumerate(items):
                parts.append(f"{k} &= {LaTeXExporter._to_latex(v)}" + (r" \\" if i<len(items)-1 else ""))
            parts.append(r"\end{align}")
        if cls:
            parts += [r"\section{Classification}", r"\begin{itemize}"]
            parts.append(rf"\item Type: {cls.get('type','Unknown')}")
            parts.append(rf"\item Order: {cls.get('order','Unknown')}")
            parts.append(rf"\item Linearity: {cls.get('linearity','Unknown')}")
            if "field" in cls: parts.append(rf"\item Field: {cls['field']}")
            if "applications" in cls: parts.append(rf"\item Applications: {', '.join(cls['applications'][:5])}")
            parts.append(r"\end{itemize}")
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def make_zip(ode: Dict[str, Any], include_extras=True) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.document(ode, True))
            zf.writestr("ode.json", json.dumps(ode, indent=2, default=str))
            if include_extras:
                zf.writestr("README.txt", "Compile with: pdflatex ode_document.tex\n")
        buf.seek(0)
        return buf.getvalue()

# ---------------- state helpers ----------------
def register_generated_ode(res: dict):
    r = dict(res)
    r.setdefault("type","nonlinear")
    r.setdefault("order", 0)
    r.setdefault("function_used", str(r.get("function_used","f")))
    r.setdefault("parameters", {})
    r.setdefault("classification", {})
    r.setdefault("timestamp", datetime.now().isoformat())
    r["generator_number"] = len(st.session_state.generated_odes) + 1
    cl = dict(r.get("classification", {}))
    cl.setdefault("type", "Linear" if r["type"]=="linear" else "Nonlinear")
    cl.setdefault("order", r["order"])
    cl.setdefault("linearity", "Linear" if r["type"]=="linear" else "Nonlinear")
    cl.setdefault("field", cl.get("field","Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    r["classification"] = cl
    try:
        r.setdefault("ode", sp.Eq(r["generator"], r["rhs"]))
    except Exception:
        pass
    st.session_state.generated_odes.append(r)

# ---------------- RQ helpers: Async→Sync fallback ----------------
def _compute_sync(func_path: str, payload: dict) -> dict:
    mod_name, func_name = func_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn  = getattr(mod, func_name)
    return fn(payload)

def _try_enqueue_or_sync(func_path: str, payload: dict, description: str = "") -> dict:
    """
    1) If Redis available, enqueue job; return {"mode":"async", "job_id":...}
    2) If enqueue fails or Redis missing, run synchronously and return {"mode":"sync","result":...}
    3) If sync fails, return {"mode":"error","error":...}
    """
    st.session_state["_last_payload"] = payload  # cache for local re-run
    if has_redis():
        job_id = enqueue_job(func_path, payload, description=description)
        if job_id:
            return {"mode":"async", "job_id": job_id}
    # fallback: sync
    try:
        result = _compute_sync(func_path, payload)
        return {"mode":"sync", "result": result}
    except Exception as e:
        return {"mode":"error", "error": str(e)}

# ---------------- UI fragments ----------------
def show_result_ode(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h3>✅ ODE Generated</h3></div>', unsafe_allow_html=True)
    t1,t2,t3 = st.tabs(["📐 Equation", "💡 Solution & ICs", "📤 Export"])
    with t1:
        try:
            st.latex(sp.latex(res["generator"]) + " = " + sp.latex(res["rhs"]))
        except Exception:
            st.code(f"LHS: {res.get('generator')}\nRHS: {res.get('rhs')}")
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t2:
        try:
            st.latex("y(x) = " + sp.latex(res["solution"]))
        except Exception:
            st.code(f"y(x) = {res.get('solution')}")
        if res.get("initial_conditions"):
            st.markdown("**Initial conditions:**")
            for k,v in (res.get("initial_conditions") or {}).items():
                try:
                    st.latex(k + " = " + sp.latex(v))
                except Exception:
                    st.write(k, "=", v)
        st.markdown("**Parameters:**")
        p = res.get("parameters", {})
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        if res.get("f_expr_preview"):
            st.write(f"**f(z):** {res['f_expr_preview']}")
    with t3:
        ode_doc = {
            "generator": res.get("generator"),
            "rhs":       res.get("rhs"),
            "solution":  res.get("solution"),
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
            "generator_number": len(st.session_state.generated_odes),
            "type": res.get("type","nonlinear"),
            "order": res.get("order", 0),
        }
        latex_full = LaTeXExporter.document(ode_doc, include_preamble=True)
        st.download_button("📄 Download LaTeX", latex_full, file_name=f"ode_{len(st.session_state.generated_odes)}.tex", mime="text/x-latex", use_container_width=True)
        zip_pkg = LaTeXExporter.make_zip(ode_doc, include_extras=True)
        st.download_button("📦 Download Package (ZIP)", zip_pkg, file_name=f"ode_pkg_{len(st.session_state.generated_odes)}.zip", mime="application/zip", use_container_width=True)

def job_monitor_block(job_id_key: str, result_handler):
    """
    Generic block to monitor an RQ job stored in st.session_state[job_id_key].
    If finished -> call result_handler(info["result"]) and clear job id.
    If queued -> allow "Run locally now" fallback.
    """
    job_id = st.session_state.get(job_id_key)
    if not job_id:
        return

    st.markdown("### 📡 Job Monitor")
    info = fetch_job(job_id)
    if not info:
        st.warning("Job not found (might have expired).")
        st.session_state[job_id_key] = None
        return

    st.json({k: info.get(k) for k in ["id","status","origin","enqueued_at","started_at","ended_at","meta"]})
    logs = info.get("logs", [])
    if logs:
        with st.expander("🗒️ Worker logs", expanded=False):
            for line in logs[-200:]:
                st.text(line)

    status = info.get("status")
    meta = info.get("meta") or {}
    progress = meta.get("progress") or {}

    if status == "queued":
        st.info("Job is queued. If no worker is listening on this queue, it will not run.")
        if st.button("⚡ Run locally now (fallback)"):
            payload = st.session_state.get("_last_payload")
            if payload:
                try:
                    local = _compute_sync("worker.compute_job" if "ode" in result_handler.__name__ else "worker.train_job", payload)
                    result_handler(local)
                    st.session_state[job_id_key] = None
                except Exception as e:
                    st.error(f"Local run failed: {e}")
            else:
                st.error("No cached payload. Re-submit the task.")
    elif status == "failed":
        st.error("Job failed.")
        if "exc_info" in info:
            with st.expander("Traceback"):
                st.code(info["exc_info"])
        if progress.get("error"):
            st.error(progress.get("error"))
        # clear id
        st.session_state[job_id_key] = None
    elif status == "finished":
        st.success("Job finished.")
        result_handler(info.get("result"))
        st.session_state[job_id_key] = None
    else:
        st.info(f"⏳ Status: {status}. Refresh the page to update.")

# ---------------- Pages ----------------
def page_dashboard():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>🤖 ML Models</h3><h1>{"1" if st.session_state.get("ml_trained") else "0"}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>📊 Batch Rows</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c4:
        rqs = redis_status()
        ok = rqs.get("ok")
        label = "ON" if ok else "OFF"
        st.markdown(f'<div class="metric-card"><h3>🔌 Redis</h3><h1>{label}</h1></div>', unsafe_allow_html=True)
        if ok:
            st.caption(f"Queue: {rqs.get('queue')} | Workers: {', '.join(rqs.get('workers',[])) or '—'}")

    if st.session_state.generated_odes:
        st.subheader("Recent ODEs")
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type","order","generator_number","timestamp","function_used"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs yet. Try **🎯 Apply Master Theorem**.")

def page_constructor():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Build a generator LHS using terms; or use the Free‑form/Arbitrary editors in the theorem page.</div>', unsafe_allow_html=True)
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found in src/.")
        return

    with st.expander("➕ Add Term", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            dorder = st.selectbox("Derivative Order", [0,1,2,3,4,5], index=0,
                                  format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x,f"y⁽{x}⁾"))
        with c2:
            ftype = st.selectbox("Function Type", [t.value for t in DerivativeType], index=0,
                                 format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coef = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)

        c5,c6,c7 = st.columns(3)
        with c5:
            optype = st.selectbox("Operator Type", [t.value for t in OperatorType], index=0,
                                  format_func=lambda s: s.replace("_"," ").title())
        with c6:
            scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if optype in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if optype in ["delay","advance"] else None

        if st.button("Add Term", type="primary"):
            try:
                term = DerivativeTerm(
                    derivative_order=int(dorder),
                    coefficient=float(coef),
                    power=int(power),
                    function_type=DerivativeType(ftype),
                    operator_type=OperatorType(optype),
                    scaling=scaling,
                    shift=shift
                )
                st.session_state.generator_terms.append(term)
                st.success("Term added.")
            except Exception as e:
                st.error(f"Failed to add term: {e}")

    if st.session_state.generator_terms:
        st.subheader("Current Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            c1,c2 = st.columns([8,1])
            with c1:
                desc = term.get_description() if hasattr(term,"get_description") else str(term)
                st.write(f"• {desc}")
            with c2:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()

        if st.button("Build Generator Spec", type="primary"):
            try:
                spec = GeneratorSpecification(
                    terms=st.session_state.generator_terms,
                    name=f"Custom Generator {len(st.session_state.generated_odes)+1}"
                )
                st.session_state.current_generator = spec
                st.success("Specification created.")
                try: st.latex(sp.latex(spec.lhs) + " = RHS")
                except Exception: pass
            except Exception as e:
                st.error(f"Failed to build spec: {e}")

        if st.button("Clear Terms"):
            st.session_state.generator_terms = []
            st.session_state.current_generator = None

def page_apply_theorem():
    st.header("🎯 Apply Master Theorem (Exact, Async‑ready)")

    # Function library
    colA, colB = st.columns([1,1])
    with colA:
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        if lib == "Basic" and st.session_state.basic_functions:
            fnames = st.session_state.basic_functions.get_function_names()
        elif lib == "Special" and st.session_state.special_functions:
            fnames = st.session_state.special_functions.get_function_names()
        else:
            fnames = []
        func_name = st.selectbox("Select f(z)", fnames) if fnames else st.text_input("Enter f(z)", "exp(z)")

    # Parameters
    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c5,c6,c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7:
        st.info("Redis: ON" if has_redis() else "Redis: OFF")

    # LHS source
    st.session_state.lhs_source = st.radio("LHS source", ["constructor","freeform","arbitrary"],
                                           index=["constructor","freeform","arbitrary"].index(st.session_state.lhs_source),
                                           horizontal=True)

    # Freeform terms editor
    with st.expander("🧩 Free‑form LHS builder", expanded=False):
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)",
                    ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs",
                     "asin","acos","atan","asinh","acosh","atanh","erf","erfc"], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_order = st.number_input("outer m", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale a", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift b", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add free‑form term"):
                st.session_state.free_terms.append({
                    "coef": float(coef),
                    "inner_order": int(inner_order),
                    "wrapper": wrapper,
                    "power": int(power),
                    "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
                st.success("Term added.")
        if st.session_state.free_terms:
            st.write("**Current terms:**")
            for i,t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. {t}")
            cfa, cfb = st.columns(2)
            with cfa:
                if st.button("Use Free‑form LHS"):
                    st.session_state.lhs_source = "freeform"
            with cfb:
                if st.button("Clear terms"):
                    st.session_state.free_terms = []

    # Arbitrary SymPy expression
    st.subheader("✍️ Arbitrary LHS (SymPy)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Expression in x and y(x) (e.g., sin(y(x)) + y(x)*y(x).diff(x))",
        value=st.session_state.arbitrary_lhs_text or "",
        height=100
    )

    # Constructor LHS preview (if present)
    constructor_lhs = None
    if st.session_state.current_generator is not None and hasattr(st.session_state.current_generator, "lhs"):
        constructor_lhs = st.session_state.current_generator.lhs
        with st.expander("🔎 Constructor LHS"):
            try: st.latex(sp.latex(constructor_lhs))
            except Exception: st.code(str(constructor_lhs))

    # Theorem 4.2 controls
    st.markdown("---")
    colm1, colm2 = st.columns([1,1])
    with colm1:
        compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", False)
    with colm2:
        m_order = st.number_input("m", 1, 12, 1)

    # Generate ODE (Async→Sync fallback)
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        payload = {
            "func_name": func_name,
            "alpha": float(alpha),
            "beta":  float(beta),
            "n": int(n),
            "M": float(M),
            "use_exact": bool(use_exact),
            "simplify_level": simplify_level,
            "lhs_source": st.session_state.lhs_source,
            "freeform_terms": st.session_state.free_terms,
            "arbitrary_lhs_text": st.session_state.arbitrary_lhs_text,
            "function_library": lib,
            # pass constructor_lhs as string for worker; it will sympify
            "constructor_lhs": str(constructor_lhs) if constructor_lhs is not None else None,
        }
        result = _try_enqueue_or_sync("worker.compute_job", payload, description="ode-generate")
        if result["mode"] == "async":
            st.session_state["last_job_id"] = result["job_id"]
            st.success(f"Job submitted: {result['job_id']}")
        elif result["mode"] == "sync":
            out = result["result"]
            # cast strings to sympy for LaTeX
            try:
                out["generator"] = sp.sympify(out["generator"])
                out["rhs"]       = sp.sympify(out["rhs"])
                out["solution"]  = sp.sympify(out["solution"])
            except Exception:
                pass
            register_generated_ode(out)
            show_result_ode(out)
        else:
            st.error(f"Generation error: {result.get('error')}")

    # Job monitor (if async)
    def _ode_job_handler(res: dict):
        if not isinstance(res, dict):
            st.error("Empty result.")
            return
        # cast to sympy-friendly
        try:
            if "generator" in res: res["generator"] = sp.sympify(res["generator"])
            if "rhs" in res:       res["rhs"]       = sp.sympify(res["rhs"])
            if "solution" in res:  res["solution"]  = sp.sympify(res["solution"])
        except Exception:
            pass
        register_generated_ode(res)
        show_result_ode(res)

    job_monitor_block("last_job_id", _ode_job_handler)

    # Theorem 4.2 compute (synchronous)
    if compute_mth and st.button("🧮 Compute y^(m)(x)", use_container_width=True):
        try:
            lib_obj = st.session_state.basic_functions if lib=="Basic" else st.session_state.special_functions
            f_expr_preview = get_function_expr(lib_obj, func_name)
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta)  if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(f_expr_preview, α, β, int(n), int(m_order), x, simplify_level)
            st.markdown("### 🔢 Derivative")
            st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
        except Exception as e:
            st.error(f"Failed to compute y^{m_order}(x): {e}")

def page_ml_training():
    st.header("🤖 ML / DL — Training & Usage")

    colh1,colh2 = st.columns(2)
    with colh1:
        model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"],
                                  index=0, format_func=lambda s: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[s])
        hidden_dim = st.selectbox("Hidden dim", [32,64,128,256], index=1)
        normalize  = st.checkbox("Normalize inputs", False)
        use_gpu    = st.checkbox("Use GPU if available", True)
    with colh2:
        epochs  = st.slider("Epochs", 5, 500, 100)
        batch   = st.slider("Batch size", 8, 128, 32)
        samples = st.slider("Synthetic samples", 100, 5000, 1000)
        valsp   = st.slider("Validation split", 0.1, 0.4, 0.2)
        use_gen = st.checkbox("Use generator (streaming)", True)
        amp     = st.checkbox("Mixed precision (AMP)", False)

    st.caption("Training runs best on the worker via Redis. If Redis/worker not available, you can run locally (web service) as fallback.")

    # ---- Start training (Async→Sync) ----
    if st.button("🚀 Start Training", type="primary"):
        payload = {
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "normalize": normalize,
            "epochs": int(epochs),
            "batch_size": int(batch),
            "samples": int(samples),
            "validation_split": float(valsp),
            "use_generator": bool(use_gen),
            "enable_mixed_precision": bool(amp),
            # optional device hint; worker may ignore
            "device": "cuda" if (use_gpu and torch and torch.cuda.is_available()) else "cpu",
        }
        result = _try_enqueue_or_sync("worker.train_job", payload, description="ml-train")
        if result["mode"] == "async":
            st.session_state["train_job_id"] = result["job_id"]
            st.success(f"Training submitted: {result['job_id']}")
        elif result["mode"] == "sync":
            _training_result_handler(result["result"])
        else:
            st.error(f"Training error: {result.get('error')}")

    # ---- Monitor training job ----
    def _training_result_handler(res: dict):
        """
        Result is defined by your worker.train_job:
        Expect fields like:
          - status: "ok"
          - history: {...}
          - best_model_b64: optional base64 of .pth
          - best_path: path on worker (not accessible from web container)
          - metrics: {...}
        """
        if not isinstance(res, dict):
            st.error("Empty training result.")
            return
        st.session_state["training_history"] = res.get("history", {})
        st.session_state["ml_trained"] = True
        st.session_state["model_meta"] = {
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "model_type": res.get("model_type"),
            "hidden_dim": res.get("hidden_dim"),
            "normalize": res.get("normalize"),
            "metrics": res.get("metrics", {})
        }
        b64 = res.get("best_model_b64")
        if b64:
            st.session_state["model_artifact_b64"] = b64
            st.session_state["model_loaded"] = True  # mark as loadable in web
        st.success("Training complete. Model ready.")

    job_monitor_block("train_job_id", _training_result_handler)

    # ---- History / Curves ----
    hist = st.session_state.get("training_history") or {}
    if hist:
        st.subheader("📈 Training Curves")
        tr = hist.get("train_loss", [])
        va = hist.get("val_loss", [])
        if tr:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1,len(tr)+1)), y=tr, mode="lines", name="Train"))
            if va:
                fig.add_trace(go.Scatter(x=list(range(1,len(va)+1)), y=va, mode="lines", name="Val"))
            fig.update_layout(title="Loss History", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No curve data recorded.")

    # ---- Session save/load & artifact upload ----
    st.subheader("💾 Session & Model Management")
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Save Session"):
            try:
                blob = pickle.dumps({
                    "generated_odes": st.session_state.generated_odes,
                    "training_history": st.session_state.training_history,
                    "ml_trained": st.session_state.ml_trained,
                    "model_artifact_b64": st.session_state.model_artifact_b64,
                    "model_meta": st.session_state.model_meta
                })
                st.download_button("⬇️ Download Session", blob, file_name=f"session_{int(time.time())}.pkl", mime="application/octet-stream")
            except Exception as e:
                st.error(f"Save failed: {e}")
    with c2:
        upl = st.file_uploader("Upload Session (.pkl)", type=["pkl"])
        if upl:
            try:
                data = pickle.loads(upl.read())
                for k in ["generated_odes","training_history","ml_trained","model_artifact_b64","model_meta"]:
                    if k in data:
                        st.session_state[k] = data[k]
                st.success("Session restored.")
            except Exception as e:
                st.error(f"Load failed: {e}")
    with c3:
        model_upl = st.file_uploader("Upload trained model (.pth)", type=["pth"])
        if model_upl:
            try:
                raw = model_upl.read()
                st.session_state["model_artifact_b64"] = base64.b64encode(raw).decode("utf-8")
                st.session_state["ml_trained"] = True
                st.session_state["model_loaded"] = True
                st.success("Model uploaded and available in memory.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # ---- Use trained model: generate & reverse engineer ----
    if st.session_state.get("ml_trained"):
        st.subheader("🎨 Use Model")
        colg, colr = st.columns(2)
        with colg:
            num_gen = st.slider("Generate novel ODEs", 1, 10, 1)
            if st.button("Generate", type="primary"):
                _ml_generate(num=num_gen)
        with colr:
            st.markdown("**Reverse Engineering**")
            st.caption("Give a target set of parameters/features to reconstruct plausible ODE.")
            re_alpha = st.number_input("α*", value=1.0, step=0.1)
            re_beta  = st.number_input("β*", value=1.0, step=0.1)
            re_n     = st.number_input("n*", 1, 12, 2)
            re_M     = st.number_input("M*", value=0.0, step=0.1)
            if st.button("Reverse Engineer"):
                _ml_reverse_engineer(alpha=re_alpha, beta=re_beta, n=int(re_n), M=re_M)

def _get_model_bytes() -> Optional[bytes]:
    b64 = st.session_state.get("model_artifact_b64")
    if not b64:
        return None
    try:
        return base64.b64decode(b64.encode("utf-8"))
    except Exception:
        return None

def _load_trainer_local() -> Optional[Any]:
    """
    Best-effort loader for a local trainer so we can use generate/reverse.
    Works if MLTrainer exists in web image.
    """
    if not MLTrainer:
        st.warning("MLTrainer not available in web image; generation will be limited.")
        return None
    try:
        # Construct with loose defaults; handle older/newer signatures
        kwargs = dict(model_type="pattern_learner", device="cuda" if (torch and torch.cuda.is_available()) else "cpu")
        try:
            tr = MLTrainer(**kwargs)  # older signature
        except TypeError:
            # Newer signature may require config
            kwargs["config"] = {"input_dim":12,"hidden_dim":128,"output_dim":12,"learning_rate":0.001}
            tr = MLTrainer(**kwargs)
        # Load model bytes if present
        model_bytes = _get_model_bytes()
        if model_bytes:
            tmp = os.path.join(APP_DIR, "tmp_model.pth")
            with open(tmp, "wb") as f: f.write(model_bytes)
            try:
                tr.load_model(tmp)
                st.session_state["model_loaded"] = True
            except Exception:
                pass
        return tr
    except Exception as e:
        st.error(f"Trainer load failed: {e}")
        return None

def _ml_generate(num: int = 1):
    tr = _load_trainer_local()
    if not tr:
        st.warning("No local trainer available.")
        return
    with st.spinner("Generating..."):
        for i in range(num):
            try:
                res = tr.generate_new_ode()
                if res:
                    st.success(f"Generated #{i+1}")
                    register_generated_ode(res)
            except Exception as e:
                st.error(f"Generation error: {e}")

def _ml_reverse_engineer(alpha: float, beta: float, n: int, M: float):
    """
    Simple reverse engineering: build a seed vector to guide the generator.
    """
    tr = _load_trainer_local()
    if not tr:
        st.warning("No local trainer available.")
        return
    try:
        seed = torch.tensor([[alpha, beta, float(n), M, 0, 1, 3, 2, 0, 0, 0, 0.0]], dtype=torch.float32) if torch else None
        res = tr.generate_new_ode(seed=seed) if seed is not None else tr.generate_new_ode()
        if res:
            st.success("Reverse engineered ODE")
            register_generated_ode(res)
            show_result_ode(res)
    except Exception as e:
        st.error(f"Reverse engineering failed: {e}")

def page_batch():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs via your factories.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        n_odes = st.slider("Number of ODEs", 5, 500, 50)
        gtypes = st.multiselect("Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_cats = st.multiselect("Function categories", ["Basic","Special"], default=["Basic"])
        vary = st.checkbox("Vary parameters", True)
    with c3:
        if vary:
            a_rng = st.slider("α range", -10.0, 10.0, (-2.0, 2.0))
            b_rng = st.slider("β range", 0.1, 10.0, (0.5, 2.0))
            n_rng = st.slider("n range", 1, 5, (1, 3))
        else:
            a_rng=(1.0,1.0); b_rng=(1.0,1.0); n_rng=(1,1)

    with st.expander("Advanced export"):
        export_fmt = st.selectbox("Format", ["JSON","CSV","LaTeX","All"])
        include_sol = st.checkbox("Include solutions", True)
        include_class = st.checkbox("Include classification", True)

    if st.button("Run Batch", type="primary"):
        with st.spinner(f"Generating {n_odes} ODEs..."):
            out = []
            names = []
            if "Basic" in func_cats and st.session_state.basic_functions:
                names += st.session_state.basic_functions.get_function_names()
            if "Special" in func_cats and st.session_state.special_functions:
                names += st.session_state.special_functions.get_function_names()[:20]
            if not names:
                st.warning("No function names found in libraries.")
                return

            for i in range(n_odes):
                try:
                    params = {
                        "alpha": float(np.random.uniform(*a_rng)),
                        "beta":  float(np.random.uniform(*b_rng)),
                        "n": int(np.random.randint(n_rng[0], n_rng[1]+1)),
                        "M": float(np.random.uniform(-1, 1))
                    }
                    fname = np.random.choice(names)
                    gtype = np.random.choice(gtypes)

                    res = {}
                    if gtype == "linear" and CompleteLinearGeneratorFactory:
                        f = CompleteLinearGeneratorFactory()
                        gen_num = np.random.randint(1, 9)
                        if gen_num in [4,5]:
                            params["a"] = float(np.random.uniform(1,3))
                        res = f.create(gen_num, st.session_state.basic_functions.get_function(fname), **params)
                    elif gtype == "nonlinear" and CompleteNonlinearGeneratorFactory:
                        f = CompleteNonlinearGeneratorFactory()
                        gen_num = np.random.randint(1, 11)
                        if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                        if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                        if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                        res = f.create(gen_num, st.session_state.basic_functions.get_function(fname), **params)
                    elif LinearGeneratorFactory and gtype == "linear":
                        f = LinearGeneratorFactory()
                        res = f.create(1, st.session_state.basic_functions.get_function(fname), **params)
                    elif NonlinearGeneratorFactory and gtype == "nonlinear":
                        f = NonlinearGeneratorFactory()
                        res = f.create(1, st.session_state.basic_functions.get_function(fname), **params)
                    else:
                        continue

                    row = {
                        "ID": i+1, "Type": res.get("type","unknown"),
                        "Generator": res.get("generator_number","?"),
                        "Function": fname, "Order": res.get("order",0),
                        "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                    }
                    if include_sol:
                        s = str(res.get("solution",""))
                        row["Solution"] = (s[:120]+"...") if len(s)>120 else s
                    if include_class:
                        row["Subtype"] = res.get("subtype","standard")
                    out.append(row)
                except Exception as e:
                    logger.debug(f"Batch item failed: {e}")

            st.session_state.batch_results.extend(out)
            st.success(f"Generated {len(out)} rows.")
            df = pd.DataFrame(out); st.dataframe(df, use_container_width=True)

            st.subheader("Export")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.download_button("CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"batch_{int(time.time())}.csv", mime="text/csv")
            with c2:
                js = json.dumps(out, indent=2, default=str).encode("utf-8")
                st.download_button("JSON", js, file_name=f"batch_{int(time.time())}.json", mime="application/json")
            with c3:
                if export_fmt in ["LaTeX","All"]:
                    latex = "\n".join([
                        r"\begin{tabular}{|c|c|c|c|c|}",
                        r"\hline", r"ID & Type & Generator & Function & Order \\",
                        r"\hline",
                        *[f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Generator','')} & {r.get('Function','')} & {r.get('Order','')} \\\\" for r in out[:40]],
                        r"\hline", r"\end{tabular}"
                    ])
                    st.download_button("LaTeX", latex, file_name=f"batch_{int(time.time())}.tex", mime="text/x-latex")
            with c4:
                if export_fmt == "All":
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("batch.csv", df.to_csv(index=False))
                        zf.writestr("batch.json", json.dumps(out, indent=2, default=str))
                    buf.seek(0)
                    st.download_button("ZIP", buf.getvalue(), file_name=f"batch_{int(time.time())}.zip", mime="application/zip")

def page_novelty():
    st.header("🔍 Novelty Detection")
    if not st.session_state.novelty_detector:
        st.warning("Novelty detector not available.")
        return

    method = st.radio("Input", ["Use Constructor LHS","Enter ODE","Pick Generated"])
    target = None
    if method == "Use Constructor LHS":
        spec = st.session_state.current_generator
        if spec is not None and hasattr(spec, "lhs"):
            target = {"ode": spec.lhs, "type":"custom", "order": getattr(spec, "order", 2)}
        else:
            st.warning("No constructor LHS.")
    elif method == "Enter ODE":
        s = st.text_area("Enter ODE (LaTeX or text)")
        if s:
            target = {"ode": s, "type":"manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox("Select", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')} (ord {st.session_state.generated_odes[i].get('order',0)})")
            target = st.session_state.generated_odes[idx]

    if target and st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                res = st.session_state.novelty_detector.analyze(target, check_solvability=True, detailed=True)
                st.metric("Novelty", "🟢 NOVEL" if res.is_novel else "🔴 STANDARD")
                st.metric("Score", f"{res.novelty_score:.1f}/100")
                st.metric("Confidence", f"{res.confidence:.1%}")
                with st.expander("Details", True):
                    st.write(f"Complexity: {res.complexity_level}")
                    st.write(f"Solvable by standard methods: {'Yes' if res.solvable_by_standard_methods else 'No'}")
                    if res.special_characteristics:
                        st.write("Special characteristics:")
                        for t in res.special_characteristics: st.write("•", t)
                    if res.recommended_methods:
                        st.write("Recommended methods:")
                        for t in res.recommended_methods[:5]: st.write("•", t)
            except Exception as e:
                st.error(f"Novelty failed: {e}")

def page_analysis():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs yet.")
        return
    if not st.session_state.ode_classifier:
        st.warning("Classifier not available.")
        return

    st.subheader("Overview")
    summary = []
    for i, ode in enumerate(st.session_state.generated_odes[-50:]):
        summary.append({
            "ID": i+1,
            "Type": ode.get("type","?"),
            "Order": ode.get("order",0),
            "Generator": ode.get("generator_number","N/A"),
            "Function": ode.get("function_used","?"),
            "Timestamp": (ode.get("timestamp","") or "")[:19],
        })
    df = pd.DataFrame(summary); st.dataframe(df, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Linear", sum(1 for o in st.session_state.generated_odes if o.get("type")=="linear"))
    with c2: st.metric("Nonlinear", sum(1 for o in st.session_state.generated_odes if o.get("type")=="nonlinear"))
    with c3:
        orders = [o.get("order",0) for o in st.session_state.generated_odes]
        st.metric("Avg Order", f"{(np.mean(orders) if orders else 0):.1f}")
    with c4:
        unique = len(set(o.get("function_used","") for o in st.session_state.generated_odes))
        st.metric("Unique f", unique)

    st.subheader("Distributions")
    c1,c2 = st.columns(2)
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

    st.subheader("🏷️ Classify")
    if st.button("Classify All", type="primary"):
        with st.spinner("Classifying..."):
            fields = []
            for ode in st.session_state.generated_odes:
                try:
                    c = st.session_state.ode_classifier.classify_ode(ode)
                    fields.append(c.get("classification",{}).get("field","Unknown"))
                except Exception:
                    fields.append("Unknown")
            vc = pd.Series(fields).value_counts()
            fig = px.bar(x=vc.index, y=vc.values, title="Fields"); fig.update_layout(xaxis_title="Field", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

def page_physics():
    st.header("🔬 Physical Applications")
    st.markdown('<div class="info-box">Quick examples (illustrative).</div>', unsafe_allow_html=True)
    data = {
        "Mechanics": [
            {"name":"Harmonic Oscillator","equation":"y'' + ω^2 y = 0","description":"Spring-mass"},
            {"name":"Damped Oscillator","equation":"y'' + 2γ y' + ω₀² y = 0","description":"Friction"},
        ],
        "Quantum": [
            {"name":"Schrödinger (1D)","equation":"-ℏ²/(2m) y'' + V(x)y = Ey","description":"Bound states"},
        ],
    }
    cat = st.selectbox("Field", list(data.keys()))
    for app in data.get(cat, []):
        with st.expander(app["name"]):
            try: st.latex(app["equation"])
            except Exception: st.write(app["equation"])
            st.write(app["description"])

def page_visualization():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize.")
        return
    idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')} (ord {st.session_state.generated_odes[i].get('order',0)})")
    ode = st.session_state.generated_odes[idx]
    c1,c2,c3 = st.columns(3)
    with c1: ptype = st.selectbox("Plot", ["Solution","Phase Portrait","3D Surface","Direction Field"])
    with c2: x_rng = st.slider("X range", -10.0, 10.0, (-5.0, 5.0))
    with c3: npts  = st.slider("Points", 100, 2000, 500)
    if st.button("Generate Plot", type="primary"):
        with st.spinner("Plotting..."):
            try:
                x = np.linspace(x_rng[0], x_rng[1], npts)
                y = np.sin(x) * np.exp(-0.1*np.abs(x))  # placeholder; hook numeric solution here if desired
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Solution"))
                fig.update_layout(title=f"ODE {idx+1} — Solution", xaxis_title="x", yaxis_title="y")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot failed: {e}")

def page_export():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.warning("No ODEs.")
        return
    mode = st.radio("Export", ["Single","Multiple","Report"])
    if mode == "Single":
        idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                           format_func=lambda i: f"ODE {i+1}")
        ode = st.session_state.generated_odes[idx]
        st.subheader("LaTeX Preview")
        st.code(LaTeXExporter.document(ode, include_preamble=False), language="latex")
        c1,c2 = st.columns(2)
        with c1:
            full = LaTeXExporter.document(ode, include_preamble=True)
            st.download_button("LaTeX", full, file_name=f"ode_{idx+1}.tex", mime="text/x-latex")
        with c2:
            pkg = LaTeXExporter.make_zip(ode, True)
            st.download_button("ZIP", pkg, file_name=f"ode_package_{idx+1}.zip", mime="application/zip")
    elif mode == "Multiple":
        sel = st.multiselect("Select", range(len(st.session_state.generated_odes)),
                             format_func=lambda i: f"ODE {i+1}")
        if sel and st.button("Build LaTeX"):
            parts = [r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb}
\begin{document}
\title{Generated ODEs}\maketitle
"""]
            for cnt, i in enumerate(sel, 1):
                parts.append(f"\\section*{{ODE {cnt}}}")
                parts.append(LaTeXExporter.document(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            doc = "\n".join(parts)
            st.download_button("Download", doc, file_name=f"odes_{int(time.time())}.tex", mime="text/x-latex")
    else:
        if st.button("Build Report"):
            parts = [r"""\documentclass[12pt]{report}
\usepackage{amsmath,amssymb}
\begin{document}\title{Master Generators — Report}\maketitle\tableofcontents
\chapter{Summary} Auto-generated report.
\chapter{ODEs}
"""]
            for i, ode in enumerate(st.session_state.generated_odes):
                parts.append(f"\\section*{{ODE {i+1}}}")
                parts.append(LaTeXExporter.document(ode, include_preamble=False))
            parts.append(r"\end{document}")
            st.download_button("LaTeX Report", "\n".join(parts), file_name=f"report_{int(time.time())}.tex", mime="text/x-latex")

def page_examples():
    st.header("📚 Examples")
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")

def page_settings():
    st.header("⚙️ Settings & Maintenance")
    tabs = st.tabs(["General","Cache","RQ Status","About"])
    with tabs[0]:
        dark = st.checkbox("Dark mode (UI-only demo)", False)
        st.info("Settings are not persisted by default.")
    with tabs[1]:
        cm = st.session_state.get("cache_manager")
        st.metric("Cache size", len(getattr(cm,"memory_cache",{})) if cm else 0)
        if st.button("Clear Cache"):
            try:
                st.session_state.cache_manager.clear()
                st.success("Cache cleared.")
            except Exception:
                st.info("No cache manager.")
    with tabs[2]:
        rs = redis_status()
        st.json(rs)
        st.caption("Ensure your Worker is started and listening on the same queue.")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorems 4.1 & 4.2, ML/DL, Export, Novelty, RQ async jobs.")

def page_docs():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Go to **🎯 Apply Master Theorem**.
2. Select a function library (Basic/Special), pick *f(z)*, set (α, β, n, M).
3. Choose LHS source: Constructor / Free‑form / Arbitrary SymPy.
4. Click **Generate ODE**.  
   • If Redis/worker is configured, generation runs in background and shows up in the **Job Monitor**.  
   • If not, the app computes locally (synchronous fallback).
5. Export from **📤 Export & LaTeX** or visualize from **📐 Visualization**.
6. Train models in **🤖 ML / DL**. Progress is shown persistently via the **Training Monitor**.
7. Save or upload sessions in the ML page to reuse trained models.
""")

# ---------------- Main ----------------
def main():
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Constructor • Free‑form/Arbitrary LHS • Master Theorem • ML/DL • Novelty • Export • RQ Async</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard",
        "🔧 Generator Constructor",
        "🎯 Apply Master Theorem",
        "🤖 ML / DL",
        "📊 Batch Generation",
        "🔍 Novelty Detection",
        "📈 Analysis & Classification",
        "🔬 Physical Applications",
        "📐 Visualization",
        "📤 Export & LaTeX",
        "📚 Examples",
        "⚙️ Settings",
        "📖 Documentation",
    ])

    if page == "🏠 Dashboard":                    page_dashboard()
    elif page == "🔧 Generator Constructor":       page_constructor()
    elif page == "🎯 Apply Master Theorem":        page_apply_theorem()
    elif page == "🤖 ML / DL":                     page_ml_training()
    elif page == "📊 Batch Generation":            page_batch()
    elif page == "🔍 Novelty Detection":           page_novelty()
    elif page == "📈 Analysis & Classification":   page_analysis()
    elif page == "🔬 Physical Applications":       page_physics()
    elif page == "📐 Visualization":               page_visualization()
    elif page == "📤 Export & LaTeX":              page_export()
    elif page == "📚 Examples":                    page_examples()
    elif page == "⚙️ Settings":                    page_settings()
    elif page == "📖 Documentation":               page_docs()

if __name__ == "__main__":
    main()