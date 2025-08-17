# master_generators_app.py
"""
Master Generators for ODEs — COMPLETE APP (Async + ML/DL + Reverse + Export)
- Corrected queue usage and robust job polling
- Pre-training ODE-count requirement
- Model lifecycle: train/load/upload/generate/reverse
- Keeps every service and page from previous builds
"""

# ---------- stdlib ----------
import os, sys, io, json, time, base64, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- third-party ----------
import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
import plotly.express as px

# optional torch (for local generation after loading model)
try:
    import torch
except Exception:
    torch = None

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# ---------- PATHS ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------- Imports from src (best-effort; app works even if some are missing) ----------
HAVE_SRC = True
try:
    from src.generators.master_generator import (
        MasterGenerator, EnhancedMasterGenerator, CompleteMasterGenerator
    )
except Exception:
    MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
    HAVE_SRC = False

try:
    from src.generators.linear_generators import (
        LinearGeneratorFactory, CompleteLinearGeneratorFactory
    )
except Exception:
    LinearGeneratorFactory = CompleteLinearGeneratorFactory = None
    HAVE_SRC = False

try:
    from src.generators.nonlinear_generators import (
        NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
    )
except Exception:
    NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
    HAVE_SRC = False

try:
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType
    )
except Exception:
    GeneratorConstructor = GeneratorSpecification = None
    DerivativeTerm = DerivativeType = OperatorType = None
    HAVE_SRC = False

try:
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
except Exception:
    ODEClassifier = PhysicalApplication = None
    HAVE_SRC = False

try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
except Exception:
    BasicFunctions = SpecialFunctions = None
    HAVE_SRC = False

try:
    from src.ml.trainer import MLTrainer
except Exception:
    MLTrainer = None
    HAVE_SRC = False

try:
    from src.dl.novelty_detector import (
        ODENoveltyDetector, NoveltyAnalysis, ODETokenizer, ODETransformer
    )
except Exception:
    ODENoveltyDetector = NoveltyAnalysis = ODETokenizer = ODETransformer = None
    HAVE_SRC = False

# ---------- Core math & RQ utils ----------
try:
    from shared.ode_core import (
        ComputeParams, compute_ode_full, theorem_4_2_y_m_expr,
        get_function_expr, to_exact, simplify_expr, expr_to_str
    )
except Exception as e:
    log.error(f"Missing shared.ode_core: {e}")
    # Fail fast: this module is essential for function.
    raise

try:
    from rq_utils import has_redis, enqueue_job, fetch_job, list_runs
except Exception as e:
    log.warning(f"rq_utils import failed ({e}); async jobs will be disabled.")
    def has_redis(): return False
    def enqueue_job(*args, **kwargs): return None
    def fetch_job(*args, **kwargs): return None
    def list_runs(*args, **kwargs): return []

# ---------- Streamlit config ----------
st.set_page_config(
    page_title="Master Generators for ODEs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS ----------
st.markdown("""
<style>
.main-header{
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding:1.4rem;border-radius:16px;margin-bottom:1.0rem;color:white;text-align:center;
  box-shadow:0 10px 30px rgba(0,0,0,0.2);
}
.main-title{font-size:2.0rem;font-weight:700;margin-bottom:.25rem;}
.subtitle{font-size:0.98rem;opacity:.95;}
.metric-card{
  background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  color:white;padding:0.9rem;border-radius:12px;text-align:center;
  box-shadow:0 10px 20px rgba(0,0,0,0.2); min-height:90px;
}
.info-box{
  background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
  border-left:5px solid #2196f3;padding:0.9rem;border-radius:10px;margin:1rem 0;
}
.result-box{
  background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
  border:2px solid #4caf50;padding:0.9rem;border-radius:10px;margin:1rem 0;
}
.error-box{
  background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);
  border:2px solid #f44336;padding:0.9rem;border-radius:10px;margin:1rem 0;
}
.small-muted{font-size:0.85rem;opacity:0.8;}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
MIN_ODES_FOR_TRAINING = int(os.getenv("MIN_ODES_FOR_TRAINING", "20"))

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        # Fallback for older Streamlit
        try:
            st.experimental_rerun()  # type: ignore[attr-defined]
        except Exception:
            pass

def _env_queue(name: str, default: str) -> str:
    return os.getenv(name, default)

def _sympify_safe(v):
    try:
        return sp.sympify(v)
    except Exception:
        return v

def _latex(expr) -> str:
    try:
        return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
    except Exception:
        return str(expr)

# ---------- Session State ----------
class SS:
    @staticmethod
    def init():
        defaults = {
            "generator_constructor": GeneratorConstructor() if GeneratorConstructor else None,
            "generator_terms": [],
            "current_generator": None,
            "lhs_source": "constructor",     # 'constructor'|'freeform'|'arbitrary'
            "free_terms": [],
            "arbitrary_lhs_text": "",
            "generated_odes": [],
            "batch_results": [],
            "analysis_results": [],
            "export_history": [],
            "training_jobs": [],
            "compute_jobs": [],
            "reverse_jobs": [],
            "ml_trainer": None,
            "ml_trained": False,
            "ml_history": {},
            "ml_model_path": "",
            "basic_functions": BasicFunctions() if BasicFunctions else None,
            "special_functions": SpecialFunctions() if SpecialFunctions else None,
            "novelty_detector": ODENoveltyDetector() if ODENoveltyDetector else None,
            "ode_classifier": ODEClassifier() if ODEClassifier else None,
            "last_job_id": None,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

def register_generated_ode(res: dict):
    """Normalize and store a generated ODE result."""
    out = dict(res)
    out.setdefault("type", "nonlinear")
    out.setdefault("order", 0)
    out.setdefault("function_used", str(res.get("function_used", "unknown")))
    out.setdefault("parameters", {})
    out.setdefault("initial_conditions", {})
    out.setdefault("timestamp", datetime.now().isoformat())
    out["generator_number"] = len(st.session_state.generated_odes) + 1
    try:
        out.setdefault("ode", sp.Eq(_sympify_safe(out["generator"]), _sympify_safe(out["rhs"])))
    except Exception:
        pass
    st.session_state.generated_odes.append(out)

# ---------- LaTeX Export ----------
class LaTeXExporter:
    @staticmethod
    def sympy_to_latex(expr) -> str:
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
    def document(ode: Dict[str, Any], include_preamble=True) -> str:
        g = ode.get("generator", "")
        rhs = ode.get("rhs", "")
        sol = ode.get("solution", "")
        params = ode.get("parameters", {})
        ic = ode.get("initial_conditions", {})
        cls = ode.get("classification", {})
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
            f"{LaTeXExporter.sympy_to_latex(g)} = {LaTeXExporter.sympy_to_latex(rhs)}",
            r"\end{equation}",
            r"\subsection{Exact Solution}",
            r"\begin{equation}",
            f"y(x) = {LaTeXExporter.sympy_to_latex(sol)}",
            r"\end{equation}",
            r"\subsection{Parameters}",
            r"\begin{align}",
            f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha', 1))} \\\\",
            f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta', 1))} \\\\",
            f"n       &= {params.get('n', 1)} \\\\",
            f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M', 0))}",
            r"\end{align}",
        ]
        if ic:
            parts += [r"\subsection{Initial Conditions}", r"\begin{align}"]
            items = list(ic.items())
            for i, (k, v) in enumerate(items):
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}" + (r" \\" if i < len(items) - 1 else ""))
            parts.append(r"\end{align}")
        if cls:
            parts += [r"\subsection{Mathematical Classification}", r"\begin{itemize}"]
            parts.append(f"\\item \\textbf{{Type:}} {cls.get('type','Unknown')}")
            parts.append(f"\\item \\textbf{{Order:}} {cls.get('order','Unknown')}")
            parts.append(f"\\item \\textbf{{Linearity:}} {cls.get('linearity','Unknown')}")
            if "field" in cls:
                parts.append(f"\\item \\textbf{{Field:}} {cls['field']}")
            if "applications" in cls:
                apps = ", ".join(cls["applications"][:5])
                parts.append(f"\\item \\textbf{{Applications:}} {apps}")
            parts.append(r"\end{itemize}")
        parts += [
            r"\subsection{Solution Verification}",
            r"Substitute $y(x)$ into the generator to verify $L[y] = \text{RHS}$."
        ]
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def package(ode: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ode_document.tex", LaTeXExporter.document(ode, True))
            z.writestr("ode_data.json", json.dumps(ode, indent=2, default=str))
            z.writestr("README.txt", "To compile: pdflatex ode_document.tex\n")
        buf.seek(0)
        return buf.getvalue()

# ---------- PAGES ----------

def page_dashboard():
    st.header("🏠 Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>',
            unsafe_allow_html=True
        )
    with c2:
        trained = "✅ Trained" if st.session_state.get("ml_trained") else "⏳ Not Trained"
        st.markdown(
            f'<div class="metric-card"><h3>🤖 ML Model</h3><p style="font-size:1.1rem">{trained}</p></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h3>📦 Batch Rows</h3><h1>{len(st.session_state.batch_results)}</h1></div>',
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><h3>🛰️ Jobs Monitored</h3><h1>{len(list_runs())}</h1></div>',
            unsafe_allow_html=True
        )

    st.subheader("📊 Recent ODEs")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-8:])
        cols = [c for c in ["type", "order", "function_used", "generator_number", "timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs yet. Go to **Apply Master Theorem** or **Generator Constructor**.")

def page_constructor():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Build custom generator LHS using symbolic terms.</div>', unsafe_allow_html=True)
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not available. (src/ missing).")
        return

    with st.expander("➕ Add Generator Term", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox(
                "Derivative Order",
                [0,1,2,3,4,5],
                format_func=lambda x: {0:"y",1:"y'",2:"y''",3:"y'''",4:"y⁽⁴⁾",5:"y⁽⁵⁾"}.get(x, f"y⁽{x}⁾")
            )
        with c2:
            ftype = st.selectbox("Function Type", [t.value for t in DerivativeType],
                                 format_func=lambda s: s.replace("_"," ").title())
        with c3:
            coef = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)
        c5, c6, c7 = st.columns(3)
        with c5:
            optype = st.selectbox(
                "Operator Type", [t.value for t in OperatorType],
                format_func=lambda s: s.replace("_", " ").title()
            )
        with c6:
            scaling = st.number_input("Scaling (a)", 0.1, 10.0, 1.0, 0.1) if optype in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if optype in ["delay","advance"] else None
        if st.button("➕ Add Term", type="primary"):
            try:
                term = DerivativeTerm(
                    derivative_order=int(deriv_order),
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
        st.subheader("📝 Current Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            c1, c2 = st.columns([8,1])
            with c1:
                desc = term.get_description() if hasattr(term, "get_description") else str(term)
                st.write(f"• {desc}")
            with c2:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i)
                    _safe_rerun()
        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                spec = GeneratorSpecification(
                    terms=st.session_state.generator_terms,
                    name=f"Custom Generator {len(st.session_state.generated_odes)+1}"
                )
                st.session_state.current_generator = spec
                st.success("Generator specification created.")
                try:
                    st.latex(_latex(spec.lhs) + " = \\mathrm{RHS}(x,y)")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Failed to build specification: {e}")
    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None

def page_apply_theorem():
    st.header("🎯 Apply Master Theorem (Async‑ready)")
    # LHS source
    src = st.radio("LHS source", ("constructor","freeform","arbitrary"),
                   index={"constructor":0,"freeform":1,"arbitrary":2}[st.session_state["lhs_source"]],
                   horizontal=True)
    st.session_state["lhs_source"] = src

    # Function library
    colA, colB = st.columns(2)
    with colA:
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        bf = st.session_state.get("basic_functions")
        sf = st.session_state.get("special_functions")
        if lib == "Basic" and bf:
            names = bf.get_function_names()
            src_lib = bf
        elif lib == "Special" and sf:
            names = sf.get_function_names()
            src_lib = sf
        else:
            names, src_lib = [], None
        func_name = st.selectbox("Choose f(z)", names) if names else st.text_input("Enter f(z)", "exp(z)")

    # Parameters
    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c5, c6, c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7:
        async_on = has_redis()
        st.info("Async via Redis: **ON**" if async_on else "Async via Redis: **OFF**")

    # Free-form LHS builder
    st.subheader("🧩 Free‑form LHS (Builder)")
    with st.expander("Add terms", expanded=False):
        if "free_terms" not in st.session_state:
            st.session_state.free_terms = []
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_k = st.number_input("inner k (y^(k))", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)",
                            ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs",
                             "asin","acos","atan","asinh","acosh","atanh","erf","erfc"], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_m = st.number_input("outer m (D^m)", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale (a)", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift (b)", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": float(coef), "inner_order": int(inner_k), "wrapper": wrapper,
                    "power": int(power), "outer_order": int(outer_m),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write("**Current terms:**")
            for i, t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. {t}")
            ccc1, ccc2 = st.columns(2)
            with ccc1:
                if st.button("🧮 Use free‑form LHS"):
                    st.session_state["lhs_source"] = "freeform"
                    st.success("Free-form LHS selected.")
            with ccc2:
                if st.button("🗑️ Clear terms"):
                    st.session_state.free_terms = []

    # Arbitrary SymPy LHS
    st.subheader("✍️ Arbitrary LHS (SymPy)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Enter any SymPy expression in x and y(x)",
        value=st.session_state.arbitrary_lhs_text or "",
        height=90
    )
    cva, cvb = st.columns(2)
    with cva:
        if st.button("✅ Validate arbitrary LHS"):
            from shared.ode_core import parse_arbitrary_lhs  # local import
            try:
                _ = parse_arbitrary_lhs(st.session_state.arbitrary_lhs_text)
                st.success("Expression parsed successfully.")
                st.session_state["lhs_source"] = "arbitrary"
            except Exception as e:
                st.error(f"Parse error: {e}")
    with cvb:
        if st.button("↩️ Prefer Constructor LHS"):
            st.session_state["lhs_source"] = "constructor"

    # Prepare constructor LHS (if any)
    constructor_lhs = None
    spec = st.session_state.get("current_generator")
    if spec is not None and hasattr(spec, "lhs"):
        constructor_lhs = spec.lhs
        st.caption("Using constructor LHS currently in session.")

    # Theorem 4.2 option
    st.markdown("---")
    colm1, colm2 = st.columns([1,1])
    with colm1:
        compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", False)
    with colm2:
        m_order = st.number_input("m", 1, 12, 1)

    # Generate ODE (Theorem 4.1)
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        payload = {
            "func_name": func_name,
            "alpha": alpha, "beta": beta, "n": int(n), "M": M,
            "use_exact": use_exact, "simplify_level": simplify_level,
            "lhs_source": st.session_state["lhs_source"],
            "freeform_terms": st.session_state.get("free_terms"),
            "arbitrary_lhs_text": st.session_state.get("arbitrary_lhs_text"),
            "function_library": lib,
        }
        if not has_redis():
            # Run synchronously
            try:
                basic_lib = st.session_state.get("basic_functions")
                special_lib = st.session_state.get("special_functions")
                p = ComputeParams(
                    func_name=payload["func_name"],
                    alpha=payload["alpha"], beta=payload["beta"], n=payload["n"], M=payload["M"],
                    use_exact=payload["use_exact"], simplify_level=payload["simplify_level"],
                    lhs_source=payload["lhs_source"],
                    constructor_lhs=constructor_lhs,
                    freeform_terms=payload["freeform_terms"],
                    arbitrary_lhs_text=payload["arbitrary_lhs_text"],
                    function_library=payload["function_library"],
                    basic_lib=basic_lib, special_lib=special_lib
                )
                res = compute_ode_full(p)
                # convert to sympy where needed
                res["generator"] = _sympify_safe(res["generator"])
                res["rhs"] = _sympify_safe(res["rhs"])
                res["solution"] = _sympify_safe(res["solution"])
                register_generated_ode(res)
                _show_ode_result(res)
            except Exception as e:
                st.error(f"Generation error: {e}")
        else:
            # Enqueue to worker
            qname = _env_queue("RQ_QUEUE_COMPUTE", "ode_jobs")
            job_id = enqueue_job(
                "worker.compute_job",
                payload,
                queue=qname,
                job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600")),
                result_ttl=int(os.getenv("RQ_RESULT_TTL", "604800")),
                description="compute ODE",
                meta={"summary": {"kind": "compute", "func": func_name}}
            )
            if job_id:
                st.session_state["last_job_id"] = job_id
                st.success(f"Job submitted to **{qname}**. ID = {job_id}")
            else:
                st.error("Failed to submit job. Check REDIS_URL / queue.")

    # Job Status (only if async & pending job)
    if has_redis() and "last_job_id" in st.session_state and st.session_state["last_job_id"]:
        st.markdown("### 🛰️ Job Status")
        colx, coly = st.columns([1,1])
        with colx:
            if st.button("🔄 Refresh status"):
                _safe_rerun()
        info = fetch_job(st.session_state["last_job_id"])
        if info:
            st.caption(
                f"Queue: {info.get('queue')} • Status: **{info.get('status')}** • "
                f"Enqueued: {info.get('enqueued_at')} • Started: {info.get('started_at')}"
            )
            if info.get("exc_info"):
                st.error("Worker exception:")
                st.code(info["exc_info"])
            if info.get("status") == "finished":
                res = info["result"]
                # cast for pretty rendering
                try:
                    res["generator"] = _sympify_safe(res["generator"])
                    res["rhs"]       = _sympify_safe(res["rhs"])
                    res["solution"]  = _sympify_safe(res["solution"])
                except Exception:
                    pass
                register_generated_ode(res)
                _show_ode_result(res)
                st.session_state["last_job_id"] = None
        else:
            st.info("⏳ Still computing...")

    # Theorem 4.2 (synchronous)
    if compute_mth and st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2", use_container_width=True):
        try:
            lib_obj = st.session_state.get("basic_functions") if lib == "Basic" else st.session_state.get("special_functions")
            f_preview = get_function_expr(lib_obj, func_name)
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta)  if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(f_preview, α, β, int(n), int(m_order), x, simplify_level)
            st.markdown("### 🔢 Derivative")
            st.latex(fr"y^{{({int(m_order)})}}(x) = " + _latex(y_m))
        except Exception as e:
            st.error(f"Failed to compute y^{m_order}(x): {e}")

def _show_ode_result(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h3>✅ ODE Generated Successfully!</h3></div>', unsafe_allow_html=True)
    t_eq, t_sol, t_exp = st.tabs(["📐 Equation", "💡 Solution & ICs", "📤 Export"])
    with t_eq:
        try:
            st.latex(_latex(res["generator"]) + " = " + _latex(res["rhs"]))
        except Exception:
            st.write("LHS:", res.get("generator"))
            st.write("RHS:", res.get("rhs"))
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t_sol:
        try:
            st.latex("y(x) = " + _latex(res["solution"]))
        except Exception:
            st.write("y(x) =", res.get("solution"))
        if res.get("initial_conditions"):
            st.markdown("**Initial conditions:**")
            for k, v in res["initial_conditions"].items():
                try:
                    st.latex(k + " = " + _latex(v))
                except Exception:
                    st.write(k, "=", v)
        p = res.get("parameters", {})
        st.markdown("**Parameters:**")
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        st.write(f"**Function:** f(z) = {res.get('f_expr_preview')}")
    with t_exp:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res.get("generator",""),
            "rhs": res.get("rhs",""),
            "solution": res.get("solution",""),
            "parameters": res.get("parameters", {}),
            "classification": {
                "type": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "order": res.get("order", 0),
                "linearity": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "field": "Mathematical Physics",
                "applications": ["Research Equation"],
            },
            "initial_conditions": res.get("initial_conditions", {}),
            "function_used": str(res.get("function_used","?")),
            "generator_number": idx,
            "type": res.get("type","nonlinear"),
            "order": res.get("order", 0)
        }
        latex_doc = LaTeXExporter.document(ode_data, include_preamble=True)
        st.download_button("📄 LaTeX (.tex)", latex_doc, f"ode_{idx}.tex", "text/x-latex", use_container_width=True)
        pkg = LaTeXExporter.package(ode_data)
        st.download_button("📦 ZIP package", pkg, f"ode_package_{idx}.zip", "application/zip", use_container_width=True)

def page_ml_dl():
    st.header("🤖 ML / DL (Training • Loading • Generation • Reverse)")
    if not MLTrainer:
        st.warning("MLTrainer not available (src/ml/trainer.py missing).")
        return

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Generated ODEs (session)", len(st.session_state.generated_odes))
    with c2: st.metric("Batch Rows", len(st.session_state.batch_results))
    with c3: st.metric("Min ODEs for Training", MIN_ODES_FOR_TRAINING)
    with c4:
        st.metric("Model", "Trained ✅" if st.session_state.get("ml_trained") else "Not trained")

    # Model selection & training config
    model_type = st.selectbox(
        "Model",
        ["pattern_learner", "vae", "transformer"],
        format_func=lambda s: {"pattern_learner":"Pattern Learner","vae":"VAE","transformer":"Transformer"}[s]
    )

    with st.expander("🎯 Training Configuration", True):
        c1,c2,c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 8, 128, 32)
        with c2:
            learning_rate = st.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
            samples = st.slider("Training Samples (synthetic)", 200, 10000, 2000)
        with c3:
            validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2)
            use_gpu = st.checkbox("Use GPU if available", True)
        run_on_worker = st.checkbox("Run on RQ worker (async)", has_redis())
        include_user_data = st.checkbox("Include session ODEs as seed examples", True)

        # PRE-TRAINING GATE
        total_examples = len(st.session_state.generated_odes) + len(st.session_state.batch_results)
        can_train = total_examples >= MIN_ODES_FOR_TRAINING
        if not can_train:
            st.warning(
                f"Need at least {MIN_ODES_FOR_TRAINING} ODEs before training. "
                f"Current: {total_examples}. Generate more ODEs (Apply Theorem / Batch) and retry."
            )

        # Train
        clicked = st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=not can_train)
        if clicked:
            payload = {
                "model_type": model_type,
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "samples": int(samples),
                "validation_split": float(validation_split),
                "device": "cuda" if (use_gpu and torch and torch.cuda.is_available()) else "cpu",
                "include_user_data": bool(include_user_data),
                "user_odes": st.session_state.generated_odes if include_user_data else [],
                "batch_rows": st.session_state.batch_results if include_user_data else [],
            }
            if run_on_worker and has_redis():
                qname = _env_queue("RQ_QUEUE_TRAIN", "ml_jobs")
                job_id = enqueue_job(
                    "worker.train_job",
                    payload,
                    queue=qname,
                    job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "86400")),  # up to 24h
                    result_ttl=int(os.getenv("RQ_RESULT_TTL", "604800")),
                    description=f"train {model_type}",
                    meta={"summary": {"kind": "train", "model": model_type}}
                )
                if job_id:
                    st.success(f"Training job submitted to **{qname}**. ID = {job_id}")
                    st.session_state.training_jobs.append(job_id)
                else:
                    st.error("Failed to submit training job. Check Redis/queue.")
            else:
                # Run locally (blocking)
                try:
                    trainer = MLTrainer(model_type=model_type, learning_rate=float(learning_rate), device=payload["device"])
                    st.session_state.ml_trainer = trainer
                    prog = st.progress(0); status = st.empty()
                    def cb(e, tot): prog.progress(min(1.0, e/tot)); status.text(f"Epoch {e}/{tot}")
                    trainer.train(
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        samples=int(samples),
                        validation_split=float(validation_split),
                        progress_callback=cb
                    )
                    # Save best model
                    ckpt_dir = "checkpoints"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    best_path = os.path.join(ckpt_dir, f"{model_type}_best.pth")
                    trainer.save_model(best_path)
                    st.session_state.ml_model_path = best_path
                    st.session_state.ml_trained = True
                    st.session_state.ml_history = trainer.history
                    st.success(f"Training completed. Best saved at {best_path}")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    # Monitor training jobs
    if has_redis() and st.session_state.training_jobs:
        st.subheader("🛰️ Training Jobs")
        for jid in list(st.session_state.training_jobs):
            info = fetch_job(jid)
            if not info: 
                st.info(f"{jid}: (not found yet)")
                continue
            st.write(f"**{jid}** — {info.get('status')} — {info.get('description')}")
            if info.get("exc_info"):
                st.error("Exception:"); st.code(info["exc_info"])
            if info.get("status") == "finished":
                res = info.get("result", {})
                best_path = res.get("best_path") or res.get("model_path") or ""
                if best_path:
                    st.session_state.ml_model_path = best_path
                    st.session_state.ml_trained = True
                    st.session_state.ml_history = res.get("history", {})
                    st.success(f"Model available: {best_path}")
                # remove from active list
                st.session_state.training_jobs.remove(jid)

    # Load / Upload model
    st.subheader("📦 Load / Upload Model")
    cols = st.columns(3)
    with cols[0]:
        mdl_path = st.text_input("Best checkpoint path", st.session_state.get("ml_model_path",""))
        if st.button("📂 Load from path"):
            if not mdl_path or not os.path.exists(mdl_path):
                st.error("Path not found.")
            else:
                try:
                    trainer = MLTrainer(model_type=model_type, device="cpu")
                    ok = trainer.load_model(mdl_path)
                    if ok:
                        st.session_state.ml_trainer = trainer
                        st.session_state.ml_trained = True
                        st.session_state.ml_model_path = mdl_path
                        st.session_state.ml_history = getattr(trainer, "history", {})
                        st.success("Model loaded.")
                    else:
                        st.error("Load returned False.")
                except Exception as e:
                    st.error(f"Load failed: {e}")
    with cols[1]:
        up = st.file_uploader("Upload .pth", type=["pth"])
        if up is not None:
            try:
                os.makedirs("checkpoints", exist_ok=True)
                target = os.path.join("checkpoints", up.name)
                with open(target, "wb") as f:
                    f.write(up.read())
                trainer = MLTrainer(model_type=model_type, device="cpu")
                ok = trainer.load_model(target)
                if ok:
                    st.session_state.ml_trainer = trainer
                    st.session_state.ml_trained = True
                    st.session_state.ml_model_path = target
                    st.session_state.ml_history = getattr(trainer, "history", {})
                    st.success(f"Uploaded & loaded: {target}")
                else:
                    st.error("Uploaded but load failed.")
            except Exception as e:
                st.error(f"Upload/load error: {e}")
    with cols[2]:
        if st.button("🧹 Clear model from memory"):
            st.session_state.ml_trainer = None
            st.session_state.ml_trained = False
            st.session_state.ml_model_path = ""
            st.success("Cleared.")

    # Generate from model
    st.subheader("🎨 Generate Novel ODEs from Trained Model")
    gen_cols = st.columns([1,1,2])
    with gen_cols[0]:
        n_gen = st.slider("How many", 1, 10, 1)
    with gen_cols[1]:
        do_register = st.checkbox("Add to session list", True)
    with gen_cols[2]:
        if st.button("🎲 Generate", type="primary") and st.session_state.get("ml_trained") and st.session_state.get("ml_trainer"):
            try:
                for i in range(n_gen):
                    res = st.session_state.ml_trainer.generate_new_ode()
                    if res:
                        # normalize view
                        try:
                            res["generator"] = _sympify_safe(res["generator"])
                            res["rhs"] = _sympify_safe(res["rhs"])
                        except Exception:
                            pass
                        st.success(f"Generated ODE #{i+1}")
                        with st.expander(f"ODE {i+1}"):
                            try:
                                st.latex(_latex(res.get("generator","")) + " = " + _latex(res.get("rhs","")))
                            except Exception:
                                st.code(str(res))
                        if do_register:
                            register_generated_ode(res)
            except Exception as e:
                st.error(f"Generation failed: {e}")

    # Reverse engineering (RQ)
    st.subheader("🔁 Reverse Engineering (Predict from learned patterns)")
    rev_cols = st.columns([1,2])
    with rev_cols[0]:
        run_rev_worker = st.checkbox("Run on worker", has_redis())
    with rev_cols[1]:
        text_input = st.text_area("Enter an ODE or solution hint (text/SymPy/LaTeX acceptable):", height=80)
    if st.button("🔎 Reverse Engineer", type="primary") and text_input.strip():
        payload = {
            "hint": text_input.strip(),
            "model_path": st.session_state.get("ml_model_path", ""),
            "model_type": model_type
        }
        if run_rev_worker and has_redis():
            qname = _env_queue("RQ_QUEUE_REVERSE", "reverse_jobs")
            job_id = enqueue_job(
                "worker.reverse_job",
                payload,
                queue=qname,
                job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT", "3600")),
                result_ttl=int(os.getenv("RQ_RESULT_TTL", "604800")),
                description="reverse",
                meta={"summary": {"kind": "reverse"}}
            )
            if job_id:
                st.success(f"Reverse job submitted to **{qname}**. ID = {job_id}")
                st.session_state.reverse_jobs.append(job_id)
        else:
            # If you have a local reverse function, call it here.
            st.info("Local reverse is not implemented here; use worker.")

    if has_redis() and st.session_state.reverse_jobs:
        st.subheader("🔭 Reverse Jobs")
        for jid in list(st.session_state.reverse_jobs):
            info = fetch_job(jid)
            if not info:
                st.write(f"{jid}: ...")
                continue
            st.write(f"**{jid}** — {info.get('status')}")
            if info.get("exc_info"):
                st.error("Exception:"); st.code(info["exc_info"])
            if info.get("status") == "finished":
                res = info.get("result", {})
                st.success("Reverse result:")
                st.json(res)
                st.session_state.reverse_jobs.remove(jid)

def page_batch():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs with your factories.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 500, 50)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_cats = st.multiselect("Function Categories", ["Basic","Special"], default=["Basic"])
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
            results = []
            prog = st.progress(0); status = st.empty()

            all_functions = []
            if "Basic" in func_cats and st.session_state.get("basic_functions"):
                all_functions += st.session_state.basic_functions.get_function_names()
            if "Special" in func_cats and st.session_state.get("special_functions"):
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

                    row = {
                        "ID": i+1,
                        "Type": gt,
                        "Function": func_name,
                        "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                    }
                    # Just record summarized items for table; production calls would call factories
                    if include_solutions:
                        row["Solution"] = " (omitted in batch preview) "
                    if include_classification:
                        row["Subtype"] = "standard"
                    results.append(row)
                except Exception as e:
                    log.debug(f"Failed to generate ODE {i+1}: {e}")

            st.session_state.batch_results.extend(results)
            st.success(f"Generated {len(results)} batch rows.")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            st.subheader("📤 Export Results")
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📊 CSV", csv, f"batch_{datetime.now():%Y%m%d_%H%M%S}.csv","text/csv")
            with c2:
                js = json.dumps(results, indent=2, default=str).encode("utf-8")
                st.download_button("📄 JSON", js, f"batch_{datetime.now():%Y%m%d_%H%M%S}.json","application/json")
            with c3:
                if export_format in ["LaTeX","All"]:
                    latex = "\\n".join([
                        r"\begin{tabular}{|c|c|c|c|c|}", r"\hline", r"ID & Type & Function & α & n \\",
                        r"\hline", *[f"{r.get('ID','')} & {r.get('Type','')} & {r.get('Function','')} & {r.get('α','')} & {r.get('n','')} \\\\" for r in results[:30]],
                        r"\hline", r"\end{tabular}"
                    ])
                    st.download_button("📝 LaTeX", latex, f"batch_{datetime.now():%Y%m%d_%H%M%S}.tex","text/x-latex")
            with c4:
                if export_format == "All":
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("batch_results.csv", df.to_csv(index=False))
                        zf.writestr("batch_results.json", json.dumps(results, indent=2, default=str))
                    zbuf.seek(0)
                    st.download_button("📦 ZIP", zbuf.getvalue(), f"batch_{datetime.now():%Y%m%d_%H%M%S}.zip","application/zip")

def page_novelty():
    st.header("🔍 Novelty Detection")
    if not st.session_state.get("novelty_detector"):
        st.warning("Novelty detector not available.")
        return
    method = st.radio("Input Method", ["Use Current Constructor LHS", "Enter ODE Manually", "Select from Generated"])
    ode = None
    if method == "Use Current Constructor LHS":
        spec = st.session_state.get("current_generator")
        if spec is not None and hasattr(spec, "lhs"):
            ode = {"ode": spec.lhs, "type": "custom", "order": getattr(spec, "order", 2)}
        else:
            st.warning("No constructor spec yet.")
    elif method == "Enter ODE Manually":
        ode_str = st.text_area("Enter ODE (LaTeX or text):")
        if ode_str:
            ode = {"ode": ode_str, "type": "manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','Unknown')}")
            ode = st.session_state.generated_odes[idx]
    if ode and st.button("🔍 Analyze Novelty", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                nd = st.session_state.novelty_detector
                analysis = nd.analyze(ode, check_solvability=True, detailed=True)
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
                    st.download_button("📥 Download Report", analysis.detailed_report,
                        f"novelty_report_{datetime.now():%Y%m%d_%H%M%S}.txt", "text/plain")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

def page_analysis():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    if not st.session_state.get("ode_classifier"):
        st.warning("Classifier unavailable.")
        return
    st.subheader("📊 Generated ODEs Overview")
    rows = []
    for i, ode in enumerate(st.session_state.generated_odes[-50:]):
        rows.append({
            "ID": i+1, "Type": ode.get("type","Unknown"), "Order": ode.get("order",0),
            "Generator": ode.get("generator_number","N/A"), "Function": ode.get("function_used","Unknown"),
            "Timestamp": ode.get("timestamp","")[:19]
        })
    df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)
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
                fig = px.bar(x=vc.index, y=vc.values, title="Classification by Field")
                fig.update_layout(xaxis_title="Field", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Classification failed: {e}")

def page_physics():
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
    cat = st.selectbox("Select Application Field", list(applications.keys()))
    for app in applications.get(cat, []):
        with st.expander(f"📚 {app['name']}"):
            try: st.latex(app["equation"])
            except Exception: st.write(app["equation"])
            st.write("Description:", app["description"])

def page_viz():
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
                # Placeholder visualization; plug actual numeric eval of solution if desired
                x = np.linspace(x_range[0], x_range[1], num_points)
                y = np.sin(x) * np.exp(-0.1*np.abs(x))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Solution"))
                fig.update_layout(title="ODE Solution", xaxis_title="x", yaxis_title="y(x)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Visualization failed: {e}")

def page_export():
    st.header("📤 Export & LaTeX")
    st.markdown('<div class="info-box">Export ODEs in publication‑ready LaTeX.</div>', unsafe_allow_html=True)
    if not st.session_state.generated_odes:
        st.warning("No ODEs to export."); return
    mode = st.radio("Export Type", ["Single ODE","Multiple ODEs","Complete Report"])
    if mode == "Single ODE":
        idx = st.selectbox(
            "Select ODE", range(len(st.session_state.generated_odes)),
            format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}"
        )
        ode = st.session_state.generated_odes[idx]
        st.subheader("📋 LaTeX Preview")
        tex = LaTeXExporter.document(ode, include_preamble=False)
        st.code(tex, language="latex")
        c1, c2 = st.columns(2)
        with c1:
            full = LaTeXExporter.document(ode, include_preamble=True)
            st.download_button("📄 LaTeX", full, f"ode_{idx+1}.tex", "text/x-latex")
        with c2:
            pkg = LaTeXExporter.package(ode)
            st.download_button("📦 Package", pkg, f"ode_package_{idx+1}.zip","application/zip")
    elif mode == "Multiple ODEs":
        sel = st.multiselect(
            "Select ODEs", range(len(st.session_state.generated_odes)),
            format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')}"
        )
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
                parts.append(LaTeXExporter.document(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            doc = "\n".join(parts)
            st.download_button("📄 Multi-ODE LaTeX", doc, f"odes_{datetime.now():%Y%m%d_%H%M%S}.tex","text/x-latex")
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
                parts.append(LaTeXExporter.document(ode, include_preamble=False))
            parts.append(r"""
\chapter{Conclusions}
The system successfully generated and analyzed multiple ODEs.
\end{document}
""")
            st.download_button("📄 Complete Report", "\n".join(parts),
                               f"complete_report_{datetime.now():%Y%m%d_%H%M%S}.tex", "text/x-latex")

def page_examples():
    st.header("📚 Examples Library")
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")

def page_settings():
    st.header("⚙️ Settings")
    tabs = st.tabs(["General","Export","Jobs","About"])
    with tabs[0]:
        st.checkbox("Dark mode", False)
        st.caption("General settings are illustrative.")
    with tabs[1]:
        st.checkbox("Include LaTeX preamble by default", True)
    with tabs[2]:
        if has_redis():
            st.subheader("Active & recent jobs")
            runs = list_runs()
            if not runs:
                st.info("No jobs found.")
            else:
                st.dataframe(pd.DataFrame(runs), use_container_width=True)
        else:
            st.info("Redis not configured; no job monitoring.")
    with tabs[3]:
        st.markdown("**Master Generators for ODEs** — Theorems 4.1 & 4.2, ML/DL, Reverse, Export, Novelty.")

def page_docs():
    st.header("📖 Documentation")
    st.markdown(f"""
**Quick Start**
1. Go to **Apply Master Theorem** and generate ODEs (or use **Batch** to generate many).
2. After you have **≥ {MIN_ODES_FOR_TRAINING}** ODEs (session + batch), go to **ML / DL** and train.
3. Load or upload models from **ML / DL** at any time; generate novel ODEs and try **Reverse**.
4. Export equations and reports from **Export & LaTeX**.
5. Monitor async jobs in **Settings → Jobs**.

**Queues / Worker**
- Set env: `RQ_QUEUE_COMPUTE=ode_jobs`, `RQ_QUEUE_TRAIN=ml_jobs`, `RQ_QUEUE_REVERSE=reverse_jobs`.
- Start worker on all queues (priority order):  
  `rq worker -u "$REDIS_URL" $RQ_QUEUE_COMPUTE $RQ_QUEUE_TRAIN $RQ_QUEUE_REVERSE`
""")

# ---------- Main ----------
def main():
    SS.init()
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Free‑form / Arbitrary LHS • Theorem 4.1 & 4.2 • Async Jobs • ML/DL • Reverse • Export • Novelty</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "📍 Navigation",
        [
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
            "📚 Examples Library",
            "⚙️ Settings",
            "📖 Documentation",
        ]
    )

    if page == "🏠 Dashboard": page_dashboard()
    elif page == "🔧 Generator Constructor": page_constructor()
    elif page == "🎯 Apply Master Theorem": page_apply_theorem()
    elif page == "🤖 ML / DL": page_ml_dl()
    elif page == "📊 Batch Generation": page_batch()
    elif page == "🔍 Novelty Detection": page_novelty()
    elif page == "📈 Analysis & Classification": page_analysis()
    elif page == "🔬 Physical Applications": page_physics()
    elif page == "📐 Visualization": page_viz()
    elif page == "📤 Export & LaTeX": page_export()
    elif page == "📚 Examples Library": page_examples()
    elif page == "⚙️ Settings": page_settings()
    elif page == "📖 Documentation": page_docs()

if __name__ == "__main__":
    main()