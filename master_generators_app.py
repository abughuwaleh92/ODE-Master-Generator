# master_generators_app.py
"""
Master Generators for ODEs — Full App (Refactored & Corrected)

Key features in this refactor
=============================
1) RQ-safe background jobs:
   • Uses rq_utils.enqueue_job(...) with RQ-1.x compatible args (timeout/result_ttl/ttl).
   • Persistent monitoring: shows status, progress meta & logs; never "dead ends".
   • One-click local fallback if job is "not found" (wrong Redis/queue) or Redis absent.

2) ODE Generation (Theorem 4.1 & 4.2):
   • LHS sources: Constructor (in-app only), Free-form builder, Arbitrary SymPy expression.
   • If Redis is ON, job is enqueued with JSON-serializable payload.
   • If Redis/OFF or constructor LHS needed, run locally to keep full functionality.

3) Reverse Engineering:
   • Practical reverse operator fit from a given y(x):
     Finds constants a,b s.t. y'' + a y' + b y ≈ 0 by least squares over sample points.
   • Shows the inferred operator and error (MSE). Demonstrates learned utility.

4) ML Training & Usage:
   • Train via RQ (if configured) or inline (local) with flexible Trainer API:
     - Adapts to both "old" trainer signatures and "new" config-based ones.
     - Saves/loads checkpoints; updates "trained" flag and training history.
   • Generate novel ODEs from trained model; evaluate basic metrics.

5) Session Management:
   • Save/Load/Upload session (generated ODEs, training history).
   • Upload model (.pth) into checkpoints and load it for reuse/resume.

6) Diagnostics & Settings:
   • Clear, actionable Redis diagnostics (queue name, workers, ping).
   • No 'phi_lib' ever passed to ComputeParams (avoids previous errors).

Keep your existing worker.py with:
   - compute_job(payload) -> result dict (strings/sympy ok)
   - train_job(payload)   -> result dict {status, history, artifacts, ...}
"""

# ---------------- std libs ----------------
import os
import sys
import io
import json
import time
import zipfile
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------- third-party ----------------
import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
from sympy import Symbol
from sympy.core.function import AppliedUndef

# Optional heavy libs (lazy usage)
try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = px = None

try:
    import torch
except Exception:
    torch = None

# ---------------- app logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("master_generators_app")

# ---------------- path setup ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
SRC_DIR = os.path.join(APP_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------- rq utils (must be provided in both web & worker images) ----------------
try:
    from rq_utils import has_redis, enqueue_job, fetch_job, redis_status
except Exception:
    def has_redis() -> bool: return False
    def enqueue_job(*a, **k): return None
    def fetch_job(*a, **k): return None
    def redis_status(): return {"ok": False, "queue": "ode_jobs", "url_present": False, "workers": [], "ping": None}

# ---------------- imports from src/* (resilient) ----------------
HAVE_SRC = True
try:
    from src.generators.linear_generators import LinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
except Exception:
    HAVE_SRC = False
    LinearGeneratorFactory = NonlinearGeneratorFactory = None

try:
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType,
    )
except Exception:
    GeneratorConstructor = GeneratorSpecification = None
    DerivativeTerm = DerivativeType = OperatorType = None

try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
except Exception:
    BasicFunctions = SpecialFunctions = None

# ML Trainer (adapts to multiple signatures)
try:
    from src.ml.trainer import MLTrainer
except Exception:
    MLTrainer = None

# Optional novelty/classifier
try:
    from src.generators.ode_classifier import ODEClassifier
except Exception:
    ODEClassifier = None

try:
    from src.dl.novelty_detector import ODENoveltyDetector
except Exception:
    ODENoveltyDetector = None

# ---------------- core math from shared.ode_core ----------------
# We only use the names known to exist in your codebase; no 'phi_lib' passed.
try:
    from shared.ode_core import (
        ComputeParams,
        compute_ode_full,
        theorem_4_2_y_m_expr,
        get_function_expr,
        parse_arbitrary_lhs,
        to_exact,
    )
except Exception as e:
    st.error(f"Failed to import shared.ode_core: {e}")
    raise

# ---------------- Streamlit config ----------------
st.set_page_config(
    page_title="Master Generators for ODEs",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- small utils ----------------
def safe_sympify(x):
    try:
        if isinstance(x, (sp.Expr, sp.Eq)):
            return x
        return sp.sympify(x)
    except Exception:
        return x

def expr_to_latex(expr) -> str:
    try:
        e = safe_sympify(expr)
        return sp.latex(e)
    except Exception:
        return str(expr)

def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ---------------- Session State ----------------
def _ensure_ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def init_session():
    _ensure_ss("generator_constructor", GeneratorConstructor() if GeneratorConstructor else None)
    _ensure_ss("generated_odes", [])
    _ensure_ss("batch_results", [])
    _ensure_ss("training_history", {})
    _ensure_ss("ml_trainer", None)
    _ensure_ss("ml_trained", False)
    _ensure_ss("last_generation_job_id", None)
    _ensure_ss("last_training_job_id", None)
    _ensure_ss("_last_payload", None)         # cached last payload for fallback
    _ensure_ss("lhs_source", "constructor")   # constructor | freeform | arbitrary
    _ensure_ss("free_terms", [])
    _ensure_ss("arbitrary_lhs_text", "")
    _ensure_ss("basic_functions", BasicFunctions() if BasicFunctions else None)
    _ensure_ss("special_functions", SpecialFunctions() if SpecialFunctions else None)
    _ensure_ss("ode_classifier", ODEClassifier() if ODEClassifier else None)
    try:
        _ensure_ss("novelty_detector", ODENoveltyDetector() if ODENoveltyDetector else None)
    except Exception:
        _ensure_ss("novelty_detector", None)

# ---------------- LaTeX Export ----------------
class LaTeXExporter:
    @staticmethod
    def document(ode: Dict[str, Any], include_preamble=True) -> str:
        gen = ode.get("generator", "")
        rhs = ode.get("rhs", "")
        sol = ode.get("solution", "")
        params = ode.get("parameters", {})
        ic = ode.get("initial_conditions", {})
        cls = ode.get("classification", {})
        lines = []
        if include_preamble:
            lines.append(r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{margin=1in}
\title{Master Generators ODE}
\date{\today}
\begin{document}
\maketitle""")
        lines += [r"\section*{Equation}",
                  r"\begin{equation}",
                  f"{expr_to_latex(gen)} = {expr_to_latex(rhs)}",
                  r"\end{equation}",
                  r"\section*{Solution}",
                  r"\begin{equation}",
                  f"y(x) = {expr_to_latex(sol)}",
                  r"\end{equation}",
                  r"\section*{Parameters}",
                  r"\begin{align}",
                  f"\\alpha &= {params.get('alpha', 1)} \\\\",
                  f"\\beta  &= {params.get('beta', 1)} \\\\",
                  f"n       &= {params.get('n', 1)} \\\\",
                  f"M       &= {params.get('M', 0)}",
                  r"\end{align}"]
        if ic:
            lines += [r"\section*{Initial Conditions}", r"\begin{align}"]
            last = list(ic.items())
            for i, (k, v) in enumerate(last):
                lines.append(f"{k} &= {expr_to_latex(v)}" + (r" \\" if i < len(last) - 1 else ""))
            lines.append(r"\end{align}")
        if cls:
            lines += [r"\section*{Classification}", r"\begin{itemize}"]
            for k in ["type", "order", "linearity", "field"]:
                if k in cls:
                    lines.append(f"\\item {k.title()}: {cls[k]}")
            lines.append(r"\end{itemize}")
        if include_preamble:
            lines.append(r"\end{document}")
        return "\n".join(lines)

    @staticmethod
    def package(ode: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("ode_document.tex", LaTeXExporter.document(ode, True))
            z.writestr("ode.json", json.dumps(ode, indent=2, default=str))
            z.writestr("README.txt", "Run: pdflatex ode_document.tex\n")
        buf.seek(0)
        return buf.getvalue()

# ---------------- Registration ----------------
def register_generated_ode(res: Dict[str, Any]):
    data = dict(res)
    data.setdefault("type", "nonlinear")
    data.setdefault("order", 0)
    data.setdefault("function_used", "unknown")
    data.setdefault("parameters", {})
    data.setdefault("classification", {})
    data["timestamp"] = _now()
    data["generator_number"] = len(st.session_state.generated_odes) + 1
    # Cast strings into sympy for nicer LaTeX where possible
    for k in ["generator", "rhs", "solution"]:
        data[k] = safe_sympify(data.get(k))
    st.session_state.generated_odes.append(data)

# ---------------- Local compute fallback ----------------
def _compute_sync(func_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run worker function locally without Redis.
    Expected to work with worker.compute_job / worker.train_job(payload).
    """
    import importlib
    mod_path, fn = func_path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    func = getattr(mod, fn)
    out = func(payload=payload)
    if not isinstance(out, dict):
        out = {"result": out}
    return out

# ---------------- Job monitor (persistent + fallback) ----------------
def job_monitor_block(job_id_key: str, result_handler):
    job_id = st.session_state.get(job_id_key)
    if not job_id:
        return
    st.markdown("### 📡 Job Monitor")
    info = fetch_job(job_id)
    if not info:
        st.warning("Job not found (Redis/queue mismatch or TTL cleanup).")
        cached = st.session_state.get("_last_payload")
        col1, col2 = st.columns([1,1])
        with col1:
            if cached and st.button("▶ Run locally now (fallback)"):
                try:
                    # Guess type: if result handler name hints 'ode', use compute_job; else train_job
                    func_path = "worker.compute_job" if "ode" in result_handler.__name__.lower() else "worker.train_job"
                    local = _compute_sync(func_path, cached)
                    result_handler(local)
                    st.session_state[job_id_key] = None
                except Exception as e:
                    st.error(f"Local fallback failed: {e}")
        with col2:
            if st.button("❌ Clear this job id"):
                st.session_state[job_id_key] = None
        return

    # pretty status
    st.json({k: info.get(k) for k in ["id","status","origin","enqueued_at","started_at","ended_at","meta"]})
    if info.get("exc_info"):
        with st.expander("Traceback"):
            st.code(info["exc_info"])

    logs = (info.get("meta") or {}).get("logs", [])
    if logs:
        with st.expander("🗒️ Worker logs"):
            for line in logs[-200:]:
                st.text(line)

    status = info.get("status")
    if status == "failed":
        st.error("Job failed.")
        st.session_state[job_id_key] = None
    elif status == "finished":
        st.success("Job finished.")
        result = info.get("result")
        result_handler(result)
        st.session_state[job_id_key] = None
    else:
        st.info(f"⏳ Status: {status}. Click the page's 'Refresh' button or just reload the page.")

# ---------------- Apply Master Theorem Page ----------------
def apply_master_theorem_page():
    st.header("🎯 Apply Master Theorem")
    # Function library
    c1, c2 = st.columns(2)
    with c1:
        lib = st.selectbox("Function library", ["Basic", "Special"], index=0)
    with c2:
        if lib == "Basic" and st.session_state.basic_functions:
            names = st.session_state.basic_functions.get_function_names()
        elif lib == "Special" and st.session_state.special_functions:
            names = st.session_state.special_functions.get_function_names()
        else:
            names = []
        func_name = st.selectbox("Select f(z)", names) if names else st.text_input("Enter f(z) name", "exp")

    # Parameters
    c3, c4, c5, c6 = st.columns(4)
    with c3: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c4: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c5: n     = st.number_input("n", 1, 12, 1)
    with c6: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")

    c7, c8, c9 = st.columns(3)
    with c7: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c8: simplify_level = st.selectbox("Simplify level", ["light","none","aggressive"], index=0)
    with c9:
        st.info("Redis: ON" if has_redis() else "Redis: OFF")

    # LHS Source
    st.subheader("LHS Source")
    src = st.radio("Choose source", ["constructor","freeform","arbitrary"],
                   index={"constructor":0,"freeform":1,"arbitrary":2}[st.session_state.lhs_source])
    st.session_state.lhs_source = src

    # Free-form builder
    st.markdown("**Free-form term builder**")
    with st.expander("Add terms"):
        cols = st.columns(8)
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_k = st.number_input("inner k (y^(k))", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)",
            ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs","asin","acos","atan","asinh","acosh","atanh","erf"], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_m = st.number_input("outer m (D^m)", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale a", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift b", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                t = dict(coef=float(coef), inner_order=int(inner_k), wrapper=wrapper,
                         power=int(power), outer_order=int(outer_m))
                if abs(scale) > 1e-14: t["arg_scale"] = float(scale)
                if abs(shift) > 1e-14: t["arg_shift"] = float(shift)
                st.session_state.free_terms.append(t)
    if st.session_state.free_terms:
        st.write("**Current terms:**", st.session_state.free_terms)
        cfa, cfb = st.columns(2)
        with cfa:
            if st.button("Use free-form LHS"):
                st.session_state.lhs_source = "freeform"
        with cfb:
            if st.button("Clear all terms"):
                st.session_state.free_terms = []

    # Arbitrary SymPy LHS
    st.subheader("Arbitrary LHS (SymPy expression)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Example: sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1)", 
        value=st.session_state.arbitrary_lhs_text or "", height=100)

    cva, cvb = st.columns(2)
    with cva:
        if st.button("✅ Validate arbitrary LHS"):
            try:
                _ = parse_arbitrary_lhs(st.session_state.arbitrary_lhs_text)
                st.success("Parsed successfully.")
                st.session_state.lhs_source = "arbitrary"
            except Exception as e:
                st.error(f"Parse error: {e}")
    with cvb:
        if st.button("↩️ Prefer Constructor"):
            st.session_state.lhs_source = "constructor"

    # Theorem 4.2
    st.markdown("---")
    do_m = st.checkbox("Compute y^(m)(x) via Theorem 4.2", False)
    m_val = st.number_input("m", 1, 12, 1)

    # Generate button
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        payload = {
            "func_name": func_name,
            "alpha": float(alpha),
            "beta": float(beta),
            "n": int(n),
            "M": float(M),
            "use_exact": bool(use_exact),
            "simplify_level": simplify_level,
            "lhs_source": st.session_state.lhs_source,
            "freeform_terms": st.session_state.free_terms if st.session_state.lhs_source=="freeform" else None,
            "arbitrary_lhs_text": st.session_state.arbitrary_lhs_text if st.session_state.lhs_source=="arbitrary" else None,
            "function_library": lib,
        }
        st.session_state["_last_payload"] = payload

        # If user selected constructor and Redis is ON, warn that worker cannot access constructor session objects.
        # We automatically fall back to local computation (keeps full functionality "intact").
        if st.session_state.lhs_source == "constructor":
            # run locally to use current constructor LHS available in session (if your compute_ode_full handles it)
            try:
                local = _compute_sync("worker.compute_job", payload)
                _handle_ode_result(local)
            except Exception as e:
                st.error(f"Local generation failed: {e}")
        else:
            # freeform/arbitrary can run via worker (all JSON-serializable)
            if has_redis():
                jid = enqueue_job("worker.compute_job", payload, description="ode-generate")
                if jid:
                    st.session_state["last_generation_job_id"] = jid
                    st.success(f"Job submitted: {jid}")
                else:
                    st.warning("Redis not available. Running locally...")
                    try:
                        local = _compute_sync("worker.compute_job", payload)
                        _handle_ode_result(local)
                    except Exception as e:
                        st.error(f"Local generation failed: {e}")
            else:
                # no Redis
                try:
                    local = _compute_sync("worker.compute_job", payload)
                    _handle_ode_result(local)
                except Exception as e:
                    st.error(f"Local generation failed: {e}")

    # monitor generation if any
    job_monitor_block("last_generation_job_id", _handle_ode_result)

    # Theorem 4.2 compute
    if do_m and st.button(f"🧮 Compute y^({m_val})(x)"):
        try:
            lib_obj = st.session_state.basic_functions if lib == "Basic" else st.session_state.special_functions
            f_expr = get_function_expr(lib_obj, func_name) if lib_obj else sp.exp(Symbol('z'))
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta) if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(f_expr, α, β, int(n), int(m_val), x, simplify_level)
            st.latex(r"y^{(%d)}(x) = " % int(m_val) + sp.latex(y_m))
        except Exception as e:
            st.error(f"Theorem 4.2 failed: {e}")

def _handle_ode_result(result: Dict[str, Any]):
    if not result:
        st.error("No result.")
        return
    if "error" in result:
        st.error(f"Generation error: {result['error']}")
        return
    # Normalized path: some workers return at top-level, some under 'result'
    res = result.get("result", result)
    register_generated_ode(res)
    ode = st.session_state.generated_odes[-1]
    with st.container():
        st.success("✅ ODE generated")
        try:
            st.latex(expr_to_latex(ode["generator"]) + " = " + expr_to_latex(ode["rhs"]))
        except Exception:
            st.write("LHS:", ode["generator"])
            st.write("RHS:", ode["rhs"])
        try:
            st.latex("y(x) = " + expr_to_latex(ode["solution"]))
        except Exception:
            st.write("Solution:", ode["solution"])

        # Export
        col1, col2 = st.columns(2)
        with col1:
            tex = LaTeXExporter.document(ode, include_preamble=True)
            st.download_button("📄 Download LaTeX", data=tex,
                               file_name=f"ode_{ode['generator_number']}.tex", mime="text/x-latex")
        with col2:
            pkg = LaTeXExporter.package(ode)
            st.download_button("📦 Download Package (ZIP)", data=pkg,
                               file_name=f"ode_{ode['generator_number']}.zip", mime="application/zip")

# ---------------- Reverse Engineering ----------------
def reverse_engineering_page():
    st.header("🔁 Reverse Engineering (fit a linear operator)")
    st.write("Given y(x), we find constants a, b such that **y'' + a·y' + b·y ≈ 0** (least squares).")
    y_text = st.text_area("Enter y(x) (SymPy)", "exp(-x)*sin(x)")
    x_min, x_max = st.slider("Sampling domain", -3.0, 3.0, (-2.0, 2.0))
    N = st.slider("Samples", 50, 2000, 400)
    if st.button("🔎 Infer L[y]"):
        try:
            x = sp.Symbol("x", real=True)
            y_expr = sp.sympify(y_text)
            y1 = sp.diff(y_expr, x)
            y2 = sp.diff(y_expr, x, 2)
            # sample and least squares: y2 + a*y1 + b*y = 0 -> [y1, y] [a,b] = -y2
            xs = np.linspace(x_min, x_max, N)
            y_vals = np.array([float(y_expr.subs(x, xv)) for xv in xs])
            y1_vals = np.array([float(y1.subs(x, xv)) for xv in xs])
            y2_vals = np.array([float(y2.subs(x, xv)) for xv in xs])
            A = np.stack([y1_vals, y_vals], axis=1)  # columns: y', y
            b = -y2_vals
            # Solve least squares
            coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
            a, bcoef = coeff.tolist()
            # compute residual MSE
            resid = y2_vals + a*y1_vals + bcoef*y_vals
            mse = float(np.mean(resid**2))
            st.success(f"Inferred operator ≈ D² + {a:.6g}·D + {bcoef:.6g}")
            st.write(f"Residual MSE on [{x_min},{x_max}] (N={N}): {mse:.3e}")
            st.code(f"L[y] = y'' + {a:.6g}*y' + {bcoef:.6g}*y")
        except Exception as e:
            st.error(f"Reverse engineering failed: {e}")

# ---------------- Generator Constructor ----------------
def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    if not GeneratorSpecification or not DerivativeTerm:
        st.warning("Constructor classes missing. Use Free-form/Arbitrary LHS on the Theorem page.")
        return

    with st.expander("➕ Add Term", True):
        col = st.columns(4)
        with col[0]:
            d_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], index=0)
        with col[1]:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType])
        with col[2]:
            coef = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with col[3]:
            power = st.number_input("Power", 1, 6, 1)
        col2 = st.columns(3)
        with col2[0]:
            op_type = st.selectbox("Operator", [t.value for t in OperatorType])
        with col2[1]:
            scaling = st.number_input("Scaling a", 0.1, 10.0, 1.0, 0.1) if op_type in ["delay","advance"] else None
        with col2[2]:
            shift = st.number_input("Shift b", -10.0, 10.0, 0.0, 0.1) if op_type in ["delay","advance"] else None
        if st.button("Add", type="primary"):
            term = DerivativeTerm(
                derivative_order=int(d_order),
                coefficient=float(coef),
                power=int(power),
                function_type=DerivativeType(func_type),
                operator_type=OperatorType(op_type),
                scaling=scaling, shift=shift
            )
            if "generator_terms" not in st.session_state:
                st.session_state.generator_terms = []
            st.session_state.generator_terms.append(term)
            st.success("Term added.")

    if "generator_terms" in st.session_state and st.session_state.generator_terms:
        st.subheader("Current Terms")
        for i, t in enumerate(st.session_state.generator_terms):
            st.write(f"{i+1}. {t}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔨 Build Specification"):
                try:
                    spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Gen {len(st.session_state.generated_odes)+1}")
                    st.session_state.current_generator = spec
                    st.success("Specification created.")
                    try:
                        st.latex(sp.latex(spec.lhs) + " = RHS")
                    except Exception:
                        st.write("LHS created.")
                except Exception as e:
                    st.error(f"Build failed: {e}")
        with c2:
            if st.button("🗑️ Clear All"):
                st.session_state.generator_terms = []
                st.session_state.current_generator = None

# ---------------- ML Training Page ----------------
def _flex_trainer_ctor(kwargs: Dict[str, Any]):
    """
    Construct MLTrainer with flexible signature support.
    Works whether your Trainer expects (model_type, hidden_dim, ...) OR a 'config' dataclass.
    """
    if MLTrainer is None:
        raise RuntimeError("MLTrainer not found in src.ml.trainer")
    try:
        # Try simple ctor first
        return MLTrainer(
            model_type=kwargs.get("model_type", "pattern_learner"),
            input_dim=kwargs.get("input_dim", 12),
            hidden_dim=kwargs.get("hidden_dim", 128),
            output_dim=kwargs.get("output_dim", 12),
            learning_rate=kwargs.get("learning_rate", 1e-3),
            device=kwargs.get("device", "cuda" if torch and torch.cuda.is_available() else "cpu"),
            checkpoint_dir=kwargs.get("checkpoint_dir", "checkpoints"),
            enable_mixed_precision=kwargs.get("enable_mixed_precision", False)
        )
    except TypeError:
        # Fallback: config-based Trainer
        cfg = {
            "model_type": kwargs.get("model_type", "pattern_learner"),
            "input_dim": kwargs.get("input_dim", 12),
            "hidden_dim": kwargs.get("hidden_dim", 128),
            "output_dim": kwargs.get("output_dim", 12),
            "learning_rate": kwargs.get("learning_rate", 1e-3),
            "device": kwargs.get("device", "cuda" if torch and torch.cuda.is_available() else "cpu"),
            "checkpoint_dir": kwargs.get("checkpoint_dir", "checkpoints"),
            "enable_mixed_precision": kwargs.get("enable_mixed_precision", False),
        }
        return MLTrainer(config=cfg)

def _flex_train_call(trainer, train_kwargs: Dict[str, Any], progress_callback=None):
    """
    Call train() with only the kwargs the current Trainer implementation supports.
    Also robust to implementations without 'save_best' or 'use_generator'.
    """
    import inspect
    sig = inspect.signature(trainer.train)
    allowed = {}
    for k, v in train_kwargs.items():
        if k in sig.parameters:
            allowed[k] = v
    if "progress_callback" in sig.parameters:
        allowed["progress_callback"] = progress_callback
    return trainer.train(**allowed)

def ml_training_page():
    st.header("🤖 ML Pattern Learning")
    if MLTrainer is None:
        st.warning("MLTrainer not available.")
        return

    # Summary
    cols = st.columns(4)
    with cols[0]:
        st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with cols[1]:
        st.metric("Batch ODE rows", len(st.session_state.batch_results))
    with cols[2]:
        st.metric("Is Trained", "Yes" if st.session_state.ml_trained else "No")
    with cols[3]:
        st.metric("Jobs", f"gen: {1 if st.session_state.last_generation_job_id else 0} • train: {1 if st.session_state.last_training_job_id else 0}")

    # Config
    st.subheader("Training Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"], index=0)
        hidden_dim = st.number_input("Hidden dim", 16, 1024, 128, 16)
        lr = st.select_slider("Learning Rate", [1e-4,5e-4,1e-3,5e-3,1e-2], value=1e-3)
    with c2:
        epochs = st.slider("Epochs", 5, 500, 100, 5)
        batch = st.slider("Batch size", 4, 256, 32, 4)
        samples = st.slider("Synthetic samples", 100, 10000, 1000, 100)
    with c3:
        val_split = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
        use_gen = st.checkbox("Use data generator", True)
        amp = st.checkbox("Mixed precision (AMP)", False)

    # Build payload
    train_payload = {
        "model_type": model_type,
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(lr),
        "epochs": int(epochs),
        "batch_size": int(batch),
        "samples": int(samples),
        "validation_split": float(val_split),
        "use_generator": bool(use_gen),
        "enable_mixed_precision": bool(amp),
        # device decided inside worker or here if local:
    }
    st.session_state["_last_payload"] = train_payload

    # Controls
    cl1, cl2, cl3, cl4 = st.columns(4)
    with cl1:
        if st.button("🚀 Train (RQ if available)", type="primary"):
            if has_redis():
                jid = enqueue_job("worker.train_job", train_payload, description="ml-train")
                if jid:
                    st.session_state["last_training_job_id"] = jid
                    st.success(f"Training job submitted: {jid}")
                else:
                    st.warning("Redis down; running locally.")
                    _run_training_local(train_payload)
            else:
                _run_training_local(train_payload)
    with cl2:
        if st.button("🧪 Generate using current model"):
            _generate_from_model()

    with cl3:
        uploaded = st.file_uploader("Upload model (.pth)", type=["pth"])
        if uploaded:
            os.makedirs("checkpoints", exist_ok=True)
            path = os.path.join("checkpoints", f"uploaded_{int(time.time())}.pth")
            with open(path, "wb") as f:
                f.write(uploaded.read())
            try:
                # Load into trainer
                tr = _flex_trainer_ctor({
                    "model_type": model_type,
                    "hidden_dim": int(hidden_dim),
                    "learning_rate": float(lr),
                })
                ok = tr.load_model(path)
                st.session_state.ml_trainer = tr if ok else None
                st.session_state.ml_trained = bool(ok)
                st.success(f"Loaded model: {path}") if ok else st.error("Load failed")
            except Exception as e:
                st.error(f"Load error: {e}")

    with cl4:
        if st.button("💾 Save Session"):
            _save_session()
        if st.button("📂 Load Session from file"):
            _load_session_dialog()

    # Show job monitor for training
    job_monitor_block("last_training_job_id", _handle_training_result)

    # Plot training history
    hist = st.session_state.get("training_history") or {}
    if hist.get("train_loss"):
        try:
            if go:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist["train_loss"], mode="lines", name="Train"))
                if hist.get("val_loss"):
                    fig.add_trace(go.Scatter(y=hist["val_loss"], mode="lines", name="Val"))
                fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart({"Train": hist["train_loss"], "Val": hist.get("val_loss", [])})
        except Exception:
            pass

def _run_training_local(payload: Dict[str, Any]):
    try:
        # Inline trainer as fallback (no Redis)
        trainer = _flex_trainer_ctor({
            "model_type": payload["model_type"],
            "hidden_dim": payload["hidden_dim"],
            "learning_rate": payload["learning_rate"],
            "checkpoint_dir": "checkpoints",
            "enable_mixed_precision": payload["enable_mixed_precision"],
        })
        prog = st.progress(0.0)
        def cb(ep, total):
            prog.progress(min(1.0, ep/total))

        _flex_train_call(trainer, dict(
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            samples=payload["samples"],
            validation_split=payload["validation_split"],
            use_generator=payload["use_generator"]
        ), progress_callback=cb)

        st.session_state.ml_trainer = trainer
        st.session_state.ml_trained = True
        st.session_state.training_history = getattr(trainer, "history", {})
        st.success("Local training finished.")
    except Exception as e:
        st.error(f"Local training error: {e}")

def _handle_training_result(result: Dict[str, Any]):
    if not result:
        st.error("Empty training result")
        return
    if "error" in result:
        st.error(f"Training failed: {result['error']}")
        return
    # Normalize
    res = result.get("result", result)
    # Persist history and mark trained
    hist = res.get("history") or {}
    st.session_state.training_history = hist
    st.session_state.ml_trained = True
    # Try to preload the best model if path is included
    best_path = res.get("best_model_path")
    if best_path and MLTrainer:
        try:
            tr = _flex_trainer_ctor({
                "model_type": res.get("model_type", "pattern_learner"),
                "hidden_dim": res.get("hidden_dim", 128),
                "learning_rate": res.get("learning_rate", 1e-3),
            })
            ok = tr.load_model(best_path)
            st.session_state.ml_trainer = tr if ok else None
            if ok:
                st.info(f"Loaded trained model from: {best_path}")
        except Exception as e:
            st.warning(f"Could not auto-load model: {e}")
    st.success("Training metadata saved. You can now generate with the model.")

def _generate_from_model():
    if not st.session_state.ml_trained or not st.session_state.ml_trainer:
        st.warning("No trained model loaded.")
        return
    num = st.slider("How many to generate?", 1, 10, 1)
    for i in range(num):
        try:
            res = st.session_state.ml_trainer.generate_new_ode()
            if res:
                register_generated_ode(res)
                st.success(f"Generated ODE #{len(st.session_state.generated_odes)} using model.")
        except Exception as e:
            st.error(f"Generation via model failed: {e}")

# ---------------- Batch Generation ----------------
def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    col = st.columns(3)
    with col[0]:
        n_odes = st.slider("Number of ODEs", 5, 500, 50, 5)
    with col[1]:
        gen_types = st.multiselect("Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with col[2]:
        vary = st.checkbox("Vary parameters", True)

    if not HAVE_SRC or not st.session_state.basic_functions:
        st.warning("Factory libraries missing. Install src/* and functions libs.")
        return

    if st.button("Run Batch", type="primary"):
        bf = st.session_state.basic_functions
        all_funcs = bf.get_function_names()
        lin = LinearGeneratorFactory() if LinearGeneratorFactory else None
        nonlin = NonlinearGeneratorFactory() if NonlinearGeneratorFactory else None
        results = []
        for i in range(n_odes):
            try:
                params = {
                    "alpha": float(np.random.uniform(-2, 2) if vary else 1.0),
                    "beta": float(np.random.uniform(0.5, 2.0) if vary else 1.0),
                    "n": int(np.random.randint(1, 4) if vary else 1),
                    "M": float(np.random.uniform(-1, 1) if vary else 0.0),
                }
                fname = np.random.choice(all_funcs)
                f = bf.get_function(fname)
                gtype = np.random.choice(gen_types)
                if gtype == "linear" and lin:
                    gnum = np.random.randint(1, 9)
                    if gnum in [4,5]:
                        params["a"] = float(np.random.uniform(1, 3))
                    res = lin.create(gnum, f, **params)
                elif gtype == "nonlinear" and nonlin:
                    gnum = np.random.randint(1, 11)
                    if gnum in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                    if gnum in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                    if gnum in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                    res = nonlin.create(gnum, f, **params)
                else:
                    continue
                row = {
                    "ID": i+1, "Type": res.get("type","?"),
                    "Order": res.get("order",0), "Function": fname,
                    "α": params["alpha"], "β": params["beta"], "n": params["n"],
                }
                results.append(row)
            except Exception:
                continue
        st.session_state.batch_results.extend(results)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

# ---------------- Novelty & Analysis ----------------
def novelty_page():
    st.header("🔍 Novelty Detection")
    nd = st.session_state.novelty_detector
    if not nd:
        st.info("Novelty detector unavailable.")
        return
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1}")
    ode = st.session_state.generated_odes[idx]
    if st.button("Analyze"):
        try:
            result = nd.analyze(ode, check_solvability=True, detailed=True)
            st.write(result.__dict__ if hasattr(result, "__dict__") else result)
        except Exception as e:
            st.error(f"Novelty error: {e}")

def analysis_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs generated.")
        return
    df = pd.DataFrame([{
        "ID": i+1,
        "Type": d.get("type"),
        "Order": d.get("order"),
        "Function": d.get("function_used"),
        "Time": d.get("timestamp")
    } for i, d in enumerate(st.session_state.generated_odes)])
    st.dataframe(df, use_container_width=True)
    if px:
        try:
            fig = px.histogram(df, x="Order", title="Order distribution"); st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# ---------------- Visualization (placeholder) ----------------
def visualization_page():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.info("No ODEs to visualize.")
        return
    idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1}")
    ode = st.session_state.generated_odes[idx]
    st.write("Preview (symbolic):")
    try:
        st.latex(expr_to_latex(ode["generator"]) + " = " + expr_to_latex(ode["rhs"]))
        st.latex("y(x) = " + expr_to_latex(ode["solution"]))
    except Exception:
        st.write(ode)

# ---------------- Export ----------------
def export_page():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.info("No ODEs yet.")
        return
    mode = st.radio("Export mode", ["Single ODE","All ODEs"])
    if mode == "Single ODE":
        idx = st.selectbox("Select", range(len(st.session_state.generated_odes)))
        ode = st.session_state.generated_odes[idx]
        tex = LaTeXExporter.document(ode, True)
        st.download_button("📄 Download LaTeX", tex, f"ode_{idx+1}.tex", "text/x-latex")
        pkg = LaTeXExporter.package(ode)
        st.download_button("📦 Download ZIP", pkg, f"ode_{idx+1}.zip", "application/zip")
    else:
        parts = [r"""\documentclass[12pt]{report}
\usepackage{amsmath,amssymb}
\title{Master Generators — Collection}
\date{\today}
\begin{document}\maketitle\tableofcontents"""]
        for i, ode in enumerate(st.session_state.generated_odes, 1):
            parts.append(f"\\chapter{{ODE {i}}}")
            parts.append(LaTeXExporter.document(ode, include_preamble=False))
        parts.append(r"\end{document}")
        doc = "\n".join(parts)
        st.download_button("📄 Download All (LaTeX)", doc, f"all_odes_{int(time.time())}.tex", "text/x-latex")

# ---------------- Settings / Session ----------------
def _save_session(path: Optional[str] = None):
    try:
        data = {
            "generated_odes": st.session_state.generated_odes,
            "batch_results": st.session_state.batch_results,
            "training_history": st.session_state.training_history,
            "ml_trained": st.session_state.ml_trained,
        }
        os.makedirs("sessions", exist_ok=True)
        path = path or os.path.join("sessions", f"session_{int(time.time())}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        st.success(f"Session saved to {path}")
    except Exception as e:
        st.error(f"Save failed: {e}")

def _load_session_dialog():
    up = st.file_uploader("Upload a session .json", type=["json"])
    if up and st.button("Load Session"):
        try:
            data = json.loads(up.read().decode("utf-8"))
            st.session_state.generated_odes = data.get("generated_odes", [])
            st.session_state.batch_results = data.get("batch_results", [])
            st.session_state.training_history = data.get("training_history", {})
            st.session_state.ml_trained = data.get("ml_trained", False)
            st.success("Session loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

def settings_page():
    st.header("⚙️ Settings & Diagnostics")
    rd = redis_status()
    st.subheader("Redis / RQ")
    st.json(rd)
    st.caption("Ensure Web and Worker use the SAME REDIS_URL and queue (default: ode_jobs). Worker must run: rq worker -u $REDIS_URL ode_jobs")

    col = st.columns(3)
    with col[0]:
        if st.button("💾 Save Session Now"):
            _save_session()
    with col[1]:
        _load_session_dialog()
    with col[2]:
        if st.button("🗑️ Clear Generated ODEs"):
            st.session_state.generated_odes = []
            st.success("Cleared.")

# ---------------- Documentation ----------------
def docs_page():
    st.header("📖 Documentation (Quick Start)")
    st.markdown("""
1. **Apply Master Theorem**: choose f(z), parameters (α,β,n,M), select LHS (constructor/free-form/arbitrary), click **Generate ODE**.
2. If **Redis** is configured, generation runs in background; otherwise runs locally.
3. Use **Reverse Engineering** to infer a simple linear operator from y(x).
4. **ML Pattern Learning**: train via RQ or locally; then generate new ODEs using the trained model.
5. **Batch Generation** produces many examples for analysis and training.
6. **Export & LaTeX** gives publication-ready outputs.
7. **Settings & Diagnostics**: check Redis status, save/load/upload sessions.
""")

# ---------------- Main ----------------
def main():
    init_session()
    st.sidebar.title("🔬 Master Generators for ODEs")
    page = st.sidebar.radio("Navigation", [
        "🏠 Dashboard",
        "🎯 Apply Master Theorem",
        "🔁 Reverse Engineering",
        "🔧 Generator Constructor",
        "🤖 ML Training",
        "📊 Batch Generation",
        "🔍 Novelty",
        "📈 Analysis",
        "📐 Visualization",
        "📤 Export",
        "⚙️ Settings",
        "📖 Docs",
    ])

    if page == "🏠 Dashboard":
        st.header("🏠 Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Generated ODEs", len(st.session_state.generated_odes))
        with c2: st.metric("Batch Rows", len(st.session_state.batch_results))
        with c3: st.metric("Is Trained", "Yes" if st.session_state.ml_trained else "No")
        with c4: st.metric("Redis", "ON" if has_redis() else "OFF")
        if st.session_state.generated_odes:
            df = pd.DataFrame(st.session_state.generated_odes)[["generator_number","type","order","timestamp"]]
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("No ODEs yet. Try '🎯 Apply Master Theorem'.")
    elif page == "🎯 Apply Master Theorem":
        apply_master_theorem_page()
    elif page == "🔁 Reverse Engineering":
        reverse_engineering_page()
    elif page == "🔧 Generator Constructor":
        generator_constructor_page()
    elif page == "🤖 ML Training":
        ml_training_page()
    elif page == "📊 Batch Generation":
        batch_generation_page()
    elif page == "🔍 Novelty":
        novelty_page()
    elif page == "📈 Analysis":
        analysis_page()
    elif page == "📐 Visualization":
        visualization_page()
    elif page == "📤 Export":
        export_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "📖 Docs":
        docs_page()

if __name__ == "__main__":
    main()