# master_generators_app.py
import os, sys, io, json, zipfile, logging, pickle
from datetime import datetime
from typing import Any, Dict, Optional, List

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---- RQ utils (persistent) ----
try:
    from rq_utils import has_redis, enqueue_job, fetch_job, get_progress, get_logs, get_artifacts
except Exception as e:
    logger.warning(f"rq_utils missing: {e}")
    def has_redis(): return False
    def enqueue_job(*a, **k): return None
    def fetch_job(*a, **k): return None
    def get_progress(*a, **k): return {}
    def get_logs(*a, **k): return []
    def get_artifacts(*a, **k): return {}

# ---- Core engine ----
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

# ---- Optional src imports ----
MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
LinearGeneratorFactory = CompleteLinearGeneratorFactory = None
NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
GeneratorConstructor = GeneratorSpecification = None
DerivativeTerm = DerivativeType = OperatorType = None
ODEClassifier = PhysicalApplication = None
BasicFunctions = SpecialFunctions = None
MLTrainer = TrainConfig = None
ODENoveltyDetector = None

try:
    # Factories / Constructor
    try:
        from src.generators.master_generator import (MasterGenerator, EnhancedMasterGenerator, CompleteMasterGenerator)
        try:
            from src.generators.master_generator import (CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory)
        except Exception:
            from src.generators.linear_generators import (LinearGeneratorFactory, CompleteLinearGeneratorFactory)
            from src.generators.nonlinear_generators import (NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory)
    except Exception:
        from src.generators.linear_generators import (LinearGeneratorFactory, CompleteLinearGeneratorFactory)
        from src.generators.nonlinear_generators import (NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory)

    try:
        from src.generators.generator_constructor import (
            GeneratorConstructor, GeneratorSpecification,
            DerivativeTerm, DerivativeType, OperatorType
        )
    except Exception:
        pass

    # Function libs
    try:
        from src.functions.basic_functions import BasicFunctions
        from src.functions.special_functions import SpecialFunctions
    except Exception:
        pass

    # ML trainer (enhanced)
    try:
        from src.ml.trainer import MLTrainer, TrainConfig
    except Exception:
        pass

    # Novelty detector
    try:
        from src.dl.novelty_detector import ODENoveltyDetector
    except Exception:
        pass

    # Classifier
    try:
        from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
    except Exception:
        pass
except Exception as e:
    logger.warning(f"optional imports failed: {e}")

# ---- Streamlit page config ----
st.set_page_config(page_title="Master Generators for ODEs", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# ---- CSS ----
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
padding:1.2rem;border-radius:14px;margin-bottom:1.2rem;color:white;text-align:center;
box-shadow:0 10px 30px rgba(0,0,0,0.25);}
.main-title{font-size:1.9rem;font-weight:700;margin-bottom:.35rem;}
.subtitle{font-size:.98rem;opacity:.95;}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
color:white;padding:.9rem;border-radius:12px;text-align:center;
box-shadow:0 10px 20px rgba(0,0,0,0.2);}
.info-box{background:linear-gradient(135deg,#e3f2fd 0%,#bbdefb 100%);
border-left:5px solid #2196f3;padding:1rem;border-radius:10px;margin:1rem 0;}
.result-box{background:linear-gradient(135deg,#e8f5e9 0%,#c8e6c9 100%);
border:2px solid #4caf50;padding:1rem;border-radius:10px;margin:1rem 0;}
.error-box{background:linear-gradient(135deg,#ffebee 0%,#ffcdd2 100%);
border:2px solid #f44336;padding:1rem;border-radius:10px;margin:1rem 0;}
.small{font-size:0.85rem;opacity:.85;}
</style>
""", unsafe_allow_html=True)

# ---- Session init ----
def _ensure(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def _init_session():
    _ensure("generator_constructor", GeneratorConstructor() if GeneratorConstructor else None)
    _ensure("generator_terms", [])
    _ensure("generated_odes", [])
    _ensure("batch_results", [])
    _ensure("analysis_results", [])
    _ensure("export_history", [])
    _ensure("training_history", [])
    _ensure("ml_trainer", None)
    _ensure("ml_trained", False)
    _ensure("ml_trained_models", [])
    _ensure("ml_job_id", None)
    _ensure("last_compute_job_id", None)
    _ensure("lhs_source", "constructor")
    _ensure("free_terms", [])
    _ensure("arbitrary_lhs_text", "")
    _ensure("current_generator", None)
    _ensure("basic_functions", BasicFunctions() if BasicFunctions else None)
    _ensure("special_functions", SpecialFunctions() if SpecialFunctions else None)
    _ensure("ode_classifier", ODEClassifier() if ODEClassifier else None)
    _ensure("novelty_detector", ODENoveltyDetector() if ODENoveltyDetector else None)
    _ensure("loaded_trainer", None)

# ---- LaTeX Export ----
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
        parts += [r"\subsection{Verification}", r"Substitute $y(x)$ into $L$ to verify $L[y]=\mathrm{RHS}$."]

        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Compile with: pdflatex ode_document.tex\n")
        buf.seek(0)
        return buf.getvalue()

# ---- ODE register/show ----
def _register_generated_ode(res: Dict[str, Any]):
    d = dict(res)
    d.setdefault("type", "nonlinear")
    d.setdefault("order", 0)
    d.setdefault("parameters", {})
    d.setdefault("classification", {})
    d.setdefault("timestamp", datetime.now().isoformat())
    d["generator_number"] = len(st.session_state.generated_odes) + 1
    cl = dict(d.get("classification", {}))
    cl.setdefault("type", "Linear" if d["type"]=="linear" else "Nonlinear")
    cl.setdefault("order", d.get("order", 0))
    cl.setdefault("linearity", "Linear" if d["type"]=="linear" else "Nonlinear")
    cl.setdefault("field", cl.get("field", "Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    d["classification"] = cl
    try:
        if isinstance(d.get("generator"), str): d["generator"] = sp.sympify(d["generator"])
        if isinstance(d.get("rhs"), str):       d["rhs"]       = sp.sympify(d["rhs"])
        if isinstance(d.get("solution"), str):  d["solution"]  = sp.sympify(d["solution"])
        d.setdefault("ode", sp.Eq(d["generator"], d["rhs"]))
    except Exception:
        pass
    st.session_state.generated_odes.append(d)

def _show_ode_result(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h3>✅ ODE Generated Successfully</h3></div>', unsafe_allow_html=True)
    tabs = st.tabs(["📐 Equation","💡 Solution & ICs","📤 Export"])
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
            for k,v in res["initial_conditions"].items():
                try: st.latex(k + " = " + sp.latex(v))
                except Exception: st.write(k, "=", v)
        p = res.get("parameters", {})
        st.write(f"α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        if res.get("f_expr_preview"):
            st.write(f"**Function:** f(z) = {res['f_expr_preview']}")
    with tabs[2]:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res.get("generator"), "rhs": res.get("rhs"), "solution": res.get("solution"),
            "parameters": res.get("parameters", {}), "classification": res.get("classification", {}),
            "initial_conditions": res.get("initial_conditions", {}),
            "function_used": str(res.get("function_used","?")),
            "generator_number": idx, "type": res.get("type","nonlinear"), "order": res.get("order",0)
        }
        tex = LaTeXExporter.generate_latex_document(ode_data, include_preamble=True)
        st.download_button("📄 Download LaTeX", tex, f"ode_{idx}.tex", "text/x-latex", use_container_width=True)
        pkg = LaTeXExporter.create_export_package(ode_data)
        st.download_button("📦 Download ZIP (data+LaTeX)", pkg, f"ode_{idx}.zip", "application/zip", use_container_width=True)

# ---- Pages ----
def page_dashboard():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>🤖 Trained Models</h3><h1>{len(st.session_state.ml_trained_models)}</h1></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h3>📊 Batch Results</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><h3>🧵 Redis (RQ)</h3><h1>{"ON" if has_redis() else "OFF"}</h1></div>', unsafe_allow_html=True)

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    st.markdown('<div class="info-box">Build custom generators; or use Free‑form / Arbitrary LHS on the theorem page.</div>', unsafe_allow_html=True)
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found. Use Free‑form/Arbitrary LHS in Theorem page.")
        return

    with st.expander("➕ Add Generator Term", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        with c1: deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5])
        with c2:
            try: ft_list = [t.value for t in DerivativeType]
            except Exception: ft_list = ["scalar"]
            func_type = st.selectbox("Function Type", ft_list)
        with c3: coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4: power = st.number_input("Power", 1, 6, 1)
        c5,c6,c7 = st.columns(3)
        with c5:
            try: op_list = [t.value for t in OperatorType]
            except Exception: op_list = ["identity"]
            operator_type = st.selectbox("Operator Type", op_list)
        with c6: scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if operator_type in ["delay","advance"] else None
        with c7: shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay","advance"] else None

        if st.button("➕ Add Term", type="primary"):
            try:
                term = DerivativeTerm(
                    derivative_order=int(deriv_order),
                    coefficient=float(coefficient),
                    power=int(power),
                    function_type=DerivativeType(func_type) if hasattr(DerivativeType, "__call__") else func_type,
                    operator_type=OperatorType(operator_type) if hasattr(OperatorType, "__call__") else operator_type,
                    scaling=scaling, shift=shift
                )
                st.session_state.generator_terms.append(term)
                st.success("Term added.")
            except Exception as e:
                st.error(f"Add term failed: {e}")

    if st.session_state.generator_terms:
        st.subheader("📝 Current Terms")
        for i, term in enumerate(list(st.session_state.generator_terms)):
            c1,c2 = st.columns([8,1])
            with c1:
                desc = term.get_description() if hasattr(term, "get_description") else str(term)
                st.write(f"• {desc}")
            with c2:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i); st.experimental_rerun()

        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Generator {len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = gen_spec
                st.success("Specification created.")
                try: st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                except Exception: pass
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None

def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem")

    src = st.radio("Generator LHS source", ("constructor","freeform","arbitrary"),
                   index={"constructor":0,"freeform":1,"arbitrary":2}.get(st.session_state.lhs_source,0),
                   horizontal=True)
    st.session_state.lhs_source = src

    colA,colB = st.columns(2)
    with colA:
        lib_choice = st.selectbox("Function library", ["Basic","Special"], index=0)
    with colB:
        lib_inst = st.session_state.basic_functions if lib_choice == "Basic" else st.session_state.special_functions
        names = lib_inst.get_function_names() if lib_inst else []
        func_name = st.selectbox("Choose f(z)", names) if names else st.text_input("Enter f(z)", "exp(z)")

    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")
    c5,c6,c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7: st.info(f"Redis: {'ON' if has_redis() else 'OFF'}")

    # Free-form helper
    st.subheader("🧩 Free‑form LHS (Builder)")
    with st.expander("Add or reuse terms", expanded=False):
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs","asin","acos","atan","asinh","acosh","atanh","erf","erfc"], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_order = st.number_input("outer m", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale (a)", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift (b)", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": float(coef), "inner_order": int(inner_order), "wrapper": wrapper,
                    "power": int(power), "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale)>1e-14 else None,
                    "arg_shift": float(shift) if abs(shift)>1e-14 else None
                })
        if st.session_state.free_terms:
            st.write("**Current terms:**")
            for i,t in enumerate(st.session_state.free_terms):
                st.write(f"{i+1}. {t}")
            cc1,cc2 = st.columns(2)
            with cc1:
                if st.button("🧮 Use free‑form LHS"):
                    st.session_state.lhs_source = "freeform"; st.success("Selected free‑form.")
            with cc2:
                if st.button("🗑️ Clear"):
                    st.session_state.free_terms = []

    # Arbitrary SymPy
    st.subheader("✍️ Arbitrary LHS (SymPy expression)")
    st.session_state.arbitrary_lhs_text = st.text_area("Use x and y(x) (e.g., sin(y(x)) + y(x)*y(x).diff(x) - y(x/2-1))",
                                                       value=st.session_state.arbitrary_lhs_text or "", height=100)
    cva,cvb = st.columns(2)
    with cva:
        if st.button("✅ Validate"):
            try:
                sp.sympify(st.session_state.arbitrary_lhs_text)
                st.success("Parsed OK."); st.session_state.lhs_source = "arbitrary"
            except Exception as e:
                st.error(f"Parse error: {e}")
    with cvb:
        if st.button("↩️ Prefer Constructor LHS"):
            st.session_state.lhs_source = "constructor"

    # Theorem 4.2
    st.markdown("---")
    t1,t2 = st.columns([1,1])
    with t1: compute_mth = st.checkbox("Compute y^(m)(x) (Theorem 4.2)", False)
    with t2: m_order = st.number_input("m", 1, 12, 1)

    # Generate (async via RQ if available)
    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        if ComputeParams is None or compute_ode_full is None:
            st.error("Core engine not available.")
        else:
            payload = {
                "func_name": func_name, "alpha": float(alpha), "beta": float(beta),
                "n": int(n), "M": float(M), "use_exact": bool(use_exact),
                "simplify_level": simplify_level, "lhs_source": st.session_state.lhs_source,
                "freeform_terms": st.session_state.get("free_terms"),
                "arbitrary_lhs_text": st.session_state.get("arbitrary_lhs_text"),
                "function_library": lib_choice,
            }
            if has_redis():
                job_id = enqueue_job(
                    "worker.compute_job", payload,
                    job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT","3600")),
                    result_ttl=int(os.getenv("RQ_RESULT_TTL","604800"))
                )
                if job_id:
                    st.session_state.last_compute_job_id = job_id
                    st.success(f"Job submitted: {job_id}")
                else:
                    st.error("Failed to submit job (check REDIS_URL / RQ worker).")
            else:
                try:
                    p = ComputeParams(
                        func_name=func_name, alpha=float(alpha), beta=float(beta), n=int(n), M=float(M),
                        use_exact=use_exact, simplify_level=simplify_level, lhs_source=st.session_state.lhs_source,
                        constructor_lhs=None,  # only UI session knows this
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

    # Poll async job (robust visibility)
    if has_redis() and st.session_state.last_compute_job_id:
        st.markdown("### 📡 Compute Job")
        colx, coly, colz = st.columns([1,1,1])
        with colx:
            if st.button("🔄 Refresh status"):
                pass
        with coly:
            if st.button("🧹 Clear job id"):
                st.session_state.last_compute_job_id = None
        with colz:
            st.caption("Worker queue must be **ode_jobs**.")

        info = fetch_job(st.session_state.last_compute_job_id)
        prog = get_progress(st.session_state.last_compute_job_id)
        logs = get_logs(st.session_state.last_compute_job_id, start=0, end=-1)

        st.write("**Job Info:**")
        st.json(info or {})

        st.write("**Progress:**")
        st.json(prog or {})

        with st.expander("🗒️ Logs", True):
            if logs:
                st.code("\n".join(logs[-200:]), language="text")
            else:
                st.info("No logs yet.")

        if info and info.get("status") == "finished":
            res = info.get("result")
            if res:
                for k in ["generator","rhs","solution"]:
                    try: res[k] = sp.sympify(res[k])
                    except Exception: pass
                _register_generated_ode(res)
                _show_ode_result(res)
            st.session_state.last_compute_job_id = None
        elif info and info.get("status") == "failed":
            st.error("Job failed.")
            st.session_state.last_compute_job_id = None
        else:
            st.info("⏳ Still computing...")

    # Theorem 4.2 immediate
    if compute_mth and theorem_4_2_y_m_expr and get_function_expr and to_exact:
        if st.button("🧮 Compute y^{(m)}(x)", use_container_width=True):
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

def page_ml():
    st.header("🤖 ML Pattern Learning (RQ Persistent)")
    if not has_redis():
        st.info("Redis not configured; background training disabled here.")
    if not (MLTrainer and TrainConfig):
        st.warning("Trainer not found, UI shown for completeness.")

    with st.expander("🎯 Training Configuration", True):
        c1,c2,c3 = st.columns(3)
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

    col = st.columns(4)
    with col[0]:
        if has_redis() and st.button("🚀 Start Training (RQ)", type="primary"):
            payload = {
                "model_type": model_type, "hidden_dim": hidden_dim, "normalize": normalize,
                "epochs": epochs, "batch_size": batch_size, "samples": samples,
                "validation_split": validation_split, "use_generator": use_generator,
                "enable_mixed_precision": enable_amp
            }
            job_id = enqueue_job(
                "worker.train_job", payload,
                job_timeout=int(os.getenv("RQ_DEFAULT_JOB_TIMEOUT","86400")),
                result_ttl=int(os.getenv("RQ_RESULT_TTL","604800"))
            )
            if job_id:
                st.session_state.ml_job_id = job_id
                st.success(f"Training job enqueued: {job_id}")
            else:
                st.error("Failed to enqueue. Check REDIS_URL.")
    with col[1]:
        if st.session_state.ml_job_id and st.button("🔄 Refresh"):
            pass
    with col[2]:
        if st.session_state.ml_job_id and st.button("🧹 Clear Job ID"):
            st.session_state.ml_job_id = None
    with col[3]:
        st.metric("Trained Models", len(st.session_state.ml_trained_models))

    if st.session_state.ml_job_id:
        st.subheader("📡 Training Monitor")
        info = fetch_job(st.session_state.ml_job_id)
        prog = get_progress(st.session_state.ml_job_id)
        logs = get_logs(st.session_state.ml_job_id)
        arts = get_artifacts(st.session_state.ml_job_id)
        st.write("**Job Info:**"); st.json(info or {})
        st.write("**Progress:**"); st.json(prog or {})
        with st.expander("🗒️ Logs", True):
            st.code("\n".join(logs[-400:]) if logs else "No logs yet.", language="text")
        st.write("**Artifacts:**"); st.json(arts or {})
        if arts.get("session_zip"):
            try:
                with open(arts["session_zip"], "rb") as f:
                    st.download_button("⬇️ Download Session ZIP", f.read(), file_name="session.zip")
            except Exception:
                st.info("Artifact path not accessible in this process.")
        if info and info.get("status") == "finished":
            st.session_state.ml_trained = True
            st.session_state.ml_trained_models.append({
                "job_id": st.session_state.ml_job_id,
                "best_model": arts.get("best_model"),
                "session_zip": arts.get("session_zip"),
                "when": datetime.now().isoformat()
            })
            st.success("Training finished.")
            st.session_state.ml_job_id = None

    st.markdown("---")
    st.subheader("💾 Session (Upload/Load)")
    up = st.file_uploader("Upload a session ZIP to load locally", type=["zip"])
    if up and (MLTrainer and TrainConfig):
        tmp = os.path.join("artifacts", f"uploaded_{datetime.now().timestamp()}.zip")
        os.makedirs("artifacts", exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(up.read())
        # Light local loader in app (optional)
        try:
            cfg = TrainConfig(model_type="pattern_learner", hidden_dim=128, normalize=True)
            tr = MLTrainer(cfg)
            ok = False
            if hasattr(tr, "load_session_zip"):
                ok = tr.load_session_zip(tmp)
            if ok:
                st.session_state.loaded_trainer = tr
                st.success("Session loaded into memory (local demo).")
            else:
                st.info("Loaded ZIP saved; worker can resume from checkpoint if shared.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    st.markdown("---")
    st.subheader("🧪 Generate / Reverse (Local demo)")
    c1,c2 = st.columns(2)
    with c1:
        num_gen = st.slider("Generate N", 1, 10, 3)
        if st.button("🎲 Generate Params"):
            if not (MLTrainer and TrainConfig):
                st.warning("Trainer not available.")
            else:
                try:
                    cfg = TrainConfig(model_type="pattern_learner", hidden_dim=128, normalize=True)
                    tr = st.session_state.get("loaded_trainer") or MLTrainer(cfg)
                    res = tr.generate_new_ode(num=num_gen)
                    st.json(res)
                except Exception as e:
                    st.error(f"Generate failed: {e}")
    with c2:
        t = st.text_input("Reverse-engineer 12 floats", "1,1,2,0,3,1,4,2,0,0,0,0")
        if st.button("🔁 Reverse Engineer"):
            if not (MLTrainer and TrainConfig):
                st.warning("Trainer not available.")
            else:
                try:
                    vec = np.array([float(x.strip()) for x in t.split(",")], dtype=np.float32)
                    cfg = TrainConfig(model_type="pattern_learner", hidden_dim=128, normalize=True)
                    tr = st.session_state.get("loaded_trainer") or MLTrainer(cfg)
                    out = tr.reverse_engineer(vec)
                    st.json(out)
                except Exception as e:
                    st.error(f"Reverse failed: {e}")

def page_batch():
    st.header("📊 Batch ODE Generation")
    st.markdown('<div class="info-box">Generate many ODEs with your factories.</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 500, 50)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_categories = st.multiselect("Function Categories", ["Basic","Special"], default=["Basic"])
        vary = st.checkbox("Vary Parameters", True)
    with c3:
        if vary:
            alpha_range = st.slider("α range", -10.0, 10.0, (-2.0, 2.0))
            beta_range  = st.slider("β range", 0.1, 10.0, (0.5, 2.0))
            n_range     = st.slider("n range", 1, 5, (1, 3))
        else:
            alpha_range=(1.0,1.0); beta_range=(1.0,1.0); n_range=(1,1)

    if st.button("🚀 Generate Batch", type="primary"):
        if not (BasicFunctions or st.session_state.get("basic_functions")):
            st.warning("Function libraries not found.")
            return
        all_functions = []
        if "Basic" in func_categories and st.session_state.basic_functions:
            all_functions += st.session_state.basic_functions.get_function_names()
        if "Special" in func_categories and st.session_state.special_functions:
            all_functions += st.session_state.special_functions.get_function_names()[:20]
        if not all_functions:
            st.warning("No function names available.")
            return

        batch_results = []
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
                if gt == "linear" and CompleteLinearGeneratorFactory:
                    fac = CompleteLinearGeneratorFactory()
                    gen_num = np.random.randint(1, 9)
                    if gen_num in [4,5]: params["a"] = float(np.random.uniform(1,3))
                    res = fac.create(gen_num, st.session_state.basic_functions.get_function(func_name), **params)
                elif gt == "linear" and LinearGeneratorFactory:
                    fac = LinearGeneratorFactory()
                    res = fac.create(1, st.session_state.basic_functions.get_function(func_name), **params)
                elif gt == "nonlinear" and CompleteNonlinearGeneratorFactory:
                    fac = CompleteNonlinearGeneratorFactory()
                    gen_num = np.random.randint(1, 11)
                    if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                    if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                    if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                    res = fac.create(gen_num, st.session_state.basic_functions.get_function(func_name), **params)
                elif gt == "nonlinear" and NonlinearGeneratorFactory:
                    fac = NonlinearGeneratorFactory()
                    res = fac.create(1, st.session_state.basic_functions.get_function(func_name), **params)
                if not res: continue
                row = {
                    "ID": i+1, "Type": res.get("type","unknown"),
                    "Generator": res.get("generator_number","?"),
                    "Function": func_name, "Order": res.get("order",0),
                    "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                }
                batch_results.append(row)
            except Exception as e:
                logger.debug(f"Batch item {i}: {e}")

        st.session_state.batch_results.extend(batch_results)
        df = pd.DataFrame(batch_results)
        st.success(f"Generated {len(batch_results)} ODEs.")
        st.dataframe(df, use_container_width=True)

def page_novelty():
    st.header("🔍 Novelty Detection")
    if not st.session_state.get("novelty_detector"):
        st.warning("Novelty detector not found.")
        return
    method = st.radio("Input", ["Current LHS", "Enter ODE", "From Generated"])
    target = None
    if method == "Current LHS":
        gs = st.session_state.get("current_generator")
        if gs is not None and hasattr(gs, "lhs"):
            target = {"ode": gs.lhs, "type":"custom", "order": getattr(gs, "order", 2)}
        else:
            st.warning("No current generator.")
    elif method == "Enter ODE":
        ode_str = st.text_area("ODE (LaTeX or text)")
        if ode_str:
            target = {"ode": ode_str, "type":"manual", "order": st.number_input("Order", 1, 10, 2)}
    else:
        if st.session_state.generated_odes:
            sel = st.selectbox("Select", range(len(st.session_state.generated_odes)),
                               format_func=lambda i: f"ODE {i+1} ({st.session_state.generated_odes[i].get('type','?')})")
            target = st.session_state.generated_odes[sel]

    if target and st.button("Analyze", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                analysis = st.session_state.novelty_detector.analyze(target, check_solvability=True, detailed=True)
                st.metric("Novelty", "🟢 NOVEL" if analysis.is_novel else "🔴 STANDARD")
                st.metric("Score", f"{analysis.novelty_score:.1f}/100")
                st.metric("Confidence", f"{analysis.confidence:.1%}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

def page_analysis():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes:
        st.info("No ODEs yet.")
        return
    if not st.session_state.get("ode_classifier"):
        st.warning("Classifier not found.")
        return
    st.subheader("Overview")
    rows = []
    for i, o in enumerate(st.session_state.generated_odes[-100:]):
        rows.append({"ID": i+1, "Type": o.get("type","?"), "Order": o.get("order",0),
                     "Generator": o.get("generator_number","?"),
                     "Function": o.get("function_used","?"),
                     "Timestamp": o.get("timestamp","")[:19]})
    df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)

def page_visual():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs."); return
    idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)))
    x = np.linspace(-5, 5, 500)
    y = np.sin(x) * np.exp(-0.1*np.abs(x))
    if _PLOTLY_OK:
        fig = go.Figure(); fig.add_trace(go.Scatter(x=x,y=y,mode="lines"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pd.DataFrame({"x":x,"y":y}).set_index("x"))

def page_export():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to export."); return
    idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)))
    ode = st.session_state.generated_odes[idx]
    tex = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
    st.download_button("📄 Download LaTeX", tex, f"ode_{idx+1}.tex", "text/x-latex")
    pkg = LaTeXExporter.create_export_package(ode)
    st.download_button("📦 Download ZIP", pkg, f"ode_{idx+1}.zip", "application/zip")

def page_examples():
    st.header("📚 Examples")
    with st.expander("Simple Harmonic Oscillator"): st.latex("y'' + y = 0")

def page_reverse():
    st.header("🔁 Reverse Engineering (Local)")
    txt = st.text_input("12 floats", "1,1,2,0,3,1,4,2,0,0,0,0")
    if st.button("Run"):
        if not (MLTrainer and TrainConfig):
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

def page_settings():
    st.header("⚙️ Settings")
    if st.button("Save Session Snapshot"):
        try:
            snap = {
                "generated_odes": st.session_state.get("generated_odes", []),
                "batch_results": st.session_state.get("batch_results", []),
                "training_history": st.session_state.get("training_history", {}),
                "ml_trained_models": st.session_state.get("ml_trained_models", []),
            }
            with open("session_state.pkl","wb") as f: pickle.dump(snap, f)
            st.success("Saved session_state.pkl")
        except Exception as e:
            st.error(f"Failed: {e}")

def page_docs():
    st.header("📖 Documentation")
    st.markdown("""
1. **Apply Master Theorem** → choose f(z), parameters, and LHS source (Constructor/Free‑form/Arbitrary).
2. Click **Generate ODE**. With Redis configured, job runs in background; otherwise it runs locally.
3. Use **ML Pattern Learning** to train in the background (progress/logs/artifacts persist).
4. **Reverse Engineering** allows parameter inference from feature vectors.
5. Export LaTeX/ZIP from **Export & LaTeX**.
""")

# ---- Main ----
def main():
    _init_session()
    st.markdown("""
    <div class="main-header">
      <div class="main-title">🔬 Master Generators for ODEs</div>
      <div class="subtitle">Free‑form/Arbitrary generators • Theorem 4.1/4.2 • ML/DL • Export • Novelty • RQ jobs</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard","🔧 Generator Constructor","🎯 Apply Master Theorem","🤖 ML Pattern Learning",
        "📊 Batch Generation","🔍 Novelty Detection","📈 Analysis & Classification","📐 Visualization",
        "📤 Export & LaTeX","📚 Examples","🔁 Reverse Engineering","⚙️ Settings","📖 Documentation"
    ])

    if page == "🏠 Dashboard": page_dashboard()
    elif page == "🔧 Generator Constructor": page_generator_constructor()
    elif page == "🎯 Apply Master Theorem": page_apply_master_theorem()
    elif page == "🤖 ML Pattern Learning": page_ml()
    elif page == "📊 Batch Generation": page_batch()
    elif page == "🔍 Novelty Detection": page_novelty()
    elif page == "📈 Analysis & Classification": page_analysis()
    elif page == "📐 Visualization": page_visual()
    elif page == "📤 Export & LaTeX": page_export()
    elif page == "📚 Examples": page_examples()
    elif page == "🔁 Reverse Engineering": page_reverse()
    elif page == "⚙️ Settings": page_settings()
    elif page == "📖 Documentation": page_docs()

if __name__ == "__main__":
    main()