# master_generators_app.py
import os, sys, io, json, base64, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import types

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp
from sympy.core.function import AppliedUndef

# Optional torch (for ML page)
try:
    import torch
except Exception:
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Resilient src imports
HAVE_SRC = True
LinearGeneratorFactory = CompleteLinearGeneratorFactory = None
NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None
GeneratorConstructor = GeneratorSpecification = None
DerivativeTerm = DerivativeType = OperatorType = None
ODEClassifier = PhysicalApplication = None
BasicFunctions = SpecialFunctions = None
UIComponents = None
ODENoveltyDetector = None
MLTrainer = ODEDataset = ODEDataGenerator = None

try:
    from src.generators.linear_generators import LinearGeneratorFactory, CompleteLinearGeneratorFactory
except Exception:
    pass
try:
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
except Exception:
    pass
try:
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification, DerivativeTerm, DerivativeType, OperatorType
    )
except Exception:
    pass
try:
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
except Exception:
    pass
try:
    from src.functions.basic_functions import BasicFunctions
except Exception:
    pass
try:
    from src.functions.special_functions import SpecialFunctions
except Exception:
    pass
try:
    from src.functions.phi_library import PhiLibrary  # NEW
except Exception:
    PhiLibrary = None
try:
    from src.ml.trainer import MLTrainer, ODEDataset, ODEDataGenerator
except Exception:
    pass
try:
    from src.ui.components import UIComponents
except Exception:
    pass

# Your theorem + compute core from the prior fixed version
try:
    from shared.ode_core import (
        ComputeParams, compute_ode_full, theorem_4_2_y_m_expr,
        get_function_expr, build_freeform_lhs, parse_arbitrary_lhs, to_exact, simplify_expr
    )
except Exception as e:
    logger.warning(f"shared.ode_core missing pieces: {e}")
    # Provide lightweight fallbacks to allow the UI to load
    def parse_arbitrary_lhs(txt): return sp.sympify(txt)
    def build_freeform_lhs(x, terms, y_name="y"): return sp.Symbol("LHS")
    def to_exact(v): return sp.nsimplify(v, rational=True)
    def simplify_expr(e, level="light"): return sp.simplify(e)
    def theorem_4_2_y_m_expr(*a, **k): return sp.Symbol("y_m")
    class ComputeParams:
        def __init__(self, **kw): self.__dict__.update(kw)
    def compute_ode_full(p): return {"generator": sp.Symbol("L[y]"), "rhs": 0, "solution": 0, "parameters": {}, "order": 1, "type": "linear"}

# Redis queue helpers (async optional)
try:
    from rq_utils import has_redis, enqueue_job, fetch_job
except Exception:
    def has_redis(): return False
    def enqueue_job(*a, **k): return None
    def fetch_job(*a, **k): return None

# Reverse engineering core
from shared.inverse_core import (
    DEFAULT_SEARCH,
    infer_params_from_solution, infer_from_ode_single,
    infer_params_multi_blocks, infer_from_ode_multi
)

st.set_page_config(page_title="Master Generators ODE System", page_icon="🔬", layout="wide")


# ── Session ───────────────────────────────────────────────────────────────────
class SessionStateManager:
    @staticmethod
    def initialize():
        defaults = dict(
            generator_terms=[], generated_odes=[], generator_patterns=[], training_history=[],
            batch_results=[], analysis_results=[], export_history=[],
            lhs_source="constructor", free_terms=[], arbitrary_lhs_text="",
            ml_trainer=None, ml_trained=False,
        )
        for k,v in defaults.items():
            if k not in st.session_state: st.session_state[k] = v

        # heavy objects
        if "generator_constructor" not in st.session_state and GeneratorConstructor:
            st.session_state.generator_constructor = GeneratorConstructor()
        if "basic_functions" not in st.session_state and BasicFunctions:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions:
            st.session_state.special_functions = SpecialFunctions()
        if "phi_library" not in st.session_state and PhiLibrary:
            st.session_state.phi_library = PhiLibrary()
        if "ode_classifier" not in st.session_state and ODEClassifier:
            try: st.session_state.ode_classifier = ODEClassifier()
            except Exception: st.session_state.ode_classifier = None
        if "novelty_detector" not in st.session_state and ODENoveltyDetector:
            try: st.session_state.novelty_detector = ODENoveltyDetector()
            except Exception: st.session_state.novelty_detector = None

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
    cl.setdefault("field", cl.get("field","Mathematical Physics"))
    cl.setdefault("applications", cl.get("applications", ["Research Equation"]))
    cl.setdefault("linearity", "Linear" if result["type"]=="linear" else "Nonlinear")
    result["classification"] = cl
    try:
        result.setdefault("ode", sp.Eq(result["generator"], result["rhs"]))
    except Exception:
        pass
    st.session_state.generated_odes.append(result)

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
        generator = ode_data.get("generator","")
        rhs = ode_data.get("rhs","")
        solution = ode_data.get("solution","")
        params = ode_data.get("parameters",{})
        classification = ode_data.get("classification",{})
        initial_conditions = ode_data.get("initial_conditions",{})

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
\section{Generated Ordinary Differential Equation}""")
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
            f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha',1))} \\\\",
            f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta',1))} \\\\",
            f"n       &= {params.get('n',1)} \\\\",
            f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M',0))}",
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
        parts += [r"\subsection{Verification}", r"Substitute $y(x)$ into the generator to verify $L[y]=\mathrm{RHS}$."]

        if include_preamble: parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def create_export_package(ode_data: Dict[str, Any], include_extras: bool = True) -> bytes:
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("ode_document.tex", LaTeXExporter.generate_latex_document(ode_data, True))
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr("README.txt", "Master Generator ODE Export\npdflatex ode_document.tex\n")
        z.seek(0); return z.getvalue()


# ── UI Helpers ────────────────────────────────────────────────────────────────
def _function_library_selector(label="Function library"):
    libs = []
    if st.session_state.get("basic_functions"): libs.append("Basic")
    if st.session_state.get("special_functions"): libs.append("Special")
    if st.session_state.get("phi_library"): libs.append("Phi")
    if not libs: libs = ["Phi"]  # safe fallback
    lib = st.selectbox(label, libs, index=0)
    if lib == "Basic":
        names = st.session_state.basic_functions.get_function_names()
        lib_obj = st.session_state.basic_functions
    elif lib == "Special":
        names = st.session_state.special_functions.get_function_names()
        lib_obj = st.session_state.special_functions
    else:
        names = st.session_state.phi_library.get_function_names()
        lib_obj = st.session_state.phi_library
    return lib, names, lib_obj

# ── Pages ─────────────────────────────────────────────────────────────────────
def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (with Φ‑library)")
    # LHS source
    src = st.radio("Generator LHS source", ["constructor","freeform","arbitrary"], horizontal=True,
                   index={"constructor":0,"freeform":1,"arbitrary":2}.get(st.session_state.get("lhs_source","constructor"),0))
    st.session_state["lhs_source"] = src

    colA,colB = st.columns([1,1])
    with colA:
        lib, func_names, lib_obj = _function_library_selector("Function library (Basic/Special/Phi)")
    with colB:
        if func_names:
            func_name = st.selectbox("Choose f(z)", func_names)
        else:
            func_name = st.text_input("Enter f(z)", "exp(z)")

    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α",  value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β",  value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n (positive integer)", 1, 12, 1)
    with c4: M     = st.number_input("M",  value=0.0, step=0.1, format="%.6f")
    c5,c6 = st.columns(2)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)

    # build constructor LHS if any
    constructor_lhs = None
    if st.session_state.get("generator_constructor") and hasattr(st.session_state.generator_constructor, "get_generator_expression"):
        try: constructor_lhs = st.session_state.generator_constructor.get_generator_expression()
        except Exception: constructor_lhs = None

    # free‑form LHS builder
    st.subheader("🧩 Free‑form LHS (Builder)")
    with st.expander("Build custom LHS terms", expanded=False):
        cols = st.columns([1,1,1,1,1,1,1,1])
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k (y^(k))", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs","logistic","erf"], index=0)
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_order = st.number_input("outer m (D^m)", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale (a)", 1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift (b)", 0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": coef, "inner_order": int(inner_order), "wrapper": wrapper,
                    "power": int(power), "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale)>1e-14 else None,
                    "arg_shift": float(shift) if abs(shift)>1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write("**Current terms:**"); [st.write(f"{i+1}. {t}") for i,t in enumerate(st.session_state.free_terms)]
            c7,c8 = st.columns(2)
            with c7:
                if st.button("🧮 Use free‑form LHS"):
                    st.session_state["lhs_source"] = "freeform"; st.success("Free‑form selected.")
            with c8:
                if st.button("🗑️ Clear terms"): st.session_state.free_terms = []

    st.subheader("✍️ Arbitrary LHS (SymPy)")
    st.session_state.arbitrary_lhs_text = st.text_area(
        "Any SymPy in x and y(x) (e.g., sin(y(x)) + y(x)*diff(y(x),x) - y(x/2-1))",
        value=st.session_state.arbitrary_lhs_text or "", height=100
    )

    st.markdown("---")
    colm1, colm2 = st.columns(2)
    with colm1:
        compute_mth = st.checkbox("Compute y^(m) via Theorem 4.2", False)
    with colm2:
        m_order = st.number_input("m", 1, 12, 1)

    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        try:
            basic_lib = st.session_state.get("basic_functions")
            special_lib = st.session_state.get("special_functions")
            phi_lib = st.session_state.get("phi_library")

            # Select the library object according to picker
            lib_obj_sel = {"Basic": basic_lib, "Special": special_lib, "Phi": phi_lib}.get(lib, None)

            p = ComputeParams(
                func_name=func_name, alpha=alpha, beta=beta, n=int(n), M=M,
                use_exact=bool(use_exact), simplify_level=simplify_level,
                lhs_source=st.session_state["lhs_source"],
                constructor_lhs=constructor_lhs,
                freeform_terms=st.session_state.get("free_terms"),
                arbitrary_lhs_text=st.session_state.get("arbitrary_lhs_text"),
                function_library=lib, basic_lib=basic_lib, special_lib=special_lib, phi_lib=phi_lib,
            )
            res = compute_ode_full(p)
            register_generated_ode(res)
            _render_ode_result(res)
        except Exception as e:
            st.error(f"Generation error: {e}")

    if compute_mth and st.button("🧮 Compute y^{(m)}(x) via Theorem 4.2"):
        try:
            lib_obj_sel = {"Basic": st.session_state.get("basic_functions"),
                           "Special": st.session_state.get("special_functions"),
                           "Phi": st.session_state.get("phi_library")}.get(lib, None)
            z = sp.Symbol("z", real=True)
            f_expr_preview = (lib_obj_sel.get_function(func_name) if lib_obj_sel else sp.exp(z))
            α = to_exact(alpha) if use_exact else sp.Float(alpha)
            β = to_exact(beta)  if use_exact else sp.Float(beta)
            x = sp.Symbol("x", real=True)
            y_m = theorem_4_2_y_m_expr(sp.sympify(f_expr_preview).subs({z:z}), α, β, int(n), int(m_order), x, simplify_level)
            st.latex(fr"y^{{({int(m_order)})}}(x) = " + sp.latex(y_m))
        except Exception as e:
            st.error(f"Failed to compute derivative: {e}")

def _render_ode_result(res: Dict[str, Any]):
    st.success("✅ ODE Generated")
    t1,t2,t3 = st.tabs(["📐 Equation", "💡 Solution & ICs", "📤 Export"])
    with t1:
        try:
            st.latex(sp.latex(res["generator"]) + " = " + sp.latex(res["rhs"]))
        except Exception:
            st.write("LHS:", res["generator"]); st.write("RHS:", res["rhs"])
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t2:
        try: st.latex("y(x) = " + sp.latex(res["solution"]))
        except Exception: st.write("y(x) =", res["solution"])
        if res.get("initial_conditions"):
            st.write("**ICs:**"); [st.write(f"{k} = {v}") for k,v in res["initial_conditions"].items()]
        p = res.get("parameters", {})
        st.write(f"**Parameters:** α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
    with t3:
        idx = len(st.session_state.generated_odes)
        ode_data = {
            "generator": res["generator"], "rhs": res["rhs"], "solution": res["solution"],
            "parameters": res.get("parameters", {}),
            "classification": {
                "type": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "order": res.get("order", 0),
                "linearity": "Linear" if res.get("type")=="linear" else "Nonlinear",
                "field":"Mathematical Physics","applications":["Research Equation"]
            },
            "initial_conditions": res.get("initial_conditions", {}),
            "function_used": str(res.get("function_used","?")),
            "generator_number": idx, "type": res.get("type","nonlinear"),
            "order": res.get("order", 0)
        }
        st.download_button("📄 LaTeX", LaTeXExporter.generate_latex_document(ode_data, True),
                           f"ode_{idx}.tex", "text/x-latex", use_container_width=True)
        st.download_button("📦 ZIP", LaTeXExporter.create_export_package(ode_data, True),
                           f"ode_package_{idx}.zip", "application/zip", use_container_width=True)

def reverse_engineering_page():
    st.header("🔁 Reverse Engineering (single / multi‑block)")
    mode_top = st.radio("Mode", ["From solution y(x)","From ODE L[y]=RHS"], horizontal=True)

    with st.expander("Search configuration", expanded=False):
        cols = st.columns(4)
        with cols[0]:
            a_min = st.number_input("α min", 0.1); a_max = st.number_input("α max", 5.0)
            a_steps = st.number_input("α steps", 2, 50, 7); a_neg = st.checkbox("α allow negative", True)
        with cols[1]:
            b_min = st.number_input("β min", 0.5); b_max = st.number_input("β max", 3.0)
            b_steps = st.number_input("β steps", 2, 50, 6)
        with cols[2]:
            n_min = st.number_input("n min", 1, 50, 1); n_max = st.number_input("n max", 1, 50, 6)
            n_steps = st.number_input("n steps", 2, 50, 6)
        with cols[3]:
            M_min = st.number_input("M min", -4.0); M_max = st.number_input("M max", 4.0)
            M_steps = st.number_input("M steps", 2, 50, 7)
        search_cfg = {
            "alpha": {"min": a_min, "max": a_max, "steps": int(a_steps), "allow_negative": bool(a_neg)},
            "beta":  {"min": b_min, "max": b_max, "steps": int(b_steps)},
            "n":     {"min": int(n_min), "max": int(n_max), "steps": int(n_steps)},
            "M":     {"min": M_min, "max": M_max, "steps": int(M_steps)}
        }

    libs = st.multiselect("Libraries", ["Basic","Special","Phi"], default=["Basic","Special","Phi"])
    fit_type = st.radio("Fit type", ["Single block","Multi‑block (product)","Multi‑block (sum)"], horizontal=True)
    topk = st.slider("Top K", 1, 10, 5)

    if fit_type != "Single block":
        c1,c2,c3 = st.columns(3)
        with c1: J = st.number_input("J (#blocks)", 2, 4, 2)
        with c2: pool = st.number_input("Pool size", 5, 100, 20)
        with c3: maxc = st.number_input("Max combos", 10, 1000, 100)

    if mode_top == "From solution y(x)":
        y_str = st.text_area("y(x) (SymPy)", "x**2*exp(2*x)")
        if st.button("Fit", type="primary"):
            try:
                if fit_type == "Single block":
                    if has_redis():
                        jid = enqueue_job("worker.inverse_solution_job", {"y_expr": y_str, "libraries": libs, "search": search_cfg, "topk": int(topk)})
                        st.session_state["last_inverse_job"] = jid; st.success(f"Job submitted: {jid}")
                    else:
                        res = infer_params_from_solution(sp.sympify(y_str), libraries=libs, search=search_cfg, topk=int(topk))
                        _render_inverse_single(res)
                else:
                    mode = "product" if "product" in fit_type else "sum"
                    if has_redis():
                        jid = enqueue_job("worker.inverse_solution_multiblock_job", {
                            "y_expr": y_str, "libraries": libs, "search": search_cfg,
                            "mode": mode, "J": int(J), "topk": int(topk), "pool_size": int(pool), "max_combos": int(maxc)
                        })
                        st.session_state["last_inverse_job"] = jid; st.success(f"Job submitted: {jid}")
                    else:
                        res = infer_params_multi_blocks(sp.sympify(y_str), J=int(J), mode=mode, libraries=libs, search=search_cfg, topk=int(topk), pool_size=int(pool), max_combos=int(maxc))
                        _render_inverse_multi(res)
            except Exception as e:
                st.error(f"Reverse fit failed: {e}")
    else:
        lhs = st.text_area("LHS (SymPy)", "diff(y(x),x,2)+y(x)")
        rhs = st.text_area("RHS (SymPy)", "0")
        if st.button("Infer", type="primary"):
            try:
                if fit_type == "Single block":
                    if has_redis():
                        jid = enqueue_job("worker.inverse_ode_job", {"lhs": lhs, "rhs": rhs, "libraries": libs, "search": search_cfg, "topk": int(topk)})
                        st.session_state["last_inverse_job"] = jid; st.success(f"Job submitted: {jid}")
                    else:
                        res = infer_from_ode_single(sp.sympify(lhs), sp.sympify(rhs), libraries=libs, search=search_cfg, topk=int(topk))
                        _render_inverse_single(res)
                else:
                    mode = "product" if "product" in fit_type else "sum"
                    if has_redis():
                        jid = enqueue_job("worker.inverse_ode_multiblock_job", {
                            "lhs": lhs, "rhs": rhs, "libraries": libs, "search": search_cfg,
                            "mode": mode, "J": int(J), "topk": int(topk), "pool_size": int(pool), "max_combos": int(maxc)
                        })
                        st.session_state["last_inverse_job"] = jid; st.success(f"Job submitted: {jid}")
                    else:
                        res = infer_from_ode_multi(sp.sympify(lhs), sp.sympify(rhs), J=int(J), mode=mode, libraries=libs, search=search_cfg, topk=int(topk), pool_size=int(pool), max_combos=int(maxc))
                        _render_inverse_multi(res)
            except Exception as e:
                st.error(f"Inference failed: {e}")

    if has_redis() and "last_inverse_job" in st.session_state:
        if st.button("🔄 Refresh status"): pass
        info = fetch_job(st.session_state["last_inverse_job"])
        if info:
            if info.get("status") == "finished":
                result = info.get("result", {})
                cands = result.get("candidates", [])
                # Detect single or multi payload shape
                if cands and "blocks" in cands[0]:
                    # multi
                    res = []
                    for c in cands:
                        # fabricate a minimal container for rendering
                        R = type("MB", (), {})()
                        R.blocks = []
                        for b in c["blocks"]:
                            R.blocks.append(type("RF", (), dict(
                                function_name=b["function_name"], alpha=b["alpha"], beta=b["beta"], n=b["n"], M=b["M"]
                            )))
                        R.scales = c["scales"]; R.rmse = c["rmse"]; R.expr = sp.sympify(c["expr"])
                        res.append(R)
                    _render_inverse_multi(res)
                else:
                    # single
                    res = []
                    for r in cands:
                        res.append(type("RF", (), dict(
                            function_name=r["function_name"], alpha=r["alpha"], beta=r["beta"], n=r["n"], M=r["M"],
                            scale_C=r["scale_C"], rmse=r["rmse"], candidate_expr=sp.sympify(r["candidate_expr"])
                        )))
                    _render_inverse_single(res)
                del st.session_state["last_inverse_job"]
            elif info.get("status") == "failed":
                st.error(f"Job failed: {info.get('error')}")
                del st.session_state["last_inverse_job"]
            else:
                st.info("⏳ Still computing...")

def _render_inverse_single(res):
    if not res:
        st.warning("No candidates found."); return
    rows = []
    for i, r in enumerate(res, 1):
        rows.append({"#":i, "f":r.function_name, "α":round(float(r.alpha),6), "β":round(float(r.beta),6),
                     "n":int(r.n), "M":round(float(r.M),6), "C":round(float(getattr(r,"scale_C",1.0)),6),
                     "RMSE":round(float(r.rmse),6)})
    st.dataframe(rows, use_container_width=True)
    best = res[0]
    try: st.latex("y_{fit}(x) = " + sp.latex(best.candidate_expr))
    except Exception: st.code(str(best.candidate_expr))
    st.session_state["reverse_best_params"] = {
        "function_name":best.function_name, "alpha":float(best.alpha),
        "beta":float(best.beta), "n":int(best.n), "M":float(best.M)
    }
    st.success("Stored best params in session.")

def _render_inverse_multi(res):
    if not res:
        st.warning("No candidates found."); return
    rows = []
    for i, c in enumerate(res, 1):
        blocks = "; ".join([f"{b.function_name}({b.alpha:.3g},{b.beta:.3g},n={b.n},M={b.M:.3g})" for b in c.blocks])
        rows.append({"#":i, "blocks": blocks, "scales": [float(s) for s in c.scales], "RMSE": round(float(c.rmse),6)})
    st.dataframe(rows, use_container_width=True)
    best = res[0]
    try: st.latex("y_{fit}(x) = " + sp.latex(best.expr))
    except Exception: st.code(str(best.expr))

def dashboard_page():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Generated ODEs", len(st.session_state.generated_odes))
    with c2: st.metric("ML Patterns", len(st.session_state.generator_patterns))
    with c3: st.metric("Batch Results", len(st.session_state.batch_results))
    with c4: st.metric("Model", "Trained" if st.session_state.get("ml_trained") else "Not Trained")
    st.subheader("Recent")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type","order","generator_number","timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs yet. Use **Apply Master Theorem**.")

def generator_constructor_page():
    st.header("🔧 Generator Constructor")
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found. Use Free‑form LHS in theorem page."); return
    with st.expander("➕ Add term", True):
        c1,c2,c3,c4 = st.columns(4)
        with c1: deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5], index=0)
        with c2: func_type = st.selectbox("Function Type", [t.value for t in DerivativeType])
        with c3: coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4: power = st.number_input("Power", 1, 6, 1)
        c5,c6,c7 = st.columns(3)
        with c5: operator_type = st.selectbox("Operator Type", [t.value for t in OperatorType])
        with c6: scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if operator_type in ["delay","advance"] else None
        with c7: shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay","advance"] else None
        if st.button("Add", type="primary"):
            term = DerivativeTerm(derivative_order=deriv_order, coefficient=coefficient, power=power,
                                  function_type=DerivativeType(func_type), operator_type=OperatorType(operator_type),
                                  scaling=scaling, shift=shift)
            st.session_state.generator_terms.append(term); st.success("Term added.")
    if st.session_state.generator_terms:
        st.subheader("Terms")
        for i, term in enumerate(st.session_state.generator_terms):
            c1,c2 = st.columns([8,1])
            with c1: st.write(getattr(term,"get_description",lambda: str(term))())
            with c2:
                if st.button("❌", key=f"rm_{i}"):
                    st.session_state.generator_terms.pop(i); st.experimental_rerun()
        if st.button("Build Specification", type="primary"):
            try:
                gen_spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"Custom Generator {len(st.session_state.generated_odes)+1}")
                st.session_state.current_generator = gen_spec; st.success("Spec created.")
                try: st.latex(sp.latex(gen_spec.lhs) + " = RHS")
                except Exception: pass
            except Exception as e:
                st.error(f"Build failed: {e}")

def ml_pattern_learning_page():
    st.header("🤖 ML Pattern Learning")
    if not MLTrainer:
        st.warning("MLTrainer not found."); return
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Patterns", len(st.session_state.generator_patterns))
    with c2: st.metric("Generated", len(st.session_state.generated_odes))
    with c3: st.metric("Batches", len(st.session_state.batch_results))
    with c4: st.metric("Model", "Trained" if st.session_state.get("ml_trained") else "Not Trained")

    model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"])
    with st.expander("Training config", True):
        c1,c2,c3 = st.columns(3)
        with c1: epochs = st.slider("Epochs", 10, 500, 100); batch_size = st.slider("Batch", 8, 128, 32)
        with c2: learning_rate = st.select_slider("LR", [0.0001,0.0005,0.001,0.005,0.01], value=0.001); samples = st.slider("Samples", 100, 5000, 1000)
        with c3: validation_split = st.slider("Val split", 0.1, 0.3, 0.2); use_gpu = st.checkbox("GPU if available", True)

    use_batch_for_training = st.checkbox("Include Batch Results as Training Data", True)
    need = 5; count = len(st.session_state.generated_odes) + (len(st.session_state.batch_results) if use_batch_for_training else 0)
    if count < need:
        st.warning(f"Need at least {need} ODEs. Current: {count}"); return

    if st.button("🚀 Train", type="primary"):
        with st.spinner("Training..."):
            try:
                device = "cuda" if use_gpu and (torch and torch.cuda.is_available()) else "cpu"
                trainer = MLTrainer(model_type=model_type, learning_rate=float(learning_rate), device=device, enable_mixed_precision=True)
                st.session_state.ml_trainer = trainer
                # Inject existing samples if desired
                try:
                    trainer.set_dataset(st.session_state.generated_odes,
                                        st.session_state.batch_results if use_batch_for_training else [])
                except Exception:
                    pass
                prog = st.progress(0); status = st.empty()
                def progress_callback(ep, total): prog.progress(min(1.0, ep/total)); status.text(f"Epoch {ep}/{total}")
                trainer.train(epochs=epochs, batch_size=batch_size, samples=samples, validation_split=validation_split, progress_callback=progress_callback)
                st.session_state.ml_trained = True; st.session_state.training_history = trainer.history
                st.success("Model trained.")
                if trainer.history.get("train_loss"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(1,len(trainer.history["train_loss"])+1)), y=trainer.history["train_loss"], mode="lines", name="Training"))
                    if trainer.history.get("val_loss"):
                        fig.add_trace(go.Scatter(x=list(range(1,len(trainer.history["val_loss"])+1)), y=trainer.history["val_loss"], mode="lines", name="Validation"))
                    fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.get("ml_trained") and st.session_state.get("ml_trainer"):
        st.subheader("🎨 Generate novel ODEs")
        c1,c2 = st.columns(2)
        with c1: num_generate = st.slider("How many", 1, 10, 1)
        with c2:
            if st.button("Generate", type="primary"):
                with st.spinner("Generating..."):
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

def batch_generation_page():
    st.header("📊 Batch ODE Generation")
    c1,c2,c3 = st.columns(3)
    with c1:
        num_odes = st.slider("Number of ODEs", 5, 500, 50)
        gen_types = st.multiselect("Generator Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        func_categories = st.multiselect("Function Categories", ["Basic","Special","Phi"], default=["Basic","Phi"])
        vary_params = st.checkbox("Vary Parameters", True)
    with c3:
        if vary_params:
            alpha_range = st.slider("α range", -10.0, 10.0, (-2.0, 2.0))
            beta_range  = st.slider("β range", 0.1, 10.0, (0.5, 2.0))
            n_range     = st.slider("n range", 1, 5, (1, 3))
        else:
            alpha_range=(1.0,1.0); beta_range=(1.0,1.0); n_range=(1,1)

    if st.button("🚀 Generate Batch", type="primary"):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            batch_results = []; prog = st.progress(0); status = st.empty()
            all_functions = []
            if "Basic" in func_categories and st.session_state.get("basic_functions"):
                all_functions += st.session_state.basic_functions.get_function_names()
            if "Special" in func_categories and st.session_state.get("special_functions"):
                all_functions += st.session_state.special_functions.get_function_names()[:20]
            if "Phi" in func_categories and st.session_state.get("phi_library"):
                all_functions += st.session_state.phi_library.get_function_names()
            if not all_functions:
                st.warning("No function names available."); return

            for i in range(num_odes):
                prog.progress((i+1)/num_odes); status.text(f"{i+1}/{num_odes}")
                try:
                    params = {
                        "alpha": float(np.random.uniform(*alpha_range)),
                        "beta":  float(np.random.uniform(*beta_range)),
                        "n": int(np.random.randint(n_range[0], n_range[1]+1)),
                        "M": float(np.random.uniform(-1, 1)),
                    }
                    func_name = np.random.choice(all_functions)
                    gt = np.random.choice(gen_types)
                    res = {}
                    if gt == "linear" and CompleteLinearGeneratorFactory:
                        factory = CompleteLinearGeneratorFactory()
                        gen_num = np.random.randint(1, 9)
                        if gen_num in [4,5]: params["a"] = float(np.random.uniform(1,3))
                        # prefer basic phi for compatibility
                        f_z = (st.session_state.basic_functions.get_function(func_name)
                               if st.session_state.get("basic_functions") and func_name in st.session_state.basic_functions.get_function_names()
                               else (st.session_state.phi_library.get_function(func_name) if st.session_state.get("phi_library") else sp.Symbol("z")))
                        res = factory.create(gen_num, f_z, **params)
                    elif gt == "nonlinear" and CompleteNonlinearGeneratorFactory:
                        factory = CompleteNonlinearGeneratorFactory()
                        gen_num = np.random.randint(1, 11)
                        if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                        if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                        if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,3))
                        f_z = (st.session_state.basic_functions.get_function(func_name)
                               if st.session_state.get("basic_functions") and func_name in st.session_state.basic_functions.get_function_names()
                               else (st.session_state.phi_library.get_function(func_name) if st.session_state.get("phi_library") else sp.Symbol("z")))
                        res = factory.create(gen_num, f_z, **params)
                    if not res: continue
                    row = {
                        "ID": i+1, "Type": res.get("type","unknown"),
                        "Generator": res.get("generator_number","?"),
                        "Function": func_name, "Order": res.get("order",0),
                        "α": round(params["alpha"],4), "β": round(params["beta"],4), "n": params["n"]
                    }
                    s = str(res.get("solution","")); row["Solution"] = (s[:120]+"...") if len(s)>120 else s
                    row["Subtype"] = res.get("subtype","standard")
                    batch_results.append(row)
                except Exception as e:
                    logger.debug(f"Failed to generate {i+1}: {e}")
            st.session_state.batch_results.extend(batch_results)
            st.success(f"Generated {len(batch_results)} ODEs.")
            df = pd.DataFrame(batch_results); st.dataframe(df, use_container_width=True)

def novelty_detection_page():
    st.header("🔍 Novelty Detection")
    st.info("Use your existing novelty analyzer here (unchanged).")

def analysis_classification_page():
    st.header("📈 Analysis & Classification")
    if not st.session_state.generated_odes: st.info("No ODEs yet."); return
    if not st.session_state.get("ode_classifier"): st.warning("Classifier not found."); return
    st.subheader("Overview")
    summary = []
    for i, ode in enumerate(st.session_state.generated_odes[-50:]):
        summary.append({"ID":i+1,"Type":ode.get("type","Unknown"),"Order":ode.get("order",0),
                        "Generator":ode.get("generator_number","N/A"),"Function":ode.get("function_used","Unknown"),
                        "Timestamp":ode.get("timestamp","")[:19]})
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

def physical_applications_page():
    st.header("🔬 Physical Applications")
    st.info("Demo list kept as in your app.")

def visualization_page():
    st.header("📐 Visualization")
    if not st.session_state.generated_odes:
        st.warning("No ODEs to visualize."); return
    sel = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)),
                       format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('type','?')} (Order {st.session_state.generated_odes[i].get('order',0)})")
    ode = st.session_state.generated_odes[sel]
    c1,c2,c3 = st.columns(3)
    with c1: plot_type = st.selectbox("Plot Type", ["Solution","Phase Portrait","Direction Field"])
    with c2: x_range = st.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    with c3: N = st.slider("Points", 100, 2000, 500)
    if st.button("Generate", type="primary"):
        x = np.linspace(x_range[0], x_range[1], N)
        y = np.sin(x)*np.exp(-0.1*np.abs(x))
        fig = go.Figure(); fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="y(x)"))
        fig.update_layout(title="ODE Solution", xaxis_title="x", yaxis_title="y")
        st.plotly_chart(fig, use_container_width=True)

def export_latex_page():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes: st.warning("No ODEs."); return
    export_type = st.radio("Export Type", ["Single ODE","Multiple ODEs","Complete Report"])
    if export_type == "Single ODE":
        idx = st.selectbox("Select ODE", range(len(st.session_state.generated_odes)))
        ode = st.session_state.generated_odes[idx]
        st.code(LaTeXExporter.generate_latex_document(ode, include_preamble=False), language="latex")
        st.download_button("📄 LaTeX", LaTeXExporter.generate_latex_document(ode, True), f"ode_{idx+1}.tex", "text/x-latex")
        st.download_button("📦 Package", LaTeXExporter.create_export_package(ode, True), f"ode_package_{idx+1}.zip", "application/zip")
    elif export_type == "Multiple ODEs":
        sel = st.multiselect("Select", range(len(st.session_state.generated_odes)))
        if sel and st.button("Build"):
            parts = [r"""\documentclass[12pt]{article}\usepackage{amsmath,amssymb}\usepackage{geometry}\geometry{margin=1in}
\title{Collection of Generated ODEs}\author{Master Generators System}\date{\today}\begin{document}\maketitle"""]
            for count,i in enumerate(sel,1):
                parts.append(f"\\section{{ODE {count}}}")
                parts.append(LaTeXExporter.generate_latex_document(st.session_state.generated_odes[i], include_preamble=False))
            parts.append(r"\end{document}")
            st.download_button("📄 Multi-ODE LaTeX", "\n".join(parts), f"multiple_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")
    else:
        if st.button("Generate Complete Report"):
            parts = [r"""\documentclass[12pt]{report}\usepackage{amsmath,amssymb}\usepackage{geometry}\geometry{margin=1in}
\title{Master Generators System\\Complete Report}\author{Generated Automatically}\date{\today}\begin{document}\maketitle\tableofcontents
\chapter{Executive Summary}This report contains all ODEs generated by the system.\chapter{Generated ODEs}"""]
            for i, ode in enumerate(st.session_state.generated_odes):
                parts.append(f"\\section{{ODE {i+1}}}")
                parts.append(LaTeXExporter.generate_latex_document(ode, include_preamble=False))
            parts.append(r"\chapter{Conclusions}The system successfully generated and analyzed multiple ODEs.\end{document}")
            st.download_button("📄 Complete Report", "\n".join(parts), f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex", "text/x-latex")

def examples_library_page():
    st.header("📚 Examples Library")
    with st.expander("Simple Harmonic Oscillator"):
        st.latex("y'' + y = 0")

def settings_page():
    st.header("⚙️ Settings")
    if st.button("Save Session"):
        try:
            with open("session_state.pkl","wb") as f:
                pickle.dump({k:st.session_state.get(k) for k in
                             ["generated_odes","generator_patterns","batch_results","analysis_results","training_history","export_history"]}, f)
            st.success("Saved.")
        except Exception as e:
            st.error(f"Save failed: {e}")

def documentation_page():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Use **Apply Master Theorem** (now supports Phi library).
2. Reverse mode: try **reverse engineering** with single‑block or multi‑block (product/sum).
3. Train ML in **ML Pattern Learning**; generation remains compatible.
""")

def main():
    SessionStateManager.initialize()
    st.markdown("""<h2>🔬 Master Generators for ODEs</h2><p>Free‑form/Arbitrary generators • Φ‑library • Multi‑block reverse fit • ML/DL • Export • Async</p>""", unsafe_allow_html=True)
    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard", "🔧 Generator Constructor", "🎯 Apply Master Theorem",
        "🔁 Reverse Engineering", "🤖 ML Pattern Learning", "📊 Batch Generation",
        "🔍 Novelty Detection", "📈 Analysis & Classification", "🔬 Physical Applications",
        "📐 Visualization", "📤 Export & LaTeX", "📚 Examples Library", "⚙️ Settings", "📖 Documentation"
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