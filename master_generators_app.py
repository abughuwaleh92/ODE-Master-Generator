# master_generators_app.py
# Streamlit UI for Master Generators (Theorem 4.2, symbolic n/m, free-form generators, ML/DL, API probe)
from __future__ import annotations

import os
import json
import time
import traceback
from typing import Any, Dict, Optional, Tuple

# --- Streamlit (compatible 1.28+ and 1.33+) ---
import streamlit as st

# --- SymPy ---
import sympy as sp
from sympy import Eq

# --- Optional outbound HTTP for API probe ---
try:
    import requests
except Exception:
    requests = None

# --- Import shim for the core ---
# We try local file, then a package `mg_core`, then dynamic import.
try:
    from core_master_generators import (
        Theorem42, GeneratorBuilder, TemplateConfig,
        safe_eval_f_of_z, ode_to_json, expr_from_srepr,
        GeneratorLibrary, GeneratorPatternLearner, NoveltyDetector
    )
except Exception:
    try:
        from mg_core.core_master_generators import (
            Theorem42, GeneratorBuilder, TemplateConfig,
            safe_eval_f_of_z, ode_to_json, expr_from_srepr,
            GeneratorLibrary, GeneratorPatternLearner, NoveltyDetector
        )
    except Exception as e:
        st.error("Cannot import core_master_generators. "
                 "Ensure core_master_generators.py is next to this file "
                 "or install the package `mg_core`.")
        st.stop()

# ---------- App Config ----------
st.set_page_config(page_title="Master Generators (Theorem 4.2)", layout="wide")

# ---------- Session State ----------
if "learner" not in st.session_state:
    st.session_state.learner = GeneratorPatternLearner()
if "novelty" not in st.session_state:
    st.session_state.novelty = NoveltyDetector()
if "trainset" not in st.session_state:
    st.session_state.trainset = []  # list of (lhs_expr, rhs_expr, meta)

# ---------- Helpers ----------

def _build_theorem(n_mode: str, n_value: Optional[int]) -> Theorem42:
    """Create Theorem42 with symbolic or numeric n."""
    if n_mode == "Symbolic (n)":
        return Theorem42(n="n")
    else:
        if n_value is None or n_value <= 0:
            raise ValueError("Numeric n must be a positive integer.")
        return Theorem42(n=int(n_value))

def _latex_eq(lhs: sp.Expr, rhs: sp.Expr) -> str:
    return sp.latex(Eq(lhs, rhs))

def _make_builder(T: Theorem42) -> GeneratorBuilder:
    return GeneratorBuilder(T, TemplateConfig(alpha=T.alpha, beta=T.beta, n=T.n, m_sym=T.m_sym))

def _download(data: str, filename: str, label: str) -> None:
    st.download_button(label, data=data.encode("utf-8"), file_name=filename, mime="application/json")

def _hr():
    st.markdown("---")

# ---------- UI Pages ----------

def page_generators():
    st.title("Master Generators — Compose, Symbolize, Generate")

    colL, colR = st.columns([7, 5])

    with colL:
        st.subheader("1) Compose a Generator (free form)")
        st.markdown(
            """
            **Template language** (LHS):
            - `y` → the unknown \(y(x)\)  
            - `DyK` → \(y^{(K)}(x)\), e.g. `Dy1`, `Dy2`, `Dy3`  
            - `Dym` → \(y^{(m)}(x)\) (when \(m\) is symbolic)  
            - Aliases: `y^(m)` → `Dym`, `y^(3)` → `Dy3`  
            - Wrap with any SymPy func: `exp(Dy2)`, `sinh(Dym)`, `cos(y)`, `log(1+y)`  
            """
        )

        preset = st.selectbox(
            "Presets (optional)",
            options=[
                "— choose —",
                "Pantograph (linear): y + Dy2 with f(z)=z",
                "Nonlinear wrap: exp(Dy2) + y with f(z)=sin(z)",
                "Multi-order mix: y + Dy1 + Dy3 + sinh(Dym) with f(z)=exp(z)",
            ],
            index=0,
        )

        default_template = "y + Dy2"
        default_f = "z"
        if preset == "Pantograph (linear): y + Dy2 with f(z)=z":
            default_template, default_f = GeneratorLibrary.preset_pantograph_linear()
        elif preset == "Nonlinear wrap: exp(Dy2) + y with f(z)=sin(z)":
            default_template, default_f = GeneratorLibrary.preset_nonlinear_wrap()
        elif preset == "Multi-order mix: y + Dy1 + Dy3 + sinh(Dym) with f(z)=exp(z)":
            default_template, default_f = GeneratorLibrary.preset_multiorder_mix()

        template = st.text_input("Template for LHS (free form):", value=default_template, help="Example: y + exp(Dy2) + sinh(Dym) or y^(m) + y^(3)")
        f_str = st.text_input("Analytic f(z):", value=default_f, help="Any analytic form in SymPy syntax; examples: z, sin(z), exp(z), log(1+z)")

        _hr()
        st.subheader("2) Symbolic parameters (n, m)")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_mode = st.radio("n is", ["Symbolic (n)", "Numeric"], horizontal=True)
        with c2:
            n_value = None
            if n_mode == "Numeric":
                n_value = st.number_input("n (integer ≥ 1)", min_value=1, max_value=999, value=2, step=1)
        with c3:
            m_mode = st.radio("m is", ["Symbolic (m)", "Numeric"], horizontal=True)
            m_value = None
            if m_mode == "Numeric":
                m_value = st.number_input("m (integer ≥ 1)", min_value=1, max_value=999, value=3, step=1)

        complex_form = st.checkbox("Keep complex form (uncheck to take real part)", value=True)

        _hr()
        st.subheader("3) Build ODE (Theorem 4.2)")
        build = st.button("Build ODE")

    with colR:
        st.subheader("Parameters for numeric preview (optional)")
        alpha_val = st.text_input("alpha (numeric; e.g. 1)", value="1")
        beta_val  = st.text_input("beta (numeric; e.g. 0.5)", value="0.5")
        x_val     = st.text_input("x (numeric; e.g. 0.7)", value="0.7")
        st.caption("These are only used if you click *Evaluate Sample*. They do not affect symbolic construction.")

        _hr()
        st.subheader("Evaluate sample")
        eval_btn = st.button("Evaluate Sample (numeric values above)")

    # Results area
    if build:
        try:
            f = safe_eval_f_of_z(f_str)
            T = _build_theorem(n_mode, n_value)
            G = _make_builder(T)

            # Build
            lhs, rhs = G.build(
                template=template,
                f=f,
                m=(T.m_sym if m_mode == "Symbolic (m)" else int(m_value)),
                n_override=None,
                complex_form=complex_form
            )

            st.success("ODE constructed.")
            st.latex(_latex_eq(lhs, rhs))

            # JSON export
            meta = {
                "template": template,
                "f_str": f_str,
                "n_mode": n_mode,
                "n_value": int(n_value) if n_mode == "Numeric" else "n",
                "m_mode": m_mode,
                "m_value": int(m_value) if m_mode == "Numeric" else "m",
                "complex_form": complex_form
            }
            ode_json = ode_to_json(lhs, rhs, meta=meta)
            _hr()
            st.subheader("Serialized ODE (JSON)")
            st.json(json.loads(ode_json))
            _download(ode_json, filename="ode.json", label="⬇️ Download ODE JSON")

            # Keep current ODE in state for ML/DL page
            st.session_state.current_ode = (lhs, rhs, meta)

        except Exception as e:
            st.error(f"Failed to build ODE.\n{e}")
            st.code(traceback.format_exc())

    if eval_btn:
        try:
            if "current_ode" not in st.session_state:
                st.warning("Build an ODE first.")
            else:
                lhs, rhs, _ = st.session_state.current_ode
                subs = {
                    sp.Symbol("alpha"): sp.sympify(alpha_val),
                    sp.Symbol("beta"): sp.sympify(beta_val),
                    sp.Symbol("x"): sp.sympify(x_val),
                }
                lhs_num = sp.N(lhs.subs(subs))
                rhs_num = sp.N(rhs.subs(subs))
                st.write("**Numeric preview** (with alpha, beta, x):")
                st.write(f"LHS = {lhs_num}")
                st.write(f"RHS = {rhs_num}")
        except Exception as e:
            st.error(f"Evaluation failed.\n{e}")
            st.code(traceback.format_exc())

    _hr()
    st.subheader("Where to hook in more")
    st.markdown(
        """
        - Add more **preset recipes** here (pantograph families, multi‑order mixes, nonlinear wraps).
        - Extend **templates**: introduce named blocks (e.g., `G1 = y + Dy2`) and reuse in the textarea.
        - Use the **API Probe** page to persist ODEs/models to the FastAPI backend.
        - Enrich ML labels (stiffness, linearity degree, solvability class), and let the Transformer score novelty/complexity.
        """
    )


def page_ml():
    st.title("ML & DL — Classify, Rank, and Train")

    if "current_ode" not in st.session_state:
        st.info("Build or load an ODE in the Generators page first.")
        return

    lhs, rhs, meta = st.session_state.current_ode

    st.subheader("Selected ODE")
    st.latex(sp.latex(Eq(lhs, rhs)))

    # Novelty score (Transformer or heuristic)
    novelty = st.session_state.novelty
    score = novelty.score(lhs, rhs)
    st.metric("Novelty/Complexity Score", f"{score:.4f}")

    _hr()
    st.subheader("Classification (linearity, stiffness, solvability)")
    learner = st.session_state.learner
    preds = learner.predict([(lhs, rhs)])
    st.json(preds[0])

    _hr()
    st.subheader("Training data")
    with st.expander("Append current ODE to training set"):
        col1, col2, col3 = st.columns(3)
        linear = col1.selectbox("linear", options=[0, 1], index=0, help="0: nonlinear, 1: linear")
        stiff  = col2.selectbox("stiffness", options=[0, 1, 2], index=0)
        solv   = col3.selectbox("solvability", options=[0, 1, 2], index=0)

        if st.button("➕ Add to training set"):
            st.session_state.trainset.append((lhs, rhs, {"linear": linear, "stiffness": stiff, "solvability": solv}))
            st.success(f"Added. Training set size: {len(st.session_state.trainset)}")

    with st.expander("Bulk load training set (JSONL; each line has lhs_srepr, rhs_srepr, meta)"):
        up = st.file_uploader("Upload JSONL", type=["jsonl", "txt"])
        if up is not None:
            lines = up.read().decode("utf-8").strip().splitlines()
            added = 0
            for line in lines:
                try:
                    obj = json.loads(line)
                    lhs_e = expr_from_srepr(obj["lhs_srepr"])
                    rhs_e = expr_from_srepr(obj["rhs_srepr"])
                    meta_e = obj.get("meta", {})
                    st.session_state.trainset.append((lhs_e, rhs_e, meta_e))
                    added += 1
                except Exception as e:
                    st.warning(f"Skipped a line: {e}")
            st.success(f"Loaded {added} samples.")

    _hr()
    if st.button("🧠 Train classifier (sklearn if installed; else heuristics active)"):
        try:
            learner.train(st.session_state.trainset)
            st.success("Training finished.")
        except Exception as e:
            st.error(f"Training failed: {e}")

    # Predict again (post-train)
    preds2 = learner.predict([(lhs, rhs)])
    st.subheader("Prediction (post-train)")
    st.json(preds2[0])


def page_api_probe():
    st.title("API Probe — FastAPI backend integration")

    if requests is None:
        st.warning("The `requests` package is not installed. API probe disabled.")
        return

    base = st.text_input("Base URL", value=os.getenv("MG_API_BASE", "http://localhost:8000"))

    _hr()
    st.subheader("Health")
    if st.button("Check /health"):
        try:
            r = requests.get(f"{base}/health", timeout=5)
            st.code(f"HTTP {r.status_code}\n{r.text}")
        except Exception as e:
            st.error(e)

    _hr()
    st.subheader("Generate (POST /generate)")
    gen_col1, gen_col2 = st.columns(2)

    with gen_col1:
        template = st.text_input("template", value="y + Dy2")
        f_str = st.text_input("f_str", value="z")
        n_mode = st.selectbox("n mode", options=["symbolic", "numeric"], index=0)
        n_value = st.number_input("n (used if numeric)", min_value=1, max_value=999, value=2, step=1)
        m_mode = st.selectbox("m mode", options=["symbolic", "numeric"], index=1)
        m_value = st.number_input("m (used if numeric)", min_value=1, max_value=999, value=3, step=1)
        complex_form = st.checkbox("complex_form", value=True)
    with gen_col2:
        if st.button("POST /generate"):
            payload = {
                "template": template,
                "f_str": f_str,
                "n_mode": n_mode,
                "n_value": int(n_value),
                "m_mode": m_mode,
                "m_value": int(m_value),
                "complex_form": complex_form,
                "persist": True
            }
            try:
                r = requests.post(f"{base}/generate", json=payload, timeout=20)
                st.code(f"HTTP {r.status_code}")
                st.json(r.json())
            except Exception as e:
                st.error(e)

    _hr()
    st.subheader("Train (POST /train)")
    with st.expander("Build payload from current training set"):
        if st.button("POST /train (from app memory)"):
            if not st.session_state.trainset:
                st.warning("Training set is empty.")
            else:
                def _ser(expr):
                    return sp.srepr(expr)
                items = [{"lhs_srepr": _ser(l), "rhs_srepr": _ser(r), "meta": m} for (l, r, m) in st.session_state.trainset]
                try:
                    r = requests.post(f"{base}/train", json={"samples": items}, timeout=60)
                    st.code(f"HTTP {r.status_code}")
                    st.json(r.json())
                except Exception as e:
                    st.error(e)

    _hr()
    st.subheader("List ODEs (GET /odes)")
    if st.button("GET /odes"):
        try:
            r = requests.get(f"{base}/odes?limit=10", timeout=10)
            st.code(f"HTTP {r.status_code}")
            st.json(r.json())
        except Exception as e:
            st.error(e)

    st.caption("Use this page to validate and evolve the API; the UI above already supports the core workflows.")


# ---------- Router ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Generators", "ML & DL", "API Probe"], index=0)

if page == "Generators":
    page_generators()
elif page == "ML & DL":
    page_ml()
else:
    page_api_probe()
