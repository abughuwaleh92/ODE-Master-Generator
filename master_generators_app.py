
# -*- coding: utf-8 -*-
"""
master_generators_app.py
========================

Streamlit UI for building free-form generators, computing RHS via Theorem 4.2,
exploring presets, and applying ML/DL (classification & novelty triage).
Also includes a simple API client to interact with a FastAPI backend.

Run:
    streamlit run master_generators_app.py
"""

import os, sys, json, time
from typing import Any, Dict, Optional, Tuple, List

import sympy as sp
import numpy as np

# Try to import local 'src' if present
for p in ('./src', './app', './code'):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

from core_master_generators import (
    Theorem42, GeneratorBuilder, GeneratorLibrary,
    GeneratorPatternLearner, NoveltyDetector,
    safe_eval_f_of_z, ode_to_json, count_symbolic_complexity
)

# Optional Streamlit (this app assumes it is present)
import streamlit as st

# ----------------------------------------------------------------------------
# Session helpers
# ----------------------------------------------------------------------------
def init_session():
    if 'odes' not in st.session_state:
        st.session_state.odes = []  # list of dicts: {'lhs': Expr, 'rhs': Expr, 'meta': {...}}
    if 'ml' not in st.session_state:
        st.session_state.ml = GeneratorPatternLearner()
    if 'dl' not in st.session_state:
        st.session_state.dl = NoveltyDetector()
    if 'api_base' not in st.session_state:
        st.session_state.api_base = "http://127.0.0.1:8000"
    if 'T' not in st.session_state:
        st.session_state.T = Theorem42()


def as_symbol_or_int(s: str, fallback_sym: sp.Symbol):
    s = s.strip()
    if s.lower() in ('n', 'm'):
        return fallback_sym
    try:
        return int(s)
    except Exception:
        return fallback_sym


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    st.title("Master Generators – Free Builder & ML/DL")
    st.markdown("""
This app lets you **compose any generator** using `y`, `Dy1`, `Dy2`, …, `Dym`, and wrappers like
`exp(.)`, `sinh(.)`, `sin(.)`, `log(.)`, etc. It computes the **RHS exactly** via a compact, general
implementation of **Theorem 4.2** (Stirling-number form).

**Highlights**
- Symbolic **n** and numeric or symbolic **m**.
- Build ODEs: your **LHS operator** vs. **RHS from Theorem 4.2** for a chosen **f(z)**.
- **Presets** include pantograph & multi-order mixes.
- **ML (enriched labels)**: linearity, stiffness, solvability.
- **DL (novelty triage)**: scores and ranks constructed ODEs.
- **API client**: push/build via a FastAPI backend and persist to DB.
    """)


def page_generators():
    st.header("Generator Builder (free form)")

    T: Theorem42 = st.session_state.T

    colA, colB = st.columns([1,1])
    with colA:
        f_str = st.text_input("Enter f(z):", value="sin(z)")
        n_str = st.text_input("n (integer or 'n' symbolic):", value="4")
        m_mode = st.selectbox("Derivative order m:", ["Specify integer m", "Symbolic m (leave unspecified)"])
        m_str = st.text_input("m (if integer chosen):", value="2", disabled=(m_mode!="Specify integer m"))
    with colB:
        complex_form = st.checkbox("Keep complex compact form", value=True)
        alpha_val = st.text_input("alpha (symbolic/number):", value="alpha")
        beta_val = st.text_input("beta (symbolic/number):", value="beta")

    template = st.text_area("Generator template (use y, Dy1, Dy2, ..., Dym):",
                            value="y + exp(Dy2) + sinh(Dym)", height=90)

    if st.button("Build Generator → ODE", type="primary"):
        try:
            f = safe_eval_f_of_z(f_str)
            f_callable = lambda zz: f(zz)
            n_val = as_symbol_or_int(n_str, T.n)
            if m_mode == "Specify integer m":
                m_val = as_symbol_or_int(m_str, T.m_sym)
            else:
                m_val = None  # symbolic

            # Allow alpha,beta override
            ns = {'alpha': T.alpha, 'beta': T.beta, 'pi': sp.pi, 'I': sp.I}
            alpha_expr = sp.sympify(alpha_val, locals=ns)
            beta_expr = sp.sympify(beta_val, locals=ns)

            # Clone T with these alpha/beta/n
            T_local = Theorem42(x=T.x, alpha=alpha_expr, beta=beta_expr, n=n_val, m_sym=T.m_sym)

            builder = GeneratorBuilder(T_local, f_callable, n_val, m_val, complex_form)
            lhs, rhs = builder.build(template)

            st.success("ODE constructed.")
            st.latex(sp.latex(sp.Eq(lhs, rhs)))

            meta = {
                "f": f_str, "template": template, "n": str(n_val), "m": str(m_val) if m_val is not None else "symbolic",
                "alpha": str(alpha_expr), "beta": str(beta_expr), "complex_form": complex_form
            }
            st.session_state.odes.append({"lhs": lhs, "rhs": rhs, "meta": meta})
        except Exception as e:
            st.error(f"Failed: {e}")


# --- Batch f(z) builder ---
    st.subheader("Batch f(z) → many ODEs")
    with st.expander("Enter one f(z) per line", expanded=False):
        batch_txt = st.text_area("f(z) lines", value="sin(z)\nexp(z)\nlog(1+z)", height=120)
        if st.button("Build batch with current template"):
            try:
                n_val = as_symbol_or_int(n_str, T.n)
                if m_mode == "Specify integer m":
                    m_val = as_symbol_or_int(m_str, T.m_sym)
                else:
                    m_val = None
                ns = {'alpha': T.alpha, 'beta': T.beta, 'pi': sp.pi, 'I': sp.I}
                alpha_expr = sp.sympify(alpha_val, locals=ns)
                beta_expr = sp.sympify(beta_val, locals=ns)
                T_local = Theorem42(x=T.x, alpha=alpha_expr, beta=beta_expr, n=n_val, m_sym=T.m_sym)
                for line in batch_txt.strip().splitlines():
                    if not line.strip():
                        continue
                    f = safe_eval_f_of_z(line.strip()); f_callable=lambda zz, f=f: f(zz)
                    builder = GeneratorBuilder(T_local, f_callable, n_val, m_val, complex_form)
                    lhs, rhs = builder.build(template)
                    meta = {"f": line.strip(), "template": template, "n": str(n_val),
                            "m": str(m_val) if m_val is not None else "symbolic",
                            "alpha": str(alpha_expr), "beta": str(beta_expr),
                            "complex_form": complex_form}
                    st.session_state.odes.append({"lhs": lhs, "rhs": rhs, "meta": meta})
                st.success("Batch created.")
            except Exception as e:
                st.error(f"Batch failed: {e}")

    st.subheader("Presets (recipes)")
    lib = GeneratorLibrary(st.session_state.T)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Pantograph (linear) preset"):
            try:
                f = safe_eval_f_of_z(f_str); f_callable=lambda z: f(z)
                preset = lib.pantograph_linear(f_callable)
                st.latex(sp.latex(sp.Eq(preset['lhs'], preset['rhs'])))
                st.session_state.odes.append({"lhs": preset['lhs'], "rhs": preset['rhs'], "meta": {"preset":"pantograph"}})
            except Exception as e:
                st.error(f"Preset failed: {e}")
    with c2:
        if st.button("Multi-order mix preset"):
            try:
                f = safe_eval_f_of_z(f_str); f_callable=lambda z: f(z)
                preset = lib.multi_order_mix(f_callable, orders=(1,2,3))
                st.latex(sp.latex(sp.Eq(preset['lhs'], preset['rhs'])))
                st.session_state.odes.append({"lhs": preset['lhs'], "rhs": preset['rhs'], "meta": {"preset":"multi-order"}})
            except Exception as e:
                st.error(f"Preset failed: {e}")
    with c3:
        if st.button("Nonlinear wrap preset: exp(y'') + y"):
            try:
                f = safe_eval_f_of_z(f_str); f_callable=lambda z: f(z)
                preset = lib.nonlinear_wrap(f_callable, wrap=sp.exp)
                st.latex(sp.latex(sp.Eq(preset['lhs'], preset['rhs'])))
                st.session_state.odes.append({"lhs": preset['lhs'], "rhs": preset['rhs'], "meta": {"preset":"nonlinear-exp"}})
            except Exception as e:
                st.error(f"Preset failed: {e}")


def page_corpus():
    st.header("Corpus & Export")
    if not st.session_state.odes:
        st.info("No ODEs yet.")
        return
    for i, rec in enumerate(st.session_state.odes):
        st.markdown(f"**ODE {i+1}**")
        st.latex(sp.latex(sp.Eq(rec['lhs'], rec['rhs'])))
        with st.expander("Metadata / Complexity"):
            st.json(rec.get("meta", {}))
            feats_L = count_symbolic_complexity(rec['lhs'])
            feats_R = count_symbolic_complexity(rec['rhs'])
            st.write({"lhs": feats_L, "rhs": feats_R})
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Export JSON"):
            data = [ ode_to_json(r['lhs'], r['rhs'], r.get('meta')) for r in st.session_state.odes ]
            path = "export_odes.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            st.success(f"Saved {path}")
            st.download_button("Download JSON", data=json.dumps(data, ensure_ascii=False, indent=2), file_name="export_odes.json")
    with c2:
        if st.button("Export LaTeX"):
            tex = "\n\n".join([ sp.latex(sp.Eq(r['lhs'], r['rhs'])) for r in st.session_state.odes ])
            st.code(tex, language="latex")


def page_ml_dl():
    st.header("ML (enriched labels) & DL (novelty triage)")
    ml = st.session_state.ml
    dl = st.session_state.dl

    st.subheader("Train ML")
    st.markdown("Provide labels per current ODEs (linearity {0,1}, stiffness int, solvability int).")
    if not st.session_state.odes:
        st.info("No ODEs to label. Build some first.")
    else:
        labels = []
        for i, rec in enumerate(st.session_state.odes):
            col1, col2, col3, col4 = st.columns([2,1,1,1])
            with col1:
                st.latex(sp.latex(sp.Eq(rec['lhs'], rec['rhs'])))
            with col2:
                lin = st.selectbox(f"Linearity (ODE {i+1})", [0,1], key=f"lin_{i}")
            with col3:
                sti = st.number_input(f"Stiffness (ODE {i+1})", min_value=0, max_value=9, value=0, key=f"sti_{i}")
            with col4:
                sol = st.number_input(f"Solvability (ODE {i+1})", min_value=0, max_value=9, value=1, key=f"sol_{i}")
            labels.append({"lhs": rec['lhs'], "rhs": rec['rhs'], "labels": {"linearity": lin, "stiffness": sti, "solvability": sol}})

        if st.button("Train classifier(s)"):
            info = ml.train(labels)
            st.write(info)

    st.subheader("Predict & triage")
    if st.session_state.odes:
        pairs = [(r['lhs'], r['rhs']) for r in st.session_state.odes]
        preds = ml.predict(pairs)
        st.write("Predictions:", preds)

        # Novelty score & rank
        ode_strs = [ sp.srepr(sp.Eq(r['lhs'], r['rhs'])) for r in st.session_state.odes ]
        scores = dl.novelty_score(ode_strs)
        order = np.argsort(-np.array(scores))
        st.write("Novelty scores:", scores)
        st.markdown("**Ranked (high → low novelty)**")
        for rank, idx in enumerate(order):
            st.write(f"#{rank+1} (score={scores[idx]:.3f})")
            r = st.session_state.odes[int(idx)]
            st.latex(sp.latex(sp.Eq(r['lhs'], r['rhs'])))

    st.subheader("Quick train novelty (optional)")
    txt = st.text_area("Provide lines: 'srepr || target' per row", height=100, value="Eq(Symbol('y')(Symbol('x')), 0) || 0.1")
    if st.button("Train novelty for a few epochs"):
        pairs = []
        for line in txt.strip().splitlines():
            if '||' in line:
                left, right = line.split('||', 1)
                try:
                    target = float(right.strip())
                except:
                    target = 0.5
                pairs.append((left.strip(), target))
        got = st.session_state.dl.quick_train(pairs, epochs=2)
        st.write(got)


def page_api_client():
    st.header("FastAPI client")
    base = st.text_input("API base URL", value=st.session_state.api_base)
    st.session_state.api_base = base

    try:
        import requests
        can = True
    except Exception:
        can = False
        st.warning("`requests` not installed; API client disabled.")

    if can and st.button("Health check"):
        try:
            r = requests.get(base + "/health", timeout=5)
            st.code(r.text, language="json")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.subheader("Push current ODEs → backend")
    if can:
        for i, rec in enumerate(st.session_state.odes):
            if st.button(f"Push ODE {i+1}"):
                payload = ode_to_json(rec['lhs'], rec['rhs'], rec.get('meta', {}))
                try:
                    r = requests.post(base + "/store/ode", json=payload, timeout=8)
                    st.code(r.text, language="json")
                except Exception as e:
                    st.error(f"Push failed: {e}")

    st.subheader("Build ODE on backend")
    with st.form("build_remote"):
        f_str = st.text_input("f(z) for backend", value="sin(z)")
        template = st.text_input("template", value="y + Dy2")
        n = st.text_input("n", value="4")
        m = st.text_input("m (int or blank)", value="2")
        alpha = st.text_input("alpha", value="alpha")
        beta = st.text_input("beta", value="beta")
        complex_form = st.checkbox("complex_form", value=True)
        submitted = st.form_submit_button("POST /build_ode")
    if can and submitted:
        try:
            payload = {"f_str": f_str, "template": template, "n": n, "m": m if m.strip() else None,
                       "alpha": alpha, "beta": beta, "complex_form": complex_form, "persist": True}
            r = requests.post(base + "/build_ode", json=payload, timeout=12)
            st.code(r.text, language="json")
        except Exception as e:
            st.error(f"Failed: {e}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    init_session()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Generator Builder", "Corpus/Export", "ML & DL", "API Client"])
    if page == "Home":
        page_home()
    elif page == "Generator Builder":
        page_generators()
    elif page == "Corpus/Export":
        page_corpus()
    elif page == "ML & DL":
        page_ml_dl()
    elif page == "API Client":
        page_api_client()

if __name__ == "__main__":
    main()
