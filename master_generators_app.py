# master_generators_app.py
# Streamlit UI for Master Generators (Theorem 4.2) with robust core-call compatibility
from __future__ import annotations

import os
import json
import traceback
from typing import Any, Dict, Optional, Tuple

import sympy as sp
from sympy import Eq

import streamlit as st

# Optional HTTP for API probe
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ---- Import shim for the core ----
# We support both a flat file and a packaged core (mg_core.core_master_generators)
CORE_IMPORT_OK = False
core_err = None
try:
    from core_master_generators import (
        Theorem42, GeneratorBuilder, TemplateConfig,
        safe_eval_f_of_z, ode_to_json, expr_from_srepr,
        GeneratorLibrary, GeneratorPatternLearner, NoveltyDetector
    )
    CORE_IMPORT_OK = True
except Exception as e1:
    try:
        from mg_core.core_master_generators import (
            Theorem42, GeneratorBuilder, TemplateConfig,
            safe_eval_f_of_z, ode_to_json, expr_from_srepr,
            GeneratorLibrary, GeneratorPatternLearner, NoveltyDetector
        )
        CORE_IMPORT_OK = True
    except Exception as e2:
        core_err = (e1, e2)
        CORE_IMPORT_OK = False

if not CORE_IMPORT_OK:
    st.error(
        "Cannot import `core_master_generators`. "
        "Place `core_master_generators.py` next to this file or install the `mg_core` package."
    )
    if core_err:
        st.code("\n".join([repr(core_err[0]), repr(core_err[1])]))
    st.stop()

# ---- Page config ----
st.set_page_config(page_title="Master Generators (Theorem 4.2)", layout="wide")

# ---- Session init ----
if "learner" not in st.session_state:
    st.session_state.learner = GeneratorPatternLearner()
if "novelty" not in st.session_state:
    st.session_state.novelty = NoveltyDetector()
if "trainset" not in st.session_state:
    st.session_state.trainset = []  # list of (lhs_expr, rhs_expr, meta)
if "current_ode" not in st.session_state:
    st.session_state.current_ode = None

# ---- Helpers ----

def hr():
    st.markdown("---")

def to_latex(lhs: sp.Expr, rhs: sp.Expr) -> str:
    return sp.latex(Eq(lhs, rhs))

def build_theorem(n_mode: str, n_value: Optional[int]) -> Theorem42:
    """
    Build Theorem42 with symbolic or numeric n.
    Some cores accept str('n'); others require a SymPy Symbol. We try both.
    """
    if n_mode == "Symbolic (n)":
        # Try SymPy symbol first
        try:
            return Theorem42(n=sp.Symbol("n"))
        except TypeError:
            return Theorem42(n="n")
        except Exception:
            return Theorem42()  # as last resort
    else:
        if n_value is None or int(n_value) <= 0:
            raise ValueError("Numeric n must be a positive integer.")
        return Theorem42(n=int(n_value))

def make_builder(T: Theorem42) -> GeneratorBuilder:
    """
    Compatibility wrapper around GeneratorBuilder construction.
    Supports:
      - GeneratorBuilder(T, TemplateConfig(...))
      - GeneratorBuilder(T)
      - legacy: (GeneratorBuilder().init(T)) or init(T, TemplateConfig(...))
    """
    cfg = TemplateConfig(
        alpha=getattr(T, "alpha", sp.Symbol("alpha")),
        beta=getattr(T, "beta", sp.Symbol("beta")),
        n=getattr(T, "n", sp.Symbol("n")),
        m_sym=getattr(T, "m_sym", sp.Symbol("m")),
    )
    # Try preferred signature: (theorem, config)
    try:
        return GeneratorBuilder(T, cfg)  # type: ignore[arg-type]
    except TypeError:
        pass
    except Exception:
        pass
    # Try minimal: (theorem)
    try:
        return GeneratorBuilder(T)  # type: ignore[arg-type]
    except TypeError:
        pass
    except Exception:
        pass
    # Legacy: instance + .init(...)
    gb = GeneratorBuilder  # class
    try:
        inst = gb()  # type: ignore[call-arg]
        if hasattr(inst, "init"):
            try:
                return inst.init(T, cfg)  # type: ignore[attr-defined]
            except TypeError:
                return inst.init(T)  # type: ignore[attr-defined]
        # If no .init, but constructor requires no args:
        return inst  # type: ignore[return-value]
    except Exception as e:
        raise TypeError(
            f"Cannot create GeneratorBuilder with any supported signature: {e}"
        )

def call_build(G: Any, *, template: str, f: Any, m: Any, complex_form: bool) -> Tuple[sp.Expr, sp.Expr]:
    """
    Robust call into GeneratorBuilder.build(), trying common signatures.
    Prioritized signature:
        build(template=..., f=..., m=..., n_override=None, complex_form=True)
    Fallbacks try removing unknown keywords / renaming flags.
    """
    # 1) Preferred keyworded signature
    try:
        return G.build(template=template, f=f, m=m, n_override=None, complex_form=complex_form)
    except TypeError:
        pass
    # 2) Without n_override
    try:
        return G.build(template=template, f=f, m=m, complex_form=complex_form)
    except TypeError:
        pass
    # 3) Positional with complex_form
    try:
        return G.build(template, f, m, complex_form)
    except TypeError:
        pass
    # 4) Positional without complex_form
    try:
        return G.build(template, f, m)
    except TypeError as e:
        raise TypeError(f"GeneratorBuilder.build() signature mismatch: {e}")

def ui_presets() -> Tuple[str, str]:
    """
    Provide preset (template, f_str) without instantiating GeneratorLibrary.
    """
    # Return values from static/class helpers when available, else built-ins
    try:
        return GeneratorLibrary.preset_pantograph_linear()  # type: ignore[attr-defined]
    except Exception:
        return ("y + Dy2", "z")

def ui_presets_nonlinear() -> Tuple[str, str]:
    try:
        return GeneratorLibrary.preset_nonlinear_wrap()  # type: ignore[attr-defined]
    except Exception:
        return ("exp(Dy2) + y", "sin(z)")

def ui_presets_multi() -> Tuple[str, str]:
    try:
        return GeneratorLibrary.preset_multiorder_mix()  # type: ignore[attr-defined]
    except Exception:
        return ("y + Dy1 + Dy3 + sinh(Dym)", "exp(z)")

# ---- Pages ----

def page_generators():
    st.title("Master Generators — Compose & Generate (Theorem 4.2)")

    colL, colR = st.columns([7, 5])

    with colL:
        st.subheader("1) Compose LHS freely")
        st.markdown(
            """
            **Atoms**:  
            - `y` → \( y(x) \)  
            - `DyK` → \( y^{(K)}(x) \), e.g. `Dy1`, `Dy2`, `Dy3`  
            - `Dym` → \( y^{(m)}(x) \) when \(m\) is symbolic  
            - `y^(m)` is accepted as an alias for `Dym`  
            - Wraps: `exp(Dy2)`, `sinh(Dym)`, `cos(y)`, `log(1+y)`, etc.
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

        default_template, default_f = ("y + Dy2", "z")
        if preset == "Pantograph (linear): y + Dy2 with f(z)=z":
            default_template, default_f = ui_presets()
        elif preset == "Nonlinear wrap: exp(Dy2) + y with f(z)=sin(z)":
            default_template, default_f = ui_presets_nonlinear()
        elif preset == "Multi-order mix: y + Dy1 + Dy3 + sinh(Dym) with f(z)=exp(z)":
            default_template, default_f = ui_presets_multi()

        template = st.text_input("LHS template", value=default_template)
        f_str = st.text_input("Analytic f(z)", value=default_f)

        hr()
        st.subheader("2) Symbolic parameters")
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

        complex_form = st.checkbox("Keep complex form (uncheck to take Re)", value=True)

        hr()
        st.subheader("3) Build ODE (RHS via Theorem 4.2)")
        build_btn = st.button("Build ODE")

    with colR:
        st.subheader("Numeric preview (optional)")
        alpha_val = st.text_input("alpha", "1")
        beta_val = st.text_input("beta", "0.5")
        x_val = st.text_input("x", "0.7")
        eval_btn = st.button("Evaluate Sample")

    if build_btn:
        try:
            f = safe_eval_f_of_z(f_str)
        except Exception as e:
            st.error(f"Invalid f(z): {e}")
            return

        try:
            T = build_theorem(n_mode, n_value)
            G = make_builder(T)
        except Exception as e:
            st.error(f"Failed to initialize builder: {e}")
            st.code(traceback.format_exc())
            return

        try:
            m_arg = getattr(T, "m_sym", sp.Symbol("m")) if m_mode == "Symbolic (m)" else int(m_value)  # type: ignore[arg-type]
            lhs, rhs = call_build(G, template=template, f=f, m=m_arg, complex_form=complex_form)
        except Exception as e:
            st.error(f"Failed to build ODE: {e}")
            st.code(traceback.format_exc())
            return

        st.success("ODE constructed.")
        st.latex(to_latex(lhs, rhs))

        # JSON
        meta = {
            "template": template,
            "f_str": f_str,
            "n_mode": n_mode,
            "n_value": (int(n_value) if n_mode == "Numeric" else "n"),
            "m_mode": m_mode,
            "m_value": (int(m_value) if m_mode == "Numeric" else "m"),
            "complex_form": complex_form,
        }
        ser = ode_to_json(lhs, rhs, meta=meta)
        st.subheader("Serialized ODE (JSON)")
        st.json(json.loads(ser))
        st.download_button("⬇️ Download ODE JSON", data=ser.encode("utf-8"), file_name="ode.json", mime="application/json")

        st.session_state.current_ode = (lhs, rhs, meta)

    if eval_btn:
        if not st.session_state.current_ode:
            st.warning("Build an ODE first.")
        else:
            lhs, rhs, _ = st.session_state.current_ode
            try:
                subs = {
                    sp.Symbol("alpha"): sp.sympify(alpha_val),
                    sp.Symbol("beta"): sp.sympify(beta_val),
                    sp.Symbol("x"): sp.sympify(x_val),
                }
                st.write("**Numeric preview**")
                st.write(f"LHS = {sp.N(lhs.subs(subs))}")
                st.write(f"RHS = {sp.N(rhs.subs(subs))}")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.code(traceback.format_exc())

    hr()
    st.subheader("Where to hook in more")
    st.markdown(
        """
        - Add more **preset recipes** (pantograph families, multi-order mixes, nonlinear wraps).
        - Extend template DSL, e.g., named blocks and reuse.
        - Use **API Probe** page to persist ODEs/models to FastAPI.
        - Enrich labels (stiffness, linearity degree, solvability) and triage by novelty score.
        """
    )

def page_ml():
    st.title("ML & DL — Classify, Rank, Train")

    if not st.session_state.current_ode:
        st.info("Build an ODE on the Generators page first.")
        return

    lhs, rhs, meta = st.session_state.current_ode
    st.subheader("Selected ODE")
    st.latex(sp.latex(Eq(lhs, rhs)))

    novelty = st.session_state.novelty
    score = novelty.score(lhs, rhs)
    st.metric("Novelty/Complexity Score", f"{score:.4f}")

    hr()
    st.subheader("Classification (linearity, stiffness, solvability)")
    learner = st.session_state.learner
    preds = learner.predict([(lhs, rhs)])
    st.json(preds[0])

    hr()
    st.subheader("Training data")
    with st.expander("Append current ODE to training set"):
        col1, col2, col3 = st.columns(3)
        linear = col1.selectbox("linear", options=[0, 1], index=0)
        stiff  = col2.selectbox("stiffness", options=[0, 1, 2], index=0)
        solv   = col3.selectbox("solvability", options=[0, 1, 2], index=0)
        if st.button("➕ Add"):
            st.session_state.trainset.append((lhs, rhs, {"linear": linear, "stiffness": stiff, "solvability": solv}))
            st.success(f"Added. Size: {len(st.session_state.trainset)}")

    with st.expander("Bulk load training set (JSONL: lhs_srepr, rhs_srepr, meta)"):
        up = st.file_uploader("Upload JSONL", type=["jsonl", "txt"])
        if up is not None:
            lines = up.read().decode("utf-8").splitlines()
            added = 0
            for line in lines:
                try:
                    obj = json.loads(line)
                    l = expr_from_srepr(obj["lhs_srepr"])
                    r = expr_from_srepr(obj["rhs_srepr"])
                    m = obj.get("meta", {})
                    st.session_state.trainset.append((l, r, m))
                    added += 1
                except Exception as e:
                    st.warning(f"Skipped: {e}")
            st.success(f"Loaded {added} samples.")

    if st.button("🧠 Train classifier (sklearn if present; else heuristics)"):
        try:
            st.session_state.learner.train(st.session_state.trainset)
            st.success("Training complete.")
        except Exception as e:
            st.error(f"Training failed: {e}")

    preds2 = learner.predict([(lhs, rhs)])
    st.subheader("Prediction (post-train)")
    st.json(preds2[0])

def page_api_probe():
    st.title("API Probe — FastAPI backend")

    if requests is None:
        st.warning("The `requests` package is not installed. API probe disabled.")
        return

    base = st.text_input("Base URL", value=os.getenv("MG_API_BASE", "http://localhost:8000"))

    hr()
    st.subheader("Health")
    if st.button("GET /health"):
        try:
            r = requests.get(f"{base}/health", timeout=5)
            st.code(f"HTTP {r.status_code}\n{r.text}")
        except Exception as e:
            st.error(e)

    hr()
    st.subheader("Generate (POST /generate)")
    c1, c2 = st.columns(2)
    with c1:
        template = st.text_input("template", value="y + Dy2", key="api_template")
        f_str    = st.text_input("f_str", value="z", key="api_fstr")
        n_mode   = st.selectbox("n_mode", options=["symbolic", "numeric"], index=0, key="api_nmode")
        n_value  = st.number_input("n_value", min_value=1, max_value=999, value=2, step=1, key="api_nvalue")
        m_mode   = st.selectbox("m_mode", options=["symbolic", "numeric"], index=1, key="api_mmode")
        m_value  = st.number_input("m_value", min_value=1, max_value=999, value=3, step=1, key="api_mvalue")
        complex_form = st.checkbox("complex_form", value=True, key="api_cplx")
    with c2:
        if st.button("POST /generate"):
            payload = {
                "template": template,
                "f_str": f_str,
                "n_mode": n_mode,
                "n_value": int(n_value),
                "m_mode": m_mode,
                "m_value": int(m_value),
                "complex_form": complex_form,
                "persist": True,
            }
            try:
                r = requests.post(f"{base}/generate", json=payload, timeout=30)
                st.code(f"HTTP {r.status_code}")
                st.json(r.json())
            except Exception as e:
                st.error(e)

    hr()
    st.subheader("Train (POST /train)")
    if st.button("POST /train (from in-app memory)"):
        if not st.session_state.trainset:
            st.warning("Training set is empty.")
        else:
            def _ser(expr): return sp.srepr(expr)
            items = [{"lhs_srepr": _ser(l), "rhs_srepr": _ser(r), "meta": m} for (l, r, m) in st.session_state.trainset]
            try:
                r = requests.post(f"{base}/train", json={"samples": items}, timeout=60)
                st.code(f"HTTP {r.status_code}")
                st.json(r.json())
            except Exception as e:
                st.error(e)

    hr()
    st.subheader("List ODEs (GET /odes)")
    if st.button("GET /odes?limit=10"):
        try:
            r = requests.get(f"{base}/odes?limit=10", timeout=10)
            st.code(f"HTTP {r.status_code}")
            st.json(r.json())
        except Exception as e:
            st.error(e)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Generators", "ML & DL", "API Probe"], index=0)
    if page == "Generators":
        page_generators()
    elif page == "ML & DL":
        page_ml()
    else:
        page_api_probe()

if __name__ == "__main__":
    main()
