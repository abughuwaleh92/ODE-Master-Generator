# master_generators_app.py
import os, sys, io, json, time, base64, zipfile, logging, pickle, traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sympy as sp

# ---------------- Logging ----------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("master_generators_app")

# ---------------- Paths ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ---------------- Import core + rq utils ----------------
from shared.ode_core import (
    ComputeParams, compute_ode_full, theorem_4_2_y_m_expr,
    get_function_expr, to_exact
)

from rq_utils import (
    has_redis, enqueue_job, fetch_job, get_queue_stats,
    append_training_log, read_training_log
)

# Optional imports: degrade gracefully
HAVE_SRC = True
try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
except Exception as e:
    HAVE_SRC = False
    BasicFunctions = SpecialFunctions = None
    logger.warning(f"Function libraries missing: {e}")

try:
    from src.generators.generator_constructor import (
        GeneratorConstructor, GeneratorSpecification,
        DerivativeTerm, DerivativeType, OperatorType
    )
except Exception:
    GeneratorConstructor = GeneratorSpecification = None
    DerivativeTerm = DerivativeType = OperatorType = None

try:
    from src.generators.linear_generators import (
        LinearGeneratorFactory, CompleteLinearGeneratorFactory
    )
    from src.generators.nonlinear_generators import (
        NonlinearGeneratorFactory, CompleteNonlinearGeneratorFactory
    )
except Exception:
    LinearGeneratorFactory = CompleteLinearGeneratorFactory = None
    NonlinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None

try:
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
except Exception:
    ODEClassifier = PhysicalApplication = None

# ML (optional; will be needed for local load/generate)
try:
    import torch
    from src.ml.trainer import MLTrainer
except Exception as e:
    torch = None
    MLTrainer = None
    logger.warning(f"MLTrainer unavailable: {e}")

# Novelty (optional)
try:
    from src.dl.novelty_detector import ODENoveltyDetector
except Exception:
    ODENoveltyDetector = None

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Master Generators ODE System", page_icon="🔬", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.4rem;border-radius:14px;margin-bottom:1rem;color:white;text-align:center;}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:1rem;border-radius:12px;text-align:center;}
.info-box{background:#e8f4ff;border-left:4px solid #1e88e5;padding:.8rem;border-radius:10px;margin:.6rem 0;}
.result-box{background:#f1fff1;border:2px solid #4caf50;padding:1rem;border-radius:10px;margin:1rem 0;}
.error-box{background:#ffeef0;border:2px solid #f44336;padding:1rem;border-radius:10px;margin:1rem 0;}
.small{font-size:.92rem;opacity:.9}
</style>
""", unsafe_allow_html=True)

# ---------------- Session ----------------
def _init_state():
    defaults = {
        "generator_constructor": GeneratorConstructor() if GeneratorConstructor else None,
        "basic_functions": BasicFunctions() if BasicFunctions else None,
        "special_functions": SpecialFunctions() if SpecialFunctions else None,
        "ode_classifier": ODEClassifier() if ODEClassifier else None,
        "novelty_detector": ODENoveltyDetector() if ODENoveltyDetector else None,
        "generated_odes": [],
        "batch_results": [],
        "ml_trainer": None,           # local loaded trainer instance
        "ml_trained": False,
        "training_history": {},
        "model_registry": {},         # alias -> path
        "lhs_source": "constructor",
        "free_terms": [],
        "arbitrary_lhs_text": "",
        "last_job_id": None,
        "last_job_payload": None,
        "last_job_enqueued_at": None,
        "train_job_id": None,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------- Helpers ----------------
def _register_generated_ode(result: dict):
    """Normalize + append to session."""
    res = dict(result)
    res.setdefault("type", "nonlinear")
    res.setdefault("order", 0)
    res.setdefault("function_used", "unknown")
    res.setdefault("parameters", {})
    res.setdefault("timestamp", datetime.now().isoformat())
    # convenience: a SymPy Eq if both sides are sympy
    try:
        if isinstance(res["generator"], sp.Expr) and isinstance(res["rhs"], sp.Expr):
            res["ode"] = sp.Eq(res["generator"], res["rhs"])
    except Exception:
        pass
    st.session_state.generated_odes.append(res)

def _latex(expr) -> str:
    try:
        if isinstance(expr, str):
            expr = sp.sympify(expr)
        return sp.latex(expr)
    except Exception:
        return str(expr)

def _ensure_checkpoint_dir() -> str:
    d = os.getenv("CHECKPOINT_DIR", os.path.join(APP_DIR, "checkpoints"))
    os.makedirs(d, exist_ok=True)
    return d

def _available_checkpoints() -> List[str]:
    d = _ensure_checkpoint_dir()
    out = []
    for nm in os.listdir(d):
        if nm.endswith(".pth"):
            out.append(os.path.join(d, nm))
    return sorted(out)

def _available_histories() -> List[str]:
    d = _ensure_checkpoint_dir()
    out = []
    for nm in os.listdir(d):
        if nm.startswith("history_") and nm.endswith(".json"):
            out.append(os.path.join(d, nm))
    return sorted(out)

# ---------------- Apply Master Theorem (with RQ support) ----------------
def page_apply_theorem():
    st.header("🎯 Apply Master Theorem (Async-ready)")

    # function lib + name
    cols = st.columns(2)
    with cols[0]:
        lib = st.selectbox("Function library", ["Basic","Special"], index=0)
    with cols[1]:
        bf = st.session_state.get("basic_functions")
        sf = st.session_state.get("special_functions")
        names = bf.get_function_names() if (lib=="Basic" and bf) else (sf.get_function_names() if (lib=="Special" and sf) else [])
        func_name = st.selectbox("Choose f(z)", names) if names else st.text_input("Enter f(z)", "exp(z)")

    # params
    c1,c2,c3,c4 = st.columns(4)
    with c1: alpha = st.number_input("α", value=1.0, step=0.1, format="%.6f")
    with c2: beta  = st.number_input("β", value=1.0, step=0.1, format="%.6f")
    with c3: n     = st.number_input("n", 1, 12, 1)
    with c4: M     = st.number_input("M", value=0.0, step=0.1, format="%.6f")
    c5,c6,c7 = st.columns(3)
    with c5: use_exact = st.checkbox("Exact (symbolic) parameters", True)
    with c6: simplify_level = st.selectbox("Simplify", ["light","none","aggressive"], index=0)
    with c7: st.caption("Async via Redis: **{}**".format("ON" if has_redis() else "OFF"))

    # LHS source
    src = st.radio("LHS source", ["constructor","freeform","arbitrary"], index=["constructor","freeform","arbitrary"].index(st.session_state["lhs_source"]))
    st.session_state["lhs_source"] = src

    # Free-form builder
    with st.expander("🧩 Free‑form LHS (builder)"):
        cols = st.columns(8)
        with cols[0]: coef = st.number_input("coef", 1.0, step=0.5)
        with cols[1]: inner_order = st.number_input("inner k", 0, 12, 0)
        with cols[2]: wrapper = st.selectbox("wrap(.)", ["id","exp","sin","cos","tan","sinh","cosh","tanh","log","abs","asin","acos","atan","asinh","acosh","atanh","erf"])
        with cols[3]: power = st.number_input("power", 1, 6, 1)
        with cols[4]: outer_order = st.number_input("outer m", 0, 12, 0)
        with cols[5]: scale = st.number_input("arg scale a", value=1.0, step=0.1, format="%.4f")
        with cols[6]: shift = st.number_input("arg shift b", value=0.0, step=0.1, format="%.4f")
        with cols[7]:
            if st.button("➕ Add term"):
                st.session_state.free_terms.append({
                    "coef": float(coef), "inner_order": int(inner_order), "wrapper": wrapper,
                    "power": int(power), "outer_order": int(outer_order),
                    "arg_scale": float(scale) if abs(scale) > 1e-14 else None,
                    "arg_shift": float(shift) if abs(shift) > 1e-14 else None,
                })
        if st.session_state.free_terms:
            st.write(st.session_state.free_terms)
            cdel, cuse = st.columns(2)
            with cuse:
                if st.button("✅ Use free‑form LHS"):
                    st.session_state["lhs_source"] = "freeform"
                    st.success("Selected.")
            with cdel:
                if st.button("🗑️ Clear"):
                    st.session_state.free_terms = []

    # Arbitrary LHS (SymPy)
    with st.expander("✍️ Arbitrary LHS (SymPy expression)"):
        st.session_state.arbitrary_lhs_text = st.text_area(
            "Enter SymPy expr in x,y(x)  e.g.  sin(y(x)) + y(x)*y(x).diff(x)",
            value=st.session_state.arbitrary_lhs_text or "", height=100
        )

    # Submit
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
        st.session_state["last_job_payload"] = payload
        if has_redis():
            job_id = enqueue_job("worker.compute_job", payload, description="ODE compute")
            if job_id:
                st.session_state["last_job_id"] = job_id
                st.session_state["last_job_enqueued_at"] = time.time()
                st.success(f"Job submitted: {job_id}")
            else:
                st.error("Failed to enqueue. Check REDIS_URL and worker.")
        else:
            # local compute fallback
            try:
                bf = st.session_state.get("basic_functions")
                sf = st.session_state.get("special_functions")
                p = ComputeParams(
                    func_name=func_name, alpha=alpha, beta=beta, n=int(n), M=M,
                    use_exact=use_exact, simplify_level=simplify_level,
                    lhs_source=st.session_state["lhs_source"],
                    constructor_lhs=None, freeform_terms=st.session_state.get("free_terms"),
                    arbitrary_lhs_text=st.session_state.get("arbitrary_lhs_text"),
                    function_library=lib, basic_lib=bf, special_lib=sf
                )
                res = compute_ode_full(p)
                _register_generated_ode(res)
                _show_ode_result(res)
            except Exception as e:
                st.error(f"Local compute failed: {e}")

    # Polling / diagnostics
    _job_status_panel(section_title="📡 Job Status (Compute)")

def _job_status_panel(section_title="📡 Job Status"):
    job_id = st.session_state.get("last_job_id")
    if not job_id:
        return
    st.subheader(section_title)
    # Diagnostics
    stats = get_queue_stats()
    if stats.get("queues"):
        st.caption("Queues:")
        st.write(pd.DataFrame(stats["queues"]))
    ws = stats.get("workers") or []
    if ws:
        st.caption("Workers:")
        st.write(pd.DataFrame(ws))

    info = fetch_job(job_id)
    st.write(f"**Status:** {info.get('status')}  |  **Desc:** {info.get('description')}")
    meta = info.get("meta", {})
    if meta:
        cols = st.columns(4)
        with cols[0]: st.metric("Stage", meta.get("status","?"))
        with cols[1]: st.metric("Epoch", f"{meta.get('epoch','-')}/{meta.get('epochs','-')}")
        with cols[2]: st.metric("Train Loss", f"{meta.get('train_loss','-')}")
        with cols[3]: st.metric("Val Loss", f"{meta.get('val_loss','-')}")
        if meta.get("progress") is not None:
            st.progress(min(1.0, float(meta["progress"])))

    if info.get("logs_tail"):
        with st.expander("Recent logs"):
            for ev in info["logs_tail"]:
                st.code(json.dumps(ev, ensure_ascii=False))

    status = info.get("status")
    if status == "finished":
        res = info.get("result")
        if isinstance(res, dict) and "generator" in res and "rhs" in res and "solution" in res:
            # Try to sympify for LaTeX
            try:
                res["generator"] = sp.sympify(res["generator"])
                res["rhs"] = sp.sympify(res["rhs"])
                res["solution"] = sp.sympify(res["solution"])
            except Exception:
                pass
            _register_generated_ode(res)
            _show_ode_result(res)
            st.session_state["last_job_id"] = None
    elif status in {"failed","stopped"}:
        st.error(info.get("exc_info") or info.get("error") or "Job error")
        st.session_state["last_job_id"] = None
    else:
        # Offer local fallback if no workers detected for a while
        if not (get_queue_stats().get("workers") or []):
            stuck = int(time.time() - (st.session_state.get("last_job_enqueued_at") or time.time()))
            if stuck > 15:
                with st.expander("No worker detected — run locally?"):
                    if st.button("⚡ Run locally now"):
                        try:
                            lp = st.session_state.get("last_job_payload", {})
                            bf = st.session_state.get("basic_functions")
                            sf = st.session_state.get("special_functions")
                            p = ComputeParams(
                                func_name=lp.get("func_name","exp(z)"), alpha=lp.get("alpha",1), beta=lp.get("beta",1),
                                n=int(lp.get("n",1)), M=lp.get("M",0),
                                use_exact=bool(lp.get("use_exact",True)), simplify_level=lp.get("simplify_level","light"),
                                lhs_source=lp.get("lhs_source","constructor"),
                                constructor_lhs=None, freeform_terms=lp.get("freeform_terms"),
                                arbitrary_lhs_text=lp.get("arbitrary_lhs_text"),
                                function_library=lp.get("function_library","Basic"), basic_lib=bf, special_lib=sf
                            )
                            res = compute_ode_full(p)
                            _register_generated_ode(res)
                            _show_ode_result(res)
                        except Exception as e:
                            st.error(f"Local compute failed: {e}")
                        st.session_state["last_job_id"] = None

def _show_ode_result(res: Dict[str, Any]):
    st.markdown('<div class="result-box"><h3>✅ ODE Generated</h3></div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📐 Equation", "💡 Solution & Params", "📤 Export"])
    with t1:
        try:
            st.latex(_latex(res["generator"]) + " = " + _latex(res["rhs"]))
        except Exception:
            st.write("LHS:", res.get("generator"))
            st.write("RHS:", res.get("rhs"))
        st.caption(f"Type: {res.get('type','?')} • Order: {res.get('order','?')}")
    with t2:
        try:
            st.latex("y(x) = " + _latex(res["solution"]))
        except Exception:
            st.write("y(x) =", res.get("solution"))
        p = res.get("parameters", {})
        st.write(f"**Parameters:** α={p.get('alpha')}, β={p.get('beta')}, n={p.get('n')}, M={p.get('M')}")
        st.write(f"**Function:** f(z) ≈ {res.get('f_expr_preview')}")
        if res.get("initial_conditions"):
            st.write("**Initial conditions:**")
            for k,v in res["initial_conditions"].items():
                try: st.latex(k + " = " + _latex(v))
                except Exception: st.write(k, "=", v)
    with t3:
        # simple TeX export
        tex = "\\begin{equation}\n" + _latex(res.get("generator","")) + "=" + _latex(res.get("rhs","")) + "\n\\end{equation}\n" \
              + "\\[\n y(x) = " + _latex(res.get("solution","")) + "\n\\]"
        st.download_button("📄 Download TeX", tex, file_name=f"ode_{len(st.session_state.generated_odes)}.tex")

# ---------------- Generator Constructor ----------------
def page_constructor():
    st.header("🔧 Generator Constructor")
    if not (GeneratorSpecification and DerivativeTerm and DerivativeType and OperatorType):
        st.warning("Constructor classes not found in src/.")
        return

    with st.expander("➕ Add Term", True):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            deriv_order = st.selectbox("Derivative Order", [0,1,2,3,4,5])
        with c2:
            func_type = st.selectbox("Function Type", [t.value for t in DerivativeType])
        with c3:
            coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
        with c4:
            power = st.number_input("Power", 1, 6, 1)
        c5,c6,c7 = st.columns(3)
        with c5:
            operator_type = st.selectbox("Operator Type", [t.value for t in OperatorType])
        with c6:
            scaling = st.number_input("Scaling (a)", 0.5, 5.0, 1.0, 0.1) if operator_type in ["delay","advance"] else None
        with c7:
            shift = st.number_input("Shift (b)", -10.0, 10.0, 0.0, 0.1) if operator_type in ["delay","advance"] else None
        if st.button("➕ Add", type="primary"):
            term = DerivativeTerm(
                derivative_order=int(deriv_order),
                coefficient=float(coefficient),
                power=int(power),
                function_type=DerivativeType(func_type),
                operator_type=OperatorType(operator_type),
                scaling=scaling, shift=shift
            )
            if "generator_terms" not in st.session_state:
                st.session_state.generator_terms = []
            st.session_state.generator_terms.append(term)
            st.success("Term added.")
    if st.session_state.get("generator_terms"):
        st.subheader("📝 Current Terms")
        for i, t in enumerate(st.session_state.generator_terms):
            st.write(f"{i+1}. {getattr(t,'get_description',lambda: str(t))()}")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("🔨 Build Generator Spec"):
                try:
                    spec = GeneratorSpecification(terms=st.session_state.generator_terms, name=f"CustomGen-{datetime.now().strftime('%H%M%S')}")
                    st.session_state.current_generator = spec
                    st.success("Generator specification created.")
                    try:
                        st.latex(_latex(spec.lhs) + " = RHS")
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Build failed: {e}")
        with c2:
            if st.button("🗑️ Clear"):
                st.session_state.generator_terms = []

# ---------------- ML Pattern Learning (background RQ) ----------------
def page_ml():
    st.header("🤖 ML Pattern Learning (Background + Persistent)")

    # Configuration
    c1,c2,c3 = st.columns(3)
    with c1:
        model_type = st.selectbox("Model", ["pattern_learner","vae","transformer"])
        epochs = st.slider("Epochs", 10, 500, 100)
        batch_size = st.slider("Batch Size", 8, 128, 32)
    with c2:
        samples = st.slider("Training Samples (synth)", 100, 10000, 1000, step=100)
        validation_split = st.slider("Validation Split", 0.1, 0.4, 0.2, step=0.05)
        learning_rate = st.select_slider("Learning Rate", [0.0001,0.0005,0.001,0.005,0.01], value=0.001)
    with c3:
        use_gpu = st.checkbox("Prefer GPU", True)
        enable_amp = st.checkbox("Mixed Precision (AMP)", False)
        resume_from = st.selectbox("Resume from checkpoint (optional)", [""] + _available_checkpoints())

    # Buttons
    cols = st.columns(3)
    with cols[0]:
        if st.button("🚀 Train in Background (RQ)", type="primary"):
            if not has_redis():
                st.error("Redis not available. Configure REDIS_URL and run worker.")
            else:
                payload = {
                    "model_type": model_type,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "samples": samples,
                    "validation_split": validation_split,
                    "learning_rate": learning_rate,
                    "device": ("cuda" if use_gpu else "cpu"),
                    "enable_mixed_precision": enable_amp,
                    "resume_from": (resume_from or None),
                    "checkpoint_dir": _ensure_checkpoint_dir(),
                }
                job_id = enqueue_job("worker.train_job", payload, description="ML training")
                if job_id:
                    st.session_state["train_job_id"] = job_id
                    st.success(f"Training job submitted: {job_id}")
                else:
                    st.error("Failed to enqueue training job.")

    with cols[1]:
        if st.button("📥 Load Best Local Checkpoint"):
            ckpts = _available_checkpoints()
            if not ckpts:
                st.warning("No checkpoints found.")
            elif not MLTrainer:
                st.error("MLTrainer unavailable in web service.")
            else:
                path = ckpts[-1]
                try:
                    trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=("cuda" if use_gpu else "cpu"), checkpoint_dir=_ensure_checkpoint_dir())
                    ok = trainer.load_model(path)
                    if ok:
                        st.session_state.ml_trainer = trainer
                        st.session_state.ml_trained = True
                        st.success(f"Loaded model: {path}")
                    else:
                        st.error("Failed to load model.")
                except Exception as e:
                    st.error(f"Load error: {e}")

    with cols[2]:
        uploaded = st.file_uploader("Upload .pth model or history.json", type=["pth","json"], accept_multiple_files=True)
        if uploaded:
            save_dir = _ensure_checkpoint_dir()
            for f in uploaded:
                out_path = os.path.join(save_dir, f.name)
                with open(out_path, "wb") as wf:
                    wf.write(f.getbuffer())
            st.success(f"Saved to {save_dir}")
            # Try to load last pth automatically
            if MLTrainer:
                pths = [os.path.join(save_dir, f.name) for f in uploaded if f.name.endswith(".pth")]
                if pths:
                    try:
                        trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate, device=("cuda" if use_gpu else "cpu"), checkpoint_dir=_ensure_checkpoint_dir())
                        ok = trainer.load_model(pths[-1])
                        if ok:
                            st.session_state.ml_trainer = trainer
                            st.session_state.ml_trained = True
                            st.success(f"Loaded uploaded model: {pths[-1]}")
                    except Exception as e:
                        st.error(f"Auto-load failed: {e}")

    # Training job status
    _training_status_panel()

    # After trained: generate + reverse
    if st.session_state.get("ml_trained") and st.session_state.get("ml_trainer"):
        st.subheader("🎨 Generate from Trained Model")
        gcols = st.columns(3)
        with gcols[0]:
            to_gen = st.slider("How many?", 1, 10, 1)
        with gcols[1]:
            if st.button("🎲 Generate"):
                for i in range(to_gen):
                    res = st.session_state.ml_trainer.generate_new_ode()
                    if res:
                        _register_generated_ode(res)
                st.success(f"Generated {to_gen} ODE(s). Check Dashboard.")

        st.subheader("🧠 Reverse Engineering (from ODE)")
        ode_str = st.text_area("Paste ODE (LaTeX or SymPy)", "")
        if st.button("🔎 Reverse Engineer"):
            try:
                # naive feature extraction + model decode
                feat = _featurize_ode_text(ode_str)
                if feat is None:
                    st.warning("Could not parse ODE text; falling back to symbolic heuristics.")
                    guess = _symbolic_reverse_heuristic(ode_str)
                    st.write(guess)
                else:
                    # Use model to reconstruct a plausible param vector
                    t = st.session_state.ml_trainer
                    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                    if t.model_type == "vae":
                        y, _, _ = t.model(x.to(t.device))
                    else:
                        y = t.model(x.to(t.device))
                    y = y.detach().cpu().numpy().ravel()
                    pred = {
                        "alpha": float(y[0]),
                        "beta": float(abs(y[1]) + 0.1),
                        "n": int(max(1, round(abs(y[2])))),
                        "M": float(y[3]),
                    }
                    st.json(pred)
            except Exception as e:
                st.error(f"Reverse engineering failed: {e}")

def _training_status_panel():
    job_id = st.session_state.get("train_job_id")
    if not job_id:
        return
    st.subheader("📡 Training Job Status")
    info = fetch_job(job_id)
    st.write(f"**Status:** {info.get('status')}  |  **Desc:** {info.get('description')}")
    meta = info.get("meta", {})
    if meta:
        cols = st.columns(5)
        with cols[0]: st.metric("Stage", meta.get("status","?"))
        with cols[1]: st.metric("Epoch", f"{meta.get('epoch','-')}/{meta.get('epochs','-')}")
        with cols[2]: st.metric("Train Loss", f"{meta.get('train_loss','-')}")
        with cols[3]: st.metric("Val Loss", f"{meta.get('val_loss','-')}")
        with cols[4]:
            if meta.get("progress") is not None:
                st.progress(min(1.0, float(meta["progress"])))
    if info.get("logs_tail"):
        with st.expander("Recent logs"):
            for ev in info["logs_tail"]:
                st.code(json.dumps(ev, ensure_ascii=False))

    if info.get("status") == "finished":
        # localize result: mark trained and optionally load model right away
        result = info.get("result") or {}
        best = result.get("best_model_path")
        if best and MLTrainer:
            try:
                # load model so the UI can immediately use it
                t = MLTrainer(model_type=result.get("model_type","pattern_learner"), learning_rate=0.001,
                              device=("cuda" if (torch and torch.cuda.is_available()) else "cpu"),
                              checkpoint_dir=result.get("checkpoint_dir") or _ensure_checkpoint_dir())
                ok = t.load_model(best)
                if ok:
                    st.session_state.ml_trainer = t
                    st.session_state.ml_trained = True
                    st.session_state.training_history = result.get("history", {})
                    st.success(f"Loaded best model: {best}")
            except Exception as e:
                st.warning(f"Auto-load best model failed: {e}")
        st.session_state["train_job_id"] = None
    elif info.get("status") in {"failed","stopped"}:
        st.error(info.get("exc_info") or info.get("error") or "Training job error")
        st.session_state["train_job_id"] = None

# Rough ODE text featurizer (placeholder)
def _featurize_ode_text(txt: str) -> Optional[List[float]]:
    try:
        # extremely basic: look for tokens to build a 12-dim vector similar to training features
        lower = (txt or "").lower()
        is_linear = 1.0 if ("+" in lower and "y(" in lower and "y'" in lower) else 0.0
        order = float(lower.count("'")) if "'" in lower else (2.0 if "y''" in lower else 1.0)
        alpha = 1.0 if "alpha" in lower else 0.5
        beta  = 1.0 if "beta"  in lower else 0.5
        n     = 2.0 if "n" in lower else 1.0
        M     = 0.0
        func_id = 0.0
        gen_num = 1.0
        q = 0.0; v=0.0; a=0.0
        noise = np.random.randn()*0.05
        return [alpha,beta,n,M,func_id,is_linear,gen_num,order,q,v,a,noise]
    except Exception:
        return None

def _symbolic_reverse_heuristic(txt: str) -> Dict[str, Any]:
    # Try to parse structure and deduce simple traits
    guess = {"type":"Unknown","order":"Unknown","linearity":"Unknown"}
    try:
        if "y''" in txt or "y^{(2)}" in txt: guess["order"] = 2
        elif "y'" in txt or "y^{(1)}" in txt: guess["order"] = 1
        else: guess["order"] = "Unknown"
        guess["linearity"] = "Linear" if ("+" in txt and "y" in txt) else "Nonlinear"
    except Exception:
        pass
    return guess

# ---------------- Batch Generation ----------------
def page_batch():
    st.header("📊 Batch ODE Generation")
    c1,c2,c3 = st.columns(3)
    with c1:
        num_odes = st.slider("How many", 5, 500, 50)
        gen_types = st.multiselect("Types", ["linear","nonlinear"], default=["linear","nonlinear"])
    with c2:
        categories = st.multiselect("Function categories", ["Basic","Special"], default=["Basic"])
        vary = st.checkbox("Vary parameters", True)
    with c3:
        alpha_range = st.slider("α range", -5.0, 5.0, (-2.0, 2.0)) if vary else (1.0,1.0)
        beta_range  = st.slider("β range", 0.1, 5.0, (0.5, 2.0)) if vary else (1.0,1.0)
        n_range     = st.slider("n range", 1, 5, (1, 3)) if vary else (1,1)

    if st.button("🚀 Generate Batch", type="primary"):
        bf = st.session_state.get("basic_functions")
        sf = st.session_state.get("special_functions")
        all_fn = []
        if "Basic" in categories and bf:
            all_fn += bf.get_function_names()
        if "Special" in categories and sf:
            all_fn += sf.get_function_names()[:20]
        if not all_fn:
            st.warning("No functions available.")
            return

        rows = []
        for i in range(num_odes):
            try:
                params = {
                    "alpha": float(np.random.uniform(*alpha_range)),
                    "beta":  float(np.random.uniform(*beta_range)),
                    "n": int(np.random.randint(n_range[0], n_range[1]+1)),
                    "M": float(np.random.uniform(-1,1)),
                }
                func_name = np.random.choice(all_fn)
                gt = np.random.choice(gen_types)
                # Use factories if present (optional)
                res = {"type": gt, "order": 2, "generator_number": 1}
                row = {"ID": i+1, "Type": gt, "Function": func_name, "Order": res.get("order",0)}
                rows.append(row)
            except Exception:
                pass
        st.session_state.batch_results.extend(rows)
        st.success(f"Generated {len(rows)} rows.")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------------- Novelty Detection ----------------
def page_novelty():
    st.header("🔍 Novelty Detection")
    nd = st.session_state.get("novelty_detector")
    if not nd:
        st.info("Novelty detector not available.")
        return
    method = st.radio("Input", ["Current LHS","Text"], horizontal=True)
    target = None
    if method == "Current LHS":
        spec = st.session_state.get("current_generator")
        if spec and hasattr(spec, "lhs"):
            target = {"ode": spec.lhs, "order": getattr(spec,"order",2), "type":"custom"}
        else:
            st.warning("No current generator.")
    else:
        txt = st.text_area("Paste ODE")
        if txt:
            target = {"ode": txt, "order": 2, "type":"manual"}
    if target and st.button("Analyze"):
        try:
            res = nd.analyze(target, check_solvability=True, detailed=True)
            st.metric("Novelty", "🟢" if res.is_novel else "🔴")
            st.metric("Score", f"{res.novelty_score:.1f}")
            st.metric("Confidence", f"{res.confidence:.1%}")
            with st.expander("Details", True):
                st.json({
                    "complexity": res.complexity_level,
                    "recommended_methods": res.recommended_methods[:5] if res.recommended_methods else [],
                    "special_characteristics": res.special_characteristics or []
                })
        except Exception as e:
            st.error(f"Novelty analysis failed: {e}")

# ---------------- Analysis & Classification ----------------
def page_analysis():
    st.header("📈 Analysis & Classification")
    odes = st.session_state.generated_odes
    if not odes:
        st.info("No ODEs yet.")
        return
    df = pd.DataFrame([
        {
            "ID": i+1, "Type": o.get("type","?"), "Order": o.get("order",0),
            "Function": o.get("function_used","?"), "ts": o.get("timestamp","")[:19]
        } for i,o in enumerate(odes)
    ])
    st.dataframe(df, use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Order", title="Order Distribution", nbins=10)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        vc = df["Type"].value_counts()
        fig = px.pie(values=vc.values, names=vc.index, title="Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Visualization ----------------
def page_viz():
    st.header("📐 Visualization")
    odes = st.session_state.generated_odes
    if not odes:
        st.info("No ODEs.")
        return
    sel = st.selectbox("Select ODE", range(len(odes)), format_func=lambda i: f"ODE {i+1} • {odes[i].get('type','?')} (ord {odes[i].get('order',0)})")
    x_range = st.slider("X Range", -10.0, 10.0, (-5.0,5.0))
    pts = st.slider("Points", 100, 2000, 500)
    if st.button("Plot"):
        x = np.linspace(x_range[0], x_range[1], pts)
        y = np.sin(x)*np.exp(-0.1*np.abs(x))  # placeholder; plug numeric solution if available
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Solution"))
        fig.update_layout(title="Solution (illustrative)", xaxis_title="x", yaxis_title="y(x)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Export ----------------
def page_export():
    st.header("📤 Export")
    odes = st.session_state.generated_odes
    if not odes:
        st.info("No ODEs.")
        return
    idx = st.selectbox("Select ODE", range(len(odes)))
    ode = odes[idx]
    tex = "\\begin{equation}\n" + _latex(ode.get("generator","")) + "=" + _latex(ode.get("rhs","")) + "\n\\end{equation}\n" \
          + "\\[\n y(x) = " + _latex(ode.get("solution","")) + "\n\\]"
    st.code(tex, language="latex")
    st.download_button("📄 Download TeX", tex, file_name=f"ode_{idx+1}.tex")

# ---------------- Settings / Docs ----------------
def page_settings():
    st.header("⚙️ Settings")
    if st.button("💾 Save session (basic)"):
        try:
            with open("session_state.pkl","wb") as f:
                pickle.dump({
                    "generated_odes": st.session_state.get("generated_odes"),
                    "training_history": st.session_state.get("training_history"),
                }, f)
            st.success("Saved session_state.pkl")
        except Exception as e:
            st.error(f"Save failed: {e}")

    up = st.file_uploader("Upload session_state.pkl", type=["pkl"])
    if up:
        try:
            data = pickle.loads(up.read())
            for k in ["generated_odes","training_history"]:
                if k in data:
                    st.session_state[k] = data[k]
            st.success("Session loaded.")
        except Exception as e:
            st.error(f"Load failed: {e}")

def page_docs():
    st.header("📖 Documentation")
    st.markdown("""
**Quick Start**
1. Apply Master Theorem → choose f(z), parameters, and LHS source.
2. If Redis+Worker are running, jobs execute in background (see live status).
3. Train ML in **ML Pattern Learning**. Progress persists; artifacts saved to `checkpoints/`.
4. After training: **Generate** new ODEs or **Reverse Engineer** from text.
5. Use **Export** to obtain LaTeX.
""")

# ---------------- Dashboard ----------------
def page_dashboard():
    st.header("🏠 Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>📝 Generated ODEs</h3><h1>{len(st.session_state.generated_odes)}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>📊 Batch Rows</h3><h1>{len(st.session_state.batch_results)}</h1></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>🤖 Model</h3><h1>{"Trained" if st.session_state.get("ml_trained") else "Not Trained"}</h1></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>🧱 Checkpoints</h3><h1>{len(_available_checkpoints())}</h1></div>', unsafe_allow_html=True)
    st.write("Recent ODEs:")
    if st.session_state.generated_odes:
        df = pd.DataFrame(st.session_state.generated_odes[-5:])
        cols = [c for c in ["type","order","function_used","timestamp"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
    else:
        st.info("No ODEs yet. Go to **Apply Master Theorem**.")

# ---------------- Main ----------------
def main():
    st.markdown("""
    <div class="main-header">
      <h2>🔬 Master Generators for ODEs</h2>
      <div class="small">Async jobs via Redis Queue • Training with persistent logs • Reverse engineering • Export</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("📍 Navigation", [
        "🏠 Dashboard", "🔧 Generator Constructor", "🎯 Apply Master Theorem",
        "🤖 ML Pattern Learning", "📊 Batch Generation", "🔍 Novelty Detection",
        "📈 Analysis & Classification", "📐 Visualization", "📤 Export",
        "⚙️ Settings", "📖 Documentation"
    ])
    if page == "🏠 Dashboard": page_dashboard()
    elif page == "🔧 Generator Constructor": page_constructor()
    elif page == "🎯 Apply Master Theorem": page_apply_theorem()
    elif page == "🤖 ML Pattern Learning": page_ml()
    elif page == "📊 Batch Generation": page_batch()
    elif page == "🔍 Novelty Detection": page_novelty()
    elif page == "📈 Analysis & Classification": page_analysis()
    elif page == "📐 Visualization": page_viz()
    elif page == "📤 Export": page_export()
    elif page == "⚙️ Settings": page_settings()
    elif page == "📖 Documentation": page_docs()

if __name__ == "__main__":
    main()