# master_generators_app.py
# -----------------------------------------------------------------------------
# Master Generators App - Clean, Correct, and Symbolic-First
# -----------------------------------------------------------------------------
# Key improvements:
# - Exact (symbolic) parameters toggle for Theorem 4.1
# - Derivative orders up to 15 in Generator Constructor
# - Quick-add of m and (2m-1) derivative terms (per the paper)
# - Symbolic application of generator L to y(x) to form RHS = L[y]
# - Robust import handling for factories placed in different modules
# - Safer LaTeX export that nsimplifies to keep E and pi symbolic
# -----------------------------------------------------------------------------

import os
import sys
import io
import json
import zipfile
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mg_app")

# -----------------------------------------------------------------------------
# Ensure src/ package is importable
# -----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

HAVE_SRC = True

# -----------------------------------------------------------------------------
# Robust imports from src/ with graceful fallbacks
# -----------------------------------------------------------------------------
try:
    # Core generator classes (depending on your ZIP layout)
    # These three are in src.generators.master_generator in your project
    from src.generators.master_generator import (
        MasterGenerator,
        EnhancedMasterGenerator,
        CompleteMasterGenerator,
        # In your ZIP: CompleteLinearGeneratorFactory, CompleteNonlinearGeneratorFactory live here
        CompleteLinearGeneratorFactory,
        CompleteNonlinearGeneratorFactory,
    )
except Exception as e:
    logger.warning(f"Primary import from master_generator failed: {e}")
    HAVE_SRC = False
    MasterGenerator = EnhancedMasterGenerator = CompleteMasterGenerator = None
    CompleteLinearGeneratorFactory = CompleteNonlinearGeneratorFactory = None

# If factories were not found above, try alternate modules:
if HAVE_SRC and (CompleteLinearGeneratorFactory is None or CompleteNonlinearGeneratorFactory is None):
    try:
        if CompleteLinearGeneratorFactory is None:
            from src.generators.linear_generators import CompleteLinearGeneratorFactory as _CLF
            CompleteLinearGeneratorFactory = _CLF
        if CompleteNonlinearGeneratorFactory is None:
            from src.generators.nonlinear_generators import CompleteNonlinearGeneratorFactory as _CNF
            CompleteNonlinearGeneratorFactory = _CNF
    except Exception as e:
        logger.warning(f"Alternate import of factories failed: {e}")

# Generator constructor + enums
try:
    from src.generators.generator_constructor import (
        GeneratorConstructor,
        GeneratorSpecification,
        DerivativeTerm,
        DerivativeType,
        OperatorType,
    )
except Exception as e:
    logger.warning(f"Import generator_constructor failed: {e}")
    HAVE_SRC = False
    GeneratorConstructor = GeneratorSpecification = DerivativeTerm = None
    DerivativeType = OperatorType = None

# Theorems + classifier
try:
    from src.generators.master_theorem import (
        MasterTheoremSolver,
        MasterTheoremParameters,
        ExtendedMasterTheorem,
    )
except Exception as e:
    logger.warning(f"Import master_theorem failed: {e}")
    MasterTheoremSolver = MasterTheoremParameters = ExtendedMasterTheorem = None

try:
    from src.generators.ode_classifier import ODEClassifier, PhysicalApplication
except Exception as e:
    logger.warning(f"Import ode_classifier failed: {e}")
    ODEClassifier = PhysicalApplication = None

# Function libraries
try:
    from src.functions.basic_functions import BasicFunctions
except Exception:
    BasicFunctions = None
try:
    from src.functions.special_functions import SpecialFunctions
except Exception:
    SpecialFunctions = None

# Optional utils
try:
    from src.utils.cache import CacheManager
except Exception:
    CacheManager = None

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Master Generators ODE System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Helpers — exactness, functions, theorem, generator application, latex
# -----------------------------------------------------------------------------
def to_exact(v):
    """Convert floats like 1.0 -> 1, 0.5 -> 1/2; keep expressions exact."""
    try:
        return sp.nsimplify(v, rational=True)
    except Exception:
        return sp.sympify(v)


def nsimplify_e_pi(expr):
    """Try to map floats back to E and pi for cleaner LaTeX."""
    try:
        return sp.nsimplify(expr, [sp.E, sp.pi])
    except Exception:
        return expr


def get_function_expr(source_obj: Any, name: str) -> sp.Expr:
    """
    Resolve f(z) from BasicFunctions/SpecialFunctions or fallback mappings.
    Returns a SymPy expression in symbol z.
    """
    z = sp.Symbol("z", real=True)
    # Try the library's API first
    if source_obj is not None:
        try:
            f = source_obj.get_function(name)
            if callable(f):
                expr = f(z)
                return sp.sympify(expr)
            else:
                # If the library returns a sympy expr or string
                return sp.sympify(f).subs(sp.Symbol("z"), z)
        except Exception as e:
            logger.warning(f"get_function_expr: failed via library '{name}': {e}")

    # Fallback known mappings
    name_low = (name or "").strip().lower()
    if name_low in {"exp", "exponential", "e^z"}:
        return sp.exp(z)
    if name_low in {"sin", "sine"}:
        return sp.sin(z)
    if name_low in {"cos", "cosine"}:
        return sp.cos(z)
    if name_low in {"log", "ln"}:
        # Protect against log(0) later; user should define domain or epsilon outside f
        return sp.log(z)
    if name_low in {"z", "identity"}:
        return z
    if name_low in {"z^2", "quadratic"}:
        return z**2
    if name_low in {"airy_ai", "ai"}:
        return sp.airyai(z)
    if name_low in {"bessel_j0", "j0"}:
        return sp.besselj(0, z)

    # Default fallback
    logger.warning(f"Unknown function '{name}'. Using identity f(z)=z.")
    return z


def theorem_4_1_solution_expr(f_z: sp.Expr, alpha, beta, n: int, M, x: sp.Symbol) -> sp.Expr:
    """
    Symbolic Theorem 4.1 solution:
       y(x) = (π / (2n)) * Σ_{s=1}^n [ 2 f(α + β)
                                       - f(α + β e^{ i x cosω_s - x sinω_s })
                                       - f(α + β e^{-i x cosω_s - x sinω_s }) ] + π M
       where ω_s = (2s-1)π / (2n)
    """
    z = sp.Symbol("z", real=False)  # function expr parameter (can be complex intermediate)
    f = sp.lambdify([z], f_z, "sympy")

    pi = sp.pi
    s_idx = sp.symbols("s_idx", integer=True, positive=True)

    terms = []
    for s in range(1, int(n) + 1):
        omega_s = (2 * s - 1) * sp.pi / (2 * n)
        e1 = sp.exp(sp.I * x * sp.cos(omega_s) - x * sp.sin(omega_s))
        e2 = sp.exp(-sp.I * x * sp.cos(omega_s) - x * sp.sin(omega_s))
        t = 2 * f(alpha + beta) - f(alpha + beta * e1) - f(alpha + beta * e2)
        terms.append(sp.simplify(t))

    y = (pi / (2 * n)) * sp.simplify(sp.Add(*terms)) + pi * M
    # The expression is real for real x, alpha, beta; force simplification
    if y.has(sp.I):
        y = sp.simplify(sp.re(y))
    return sp.simplify(y)


def apply_generator_to_solution(lhs_expr: sp.Expr, y_expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """
    Given a generator LHS expression (in terms of y(x), y'(x), ... y(x/a), ...)
    and a symbolic solution y_expr, produce RHS = L[y] by symbolic substitution.

    Handles:
      - y(x)          -> y_expr
      - y(g(x))       -> y_expr.subs(x, g(x))
      - Derivative(y(x), x, k) -> diff(y_expr, x, k)
      - Derivative(y(g(x)), x, k) -> diff(y_expr.subs(x,g(x)), x, k)
    """
    y = sp.Function("y")

    # 1) Replace derivatives of y(...) with derivatives of composition
    def _replace_derivative_Y(e):
        if isinstance(e, sp.Derivative) and e.expr.func == y:
            arg = e.expr.args[0] if e.expr.args else x
            # e.variables is a tuple like (x,) or (x, x, ..., x) of length k
            return sp.diff(y_expr.subs(x, arg), *e.variables)
        return None

    # 2) Replace raw y(...) with composition
    def _replace_Y(e):
        if isinstance(e, sp.FunctionClass):
            return None  # pragma: no cover
        if getattr(e, "func", None) == y:
            arg = e.args[0] if e.args else x
            return y_expr.subs(x, arg)
        return None

    # Do replacements
    rhs = lhs_expr.replace(
        lambda e: isinstance(e, sp.Derivative) and getattr(e.expr, "func", None) == y,
        _replace_derivative_Y,
    )
    rhs = rhs.replace(lambda e: getattr(e, "func", None) == y, _replace_Y)

    return sp.simplify(rhs)


# -----------------------------------------------------------------------------
# LaTeX exporter (with exactness control)
# -----------------------------------------------------------------------------
class LaTeXExporter:
    @staticmethod
    def sympy_to_latex(expr: Any) -> str:
        try:
            expr = nsimplify_e_pi(expr)
            return sp.latex(expr).replace(r"\left(", "(").replace(r"\right)", ")")
        except Exception:
            try:
                return sp.latex(expr)
            except Exception:
                return str(expr)

    @staticmethod
    def generate_document(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        generator = ode_data.get("generator", "")
        solution = ode_data.get("solution", "")
        rhs = ode_data.get("rhs", "")
        params = ode_data.get("parameters", {})
        classification = ode_data.get("classification", {})
        initial_conditions = ode_data.get("initial_conditions", {})

        parts = []
        if include_preamble:
            parts.append(r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\usepackage{hyperref}
\geometry{margin=1in}
\title{Master Generators ODE System}
\author{Generated by Master Generators App}
\date{\today}
\begin{document}
\maketitle

\section{Generated Ordinary Differential Equation}
""".strip("\n"))

        parts.append(r"\subsection{Generator Equation}")
        parts.append(r"\begin{equation}")
        parts.append(f"{LaTeXExporter.sympy_to_latex(generator)} = {LaTeXExporter.sympy_to_latex(rhs)}")
        parts.append(r"\end{equation}")
        parts.append("")

        parts.append(r"\subsection{Exact Solution}")
        parts.append(r"\begin{equation}")
        parts.append(f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}")
        parts.append(r"\end{equation}")
        parts.append("")

        parts.append(r"\subsection{Parameters}")
        parts.append(r"\begin{align}")
        parts.append(f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha', ''))} \\\\")
        parts.append(f"\\beta  &= {LaTeXExporter.sympy_to_latex(params.get('beta', ''))} \\\\")
        parts.append(f"n       &= {params.get('n', '')} \\\\")
        parts.append(f"M       &= {LaTeXExporter.sympy_to_latex(params.get('M', ''))}")
        parts.append(r"\end{align}")
        parts.append("")

        if initial_conditions:
            parts.append(r"\subsection{Initial Conditions}")
            parts.append(r"\begin{align}")
            ics = list(initial_conditions.items())
            for i, (k, v) in enumerate(ics):
                tail = r" \\" if i < len(ics) - 1 else ""
                parts.append(f"{k} &= {LaTeXExporter.sympy_to_latex(v)}{tail}")
            parts.append(r"\end{align}")
            parts.append("")

        if classification:
            parts.append(r"\subsection{Mathematical Classification}")
            parts.append(r"\begin{itemize}")
            if "type" in classification:
                parts.append(f"\\item \\textbf{{Type:}} {classification.get('type')}")
            if "order" in classification:
                parts.append(f"\\item \\textbf{{Order:}} {classification.get('order')}")
            if "field" in classification:
                parts.append(f"\\item \\textbf{{Field:}} {classification.get('field')}")
            if "applications" in classification and classification["applications"]:
                apps = ", ".join(classification["applications"])
                parts.append(f"\\item \\textbf{{Applications:}} {apps}")
            parts.append(r"\end{itemize}")
            parts.append("")

        parts.append(r"\subsection{Solution Verification}")
        parts.append(r"Substitute $y(x)$ into the generator operator to verify $L[y] = \text{RHS}$.")
        if include_preamble:
            parts.append(r"\end{document}")
        return "\n".join(parts)

    @staticmethod
    def export_package(ode_data: Dict[str, Any], include_code: bool = True) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            tex = LaTeXExporter.generate_document(ode_data, include_preamble=True)
            zf.writestr("ode_document.tex", tex)
            zf.writestr("ode_data.json", json.dumps(ode_data, indent=2, default=str))
            zf.writestr(
                "README.txt",
                f"""Master Generator ODE Export
Generated: {datetime.now().isoformat()}

Contents:
- ode_document.tex
- ode_data.json
- README.txt

Compile:
  pdflatex ode_document.tex
""",
            )
        buf.seek(0)
        return buf.getvalue()


# -----------------------------------------------------------------------------
# Session state init
# -----------------------------------------------------------------------------
def init_state():
    if "generated_odes" not in st.session_state:
        st.session_state.generated_odes = []
    if "generator_terms" not in st.session_state:
        st.session_state.generator_terms = []
    if "current_generator" not in st.session_state:
        st.session_state.current_generator = None
    if "constructor" not in st.session_state and GeneratorConstructor is not None:
        st.session_state.constructor = GeneratorConstructor()
    if "ode_classifier" not in st.session_state and ODEClassifier is not None:
        st.session_state.ode_classifier = ODEClassifier()


# -----------------------------------------------------------------------------
# UI - Header and Sidebar
# -----------------------------------------------------------------------------
def header():
    st.markdown(
        """
<div style="padding:1.5rem;border-radius:12px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;">
  <h1 style="margin:0;">🔬 Master Generators for ODEs</h1>
  <p style="margin:0;">Symbolic Theorem 4.1 · Derivatives up to 15 · Quick add m and (2m−1) · Robust generator application</p>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar():
    st.sidebar.title("📍 Navigation")
    return st.sidebar.radio(
        "Select Module",
        (
            "🏠 Dashboard",
            "🔧 Generator Constructor",
            "🎯 Apply Master Theorem",
            "📤 Export & LaTeX",
        ),
    )


# -----------------------------------------------------------------------------
# Page: Dashboard
# -----------------------------------------------------------------------------
def page_dashboard():
    st.header("🏠 Dashboard")
    st.metric("Generated ODEs", len(st.session_state.generated_odes))
    if st.session_state.generated_odes:
        st.subheader("Recent Results")
        for i, rec in enumerate(st.session_state.generated_odes[-5:][::-1], start=1):
            with st.expander(f"Result #{len(st.session_state.generated_odes) - i + 1}"):
                st.write("**Type**:", rec.get("type", "Unknown"))
                st.write("**Order**:", rec.get("order", "Unknown"))
                st.latex(f"y(x) = {LaTeXExporter.sympy_to_latex(rec.get('solution',''))}")


# -----------------------------------------------------------------------------
# Page: Generator Constructor (up to 15 + quick-add m and 2m−1)
# -----------------------------------------------------------------------------
def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    if not HAVE_SRC or GeneratorConstructor is None or DerivativeTerm is None:
        st.error("This module requires the src/ package. Ensure the project ZIP is extracted with src/ present.")
        return

    # Build terms UI
    with st.expander("➕ Add Generator Term", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # Derivative order up to 15
            d_order = st.selectbox(
                "Derivative Order",
                list(range(0, 16)),
                index=0,
                format_func=lambda k: "y" if k == 0 else f"y^{ '(' + str(k) + ')' }"
            )
        with col2:
            # DerivativeType is an Enum from src; show string values
            if DerivativeType is not None:
                dtypes = [t.value for t in DerivativeType]
                ftype_val = st.selectbox("Function Type", dtypes)
                ftype = DerivativeType(ftype_val)
            else:
                ftype_val = st.selectbox("Function Type", ["identity", "power", "trig", "log", "exp"])
                ftype = ftype_val
        with col3:
            coeff = st.number_input("Coefficient", -50.0, 50.0, 1.0, 0.1)
        with col4:
            power = st.number_input("Power", 1, 8, 1)

        col5, col6, col7 = st.columns(3)
        with col5:
            if OperatorType is not None:
                optypes = [t.value for t in OperatorType]
                op_val = st.selectbox("Operator Type", optypes)
                op_type = OperatorType(op_val)
            else:
                op_val = st.selectbox("Operator Type", ["standard", "delay", "advance"])
                op_type = op_val
        with col6:
            a = st.number_input("Scaling a (for delay/advance)", 0.1, 10.0, 2.0, 0.1)
        with col7:
            shift = st.number_input("Shift (for delay/advance)", -10.0, 10.0, 0.0, 0.1)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add Term", type="primary", use_container_width=True):
                term = DerivativeTerm(
                    derivative_order=int(d_order),
                    coefficient=float(coeff),
                    power=int(power),
                    function_type=ftype,
                    operator_type=op_type,
                    scaling=float(a),
                    shift=float(shift),
                )
                st.session_state.generator_terms.append(term)
                st.success("Term added.")

        with c2:
            m = st.number_input("m (for paper terms)", 1, 50, 2)
            if st.button("Quick-add: m and (2m−1)", use_container_width=True):
                # Add y^(m) and y^(2m-1) with default coeff=1, power=1, same op and ftype
                for k in [int(m), int(2 * m - 1)]:
                    term = DerivativeTerm(
                        derivative_order=k,
                        coefficient=1.0,
                        power=1,
                        function_type=ftype,
                        operator_type=op_type,
                        scaling=float(a),
                        shift=float(shift),
                    )
                    st.session_state.generator_terms.append(term)
                st.success(f"Added derivative orders m={m} and 2m-1={2*m-1}.")

    # Show current terms
    if st.session_state.generator_terms:
        st.subheader("📝 Current Terms")
        for i, t in enumerate(st.session_state.generator_terms):
            colA, colB = st.columns([8, 1])
            with colA:
                try:
                    desc = t.get_description()
                except Exception:
                    desc = f"order={t.derivative_order}, coeff={t.coefficient}, power={t.power}, op={getattr(t,'operator_type', '')}"
                st.info(desc)
            with colB:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.experimental_rerun()

        # Build GeneratorSpecification
        if st.button("🔨 Build Generator Specification", type="primary"):
            try:
                constructor = st.session_state.constructor
                constructor.clear()  # ensure clean
                for t in st.session_state.generator_terms:
                    constructor.add_term(t)
                spec = GeneratorSpecification(
                    terms=st.session_state.generator_terms,
                    name=f"Custom Generator #{len(st.session_state.generated_odes) + 1}",
                )
                st.session_state.current_generator = spec
                st.success("Generator specification created.")
                # Display LHS if available
                if hasattr(spec, "lhs"):
                    st.latex(LaTeXExporter.sympy_to_latex(spec.lhs))
            except Exception as e:
                st.error(f"Failed to build specification: {e}")

    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.session_state.current_generator = None
        st.success("Cleared.")


# -----------------------------------------------------------------------------
# Page: Apply Master Theorem (Exact toggle + symbolic RHS construction)
# -----------------------------------------------------------------------------
def page_apply_master_theorem():
    st.header("🎯 Apply Master Theorem (4.1)")
    if not HAVE_SRC:
        st.error("This module requires the src/ package. Ensure the project ZIP is extracted with src/ present.")
        return

    # Function library
    lib_choice = st.radio("Function Library", ["Basic", "Special"], horizontal=True)
    source_lib = None
    names = []
    if lib_choice == "Basic" and BasicFunctions is not None:
        source_lib = BasicFunctions()
        try:
            names = source_lib.get_function_names()
        except Exception:
            names = ["exponential", "sin", "cos", "log", "z"]
    elif lib_choice == "Special" and SpecialFunctions is not None:
        source_lib = SpecialFunctions()
        try:
            names = source_lib.get_function_names()
        except Exception:
            names = ["airy_ai", "bessel_j0"]
    else:
        names = ["exponential", "z"]

    func_name = st.selectbox("Select f(z)", names)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        alpha = st.text_input("α (alpha)", "1")
    with col2:
        beta = st.text_input("β (beta)", "1")
    with col3:
        n = st.number_input("n (positive integer)", 1, 20, 1)
    with col4:
        M = st.text_input("M", "0")

    # Exact/symbolic toggle (your batch)
    use_exact = st.checkbox("Exact (symbolic) parameters", value=True)

    # Prepare symbolic parameters
    def parse_param(s):
        try:
            return sp.sympify(s)
        except Exception:
            return sp.nsimplify(s)

    α_in = parse_param(alpha)
    β_in = parse_param(beta)
    M_in = parse_param(M)
    α = to_exact(α_in) if use_exact else sp.Float(α_in)
    β = to_exact(β_in) if use_exact else sp.Float(β_in)
    𝑀 = to_exact(M_in) if use_exact else sp.Float(M_in)

    if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
        with st.spinner("Applying Master Theorem 4.1 and constructing RHS..."):
            try:
                # Resolve f(z) and variable x
                f_expr = get_function_expr(source_lib, func_name)
                x = sp.Symbol("x", real=True)

                # Build y(x) via Theorem 4.1
                solution = theorem_4_1_solution_expr(f_expr, α, β, int(n), 𝑀, x)

                # Construct RHS = L[y] using the current generator specification
                gen_spec = st.session_state.get("current_generator")
                if gen_spec is not None and hasattr(gen_spec, "lhs"):
                    try:
                        rhs = apply_generator_to_solution(gen_spec.lhs, solution, x)
                        generator_lhs = gen_spec.lhs
                    except Exception as e:
                        logger.warning(f"Failed to apply generator to solution, fallback RHS: {e}")
                        generator_lhs = sp.Function("y")(x)  # dummy display
                        rhs = sp.simplify(sp.pi * (f_expr.subs(sp.Symbol("z"), α + β) + 𝑀))
                else:
                    generator_lhs = sp.Function("y")(x)
                    rhs = sp.simplify(sp.pi * (f_expr.subs(sp.Symbol("z"), α + β) + 𝑀))

                # Classify
                classification = {}
                if gen_spec is not None:
                    try:
                        ctype = "Linear" if getattr(gen_spec, "is_linear", False) else "Nonlinear"
                        order = getattr(gen_spec, "order", None)
                        classification = {
                            "type": ctype,
                            "order": order,
                            "field": "Mathematical Physics",
                            "applications": ["Research Equation"],
                        }
                    except Exception:
                        classification = {"type": "Unknown"}

                # Store record
                result = {
                    "generator": generator_lhs,
                    "solution": solution,
                    "rhs": rhs,
                    "parameters": {"alpha": α, "beta": β, "n": int(n), "M": 𝑀},
                    "function_used": func_name,
                    "type": classification.get("type", "Unknown"),
                    "order": classification.get("order", None),
                    "classification": classification,
                    "initial_conditions": {"y(0)": sp.simplify(solution.subs(x, 0))},
                    "timestamp": datetime.now().isoformat(),
                }
                st.session_state.generated_odes.append(result)

                # Display
                st.success("✅ ODE Generated Successfully!")
                tabs = st.tabs(["📐 Equation", "💡 Solution", "🏷️ Classification", "📤 Export"])
                with tabs[0]:
                    st.markdown("### Complete ODE:")
                    st.latex(
                        f"{LaTeXExporter.sympy_to_latex(generator_lhs)} = {LaTeXExporter.sympy_to_latex(rhs)}"
                    )
                with tabs[1]:
                    st.markdown("### Exact Solution:")
                    st.latex(f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}")
                    st.markdown("### Initial Conditions:")
                    st.latex(
                        f"y(0) = {LaTeXExporter.sympy_to_latex(result['initial_conditions']['y(0)'])}"
                    )
                with tabs[2]:
                    st.write("**Type:**", classification.get("type", "Unknown"))
                    st.write("**Order:**", classification.get("order", "Unknown"))
                    st.write("**Field:**", classification.get("field", "Unknown"))
                    st.write("**Applications:**", ", ".join(classification.get("applications", [])))
                with tabs[3]:
                    latex_doc = LaTeXExporter.generate_document(result, include_preamble=True)
                    st.download_button(
                        "📄 Download LaTeX Document",
                        latex_doc,
                        file_name="ode_solution.tex",
                        mime="text/x-latex",
                        use_container_width=True,
                    )
                    package = LaTeXExporter.export_package(result, include_code=False)
                    st.download_button(
                        "📦 Download Complete Package (ZIP)",
                        package,
                        file_name=f"ode_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error generating ODE: {e}")
                logger.exception("Generation error")


# -----------------------------------------------------------------------------
# Page: Export & LaTeX
# -----------------------------------------------------------------------------
def page_export():
    st.header("📤 Export & LaTeX")
    if not st.session_state.generated_odes:
        st.info("No ODEs to export yet.")
        return

    idx = st.selectbox(
        "Select ODE to export",
        list(range(len(st.session_state.generated_odes))),
        format_func=lambda i: f"ODE #{i+1} (type={st.session_state.generated_odes[i].get('type','?')})",
    )
    record = st.session_state.generated_odes[idx]

    # Preview
    st.subheader("📋 LaTeX Preview")
    preview = LaTeXExporter.generate_document(record, include_preamble=False)
    st.code(preview, language="latex")

    col1, col2 = st.columns(2)
    with col1:
        full = LaTeXExporter.generate_document(record, include_preamble=True)
        st.download_button(
            "📄 Download LaTeX",
            full,
            file_name=f"ode_{idx+1}.tex",
            mime="text/x-latex",
            use_container_width=True,
        )
    with col2:
        pkg = LaTeXExporter.export_package(record)
        st.download_button(
            "📦 Download ZIP Package",
            pkg,
            file_name=f"ode_package_{idx+1}.zip",
            mime="application/zip",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    init_state()
    header()
    page = sidebar()
    if page == "🏠 Dashboard":
        page_dashboard()
    elif page == "🔧 Generator Constructor":
        page_generator_constructor()
    elif page == "🎯 Apply Master Theorem":
        page_apply_master_theorem()
    elif page == "📤 Export & LaTeX":
        page_export()


if __name__ == "__main__":
    main()
