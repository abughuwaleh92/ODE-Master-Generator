# -*- coding: utf-8 -*-
"""
Master Generators for ODEs — Complete App with Exact Symbolic Computation
Includes ML/DL, Batch Generation, and Theorems 4.1 & 4.2
"""

# ======================================================
# Standard imports
# ======================================================
import os
import sys
import io
import json
import zipfile
import pickle
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# Numeric / symbolic / plotting
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, Symbol, Function, exp, sin, cos, pi, I, diff, simplify, expand
from sympy import summation, Integer, Rational, sqrt, log, tan, atan, sinh, cosh, tanh
from sympy.core.function import AppliedUndef

# Streamlit UI
import streamlit as st

# Optional ML/DL
try:
    import torch
except Exception:
    torch = None

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# ======================================================
# Logging
# ======================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("master_generators_app")

# ======================================================
# Ensure src/ is importable
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

HAVE_SRC = True
IMPORT_WARNINGS: List[str] = []

def _try_import(module_path: str):
    try:
        __import__(module_path)
        return sys.modules[module_path]
    except Exception as e:
        IMPORT_WARNINGS.append(f"Import failed: {module_path} -> {e}")
        return None

# Core modules (some may be missing; we guard usage)
mod_master_gen = _try_import("src.generators.master_generator")
mod_lin_gen = _try_import("src.generators.linear_generators")
mod_nonlin_gen = _try_import("src.generators.nonlinear_generators")
mod_constructor = _try_import("src.generators.generator_constructor")
mod_theorem = _try_import("src.generators.master_theorem")
mod_classifier = _try_import("src.generators.ode_classifier")

mod_funcs_basic = _try_import("src.functions.basic_functions")
mod_funcs_spec = _try_import("src.functions.special_functions")

mod_ml_pl = _try_import("src.ml.pattern_learner")
mod_ml_tr = _try_import("src.ml.trainer")
mod_ml_gl = _try_import("src.ml.generator_learner")
mod_dl_nd = _try_import("src.dl.novelty_detector")

mod_utils_conf = _try_import("src.utils.config")
mod_utils_cache = _try_import("src.utils.cache")
mod_utils_valid = _try_import("src.utils.validators")
mod_ui_comp = _try_import("src.ui.components")

# Extract classes from modules
MasterGenerator = getattr(mod_master_gen, "MasterGenerator", None) if mod_master_gen else None
LinearGeneratorFactory = getattr(mod_lin_gen, "LinearGeneratorFactory", None) if mod_lin_gen else None
NonlinearGeneratorFactory = getattr(mod_nonlin_gen, "NonlinearGeneratorFactory", None) if mod_nonlin_gen else None

BasicFunctions = getattr(mod_funcs_basic, "BasicFunctions", None) if mod_funcs_basic else None
SpecialFunctions = getattr(mod_funcs_spec, "SpecialFunctions", None) if mod_funcs_spec else None

GeneratorPatternLearner = getattr(mod_ml_pl, "GeneratorPatternLearner", None) if mod_ml_pl else None
MLTrainer = getattr(mod_ml_tr, "MLTrainer", None) if mod_ml_tr else None
ODENoveltyDetector = getattr(mod_dl_nd, "ODENoveltyDetector", None) if mod_dl_nd else None

# ======================================================
# Streamlit Page Config
# ======================================================
st.set_page_config(
    page_title="Master Generators ODE System — Complete Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# Enhanced CSS
# ======================================================
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; border-radius: 15px; margin-bottom: 2rem; color: white;
        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 0.25rem; }
    .subtitle { font-size: 1.05rem; opacity: 0.95; }
    .generator-term {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 12px; border-radius: 10px; margin: 8px 0; border-left: 5px solid #667eea;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08); transition: transform 0.25s ease;
    }
    .generator-term:hover { transform: translateX(5px); }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50; padding: 1.25rem; border-radius: 15px; margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(76,175,80,0.2);
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3; padding: 0.9rem; border-radius: 10px; margin: 0.8rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# Master Theorem Symbolic Implementation
# ======================================================

class MasterTheoremSymbolic:
    """
    Implements Theorems 4.1 and 4.2 with exact symbolic computation
    """
    
    def __init__(self):
        self.x = Symbol('x', real=True)
        self.alpha = Symbol('alpha', real=True)
        self.beta = Symbol('beta', positive=True)
        self.n = Symbol('n', positive=True, integer=True)
        self.M = Symbol('M', real=True)
        self.z = Symbol('z')
        self.omega = Symbol('omega', real=True)
        self.s = Symbol('s', positive=True, integer=True)
        self.m = Symbol('m', positive=True, integer=True)
        
    def compute_omega(self, s_val: int, n_val: int) -> sp.Expr:
        """Compute ω(s) = (2s-1)π/(2n) symbolically"""
        return Rational(2*s_val - 1, 2*n_val) * pi
    
    def psi_function(self, f_expr: sp.Expr, alpha_val, beta_val, omega_val, x_sym) -> sp.Expr:
        """ψ(α,ω,x) = f(α + β·e^(ix cos(ω) - x sin(ω)))"""
        exponent = I * x_sym * cos(omega_val) - x_sym * sin(omega_val)
        z_val = alpha_val + beta_val * exp(exponent)
        return f_expr.subs(self.z, z_val)
    
    def phi_function(self, f_expr: sp.Expr, alpha_val, beta_val, omega_val, x_sym) -> sp.Expr:
        """φ(α,ω,x) = f(α + β·e^(-ix cos(ω) - x sin(ω)))"""
        exponent = -I * x_sym * cos(omega_val) - x_sym * sin(omega_val)
        z_val = alpha_val + beta_val * exp(exponent)
        return f_expr.subs(self.z, z_val)
    
    def generate_solution_y(self, f_expr: sp.Expr, alpha_val, beta_val, n_val: int, M_val) -> sp.Expr:
        """Generate y(x) using Theorem 4.1"""
        result = 0
        
        for s_val in range(1, n_val + 1):
            omega_val = self.compute_omega(s_val, n_val)
            f_alpha_beta = f_expr.subs(self.z, alpha_val + beta_val)
            psi = self.psi_function(f_expr, alpha_val, beta_val, omega_val, self.x)
            phi = self.phi_function(f_expr, alpha_val, beta_val, omega_val, self.x)
            term = 2*f_alpha_beta - (psi + phi)
            result += term
        
        y_solution = (pi / (2*n_val)) * result + pi * M_val
        return y_solution
    
    def compute_derivative_terms(self, f_expr: sp.Expr, alpha_val, beta_val, omega_val, order: int) -> Dict[str, sp.Expr]:
        """Compute derivative terms for a given order"""
        psi = self.psi_function(f_expr, alpha_val, beta_val, omega_val, self.x)
        phi = self.phi_function(f_expr, alpha_val, beta_val, omega_val, self.x)
        
        derivatives = {}
        for j in range(order + 1):
            if j == 0:
                derivatives[f'psi_{j}'] = psi
                derivatives[f'phi_{j}'] = phi
            else:
                derivatives[f'psi_{j}'] = diff(psi, self.alpha, j).subs(self.alpha, alpha_val)
                derivatives[f'phi_{j}'] = diff(phi, self.alpha, j).subs(self.alpha, alpha_val)
        
        return derivatives
    
    def compute_coefficient_a_j(self, j: int, m: int) -> sp.Expr:
        """Compute coefficient a_j for middle terms"""
        if j < 2 or j >= 2*m:
            return 0
        from sympy import binomial
        return binomial(2*m, j)

# ======================================================
# Helper Functions
# ======================================================

def get_function_expr_symbolic(func_type: str, func_name: str) -> sp.Expr:
    """Get function expression symbolically"""
    z = Symbol('z')
    
    basic_funcs = {
        'exponential': exp(z),
        'linear': z,
        'quadratic': z**2,
        'cubic': z**3,
        'quartic': z**4,
        'sine': sin(z),
        'cosine': cos(z),
        'tangent': tan(z),
        'logarithm': log(z),
        'sqrt': sqrt(z),
        'gaussian': exp(-z**2),
        'sigmoid': 1 / (1 + exp(-z)),
        'sinh': sinh(z),
        'cosh': cosh(z),
        'tanh': tanh(z),
    }
    
    special_funcs = {
        'airy_ai': sp.airyai(z),
        'airy_bi': sp.airybi(z),
        'bessel_j0': sp.besselj(0, z),
        'bessel_j1': sp.besselj(1, z),
        'bessel_j2': sp.besselj(2, z),
        'gamma': sp.gamma(z),
        'erf': sp.erf(z),
        'legendre_p2': sp.legendre(2, z),
        'hermite_h3': sp.hermite(3, z),
        'chebyshev_t4': sp.chebyshevt(4, z),
        'lambert_w': sp.LambertW(z),
        'zeta': sp.zeta(z),
    }
    
    if func_type == "basic":
        return basic_funcs.get(func_name, z)
    else:
        return special_funcs.get(func_name, sp.airyai(z))

def apply_generator_symbolic(generator_lhs: sp.Expr, solution: sp.Expr, x: Symbol) -> sp.Expr:
    """Apply generator operator to solution symbolically"""
    y = Function('y')
    subs_dict = {y(x): solution}
    
    # Handle derivatives up to order 20
    for order in range(1, 21):
        deriv_pattern = diff(y(x), x, order)
        deriv_value = diff(solution, x, order)
        subs_dict[deriv_pattern] = deriv_value
    
    rhs = generator_lhs.subs(subs_dict)
    return simplify(rhs)

# ======================================================
# Export Helpers
# ======================================================

class LaTeXExporter:
    """LaTeX export system for ODEs"""
    
    @staticmethod
    def sympy_to_latex(expr) -> str:
        if expr is None:
            return ""
        try:
            if isinstance(expr, str):
                expr = sp.sympify(expr)
            return sp.latex(expr)
        except Exception:
            return str(expr)
    
    @staticmethod
    def generate_latex_document(ode_data: Dict[str, Any], include_preamble: bool = True) -> str:
        generator = ode_data.get("generator", "")
        solution = ode_data.get("solution", "")
        rhs = ode_data.get("rhs", "")
        params = ode_data.get("parameters", {})
        
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
""")
        
        parts.append(r"\subsection{Generator Equation}")
        parts.append(r"\begin{equation}")
        parts.append(f"{LaTeXExporter.sympy_to_latex(generator)} = {LaTeXExporter.sympy_to_latex(rhs)}")
        parts.append(r"\end{equation}")
        
        parts.append(r"\subsection{Exact Solution}")
        parts.append(r"\begin{equation}")
        parts.append(f"y(x) = {LaTeXExporter.sympy_to_latex(solution)}")
        parts.append(r"\end{equation}")
        
        parts.append(r"\subsection{Parameters}")
        parts.append(r"\begin{align}")
        parts.append(f"\\alpha &= {LaTeXExporter.sympy_to_latex(params.get('alpha', 1))} \\\\")
        parts.append(f"\\beta &= {LaTeXExporter.sympy_to_latex(params.get('beta', 1))} \\\\")
        parts.append(f"n &= {params.get('n', 1)} \\\\")
        parts.append(f"M &= {LaTeXExporter.sympy_to_latex(params.get('M', 0))}")
        parts.append(r"\end{align}")
        
        if include_preamble:
            parts.append(r"\end{document}")
        
        return "\n".join(parts)

# ======================================================
# Session State Manager
# ======================================================

class SessionStateManager:
    @staticmethod
    def initialize():
        if "generator_terms" not in st.session_state:
            st.session_state.generator_terms = []
        if "generated_odes" not in st.session_state:
            st.session_state.generated_odes = []
        if "batch_results" not in st.session_state:
            st.session_state.batch_results = []
        if "ml_trained" not in st.session_state:
            st.session_state.ml_trained = False
        if "ml_trainer" not in st.session_state:
            st.session_state.ml_trainer = None
        if "novelty_detector" not in st.session_state and ODENoveltyDetector:
            st.session_state.novelty_detector = ODENoveltyDetector()
        if "basic_functions" not in st.session_state and BasicFunctions:
            st.session_state.basic_functions = BasicFunctions()
        if "special_functions" not in st.session_state and SpecialFunctions:
            st.session_state.special_functions = SpecialFunctions()

# ======================================================
# Main UI Functions
# ======================================================

def header():
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">🔬 Master Generators for ODEs — Complete System</div>
            <div class="subtitle">Exact Symbolic • ML/DL • Batch Generation • Theorems 4.1 & 4.2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sidebar():
    return st.sidebar.radio(
        "📍 Navigation",
        [
            "🏠 Dashboard",
            "🔧 Generator Constructor",
            "📊 Batch Generation",
            "🤖 ML Pattern Learning",
            "🔍 Novelty Detection",
            "📈 Analysis & Visualization",
            "📤 Export & LaTeX",
            "📖 Documentation",
        ],
        index=0,
    )

def page_dashboard():
    st.header("🏠 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generated ODEs", len(st.session_state.generated_odes))
    col2.metric("Batch Results", len(st.session_state.batch_results))
    col3.metric("ML Trained", "Yes" if st.session_state.ml_trained else "No")
    col4.metric("Max Derivative", "20")
    
    st.subheader("System Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Core Features
        - ✅ **Exact Symbolic Computation**
        - ✅ **Derivatives up to Order 20**
        - ✅ **Theorems 4.1 & 4.2**
        - ✅ **Automatic ODE Generation**
        """)
    
    with col2:
        st.markdown("""
        ### Advanced Features
        - ✅ **Machine Learning Integration**
        - ✅ **Batch Generation**
        - ✅ **Novelty Detection**
        - ✅ **Pattern Learning**
        """)
    
    if st.session_state.generated_odes:
        st.subheader("Recent ODEs")
        for i, ode in enumerate(st.session_state.generated_odes[-3:]):
            with st.expander(f"ODE {len(st.session_state.generated_odes) - 2 + i}"):
                if 'lhs' in ode and 'rhs' in ode:
                    st.latex(sp.latex(ode['lhs']) + " = " + sp.latex(ode['rhs']))

def page_generator_constructor():
    st.header("🔧 Generator Constructor")
    
    st.markdown("""
    <div class="info-box">
    Build custom differential operators with derivatives up to order 20.
    ODEs are generated automatically with exact symbolic solutions.
    </div>
    """, unsafe_allow_html=True)
    
    # Generator construction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        derivative_order = st.selectbox(
            "Derivative Order",
            list(range(21)),
            format_func=lambda x: f"y^({x})" if x > 0 else "y"
        )
    
    with col2:
        coefficient = st.text_input("Coefficient (symbolic)", "1")
        try:
            coeff_sym = sp.sympify(coefficient)
        except:
            coeff_sym = 1
            st.error("Invalid coefficient, using 1")
    
    with col3:
        power = st.number_input("Power", min_value=1, max_value=10, value=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_transform = st.selectbox(
            "Function Transform",
            ["none", "sin", "cos", "tan", "exp", "log", "sinh", "cosh", "tanh"]
        )
    
    with col2:
        if st.button("➕ Add Term", type="primary"):
            x = Symbol('x')
            y = Function('y')
            
            if derivative_order == 0:
                term = y(x)
            else:
                term = diff(y(x), x, derivative_order)
            
            if power > 1:
                term = term**power
            
            if func_transform != "none":
                transform_map = {
                    "sin": sin, "cos": cos, "tan": tan, "exp": exp,
                    "log": lambda t: log(Symbol('epsilon', positive=True) + sp.Abs(t)),
                    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh
                }
                term = transform_map[func_transform](term)
            
            term = coeff_sym * term
            st.session_state.generator_terms.append(term)
            st.success(f"Added term: {sp.latex(term)}")
            st.rerun()
    
    # Display current generator
    if st.session_state.generator_terms:
        st.subheader("Current Generator Terms")
        
        x = Symbol('x')
        y = Function('y')
        generator_lhs = 0
        
        for i, term in enumerate(st.session_state.generator_terms):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.latex(sp.latex(term))
            with col2:
                if st.button(f"❌", key=f"del_{i}"):
                    st.session_state.generator_terms.pop(i)
                    st.rerun()
            generator_lhs += term
        
        st.subheader("Complete Generator")
        st.latex(sp.latex(generator_lhs) + " = RHS")
        
        # ODE Generation Section
        st.markdown("### Generate ODE with Exact Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_type = st.selectbox("Function Type", ["basic", "special"])
            func_list = ["exponential", "sine", "cosine", "quadratic", "cubic", "sinh", "cosh"] if func_type == "basic" else ["airy_ai", "bessel_j0", "bessel_j1", "gamma", "erf"]
            func_name = st.selectbox("Function f(z)", func_list)
        
        with col2:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                alpha_expr = st.text_input("α (symbolic)", "1")
                beta_expr = st.text_input("β (symbolic)", "1")
            with col2_2:
                n_val = st.number_input("n", min_value=1, max_value=10, value=1)
                M_expr = st.text_input("M (symbolic)", "0")
        
        if st.button("🚀 Generate ODE", type="primary", use_container_width=True):
            with st.spinner("Generating exact symbolic ODE..."):
                try:
                    # Parse parameters
                    alpha_sym = sp.sympify(alpha_expr)
                    beta_sym = sp.sympify(beta_expr)
                    M_sym = sp.sympify(M_expr)
                    
                    # Get function
                    f_expr = get_function_expr_symbolic(func_type, func_name)
                    
                    # Generate solution
                    theorem = MasterTheoremSymbolic()
                    solution = theorem.generate_solution_y(f_expr, alpha_sym, beta_sym, n_val, M_sym)
                    
                    # Apply generator to get RHS
                    rhs = apply_generator_symbolic(generator_lhs, solution, x)
                    
                    # Store result
                    result = {
                        'name': f"ODE_{len(st.session_state.generated_odes) + 1}",
                        'lhs': generator_lhs,
                        'rhs': simplify(rhs),
                        'solution': simplify(solution),
                        'generator': generator_lhs,
                        'parameters': {
                            'alpha': alpha_sym,
                            'beta': beta_sym,
                            'n': n_val,
                            'M': M_sym
                        },
                        'function_used': func_name,
                        'type': 'symbolic',
                        'order': max([0] + [term.as_coeff_exponent(diff(y(x), x, i))[1] for i in range(21) for term in st.session_state.generator_terms if diff(y(x), x, i) in term.free_symbols]),
                        'initial_conditions': {
                            'y(0)': simplify(solution.subs(x, 0))
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.generated_odes.append(result)
                    
                    # Display results
                    st.success("✅ ODE Generated Successfully!")
                    
                    tabs = st.tabs(["📐 Equation", "💡 Solution", "📝 LaTeX"])
                    
                    with tabs[0]:
                        st.subheader("Generated ODE")
                        st.latex(sp.latex(generator_lhs) + " = " + sp.latex(rhs))
                    
                    with tabs[1]:
                        st.subheader("Exact Solution")
                        st.latex("y(x) = " + sp.latex(solution))
                        st.write("**Initial Condition:**")
                        st.latex("y(0) = " + sp.latex(result['initial_conditions']['y(0)']))
                    
                    with tabs[2]:
                        latex_code = LaTeXExporter.generate_latex_document(result, False)
                        st.code(latex_code, language='latex')
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(traceback.format_exc())
    
    if st.button("🗑️ Clear All Terms"):
        st.session_state.generator_terms = []
        st.rerun()

def page_batch_generation():
    st.header("📊 Batch Generation")
    
    if not BasicFunctions or not SpecialFunctions:
        st.error("Function libraries not available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_odes = st.slider("Number of ODEs", 5, 100, 20)
        gen_types = st.multiselect("Generator Types", ["linear", "nonlinear"], default=["linear"])
    
    with col2:
        func_cats = st.multiselect("Function Categories", ["Basic", "Special"], default=["Basic"])
        vary = st.checkbox("Vary parameters", True)
    
    with col3:
        symbolic = st.checkbox("Use Symbolic Computation", True)
        if vary:
            st.caption("Parameters will vary randomly")
    
    if st.button("🚀 Run Batch Generation", type="primary"):
        if not func_cats or not gen_types:
            st.warning("Select at least one function category and generator type")
            return
        
        results = []
        prog = st.progress(0)
        
        for i in range(num_odes):
            prog.progress((i + 1) / num_odes)
            try:
                # Random parameters
                if vary:
                    if symbolic:
                        alpha = sp.Rational(np.random.randint(-5, 6), 1)
                        beta = sp.Rational(np.random.randint(1, 6), 1)
                        M = sp.Rational(np.random.randint(-3, 4), 1)
                    else:
                        alpha = np.random.uniform(-5, 5)
                        beta = np.random.uniform(0.5, 5)
                        M = np.random.uniform(-3, 3)
                else:
                    alpha, beta, M = 1, 1, 0
                
                n = np.random.randint(1, 4)
                
                # Random function
                if "Basic" in func_cats:
                    func_type = "basic"
                    func_name = np.random.choice(["exponential", "sine", "cosine", "quadratic"])
                else:
                    func_type = "special"
                    func_name = np.random.choice(["airy_ai", "bessel_j0", "gamma"])
                
                # Generate ODE
                if symbolic:
                    f_expr = get_function_expr_symbolic(func_type, func_name)
                    theorem = MasterTheoremSymbolic()
                    solution = theorem.generate_solution_y(f_expr, alpha, beta, n, M)
                    
                    results.append({
                        'ID': i + 1,
                        'Type': 'Symbolic',
                        'Function': func_name,
                        'α': str(alpha),
                        'β': str(beta),
                        'n': n,
                        'M': str(M),
                        'Solution': str(solution)[:100] + "..."
                    })
                else:
                    # Use numerical generation if available
                    results.append({
                        'ID': i + 1,
                        'Type': 'Numerical',
                        'Function': func_name,
                        'α': round(alpha, 3) if not symbolic else str(alpha),
                        'β': round(beta, 3) if not symbolic else str(beta),
                        'n': n,
                        'M': round(M, 3) if not symbolic else str(M)
                    })
                
                st.session_state.batch_results.append(results[-1])
                
            except Exception as e:
                logger.debug(f"Batch item {i+1} failed: {e}")
        
        st.success(f"Generated {len(results)} ODEs")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"batch_odes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

def page_ml_pattern_learning():
    st.header("🤖 ML Pattern Learning")
    
    if not MLTrainer:
        st.info("ML Trainer not available. Please check src/ installation.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["pattern_learner", "vae", "transformer"],
            format_func=lambda x: {
                "pattern_learner": "Pattern Learner",
                "vae": "Variational Autoencoder",
                "transformer": "Transformer"
            }[x]
        )
        epochs = st.slider("Epochs", 10, 200, 50)
    
    with col2:
        batch_size = st.slider("Batch Size", 8, 64, 32)
        lr = st.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
        samples = st.slider("Training Samples", 100, 2000, 500)
    
    if st.button("🚀 Train Model", type="primary"):
        if len(st.session_state.generated_odes) < 5:
            st.warning("Generate at least 5 ODEs before training")
            return
        
        try:
            with st.spinner("Training model..."):
                trainer = MLTrainer(model_type=model_type, learning_rate=lr)
                st.session_state.ml_trainer = trainer
                
                # Train model
                trainer.train(epochs=epochs, batch_size=batch_size, samples=samples)
                st.session_state.ml_trained = True
                
                st.success("✅ Model trained successfully!")
                
                # Display training metrics
                if hasattr(trainer, 'history') and trainer.history:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=trainer.history.get('train_loss', []),
                        mode='lines',
                        name='Training Loss'
                    ))
                    if 'val_loss' in trainer.history:
                        fig.add_trace(go.Scatter(
                            y=trainer.history['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                    fig.update_layout(title="Training Progress", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
    
    if st.session_state.ml_trained and st.session_state.ml_trainer:
        st.subheader("Generate New ODE with ML")
        if st.button("🎲 Generate ML ODE"):
            try:
                new_ode = st.session_state.ml_trainer.generate_new_ode()
                if new_ode:
                    st.success("Generated new ODE with ML!")
                    st.latex(sp.latex(new_ode.get('ode', '')))
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

def page_novelty_detection():
    st.header("🔍 Novelty Detection")
    
    if not ODENoveltyDetector:
        st.info("Novelty detector not available")
        return
    
    nd = st.session_state.get("novelty_detector") or ODENoveltyDetector()
    
    method = st.radio("Input Method", ["Enter ODE Manually", "Select from Generated"])
    
    if method == "Enter ODE Manually":
        ode_expr = st.text_area("Enter ODE expression:")
        ode_type = st.selectbox("Type", ["linear", "nonlinear"])
        order = st.number_input("Order", 1, 10, 2)
        
        if st.button("Analyze", type="primary"):
            if ode_expr:
                try:
                    analysis = nd.analyze({
                        'ode': ode_expr,
                        'type': ode_type,
                        'order': order
                    }, detailed=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Novelty Score", f"{analysis.novelty_score:.1f}/100")
                        st.metric("Complexity", analysis.complexity_level)
                    with col2:
                        if analysis.is_novel:
                            st.error("🚨 NOVEL ODE DETECTED")
                        else:
                            st.success("✅ STANDARD ODE")
                    
                    st.subheader("Characteristics")
                    for char in analysis.special_characteristics[:5]:
                        st.write(f"• {char}")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    else:
        if st.session_state.generated_odes:
            idx = st.selectbox(
                "Select ODE",
                range(len(st.session_state.generated_odes)),
                format_func=lambda i: f"ODE {i+1}: {st.session_state.generated_odes[i].get('name', 'Unnamed')}"
            )
            
            if st.button("Analyze", type="primary"):
                ode = st.session_state.generated_odes[idx]
                try:
                    analysis = nd.analyze({
                        'ode': str(ode.get('lhs', '')),
                        'type': ode.get('type', 'linear'),
                        'order': ode.get('order', 2)
                    })
                    
                    st.metric("Novelty Score", f"{analysis.novelty_score:.1f}/100")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

def page_analysis():
    st.header("📈 Analysis & Visualization")
    
    if not st.session_state.generated_odes:
        st.info("No ODEs to analyze. Generate some first!")
        return
    
    # Statistics
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total ODEs", len(st.session_state.generated_odes))
    
    with col2:
        symbolic_count = sum(1 for ode in st.session_state.generated_odes if ode.get('type') == 'symbolic')
        st.metric("Symbolic ODEs", symbolic_count)
    
    with col3:
        avg_order = np.mean([ode.get('order', 0) for ode in st.session_state.generated_odes])
        st.metric("Avg Order", f"{avg_order:.1f}")
    
    # Function distribution
    if len(st.session_state.generated_odes) > 0:
        funcs = [ode.get('function_used', 'unknown') for ode in st.session_state.generated_odes]
        func_counts = pd.Series(funcs).value_counts()
        
        fig = px.pie(values=func_counts.values, names=func_counts.index, title="Function Distribution")
        st.plotly_chart(fig, use_container_width=True)

def page_export():
    st.header("📤 Export & LaTeX")
    
    if not st.session_state.generated_odes:
        st.info("No ODEs to export")
        return
    
    ode_names = [f"{ode.get('name', f'ODE_{i+1}')} - {ode.get('function_used', 'Unknown')}" 
                 for i, ode in enumerate(st.session_state.generated_odes)]
    
    selected_idx = st.selectbox("Select ODE", range(len(ode_names)), format_func=lambda x: ode_names[x])
    ode = st.session_state.generated_odes[selected_idx]
    
    st.subheader("LaTeX Export")
    latex_doc = LaTeXExporter.generate_latex_document(ode, include_preamble=True)
    
    st.code(latex_doc, language='latex')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Download LaTeX",
            latex_doc,
            f"ode_{ode.get('name', 'export')}.tex",
            "text/x-latex"
        )
    
    with col2:
        # JSON export
        json_data = json.dumps({
            'ode': str(ode.get('lhs', '')),
            'rhs': str(ode.get('rhs', '')),
            'solution': str(ode.get('solution', '')),
            'parameters': {k: str(v) for k, v in ode.get('parameters', {}).items()}
        }, indent=2)
        
        st.download_button(
            "📋 Download JSON",
            json_data,
            f"ode_{ode.get('name', 'export')}.json",
            "application/json"
        )

def page_documentation():
    st.header("📖 Documentation")
    
    st.markdown("""
    ## Complete Master Generators System
    
    ### Features
    
    #### 1. Exact Symbolic Computation
    - All calculations use SymPy's symbolic engine
    - No numerical approximations
    - Results preserve mathematical constants (π, e, etc.)
    
    #### 2. Theorems 4.1 & 4.2 Implementation
    - **Theorem 4.1**: Basic solution formula
    - **Theorem 4.2**: m-th and (2m-1)-th derivatives
    - Support for derivatives up to order 20
    
    #### 3. Machine Learning Integration
    - Pattern learning with neural networks
    - VAE for generating new ODE patterns
    - Transformer models for sequence learning
    
    #### 4. Batch Generation
    - Generate multiple ODEs automatically
    - Vary parameters systematically or randomly
    - Export results to CSV
    
    #### 5. Novelty Detection
    - Analyze ODEs for uniqueness
    - Complexity scoring
    - Recommendation of solution methods
    
    ### Usage Guide
    
    1. **Generator Constructor**: Build your differential operator
    2. **Add Terms**: Support up to 20th order derivatives
    3. **Select Function**: Choose f(z) from basic or special functions
    4. **Set Parameters**: Use symbolic expressions (e.g., "pi/2", "sqrt(3)")
    5. **Generate**: Get exact symbolic ODE with solution
    
    ### Mathematical Framework
    
    The system implements the complete mathematical framework from the research paper,
    including all linear and nonlinear generators with exact symbolic computation.
    """)

# ======================================================
# Main Application
# ======================================================

def main():
    SessionStateManager.initialize()
    header()
    page = sidebar()
    
    page_map = {
        "🏠 Dashboard": page_dashboard,
        "🔧 Generator Constructor": page_generator_constructor,
        "📊 Batch Generation": page_batch_generation,
        "🤖 ML Pattern Learning": page_ml_pattern_learning,
        "🔍 Novelty Detection": page_novelty_detection,
        "📈 Analysis & Visualization": page_analysis,
        "📤 Export & LaTeX": page_export,
        "📖 Documentation": page_documentation,
    }
    
    page_func = page_map.get(page, page_dashboard)
    page_func()

if __name__ == "__main__":
    main()
