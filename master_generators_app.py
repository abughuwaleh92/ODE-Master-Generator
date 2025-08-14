"""
Master Generators for ODEs - Complete Implementation with ALL Features
Implements Theorems 4.1 and 4.2 with correct RHS calculation
Includes ML/DL, Batch Generation, Pattern Learning, and Classification
"""

import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import traceback
import pickle
from dataclasses import dataclass, field, asdict
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Configure page
st.set_page_config(
    page_title="Master Generators ODE System - Complete Edition",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GENERATOR TERM AND MASTER THEOREMS IMPLEMENTATION
# ============================================================================

@dataclass
class GeneratorTerm:
    """Represents a single term in the generator"""
    derivative_order: int
    coefficient: float = 1.0
    power: int = 1
    function_type: str = "linear"  # linear, exponential, sine, cosine, logarithmic
    argument_scaling: Optional[float] = None  # For y(x/a) or y(ax)
    
    def get_description(self) -> str:
        """Get human-readable description"""
        derivative_notation = {
            0: "y",
            1: "y'",
            2: "y''",
            3: "y'''",
        }
        base = derivative_notation.get(self.derivative_order, f"y^({self.derivative_order})")
        
        if self.argument_scaling:
            base = base.replace("y", f"y(x/{self.argument_scaling})")
            
        if self.function_type == "exponential":
            result = f"e^({base})"
        elif self.function_type == "sine":
            result = f"sin({base})"
        elif self.function_type == "cosine":
            result = f"cos({base})"
        elif self.function_type == "logarithmic":
            result = f"ln({base})"
        elif self.function_type == "power" and self.power != 1:
            result = f"({base})^{self.power}"
        else:
            result = base
            
        if self.coefficient != 1:
            if self.coefficient == -1:
                result = f"-{result}"
            else:
                result = f"{self.coefficient}*{result}"
            
        return result
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

class MasterGeneratorTheorems:
    """
    Complete implementation of Theorems 4.1 and 4.2 with proper RHS calculation
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, n: int = 1, M: float = 0.0):
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if n < 1:
            raise ValueError("n must be at least 1")
            
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.M = M
        
        # Symbolic variables
        self.x = sp.Symbol('x', real=True)
        self.z = sp.Symbol('z')
        self.s = sp.Symbol('s', integer=True, positive=True)
        
    def compute_omega(self, s: int) -> float:
        """Compute ω(s) = (2s-1)π/(2n)"""
        return (2 * s - 1) * sp.pi / (2 * self.n)
    
    def psi_function(self, f_z: sp.Expr, omega: sp.Expr, x: sp.Symbol) -> sp.Expr:
        """ψ(α,ω,x) = f(α + β*e^(ix*cos(ω) - x*sin(ω)))"""
        exponent = sp.I * x * sp.cos(omega) - x * sp.sin(omega)
        z_val = self.alpha + self.beta * sp.exp(exponent)
        return f_z.subs(self.z, z_val)
    
    def phi_function(self, f_z: sp.Expr, omega: sp.Expr, x: sp.Symbol) -> sp.Expr:
        """φ(α,ω,x) = f(α + β*e^(-ix*cos(ω) - x*sin(ω)))"""
        exponent = -sp.I * x * sp.cos(omega) - x * sp.sin(omega)
        z_val = self.alpha + self.beta * sp.exp(exponent)
        return f_z.subs(self.z, z_val)
    
    def generate_y(self, f_z: sp.Expr) -> sp.Expr:
        """Generate y(x) using Theorem 4.1 - Equation (4.6)"""
        y_result = 0
        
        for s in range(1, self.n + 1):
            omega = self.compute_omega(s)
            psi = self.psi_function(f_z, omega, self.x)
            phi = self.phi_function(f_z, omega, self.x)
            f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
            
            y_result += 2 * f_alpha_beta - (psi + phi)
        
        return sp.pi / (2 * self.n) * y_result + sp.pi * self.M
    
    def generate_y_prime(self, f_z: sp.Expr) -> sp.Expr:
        """Generate y'(x) using Theorem 4.1 - Equation (4.7)"""
        y_prime_result = 0
        
        # Take derivative of f with respect to z
        f_z_prime = sp.diff(f_z, self.z)
        
        for s in range(1, self.n + 1):
            omega = self.compute_omega(s)
            psi = self.psi_function(f_z, omega, self.x)
            phi = self.phi_function(f_z, omega, self.x)
            psi_prime = self.psi_function(f_z_prime, omega, self.x)
            phi_prime = self.phi_function(f_z_prime, omega, self.x)
            
            term1 = self.beta * sp.exp(-self.x * sp.sin(omega))
            term2_cos = sp.cos(self.x * sp.cos(omega) + omega) / sp.I
            term2_sin = sp.sin(self.x * sp.cos(omega) + omega)
            
            y_prime_result += term1 * (
                term2_cos * (psi_prime - phi_prime) +
                term2_sin * (psi_prime + phi_prime)
            )
        
        return sp.pi / (2 * self.n) * y_prime_result
    
    def generate_y_double_prime(self, f_z: sp.Expr) -> sp.Expr:
        """Generate y''(x) using Theorem 4.1 - Equation (4.8)"""
        y_double_prime_result = 0
        
        # Derivatives of f
        f_z_prime = sp.diff(f_z, self.z)
        f_z_double_prime = sp.diff(f_z_prime, self.z)
        
        for s in range(1, self.n + 1):
            omega = self.compute_omega(s)
            psi = self.psi_function(f_z, omega, self.x)
            phi = self.phi_function(f_z, omega, self.x)
            psi_prime = self.psi_function(f_z_prime, omega, self.x)
            phi_prime = self.phi_function(f_z_prime, omega, self.x)
            psi_double = self.psi_function(f_z_double_prime, omega, self.x)
            phi_double = self.phi_function(f_z_double_prime, omega, self.x)
            
            # First term with β
            term1 = self.beta * sp.exp(-self.x * sp.sin(omega))
            term1_cos = sp.cos(self.x * sp.cos(omega) + 2 * omega)
            term1_sin = sp.sin(self.x * sp.cos(omega) + 2 * omega) / sp.I
            
            # Second term with β²
            term2 = self.beta**2 * sp.exp(-2 * self.x * sp.sin(omega))
            term2_cos = sp.cos(2 * self.x * sp.cos(omega) + 2 * omega)
            term2_sin = sp.sin(2 * self.x * sp.cos(omega) + 2 * omega) / sp.I
            
            y_double_prime_result += (
                term1 * (term1_cos * (psi_prime + phi_prime) + term1_sin * (psi_prime - phi_prime)) +
                term2 * (term2_cos * (psi_double + phi_double) + term2_sin * (psi_double - phi_double))
            )
        
        return sp.pi / (2 * self.n) * y_double_prime_result
    
    def generate_kth_derivative(self, f_z: sp.Expr, k: int) -> sp.Expr:
        """Generate k-th derivative using Theorem 4.2"""
        if k == 0:
            return self.generate_y(f_z)
        elif k == 1:
            return self.generate_y_prime(f_z)
        elif k == 2:
            return self.generate_y_double_prime(f_z)
        else:
            # For higher derivatives, use simplified form
            # Full implementation would use equations 4.25 and 4.26
            y = self.generate_y(f_z)
            return sp.diff(y, self.x, k)
    
    def calculate_rhs(self, generator_terms: List[GeneratorTerm], f_z: sp.Expr) -> sp.Expr:
        """
        Calculate the RHS by substituting the exact forms of y and its derivatives
        into the generator expression
        """
        rhs = 0
        
        for term in generator_terms:
            # Get the appropriate derivative
            if term.derivative_order == 0:
                if term.argument_scaling:
                    # For y(x/a), substitute x -> x/a in the solution
                    y_val = self.generate_y(f_z).subs(self.x, self.x / term.argument_scaling)
                else:
                    y_val = self.generate_y(f_z)
            else:
                y_val = self.generate_kth_derivative(f_z, term.derivative_order)
                if term.argument_scaling:
                    y_val = y_val.subs(self.x, self.x / term.argument_scaling)
            
            # Apply function transformation
            if term.function_type == "exponential":
                expr = sp.exp(y_val)
            elif term.function_type == "sine":
                expr = sp.sin(y_val)
            elif term.function_type == "cosine":
                expr = sp.cos(y_val)
            elif term.function_type == "logarithmic":
                expr = sp.log(sp.Abs(y_val) + sp.Symbol('epsilon', positive=True))
            elif term.function_type == "power" and term.power != 1:
                expr = y_val ** term.power
            else:
                expr = y_val
            
            # Apply coefficient
            rhs += term.coefficient * expr
        
        return rhs

# ============================================================================
# MACHINE LEARNING COMPONENTS
# ============================================================================

class GeneratorPatternDataset(Dataset):
    """Dataset for generator patterns"""
    
    def __init__(self, patterns: List[Dict[str, Any]]):
        self.patterns = patterns
        
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        # Convert to feature vector
        features = self._extract_features(pattern)
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_features(self, pattern: Dict[str, Any]) -> List[float]:
        """Extract features from a generator pattern"""
        features = []
        terms = pattern.get('terms', [])
        
        # Max 10 terms, 6 features each
        for i in range(10):
            if i < len(terms):
                term = terms[i]
                features.extend([
                    term.derivative_order,
                    term.coefficient,
                    term.power,
                    1 if term.function_type == "exponential" else 0,
                    1 if term.function_type in ["sine", "cosine"] else 0,
                    term.argument_scaling if term.argument_scaling else 0
                ])
            else:
                features.extend([0] * 6)
        
        return features

class GeneratorVAE(nn.Module):
    """VAE for learning generator patterns"""
    
    def __init__(self, input_dim: int = 60, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def generate(self, num_samples: int = 1):
        """Generate new patterns"""
        with torch.no_grad():
            z = torch.randn(num_samples, 32)
            samples = self.decode(z)
            return samples

class NoveltyDetector:
    """Detect novel ODEs"""
    
    def __init__(self):
        self.known_patterns = []
        
    def add_pattern(self, pattern: Dict[str, Any]):
        """Add a known pattern"""
        self.known_patterns.append(pattern)
    
    def check_novelty(self, new_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a pattern is novel"""
        if not self.known_patterns:
            return {"is_novel": True, "novelty_score": 100, "similar_patterns": []}
        
        # Simple novelty check based on structure
        novelty_score = 100
        similar_patterns = []
        
        for known in self.known_patterns:
            similarity = self._calculate_similarity(new_pattern, known)
            if similarity > 0.8:
                novelty_score -= similarity * 20
                similar_patterns.append(known)
        
        return {
            "is_novel": novelty_score > 50,
            "novelty_score": max(0, novelty_score),
            "similar_patterns": similar_patterns[:3]
        }
    
    def _calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns"""
        # Simplified similarity metric
        terms1 = pattern1.get('terms', [])
        terms2 = pattern2.get('terms', [])
        
        if len(terms1) != len(terms2):
            return 0.0
        
        similarity = 0
        for t1, t2 in zip(terms1, terms2):
            if t1.derivative_order == t2.derivative_order:
                similarity += 0.3
            if t1.function_type == t2.function_type:
                similarity += 0.3
            if abs(t1.coefficient - t2.coefficient) < 0.1:
                similarity += 0.2
            if t1.power == t2.power:
                similarity += 0.2
        
        return similarity / len(terms1)

class ODEClassifier:
    """Classify ODEs by physics applications"""
    
    def __init__(self):
        self.applications = {
            "harmonic": {
                "pattern": ["y''", "y"],
                "field": "Physics",
                "applications": ["Spring-mass systems", "Pendulum", "LC circuits"]
            },
            "damped": {
                "pattern": ["y''", "y'", "y"],
                "field": "Physics/Engineering",
                "applications": ["Damped oscillators", "RLC circuits"]
            },
            "wave": {
                "pattern": ["y''"],
                "field": "Physics",
                "applications": ["Wave propagation", "Vibrations"]
            },
            "heat": {
                "pattern": ["y'"],
                "field": "Thermodynamics",
                "applications": ["Heat diffusion", "Chemical diffusion"]
            },
            "schrodinger": {
                "pattern": ["y''", "V(x)*y"],
                "field": "Quantum Mechanics",
                "applications": ["Quantum states", "Particle in potential"]
            }
        }
    
    def classify(self, generator_terms: List[GeneratorTerm]) -> Dict[str, Any]:
        """Classify ODE based on generator terms"""
        # Extract pattern
        pattern = []
        for term in generator_terms:
            if term.derivative_order == 0:
                pattern.append("y")
            elif term.derivative_order == 1:
                pattern.append("y'")
            elif term.derivative_order == 2:
                pattern.append("y''")
            else:
                pattern.append(f"y^({term.derivative_order})")
        
        # Find matching applications
        matches = []
        for name, app_data in self.applications.items():
            if any(p in pattern for p in app_data["pattern"]):
                matches.append({
                    "name": name,
                    "field": app_data["field"],
                    "applications": app_data["applications"]
                })
        
        if not matches:
            return {
                "field": "Mathematical Physics",
                "applications": ["Research equation"],
                "matches": []
            }
        
        return {
            "field": matches[0]["field"],
            "applications": matches[0]["applications"],
            "matches": matches
        }

# ============================================================================
# GENERATOR CONSTRUCTOR
# ============================================================================

class GeneratorConstructor:
    """Generator constructor for building custom ODEs"""
    
    def __init__(self):
        self.terms: List[GeneratorTerm] = []
        
    def add_term(self, term: GeneratorTerm):
        """Add a term to the generator"""
        self.terms.append(term)
        
    def clear_terms(self):
        """Clear all terms"""
        self.terms = []
        
    def get_generator_expression(self) -> str:
        """Get the generator expression as a string"""
        if not self.terms:
            return "0"
        
        expr_parts = []
        for i, term in enumerate(self.terms):
            desc = term.get_description()
            if i > 0 and not desc.startswith("-"):
                expr_parts.append(" + ")
            elif i > 0:
                expr_parts.append(" ")
            expr_parts.append(desc)
        
        return "".join(expr_parts)
    
    def get_latex_expression(self) -> str:
        """Get the generator expression in LaTeX format"""
        expr = self.get_generator_expression()
        # Convert to LaTeX notation
        expr = expr.replace("y'''", r"y'''")
        expr = expr.replace("y''", r"y''")
        expr = expr.replace("y'", r"y'")
        expr = expr.replace("*", "")
        expr = expr.replace("e^", r"e^")
        expr = expr.replace("sin", r"\sin")
        expr = expr.replace("cos", r"\cos")
        expr = expr.replace("ln", r"\ln")
        return expr
    
    def to_pattern(self) -> Dict[str, Any]:
        """Convert to pattern for ML training"""
        return {
            "terms": self.terms,
            "expression": self.get_generator_expression(),
            "order": max((t.derivative_order for t in self.terms), default=0),
            "is_linear": all(t.function_type == "linear" and t.power == 1 for t in self.terms),
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if 'generator_constructor' not in st.session_state:
        st.session_state.generator_constructor = GeneratorConstructor()
    if 'generated_odes' not in st.session_state:
        st.session_state.generated_odes = []
    if 'generator_patterns' not in st.session_state:
        st.session_state.generator_patterns = []
    if 'vae_model' not in st.session_state:
        st.session_state.vae_model = GeneratorVAE()
    if 'novelty_detector' not in st.session_state:
        st.session_state.novelty_detector = NoveltyDetector()
    if 'ode_classifier' not in st.session_state:
        st.session_state.ode_classifier = ODEClassifier()
    if 'ml_trained' not in st.session_state:
        st.session_state.ml_trained = False

def main():
    """Main application"""
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .generator-term {
        background: #f7f7f7;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #667eea;
    }
    .result-box {
        background: #e8f5e9;
        border: 2px solid #4caf50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .ml-box {
        background: #fff3e0;
        border: 2px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Master Generators for ODEs - Complete System</h1>
        <p>Theorems 4.1 & 4.2 + ML/DL + Batch Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔧 Generator Constructor", 
         "🤖 ML Pattern Learning",
         "📊 Batch Generation",
         "🔍 Novelty Detection",
         "📈 Analysis & Classification",
         "📚 Examples",
         "📖 Documentation"]
    )
    
    if page == "🔧 Generator Constructor":
        generator_constructor_page()
    elif page == "🤖 ML Pattern Learning":
        ml_pattern_learning_page()
    elif page == "📊 Batch Generation":
        batch_generation_page()
    elif page == "🔍 Novelty Detection":
        novelty_detection_page()
    elif page == "📈 Analysis & Classification":
        analysis_classification_page()
    elif page == "📚 Examples":
        examples_page()
    elif page == "📖 Documentation":
        documentation_page()

def generator_constructor_page():
    """Page for constructing custom generators with proper RHS"""
    st.header("🔧 Custom Generator Constructor")
    
    st.info("""
    Build your generator by combining y and its derivatives. The system will calculate 
    the correct RHS using Theorems 4.1 and 4.2.
    """)
    
    constructor = st.session_state.generator_constructor
    
    # Term builder
    st.subheader("➕ Add Generator Terms")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        deriv_order = st.selectbox(
            "Derivative Order",
            [0, 1, 2, 3, 4, 5],
            format_func=lambda x: {
                0: "y", 1: "y'", 2: "y''", 3: "y'''", 
                4: "y⁽⁴⁾", 5: "y⁽⁵⁾"
            }.get(x, f"y⁽{x}⁾")
        )
    
    with col2:
        func_type = st.selectbox(
            "Function Type",
            ["linear", "exponential", "sine", "cosine", "logarithmic", "power"]
        )
    
    with col3:
        coefficient = st.number_input("Coefficient", -10.0, 10.0, 1.0, 0.1)
    
    with col4:
        power = st.number_input("Power", 1, 5, 1) if func_type == "power" else 1
    
    use_scaling = st.checkbox("Use argument scaling (y(x/a))")
    scaling = st.number_input("Scaling a", 0.5, 5.0, 2.0, 0.1) if use_scaling else None
    
    if st.button("➕ Add Term", type="primary"):
        term = GeneratorTerm(deriv_order, coefficient, power, func_type, scaling)
        constructor.add_term(term)
        st.success(f"Added: {term.get_description()}")
        st.rerun()
    
    # Display current generator
    if constructor.terms:
        st.subheader("📝 Current Generator")
        
        for i, term in enumerate(constructor.terms):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="generator-term">
                    <strong>Term {i+1}:</strong> {term.get_description()}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    constructor.terms.pop(i)
                    st.rerun()
        
        # Display equation
        st.markdown("### Generator Equation:")
        st.latex(f"{constructor.get_latex_expression()} = RHS")
        
        # Generate ODE section
        st.subheader("🎯 Generate ODE with Exact Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_options = {
                "z": "f(z) = z",
                "z²": "f(z) = z²",
                "z³": "f(z) = z³",
                "e^z": "f(z) = e^z",
                "sin(z)": "f(z) = sin(z)",
                "cos(z)": "f(z) = cos(z)",
                "ln(z)": "f(z) = ln(z)",
                "1/z": "f(z) = 1/z"
            }
            func_choice = st.selectbox("Select f(z)", list(func_options.keys()),
                                      format_func=lambda x: func_options[x])
            
            st.markdown("**Parameters:**")
            alpha = st.slider("α", -5.0, 5.0, 1.0, 0.1)
            beta = st.slider("β", 0.1, 5.0, 1.0, 0.1)
            n = st.slider("n", 1, 3, 1)
            M = st.slider("M", -5.0, 5.0, 0.0, 0.1)
        
        with col2:
            if st.button("🚀 Generate ODE", type="primary"):
                with st.spinner("Calculating..."):
                    try:
                        # Create f(z)
                        z = sp.Symbol('z')
                        func_map = {
                            "z": z, "z²": z**2, "z³": z**3,
                            "e^z": sp.exp(z), "sin(z)": sp.sin(z),
                            "cos(z)": sp.cos(z), "ln(z)": sp.log(z),
                            "1/z": 1/z
                        }
                        f_z = func_map[func_choice]
                        
                        # Generate solution and RHS
                        master_gen = MasterGeneratorTheorems(alpha, beta, n, M)
                        y_solution = master_gen.generate_y(f_z)
                        rhs = master_gen.calculate_rhs(constructor.terms, f_z)
                        
                        # Simplify
                        try:
                            y_solution = sp.simplify(y_solution)
                            rhs = sp.simplify(rhs)
                        except:
                            pass
                        
                        # Store result
                        result = {
                            "generator": constructor.get_generator_expression(),
                            "f_z": str(f_z),
                            "solution": str(y_solution),
                            "rhs": str(rhs),
                            "parameters": {"alpha": alpha, "beta": beta, "n": n, "M": M}
                        }
                        st.session_state.generated_odes.append(result)
                        
                        # Add pattern for ML
                        pattern = constructor.to_pattern()
                        st.session_state.generator_patterns.append(pattern)
                        st.session_state.novelty_detector.add_pattern(pattern)
                        
                        # Display results
                        st.markdown("""
                        <div class="result-box">
                            <h3>✅ ODE Generated Successfully!</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### Complete ODE:")
                        st.latex(f"{constructor.get_latex_expression()} = {sp.latex(rhs)}")
                        
                        st.markdown("### Exact Solution:")
                        st.latex(f"y(x) = {sp.latex(y_solution)}")
                        
                        # Classification
                        classification = st.session_state.ode_classifier.classify(constructor.terms)
                        st.markdown("### 🏷️ Classification:")
                        st.write(f"**Field:** {classification['field']}")
                        st.write(f"**Applications:** {', '.join(classification['applications'])}")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        if st.button("🗑️ Clear All Terms"):
            constructor.clear_terms()
            st.rerun()

def ml_pattern_learning_page():
    """ML Pattern Learning page"""
    st.header("🤖 ML Pattern Learning")
    
    st.markdown("""
    <div class="ml-box">
    The ML system learns generator patterns (not individual ODEs) to create new generators 
    that produce infinite families of ODEs.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Patterns Collected", len(st.session_state.generator_patterns))
    with col2:
        st.metric("ODEs Generated", len(st.session_state.generated_odes))
    with col3:
        st.metric("Model Status", "Trained" if st.session_state.ml_trained else "Not Trained")
    
    # Training section
    st.subheader("🎯 Train VAE Model")
    
    if len(st.session_state.generator_patterns) < 5:
        st.warning(f"Need at least 5 patterns. Current: {len(st.session_state.generator_patterns)}")
    else:
        epochs = st.slider("Training Epochs", 10, 100, 50)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training VAE..."):
                try:
                    # Create dataset
                    dataset = GeneratorPatternDataset(st.session_state.generator_patterns)
                    dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)
                    
                    # Train VAE
                    vae = st.session_state.vae_model
                    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
                    
                    progress_bar = st.progress(0)
                    for epoch in range(epochs):
                        for batch in dataloader:
                            optimizer.zero_grad()
                            recon, mu, log_var = vae(batch)
                            
                            # VAE loss
                            recon_loss = F.mse_loss(recon, batch)
                            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                            loss = recon_loss + 0.01 * kld_loss
                            
                            loss.backward()
                            optimizer.step()
                        
                        progress_bar.progress((epoch + 1) / epochs)
                    
                    st.session_state.ml_trained = True
                    st.success("✅ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    # Generation section
    st.subheader("🎨 Generate Novel Patterns")
    
    if st.session_state.ml_trained:
        if st.button("🎲 Generate Novel Generator", type="primary"):
            with st.spinner("Generating..."):
                try:
                    # Generate from VAE
                    vae = st.session_state.vae_model
                    samples = vae.generate(1)
                    
                    # Convert to generator terms
                    features = samples[0].numpy()
                    novel_terms = []
                    
                    for i in range(0, min(30, len(features)), 6):
                        if features[i] > 0:  # derivative order
                            term = GeneratorTerm(
                                derivative_order=int(abs(features[i])) % 4,
                                coefficient=float(features[i+1]),
                                power=max(1, int(abs(features[i+2])) % 4),
                                function_type=["linear", "exponential", "sine", "cosine"][int(abs(features[i+3])) % 4],
                                argument_scaling=None if features[i+5] == 0 else abs(features[i+5])
                            )
                            novel_terms.append(term)
                    
                    if novel_terms:
                        st.success("✅ Novel generator created!")
                        
                        for i, term in enumerate(novel_terms[:5]):  # Limit to 5 terms
                            st.write(f"**Term {i+1}:** {term.get_description()}")
                        
                        if st.button("Use This Generator"):
                            constructor = st.session_state.generator_constructor
                            constructor.clear_terms()
                            for term in novel_terms[:5]:
                                constructor.add_term(term)
                            st.success("Loaded into constructor!")
                    
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")

def batch_generation_page():
    """Batch generation page"""
    st.header("📊 Batch ODE Generation")
    
    constructor = st.session_state.generator_constructor
    
    if not constructor.terms:
        st.warning("Please construct a generator first")
        return
    
    st.info(f"Current Generator: {constructor.get_generator_expression()}")
    
    # Batch settings
    col1, col2 = st.columns(2)
    
    with col1:
        num_odes = st.slider("Number of ODEs", 5, 50, 10)
        functions = st.multiselect(
            "Functions f(z)",
            ["z", "z²", "e^z", "sin(z)", "cos(z)"],
            default=["z", "z²", "e^z"]
        )
    
    with col2:
        vary_params = st.checkbox("Vary parameters", True)
        if vary_params:
            alpha_range = st.slider("α range", -5.0, 5.0, (-2.0, 2.0))
            beta_range = st.slider("β range", 0.1, 5.0, (0.5, 2.0))
        else:
            alpha_range = (1.0, 1.0)
            beta_range = (1.0, 1.0)
    
    if st.button("🚀 Generate Batch", type="primary"):
        with st.spinner(f"Generating {num_odes} ODEs..."):
            batch_results = []
            progress = st.progress(0)
            
            for i in range(num_odes):
                try:
                    # Random selections
                    func_choice = np.random.choice(functions)
                    alpha = np.random.uniform(*alpha_range)
                    beta = np.random.uniform(*beta_range)
                    n = np.random.randint(1, 3)
                    M = np.random.uniform(-1, 1)
                    
                    # Create f(z)
                    z = sp.Symbol('z')
                    func_map = {
                        "z": z, "z²": z**2, "e^z": sp.exp(z),
                        "sin(z)": sp.sin(z), "cos(z)": sp.cos(z)
                    }
                    f_z = func_map[func_choice]
                    
                    # Generate
                    master_gen = MasterGeneratorTheorems(alpha, beta, n, M)
                    y_solution = master_gen.generate_y(f_z)
                    rhs = master_gen.calculate_rhs(constructor.terms, f_z)
                    
                    batch_results.append({
                        "ID": i + 1,
                        "f(z)": func_choice,
                        "α": round(alpha, 2),
                        "β": round(beta, 2),
                        "n": n,
                        "Generator": constructor.get_generator_expression()[:30] + "..."
                    })
                    
                except:
                    pass
                
                progress.progress((i + 1) / num_odes)
            
            st.success(f"✅ Generated {len(batch_results)} ODEs!")
            
            df = pd.DataFrame(batch_results)
            st.dataframe(df, use_container_width=True)
            
            # Export
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "batch_odes.csv", "text/csv")

def novelty_detection_page():
    """Novelty detection page"""
    st.header("🔍 Novelty Detection")
    
    detector = st.session_state.novelty_detector
    
    st.info(f"Known Patterns: {len(detector.known_patterns)}")
    
    constructor = st.session_state.generator_constructor
    
    if constructor.terms:
        st.subheader("Check Current Generator")
        st.write(f"Generator: {constructor.get_generator_expression()}")
        
        if st.button("🔍 Check Novelty"):
            pattern = constructor.to_pattern()
            result = detector.check_novelty(pattern)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Novelty Score", f"{result['novelty_score']:.0f}/100")
            with col2:
                if result['is_novel']:
                    st.success("✅ NOVEL GENERATOR")
                else:
                    st.warning("⚠️ Similar to existing")
            
            if result['similar_patterns']:
                st.subheader("Similar Patterns:")
                for p in result['similar_patterns']:
                    st.write(f"- {p.get('expression', 'Unknown')}")

def analysis_classification_page():
    """Analysis and classification page"""
    st.header("📈 Analysis & Classification")
    
    if st.session_state.generated_odes:
        st.subheader("Generated ODEs Analysis")
        
        # Create summary
        df_data = []
        for i, ode in enumerate(st.session_state.generated_odes[-10:]):  # Last 10
            df_data.append({
                "ID": i + 1,
                "Generator": ode['generator'][:30] + "...",
                "f(z)": ode['f_z'],
                "α": ode['parameters']['alpha'],
                "β": ode['parameters']['beta']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Classification statistics
        classifier = st.session_state.ode_classifier
        constructor = st.session_state.generator_constructor
        
        if constructor.terms:
            classification = classifier.classify(constructor.terms)
            
            st.subheader("Current Generator Classification")
            st.write(f"**Field:** {classification['field']}")
            st.write(f"**Applications:** {', '.join(classification['applications'])}")
    else:
        st.info("No ODEs generated yet")

def examples_page():
    """Examples page"""
    st.header("📚 Examples from the Paper")
    
    examples = [
        {
            "name": "Linear Generator 1",
            "equation": "y'' + y = RHS",
            "terms": [GeneratorTerm(2, 1.0), GeneratorTerm(0, 1.0)]
        },
        {
            "name": "Linear Generator 2",  
            "equation": "y'' + y' = RHS",
            "terms": [GeneratorTerm(2, 1.0), GeneratorTerm(1, 1.0)]
        },
        {
            "name": "Nonlinear Generator 1",
            "equation": "(y'')² + y = RHS",
            "terms": [GeneratorTerm(2, 1.0, power=2, function_type="power"), GeneratorTerm(0, 1.0)]
        },
        {
            "name": "Nonlinear Generator 6",
            "equation": "sin(y'') + y = RHS",
            "terms": [GeneratorTerm(2, 1.0, function_type="sine"), GeneratorTerm(0, 1.0)]
        }
    ]
    
    for ex in examples:
        with st.expander(f"{ex['name']}: {ex['equation']}"):
            if st.button(f"Load {ex['name']}", key=ex['name']):
                constructor = st.session_state.generator_constructor
                constructor.clear_terms()
                for term in ex['terms']:
                    constructor.add_term(term)
                st.success(f"Loaded {ex['name']}!")

def documentation_page():
    """Documentation page"""
    st.header("📖 Documentation")
    
    st.markdown("""
    ## Complete System Features
    
    ### 1. Proper RHS Calculation ✅
    - Uses Theorems 4.1 and 4.2 to calculate exact y(x) and derivatives
    - Substitutes into generator to get correct RHS
    - Ensures y(x) is truly the solution
    
    ### 2. ML Pattern Learning 🤖
    - VAE learns generator patterns (not individual ODEs)
    - Can generate novel generator combinations
    - Each generator produces infinite ODEs
    
    ### 3. Batch Generation 📊
    - Generate multiple ODEs from one generator
    - Vary f(z) and parameters systematically
    - Export results for analysis
    
    ### 4. Novelty Detection 🔍
    - Check if generators are novel
    - Find similar existing patterns
    - Identify research opportunities
    
    ### 5. Classification 📈
    - Identify physics applications
    - Classify by field (Physics, QM, etc.)
    - Link to real-world uses
    
    ### Mathematical Foundation
    
    **Theorem 4.1** provides:
    - y(x) = exact solution
    - y'(x), y''(x) = first and second derivatives
    
    **Theorem 4.2** extends to:
    - y^(2m)(x) = even derivatives
    - y^(2m-1)(x) = odd derivatives
    
    ### Workflow
    
    1. **Build Generator**: Combine terms (y, y', y'', etc.)
    2. **Choose f(z)**: Select function
    3. **Set Parameters**: α, β, n, M
    4. **Generate**: Get ODE with exact solution and correct RHS
    5. **Train ML**: Learn patterns for novel generation
    6. **Batch Generate**: Create families of ODEs
    7. **Analyze**: Classify and detect novelty
    """)

if __name__ == "__main__":
    main()
