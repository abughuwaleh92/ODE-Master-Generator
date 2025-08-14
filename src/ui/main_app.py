"""
Main Streamlit Application for Master ODE Generator System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import sympy as sp
from typing import Dict, Any, List, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from generators.generator_constructor import (
    GeneratorConstructor, GeneratorSpecification, 
    DerivativeTerm, DerivativeType, OperatorType
)
from generators.master_theorem import MasterTheoremSolver, MasterTheoremParameters
from generators.ode_classifier import ODEClassifier
from ml.pattern_learner import PatternLearningSystem, GeneratorPattern
from functions.function_library import FunctionLibrary
from solvers.numerical_solver import NumericalSolver
from utils.visualization import Visualizer

# Configure Streamlit
st.set_page_config(
    page_title="Master ODE Generator System",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e5266;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1e3d59;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #228b22;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffacd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffd700;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class MasterODEApp:
    """Main application class"""
    
    def __init__(self):
        self.initialize_session_state()
        self.constructor = GeneratorConstructor()
        self.solver = MasterTheoremSolver()
        self.classifier = ODEClassifier()
        self.ml_system = self.initialize_ml_system()
        self.visualizer = Visualizer()
        self.function_library = FunctionLibrary()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'generator_terms' not in st.session_state:
            st.session_state.generator_terms = []
        if 'generated_odes' not in st.session_state:
            st.session_state.generated_odes = []
        if 'ml_patterns' not in st.session_state:
            st.session_state.ml_patterns = []
        if 'current_generator' not in st.session_state:
            st.session_state.current_generator = None
        if 'training_history' not in st.session_state:
            st.session_state.training_history = []
    
    def initialize_ml_system(self):
        """Initialize ML system"""
        try:
            system = PatternLearningSystem(model_type='vae')
            # Load pre-trained model if available
            if os.path.exists('models/pretrained_vae.pth'):
                system.load_model('models/pretrained_vae.pth')
            return system
        except Exception as e:
            st.warning(f"ML system initialization failed: {e}")
            return None
    
    def run(self):
        """Run the main application"""
        st.markdown('<h1 class="main-header">🧮 Master ODE Generator System</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            [
                "🏠 Home",
                "🔧 Generator Constructor",
                "🎯 Apply Master Theorem",
                "🤖 ML Pattern Learning",
                "📊 Analysis & Classification",
                "🔬 Physical Applications",
                "📈 Visualization",
                "💾 Library",
                "⚙️ Settings"
            ]
        )
        
        # Route to appropriate page
        if page == "🏠 Home":
            self.home_page()
        elif page == "🔧 Generator Constructor":
            self.generator_constructor_page()
        elif page == "🎯 Apply Master Theorem":
            self.master_theorem_page()
        elif page == "🤖 ML Pattern Learning":
            self.ml_learning_page()
        elif page == "📊 Analysis & Classification":
            self.analysis_page()
        elif page == "🔬 Physical Applications":
            self.applications_page()
        elif page == "📈 Visualization":
            self.visualization_page()
        elif page == "💾 Library":
            self.library_page()
        elif page == "⚙️ Settings":
            self.settings_page()
    
    def home_page(self):
        """Home page with overview and quick stats"""
        st.markdown('<h2 class="sub-header">Welcome to the Master ODE Generator System</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📝</h3>
                <h2>{}</h2>
                <p>Generated ODEs</p>
            </div>
            """.format(len(st.session_state.generated_odes)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🧬</h3>
                <h2>{}</h2>
                <p>ML Patterns</p>
            </div>
            """.format(len(st.session_state.ml_patterns)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🔧</h3>
                <h2>{}</h2>
                <p>Active Terms</p>
            </div>
            """.format(len(st.session_state.generator_terms)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>📚</h3>
                <h2>{}</h2>
                <p>Functions</p>
            </div>
            """.format(len(self.function_library.get_all_functions())), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 System Capabilities</h3>
        
        This advanced system implements the Master Theorems for generating exact solutions to differential equations:
        
        - **Generator Construction**: Build custom differential operators by combining derivatives with various transformations
        - **Exact Solutions**: Apply Master Theorems to generate ODEs with known analytical solutions
        - **ML Pattern Learning**: Train neural networks to learn and generate novel ODE patterns
        - **Physical Applications**: Identify real-world applications and classify ODEs
        - **Advanced Analysis**: Stability, symmetry, and conservation law analysis
        
        <br>
        <b>Quick Start:</b> Navigate to the Generator Constructor to begin creating your first custom ODE!
        </div>
        """, unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("### 📊 Recent Activity")
        
        if st.session_state.generated_odes:
            recent_odes = st.session_state.generated_odes[-5:]
            for i, ode in enumerate(reversed(recent_odes)):
                with st.expander(f"ODE {len(st.session_state.generated_odes) - i}: {ode.get('name', 'Unnamed')}"):
                    st.latex(ode.get('latex', ''))
                    st.write(f"**Order:** {ode.get('order', 'N/A')}")
                    st.write(f"**Type:** {ode.get('type', 'N/A')}")
        else:
            st.info("No ODEs generated yet. Start by constructing a generator!")
    
    def generator_constructor_page(self):
        """Page for constructing custom generators"""
        st.markdown('<h2 class="sub-header">🔧 Generator Constructor</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Build custom differential operators by combining derivative terms with various transformations.
        Create equations like: <b>y + sin(y') + e^(y'') = RHS</b>
        </div>
        """, unsafe_allow_html=True)
        
        # Term builder
        with st.container():
            st.markdown("### Add New Term")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                derivative_order = st.selectbox(
                    "Derivative Order",
                    options=list(range(11)),
                    format_func=lambda x: {
                        0: "y (no derivative)",
                        1: "y' (first)",
                        2: "y'' (second)",
                        3: "y''' (third)"
                    }.get(x, f"y^({x})")
                )
            
            with col2:
                function_type = st.selectbox(
                    "Function Type",
                    options=[t.value for t in DerivativeType],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            with col3:
                operator_type = st.selectbox(
                    "Operator Type",
                    options=[t.value for t in OperatorType],
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            with col4:
                coefficient = st.number_input(
                    "Coefficient",
                    min_value=-100.0,
                    max_value=100.0,
                    value=1.0,
                    step=0.1
                )
            
            # Additional parameters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                power = st.number_input("Power", min_value=1, max_value=10, value=1)
            
            with col2:
                if operator_type in ['delay', 'advance']:
                    scaling = st.number_input("Scaling", min_value=0.1, max_value=10.0, value=2.0)
                else:
                    scaling = None
            
            with col3:
                if operator_type in ['delay', 'advance']:
                    shift = st.number_input("Shift", min_value=-10.0, max_value=10.0, value=0.0)
                else:
                    shift = None
            
            with col4:
                additional_params = {}
                if function_type == 'trigonometric':
                    trig_func = st.selectbox(
                        "Trig Function",
                        ['sin', 'cos', 'tan', 'sec', 'csc', 'cot']
                    )
                    additional_params['trig_func'] = trig_func
                elif function_type == 'hyperbolic':
                    hyp_func = st.selectbox(
                        "Hyperbolic Function",
                        ['sinh', 'cosh', 'tanh', 'sech', 'csch', 'coth']
                    )
                    additional_params['hyp_func'] = hyp_func
            
            # Add term button
            if st.button("➕ Add Term", type="primary", use_container_width=True):
                term = DerivativeTerm(
                    derivative_order=derivative_order,
                    coefficient=coefficient,
                    power=power,
                    function_type=DerivativeType(function_type),
                    operator_type=OperatorType(operator_type),
                    scaling=scaling if scaling else None,
                    shift=shift if shift and shift != 0 else None,
                    additional_params=additional_params if additional_params else {}
                )
                st.session_state.generator_terms.append(term)
                st.success(f"Added term: {term.get_description()}")
                st.rerun()
        
        # Display current terms
        if st.session_state.generator_terms:
            st.markdown("### Current Generator Terms")
            
            for i, term in enumerate(st.session_state.generator_terms):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**Term {i+1}:** {term.get_description()}")
                
                with col2:
                    if st.button(f"🔧 Edit", key=f"edit_{i}"):
                        st.info("Edit functionality in development")
                
                with col3:
                    if st.button(f"❌ Remove", key=f"remove_{i}"):
                        st.session_state.generator_terms.pop(i)
                        st.rerun()
            
            # Construct generator
            st.markdown("### Construct Generator")
            
            col1, col2 = st.columns(2)
            
            with col1:
                gen_name = st.text_input("Generator Name (optional)")
            
            with col2:
                gen_desc = st.text_input("Description (optional)")
            
            if st.button("🔨 Construct Generator", type="primary", use_container_width=True):
                try:
                    generator_spec = GeneratorSpecification(
                        terms=st.session_state.generator_terms,
                        name=gen_name if gen_name else None,
                        description=gen_desc if gen_desc else None
                    )
                    
                    st.session_state.current_generator = generator_spec
                    
                    st.markdown("""
                    <div class="success-box">
                    <h3>✅ Generator Constructed Successfully!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display generator info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Properties")
                        st.write(f"**Name:** {generator_spec.name}")
                        st.write(f"**Order:** {generator_spec.order}")
                        st.write(f"**Type:** {'Linear' if generator_spec.is_linear else 'Nonlinear'}")
                        st.write(f"**Special Features:** {', '.join(generator_spec.special_features) if generator_spec.special_features else 'None'}")
                    
                    with col2:
                        st.markdown("#### Equation")
                        st.latex(f"{sp.latex(generator_spec.lhs)} = f(x)")
                    
                    # Save to session
                    st.session_state.generated_odes.append({
                        'name': generator_spec.name,
                        'latex': sp.latex(generator_spec.lhs),
                        'order': generator_spec.order,
                        'type': 'Linear' if generator_spec.is_linear else 'Nonlinear',
                        'spec': generator_spec,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error constructing generator: {str(e)}")
        
        # Clear button
        if st.button("🗑️ Clear All Terms"):
            st.session_state.generator_terms = []
            st.rerun()
    
    def master_theorem_page(self):
        """Page for applying Master Theorem"""
        st.markdown('<h2 class="sub-header">🎯 Apply Master Theorem</h2>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.current_generator:
            st.warning("Please construct a generator first!")
            return
        
        generator_spec = st.session_state.current_generator
        
        st.markdown(f"### Current Generator: {generator_spec.name}")
        st.latex(f"{sp.latex(generator_spec.lhs)} = RHS")
        
        # Function selection
        st.markdown("### Select Function f(z)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            func_category = st.selectbox(
                "Function Category",
                self.function_library.get_categories()
            )
        
        with col2:
            func_name = st.selectbox(
                "Function",
                self.function_library.get_functions_in_category(func_category)
            )
        
        with col3:
            st.markdown("#### Function Preview")
            func_expr = self.function_library.get_function(func_name)
            st.latex(f"f(z) = {sp.latex(func_expr)}")
        
        # Parameters
        st.markdown("### Master Theorem Parameters")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            alpha = st.number_input("α", -10.0, 10.0, 1.0, 0.1)
        
        with col2:
            beta = st.number_input("β", 0.1, 10.0, 1.0, 0.1)
        
        with col3:
            n = st.number_input("n", 1, 10, 1)
        
        with col4:
            M = st.number_input("M", -10.0, 10.0, 0.0, 0.1)
        
        with col5:
            precision = st.number_input("Precision", 10, 30, 15)
        
        # Apply theorem
        if st.button("🚀 Generate ODE with Exact Solution", type="primary", use_container_width=True):
            try:
                with st.spinner("Applying Master Theorem..."):
                    # Create parameters
                    params = MasterTheoremParameters(
                        f_z=func_expr,
                        alpha=alpha,
                        beta=beta,
                        n=n,
                        M=M,
                        precision=precision
                    )
                    
                    # Apply theorem
                    result = self.solver.apply_theorem_4_1(generator_spec, params)
                    
                    # Classify ODE
                    classification = self.classifier.classify(generator_spec)
                    
                    st.markdown("""
                    <div class="success-box">
                    <h3>✅ ODE Generated with Exact Solution!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results in tabs
                    tabs = st.tabs([
                        "📐 Equation",
                        "💡 Solution", 
                        "🏷️ Classification",
                        "🔬 Applications",
                        "📊 Verification"
                    ])
                    
                    with tabs[0]:
                        st.markdown("#### Complete ODE")
                        st.latex(sp.latex(result['ode']['equation']))
                        
                        st.markdown("#### Right-Hand Side")
                        st.latex(f"RHS = {sp.latex(result['ode']['rhs'])}")
                    
                    with tabs[1]:
                        st.markdown("#### Exact Solution")
                        st.latex(f"y(x) = {sp.latex(result['solution'])}")
                        
                        st.markdown("#### Initial Conditions")
                        for ic, value in result['initial_conditions'].items():
                            st.write(f"• **{ic}** = {value}")
                        
                        st.markdown("#### Parameters Used")
                        params_df = pd.DataFrame({
                            'Parameter': ['α', 'β', 'n', 'M'],
                            'Value': [alpha, beta, n, M]
                        })
                        st.dataframe(params_df)
                    
                    with tabs[2]:
                        st.markdown("#### Mathematical Classification")
                        
                        math_class = classification['mathematical_type']
                        class_df = pd.DataFrame({
                            'Property': list(math_class.keys()),
                            'Value': list(math_class.values())
                        })
                        st.dataframe(class_df)
                        
                        st.markdown("#### Structural Properties")
                        struct_props = classification['structural_properties']
                        st.json(struct_props)
                    
                    with tabs[3]:
                        st.markdown("#### Physical Applications")
                        
                        apps = classification['physical_applications']
                        if apps:
                            for app in apps:
                                with st.expander(f"📚 {app.name} ({app.field})"):
                                    st.write(f"**Description:** {app.description}")
                                    
                                    st.write("**Parameter Meanings:**")
                                    for param, meaning in app.parameters_meaning.items():
                                        st.write(f"• {param}: {meaning}")
                                    
                                    if app.typical_values:
                                        st.write("**Typical Values:**")
                                        values_df = pd.DataFrame({
                                            'Parameter': list(app.typical_values.keys()),
                                            'Value': list(# Complete Master ODE Generator System
## Advanced Implementation with Pattern Learning and Classification

## Project Structure
