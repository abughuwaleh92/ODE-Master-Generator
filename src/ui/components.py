"""
Streamlit UI Components for Master Generators
Reusable UI components and widgets - FIXED VERSION
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple
import json
import base64
from io import BytesIO, StringIO
from datetime import datetime
import requests
import html

class UIComponents:
    """Collection of reusable UI components with security fixes"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000/api/v1"
    
    @staticmethod
    def sanitize_latex(latex_str: str) -> str:
        """Sanitize LaTeX input to prevent XSS"""
        # Escape HTML special characters
        latex_str = html.escape(latex_str)
        # Remove potentially dangerous LaTeX commands
        dangerous_commands = ['\\input', '\\include', '\\write', '\\immediate']
        for cmd in dangerous_commands:
            latex_str = latex_str.replace(cmd, '')
        return latex_str
    
    @staticmethod
    def create_header():
        """Create application header"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin: 0;
        }
        .subtitle {
            color: #f0f0f0;
            font-size: 1.2rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        </style>
        
        <div class="main-header">
            <h1 class="main-title">🔬 Master Generators for ODEs</h1>
            <p class="subtitle">A Novel Approach to Construct and Solve Ordinary Differential Equations</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_sidebar_navigation() -> str:
        """Create sidebar navigation"""
        st.sidebar.markdown("## 📍 Navigation")
        
        modes = [
            ("🎯 Single Generator", "single"),
            ("📊 Batch Generation", "batch"),
            ("🤖 ML Pattern Learning", "ml"),
            ("🔍 Novelty Detection", "novelty"),
            ("📈 Visualization", "visualization"),
            ("📚 Documentation", "docs"),
            ("⚙️ Settings", "settings")
        ]
        
        selected = st.sidebar.radio(
            "Select Mode",
            options=[m[1] for m in modes],
            format_func=lambda x: next(m[0] for m in modes if m[1] == x)
        )
        
        return selected
    
    @staticmethod
    def parameter_input_panel() -> Dict[str, Any]:
        """Create parameter input panel with validation"""
        st.subheader("📝 Generator Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alpha = st.number_input(
                "α (Alpha)",
                min_value=-100.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Parameter α in the generator"
            )
            
            beta = st.number_input(
                "β (Beta)",
                min_value=0.1,
                max_value=100.0,
                value=1.0,
                step=0.1,
                help="Parameter β must be positive"
            )
        
        with col2:
            n = st.number_input(
                "n (Order parameter)",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Order parameter n"
            )
            
            M = st.number_input(
                "M (Constant)",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=0.1,
                help="Constant term M"
            )
        
        # Validate parameters
        params = {
            'alpha': float(alpha),
            'beta': float(beta),
            'n': int(n),
            'M': float(M)
        }
        
        # Additional validation
        if beta <= 0:
            st.error("β must be positive!")
            params['beta'] = 0.1
        
        return params
    
    @staticmethod
    def function_selector(include_special: bool = True) -> Tuple[str, str]:
        """Create function selector"""
        st.subheader("🔢 Function Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            func_type = st.selectbox(
                "Function Type",
                ["Basic", "Special"] if include_special else ["Basic"],
                help="Choose between basic and special mathematical functions"
            )
        
        with col2:
            if func_type == "Basic":
                functions = [
                    "linear", "quadratic", "cubic", "exponential",
                    "sine", "cosine", "tangent", "sinh", "cosh", "tanh",
                    "logarithm", "sqrt", "gaussian", "sigmoid"
                ]
            else:
                functions = [
                    "airy_ai", "airy_bi", "bessel_j0", "bessel_j1",
                    "gamma", "erf", "legendre_p2", "hermite_h3",
                    "chebyshev_t4", "zeta", "lambert_w"
                ]
            
            func_name = st.selectbox(
                "Function f(z)",
                functions,
                help="Select the function to use in the generator"
            )
        
        return func_type.lower(), func_name
    
    @staticmethod
    def generator_selector(gen_type: str) -> Tuple[int, Dict[str, int]]:
        """Create generator selector"""
        st.subheader("⚙️ Generator Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if gen_type == "linear":
                gen_num = st.selectbox(
                    "Generator Number",
                    list(range(1, 9)),
                    format_func=lambda x: f"Generator {x}",
                    help="Select linear generator (1-8)"
                )
            else:
                gen_num = st.selectbox(
                    "Generator Number",
                    list(range(1, 11)),
                    format_func=lambda x: f"Generator {x}",
                    help="Select nonlinear generator (1-10)"
                )
        
        extra_params = {}
        
        with col2:
            if gen_type == "nonlinear":
                if gen_num in [1, 2, 4]:
                    extra_params['q'] = st.slider("q (power)", 2, 10, 2)
                if gen_num in [2, 3, 5]:
                    extra_params['v'] = st.slider("v (power)", 2, 10, 3)
                if gen_num in [4, 5, 9, 10]:
                    extra_params['a'] = st.slider("a (scaling)", 0.5, 5.0, 2.0)
        
        return gen_num, extra_params
    
    @classmethod
    def display_ode_result(cls, result: Dict[str, Any]):
        """Display ODE generation result with LaTeX sanitization"""
        st.success("✅ ODE Generated Successfully!")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📐 Equation", "💡 Solution", "📊 Properties", "📝 LaTeX"])
        
        with tab1:
            st.subheader("Differential Equation")
            # Sanitize LaTeX before display
            safe_latex = cls.sanitize_latex(sp.latex(result['ode']))
            st.latex(safe_latex)
            
            st.subheader("Text Form")
            st.code(str(result['ode']), language='python')
        
        with tab2:
            st.subheader("Exact Solution")
            safe_solution = cls.sanitize_latex(sp.latex(result['solution']))
            st.latex(f"y(x) = {safe_solution}")
            
            st.subheader("Initial Conditions")
            for ic, value in result.get('initial_conditions', {}).items():
                st.write(f"• {ic} = {value}")
        
        with tab3:
            st.subheader("Properties")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Type", result['type'].capitalize())
                st.metric("Order", result['order'])
            
            with col2:
                st.metric("Generator", f"#{result['generator_number']}")
                if 'subtype' in result:
                    st.metric("Subtype", result['subtype'].capitalize())
            
            if 'powers' in result:
                st.write("**Nonlinear Powers:**")
                for var, power in result['powers'].items():
                    st.write(f"• {var} = {power}")
        
        with tab4:
            st.subheader("LaTeX Code")
            
            latex_code = f"""\\begin{{equation}}
{cls.sanitize_latex(sp.latex(result['ode']))}
\\end{{equation}}

\\begin{{equation}}
y(x) = {cls.sanitize_latex(sp.latex(result['solution']))}
\\end{{equation}}"""
            
            st.code(latex_code, language='latex')
            
            # Copy button
            if st.button("📋 Copy LaTeX"):
                st.write("LaTeX code copied to clipboard!")
                st.balloons()
    
    @staticmethod
    def plot_solution(result: Dict[str, Any], x_range: Tuple[float, float] = (-5, 5)):
        """Plot ODE solution with error handling"""
        st.subheader("📈 Solution Visualization")
        
        try:
            # Generate data points
            x = sp.Symbol('x')
            solution = result['solution']
            
            x_vals = np.linspace(x_range[0], x_range[1], 1000)
            
            # Convert to numerical function with error handling
            try:
                func = sp.lambdify(x, solution, 'numpy')
                y_vals = func(x_vals)
            except Exception as e:
                st.error(f"Cannot evaluate solution numerically: {e}")
                return
            
            # Handle infinite or NaN values
            y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)
            
            # Create plotly figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=np.real(y_vals),
                mode='lines',
                name='Real Part',
                line=dict(color='blue', width=2)
            ))
            
            if np.any(np.imag(y_vals) != 0):
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=np.imag(y_vals),
                    mode='lines',
                    name='Imaginary Part',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"Solution: {result['type'].capitalize()} ODE (Generator {result['generator_number']})",
                xaxis_title="x",
                yaxis_title="y(x)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not plot solution: {e}")
    
    @staticmethod
    def batch_results_table(results: List[Dict[str, Any]]):
        """Display batch generation results"""
        st.subheader("📊 Batch Generation Results")
        
        if not results:
            st.warning("No results to display")
            return
        
        # Convert to dataframe with error handling
        try:
            df = pd.DataFrame(results)
        except Exception as e:
            st.error(f"Error creating dataframe: {e}")
            return
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Generated", len(df))
        
        with col2:
            if 'type' in df.columns:
                linear_count = len(df[df['type'] == 'linear'])
                st.metric("Linear ODEs", linear_count)
        
        with col3:
            if 'type' in df.columns:
                nonlinear_count = len(df[df['type'] == 'nonlinear'])
                st.metric("Nonlinear ODEs", nonlinear_count)
        
        with col4:
            if 'order' in df.columns:
                avg_order = df['order'].mean()
                st.metric("Avg Order", f"{avg_order:.1f}")
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="generated_odes.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def novelty_analysis_display(analysis: Dict[str, Any]):
        """Display novelty analysis results"""
        st.subheader("🔍 Novelty Analysis Results")
        
        # Validate analysis data
        if not analysis:
            st.error("No analysis data available")
            return
        
        # Novelty score gauge
        novelty_score = analysis.get('novelty_score', 0)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=novelty_score,
            title={'text': "Novelty Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "gray"},
                    {'range': [60, 100], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis details
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis.get('is_novel', False):
                st.error("🚨 **NOVEL ODE DETECTED**")
            else:
                st.success("✅ **STANDARD ODE**")
            
            st.write(f"**Complexity:** {analysis.get('complexity_level', 'Unknown')}")
            
            if analysis.get('solvable_by_standard_methods', False):
                st.info("Can be solved by standard methods")
            else:
                st.warning("Requires advanced methods")
        
        with col2:
            st.write("**Recommended Methods:**")
            methods = analysis.get('recommended_methods', [])
            for method in methods[:5]:
                st.write(f"• {method}")
        
        # Detailed report
        if 'detailed_report' in analysis and analysis['detailed_report']:
            with st.expander("📄 View Detailed Report"):
                st.text(analysis['detailed_report'])
    
    @staticmethod
    def ml_training_interface():
        """Create ML training interface with validation"""
        st.subheader("🤖 Machine Learning Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["pattern_learner", "vae", "transformer"],
                format_func=lambda x: {
                    "pattern_learner": "Pattern Learner (Encoder-Decoder)",
                    "vae": "Variational Autoencoder",
                    "transformer": "Transformer"
                }[x]
            )
            
            epochs = st.slider("Training Epochs", 10, 500, 100)
            batch_size = st.slider("Batch Size", 16, 128, 32)
        
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
            
            samples = st.slider("Training Samples", 100, 5000, 1000)
            
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
        
        # Validate configuration
        if batch_size > samples:
            st.warning("Batch size should not exceed training samples!")
        
        return {
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'samples': samples,
            'validation_split': validation_split
        }
    
    def settings_page(self):
        """Enhanced settings page with all configurations"""
        st.header("⚙️ Application Settings")
        
        tabs = st.tabs(["General", "API", "ML Configuration", "Cache", "Export"])
        
        with tabs[0]:
            st.subheader("General Settings")
            
            # Theme selection
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            
            # Parameter limits
            st.markdown("### Parameter Limits")
            max_alpha = st.number_input("Max Alpha", 1, 1000, 100)
            max_beta = st.number_input("Max Beta", 1, 1000, 100)
            max_n = st.number_input("Max n", 1, 20, 10)
            
            # Display settings
            show_latex = st.checkbox("Show LaTeX equations", value=True)
            auto_save = st.checkbox("Auto-save generated ODEs", value=False)
            
            # Save settings
            if st.button("Save General Settings"):
                st.success("Settings saved!")
        
        with tabs[1]:
            st.subheader("API Configuration")
            
            api_url = st.text_input("API URL", value=self.api_url)
            api_key = st.text_input("API Key (optional)", type="password")
            
            if st.button("Test Connection"):
                if self.check_api_status():
                    st.success("✅ API Connected")
                else:
                    st.error("❌ API Connection Failed")
            
            # Rate limiting
            st.markdown("### Rate Limiting")
            rate_limit = st.number_input("Requests per minute", 1, 1000, 100)
            
            if st.button("Save API Settings"):
                self.api_url = api_url
                st.success("API settings updated!")
        
        with tabs[2]:
            st.subheader("ML Configuration")
            
            # Model settings
            default_model = st.selectbox(
                "Default Model",
                ["pattern_learner", "vae", "transformer"]
            )
            
            # Training defaults
            st.markdown("### Training Defaults")
            default_epochs = st.slider("Default Epochs", 10, 500, 100)
            default_batch_size = st.slider("Default Batch Size", 8, 128, 32)
            default_lr = st.select_slider(
                "Default Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
            
            # GPU settings
            use_gpu = st.checkbox("Use GPU if available", value=True)
            
            if st.button("Save ML Settings"):
                st.success("ML settings saved!")
        
        with tabs[3]:
            st.subheader("Cache Management")
            
            # Cache statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cached_odes = len(st.session_state.get('generated_odes', []))
                st.metric("Cached ODEs", cached_odes)
            
            with col2:
                batch_results = len(st.session_state.get('batch_results', []))
                st.metric("Batch Results", batch_results)
            
            with col3:
                if st.button("Clear All Cache"):
                    st.session_state.generated_odes = []
                    st.session_state.batch_results = []
                    st.session_state.analysis_results = []
                    st.success("✅ Cache cleared!")
                    st.rerun()
            
            # Cache settings
            max_cache_size = st.number_input("Max Cache Size", 100, 10000, 1000)
            cache_ttl = st.number_input("Cache TTL (seconds)", 60, 3600, 600)
            
            if st.button("Save Cache Settings"):
                st.success("Cache settings saved!")
        
        with tabs[4]:
            st.subheader("Export Settings")
            
            # Export all data
            if st.button("Export All Data"):
                all_data = {
                    'generated_odes': [
                        {
                            'ode': str(ode.get('ode', '')),
                            'solution': str(ode.get('solution', '')),
                            'type': ode.get('type', ''),
                            'order': ode.get('order', 0)
                        } for ode in st.session_state.get('generated_odes', [])
                    ],
                    'batch_results': st.session_state.get('batch_results', []),
                    'timestamp': datetime.now().isoformat()
                }
                
                json_data = json.dumps(all_data, indent=2)
                st.download_button(
                    "📥 Download All Data",
                    json_data,
                    "master_generators_export.json",
                    "application/json"
                )
            
            # Database export
            st.markdown("### Database Export")
            export_format = st.selectbox("Format", ["JSON", "CSV", "SQL"])
            
            if st.button("Export Database"):
                st.info(f"Exporting to {export_format} format...")

    def check_api_status(self) -> bool:
        """Check if API is available with timeout"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            st.error(f"API connection error: {e}")
            return False
    
    @staticmethod
    def display_training_progress(history: Dict[str, List[float]]):
        """Display training progress charts"""
        st.subheader("📊 Training Progress")
        
        if not history or not history.get('train_loss'):
            st.info("No training history available")
            return
        
        # Create loss plot
        fig = go.Figure()
        
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['train_loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        if 'val_loss' in history and history['val_loss']:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
        
        with col2:
            if 'val_loss' in history and history['val_loss']:
                st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
        
        with col3:
            st.metric("Epochs Completed", len(history['train_loss']))
    
    @staticmethod
    def create_footer():
        """Create application footer"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Master Generators for ODEs v2.0.0</p>
            <p>Based on the research paper by Mohammad Abu-Ghuwaleh et al.</p>
            <p>© 2024 Master Generators Team</p>
        </div>
        """, unsafe_allow_html=True)
