# master_generators_app_integrated.py
"""
Streamlit UI for Master Generators using Unified Core
"""

import streamlit as st
import sympy as sp
import json
import traceback
from typing import Dict, Any, Optional
import datetime as dt

# Import the unified core module
from unified_core_fixed import (
    UnifiedMasterGenerator,
    safe_eval_f_of_z,
    Theorem42,
    GeneratorBuilder,
    LinearGeneratorFactory,
    NonlinearGeneratorFactory
)

# ---- Page config ----
st.set_page_config(page_title="Master Generators (Unified)", layout="wide")

# ---- Session init ----
if "current_generator" not in st.session_state:
    st.session_state.current_generator = None
if "generated_odes" not in st.session_state:
    st.session_state.generated_odes = []

# ---- Helpers ----
def hr():
    st.markdown("---")

def to_latex(expr: sp.Expr) -> str:
    return sp.latex(expr)

# ---- Pages ----

def page_generators():
    st.title("Master Generators — Unified System")
    
    # Choose generation mode
    mode = st.radio(
        "Generation Mode",
        ["Template-based (Theorem 4.2)", "Standard Generators (Linear/Nonlinear)"],
        horizontal=True
    )
    
    colL, colR = st.columns([7, 5])
    
    with colL:
        if mode == "Template-based (Theorem 4.2)":
            st.subheader("1) Compose LHS with Template")
            st.markdown(
                """
                **Build custom equations like:**
                - `y + Dy2` → y(x) + y''(x) = RHS
                - `sin(Dy1) + exp(Dy2)` → sin(y'(x)) + e^(y''(x)) = RHS
                - `y + Dy1 + Dy3 + sinh(Dym)` → y + y' + y''' + sinh(y^(m)) = RHS
                """
            )
            
            template = st.text_input(
                "LHS template", 
                value="y + sin(Dy1) + exp(Dy2)",
                help="Use y, Dy1, Dy2, Dy3, Dym, with functions like sin, cos, exp, log, sinh, cosh"
            )
            
        else:  # Standard Generators
            st.subheader("1) Select Standard Generator")
            
            gen_type = st.selectbox("Generator Type", ["linear", "nonlinear"])
            
            if gen_type == "linear":
                gen_num = st.selectbox(
                    "Generator Number",
                    list(range(1, 9)),
                    format_func=lambda x: {
                        1: "1: y'' + y = RHS",
                        2: "2: y'' + y' = RHS",
                        3: "3: y + y' = RHS",
                        4: "4: y'' + y(x/a) - y = RHS (Pantograph)",
                        5: "5: y(x/a) + y' = RHS",
                        6: "6: y''' + y = RHS",
                        7: "7: y''' + y' = RHS",
                        8: "8: y''' + y'' = RHS"
                    }.get(x, f"Generator {x}")
                )
            else:
                gen_num = st.selectbox(
                    "Generator Number",
                    list(range(1, 11)),
                    format_func=lambda x: {
                        1: "1: (y'')^q + y = RHS",
                        2: "2: (y'')^q + (y')^v = RHS",
                        3: "3: y + (y')^v = RHS",
                        4: "4: (y'')^q + y(x/a) - y = RHS",
                        5: "5: y(x/a) + (y')^v = RHS",
                        6: "6: sin(y'') + y = RHS",
                        7: "7: e^(y'') + e^(y') = RHS",
                        8: "8: y + e^(y') = RHS",
                        9: "9: e^(y'') + y(x/a) - y = RHS",
                        10: "10: y(x/a) + ln(y') = RHS"
                    }.get(x, f"Generator {x}")
                )
        
        # Function selection
        st.subheader("2) Select Function f(z)")
        f_str = st.text_input("Function f(z)", value="exp(z)")
        
        hr()
        st.subheader("3) Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_val = st.number_input("n (integer ≥ 1)", min_value=1, max_value=10, value=2)
        with c2:
            alpha_val = st.number_input("α", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
        with c3:
            beta_val = st.number_input("β (> 0)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        M_val = st.number_input("M (constant)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        
        # Additional parameters for specific generators
        extra_params = {}
        if mode == "Template-based (Theorem 4.2)":
            if "Dym" in template:
                m_val = st.number_input("m (for Dym)", min_value=1, max_value=10, value=3)
                extra_params['m'] = m_val
        else:
            if gen_type == "nonlinear":
                if gen_num in [1, 2, 4]:
                    extra_params['q'] = st.slider("q (power)", 2, 10, 2)
                if gen_num in [2, 3, 5]:
                    extra_params['v'] = st.slider("v (power)", 2, 10, 3)
            if (gen_type == "linear" and gen_num in [4, 5]) or \
               (gen_type == "nonlinear" and gen_num in [4, 5, 9, 10]):
                extra_params['a'] = st.slider("a (scaling)", 0.5, 5.0, 2.0)
        
        complex_form = st.checkbox("Keep complex form", value=True)
        
        hr()
        st.subheader("4) Generate ODE")
        generate_btn = st.button("🚀 Generate ODE with Exact Solution", type="primary")
    
    with colR:
        st.subheader("Preview")
        if mode == "Template-based (Theorem 4.2)":
            st.info(f"Template will create: {template} = RHS")
        else:
            st.info(f"{gen_type.capitalize()} Generator #{gen_num}")
    
    if generate_btn:
        try:
            # Create unified generator
            generator = UnifiedMasterGenerator(
                alpha=alpha_val,
                beta=beta_val,
                n=n_val,
                M=M_val
            )
            
            # Parse function
            f = safe_eval_f_of_z(f_str)
            
            # Generate based on mode
            if mode == "Template-based (Theorem 4.2)":
                result = generator.generate_from_template(
                    template=template,
                    f_z=f,
                    m=extra_params.get('m'),
                    complex_form=complex_form
                )
            else:
                result = generator.generate_standard(
                    gen_type=gen_type,
                    gen_num=gen_num,
                    f_z=sp.sympify(f_str),
                    **extra_params
                )
            
            st.success("✅ ODE Generated Successfully!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Differential Equation")
                st.latex(to_latex(result['ode']))
                
                st.subheader("Left-Hand Side")
                st.latex(to_latex(result['lhs']))
            
            with col2:
                st.subheader("Right-Hand Side")
                st.latex(to_latex(result['rhs']))
                
                st.subheader("Exact Solution")
                st.latex(f"y(x) = {to_latex(result['solution'])}")
            
            # Store in session
            st.session_state.current_generator = result
            st.session_state.generated_odes.append({
                'ode': result['ode'],
                'solution': result['solution'],
                'timestamp': dt.datetime.now().isoformat(),
                'params': {'alpha': alpha_val, 'beta': beta_val, 'n': n_val, 'M': M_val}
            })
            
            # JSON export
            with st.expander("📄 View JSON"):
                json_data = generator.export_to_json(result)
                st.json(json.loads(json_data))
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_data,
                    file_name="ode_result.json",
                    mime="application/json"
                )
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())

def page_batch():
    st.title("Batch Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        count = st.number_input("Number of ODEs", min_value=1, max_value=100, value=10)
        mode = st.selectbox("Generation Mode", ["mixed", "template", "standard"])
    
    with col2:
        # Base parameters
        alpha = st.number_input("Base α", value=1.0)
        beta = st.number_input("Base β", value=1.0, min_value=0.1)
        n = st.number_input("Base n", value=2, min_value=1)
    
    if st.button("Generate Batch", type="primary"):
        generator = UnifiedMasterGenerator(alpha=alpha, beta=beta, n=n, M=0.0)
        
        with st.spinner(f"Generating {count} ODEs..."):
            results = generator.generate_batch(count=count, mode=mode)
        
        st.success(f"Generated {len(results)} ODEs!")
        
        # Display summary
        for i, result in enumerate(results, 1):
            with st.expander(f"ODE {i}: {result['type']} (Order {result['order']})"):
                st.latex(to_latex(result['ode']))
                if st.button(f"View Details", key=f"detail_{i}"):
                    st.write(f"**Solution:** {result['solution']}")
                    st.write(f"**Description:** {result.get('description', 'N/A')}")

def page_history():
    st.title("Generation History")
    
    if not st.session_state.generated_odes:
        st.info("No ODEs generated yet.")
        return
    
    st.write(f"Total ODEs generated: {len(st.session_state.generated_odes)}")
    
    for i, ode_data in enumerate(reversed(st.session_state.generated_odes[-10:]), 1):
        with st.expander(f"ODE {len(st.session_state.generated_odes) - i + 1}"):
            st.latex(to_latex(ode_data['ode']))
            st.write(f"**Generated at:** {ode_data['timestamp']}")
            st.write(f"**Parameters:** α={ode_data['params']['alpha']}, "
                    f"β={ode_data['params']['beta']}, n={ode_data['params']['n']}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Generator", "Batch Generation", "History"],
        index=0
    )
    
    if page == "Generator":
        page_generators()
    elif page == "Batch Generation":
        page_batch()
    elif page == "History":
        page_history()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Master Generators v2.0**  
        Unified System combining:
        - Theorem 4.2 Templates
        - Standard Generators (1-8 Linear, 1-10 Nonlinear)
        """
    )

if __name__ == "__main__":
    main()
