# src/generators/generator_constructor.py
"""
Advanced Generator Constructor System
Allows creation of custom generators by combining derivatives
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

class DerivativeType(Enum):
    """Types of derivative terms"""
    LINEAR = "linear"
    POWER = "power"
    EXPONENTIAL = "exponential"
    TRIGONOMETRIC = "trigonometric"
    LOGARITHMIC = "logarithmic"
    SPECIAL = "special"

@dataclass
class DerivativeTerm:
    """Represents a single term in the generator"""
    derivative_order: int  # 0 for y, 1 for y', 2 for y'', etc.
    coefficient: float = 1.0
    power: Optional[int] = 1  # For (y^(k))^p
    function_type: DerivativeType = DerivativeType.LINEAR
    scaling: Optional[float] = None  # For y(x/a)
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}
    
    def to_sympy(self, y_func, x_sym):
        """Convert to SymPy expression"""
        # Get the derivative
        if self.derivative_order == 0:
            expr = y_func(x_sym)
        else:
            expr = y_func(x_sym).diff(x_sym, self.derivative_order)
        
        # Apply scaling if present
        if self.scaling:
            expr = expr.subs(x_sym, x_sym / self.scaling)
        
        # Apply function type
        if self.function_type == DerivativeType.POWER and self.power != 1:
            expr = expr ** self.power
        elif self.function_type == DerivativeType.EXPONENTIAL:
            expr = sp.exp(expr)
        elif self.function_type == DerivativeType.TRIGONOMETRIC:
            trig_func = self.additional_params.get('trig_func', 'sin')
            if trig_func == 'sin':
                expr = sp.sin(expr)
            elif trig_func == 'cos':
                expr = sp.cos(expr)
            elif trig_func == 'tan':
                expr = sp.tan(expr)
        elif self.function_type == DerivativeType.LOGARITHMIC:
            expr = sp.log(expr)
        
        # Apply coefficient
        return self.coefficient * expr

class GeneratorConstructor:
    """
    Advanced generator constructor using Theorem 4.1 and 4.2
    """
    
    def __init__(self):
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Function('y')
        self.z = sp.Symbol('z')
        
    def construct_generator(
        self,
        terms: List[DerivativeTerm],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Construct a custom generator from derivative terms
        
        Args:
            terms: List of derivative terms to combine
            name: Optional name for the generator
            description: Optional description
            
        Returns:
            Generator specification
        """
        # Build the left-hand side of the ODE
        lhs = 0
        for term in terms:
            lhs += term.to_sympy(self.y, self.x)
        
        # Determine generator properties
        max_order = max(term.derivative_order for term in terms)
        is_linear = all(
            term.function_type == DerivativeType.LINEAR and term.power == 1
            for term in terms
        )
        
        # Check for special types
        has_delay = any(term.scaling is not None for term in terms)
        has_trigonometric = any(
            term.function_type == DerivativeType.TRIGONOMETRIC 
            for term in terms
        )
        has_exponential = any(
            term.function_type == DerivativeType.EXPONENTIAL 
            for term in terms
        )
        
        # Generate name if not provided
        if not name:
            name = self._generate_name(terms)
        
        # Generate description if not provided
        if not description:
            description = self._generate_description(terms)
        
        return {
            'lhs': lhs,
            'terms': terms,
            'order': max_order,
            'is_linear': is_linear,
            'has_delay': has_delay,
            'has_trigonometric': has_trigonometric,
            'has_exponential': has_exponential,
            'name': name,
            'description': description,
            'latex': sp.latex(lhs),
            'string_form': str(lhs)
        }
    
    def _generate_name(self, terms: List[DerivativeTerm]) -> str:
        """Generate a name for the generator"""
        parts = []
        
        # Count derivative orders
        orders = {}
        for term in terms:
            order = term.derivative_order
            if order not in orders:
                orders[order] = []
            orders[order].append(term)
        
        # Build name based on terms
        for order in sorted(orders.keys()):
            order_terms = orders[order]
            if order == 0:
                parts.append("y")
            elif order == 1:
                parts.append("y'")
            elif order == 2:
                parts.append("y''")
            else:
                parts.append(f"y^({order})")
        
        return "-".join(parts) + " Generator"
    
    def _generate_description(self, terms: List[DerivativeTerm]) -> str:
        """Generate a description for the generator"""
        parts = []
        
        for i, term in enumerate(terms):
            # Build term description
            term_str = ""
            
            # Add coefficient if not 1
            if term.coefficient != 1:
                if term.coefficient == -1:
                    term_str = "-"
                else:
                    term_str = f"{term.coefficient}"
            
            # Add derivative
            if term.derivative_order == 0:
                base = "y"
            elif term.derivative_order == 1:
                base = "y'"
            elif term.derivative_order == 2:
                base = "y''"
            else:
                base = f"y^({term.derivative_order})"
            
            # Add scaling
            if term.scaling:
                base = base.replace("y", f"y(x/{term.scaling})")
            
            # Apply function type
            if term.function_type == DerivativeType.POWER and term.power != 1:
                base = f"({base})^{term.power}"
            elif term.function_type == DerivativeType.EXPONENTIAL:
                base = f"e^({base})"
            elif term.function_type == DerivativeType.TRIGONOMETRIC:
                trig_func = term.additional_params.get('trig_func', 'sin')
                base = f"{trig_func}({base})"
            elif term.function_type == DerivativeType.LOGARITHMIC:
                base = f"ln({base})"
            
            term_str += base
            
            # Add sign for next term
            if i > 0:
                if term.coefficient >= 0:
                    parts.append("+")
            
            parts.append(term_str)
        
        return " ".join(parts) + " = RHS"
    
    def apply_master_theorem(
        self,
        generator_spec: Dict[str, Any],
        f_z: sp.Expr,
        alpha: float = 1.0,
        beta: float = 1.0,
        n: int = 1,
        M: float = 0.0
    ) -> Dict[str, Any]:
        """
        Apply Master Theorem to generate the solution
        
        Uses Theorem 4.1 and 4.2 to generate the exact solution
        """
        # Calculate omega values
        omegas = [(2*s - 1) * sp.pi / (2*n) for s in range(1, n+1)]
        
        # Calculate psi and phi functions
        def psi_func(omega, x_val):
            exp_arg = sp.I * x_val * sp.cos(omega) - x_val * sp.sin(omega)
            z_val = alpha + beta * sp.exp(exp_arg)
            return f_z.subs(self.z, z_val)
        
        def phi_func(omega, x_val):
            exp_arg = -sp.I * x_val * sp.cos(omega) - x_val * sp.sin(omega)
            z_val = alpha + beta * sp.exp(exp_arg)
            return f_z.subs(self.z, z_val)
        
        # Generate y(x) using Theorem 4.1
        y_solution = 0
        for omega in omegas:
            psi = psi_func(omega, self.x)
            phi = phi_func(omega, self.x)
            f_alpha_beta = f_z.subs(self.z, alpha + beta)
            y_solution += 2*f_alpha_beta - (psi + phi)
        
        y_solution = sp.pi / (2*n) * y_solution + sp.pi * M
        
        # Calculate derivatives needed for the generator
        derivatives = {0: y_solution}
        max_order = generator_spec['order']
        
        for k in range(1, max_order + 1):
            derivatives[k] = self._calculate_kth_derivative(
                k, f_z, alpha, beta, n, omegas, psi_func, phi_func
            )
        
        # Build the actual ODE
        lhs_evaluated = 0
        for term in generator_spec['terms']:
            deriv = derivatives[term.derivative_order]
            
            # Apply scaling if needed
            if term.scaling:
                deriv = deriv.subs(self.x, self.x / term.scaling)
            
            # Apply function type
            if term.function_type == DerivativeType.POWER and term.power != 1:
                deriv = deriv ** term.power
            elif term.function_type == DerivativeType.EXPONENTIAL:
                deriv = sp.exp(deriv)
            elif term.function_type == DerivativeType.TRIGONOMETRIC:
                trig_func = term.additional_params.get('trig_func', 'sin')
                if trig_func == 'sin':
                    deriv = sp.sin(deriv)
                elif trig_func == 'cos':
                    deriv = sp.cos(deriv)
            elif term.function_type == DerivativeType.LOGARITHMIC:
                deriv = sp.log(deriv)
            
            lhs_evaluated += term.coefficient * deriv
        
        # The RHS is what makes the equation true
        rhs = lhs_evaluated
        
        return {
            'generator': generator_spec,
            'solution': y_solution,
            'lhs': generator_spec['lhs'],
            'rhs': rhs,
            'ode': sp.Eq(generator_spec['lhs'], rhs),
            'parameters': {
                'alpha': alpha,
                'beta': beta,
                'n': n,
                'M': M,
                'f_z': str(f_z)
            },
            'initial_conditions': self._calculate_initial_conditions(
                derivatives, max_order
            )
        }
    
    def _calculate_kth_derivative(
        self, k, f_z, alpha, beta, n, omegas, psi_func, phi_func
    ):
        """
        Calculate k-th derivative using Theorem 4.2
        """
        # This is a simplified version - full implementation would use
        # the complete formulas from Theorem 4.2
        
        if k == 1:
            # Use formula from Theorem 4.1 for y'(x)
            derivative = 0
            for omega in omegas:
                psi = psi_func(omega, self.x)
                phi = phi_func(omega, self.x)
                
                # Partial derivatives
                psi_alpha = sp.diff(psi, alpha)
                phi_alpha = sp.diff(phi, alpha)
                
                term1 = sp.cos(self.x * sp.cos(omega) + omega) * (psi_alpha - phi_alpha) / sp.I
                term2 = sp.sin(self.x * sp.cos(omega) + omega) * (psi_alpha + phi_alpha)
                
                derivative += beta * sp.exp(-self.x * sp.sin(omega)) * (term1 + term2)
            
            return sp.pi / (2*n) * derivative
        
        elif k == 2:
            # Use formula from Theorem 4.1 for y''(x)
            derivative = 0
            for omega in omegas:
                psi = psi_func(omega, self.x)
                phi = phi_func(omega, self.x)
                
                # First order terms
                psi_alpha = sp.diff(psi, alpha)
                phi_alpha = sp.diff(phi, alpha)
                
                term1 = beta * sp.exp(-self.x * sp.sin(omega)) * (
                    sp.cos(self.x * sp.cos(omega) + 2*omega) * (psi_alpha + phi_alpha) +
                    sp.sin(self.x * sp.cos(omega) + 2*omega) * (psi_alpha - phi_alpha) / sp.I
                )
                
                # Second order terms
                psi_alpha2 = sp.diff(psi, alpha, 2)
                phi_alpha2 = sp.diff(phi, alpha, 2)
                
                term2 = beta**2 * sp.exp(-2*self.x * sp.sin(omega)) * (
                    sp.cos(2*self.x * sp.cos(omega) + 2*omega) * (psi_alpha2 + phi_alpha2) +
                    sp.sin(2*self.x * sp.cos(omega) + 2*omega) * (psi_alpha2 - phi_alpha2) / sp.I
                )
                
                derivative += term1 + term2
            
            return sp.pi / (2*n) * derivative
        
        else:
            # For higher derivatives, use Theorem 4.2 formulas
            # This would require implementing the coefficient table
            return sp.diff(self._calculate_kth_derivative(k-1, f_z, alpha, beta, n, omegas, psi_func, phi_func), self.x)
    
    def _calculate_initial_conditions(self, derivatives, max_order):
        """Calculate initial conditions at x=0"""
        ic = {}
        for k in range(max_order + 1):
            if k == 0:
                ic['y(0)'] = derivatives[k].subs(self.x, 0)
            elif k == 1:
                ic["y'(0)"] = derivatives[k].subs(self.x, 0)
            elif k == 2:
                ic["y''(0)"] = derivatives[k].subs(self.x, 0)
            else:
                ic[f"y^({k})(0)"] = derivatives[k].subs(self.x, 0)
        return ic
