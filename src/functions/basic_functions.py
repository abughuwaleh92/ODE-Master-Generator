"""
Basic Mathematical Functions for ODE Generators
Includes elementary functions like sin, cos, exp, log, etc.
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Callable

class BasicFunctions:
    """
    Collection of basic mathematical functions for use in generators
    """
    
    def __init__(self):
        self.z = sp.Symbol('z')
        self.functions = self._initialize_functions()
        
    def _initialize_functions(self) -> Dict[str, sp.Expr]:
        """
        Initialize all basic functions
        
        Returns:
            Dictionary of function names to symbolic expressions
        """
        return {
            # Algebraic functions
            'constant': sp.Integer(1),
            'linear': self.z,
            'quadratic': self.z**2,
            'cubic': self.z**3,
            'quartic': self.z**4,
            'quintic': self.z**5,
            'sqrt': sp.sqrt(self.z),
            'cbrt': self.z**(sp.Rational(1, 3)),
            'reciprocal': 1/self.z,
            'reciprocal_square': 1/self.z**2,
            
            # Exponential and logarithmic
            'exponential': sp.exp(self.z),
            'exp2': sp.exp(2*self.z),
            'exp_neg': sp.exp(-self.z),
            'logarithm': sp.log(self.z),
            'log10': sp.log(self.z, 10),
            'log2': sp.log(self.z, 2),
            
            # Trigonometric functions
            'sine': sp.sin(self.z),
            'cosine': sp.cos(self.z),
            'tangent': sp.tan(self.z),
            'cotangent': sp.cot(self.z),
            'secant': sp.sec(self.z),
            'cosecant': sp.csc(self.z),
            
            # Inverse trigonometric
            'arcsin': sp.asin(self.z),
            'arccos': sp.acos(self.z),
            'arctan': sp.atan(self.z),
            'arccot': sp.acot(self.z),
            'arcsec': sp.asec(self.z),
            'arccsc': sp.acsc(self.z),
            
            # Hyperbolic functions
            'sinh': sp.sinh(self.z),
            'cosh': sp.cosh(self.z),
            'tanh': sp.tanh(self.z),
            'coth': sp.coth(self.z),
            'sech': sp.sech(self.z),
            'csch': sp.csch(self.z),
            
            # Inverse hyperbolic
            'arcsinh': sp.asinh(self.z),
            'arccosh': sp.acosh(self.z),
            'arctanh': sp.atanh(self.z),
            'arccoth': sp.acoth(self.z),
            'arcsech': sp.asech(self.z),
            'arccsch': sp.acsch(self.z),
            
            # Combined functions
            'sin_squared': sp.sin(self.z)**2,
            'cos_squared': sp.cos(self.z)**2,
            'sin_cos': sp.sin(self.z) * sp.cos(self.z),
            'exp_sin': sp.exp(self.z) * sp.sin(self.z),
            'exp_cos': sp.exp(self.z) * sp.cos(self.z),
            'log_sin': sp.log(sp.sin(self.z)),
            'log_cos': sp.log(sp.cos(self.z)),
            
            # Special combinations
            'gaussian': sp.exp(-self.z**2),
            'sigmoid': 1 / (1 + sp.exp(-self.z)),
            'relu': sp.Max(0, self.z),
            'softplus': sp.log(1 + sp.exp(self.z)),
            'swish': self.z / (1 + sp.exp(-self.z)),
        }
    
    def get_function(self, name: str) -> sp.Expr:
        """
        Get a function by name
        
        Args:
            name: Function name
            
        Returns:
            Symbolic expression for the function
        """
        if name not in self.functions:
            raise ValueError(f"Unknown function: {name}")
        return self.functions[name]
    
    def get_all_functions(self) -> Dict[str, sp.Expr]:
        """
        Get all available functions
        
        Returns:
            Dictionary of all functions
        """
        return self.functions.copy()
    
    def get_function_names(self) -> list:
        """
        Get list of all function names
        
        Returns:
            List of function names
        """
        return list(self.functions.keys())
    
    def evaluate_function(self, name: str, value: float) -> float:
        """
        Evaluate a function at a specific value
        
        Args:
            name: Function name
            value: Value to evaluate at
            
        Returns:
            Function value
        """
        func = self.get_function(name)
        return float(func.subs(self.z, value))
    
    def get_derivative(self, name: str, order: int = 1) -> sp.Expr:
        """
        Get the derivative of a function
        
        Args:
            name: Function name
            order: Order of derivative
            
        Returns:
            Derivative expression
        """
        func = self.get_function(name)
        for _ in range(order):
            func = sp.diff(func, self.z)
        return func
    
    def get_integral(self, name: str) -> sp.Expr:
        """
        Get the indefinite integral of a function
        
        Args:
            name: Function name
            
        Returns:
            Integral expression
        """
        func = self.get_function(name)
        return sp.integrate(func, self.z)
    
    def get_function_properties(self, name: str) -> Dict[str, Any]:
        """
        Get properties of a function
        
        Args:
            name: Function name
            
        Returns:
            Dictionary of function properties
        """
        func = self.get_function(name)
        
        return {
            'name': name,
            'expression': str(func),
            'latex': sp.latex(func),
            'is_polynomial': func.is_polynomial(self.z),
            'is_rational': func.is_rational_function(self.z),
            'is_algebraic': func.is_algebraic_expr(self.z),
            'is_transcendental': not func.is_algebraic_expr(self.z),
            'degree': sp.degree(func, self.z) if func.is_polynomial(self.z) else None,
            'derivative': str(self.get_derivative(name)),
            'integral': str(self.get_integral(name))
        }
    
    def create_composite_function(self, f_name: str, g_name: str) -> sp.Expr:
        """
        Create composite function f(g(z))
        
        Args:
            f_name: Outer function name
            g_name: Inner function name
            
        Returns:
            Composite function expression
        """
        f = self.get_function(f_name)
        g = self.get_function(g_name)
        return f.subs(self.z, g)
    
    def create_linear_combination(self, functions: Dict[str, float]) -> sp.Expr:
        """
        Create linear combination of functions
        
        Args:
            functions: Dictionary of function names to coefficients
            
        Returns:
            Linear combination expression
        """
        result = 0
        for name, coeff in functions.items():
            result += coeff * self.get_function(name)
        return result
    
    def create_product(self, function_names: list) -> sp.Expr:
        """
        Create product of functions
        
        Args:
            function_names: List of function names
            
        Returns:
            Product expression
        """
        result = 1
        for name in function_names:
            result *= self.get_function(name)
        return result
