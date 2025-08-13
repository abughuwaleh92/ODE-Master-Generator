"""
Special Mathematical Functions for ODE Generators
Includes Airy, Bessel, Gamma, Legendre, Hermite, and other special functions
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional

class SpecialFunctions:
    """
    Collection of special mathematical functions for use in generators
    """
    
    def __init__(self):
        self.z = sp.Symbol('z')
        self.n = sp.Symbol('n', integer=True)
        self.functions = self._initialize_functions()
        
    def _initialize_functions(self) -> Dict[str, sp.Expr]:
        """
        Initialize all special functions
        
        Returns:
            Dictionary of function names to symbolic expressions
        """
        return {
            # Airy functions
            'airy_ai': sp.airyai(self.z),
            'airy_bi': sp.airybi(self.z),
            'airy_ai_prime': sp.airyaiprime(self.z),
            'airy_bi_prime': sp.airybiprime(self.z),
            
            # Bessel functions of the first kind
            'bessel_j0': sp.besselj(0, self.z),
            'bessel_j1': sp.besselj(1, self.z),
            'bessel_j2': sp.besselj(2, self.z),
            'bessel_j3': sp.besselj(3, self.z),
            'bessel_jn': sp.besselj(self.n, self.z),
            
            # Bessel functions of the second kind
            'bessel_y0': sp.bessely(0, self.z),
            'bessel_y1': sp.bessely(1, self.z),
            'bessel_y2': sp.bessely(2, self.z),
            'bessel_y3': sp.bessely(3, self.z),
            'bessel_yn': sp.bessely(self.n, self.z),
            
            # Modified Bessel functions
            'bessel_i0': sp.besseli(0, self.z),
            'bessel_i1': sp.besseli(1, self.z),
            'bessel_k0': sp.besselk(0, self.z),
            'bessel_k1': sp.besselk(1, self.z),
            
            # Hankel functions
            'hankel1': sp.hankel1(1, self.z),
            'hankel2': sp.hankel2(1, self.z),
            
            # Gamma and related functions
            'gamma': sp.gamma(self.z),
            'loggamma': sp.loggamma(self.z),
            'digamma': sp.digamma(self.z),
            'trigamma': sp.trigamma(self.z),
            'polygamma': sp.polygamma(2, self.z),
            'beta': sp.beta(self.z, 2),
            
            # Error functions
            'erf': sp.erf(self.z),
            'erfc': sp.erfc(self.z),
            'erfi': sp.erfi(self.z),
            'erf_inv': sp.erfinv(self.z),
            'erfc_inv': sp.erfcinv(self.z),
            
            # Fresnel integrals
            'fresnel_s': sp.fresnels(self.z),
            'fresnel_c': sp.fresnelc(self.z),
            
            # Exponential integrals
            'exp_integral_e': sp.expint(1, self.z),
            'exp_integral_ei': sp.Ei(self.z),
            'log_integral': sp.li(self.z),
            'sine_integral': sp.Si(self.z),
            'cosine_integral': sp.Ci(self.z),
            'sinh_integral': sp.Shi(self.z),
            'cosh_integral': sp.Chi(self.z),
            
            # Orthogonal polynomials
            'legendre_p0': sp.legendre(0, self.z),
            'legendre_p1': sp.legendre(1, self.z),
            'legendre_p2': sp.legendre(2, self.z),
            'legendre_p3': sp.legendre(3, self.z),
            'legendre_p4': sp.legendre(4, self.z),
            
            'chebyshev_t0': sp.chebyshevt(0, self.z),
            'chebyshev_t1': sp.chebyshevt(1, self.z),
            'chebyshev_t2': sp.chebyshevt(2, self.z),
            'chebyshev_t3': sp.chebyshevt(3, self.z),
            'chebyshev_t4': sp.chebyshevt(4, self.z),
            
            'chebyshev_u0': sp.chebyshevu(0, self.z),
            'chebyshev_u1': sp.chebyshevu(1, self.z),
            'chebyshev_u2': sp.chebyshevu(2, self.z),
            'chebyshev_u3': sp.chebyshevu(3, self.z),
            'chebyshev_u4': sp.chebyshevu(4, self.z),
            
            'hermite_h0': sp.hermite(0, self.z),
            'hermite_h1': sp.hermite(1, self.z),
            'hermite_h2': sp.hermite(2, self.z),
            'hermite_h3': sp.hermite(3, self.z),
            'hermite_h4': sp.hermite(4, self.z),
            
            'laguerre_l0': sp.laguerre(0, self.z),
            'laguerre_l1': sp.laguerre(1, self.z),
            'laguerre_l2': sp.laguerre(2, self.z),
            'laguerre_l3': sp.laguerre(3, self.z),
            
            'jacobi_p': sp.jacobi(2, 1, 1, self.z),
            'gegenbauer': sp.gegenbauer(2, sp.Rational(1, 2), self.z),
            
            # Zeta and related functions
            'zeta': sp.zeta(self.z),
            'dirichlet_eta': sp.dirichlet_eta(self.z),
            'lerch_phi': sp.lerchphi(self.z, 2, sp.Rational(1, 2)),
            'polylog': sp.polylog(2, self.z),
            
            # Hypergeometric functions
            'hypergeometric_0f1': sp.hyper([],[sp.Rational(1,2)], self.z),
            'hypergeometric_1f1': sp.hyper([1],[2], self.z),
            'hypergeometric_2f1': sp.hyper([sp.Rational(1,2), sp.Rational(1,3)], [sp.Rational(3,2)], self.z),
            
            # Lambert W function
            'lambert_w': sp.LambertW(self.z),
            'lambert_w_minus1': sp.LambertW(self.z, -1),
            
            # Elliptic integrals
            'elliptic_k': sp.elliptic_k(self.z),
            'elliptic_e': sp.elliptic_e(self.z),
            'elliptic_pi': sp.elliptic_pi(sp.Rational(1,2), self.z),
            
            # Mathieu functions
            'mathieu_s': sp.mathieusprime(1, 1, self.z),
            'mathieu_c': sp.mathieucprime(1, 1, self.z),
            
            # Struve functions
            'struve_h': sp.Function('StruveH')(0, self.z),
            'struve_l': sp.Function('StruveL')(0, self.z),
        }
    
    def get_function(self, name: str, **params) -> sp.Expr:
        """
        Get a special function by name
        
        Args:
            name: Function name
            **params: Optional parameters for parameterized functions
            
        Returns:
            Symbolic expression for the function
        """
        if name not in self.functions:
            raise ValueError(f"Unknown special function: {name}")
        
        func = self.functions[name]
        
        # Handle parameterized functions
        if 'n' in params and self.n in func.free_symbols:
            func = func.subs(self.n, params['n'])
        
        return func
    
    def get_all_functions(self) -> Dict[str, sp.Expr]:
        """
        Get all available special functions
        
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
    
    def get_function_by_category(self, category: str) -> Dict[str, sp.Expr]:
        """
        Get functions by category
        
        Args:
            category: Category name (airy, bessel, gamma, orthogonal, etc.)
            
        Returns:
            Dictionary of functions in that category
        """
        categories = {
            'airy': ['airy_ai', 'airy_bi', 'airy_ai_prime', 'airy_bi_prime'],
            'bessel': [k for k in self.functions.keys() if 'bessel' in k],
            'gamma': ['gamma', 'loggamma', 'digamma', 'trigamma', 'polygamma', 'beta'],
            'error': ['erf', 'erfc', 'erfi', 'erf_inv', 'erfc_inv'],
            'orthogonal': [k for k in self.functions.keys() if any(
                poly in k for poly in ['legendre', 'chebyshev', 'hermite', 'laguerre', 'jacobi', 'gegenbauer']
            )],
            'hypergeometric': [k for k in self.functions.keys() if 'hypergeometric' in k],
            'elliptic': [k for k in self.functions.keys() if 'elliptic' in k],
            'integral': [k for k in self.functions.keys() if 'integral' in k],
            'zeta': ['zeta', 'dirichlet_eta', 'lerch_phi', 'polylog'],
        }
        
        if category not in categories:
            raise ValueError(f"Unknown category: {category}")
        
        return {name: self.functions[name] for name in categories[category]}
    
    def evaluate_function(self, name: str, value: complex, **params) -> complex:
        """
        Evaluate a special function at a specific value
        
        Args:
            name: Function name
            value: Value to evaluate at
            **params: Optional parameters
            
        Returns:
            Function value
        """
        func = self.get_function(name, **params)
        result = func.subs(self.z, value)
        
        # Try to evaluate numerically
        try:
            return complex(result.evalf())
        except:
            return result
    
    def get_series_expansion(self, name: str, order: int = 5, point: float = 0) -> sp.Expr:
        """
        Get series expansion of a special function
        
        Args:
            name: Function name
            order: Order of expansion
            point: Point around which to expand
            
        Returns:
            Series expansion
        """
        func = self.get_function(name)
        return func.series(self.z, point, order).removeO()
    
    def get_asymptotic_expansion(self, name: str, order: int = 3) -> sp.Expr:
        """
        Get asymptotic expansion for large arguments
        
        Args:
            name: Function name
            order: Order of expansion
            
        Returns:
            Asymptotic expansion
        """
        func = self.get_function(name)
        # This is a simplified version; actual implementation would be more complex
        return func.series(self.z, sp.oo, order).removeO()
    
    def get_recurrence_relation(self, name: str) -> Optional[str]:
        """
        Get recurrence relation for special functions
        
        Args:
            name: Function name
            
        Returns:
            String describing the recurrence relation
        """
        relations = {
            'bessel_jn': 'J_{n+1}(z) = (2n/z)J_n(z) - J_{n-1}(z)',
            'bessel_yn': 'Y_{n+1}(z) = (2n/z)Y_n(z) - Y_{n-1}(z)',
            'legendre_pn': '(n+1)P_{n+1}(z) = (2n+1)zP_n(z) - nP_{n-1}(z)',
            'hermite_hn': 'H_{n+1}(z) = 2zH_n(z) - 2nH_{n-1}(z)',
            'chebyshev_tn': 'T_{n+1}(z) = 2zT_n(z) - T_{n-1}(z)',
            'laguerre_ln': '(n+1)L_{n+1}(z) = (2n+1-z)L_n(z) - nL_{n-1}(z)',
        }
        
        for key, relation in relations.items():
            if key in name:
                return relation
        
        return None
    
    def get_differential_equation(self, name: str) -> Optional[str]:
        """
        Get the differential equation satisfied by the special function
        
        Args:
            name: Function name
            
        Returns:
            String describing the differential equation
        """
        equations = {
            'airy_ai': "y'' - z*y = 0",
            'bessel_jn': "z²y'' + zy' + (z² - n²)y = 0",
            'legendre_pn': "(1-z²)y'' - 2zy' + n(n+1)y = 0",
            'hermite_hn': "y'' - 2zy' + 2ny = 0",
            'chebyshev_tn': "(1-z²)y'' - zy' + n²y = 0",
            'laguerre_ln': "zy'' + (1-z)y' + ny = 0",
        }
        
        for key, equation in equations.items():
            if key in name:
                return equation
        
        return None