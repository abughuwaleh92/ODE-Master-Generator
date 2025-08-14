# unified_core.py
"""
Unified Master Generators Core Module
Combines Theorem 4.2 template-based approach with complete standard generators
"""

from __future__ import annotations

import sympy as sp
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CORE THEOREM 4.2 IMPLEMENTATION
# ============================================================================

class Theorem42:
    """
    Implements the generalized (compact) form of Theorem 4.2
    Supports both symbolic and numeric n and m parameters
    """
    
    def __init__(
        self,
        x: Optional[sp.Symbol] = None,
        alpha: Optional[sp.Symbol] = None,
        beta: Optional[sp.Symbol] = None,
        n: Union[int, str, sp.Symbol] = "n",
        m_sym: Optional[sp.Symbol] = None
    ):
        self.x = x or sp.symbols("x", real=True)
        self.alpha = alpha if alpha is not None else sp.symbols("alpha", real=True)
        self.beta = beta if beta is not None else sp.symbols("beta", real=True)
        
        # Handle n as symbolic or numeric
        if isinstance(n, (int, sp.Integer)):
            self.n = int(n)
        elif isinstance(n, str):
            self.n = sp.symbols(n, integer=True, positive=True)
        else:
            self.n = n
        
        # Canonical integer symbols for summation
        self.s = sp.symbols("s", integer=True, positive=True)
        self.j = sp.symbols("j", integer=True, positive=True)
        
        # Symbolic m
        self.m_sym = m_sym if isinstance(m_sym, sp.Symbol) else sp.symbols("m", integer=True, positive=True)
        
        # Function symbol
        self.yfun = sp.Function("y")(self.x)
    
    def _omega(self, s: Optional[int] = None) -> sp.Expr:
        """Compute ω(s) = (2s-1)π/(2n)"""
        s_val = self.s if s is None else s
        return sp.pi * (2*s_val - 1) / (2*self.n)
    
    def _lambda(self, s: Optional[int] = None) -> sp.Expr:
        """λ_s = exp(i x cos ω_s − x sin ω_s)"""
        w = self._omega(s)
        return sp.exp(sp.I*self.x*sp.cos(w) - self.x*sp.sin(w))
    
    def _lambda_bar(self, s: Optional[int] = None) -> sp.Expr:
        """λ̄_s = exp(−i x cos ω_s − x sin ω_s)"""
        w = self._omega(s)
        return sp.exp(-sp.I*self.x*sp.cos(w) - self.x*sp.sin(w))
    
    def _zeta(self, s: Optional[int] = None) -> sp.Expr:
        """ζ_s = exp(−x sin ω_s)"""
        w = self._omega(s)
        return sp.exp(-self.x*sp.sin(w))
    
    def _psi(self, f: Callable[[sp.Expr], sp.Expr], s: Optional[int] = None) -> sp.Expr:
        """ψ = f(α + β λ_s)"""
        lam = self._lambda(s)
        return f(self.alpha + self.beta*lam)
    
    def _phi(self, f: Callable[[sp.Expr], sp.Expr], s: Optional[int] = None) -> sp.Expr:
        """φ = f(α + β λ̄_s)"""
        lam_bar = self._lambda_bar(s)
        return f(self.alpha + self.beta*lam_bar)
    
    def y_base(self, f: Callable[[sp.Expr], sp.Expr], n_override: Optional[Any] = None) -> sp.Expr:
        """Generate y(x) solution using Theorem 4.1"""
        nval = self.n if n_override is None else n_override
        s = self.s
        
        # Build the sum
        f_alpha_beta = f(self.alpha + self.beta)
        expr = sp.summation(
            2*f_alpha_beta - self._psi(f, s) - self._phi(f, s),
            (s, 1, nval)
        )
        
        return sp.pi/(2*nval) * expr
    
    def y_derivative(
        self,
        f: Callable[[sp.Expr], sp.Expr],
        m: Optional[Union[int, sp.Symbol]] = None,
        n_override: Optional[Any] = None,
        complex_form: bool = True
    ) -> sp.Expr:
        """
        General m-th derivative of y(x)
        Handles both numeric and symbolic m
        """
        nval = self.n if n_override is None else n_override
        s, j = self.s, self.j
        
        lam = self._lambda(s)
        lamp = self._lambda_bar(s)
        zeta = self._zeta(s)
        
        # Numeric m - explicit expansion
        if isinstance(m, (int, sp.Integer)):
            m = int(m)
            if m <= 0:
                return sp.S.Zero
            
            inner = sp.S.Zero
            for jj in range(1, m+1):
                # Stirling number of second kind
                from sympy.functions.combinatorial.numbers import stirling
                S = stirling(m, jj, kind=2)
                
                # Derivatives of psi and phi w.r.t. alpha
                dpsi = sp.Derivative(self._psi(f, s), (self.alpha, jj))
                dphi = sp.Derivative(self._phi(f, s), (self.alpha, jj))
                
                # Build term
                term = S * (self.beta**jj) * (zeta**jj) * (
                    lam**m * dpsi + lamp**m * dphi
                )
                inner += term
            
            result = -sp.pi/(2*nval) * sp.summation(inner, (s, 1, nval))
            
        else:
            # Symbolic m - use symbolic sums
            m_sym = self.m_sym if m is None else m
            from sympy.functions.combinatorial.numbers import stirling
            S2 = stirling(m_sym, j, kind=2)
            
            dpsi = sp.Derivative(self._psi(f, s), (self.alpha, j))
            dphi = sp.Derivative(self._phi(f, s), (self.alpha, j))
            
            summand = S2 * (self.beta**j) * (zeta**j) * (
                lam**m_sym * dpsi + lamp**m_sym * dphi
            )
            
            result = -sp.pi/(2*nval) * sp.summation(
                sp.summation(summand, (j, 1, m_sym)), 
                (s, 1, nval)
            )
        
        return result if complex_form else sp.re(result)


# ============================================================================
# TEMPLATE-BASED GENERATOR BUILDER
# ============================================================================

@dataclass
class TemplateConfig:
    """Configuration for template-based generators"""
    alpha: Any = sp.symbols("alpha", real=True)
    beta: Any = sp.symbols("beta", real=True)
    n: Any = sp.symbols("n", integer=True, positive=True)
    m_sym: Any = sp.symbols("m", integer=True, positive=True)


class GeneratorBuilder:
    """
    Builds ODEs from free-form templates using Theorem 4.2
    """
    
    def __init__(self, theorem: Theorem42, config: Optional[TemplateConfig] = None):
        self.T = theorem
        self.cfg = config or TemplateConfig()
        self.x = self.T.x
        self.yfun = self.T.yfun
        self.m_val: Optional[Any] = None
    
    @staticmethod
    def _parse_template(template: str) -> str:
        """Parse template aliases"""
        import re
        # Map y^(m) → Dym, y^(k) → Dyk
        s = re.sub(r"y\^\(\s*m\s*\)", "Dym", template)
        s = re.sub(r"y\^\(\s*(\d+)\s*\)", lambda m: f"Dy{m.group(1)}", s)
        return s
    
    @staticmethod
    def _extract_orders(template: str) -> List[int]:
        """Extract derivative orders from template"""
        import re
        return sorted({int(m) for m in re.findall(r"\bDy(\d+)\b", template)})
    
    def build(
        self,
        template: str,
        f: Callable[[sp.Expr], sp.Expr],
        m: Optional[Union[int, sp.Symbol]] = None,
        n_override: Optional[Any] = None,
        complex_form: bool = True
    ) -> Tuple[sp.Expr, sp.Expr]:
        """
        Build LHS and RHS from template
        
        Args:
            template: Template string like "y + sin(Dy1) + exp(Dy2)"
            f: Function f(z)
            m: Value for symbolic m
            n_override: Override for n
            complex_form: Keep complex form or take real part
            
        Returns:
            (LHS expression, RHS expression)
        """
        self.m_val = m
        template = self._parse_template(template)
        orders = self._extract_orders(template)
        uses_dym = "Dym" in template
        
        # Build namespace for LHS
        ns_lhs = _sympy_ns()
        ns_lhs.update({
            "x": self.x,
            "y": self.yfun,
            "alpha": self.T.alpha,
            "beta": self.T.beta,
            "m": m if m is not None else self.T.m_sym
        })
        
        # Add derivative symbols
        for k in orders:
            ns_lhs[f"Dy{k}"] = sp.Derivative(self.yfun, (self.x, k))
        
        if uses_dym:
            if isinstance(m, (int, sp.Integer)):
                ns_lhs["Dym"] = sp.Derivative(self.yfun, (self.x, int(m)))
            else:
                ns_lhs["Dym"] = sp.Symbol("Dym")  # Placeholder for symbolic m
        
        # Parse LHS
        try:
            lhs = sp.sympify(template, locals=ns_lhs)
        except Exception as e:
            raise ValueError(f"Cannot parse LHS template: {e}\nTemplate: {template}")
        
        # Build RHS by substituting actual expressions
        rhs_map = {}
        rhs_map["y"] = self.T.y_base(f, n_override=n_override)
        
        for k in orders:
            rhs_map[f"Dy{k}"] = self.T.y_derivative(f, m=k, n_override=n_override, complex_form=complex_form)
        
        if uses_dym:
            rhs_map["Dym"] = self.T.y_derivative(
                f,
                m=(m if m is not None else self.T.m_sym),
                n_override=n_override,
                complex_form=complex_form
            )
        
        # Build RHS namespace
        ns_rhs = ns_lhs.copy()
        ns_rhs.update(rhs_map)
        
        # Parse RHS
        try:
            rhs = sp.sympify(template, locals=ns_rhs)
        except Exception as e:
            raise ValueError(f"Cannot build RHS: {e}\nTemplate: {template}")
        
        return sp.simplify(lhs), sp.simplify(rhs)


# ============================================================================
# COMPLETE STANDARD GENERATORS
# ============================================================================

class StandardGenerator:
    """Base class for standard generators"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, n: int = 1, M: float = 0.0):
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if n < 1:
            raise ValueError("n must be at least 1")
        
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.M = M
        
        self.x = sp.Symbol('x', real=True)
        self.z = sp.Symbol('z')
        self.y = sp.Function('y')
    
    def compute_derivatives_at_exp(self, f_z: sp.Expr, max_order: int = 3) -> Dict[int, sp.Expr]:
        """Compute derivatives of f at α + βe^(-x)"""
        exp_arg = self.alpha + self.beta * sp.exp(-self.x)
        derivatives = {}
        
        derivatives[0] = f_z.subs(self.z, exp_arg)
        
        current_deriv = f_z
        for order in range(1, max_order + 1):
            current_deriv = sp.diff(current_deriv, self.z)
            derivatives[order] = current_deriv.subs(self.z, exp_arg)
        
        return derivatives
    
    def compute_derivatives_at_scaled(self, f_z: sp.Expr, a: float, max_order: int = 3) -> Dict[int, sp.Expr]:
        """Compute derivatives of f at α + βe^(-x/a)"""
        exp_arg_scaled = self.alpha + self.beta * sp.exp(-self.x/a)
        derivatives = {}
        
        derivatives[0] = f_z.subs(self.z, exp_arg_scaled)
        
        current_deriv = f_z
        for order in range(1, max_order + 1):
            current_deriv = sp.diff(current_deriv, self.z)
            derivatives[order] = current_deriv.subs(self.z, exp_arg_scaled)
        
        return derivatives


class LinearGeneratorFactory(StandardGenerator):
    """Factory for all 8 linear generators with explicit RHS"""
    
    def create(self, generator_number: int, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Create linear generator by number (1-8)"""
        
        # Initialize with parameters
        self.__init__(**params)
        
        generators = {
            1: self._generator_1,
            2: self._generator_2,
            3: self._generator_3,
            4: self._generator_4,
            5: self._generator_5,
            6: self._generator_6,
            7: self._generator_7,
            8: self._generator_8,
        }
        
        if generator_number not in generators:
            raise ValueError(f"Linear generator number must be 1-8, got {generator_number}")
        
        return generators[generator_number](f_z, **params)
    
    def _generator_1(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 1: y''(x) + y(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 2)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M 
                      - self.beta * sp.exp(-self.x) * derivs[1]
                      - self.beta**2 * sp.exp(-2*self.x) * derivs[2])
        
        lhs = self.y(self.x).diff(self.x, 2) + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        # Generate solution using Theorem 4.2
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n, m_sym=sp.Symbol('m'))
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 2,
            'generator_number': 1,
            'description': "y''(x) + y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_2(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 2: y''(x) + y'(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 2)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M
                      - self.beta**2 * sp.exp(-2*self.x) * derivs[2])
        
        lhs = self.y(self.x).diff(self.x, 2) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 2,
            'generator_number': 2,
            'description': "y''(x) + y'(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_3(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 3: y(x) + y'(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        
        rhs = sp.pi * (f_alpha_beta + self.M)
        
        lhs = self.y(self.x) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 1,
            'generator_number': 3,
            'description': "y(x) + y'(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_4(self, f_z: sp.Expr, a: float = 2, **kwargs) -> Dict[str, Any]:
        """Generator 4: y''(x) + y(x/a) - y(x) = RHS (Pantograph)"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs_exp = self.compute_derivatives_at_exp(f_z, 2)
        derivs_scaled = self.compute_derivatives_at_scaled(f_z, a, 2)
        
        rhs = sp.pi * (f_alpha_beta - derivs_exp[0] + derivs_scaled[0] - derivs_exp[0] + self.M
                      - self.beta * sp.exp(-self.x) * derivs_exp[1]
                      - self.beta**2 * sp.exp(-2*self.x) * derivs_exp[2])
        
        lhs = self.y(self.x).diff(self.x, 2) + self.y(self.x/a) - self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'subtype': 'pantograph',
            'order': 2,
            'generator_number': 4,
            'scaling_parameter': a,
            'description': f"y''(x) + y(x/{a}) - y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_5(self, f_z: sp.Expr, a: float = 2, **kwargs) -> Dict[str, Any]:
        """Generator 5: y(x/a) + y'(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs_scaled = self.compute_derivatives_at_scaled(f_z, a, 1)
        
        rhs = sp.pi * (f_alpha_beta + derivs_scaled[0] - f_alpha_beta + self.M)
        
        lhs = self.y(self.x/a) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'subtype': 'delay',
            'order': 1,
            'generator_number': 5,
            'scaling_parameter': a,
            'description': f"y(x/{a}) + y'(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_6(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 6: y'''(x) + y(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 3)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M
                      - self.beta * sp.exp(-self.x) * derivs[1]
                      - self.beta**2 * sp.exp(-2*self.x) * derivs[2]
                      - self.beta**3 * sp.exp(-3*self.x) * derivs[3])
        
        lhs = self.y(self.x).diff(self.x, 3) + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 3,
            'generator_number': 6,
            'description': "y'''(x) + y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_7(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 7: y'''(x) + y'(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 3)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M
                      - self.beta**3 * sp.exp(-3*self.x) * derivs[3])
        
        lhs = self.y(self.x).diff(self.x, 3) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 3,
            'generator_number': 7,
            'description': "y'''(x) + y'(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_8(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 8: y'''(x) + y''(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 3)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M
                      - self.beta * sp.exp(-self.x) * derivs[1]
                      - self.beta**3 * sp.exp(-3*self.x) * derivs[3])
        
        lhs = self.y(self.x).diff(self.x, 3) + self.y(self.x).diff(self.x, 2)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 3,
            'generator_number': 8,
            'description': "y'''(x) + y''(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }


class NonlinearGeneratorFactory(StandardGenerator):
    """Factory for all 10 nonlinear generators with explicit RHS"""
    
    def create(self, generator_number: int, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Create nonlinear generator by number (1-10)"""
        
        # Initialize with parameters
        self.__init__(
            alpha=params.get('alpha', 1.0),
            beta=params.get('beta', 1.0),
            n=params.get('n', 1),
            M=params.get('M', 0.0)
        )
        
        generators = {
            1: self._generator_1,
            2: self._generator_2,
            3: self._generator_3,
            4: self._generator_4,
            5: self._generator_5,
            6: self._generator_6,
            7: self._generator_7,
            8: self._generator_8,
            9: self._generator_9,
            10: self._generator_10,
        }
        
        if generator_number not in generators:
            raise ValueError(f"Nonlinear generator number must be 1-10, got {generator_number}")
        
        return generators[generator_number](f_z, **params)
    
    def _generator_1(self, f_z: sp.Expr, q: int = 2, **kwargs) -> Dict[str, Any]:
        """Generator 1: (y''(x))^q + y(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_expr = -self.beta * sp.exp(-self.x) * derivs[1] - self.beta**2 * sp.exp(-2*self.x) * derivs[2]
        y_expr = f_alpha_beta - derivs[0] + self.M
        
        rhs = sp.pi**q * y_double_prime_expr**q + sp.pi * y_expr
        
        lhs = self.y(self.x).diff(self.x, 2)**q + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'order': 2,
            'generator_number': 1,
            'powers': {'q': q},
            'description': f"(y''(x))^{q} + y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_2(self, f_z: sp.Expr, q: int = 2, v: int = 3, **kwargs) -> Dict[str, Any]:
        """Generator 2: (y''(x))^q + (y'(x))^v = RHS"""
        derivs = self.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_expr = -self.beta * sp.exp(-self.x) * derivs[1] - self.beta**2 * sp.exp(-2*self.x) * derivs[2]
        y_prime_expr = self.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.pi**q * y_double_prime_expr**q + sp.pi**v * y_prime_expr**v
        
        lhs = self.y(self.x).diff(self.x, 2)**q + self.y(self.x).diff(self.x)**v
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'order': 2,
            'generator_number': 2,
            'powers': {'q': q, 'v': v},
            'description': f"(y''(x))^{q} + (y'(x))^{v} = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_3(self, f_z: sp.Expr, v: int = 3, **kwargs) -> Dict[str, Any]:
        """Generator 3: y(x) + (y'(x))^v = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 1)
        
        y_prime_expr = self.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M) + sp.pi**v * y_prime_expr**v
        
        lhs = self.y(self.x) + self.y(self.x).diff(self.x)**v
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'order': 1,
            'generator_number': 3,
            'powers': {'v': v},
            'description': f"y(x) + (y'(x))^{v} = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_4(self, f_z: sp.Expr, q: int = 2, a: float = 2, **kwargs) -> Dict[str, Any]:
        """Generator 4: (y''(x))^q + y(x/a) - y(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs_exp = self.compute_derivatives_at_exp(f_z, 2)
        derivs_scaled = self.compute_derivatives_at_scaled(f_z, a, 2)
        
        y_double_prime_expr = -self.beta * sp.exp(-self.x) * derivs_exp[1] - self.beta**2 * sp.exp(-2*self.x) * derivs_exp[2]
        
        rhs = sp.pi**q * y_double_prime_expr**q + sp.pi * (derivs_scaled[0] - derivs_exp[0] + self.M)
        
        lhs = self.y(self.x).diff(self.x, 2)**q + self.y(self.x/a) - self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'pantograph',
            'order': 2,
            'generator_number': 4,
            'powers': {'q': q},
            'scaling_parameter': a,
            'description': f"(y''(x))^{q} + y(x/{a}) - y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_5(self, f_z: sp.Expr, v: int = 3, a: float = 2, **kwargs) -> Dict[str, Any]:
        """Generator 5: y(x/a) + (y'(x))^v = RHS"""
        derivs_exp = self.compute_derivatives_at_exp(f_z, 1)
        derivs_scaled = self.compute_derivatives_at_scaled(f_z, a, 1)
        
        y_prime_expr = self.beta * sp.exp(-self.x) * derivs_exp[1]
        
        rhs = sp.pi * (derivs_scaled[0] + self.M) + sp.pi**v * y_prime_expr**v
        
        lhs = self.y(self.x/a) + self.y(self.x).diff(self.x)**v
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'delay',
            'order': 1,
            'generator_number': 5,
            'powers': {'v': v},
            'scaling_parameter': a,
            'description': f"y(x/{a}) + (y'(x))^{v} = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_6(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 6: sin(y''(x)) + y(x) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_term = sp.pi * (-self.beta * sp.exp(-self.x) * derivs[1] 
                                      - self.beta**2 * sp.exp(-2*self.x) * derivs[2])
        y_term = sp.pi * (f_alpha_beta - derivs[0] + self.M)
        
        rhs = sp.sin(y_double_prime_term) + y_term
        
        lhs = sp.sin(self.y(self.x).diff(self.x, 2)) + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'trigonometric',
            'order': 2,
            'generator_number': 6,
            'description': "sin(y''(x)) + y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_7(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 7: e^(y''(x)) + e^(y'(x)) = RHS"""
        derivs = self.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_term = sp.pi * (-self.beta * sp.exp(-self.x) * derivs[1] 
                                      - self.beta**2 * sp.exp(-2*self.x) * derivs[2])
        y_prime_term = sp.pi * self.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.exp(y_double_prime_term) + sp.exp(y_prime_term)
        
        lhs = sp.exp(self.y(self.x).diff(self.x, 2)) + sp.exp(self.y(self.x).diff(self.x))
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'exponential',
            'order': 2,
            'generator_number': 7,
            'description': "e^(y''(x)) + e^(y'(x)) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_8(self, f_z: sp.Expr, **kwargs) -> Dict[str, Any]:
        """Generator 8: y(x) + e^(y'(x)) = RHS"""
        f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
        derivs = self.compute_derivatives_at_exp(f_z, 1)
        
        y_prime_term = sp.pi * self.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + self.M) + sp.exp(y_prime_term)
        
        lhs = self.y(self.x) + sp.exp(self.y(self.x).diff(self.x))
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'exponential',
            'order': 1,
            'generator_number': 8,
            'description': "y(x) + e^(y'(x)) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_9(self, f_z: sp.Expr, a: float = 2, **kwargs) -> Dict[str, Any]:
        """Generator 9: e^(y''(x)) + y(x/a) - y(x) = RHS"""
        derivs_exp = self.compute_derivatives_at_exp(f_z, 2)
        derivs_scaled = self.compute_derivatives_at_scaled(f_z, a, 0)
        
        y_double_prime_term = sp.pi * (-self.beta * sp.exp(-self.x) * derivs_exp[1] 
                                      - self.beta**2 * sp.exp(-2*self.x) * derivs_exp[2])
        
        rhs = sp.exp(y_double_prime_term) + sp.pi * (derivs_scaled[0] - derivs_exp[0] + self.M)
        
        lhs = sp.exp(self.y(self.x).diff(self.x, 2)) + self.y(self.x/a) - self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'exponential-pantograph',
            'order': 2,
            'generator_number': 9,
            'scaling_parameter': a,
            'description': f"e^(y''(x)) + y(x/{a}) - y(x) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }
    
    def _generator_10(self, f_z: sp.Expr, a: float = 2, **kwargs) -> Dict[str, Any]:
        """Generator 10: y(x/a) + ln(y'(x)) = RHS"""
        derivs_exp = self.compute_derivatives_at_exp(f_z, 1)
        derivs_scaled = self.compute_derivatives_at_scaled(f_z, a, 0)
        
        y_prime_term = sp.pi * self.beta * sp.exp(-self.x) * derivs_exp[1]
        
        # Handle log carefully to avoid negative arguments
        rhs = sp.pi * (derivs_scaled[0] + self.M) + sp.log(sp.Abs(y_prime_term))
        
        lhs = self.y(self.x/a) + sp.log(sp.Abs(self.y(self.x).diff(self.x)))
        ode = sp.Eq(lhs, rhs)
        
        theorem = Theorem42(alpha=self.alpha, beta=self.beta, n=self.n)
        solution = theorem.y_base(lambda z: f_z.subs(self.z, z))
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'logarithmic',
            'order': 1,
            'generator_number': 10,
            'scaling_parameter': a,
            'description': f"y(x/{a}) + ln(|y'(x)|) = RHS",
            'initial_conditions': {'y(0)': sp.pi * self.M}
        }


# ============================================================================
# UNIFIED MASTER GENERATOR SYSTEM
# ============================================================================

class UnifiedMasterGenerator:
    """
    Unified system combining template-based and standard generators
    """
    
    def __init__(self, **params):
        """
        Initialize unified generator
        
        Args:
            alpha: Parameter alpha
            beta: Parameter beta (must be positive)
            n: Order parameter (int or symbolic)
            M: Constant term
        """
        # Store parameters
        self.params = {
            'alpha': params.get('alpha', 1.0),
            'beta': params.get('beta', 1.0),
            'n': params.get('n', 1),
            'M': params.get('M', 0.0)
        }
        
        # Initialize both systems
        self.theorem42 = Theorem42(
            alpha=self.params['alpha'],
            beta=self.params['beta'],
            n=self.params['n']
        )
        
        self.generator_builder = GeneratorBuilder(self.theorem42)
        self.linear_factory = LinearGeneratorFactory(**self.params)
        self.nonlinear_factory = NonlinearGeneratorFactory(**self.params)
        
        logger.info("Unified Master Generator initialized")
    
    def generate_from_template(
        self,
        template: str,
        f_z: Union[sp.Expr, Callable],
        m: Optional[Union[int, sp.Symbol]] = None,
        complex_form: bool = True
    ) -> Dict[str, Any]:
        """
        Generate ODE using template-based approach
        
        Args:
            template: Template string (e.g., "y + sin(Dy1) + exp(Dy2)")
            f_z: Function f(z) as expression or callable
            m: Value for symbolic m in template
            complex_form: Keep complex form or take real part
            
        Returns:
            Dictionary with ODE details
        """
        # Convert f_z to callable if needed
        if isinstance(f_z, sp.Expr):
            z = sp.Symbol('z')
            f = lambda arg: f_z.subs(z, arg)
        else:
            f = f_z
        
        # Build using template
        lhs, rhs = self.generator_builder.build(
            template=template,
            f=f,
            m=m,
            complex_form=complex_form
        )
        
        # Create solution
        solution = self.theorem42.y_base(f)
        
        # Extract derivative orders for classification
        import re
        orders = sorted({int(m) for m in re.findall(r"\bDy(\d+)\b", template)})
        max_order = max(orders) if orders else 0
        
        return {
            'ode': sp.Eq(lhs, rhs),
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'template-based',
            'order': max_order,
            'template': template,
            'initial_conditions': {'y(0)': sp.pi * self.params['M']},
            'description': f"Template: {template}"
        }
    
    def generate_standard(
        self,
        gen_type: str,
        gen_num: int,
        f_z: sp.Expr,
        **extra_params
    ) -> Dict[str, Any]:
        """
        Generate ODE using standard generators
        
        Args:
            gen_type: 'linear' or 'nonlinear'
            gen_num: Generator number (1-8 for linear, 1-10 for nonlinear)
            f_z: Function f(z)
            **extra_params: Additional parameters (q, v, a)
            
        Returns:
            Dictionary with ODE details
        """
        if gen_type == 'linear':
            return self.linear_factory.create(gen_num, f_z, **extra_params)
        elif gen_type == 'nonlinear':
            return self.nonlinear_factory.create(gen_num, f_z, **extra_params)
        else:
            raise ValueError(f"Unknown generator type: {gen_type}")
    
    def generate_batch(
        self,
        count: int,
        mode: str = 'mixed',
        functions: Optional[List[sp.Expr]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple ODEs
        
        Args:
            count: Number of ODEs to generate
            mode: 'template', 'standard', or 'mixed'
            functions: List of f(z) functions to use
            
        Returns:
            List of ODE dictionaries
        """
        results = []
        
        # Default functions if not provided
        if functions is None:
            z = sp.Symbol('z')
            functions = [
                z, z**2, sp.exp(z), sp.sin(z), sp.cos(z),
                sp.sinh(z), sp.cosh(z), sp.log(z + 1)
            ]
        
        # Templates for template mode
        templates = [
            "y + Dy2",
            "y + Dy1",
            "Dy2 + Dy1",
            "y + sin(Dy1)",
            "exp(Dy2) + y",
            "Dy3 + y",
            "y + Dy1 + Dy2",
            "sinh(Dy1) + y"
        ]
        
        for i in range(count):
            # Random selection
            f_z = np.random.choice(functions)
            
            if mode == 'template' or (mode == 'mixed' and np.random.random() < 0.5):
                # Template-based generation
                template = np.random.choice(templates)
                result = self.generate_from_template(template, f_z)
            else:
                # Standard generation
                gen_type = np.random.choice(['linear', 'nonlinear'])
                
                if gen_type == 'linear':
                    gen_num = np.random.randint(1, 9)
                    result = self.generate_standard(gen_type, gen_num, f_z)
                else:
                    gen_num = np.random.randint(1, 11)
                    extra = {}
                    
                    # Add random parameters for specific generators
                    if gen_num in [1, 2, 4]:
                        extra['q'] = np.random.randint(2, 5)
                    if gen_num in [2, 3, 5]:
                        extra['v'] = np.random.randint(2, 5)
                    if gen_num in [4, 5, 9, 10]:
                        extra['a'] = np.random.uniform(1.5, 3.0)
                    
                    result = self.generate_standard(gen_type, gen_num, f_z, **extra)
            
            results.append(result)
        
        return results
    
    def export_to_json(self, result: Dict[str, Any]) -> str:
        """Export ODE result to JSON format"""
        export_data = {
            'ode': str(result['ode']),
            'lhs': str(result['lhs']),
            'rhs': str(result['rhs']),
            'solution': str(result['solution']),
            'type': result['type'],
            'order': result['order'],
            'description': result.get('description', ''),
            'initial_conditions': {k: str(v) for k, v in result.get('initial_conditions', {}).items()},
            'parameters': self.params
        }
        
        # Add extra fields if present
        for key in ['generator_number', 'template', 'powers', 'scaling_parameter', 'subtype']:
            if key in result:
                export_data[key] = result[key]
        
        return json.dumps(export_data, indent=2)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _sympy_ns() -> Dict[str, Any]:
    """Create safe SymPy namespace for sympify"""
    ns = {name: getattr(sp, name) for name in dir(sp) if not name.startswith("_")}
    ns.update({
        "I": sp.I, "pi": sp.pi, "E": sp.E,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "re": sp.re, "im": sp.im,
    })
    return ns


def safe_eval_f_of_z(f_str: str) -> Callable[[sp.Expr], sp.Expr]:
    """
    Create a function f(z) from string safely
    
    Args:
        f_str: String representation like "sin(z)", "exp(z)", etc.
        
    Returns:
        Callable that takes a SymPy expression and returns f(expression)
    """
    z = sp.Symbol('z')
    try:
        f_expr = sp.sympify(f_str, locals={**_sympy_ns(), "z": z})
    except Exception as e:
        raise ValueError(f"Cannot parse f(z) from string: {f_str!r}. Error: {e}")
    
    def f(arg: sp.Expr) -> sp.Expr:
        return sp.simplify(f_expr.xreplace({z: arg}))
    
    return f


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test the unified system
    print("Testing Unified Master Generator System")
    print("=" * 60)
    
    # Initialize unified generator
    generator = UnifiedMasterGenerator(alpha=1.0, beta=1.0, n=2, M=0.0)
    
    # Test 1: Template-based generation
    print("\n1. Template-based Generation:")
    print("-" * 40)
    
    result1 = generator.generate_from_template(
        template="y + sin(Dy1) + exp(Dy2)",
        f_z=safe_eval_f_of_z("exp(z)"),
        m=2
    )
    
    print(f"Template: y + sin(Dy1) + exp(Dy2)")
    print(f"ODE: {result1['ode']}")
    print(f"Order: {result1['order']}")
    
    # Test 2: Standard linear generator
    print("\n2. Standard Linear Generator:")
    print("-" * 40)
    
    z = sp.Symbol('z')
    result2 = generator.generate_standard(
        gen_type='linear',
        gen_num=1,
        f_z=sp.sin(z)
    )
    
    print(f"Generator: Linear #1")
    print(f"Description: {result2['description']}")
    print(f"ODE: {result2['ode']}")
    
    # Test 3: Standard nonlinear generator
    print("\n3. Standard Nonlinear Generator:")
    print("-" * 40)
    
    result3 = generator.generate_standard(
        gen_type='nonlinear',
        gen_num=6,
        f_z=sp.exp(z)
    )
    
    print(f"Generator: Nonlinear #6")
    print(f"Description: {result3['description']}")
    print(f"ODE: {result3['ode']}")
    
    # Test 4: Batch generation
    print("\n4. Batch Generation (Mixed Mode):")
    print("-" * 40)
    
    batch = generator.generate_batch(count=5, mode='mixed')
    
    for i, ode in enumerate(batch, 1):
        print(f"  {i}. Type: {ode['type']}, Order: {ode['order']}")
    
    # Test 5: JSON export
    print("\n5. JSON Export:")
    print("-" * 40)
    
    json_data = generator.export_to_json(result1)
    print(json_data[:200] + "...")
    
    print("\n✅ All tests completed successfully!")
