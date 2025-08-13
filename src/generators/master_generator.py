"""
Complete Master Generator Implementation with ALL Generators
Includes base implementation and complete generator classes
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from functools import lru_cache

class MasterGenerator:
    """
    Core Master Generator implementation based on Theorem 4.1
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                 n: int = 1, M: float = 0.0):
        """
        Initialize Master Generator with parameters
        
        Args:
            alpha: Parameter α from the paper
            beta: Parameter β from the paper (must be positive)
            n: Order parameter
            M: Constant term
        """
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if n < 1:
            raise ValueError("n must be at least 1")
            
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.M = M
        
        # Define symbolic variables
        self.x = sp.Symbol('x', real=True)
        self.z = sp.Symbol('z')
        self.y = sp.Function('y')
        
    def compute_omega(self, s: int) -> sp.Expr:
        """Compute ω(s) = (2s-1)π/(2n)"""
        return (2 * s - 1) * sp.pi / (2 * self.n)
    
    def psi_function(self, f_z: sp.Expr, x_val: sp.Symbol, omega: sp.Expr) -> sp.Expr:
        """ψ(α,ω,x) = f(α + β·e^(i x cos ω − x sin ω))"""
        exponent = sp.I * x_val * sp.cos(omega) - x_val * sp.sin(omega)
        z_val = self.alpha + self.beta * sp.exp(exponent)
        return f_z.subs(self.z, z_val)
    
    def phi_function(self, f_z: sp.Expr, x_val: sp.Symbol, omega: sp.Expr) -> sp.Expr:
        """φ(α,ω,x) = f(α + β·e^(−i x cos ω − x sin ω))"""
        exponent = -sp.I * x_val * sp.cos(omega) - x_val * sp.sin(omega)
        z_val = self.alpha + self.beta * sp.exp(exponent)
        return f_z.subs(self.z, z_val)
    
    def generate_y(self, f_z: sp.Expr) -> sp.Expr:
        """Generate y(x) solution using Theorem 4.1"""
        result = 0
        for s in range(1, self.n + 1):
            omega = self.compute_omega(s)
            psi = self.psi_function(f_z, self.x, omega)
            phi = self.phi_function(f_z, self.x, omega)
            f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
            result += 2 * f_alpha_beta - (psi + phi)
        
        return sp.pi / (2 * self.n) * result + sp.pi * self.M
    
    def generate_y_prime(self, f_z: sp.Expr) -> sp.Expr:
        """Generate y'(x)"""
        y = self.generate_y(f_z)
        return sp.diff(y, self.x)
    
    def generate_y_double_prime(self, f_z: sp.Expr) -> sp.Expr:
        """Generate y''(x)"""
        y = self.generate_y(f_z)
        return sp.diff(y, self.x, 2)
    
    def generate_higher_derivative(self, f_z: sp.Expr, order: int) -> sp.Expr:
        """Generate y^(n)(x) - nth derivative"""
        y = self.generate_y(f_z)
        return sp.diff(y, self.x, order)
    
    def get_initial_conditions(self, f_z: sp.Expr) -> Dict[str, sp.Expr]:
        """Calculate initial conditions at x=0"""
        y = self.generate_y(f_z)
        y_prime = self.generate_y_prime(f_z)
        y_double_prime = self.generate_y_double_prime(f_z)
        
        return {
            'y(0)': y.subs(self.x, 0),
            "y'(0)": y_prime.subs(self.x, 0),
            "y''(0)": y_double_prime.subs(self.x, 0)
        }
    
    def validate_parameters(self) -> bool:
        """Validate generator parameters"""
        if self.beta <= 0:
            raise ValueError("Beta must be positive")
        if self.n < 1:
            raise ValueError("n must be at least 1")
        return True


class EnhancedMasterGenerator(MasterGenerator):
    """Enhanced Master Generator with caching and optimization"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                 n: int = 1, M: float = 0.0):
        super().__init__(alpha, beta, n, M)
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def compute_omega(self, s: int) -> sp.Expr:
        """Cached version of omega computation"""
        return super().compute_omega(s)
    
    def generate_y_optimized(self, f_z: sp.Expr) -> sp.Expr:
        """Optimized y generation with caching"""
        cache_key = (str(f_z), self.alpha, self.beta, self.n, self.M)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = super().generate_y(f_z)
        self._cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        self.compute_omega.cache_clear()


class CompleteMasterGenerator:
    """
    Complete implementation with explicit RHS for all generators
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, 
                 n: int = 1, M: float = 0.0):
        """Initialize Master Generator with parameters"""
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if n < 1:
            raise ValueError("n must be at least 1")
            
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.M = M
        
        # Define symbolic variables
        self.x = sp.Symbol('x', real=True)
        self.z = sp.Symbol('z')
        self.y = sp.Function('y')
        
    def compute_omega(self, s: int) -> sp.Expr:
        """Compute ω(s) = (2s-1)π/(2n)"""
        return (2 * s - 1) * sp.pi / (2 * self.n)
    
    def psi_function(self, f_z: sp.Expr, x_val: sp.Symbol, omega: sp.Expr) -> sp.Expr:
        """ψ(α,ω,x) = f(α + β·e^(i x cos ω − x sin ω))"""
        exponent = sp.I * x_val * sp.cos(omega) - x_val * sp.sin(omega)
        z_val = self.alpha + self.beta * sp.exp(exponent)
        return f_z.subs(self.z, z_val)
    
    def phi_function(self, f_z: sp.Expr, x_val: sp.Symbol, omega: sp.Expr) -> sp.Expr:
        """φ(α,ω,x) = f(α + β·e^(−i x cos ω − x sin ω))"""
        exponent = -sp.I * x_val * sp.cos(omega) - x_val * sp.sin(omega)
        z_val = self.alpha + self.beta * sp.exp(exponent)
        return f_z.subs(self.z, z_val)
    
    def generate_solution_y(self, f_z: sp.Expr) -> sp.Expr:
        """
        Generate y(x) solution using Theorem 4.1
        This is the known solution to the ODE
        """
        result = 0
        for s in range(1, self.n + 1):
            omega = self.compute_omega(s)
            psi = self.psi_function(f_z, self.x, omega)
            phi = self.phi_function(f_z, self.x, omega)
            f_alpha_beta = f_z.subs(self.z, self.alpha + self.beta)
            result += 2 * f_alpha_beta - (psi + phi)
        
        return sp.pi / (2 * self.n) * result + sp.pi * self.M
    
    def compute_derivatives_at_exp(self, f_z: sp.Expr, max_order: int = 3) -> Dict[int, sp.Expr]:
        """Compute derivatives of f at α + βe^(-x) up to max_order"""
        exp_arg = self.alpha + self.beta * sp.exp(-self.x)
        derivatives = {}
        
        # 0th derivative (function itself)
        derivatives[0] = f_z.subs(self.z, exp_arg)
        
        # Higher derivatives
        current_deriv = f_z
        for order in range(1, max_order + 1):
            current_deriv = sp.diff(current_deriv, self.z)
            derivatives[order] = current_deriv.subs(self.z, exp_arg)
        
        return derivatives
    
    def compute_derivatives_at_scaled(self, f_z: sp.Expr, a: float, max_order: int = 3) -> Dict[int, sp.Expr]:
        """Compute derivatives of f at α + βe^(-x/a) for pantograph-type equations"""
        exp_arg_scaled = self.alpha + self.beta * sp.exp(-self.x/a)
        derivatives = {}
        
        derivatives[0] = f_z.subs(self.z, exp_arg_scaled)
        
        current_deriv = f_z
        for order in range(1, max_order + 1):
            current_deriv = sp.diff(current_deriv, self.z)
            derivatives[order] = current_deriv.subs(self.z, exp_arg_scaled)
        
        return derivatives


class CompleteLinearGeneratorFactory:
    """
    Complete Linear Generator Factory with all 8 generators and explicit RHS
    """
    
    def __init__(self):
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Function('y')
        self.z = sp.Symbol('z')
        
    def create_generator_1(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Generator 1: y''(x) + y(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        # Compute explicit RHS
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 2)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M 
                      - generator.beta * sp.exp(-self.x) * derivs[1]
                      - generator.beta**2 * sp.exp(-2*self.x) * derivs[2])
        
        lhs = self.y(self.x).diff(self.x, 2) + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 2,
            'generator_number': 1,
            'description': 'y\'\'(x) + y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_2(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Generator 2: y''(x) + y'(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 2)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M
                      - generator.beta**2 * sp.exp(-2*self.x) * derivs[2])
        
        lhs = self.y(self.x).diff(self.x, 2) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 2,
            'generator_number': 2,
            'description': 'y\'\'(x) + y\'(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_3(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Generator 3: y(x) + y'(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        
        rhs = sp.pi * (f_alpha_beta + generator.M)
        
        lhs = self.y(self.x) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 1,
            'generator_number': 3,
            'description': 'y(x) + y\'(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_4(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """Generator 4: y''(x) + y(x/a) - y(x) = RHS (Pantograph equation)"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs_exp = generator.compute_derivatives_at_exp(f_z, 2)
        derivs_scaled = generator.compute_derivatives_at_scaled(f_z, a, 2)
        
        rhs = sp.pi * (f_alpha_beta - derivs_exp[0] + derivs_scaled[0] - derivs_exp[0] + generator.M
                      - generator.beta * sp.exp(-self.x) * derivs_exp[1]
                      - generator.beta**2 * sp.exp(-2*self.x) * derivs_exp[2])
        
        lhs = self.y(self.x).diff(self.x, 2) + self.y(self.x/a) - self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
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
            'description': f'y\'\'(x) + y(x/{a}) - y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_5(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """Generator 5: y(x/a) + y'(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs_scaled = generator.compute_derivatives_at_scaled(f_z, a, 1)
        
        rhs = sp.pi * (f_alpha_beta + derivs_scaled[0] - f_alpha_beta + generator.M)
        
        lhs = self.y(self.x/a) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
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
            'description': f'y(x/{a}) + y\'(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_6(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Generator 6: y'''(x) + y(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 3)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M
                      - generator.beta * sp.exp(-self.x) * derivs[1]
                      - generator.beta**2 * sp.exp(-2*self.x) * derivs[2]
                      - generator.beta**3 * sp.exp(-3*self.x) * derivs[3])
        
        lhs = self.y(self.x).diff(self.x, 3) + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 3,
            'generator_number': 6,
            'description': 'y\'\'\'(x) + y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_7(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Generator 7: y'''(x) + y'(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 3)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M
                      - generator.beta**3 * sp.exp(-3*self.x) * derivs[3])
        
        lhs = self.y(self.x).diff(self.x, 3) + self.y(self.x).diff(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 3,
            'generator_number': 7,
            'description': 'y\'\'\'(x) + y\'(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_8(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Generator 8: y'''(x) + y''(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 3)
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M
                      - generator.beta * sp.exp(-self.x) * derivs[1]
                      - generator.beta**3 * sp.exp(-3*self.x) * derivs[3])
        
        lhs = self.y(self.x).diff(self.x, 3) + self.y(self.x).diff(self.x, 2)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'linear',
            'order': 3,
            'generator_number': 8,
            'description': 'y\'\'\'(x) + y\'\'(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create(self, generator_number: int, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Create a linear generator by number (1-8)"""
        generators = {
            1: self.create_generator_1,
            2: self.create_generator_2,
            3: self.create_generator_3,
            4: self.create_generator_4,
            5: self.create_generator_5,
            6: self.create_generator_6,
            7: self.create_generator_7,
            8: self.create_generator_8,
        }
        
        if generator_number not in generators:
            raise ValueError(f"Linear generator number must be 1-8, got {generator_number}")
        
        return generators[generator_number](f_z, **params)


class CompleteNonlinearGeneratorFactory:
    """
    Complete Nonlinear Generator Factory with all 10 generators and explicit RHS
    """
    
    def __init__(self):
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Function('y')
        self.z = sp.Symbol('z')
        
    def create_generator_1(self, f_z: sp.Expr, q: int = 2, **params) -> Dict[str, Any]:
        """Nonlinear Generator 1: (y''(x))^q + y(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_expr = -generator.beta * sp.exp(-self.x) * derivs[1] - generator.beta**2 * sp.exp(-2*self.x) * derivs[2]
        y_expr = f_alpha_beta - derivs[0] + generator.M
        
        rhs = sp.pi**q * y_double_prime_expr**q + sp.pi * y_expr
        
        lhs = self.y(self.x).diff(self.x, 2)**q + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'order': 2,
            'generator_number': 1,
            'powers': {'q': q},
            'description': f'(y\'\'(x))^{q} + y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_2(self, f_z: sp.Expr, q: int = 2, v: int = 3, **params) -> Dict[str, Any]:
        """Nonlinear Generator 2: (y''(x))^q + (y'(x))^v = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        derivs = generator.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_expr = -generator.beta * sp.exp(-self.x) * derivs[1] - generator.beta**2 * sp.exp(-2*self.x) * derivs[2]
        y_prime_expr = generator.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.pi**q * y_double_prime_expr**q + sp.pi**v * y_prime_expr**v
        
        lhs = self.y(self.x).diff(self.x, 2)**q + self.y(self.x).diff(self.x)**v
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'order': 2,
            'generator_number': 2,
            'powers': {'q': q, 'v': v},
            'description': f'(y\'\'(x))^{q} + (y\'(x))^{v} = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_3(self, f_z: sp.Expr, v: int = 3, **params) -> Dict[str, Any]:
        """Nonlinear Generator 3: y(x) + (y'(x))^v = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 1)
        
        y_prime_expr = generator.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M) + sp.pi**v * y_prime_expr**v
        
        lhs = self.y(self.x) + self.y(self.x).diff(self.x)**v
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'order': 1,
            'generator_number': 3,
            'powers': {'v': v},
            'description': f'y(x) + (y\'(x))^{v} = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_4(self, f_z: sp.Expr, q: int = 2, a: float = 2, **params) -> Dict[str, Any]:
        """Nonlinear Generator 4: (y''(x))^q + y(x/a) - y(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs_exp = generator.compute_derivatives_at_exp(f_z, 2)
        derivs_scaled = generator.compute_derivatives_at_scaled(f_z, a, 2)
        
        y_double_prime_expr = -generator.beta * sp.exp(-self.x) * derivs_exp[1] - generator.beta**2 * sp.exp(-2*self.x) * derivs_exp[2]
        
        rhs = sp.pi**q * y_double_prime_expr**q + sp.pi * (derivs_scaled[0] - derivs_exp[0] + generator.M)
        
        lhs = self.y(self.x).diff(self.x, 2)**q + self.y(self.x/a) - self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
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
            'description': f'(y\'\'(x))^{q} + y(x/{a}) - y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_5(self, f_z: sp.Expr, v: int = 3, a: float = 2, **params) -> Dict[str, Any]:
        """Nonlinear Generator 5: y(x/a) + (y'(x))^v = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        derivs_exp = generator.compute_derivatives_at_exp(f_z, 1)
        derivs_scaled = generator.compute_derivatives_at_scaled(f_z, a, 1)
        
        y_prime_expr = generator.beta * sp.exp(-self.x) * derivs_exp[1]
        
        rhs = sp.pi * (derivs_scaled[0] + generator.M) + sp.pi**v * y_prime_expr**v
        
        lhs = self.y(self.x/a) + self.y(self.x).diff(self.x)**v
        ode = sp.Eq(lhs, rhs)
        
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
            'description': f'y(x/{a}) + (y\'(x))^{v} = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_6(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Nonlinear Generator 6: sin(y''(x)) + y(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_term = sp.pi * (-generator.beta * sp.exp(-self.x) * derivs[1] 
                                      - generator.beta**2 * sp.exp(-2*self.x) * derivs[2])
        y_term = sp.pi * (f_alpha_beta - derivs[0] + generator.M)
        
        rhs = sp.sin(y_double_prime_term) + y_term
        
        lhs = sp.sin(self.y(self.x).diff(self.x, 2)) + self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'trigonometric',
            'order': 2,
            'generator_number': 6,
            'description': 'sin(y\'\'(x)) + y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_7(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Nonlinear Generator 7: e^(y''(x)) + e^(y'(x)) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        derivs = generator.compute_derivatives_at_exp(f_z, 2)
        
        y_double_prime_term = sp.pi * (-generator.beta * sp.exp(-self.x) * derivs[1] 
                                      - generator.beta**2 * sp.exp(-2*self.x) * derivs[2])
        y_prime_term = sp.pi * generator.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.exp(y_double_prime_term) + sp.exp(y_prime_term)
        
        lhs = sp.exp(self.y(self.x).diff(self.x, 2)) + sp.exp(self.y(self.x).diff(self.x))
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'exponential',
            'order': 2,
            'generator_number': 7,
            'description': 'e^(y\'\'(x)) + e^(y\'(x)) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_8(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Nonlinear Generator 8: y(x) + e^(y'(x)) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        f_alpha_beta = f_z.subs(generator.z, generator.alpha + generator.beta)
        derivs = generator.compute_derivatives_at_exp(f_z, 1)
        
        y_prime_term = sp.pi * generator.beta * sp.exp(-self.x) * derivs[1]
        
        rhs = sp.pi * (f_alpha_beta - derivs[0] + generator.M) + sp.exp(y_prime_term)
        
        lhs = self.y(self.x) + sp.exp(self.y(self.x).diff(self.x))
        ode = sp.Eq(lhs, rhs)
        
        return {
            'ode': ode,
            'lhs': lhs,
            'rhs': rhs,
            'solution': solution,
            'type': 'nonlinear',
            'subtype': 'exponential',
            'order': 1,
            'generator_number': 8,
            'description': 'y(x) + e^(y\'(x)) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_9(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """Nonlinear Generator 9: e^(y''(x)) + y(x/a) - y(x) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        derivs_exp = generator.compute_derivatives_at_exp(f_z, 2)
        derivs_scaled = generator.compute_derivatives_at_scaled(f_z, a, 0)
        
        y_double_prime_term = sp.pi * (-generator.beta * sp.exp(-self.x) * derivs_exp[1] 
                                      - generator.beta**2 * sp.exp(-2*self.x) * derivs_exp[2])
        
        rhs = sp.exp(y_double_prime_term) + sp.pi * (derivs_scaled[0] - derivs_exp[0] + generator.M)
        
        lhs = sp.exp(self.y(self.x).diff(self.x, 2)) + self.y(self.x/a) - self.y(self.x)
        ode = sp.Eq(lhs, rhs)
        
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
            'description': f'e^(y\'\'(x)) + y(x/{a}) - y(x) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create_generator_10(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """Nonlinear Generator 10: y(x/a) + ln(y'(x)) = RHS"""
        generator = CompleteMasterGenerator(**params)
        
        solution = generator.generate_solution_y(f_z)
        
        derivs_exp = generator.compute_derivatives_at_exp(f_z, 1)
        derivs_scaled = generator.compute_derivatives_at_scaled(f_z, a, 0)
        
        y_prime_term = sp.pi * generator.beta * sp.exp(-self.x) * derivs_exp[1]
        
        rhs = sp.pi * (derivs_scaled[0] + generator.M) + sp.log(y_prime_term)
        
        lhs = self.y(self.x/a) + sp.log(self.y(self.x).diff(self.x))
        ode = sp.Eq(lhs, rhs)
        
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
            'description': f'y(x/{a}) + ln(y\'(x)) = RHS',
            'initial_conditions': {'y(0)': sp.pi * params.get('M', 0)}
        }
    
    def create(self, generator_number: int, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """Create a nonlinear generator by number (1-10)"""
        generators = {
            1: self.create_generator_1,
            2: self.create_generator_2,
            3: self.create_generator_3,
            4: self.create_generator_4,
            5: self.create_generator_5,
            6: self.create_generator_6,
            7: self.create_generator_7,
            8: self.create_generator_8,
            9: self.create_generator_9,
            10: self.create_generator_10,
        }
        
        if generator_number not in generators:
            raise ValueError(f"Nonlinear generator number must be 1-10, got {generator_number}")
        
        return generators[generator_number](f_z, **params)
