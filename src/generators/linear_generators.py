"""
Linear ODE Generators Implementation
Implements generators from Table 1 of the research paper
"""

import sympy as sp
from typing import Dict, Any, Optional
from .master_generator import MasterGenerator

class LinearGeneratorFactory:
    """
    Factory for creating linear ODE generators
    """
    
    def __init__(self):
        self.z = sp.Symbol('z')
        self.x = sp.Symbol('x')
        self.a = sp.Symbol('a')
        
    def create_generator_1(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 1: y''(x) + y(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Construct the ODE
        ode = y_double_prime + y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'order': 2,
            'generator_number': 1,
            'initial_conditions': initial_conditions,
            'description': "y''(x) + y(x) = RHS"
        }
    
    def create_generator_2(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 2: y''(x) + y'(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Construct the ODE
        ode = y_double_prime + y_prime
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'order': 2,
            'generator_number': 2,
            'initial_conditions': initial_conditions,
            'description': "y''(x) + y'(x) = RHS"
        }
    
    def create_generator_3(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 3: y(x) + y'(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        
        # Construct the ODE
        ode = y + y_prime
        
        # Get initial conditions
        initial_conditions = {
            'y(0)': sp.pi * params.get('M', 0)
        }
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'order': 1,
            'generator_number': 3,
            'initial_conditions': initial_conditions,
            'description': "y(x) + y'(x) = RHS"
        }
    
    def create_generator_4(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """
        Generator 4: y''(x) + y(x/a) - y(x) = RHS (Pantograph equation)
        
        Args:
            f_z: Function f(z)
            a: Scaling parameter
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Create y(x/a) by substitution
        y_scaled = y.subs(self.x, self.x/a)
        
        # Construct the ODE
        ode = y_double_prime + y_scaled - y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'subtype': 'pantograph',
            'order': 2,
            'generator_number': 4,
            'initial_conditions': initial_conditions,
            'scaling_parameter': a,
            'description': f"y''(x) + y(x/{a}) - y(x) = RHS"
        }
    
    def create_generator_5(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """
        Generator 5: y(x/a) + y'(x) = RHS
        
        Args:
            f_z: Function f(z)
            a: Scaling parameter
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        
        # Create y(x/a) by substitution
        y_scaled = y.subs(self.x, self.x/a)
        
        # Construct the ODE
        ode = y_scaled + y_prime
        
        # Get initial conditions
        initial_conditions = {
            'y(0)': sp.pi * params.get('M', 0)
        }
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'subtype': 'delay',
            'order': 1,
            'generator_number': 5,
            'initial_conditions': initial_conditions,
            'scaling_parameter': a,
            'description': f"y(x/{a}) + y'(x) = RHS"
        }
    
    def create_generator_6(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 6: y'''(x) + y(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_triple_prime = generator.generate_higher_derivative(f_z, 3)
        
        # Construct the ODE
        ode = y_triple_prime + y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        initial_conditions["y'''(0)"] = 0
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'order': 3,
            'generator_number': 6,
            'initial_conditions': initial_conditions,
            'description': "y'''(x) + y(x) = RHS"
        }
    
    def create_generator_7(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 7: y'''(x) + y'(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        y_triple_prime = generator.generate_higher_derivative(f_z, 3)
        
        # Construct the ODE
        ode = y_triple_prime + y_prime
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        initial_conditions["y'''(0)"] = 0
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'order': 3,
            'generator_number': 7,
            'initial_conditions': initial_conditions,
            'description': "y'''(x) + y'(x) = RHS"
        }
    
    def create_generator_8(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 8: y'''(x) + y''(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        y_triple_prime = generator.generate_higher_derivative(f_z, 3)
        
        # Construct the ODE
        ode = y_triple_prime + y_double_prime
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        initial_conditions["y'''(0)"] = 0
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'linear',
            'order': 3,
            'generator_number': 8,
            'initial_conditions': initial_conditions,
            'description': "y'''(x) + y''(x) = RHS"
        }
    
    def create(self, generator_number: int, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Create a linear generator by number
        
        Args:
            generator_number: Generator number (1-8)
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generators = {
            1: self.create_generator_1,
            2: self.create_generator_2,
            3: self.create_generator_3,
            4: self.create_generator_4,
            5: self.create_generator_5,
            6: self.create_generator_6,
            7: self.create_generator_7,
            8: self.create_generator_8
        }
        
        if generator_number not in generators:
            raise ValueError(f"Generator number must be between 1 and 8, got {generator_number}")
        
        return generators[generator_number](f_z, **params)