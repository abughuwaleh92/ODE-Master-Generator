"""
Nonlinear ODE Generators Implementation
Implements generators from Table 2 of the research paper
"""

import sympy as sp
from typing import Dict, Any, Optional
from .master_generator import MasterGenerator

class NonlinearGeneratorFactory:
    """
    Factory for creating nonlinear ODE generators
    """
    
    def __init__(self):
        self.z = sp.Symbol('z')
        self.x = sp.Symbol('x')
        self.a = sp.Symbol('a')
        
    def create_generator_1(self, f_z: sp.Expr, q: int = 2, **params) -> Dict[str, Any]:
        """
        Generator 1: (y''(x))^q + y(x) = RHS
        
        Args:
            f_z: Function f(z)
            q: Power for y''
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Construct the nonlinear ODE
        ode = y_double_prime**q + y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'order': 2,
            'generator_number': 1,
            'powers': {'q': q},
            'initial_conditions': initial_conditions,
            'description': f"(y''(x))^{q} + y(x) = RHS"
        }
    
    def create_generator_2(self, f_z: sp.Expr, q: int = 2, v: int = 3, **params) -> Dict[str, Any]:
        """
        Generator 2: (y''(x))^q + (y'(x))^v = RHS
        
        Args:
            f_z: Function f(z)
            q: Power for y''
            v: Power for y'
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Construct the nonlinear ODE
        ode = y_double_prime**q + y_prime**v
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'order': 2,
            'generator_number': 2,
            'powers': {'q': q, 'v': v},
            'initial_conditions': initial_conditions,
            'description': f"(y''(x))^{q} + (y'(x))^{v} = RHS"
        }
    
    def create_generator_3(self, f_z: sp.Expr, v: int = 3, **params) -> Dict[str, Any]:
        """
        Generator 3: y(x) + (y'(x))^v = RHS
        
        Args:
            f_z: Function f(z)
            v: Power for y'
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        
        # Construct the nonlinear ODE
        ode = y + y_prime**v
        
        # Get initial conditions
        initial_conditions = {
            'y(0)': sp.pi * params.get('M', 0)
        }
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'order': 1,
            'generator_number': 3,
            'powers': {'v': v},
            'initial_conditions': initial_conditions,
            'description': f"y(x) + (y'(x))^{v} = RHS"
        }
    
    def create_generator_4(self, f_z: sp.Expr, q: int = 2, a: float = 2, **params) -> Dict[str, Any]:
        """
        Generator 4: (y''(x))^q + y(x/a) - y(x) = RHS
        
        Args:
            f_z: Function f(z)
            q: Power for y''
            a: Scaling parameter
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Create y(x/a) by substitution
        y_scaled = y.subs(self.x, self.x/a)
        
        # Construct the nonlinear ODE
        ode = y_double_prime**q + y_scaled - y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'pantograph',
            'order': 2,
            'generator_number': 4,
            'powers': {'q': q},
            'scaling_parameter': a,
            'initial_conditions': initial_conditions,
            'description': f"(y''(x))^{q} + y(x/{a}) - y(x) = RHS"
        }
    
    def create_generator_5(self, f_z: sp.Expr, v: int = 3, a: float = 2, **params) -> Dict[str, Any]:
        """
        Generator 5: y(x/a) + (y'(x))^v = RHS
        
        Args:
            f_z: Function f(z)
            v: Power for y'
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
        
        # Construct the nonlinear ODE
        ode = y_scaled + y_prime**v
        
        # Get initial conditions
        initial_conditions = {
            'y(0)': sp.pi * params.get('M', 0)
        }
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'delay',
            'order': 1,
            'generator_number': 5,
            'powers': {'v': v},
            'scaling_parameter': a,
            'initial_conditions': initial_conditions,
            'description': f"y(x/{a}) + (y'(x))^{v} = RHS"
        }
    
    def create_generator_6(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 6: sin(y''(x)) + y(x) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Construct the nonlinear ODE with trigonometric function
        ode = sp.sin(y_double_prime) + y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'trigonometric',
            'order': 2,
            'generator_number': 6,
            'initial_conditions': initial_conditions,
            'description': "sin(y''(x)) + y(x) = RHS"
        }
    
    def create_generator_7(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 7: e^(y''(x)) + e^(y'(x)) = RHS
        
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
        
        # Construct the nonlinear ODE with exponential functions
        ode = sp.exp(y_double_prime) + sp.exp(y_prime)
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'exponential',
            'order': 2,
            'generator_number': 7,
            'initial_conditions': initial_conditions,
            'description': "e^(y''(x)) + e^(y'(x)) = RHS"
        }
    
    def create_generator_8(self, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Generator 8: y(x) + e^(y'(x)) = RHS
        
        Args:
            f_z: Function f(z)
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_prime = generator.generate_y_prime(f_z)
        
        # Construct the nonlinear ODE
        ode = y + sp.exp(y_prime)
        
        # Get initial conditions
        initial_conditions = {
            'y(0)': sp.pi * params.get('M', 0)
        }
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'exponential',
            'order': 1,
            'generator_number': 8,
            'initial_conditions': initial_conditions,
            'description': "y(x) + e^(y'(x)) = RHS"
        }
    
    def create_generator_9(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """
        Generator 9: e^(y''(x)) + y(x/a) - y(x) = RHS
        
        Args:
            f_z: Function f(z)
            a: Scaling parameter
            **params: Generator parameters
            
        Returns:
            ODE dictionary with solution
        """
        generator = MasterGenerator(**params)
        
        y = generator.generate_y(f_z)
        y_double_prime = generator.generate_y_double_prime(f_z)
        
        # Create y(x/a) by substitution
        y_scaled = y.subs(self.x, self.x/a)
        
        # Construct the nonlinear ODE
        ode = sp.exp(y_double_prime) + y_scaled - y
        
        # Get initial conditions
        initial_conditions = generator.get_initial_conditions(f_z)
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'exponential-pantograph',
            'order': 2,
            'generator_number': 9,
            'scaling_parameter': a,
            'initial_conditions': initial_conditions,
            'description': f"e^(y''(x)) + y(x/{a}) - y(x) = RHS"
        }
    
    def create_generator_10(self, f_z: sp.Expr, a: float = 2, **params) -> Dict[str, Any]:
        """
        Generator 10: y(x/a) + ln(y'(x)) = RHS
        
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
        
        # Construct the nonlinear ODE
        ode = y_scaled + sp.log(y_prime)
        
        # Get initial conditions
        initial_conditions = {
            'y(0)': sp.pi * params.get('M', 0)
        }
        
        return {
            'ode': ode,
            'solution': y,
            'type': 'nonlinear',
            'subtype': 'logarithmic',
            'order': 1,
            'generator_number': 10,
            'scaling_parameter': a,
            'initial_conditions': initial_conditions,
            'description': f"y(x/{a}) + ln(y'(x)) = RHS"
        }
    
    def create(self, generator_number: int, f_z: sp.Expr, **params) -> Dict[str, Any]:
        """
        Create a nonlinear generator by number
        
        Args:
            generator_number: Generator number (1-10)
            f_z: Function f(z)
            **params: Generator parameters including q, v, a as needed
            
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
            8: self.create_generator_8,
            9: self.create_generator_9,
            10: self.create_generator_10
        }
        
        if generator_number not in generators:
            raise ValueError(f"Generator number must be between 1 and 10, got {generator_number}")
        
        return generators[generator_number](f_z, **params)