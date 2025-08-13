# tests/test_generators_improved.py - Enhanced Generator Tests
# ============================================================================
"""
Comprehensive unit tests for ODE generators with improved coverage
"""

import unittest
import pytest
import sys
import os
import numpy as np
import sympy as sp
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generators.master_generator import MasterGenerator, EnhancedMasterGenerator
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
from src.utils.validators import ParameterValidator
from src.utils.cache import CacheManager

class TestEnhancedMasterGenerator(unittest.TestCase):
    """Enhanced tests for Master Generator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = EnhancedMasterGenerator(alpha=1.0, beta=1.0, n=1, M=0)
        self.z = sp.Symbol('z')
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()
    
    def test_initialization_with_edge_cases(self):
        """Test initialization with edge case parameters"""
        # Test with maximum values
        gen = EnhancedMasterGenerator(alpha=100, beta=100, n=10, M=100)
        self.assertEqual(gen.alpha, 100)
        self.assertEqual(gen.beta, 100)
        self.assertEqual(gen.n, 10)
        self.assertEqual(gen.M, 100)
        
        # Test with minimum values
        gen = EnhancedMasterGenerator(alpha=-100, beta=0.1, n=1, M=-100)
        self.assertEqual(gen.alpha, -100)
        self.assertEqual(gen.beta, 0.1)
        self.assertEqual(gen.n, 1)
        self.assertEqual(gen.M, -100)
    
    def test_omega_caching(self):
        """Test that omega computation is cached"""
        # First call should compute
        omega1 = self.generator.compute_omega(1)
        
        # Second call should use cache
        omega2 = self.generator.compute_omega(1)
        
        self.assertEqual(omega1, omega2)
        
        # Check cache info
        cache_info = self.generator.compute_omega.cache_info()
        self.assertGreater(cache_info.hits, 0)
    
    def test_generate_y_with_various_functions(self):
        """Test y generation with different function types"""
        test_functions = [
            ('linear', self.z),
            ('quadratic', self.z**2),
            ('exponential', sp.exp(self.z)),
            ('sine', sp.sin(self.z)),
            ('bessel', sp.besselj(0, self.z))
        ]
        
        for name, f_z in test_functions:
            with self.subTest(function=name):
                y = self.generator.generate_y(f_z)
                self.assertIsInstance(y, sp.Expr)
                
                # Check that y is not trivial
                self.assertNotEqual(y, 0)
                self.assertNotEqual(y, sp.pi * self.generator.M)
    
    def test_optimized_generation(self):
        """Test optimized y generation with caching"""
        f_z = self.z**2
        
        # First generation
        y1 = self.generator.generate_y_optimized(f_z)
        
        # Second generation should use cache
        y2 = self.generator.generate_y_optimized(f_z)
        
        self.assertEqual(y1, y2)
    
    def test_derivatives_consistency(self):
        """Test consistency of derivative calculations"""
        f_z = sp.sin(self.z)
        
        y = self.generator.generate_y(f_z)
        y_prime = self.generator.generate_y_prime(f_z)
        y_double_prime = self.generator.generate_y_double_prime(f_z)
        
        # Check that derivatives are different
        self.assertNotEqual(y, y_prime)
        self.assertNotEqual(y_prime, y_double_prime)
        self.assertNotEqual(y, y_double_prime)
    
    def test_initial_conditions(self):
        """Test initial conditions calculation"""
        f_z = sp.exp(self.z)
        ic = self.generator.get_initial_conditions(f_z)
        
        self.assertIn('y(0)', ic)
        self.assertIn("y'(0)", ic)
        self.assertIn("y''(0)", ic)
        
        # Check y(0) = π*M when M=0
        if self.generator.M == 0:
            self.assertEqual(ic['y(0)'], 0)
    
    @patch('src.generators.master_generator.cache_manager')
    def test_caching_integration(self, mock_cache):
        """Test integration with cache manager"""
        mock_cache.get.return_value = None
        
        f_z = self.z
        y = self.generator.generate_y_optimized(f_z)
        
        # Check cache was accessed
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()

class TestLinearGeneratorsEnhanced(unittest.TestCase):
    """Enhanced tests for linear generators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = LinearGeneratorFactory()
        self.z = sp.Symbol('z')
        self.basic_functions = BasicFunctions()
    
    def test_all_linear_generators(self):
        """Test all 8 linear generator types"""
        for gen_num in range(1, 9):
            with self.subTest(generator=gen_num):
                f_z = self.z
                result = self.factory.create(gen_num, f_z)
                
                self.assertIsNotNone(result)
                self.assertEqual(result['type'], 'linear')
                self.assertEqual(result['generator_number'], gen_num)
                self.assertIn('ode', result)
                self.assertIn('solution', result)
                self.assertIn('initial_conditions', result)
    
    def test_pantograph_equation(self):
        """Test pantograph equation generator"""
        f_z = sp.sin(self.z)
        result = self.factory.create_generator_4(f_z, a=2)
        
        self.assertEqual(result['subtype'], 'pantograph')
        self.assertEqual(result['scaling_parameter'], 2)
        
        # Check that ODE contains y(x/a)
        ode_str = str(result['ode'])
        self.assertIn('x/2', ode_str)
    
    def test_higher_order_generators(self):
        """Test generators with third derivatives"""
        for gen_num in [6, 7, 8]:
            with self.subTest(generator=gen_num):
                f_z = sp.exp(self.z)
                result = self.factory.create(gen_num, f_z)
                
                self.assertEqual(result['order'], 3)
                self.assertIn("y'''(0)", result['initial_conditions'])
    
    def test_with_special_functions(self):
        """Test linear generators with special functions"""
        special_funcs = SpecialFunctions()
        
        test_functions = [
            'airy_ai',
            'bessel_j0',
            'gamma',
            'erf'
        ]
        
        for func_name in test_functions:
            with self.subTest(function=func_name):
                f_z = special_funcs.get_function(func_name)
                result = self.factory.create(1, f_z)
                
                self.assertIsNotNone(result)
                self.assertEqual(result['type'], 'linear')
    
    def test_invalid_generator_number(self):
        """Test error handling for invalid generator numbers"""
        with self.assertRaises(ValueError):
            self.factory.create(0, self.z)
        
        with self.assertRaises(ValueError):
            self.factory.create(9, self.z)

class TestNonlinearGeneratorsEnhanced(unittest.TestCase):
    """Enhanced tests for nonlinear generators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = NonlinearGeneratorFactory()
        self.z = sp.Symbol('z')
    
    def test_all_nonlinear_generators(self):
        """Test all 10 nonlinear generator types"""
        for gen_num in range(1, 11):
            with self.subTest(generator=gen_num):
                f_z = self.z
                
                # Add required parameters for specific generators
                kwargs = {}
                if gen_num in [1, 2, 4]:
                    kwargs['q'] = 2
                if gen_num in [2, 3, 5]:
                    kwargs['v'] = 3
                if gen_num in [4, 5, 9, 10]:
                    kwargs['a'] = 2
                
                result = self.factory.create(gen_num, f_z, **kwargs)
                
                self.assertIsNotNone(result)
                self.assertEqual(result['type'], 'nonlinear')
                self.assertEqual(result['generator_number'], gen_num)
    
    def test_power_nonlinearity(self):
        """Test generators with power nonlinearity"""
        f_z = sp.sin(self.z)
        
        # Test different power values
        for q in [2, 3, 4, 5]:
            with self.subTest(q=q):
                result = self.factory.create_generator_1(f_z, q=q)
                
                self.assertEqual(result['powers']['q'], q)
                
                # Check that ODE contains y''^q
                ode_str = str(result['ode'])
                self.assertIn('**', ode_str)
    
    def test_trigonometric_nonlinearity(self):
        """Test generators with trigonometric nonlinearity"""
        f_z = self.z
        result = self.factory.create_generator_6(f_z)
        
        self.assertEqual(result['subtype'], 'trigonometric')
        
        # Check that ODE contains sin
        ode_str = str(result['ode'])
        self.assertIn('sin', ode_str)
    
    def test_exponential_nonlinearity(self):
        """Test generators with exponential nonlinearity"""
        f_z = self.z**2
        
        # Test generator 7: e^(y'') + e^(y')
        result = self.factory.create_generator_7(f_z)
        self.assertEqual(result['subtype'], 'exponential')
        
        # Test generator 8: y + e^(y')
        result = self.factory.create_generator_8(f_z)
        self.assertEqual(result['subtype'], 'exponential')
        
        # Check that ODE contains exp
        ode_str = str(result['ode'])
        self.assertIn('exp', ode_str)
    
    def test_logarithmic_nonlinearity(self):
        """Test generator with logarithmic nonlinearity"""
        f_z = sp.cos(self.z)
        result = self.factory.create_generator_10(f_z, a=3)
        
        self.assertEqual(result['subtype'], 'logarithmic')
        self.assertEqual(result['scaling_parameter'], 3)
        
        # Check that ODE contains log
        ode_str = str(result['ode'])
        self.assertIn('log', ode_str)

class TestParameterValidation(unittest.TestCase):
    """Tests for parameter validation"""
    
    def test_validate_alpha(self):
        """Test alpha parameter validation"""
        # Valid values
        self.assertEqual(ParameterValidator.validate_alpha(0), 0.0)
        self.assertEqual(ParameterValidator.validate_alpha(50), 50.0)
        self.assertEqual(ParameterValidator.validate_alpha(-50), -50.0)
        
        # Invalid values
        with self.assertRaises(ValueError):
            ParameterValidator.validate_alpha(150)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_alpha(-150)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_alpha("not a number")
    
    def test_validate_beta(self):
        """Test beta parameter validation"""
        # Valid values
        self.assertEqual(ParameterValidator.validate_beta(0.1), 0.1)
        self.assertEqual(ParameterValidator.validate_beta(50), 50.0)
        
        # Invalid values
        with self.assertRaises(ValueError):
            ParameterValidator.validate_beta(0)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_beta(-1)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_beta(150)
    
    def test_validate_n(self):
        """Test n parameter validation"""
        # Valid values
        self.assertEqual(ParameterValidator.validate_n(1), 1)
        self.assertEqual(ParameterValidator.validate_n(5), 5)
        
        # Invalid values
        with self.assertRaises(ValueError):
            ParameterValidator.validate_n(0)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_n(15)
        
        # Type conversion
        self.assertEqual(ParameterValidator.validate_n(3.7), 3)
    
    def test_validate_parameters_dict(self):
        """Test validation of parameter dictionary"""
        params = {
            'alpha': 10,
            'beta': 2,
            'n': 3,
            'M': 5,
            'q': 3,
            'v': 4,
            'a': 2.5
        }
        
        validated = ParameterValidator.validate_parameters(params)
        
        self.assertEqual(validated['alpha'], 10.0)
        self.assertEqual(validated['beta'], 2.0)
        self.assertEqual(validated['n'], 3)
        self.assertEqual(validated['M'], 5.0)
        self.assertEqual(validated['q'], 3)
        self.assertEqual(validated['v'], 4)
        self.assertEqual(validated['a'], 2.5)
    
    def test_sanitize_parameters(self):
        """Test parameter sanitization"""
        params = {
            'alpha': 200,  # Too large
            'beta': -1,    # Negative
            'n': 20,       # Too large
            'M': 150       # Too large
        }
        
        sanitized = ParameterValidator.sanitize_parameters(params)
        
        self.assertEqual(sanitized['alpha'], 100.0)  # Clamped to max
        self.assertEqual(sanitized['beta'], 0.1)     # Clamped to min
        self.assertEqual(sanitized['n'], 10)         # Clamped to max
        self.assertEqual(sanitized['M'], 100.0)      # Clamped to max

class TestCacheManager(unittest.TestCase):
    """Tests for cache manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = CacheManager(max_memory_size=10)
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        # Set value
        self.cache.set('key1', 'value1')
        
        # Get value
        value = self.cache.get('key1')
        self.assertEqual(value, 'value1')
        
        # Get non-existent key
        value = self.cache.get('non_existent')
        self.assertIsNone(value)
        
        # Delete value
        self.cache.delete('key1')
        value = self.cache.get('key1')
        self.assertIsNone(value)
    
    def test_cache_expiration(self):
        """Test cache size limits"""
        # Fill cache beyond max size
        for i in range(15):
            self.cache.set(f'key{i}', f'value{i}')
        
        # Check that cache size is limited
        self.assertLessEqual(len(self.cache.memory_cache), 10)
    
    def test_lru_eviction(self):
        """Test least recently used eviction"""
        # Fill cache to max
        for i in range(10):
            self.cache.set(f'key{i}', f'value{i}')
        
        # Access some keys to update access times
        self.cache.get('key5')
        self.cache.get('key6')
        
        # Add new key, should evict least recently used
        self.cache.set('new_key', 'new_value')
        
        # Check that new key exists
        self.assertEqual(self.cache.get('new_key'), 'new_value')
        
        # Check cache size
        self.assertLessEqual(len(self.cache.memory_cache), 10)
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        # Add some data
        for i in range(5):
            self.cache.set(f'key{i}', f'value{i}')
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['memory_size'], 5)
        self.assertEqual(stats['max_memory_size'], 10)
        self.assertFalse(stats['redis_connected'])
