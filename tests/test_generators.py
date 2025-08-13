"""
Unit tests for ODE generators
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp
from src.generators.master_generator import MasterGenerator
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions

class TestMasterGenerator(unittest.TestCase):
    """Test cases for MasterGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = MasterGenerator(alpha=1.0, beta=1.0, n=1, M=0)
        self.z = sp.Symbol('z')
        
    def test_initialization(self):
        """Test generator initialization"""
        self.assertEqual(self.generator.alpha, 1.0)
        self.assertEqual(self.generator.beta, 1.0)
        self.assertEqual(self.generator.n, 1)
        self.assertEqual(self.generator.M, 0)
        
    def test_compute_omega(self):
        """Test omega computation"""
        omega = self.generator.compute_omega(1)
        expected = sp.pi / 2
        self.assertEqual(omega, expected)
        
    def test_generate_y_linear(self):
        """Test y generation with linear function"""
        f_z = self.z  # Linear function
        y = self.generator.generate_y(f_z)
        self.assertIsInstance(y, sp.Expr)
        
    def test_generate_y_prime(self):
        """Test y' generation"""
        f_z = self.z
        y_prime = self.generator.generate_y_prime(f_z)
        self.assertIsInstance(y_prime, sp.Expr)
        
    def test_generate_y_double_prime(self):
        """Test y'' generation"""
        f_z = self.z
        y_double_prime = self.generator.generate_y_double_prime(f_z)
        self.assertIsInstance(y_double_prime, sp.Expr)
        
    def test_initial_conditions(self):
        """Test initial conditions calculation"""
        f_z = self.z
        ic = self.generator.get_initial_conditions(f_z)
        self.assertIn('y(0)', ic)
        self.assertIn("y'(0)", ic)
        self.assertIn("y''(0)", ic)
        
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        self.assertTrue(self.generator.validate_parameters())
        
        # Invalid beta
        with self.assertRaises(ValueError):
            invalid_gen = MasterGenerator(beta=-1)
            invalid_gen.validate_parameters()
        
        # Invalid n
        with self.assertRaises(ValueError):
            invalid_gen = MasterGenerator(n=0)
            invalid_gen.validate_parameters()

class TestLinearGenerators(unittest.TestCase):
    """Test cases for linear generators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = LinearGeneratorFactory()
        self.z = sp.Symbol('z')
        self.f_z = self.z  # Simple linear function
        
    def test_generator_1(self):
        """Test linear generator 1: y'' + y = RHS"""
        result = self.factory.create_generator_1(self.f_z)
        self.assertEqual(result['type'], 'linear')
        self.assertEqual(result['order'], 2)
        self.assertEqual(result['generator_number'], 1)
        self.assertIn('ode', result)
        self.assertIn('solution', result)
        
    def test_generator_2(self):
        """Test linear generator 2: y'' + y' = RHS"""
        result = self.factory.create_generator_2(self.f_z)
        self.assertEqual(result['type'], 'linear')
        self.assertEqual(result['order'], 2)
        self.assertEqual(result['generator_number'], 2)
        
    def test_generator_3(self):
        """Test linear generator 3: y + y' = RHS"""
        result = self.factory.create_generator_3(self.f_z)
        self.assertEqual(result['type'], 'linear')
        self.assertEqual(result['order'], 1)
        self.assertEqual(result['generator_number'], 3)
        
    def test_pantograph_generator(self):
        """Test pantograph equation generator"""
        result = self.factory.create_generator_4(self.f_z, a=2)
        self.assertEqual(result['subtype'], 'pantograph')
        self.assertEqual(result['scaling_parameter'], 2)
        
    def test_create_by_number(self):
        """Test creating generator by number"""
        for i in range(1, 9):
            result = self.factory.create(i, self.f_z)
            self.assertEqual(result['generator_number'], i)
        
        # Test invalid number
        with self.assertRaises(ValueError):
            self.factory.create(0, self.f_z)
        
        with self.assertRaises(ValueError):
            self.factory.create(9, self.f_z)

class TestNonlinearGenerators(unittest.TestCase):
    """Test cases for nonlinear generators"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = NonlinearGeneratorFactory()
        self.z = sp.Symbol('z')
        self.f_z = self.z
        
    def test_generator_1(self):
        """Test nonlinear generator 1: (y'')^q + y = RHS"""
        result = self.factory.create_generator_1(self.f_z, q=2)
        self.assertEqual(result['type'], 'nonlinear')
        self.assertEqual(result['powers']['q'], 2)
        self.assertEqual(result['generator_number'], 1)
        
    def test_generator_2(self):
        """Test nonlinear generator 2: (y'')^q + (y')^v = RHS"""
        result = self.factory.create_generator_2(self.f_z, q=2, v=3)
        self.assertEqual(result['type'], 'nonlinear')
        self.assertEqual(result['powers']['q'], 2)
        self.assertEqual(result['powers']['v'], 3)
        
    def test_trigonometric_generator(self):
        """Test trigonometric nonlinear generator"""
        result = self.factory.create_generator_6(self.f_z)
        self.assertEqual(result['subtype'], 'trigonometric')
        
    def test_exponential_generator(self):
        """Test exponential nonlinear generator"""
        result = self.factory.create_generator_7(self.f_z)
        self.assertEqual(result['subtype'], 'exponential')
        
    def test_logarithmic_generator(self):
        """Test logarithmic nonlinear generator"""
        result = self.factory.create_generator_10(self.f_z)
        self.assertEqual(result['subtype'], 'logarithmic')

class TestBasicFunctions(unittest.TestCase):
    """Test cases for basic functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.functions = BasicFunctions()
        
    def test_get_function(self):
        """Test getting functions by name"""
        linear = self.functions.get_function('linear')
        self.assertEqual(linear, self.functions.z)
        
        sine = self.functions.get_function('sine')
        self.assertEqual(sine, sp.sin(self.functions.z))
        
    def test_function_names(self):
        """Test getting function names"""
        names = self.functions.get_function_names()
        self.assertIn('linear', names)
        self.assertIn('exponential', names)
        self.assertIn('sine', names)
        
    def test_evaluate_function(self):
        """Test function evaluation"""
        # Test sine(π/2) = 1
        value = self.functions.evaluate_function('sine', sp.pi/2)
        self.assertAlmostEqual(value, 1.0, places=10)
        
    def test_derivative(self):
        """Test derivative calculation"""
        # d/dz(z²) = 2z
        deriv = self.functions.get_derivative('quadratic')
        expected = 2 * self.functions.z
        self.assertEqual(deriv, expected)
        
    def test_composite_function(self):
        """Test composite function creation"""
        # sin(e^z)
        composite = self.functions.create_composite_function('sine', 'exponential')
        expected = sp.sin(sp.exp(self.functions.z))
        self.assertEqual(composite, expected)

class TestSpecialFunctions(unittest.TestCase):
    """Test cases for special functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.functions = SpecialFunctions()
        
    def test_airy_functions(self):
        """Test Airy functions"""
        ai = self.functions.get_function('airy_ai')
        self.assertIsInstance(ai, sp.Expr)
        
        bi = self.functions.get_function('airy_bi')
        self.assertIsInstance(bi, sp.Expr)
        
    def test_bessel_functions(self):
        """Test Bessel functions"""
        j0 = self.functions.get_function('bessel_j0')
        self.assertIsInstance(j0, sp.Expr)
        
        y1 = self.functions.get_function('bessel_y1')
        self.assertIsInstance(y1, sp.Expr)
        
    def test_get_by_category(self):
        """Test getting functions by category"""
        airy_funcs = self.functions.get_function_by_category('airy')
        self.assertEqual(len(airy_funcs), 4)
        
        bessel_funcs = self.functions.get_function_by_category('bessel')
        self.assertTrue(len(bessel_funcs) > 0)
        
    def test_series_expansion(self):
        """Test series expansion"""
        series = self.functions.get_series_expansion('exponential', order=3)
        self.assertIsInstance(series, sp.Expr)
        
    def test_recurrence_relation(self):
        """Test recurrence relation retrieval"""
        relation = self.functions.get_recurrence_relation('bessel_jn')
        self.assertIsNotNone(relation)
        self.assertIn('J_{n+1}', relation)

if __name__ == '__main__':
    unittest.main()