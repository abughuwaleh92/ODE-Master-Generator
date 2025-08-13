# src/generators/ode_classifier.py
"""
ODE Classification and Physical Applications System
"""

import sympy as sp
from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class PhysicalApplication:
    """Represents a physical application of an ODE"""
    field: str  # Physics, Chemistry, Biology, etc.
    name: str  # Specific application name
    description: str
    parameters_meaning: Dict[str, str]  # Physical meaning of parameters
    typical_values: Dict[str, Any]  # Typical parameter values
    units: Dict[str, str]  # Units for each parameter

class ODEClassifier:
    """
    Classifies ODEs and identifies their physical applications
    """
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, PhysicalApplication]:
        """Build knowledge base of ODE applications"""
        
        kb = {
            # Harmonic Oscillator
            "y'' + ω²y = 0": PhysicalApplication(
                field="Physics",
                name="Simple Harmonic Oscillator",
                description="Describes oscillatory motion without damping",
                parameters_meaning={
                    "ω": "Angular frequency",
                    "A": "Amplitude",
                    "φ": "Phase angle"
                },
                typical_values={"ω": 2*3.14159, "A": 1.0, "φ": 0},
                units={"ω": "rad/s", "A": "m", "φ": "rad"}
            ),
            
            # Damped Oscillator
            "y'' + 2γy' + ω₀²y = 0": PhysicalApplication(
                field="Physics",
                name="Damped Harmonic Oscillator",
                description="Oscillatory motion with damping",
                parameters_meaning={
                    "γ": "Damping coefficient",
                    "ω₀": "Natural frequency"
                },
                typical_values={"γ": 0.1, "ω₀": 1.0},
                units={"γ": "1/s", "ω₀": "rad/s"}
            ),
            
            # Heat Equation (1D steady state)
            "y'' = 0": PhysicalApplication(
                field="Physics/Engineering",
                name="1D Steady-State Heat Conduction",
                description="Temperature distribution in steady state",
                parameters_meaning={
                    "T₀": "Temperature at boundary x=0",
                    "T₁": "Temperature at boundary x=L"
                },
                typical_values={"T₀": 100, "T₁": 20},
                units={"T": "°C or K"}
            ),
            
            # Bessel's Equation
            "x²y'' + xy' + (x² - n²)y = 0": PhysicalApplication(
                field="Physics/Engineering",
                name="Bessel's Equation",
                description="Appears in cylindrical waveguides, heat conduction in cylinders",
                parameters_meaning={
                    "n": "Order of Bessel function"
                },
                typical_values={"n": 0},
                units={"n": "dimensionless"}
            ),
            
            # Airy Equation
            "y'' - xy = 0": PhysicalApplication(
                field="Physics",
                name="Airy Equation",
                description="Quantum mechanics - particle in triangular potential well",
                parameters_meaning={
                    "x": "Position coordinate"
                },
                typical_values={},
                units={"x": "nm"}
            ),
            
            # Legendre Equation
            "(1-x²)y'' - 2xy' + n(n+1)y = 0": PhysicalApplication(
                field="Physics",
                name="Legendre Equation",
                description="Appears in solving Laplace's equation in spherical coordinates",
                parameters_meaning={
                    "n": "Degree of Legendre polynomial"
                },
                typical_values={"n": 2},
                units={"n": "dimensionless"}
            ),
            
            # Van der Pol Oscillator
            "y'' - μ(1-y²)y' + y = 0": PhysicalApplication(
                field="Engineering/Biology",
                name="Van der Pol Oscillator",
                description="Self-sustained oscillations in electronic circuits and biological rhythms",
                parameters_meaning={
                    "μ": "Nonlinearity parameter"
                },
                typical_values={"μ": 1.0},
                units={"μ": "dimensionless"}
            ),
            
            # Duffing Equation
            "y'' + δy' + αy + βy³ = γcos(ωt)": PhysicalApplication(
                field="Physics/Engineering",
                name="Duffing Oscillator",
                description="Nonlinear oscillator with cubic stiffness",
                parameters_meaning={
                    "δ": "Damping",
                    "α": "Linear stiffness",
                    "β": "Nonlinear stiffness",
                    "γ": "Forcing amplitude",
                    "ω": "Forcing frequency"
                },
                typical_values={"δ": 0.3, "α": -1, "β": 1, "γ": 0.5, "ω": 1.2},
                units={"δ": "1/s", "α": "1/s²", "β": "1/(m²s²)", "γ": "m/s²", "ω": "rad/s"}
            ),
            
            # Schrödinger Equation (time-independent, 1D)
            "-ℏ²/(2m) y'' + V(x)y = Ey": PhysicalApplication(
                field="Quantum Mechanics",
                name="Time-Independent Schrödinger Equation",
                description="Quantum mechanical stationary states",
                parameters_meaning={
                    "ℏ": "Reduced Planck constant",
                    "m": "Particle mass",
                    "V(x)": "Potential energy function",
                    "E": "Energy eigenvalue"
                },
                typical_values={"ℏ": 1.054e-34, "m": 9.109e-31},
                units={"ℏ": "J·s", "m": "kg", "E": "J", "V": "J"}
            ),
            
            # Reaction-Diffusion (steady state)
            "D y'' - ky = 0": PhysicalApplication(
                field="Chemistry/Biology",
                name="Reaction-Diffusion Equation",
                description="Chemical concentration with diffusion and reaction",
                parameters_meaning={
                    "D": "Diffusion coefficient",
                    "k": "Reaction rate constant"
                },
                typical_values={"D": 1e-9, "k": 0.1},
                units={"D": "m²/s", "k": "1/s"}
            ),
            
            # Euler-Bernoulli Beam
            "EI y'''' = q(x)": PhysicalApplication(
                field="Engineering",
                name="Euler-Bernoulli Beam Equation",
                description="Deflection of elastic beams",
                parameters_meaning={
                    "E": "Young's modulus",
                    "I": "Second moment of area",
                    "q(x)": "Distributed load"
                },
                typical_values={"E": 200e9, "I": 1e-6},
                units={"E": "Pa", "I": "m⁴", "q": "N/m"}
            ),
            
            # Logistic Growth (transformed)
            "y' = ry(1 - y/K)": PhysicalApplication(
                field="Biology/Ecology",
                name="Logistic Growth Equation",
                description="Population growth with carrying capacity",
                parameters_meaning={
                    "r": "Growth rate",
                    "K": "Carrying capacity"
                },
                typical_values={"r": 0.5, "K": 1000},
                units={"r": "1/time", "K": "individuals"}
            ),
            
            # Pendulum (nonlinear)
            "y'' + (g/L)sin(y) = 0": PhysicalApplication(
                field="Physics",
                name="Nonlinear Pendulum",
                description="Exact pendulum equation without small angle approximation",
                parameters_meaning={
                    "g": "Gravitational acceleration",
                    "L": "Pendulum length"
                },
                typical_values={"g": 9.81, "L": 1.0},
                units={"g": "m/s²", "L": "m"}
            ),
            
            # Mathieu Equation
            "y'' + (a - 2q cos(2t))y = 0": PhysicalApplication(
                field="Physics/Engineering",
                name="Mathieu Equation",
                description="Parametric oscillations, stability of periodic orbits",
                parameters_meaning={
                    "a": "Characteristic value",
                    "q": "Parameter strength"
                },
                typical_values={"a": 1.0, "q": 0.5},
                units={"a": "dimensionless", "q": "dimensionless"}
            ),
            
            # Korteweg-de Vries (traveling wave)
            "y''' + 6yy' = 0": PhysicalApplication(
                field="Physics",
                name="KdV Equation (Traveling Wave)",
                description="Soliton waves in shallow water",
                parameters_meaning={
                    "c": "Wave speed"
                },
                typical_values={"c": 1.0},
                units={"c": "m/s"}
            )
        }
        
        return kb
    
    def classify_ode(self, ode_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an ODE and identify its physical applications
        
        Args:
            ode_spec: ODE specification from generator
            
        Returns:
            Classification results including name and applications
        """
        # Extract ODE structure
        ode_str = str(ode_spec.get('lhs', ''))
        order = ode_spec.get('order', 0)
        is_linear = ode_spec.get('is_linear', True)
        
        # Pattern matching for known equations
        matched_applications = []
        
        # Check for exact matches
        for pattern, app in self.knowledge_base.items():
            if self._matches_pattern(ode_str, pattern):
                matched_applications.append(app)
        
        # Classify by structure
        classification = self._structural_classification(ode_spec)
        
        # Generate standard name
        standard_name = self._generate_standard_name(ode_spec)
        
        # Identify potential applications
        potential_applications = self._identify_potential_applications(
            order, is_linear, ode_spec
        )
        
        return {
            'standard_name': standard_name,
            'classification': classification,
            'matched_applications': matched_applications,
            'potential_applications': potential_applications,
            'properties': self._analyze_properties(ode_spec)
        }
    
    def _matches_pattern(self, ode_str: str, pattern: str) -> bool:
        """Check if ODE matches a known pattern"""
        # Normalize strings for comparison
        ode_normalized = self._normalize_equation(ode_str)
        pattern_normalized = self._normalize_equation(pattern)
        
        # Check for structural similarity
        return self._structural_similarity(ode_normalized, pattern_normalized) > 0.8
    
    def _normalize_equation(self, eq_str: str) -> str:
        """Normalize equation string for comparison"""
        # Remove spaces and standardize notation
        eq_str = eq_str.replace(' ', '')
        eq_str = eq_str.replace('**', '^')
        
        # Standardize derivative notation
        eq_str = re.sub(r"y'\(x\)", "y'", eq_str)
        eq_str = re.sub(r"y''\(x\)", "y''", eq_str)
        eq_str = re.sub(r"y\(x\)", "y", eq_str)
        
        return eq_str.lower()
    
    def _structural_similarity(self, eq1: str, eq2: str) -> float:
        """Calculate structural similarity between two equations"""
        # Simple similarity based on common terms
        terms1 = set(re.findall(r'[a-z]+\'*|\^|\+|\-|\*|/', eq1))
        terms2 = set(re.findall(r'[a-z]+\'*|\^|\+|\-|\*|/', eq2))
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        
        return intersection / union if union > 0 else 0.0
    
    def _structural_classification(self, ode_spec: Dict[str, Any]) -> Dict[str, str]:
        """Classify ODE by structure"""
        classification = {}
        
        # Order classification
        order = ode_spec.get('order', 0)
        if order == 1:
            classification['order_type'] = 'First Order'
        elif order == 2:
            classification['order_type'] = 'Second Order'
        elif order == 3:
            classification['order_type'] = 'Third Order'
        else:
            classification['order_type'] = f'{order}th Order'
        
        # Linearity
        classification['linearity'] = 'Linear' if ode_spec.get('is_linear') else 'Nonlinear'
        
        # Special features
        features = []
        if ode_spec.get('has_delay'):
            features.append('Delay')
        if ode_spec.get('has_trigonometric'):
            features.append('Trigonometric')
        if ode_spec.get('has_exponential'):
            features.append('Exponential')
        
        classification['special_features'] = features
        
        # Coefficient type
        classification['coefficient_type'] = 'Constant Coefficients'  # Can be extended
        
        return classification
    
    def _generate_standard_name(self, ode_spec: Dict[str, Any]) -> str:
        """Generate standard mathematical name for ODE"""
        
        # Check if it matches known equations
        ode_str = str(ode_spec.get('lhs', ''))
        
        # Pattern matching for standard equations
        if 'bessel' in ode_str.lower():
            return "Bessel-type Equation"
        elif 'airy' in ode_str.lower():
            return "Airy-type Equation"
        elif 'legendre' in ode_str.lower():
            return "Legendre-type Equation"
        elif ode_spec.get('has_delay'):
            if ode_spec.get('is_linear'):
                return "Linear Delay Differential Equation"
            else:
                return "Nonlinear Delay Differential Equation"
        elif ode_spec.get('has_exponential') and not ode_spec.get('is_linear'):
            return "Exponential Nonlinear ODE"
        elif ode_spec.get('has_trigonometric') and not ode_spec.get('is_linear'):
            return "Trigonometric Nonlinear ODE"
        else:
            # Generic name based on structure
            order = ode_spec.get('order', 0)
            linearity = 'Linear' if ode_spec.get('is_linear') else 'Nonlinear'
            
            return f"{self._ordinal(order)} Order {linearity} ODE"
    
    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    def _identify_potential_applications(
        self, order: int, is_linear: bool, ode_spec: Dict[str, Any]
    ) -> List[str]:
        """Identify potential applications based on ODE structure"""
        applications = []
        
        if order == 2 and is_linear:
            applications.extend([
                "Mechanical vibrations",
                "Electrical circuits (RLC)",
                "Wave propagation"
            ])
        
        if order == 2 and not is_linear:
            applications.extend([
                "Nonlinear oscillations",
                "Chaos theory",
                "Population dynamics (predator-prey)"
            ])
        
        if ode_spec.get('has_delay'):
            applications.extend([
                "Control systems with time delay",
                "Population dynamics with maturation time",
                "Economic models with delay"
            ])
        
        if order == 4:
            applications.extend([
                "Beam bending (Euler-Bernoulli)",
                "Plate vibrations"
            ])
        
        if ode_spec.get('has_exponential'):
            applications.extend([
                "Chemical kinetics",
                "Radioactive decay chains"
            ])
        
        return applications
    
    def _analyze_properties(self, ode_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mathematical properties of the ODE"""
        properties = {}
        
        # Stability analysis placeholder
        properties['stability'] = 'To be determined'
        
        # Existence and uniqueness
        if ode_spec.get('is_linear'):
            properties['existence_uniqueness'] = 'Guaranteed (linear ODE)'
        else:
            properties['existence_uniqueness'] = 'Depends on specific nonlinearity'
        
        # Symmetries
        properties['symmetries'] = []
        
        # Conservation laws
        properties['conservation_laws'] = []
        
        return properties
