"""
Advanced Generator Constructor System
Implements the mathematical framework for creating custom ODE generators
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DerivativeType(Enum):
    """Types of derivative transformations"""
    LINEAR = "linear"
    POWER = "power"
    EXPONENTIAL = "exponential"
    TRIGONOMETRIC = "trigonometric"
    LOGARITHMIC = "logarithmic"
    HYPERBOLIC = "hyperbolic"
    ALGEBRAIC = "algebraic"
    SPECIAL = "special"
    COMPOSITE = "composite"

class OperatorType(Enum):
    """Types of differential operators"""
    STANDARD = "standard"
    FRACTIONAL = "fractional"
    DELAY = "delay"
    ADVANCE = "advance"
    INTEGRO_DIFFERENTIAL = "integro_differential"
    STOCHASTIC = "stochastic"

@dataclass
class DerivativeTerm:
    """
    Represents a single term in the differential operator
    Encapsulates all transformations applied to a derivative
    """
    derivative_order: int
    coefficient: Union[float, sp.Expr] = 1.0
    power: int = 1
    function_type: DerivativeType = DerivativeType.LINEAR
    operator_type: OperatorType = OperatorType.STANDARD
    scaling: Optional[float] = None  # For delay/advance: y(x/a) or y(ax)
    shift: Optional[float] = None  # For shifted arguments: y(x + c)
    modulation: Optional[Dict[str, Any]] = None  # For modulated terms
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize term parameters"""
        if self.derivative_order < 0:
            raise ValueError("Derivative order must be non-negative")
        
        if self.power < 1:
            raise ValueError("Power must be at least 1")
        
        # Ensure coefficient is symbolic if needed
        if not isinstance(self.coefficient, sp.Expr):
            self.coefficient = sp.sympify(self.coefficient)
    
    def to_sympy(self, y_func: sp.Function, x_sym: sp.Symbol) -> sp.Expr:
        """
        Convert term to SymPy expression
        
        Args:
            y_func: The function symbol y
            x_sym: The independent variable symbol
            
        Returns:
            SymPy expression representing the term
        """
        # Start with the base function
        if self.operator_type == OperatorType.DELAY and self.scaling:
            arg = x_sym / self.scaling
        elif self.operator_type == OperatorType.ADVANCE and self.scaling:
            arg = x_sym * self.scaling
        elif self.shift:
            arg = x_sym + self.shift
        else:
            arg = x_sym
        
        # Apply derivative
        if self.derivative_order == 0:
            expr = y_func(arg)
        else:
            expr = sp.diff(y_func(arg), x_sym, self.derivative_order)
        
        # Apply function transformation
        expr = self._apply_function_type(expr, x_sym)
        
        # Apply power if needed
        if self.power != 1:
            expr = expr ** self.power
        
        # Apply modulation if specified
        if self.modulation:
            expr = self._apply_modulation(expr, x_sym)
        
        # Apply coefficient
        return self.coefficient * expr
    
    def _apply_function_type(self, expr: sp.Expr, x_sym: sp.Symbol) -> sp.Expr:
        """Apply function transformation to expression"""
        if self.function_type == DerivativeType.LINEAR:
            return expr
        elif self.function_type == DerivativeType.EXPONENTIAL:
            return sp.exp(expr)
        elif self.function_type == DerivativeType.LOGARITHMIC:
            # Add safety check for positive argument
            return sp.log(sp.Abs(expr) + sp.Symbol('epsilon', positive=True))
        elif self.function_type == DerivativeType.TRIGONOMETRIC:
            trig_func = self.additional_params.get('trig_func', 'sin')
            trig_map = {
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'sec': sp.sec,
                'csc': sp.csc,
                'cot': sp.cot
            }
            return trig_map.get(trig_func, sp.sin)(expr)
        elif self.function_type == DerivativeType.HYPERBOLIC:
            hyp_func = self.additional_params.get('hyp_func', 'sinh')
            hyp_map = {
                'sinh': sp.sinh,
                'cosh': sp.cosh,
                'tanh': sp.tanh,
                'sech': sp.sech,
                'csch': sp.csch,
                'coth': sp.coth
            }
            return hyp_map.get(hyp_func, sp.sinh)(expr)
        elif self.function_type == DerivativeType.ALGEBRAIC:
            # Custom algebraic transformation
            transform = self.additional_params.get('transform', 'sqrt')
            if transform == 'sqrt':
                return sp.sqrt(sp.Abs(expr))
            elif transform == 'reciprocal':
                return 1 / (expr + sp.Symbol('epsilon', positive=True))
            else:
                return expr
        elif self.function_type == DerivativeType.COMPOSITE:
            # Apply composite function
            comp_funcs = self.additional_params.get('functions', [])
            for func_name in comp_funcs:
                if func_name in ['sin', 'cos', 'exp', 'log']:
                    expr = getattr(sp, func_name)(expr)
            return expr
        else:
            return expr
    
    def _apply_modulation(self, expr: sp.Expr, x_sym: sp.Symbol) -> sp.Expr:
        """Apply modulation to expression"""
        mod_type = self.modulation.get('type', 'amplitude')
        
        if mod_type == 'amplitude':
            # Amplitude modulation: A(x) * expr
            amp_func = self.modulation.get('function', 'constant')
            if amp_func == 'exponential':
                rate = self.modulation.get('rate', 1.0)
                return sp.exp(rate * x_sym) * expr
            elif amp_func == 'oscillatory':
                freq = self.modulation.get('frequency', 1.0)
                return sp.cos(freq * x_sym) * expr
            else:
                return expr
        elif mod_type == 'frequency':
            # Frequency modulation: expr(ω(x))
            freq_func = self.modulation.get('function', 'linear')
            if freq_func == 'linear':
                rate = self.modulation.get('rate', 1.0)
                return expr.subs(x_sym, rate * x_sym)
            elif freq_func == 'quadratic':
                rate = self.modulation.get('rate', 1.0)
                return expr.subs(x_sym, rate * x_sym**2)
            else:
                return expr
        else:
            return expr
    
    def get_description(self) -> str:
        """Generate human-readable description of the term"""
        parts = []
        
        # Coefficient
        if self.coefficient != 1:
            if self.coefficient == -1:
                parts.append("-")
            else:
                parts.append(str(self.coefficient))
        
        # Derivative notation
        if self.derivative_order == 0:
            base = "y"
        elif self.derivative_order == 1:
            base = "y'"
        elif self.derivative_order == 2:
            base = "y''"
        else:
            base = f"y^({self.derivative_order})"
        
        # Argument modification
        if self.operator_type == OperatorType.DELAY and self.scaling:
            base = base.replace("y", f"y(x/{self.scaling})")
        elif self.operator_type == OperatorType.ADVANCE and self.scaling:
            base = base.replace("y", f"y({self.scaling}x)")
        elif self.shift:
            base = base.replace("y", f"y(x+{self.shift})")
        
        # Function transformation
        if self.function_type == DerivativeType.EXPONENTIAL:
            base = f"e^({base})"
        elif self.function_type == DerivativeType.LOGARITHMIC:
            base = f"ln({base})"
        elif self.function_type == DerivativeType.TRIGONOMETRIC:
            trig = self.additional_params.get('trig_func', 'sin')
            base = f"{trig}({base})"
        elif self.function_type == DerivativeType.HYPERBOLIC:
            hyp = self.additional_params.get('hyp_func', 'sinh')
            base = f"{hyp}({base})"
        
        # Power
        if self.power != 1:
            base = f"({base})^{self.power}"
        
        parts.append(base)
        
        return "".join(parts)

class GeneratorSpecification:
    """
    Complete specification of an ODE generator
    Encapsulates all information needed to construct and analyze ODEs
    """
    
    def __init__(self, terms: List[DerivativeTerm], name: Optional[str] = None,
                 description: Optional[str] = None, metadata: Optional[Dict] = None):
        self.terms = terms
        self.name = name or self._generate_name()
        self.description = description or self._generate_description()
        self.metadata = metadata or {}
        
        # Computed properties
        self.order = self._compute_order()
        self.is_linear = self._check_linearity()
        self.has_delay = self._check_delay()
        self.has_nonlocal = self._check_nonlocal()
        self.special_features = self._identify_features()
        
        # Symbolic representation
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Function('y')
        self.lhs = self._build_lhs()
    
    def _generate_name(self) -> str:
        """Generate automatic name based on structure"""
        parts = []
        
        # Group by derivative order
        order_groups = {}
        for term in self.terms:
            order = term.derivative_order
            if order not in order_groups:
                order_groups[order] = []
            order_groups[order].append(term)
        
        # Build name components
        for order in sorted(order_groups.keys()):
            if order == 0:
                parts.append("y")
            elif order == 1:
                parts.append("y'")
            elif order == 2:
                parts.append("y''")
            else:
                parts.append(f"y^({order})")
        
        base_name = "-".join(parts)
        
        # Add special qualifiers
        qualifiers = []
        if self.has_delay:
            qualifiers.append("Delay")
        if not self.is_linear:
            qualifiers.append("Nonlinear")
        if any(t.function_type == DerivativeType.TRIGONOMETRIC for t in self.terms):
            qualifiers.append("Trigonometric")
        
        if qualifiers:
            return f"{base_name} {' '.join(qualifiers)} Generator"
        else:
            return f"{base_name} Generator"
    
    def _generate_description(self) -> str:
        """Generate automatic description"""
        term_descriptions = [term.get_description() for term in self.terms]
        
        # Combine with proper signs
        equation_parts = []
        for i, desc in enumerate(term_descriptions):
            if i > 0 and not desc.startswith("-"):
                equation_parts.append(" + ")
            elif i > 0:
                equation_parts.append(" ")
            equation_parts.append(desc)
        
        return "".join(equation_parts) + " = RHS"
    
    def _compute_order(self) -> int:
        """Compute maximum derivative order"""
        if not self.terms:
            return 0
        return max(term.derivative_order for term in self.terms)
    
    def _check_linearity(self) -> bool:
        """Check if generator is linear"""
        for term in self.terms:
            # Nonlinear if power > 1 or has nonlinear function type
            if term.power > 1:
                return False
            if term.function_type not in [DerivativeType.LINEAR]:
                return False
        return True
    
    def _check_delay(self) -> bool:
        """Check if generator has delay terms"""
        return any(
            term.operator_type in [OperatorType.DELAY, OperatorType.ADVANCE]
            or term.scaling is not None
            or term.shift is not None
            for term in self.terms
        )
    
    def _check_nonlocal(self) -> bool:
        """Check if generator has nonlocal terms"""
        return any(
            term.operator_type in [OperatorType.INTEGRO_DIFFERENTIAL, OperatorType.FRACTIONAL]
            for term in self.terms
        )
    
    def _identify_features(self) -> List[str]:
        """Identify special features"""
        features = []
        
        if self.has_delay:
            features.append("delay")
        if self.has_nonlocal:
            features.append("nonlocal")
        
        # Check function types
        func_types = set(term.function_type for term in self.terms)
        if DerivativeType.TRIGONOMETRIC in func_types:
            features.append("trigonometric")
        if DerivativeType.EXPONENTIAL in func_types:
            features.append("exponential")
        if DerivativeType.LOGARITHMIC in func_types:
            features.append("logarithmic")
        if DerivativeType.HYPERBOLIC in func_types:
            features.append("hyperbolic")
        
        # Check for modulation
        if any(term.modulation for term in self.terms):
            features.append("modulated")
        
        # Check for stochastic terms
        if any(term.operator_type == OperatorType.STOCHASTIC for term in self.terms):
            features.append("stochastic")
        
        return features
    
    def _build_lhs(self) -> sp.Expr:
        """Build left-hand side symbolic expression"""
        lhs = 0
        for term in self.terms:
            lhs += term.to_sympy(self.y, self.x)
        return lhs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'description': self.description,
            'order': self.order,
            'is_linear': self.is_linear,
            'has_delay': self.has_delay,
            'has_nonlocal': self.has_nonlocal,
            'special_features': self.special_features,
            'latex': sp.latex(self.lhs),
            'string_form': str(self.lhs),
            'terms': [asdict(term) for term in self.terms],
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = self.to_dict()
        # Convert SymPy expressions to strings
        for term_dict in data['terms']:
            if isinstance(term_dict['coefficient'], sp.Expr):
                term_dict['coefficient'] = str(term_dict['coefficient'])
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneratorSpecification':
        """Create from dictionary representation"""
        terms = []
        for term_dict in data['terms']:
            # Convert string coefficient back to SymPy if needed
            if isinstance(term_dict['coefficient'], str):
                term_dict['coefficient'] = sp.sympify(term_dict['coefficient'])
            
            # Convert string enums back to enum types
            term_dict['function_type'] = DerivativeType(term_dict['function_type'])
            term_dict['operator_type'] = OperatorType(term_dict['operator_type'])
            
            terms.append(DerivativeTerm(**term_dict))
        
        return cls(
            terms=terms,
            name=data.get('name'),
            description=data.get('description'),
            metadata=data.get('metadata', {})
        )

class GeneratorConstructor:
    """
    Main class for constructing and manipulating ODE generators
    Implements the mathematical framework from the Master Theorems
    """
    
    def __init__(self):
        self.x = sp.Symbol('x', real=True)
        self.y = sp.Function('y')
        self.z = sp.Symbol('z')
        
        # Cache for computed expressions
        self._cache = {}
        
        logger.info("GeneratorConstructor initialized")
    
    def create_standard_generator(self, template: str, **params) -> GeneratorSpecification:
        """
        Create generator from standard template
        
        Args:
            template: Name of standard template
            **params: Parameters for the template
            
        Returns:
            GeneratorSpecification object
        """
        templates = {
            'harmonic': lambda: [
                DerivativeTerm(2, 1.0),
                DerivativeTerm(0, params.get('omega', 1.0)**2)
            ],
            'damped_harmonic': lambda: [
                DerivativeTerm(2, 1.0),
                DerivativeTerm(1, 2*params.get('gamma', 0.1)),
                DerivativeTerm(0, params.get('omega', 1.0)**2)
            ],
            'van_der_pol': lambda: [
                DerivativeTerm(2, 1.0),
                DerivativeTerm(1, -params.get('mu', 1.0), power=1,
                             function_type=DerivativeType.COMPOSITE,
                             additional_params={'functions': ['nonlinear']}),
                DerivativeTerm(0, 1.0)
            ],
            'duffing': lambda: [
                DerivativeTerm(2, 1.0),
                DerivativeTerm(1, params.get('delta', 0.3)),
                DerivativeTerm(0, params.get('alpha', -1.0)),
                DerivativeTerm(0, params.get('beta', 1.0), power=3)
            ],
            'mathieu': lambda: [
                DerivativeTerm(2, 1.0),
                DerivativeTerm(0, params.get('a', 1.0) - 2*params.get('q', 0.5),
                             modulation={'type': 'amplitude', 'function': 'oscillatory',
                                        'frequency': 2.0})
            ],
            'bessel': lambda: self._create_bessel_terms(params.get('n', 0)),
            'airy': lambda: [
                DerivativeTerm(2, 1.0),
                DerivativeTerm(0, -self.x)
            ],
            'delay': lambda: [
                DerivativeTerm(1, 1.0),
                DerivativeTerm(0, params.get('a', -1.0), 
                             operator_type=OperatorType.DELAY,
                             scaling=params.get('tau', 2.0))
            ]
        }
        
        if template not in templates:
            raise ValueError(f"Unknown template: {template}")
        
        terms = templates[template]()
        
        return GeneratorSpecification(
            terms=terms,
            name=f"{template.replace('_', ' ').title()} Generator",
            metadata={'template': template, 'parameters': params}
        )
    
    def _create_bessel_terms(self, n: float) -> List[DerivativeTerm]:
        """Create terms for Bessel equation"""
        # x²y'' + xy' + (x² - n²)y = 0
        # Rewrite as: y'' + (1/x)y' + (1 - n²/x²)y = 0
        return [
            DerivativeTerm(2, 1.0),
            DerivativeTerm(1, sp.sympify(f"1/x")),
            DerivativeTerm(0, sp.sympify(f"1 - {n}**2/x**2"))
        ]
    
    def combine_generators(self, specs: List[GeneratorSpecification],
                          weights: Optional[List[float]] = None) -> GeneratorSpecification:
        """
        Combine multiple generator specifications
        
        Args:
            specs: List of generator specifications to combine
            weights: Optional weights for each generator
            
        Returns:
            Combined generator specification
        """
        if not specs:
            raise ValueError("At least one generator specification required")
        
        if weights is None:
            weights = [1.0] * len(specs)
        
        if len(weights) != len(specs):
            raise ValueError("Number of weights must match number of specifications")
        
        # Combine terms
        combined_terms = []
        for spec, weight in zip(specs, weights):
            for term in spec.terms:
                # Scale coefficient by weight
                scaled_term = DerivativeTerm(
                    derivative_order=term.derivative_order,
                    coefficient=term.coefficient * weight,
                    power=term.power,
                    function_type=term.function_type,
                    operator_type=term.operator_type,
                    scaling=term.scaling,
                    shift=term.shift,
                    modulation=term.modulation,
                    additional_params=term.additional_params
                )
                combined_terms.append(scaled_term)
        
        # Merge similar terms if possible
        merged_terms = self._merge_similar_terms(combined_terms)
        
        return GeneratorSpecification(
            terms=merged_terms,
            name=f"Combined Generator ({len(specs)} components)",
            metadata={'components': [spec.name for spec in specs], 'weights': weights}
        )
    
    def _merge_similar_terms(self, terms: List[DerivativeTerm]) -> List[DerivativeTerm]:
        """Merge terms with same structure but different coefficients"""
        merged = {}
        
        for term in terms:
            # Create key for grouping similar terms
            key = (
                term.derivative_order,
                term.power,
                term.function_type,
                term.operator_type,
                term.scaling,
                term.shift,
                json.dumps(term.modulation) if term.modulation else None,
                json.dumps(term.additional_params) if term.additional_params else None
            )
            
            if key in merged:
                # Add coefficients
                merged[key].coefficient += term.coefficient
            else:
                merged[key] = term
        
        return list(merged.values())
    
    def analyze_stability(self, spec: GeneratorSpecification) -> Dict[str, Any]:
        """
        Analyze stability properties of the generator
        
        Args:
            spec: Generator specification
            
        Returns:
            Dictionary with stability analysis results
        """
        analysis = {
            'is_autonomous': True,  # No explicit x dependence in coefficients
            'equilibrium_points': [],
            'linear_stability': None,
            'lyapunov_candidate': None
        }
        
        if spec.is_linear and spec.order == 2:
            # For second-order linear ODEs, analyze characteristic equation
            char_eq = self._get_characteristic_equation(spec)
            if char_eq:
                roots = sp.solve(char_eq, sp.Symbol('lambda'))
                analysis['characteristic_roots'] = roots
                
                # Determine stability based on roots
                if all(sp.re(root) < 0 for root in roots if root.is_number):
                    analysis['linear_stability'] = 'asymptotically stable'
                elif all(sp.re(root) <= 0 for root in roots if root.is_number):
                    analysis['linear_stability'] = 'marginally stable'
                else:
                    analysis['linear_stability'] = 'unstable'
        
        return analysis
    
    def _get_characteristic_equation(self, spec: GeneratorSpecification) -> Optional[sp.Expr]:
        """Get characteristic equation for linear ODE"""
        if not spec.is_linear:
            return None
        
        # Build characteristic polynomial
        lambda_sym = sp.Symbol('lambda')
        char_poly = 0
        
        for term in spec.terms:
            if term.function_type == DerivativeType.LINEAR and term.power == 1:
                # Only consider linear terms
                char_poly += term.coefficient * lambda_sym**term.derivative_order
        
        return char_poly
    
    def export_to_latex(self, spec: GeneratorSpecification, 
                        include_metadata: bool = True) -> str:
        """
        Export generator specification to LaTeX format
        
        Args:
            spec: Generator specification
            include_metadata: Whether to include metadata in output
            
        Returns:
            LaTeX string
        """
        latex_parts = []
        
        # Title
        latex_parts.append(f"\\section{{{spec.name}}}")
        
        # Description
        latex_parts.append(f"\\subsection{{Description}}")
        latex_parts.append(spec.description)
        
        # Equation
        latex_parts.append(f"\\subsection{{Differential Equation}}")
        latex_parts.append(f"\\begin{{equation}}")
        latex_parts.append(sp.latex(spec.lhs) + " = f(x)")
        latex_parts.append(f"\\end{{equation}}")
        
        # Properties
        latex_parts.append(f"\\subsection{{Properties}}")
        latex_parts.append(f"\\begin{{itemize}}")
        latex_parts.append(f"\\item Order: {spec.order}")
        latex_parts.append(f"\\item Type: {'Linear' if spec.is_linear else 'Nonlinear'}")
        
        if spec.special_features:
            latex_parts.append(f"\\item Special Features: {', '.join(spec.special_features)}")
        
        latex_parts.append(f"\\end{{itemize}}")
        
        # Metadata
        if include_metadata and spec.metadata:
            latex_parts.append(f"\\subsection{{Additional Information}}")
            latex_parts.append(f"\\begin{{verbatim}}")
            latex_parts.append(json.dumps(spec.metadata, indent=2))
            latex_parts.append(f"\\end{{verbatim}}")
        
        return "\n".join(latex_parts)
