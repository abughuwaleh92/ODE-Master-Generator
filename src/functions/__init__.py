# src/functions/__init__.py
# ============================================================================
"""
Mathematical functions module
Provides basic and special functions for ODE generation
"""

from .basic_functions import BasicFunctions
from .special_functions import SpecialFunctions

# Create singleton instances for convenience
basic_functions = BasicFunctions()
special_functions = SpecialFunctions()

# Get all available functions
ALL_FUNCTIONS = {
    'basic': basic_functions.get_function_names(),
    'special': special_functions.get_function_names()
}

__all__ = [
    'BasicFunctions',
    'SpecialFunctions',
    'basic_functions',
    'special_functions',
    'ALL_FUNCTIONS'
]
