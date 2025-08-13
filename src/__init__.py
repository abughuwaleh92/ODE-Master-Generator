# ============================================================================
# src/__init__.py - Main source module initialization
# ============================================================================
"""
Master Generators Source Module
Core implementation of the Master Generators system
"""

__version__ = "2.0.0"
__author__ = "Master Generators Team"
__email__ = "contact@master-generators.com"
__license__ = "MIT"

# Import core modules
from .generators.master_generator import MasterGenerator, EnhancedMasterGenerator
from .generators.linear_generators import LinearGeneratorFactory
from .generators.nonlinear_generators import NonlinearGeneratorFactory

# Import functions
from .functions.basic_functions import BasicFunctions
from .functions.special_functions import SpecialFunctions

# Import ML/DL modules
from .ml.pattern_learner import (
    GeneratorPatternLearner,
    GeneratorVAE,
    GeneratorTransformer,
    create_model
)
from .ml.trainer import MLTrainer
from .dl.novelty_detector import ODENoveltyDetector, NoveltyAnalysis

# Import utilities
from .utils.config import Settings, AppConfig
from .utils.cache import CacheManager
from .utils.validators import ParameterValidator

# Import UI components
from .ui.components import UIComponents

# Define public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Generators
    'MasterGenerator',
    'EnhancedMasterGenerator',
    'LinearGeneratorFactory',
    'NonlinearGeneratorFactory',
    
    # Functions
    'BasicFunctions',
    'SpecialFunctions',
    
    # ML/DL
    'GeneratorPatternLearner',
    'GeneratorVAE',
    'GeneratorTransformer',
    'create_model',
    'MLTrainer',
    'ODENoveltyDetector',
    'NoveltyAnalysis',
    
    # Utilities
    'Settings',
    'AppConfig',
    'CacheManager',
    'ParameterValidator',
    
    # UI
    'UIComponents'
]
