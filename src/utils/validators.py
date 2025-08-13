# src/utils/validators.py - Parameter validation utilities
# ============================================================================
"""
Parameter validation utilities
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import numpy as np

class ParameterValidator:
    """Validator for generator parameters"""
    
    # Limits
    MAX_ALPHA = 100
    MAX_BETA = 100
    MAX_N = 10
    MAX_M = 100
    MAX_POWER = 10
    MAX_SCALING = 10
    
    @classmethod
    def validate_alpha(cls, value: float) -> float:
        """Validate alpha parameter"""
        if not isinstance(value, (int, float)):
            raise ValueError("Alpha must be a number")
        
        if abs(value) > cls.MAX_ALPHA:
            raise ValueError(f"Alpha must be between -{cls.MAX_ALPHA} and {cls.MAX_ALPHA}")
        
        return float(value)
    
    @classmethod
    def validate_beta(cls, value: float) -> float:
        """Validate beta parameter"""
        if not isinstance(value, (int, float)):
            raise ValueError("Beta must be a number")
        
        if value <= 0:
            raise ValueError("Beta must be positive")
        
        if value > cls.MAX_BETA:
            raise ValueError(f"Beta must be less than {cls.MAX_BETA}")
        
        return float(value)
    
    @classmethod
    def validate_n(cls, value: int) -> int:
        """Validate n parameter"""
        if not isinstance(value, int):
            try:
                value = int(value)
            except:
                raise ValueError("n must be an integer")
        
        if value < 1:
            raise ValueError("n must be at least 1")
        
        if value > cls.MAX_N:
            raise ValueError(f"n must be less than {cls.MAX_N}")
        
        return value
    
    @classmethod
    def validate_M(cls, value: float) -> float:
        """Validate M parameter"""
        if not isinstance(value, (int, float)):
            raise ValueError("M must be a number")
        
        if abs(value) > cls.MAX_M:
            raise ValueError(f"M must be between -{cls.MAX_M} and {cls.MAX_M}")
        
        return float(value)
    
    @classmethod
    def validate_power(cls, value: int, name: str = "power") -> int:
        """Validate power parameters (q, v)"""
        if not isinstance(value, int):
            try:
                value = int(value)
            except:
                raise ValueError(f"{name} must be an integer")
        
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        
        if value > cls.MAX_POWER:
            raise ValueError(f"{name} must be less than {cls.MAX_POWER}")
        
        return value
    
    @classmethod
    def validate_scaling(cls, value: float, name: str = "scaling") -> float:
        """Validate scaling parameter (a)"""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        
        if value > cls.MAX_SCALING:
            raise ValueError(f"{name} must be less than {cls.MAX_SCALING}")
        
        return float(value)
    
    @classmethod
    def validate_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all parameters"""
        validated = {}
        
        # Required parameters
        validated['alpha'] = cls.validate_alpha(params.get('alpha', 1.0))
        validated['beta'] = cls.validate_beta(params.get('beta', 1.0))
        validated['n'] = cls.validate_n(params.get('n', 1))
        validated['M'] = cls.validate_M(params.get('M', 0.0))
        
        # Optional parameters
        if 'q' in params:
            validated['q'] = cls.validate_power(params['q'], 'q')
        
        if 'v' in params:
            validated['v'] = cls.validate_power(params['v'], 'v')
        
        if 'a' in params:
            validated['a'] = cls.validate_scaling(params['a'], 'a')
        
        return validated
    
    @classmethod
    def sanitize_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters to ensure they're within bounds"""
        sanitized = {}
        
        # Clamp values to valid ranges
        sanitized['alpha'] = np.clip(float(params.get('alpha', 1.0)), -cls.MAX_ALPHA, cls.MAX_ALPHA)
        sanitized['beta'] = np.clip(float(params.get('beta', 1.0)), 0.1, cls.MAX_BETA)
        sanitized['n'] = np.clip(int(params.get('n', 1)), 1, cls.MAX_N)
        sanitized['M'] = np.clip(float(params.get('M', 0.0)), -cls.MAX_M, cls.MAX_M)
        
        if 'q' in params:
            sanitized['q'] = np.clip(int(params['q']), 1, cls.MAX_POWER)
        
        if 'v' in params:
            sanitized['v'] = np.clip(int(params['v']), 1, cls.MAX_POWER)
        
        if 'a' in params:
            sanitized['a'] = np.clip(float(params['a']), 0.1, cls.MAX_SCALING)
        
        return sanitized
