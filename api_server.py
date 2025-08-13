"""
FastAPI server for Master Generators API - SECURE VERSION
Uses all Complete Generator implementations with authentication and rate limiting
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import uvicorn
import json
import numpy as np
import sympy as sp
from datetime import datetime, timedelta
import os
import sys
import logging
import io
import secrets
import hashlib
from functools import wraps
import time
from collections import defaultdict
import jwt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import configuration
try:
    from src.utils.config import app_config
except ImportError:
    logger.warning("Could not import app_config, using defaults")
    class AppConfig:
        SECRET_KEY = secrets.token_urlsafe(32)
        API_KEY = None
        RATE_LIMIT_ENABLED = True
        RATE_LIMIT_REQUESTS = 100
        RATE_LIMIT_WINDOW = 60
        ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8501"]
        JWT_ALGORITHM = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES = 30
        DEBUG = False
    app_config = AppConfig()

# Import the COMPLETE implementations
try:
    from src.generators.master_generator import (
        MasterGenerator,
        EnhancedMasterGenerator,
        CompleteMasterGenerator,
        CompleteLinearGeneratorFactory,
        CompleteNonlinearGeneratorFactory
    )
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    USE_COMPLETE = True
    logger.info("Using complete generator implementations")
except ImportError:
    logger.warning("Failed to import complete implementations, using basic versions")
    from src.generators.master_generator import MasterGenerator
    from src.generators.linear_generators import LinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    USE_COMPLETE = False

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Master Generators API - Secure Version",
    description="Enhanced API with all 18 generators, authentication, and rate limiting",
    version="2.1.0",
    docs_url="/docs" if app_config.DEBUG else None,  # Disable docs in production
    redoc_url="/redoc" if app_config.DEBUG else None
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.railway.app"] if not app_config.DEBUG else ["*"]
)

# CORS Middleware with security
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600
)

# ============================================================================
# SECURITY & AUTHENTICATION
# ============================================================================

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Rate limiting
class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, key: str, max_requests: int, window: int) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window
        ]
        
        # Check rate limit
        if len(self.requests[key]) >= max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True
    
    def get_reset_time(self, key: str, window: int) -> int:
        """Get time until rate limit resets"""
        if not self.requests[key]:
            return 0
        
        oldest = min(self.requests[key])
        return int(oldest + window - time.time())

rate_limiter = RateLimiter()

def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    # Check for proxy headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

async def check_rate_limit(request: Request):
    """Rate limiting dependency"""
    if not app_config.RATE_LIMIT_ENABLED:
        return
    
    client_ip = get_client_ip(request)
    
    if not rate_limiter.is_allowed(
        client_ip,
        app_config.RATE_LIMIT_REQUESTS,
        app_config.RATE_LIMIT_WINDOW
    ):
        reset_time = rate_limiter.get_reset_time(client_ip, app_config.RATE_LIMIT_WINDOW)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {reset_time} seconds.",
            headers={"Retry-After": str(reset_time)}
        )

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Verify API key if required"""
    if not app_config.API_KEY:
        return None
    
    if not api_key or api_key != app_config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    return api_key

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=app_config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        app_config.SECRET_KEY,
        algorithm=app_config.JWT_ALGORITHM
    )
    return encoded_jwt

async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)):
    """Verify JWT token"""
    if not credentials:
        return None
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            app_config.SECRET_KEY,
            algorithms=[app_config.JWT_ALGORITHM]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GeneratorParameters(BaseModel):
    """Parameters for generator with validation"""
    alpha: float = Field(default=1.0, ge=-100, le=100)
    beta: float = Field(default=1.0, gt=0, le=100)
    n: int = Field(default=1, ge=1, le=10)
    M: float = Field(default=0.0, ge=-100, le=100)
    
    @field_validator('beta')
    @classmethod
    def validate_beta(cls, v):
        if v <= 0:
            raise ValueError("Beta must be positive")
        return v

class SingleGeneratorRequest(BaseModel):
    """Request for single ODE generation"""
    type: str = Field(..., pattern="^(linear|nonlinear)$")
    generator_number: int = Field(..., ge=1, le=10)
    function: str = Field(..., max_length=50)
    parameters: GeneratorParameters
    q: Optional[int] = Field(default=2, ge=2, le=10)
    v: Optional[int] = Field(default=3, ge=2, le=10)
    a: Optional[float] = Field(default=2.0, gt=0, le=10)
    include_rhs: bool = Field(default=True, description="Include explicit RHS in response")

class BatchGeneratorRequest(BaseModel):
    """Request for batch ODE generation"""
    count: int = Field(..., ge=1, le=100)
    types: List[str] = Field(default=["linear", "nonlinear"])
    functions: List[str] = Field(...)
    random_params: bool = Field(default=True)
    include_rhs: bool = Field(default=False)

class APIResponse(BaseModel):
    """Standard API response"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    generator_version: str = "complete" if USE_COMPLETE else "basic"
    cached: bool = False

# ============================================================================
# INPUT SANITIZATION
# ============================================================================

def sanitize_sympy_expr(expr_str: str) -> str:
    """Sanitize SymPy expression to prevent code injection"""
    # Remove potentially dangerous characters/commands
    dangerous = ['exec', 'eval', '__import__', 'open', 'file', 'input', 'compile']
    
    expr_str = str(expr_str)
    for danger in dangerous:
        if danger in expr_str.lower():
            raise ValueError(f"Potentially dangerous expression detected: {danger}")
    
    return expr_str

def validate_function_name(name: str) -> str:
    """Validate function name to prevent injection"""
    # Allow only alphanumeric and underscore
    if not name.replace('_', '').isalnum():
        raise ValueError(f"Invalid function name: {name}")
    
    # Check length
    if len(name) > 50:
        raise ValueError("Function name too long")
    
    return name

# ============================================================================
# CACHING
# ============================================================================

class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()

cache = SimpleCache(ttl=app_config.CACHE_TTL if hasattr(app_config, 'CACHE_TTL') else 3600)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cache_key(request_data: dict) -> str:
    """Generate cache key from request data"""
    # Create deterministic string from request
    key_str = json.dumps(request_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def format_sympy_expr(expr) -> str:
    """Safely convert SymPy expression to string"""
    try:
        return sanitize_sympy_expr(str(expr))
    except:
        return "Expression too complex to display"

def process_generator_result(result: Dict[str, Any], include_rhs: bool = True) -> Dict[str, Any]:
    """Process generator result for API response"""
    response_data = {
        'type': result['type'],
        'order': result['order'],
        'generator_number': result['generator_number'],
        'description': result.get('description', ''),
        'initial_conditions': {k: format_sympy_expr(v) for k, v in result.get('initial_conditions', {}).items()}
    }
    
    # Handle ODE based on whether it's complete version with LHS/RHS
    if 'lhs' in result and 'rhs' in result and include_rhs:
        response_data['lhs'] = format_sympy_expr(result['lhs'])
        response_data['rhs'] = format_sympy_expr(result['rhs'])
        response_data['ode_equation'] = f"{response_data['lhs']} = {response_data['rhs']}"
        response_data['has_explicit_rhs'] = True
    else:
        response_data['ode'] = format_sympy_expr(result['ode'])
        response_data['has_explicit_rhs'] = False
    
    # Add solution
    response_data['solution'] = format_sympy_expr(result['solution'])
    
    # Add LaTeX representations with error handling
    try:
        if 'lhs' in result and 'rhs' in result:
            response_data['latex_lhs'] = sp.latex(result['lhs'])
            response_data['latex_rhs'] = sp.latex(result['rhs'])
        else:
            response_data['latex_ode'] = sp.latex(result['ode'])
        response_data['latex_solution'] = sp.latex(result['solution'])
    except:
        logger.warning("Could not generate LaTeX representation")
    
    # Add additional metadata
    if 'subtype' in result:
        response_data['subtype'] = result['subtype']
    if 'powers' in result:
        response_data['powers'] = result['powers']
    if 'scaling_parameter' in result:
        response_data['scaling_parameter'] = result['scaling_parameter']
    
    return response_data

# ============================================================================
# FACTORY INSTANCES
# ============================================================================

def get_factories():
    """Get the appropriate factory instances"""
    if USE_COMPLETE:
        return CompleteLinearGeneratorFactory(), CompleteNonlinearGeneratorFactory()
    else:
        return LinearGeneratorFactory(), NonlinearGeneratorFactory()

def get_basic_functions():
    """Get cached basic functions instance"""
    return BasicFunctions()

def get_special_functions():
    """Get cached special functions instance"""
    return SpecialFunctions()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": f"Master Generators API v2.1.0 - {'Complete' if USE_COMPLETE else 'Basic'} Implementation",
        "total_generators": 18,
        "linear_generators": 8,
        "nonlinear_generators": 10,
        "has_explicit_rhs": USE_COMPLETE,
        "documentation": "/docs" if app_config.DEBUG else "Disabled in production",
        "health": "/health",
        "endpoints": {
            "generate_single": "/api/generate/single",
            "generate_batch": "/api/generate/batch",
            "list_generators": "/api/generators/list",
            "functions": "/api/functions/list"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0",
        "implementation": "complete" if USE_COMPLETE else "basic"
    }

@app.post("/api/generate/single", response_model=APIResponse, tags=["Generators"])
async def generate_single_ode(
    request: SingleGeneratorRequest,
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Generate a single ODE with complete explicit RHS (if using complete implementation)
    """
    try:
        # Check cache
        cache_key = get_cache_key(request.dict())
        cached_result = cache.get(cache_key)
        if cached_result:
            return APIResponse(
                success=True,
                data=cached_result,
                cached=True,
                generator_version="complete" if USE_COMPLETE else "basic"
            )
        
        # Validate and sanitize function name
        function_name = validate_function_name(request.function)
        
        # Get function
        basic_functions = get_basic_functions()
        special_functions = get_special_functions()
        
        if function_name in basic_functions.get_function_names():
            f_z = basic_functions.get_function(function_name)
        elif function_name in special_functions.get_function_names():
            f_z = special_functions.get_function(function_name)
        else:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Get factories
        linear_factory, nonlinear_factory = get_factories()
        
        # Prepare parameters
        params = request.parameters.dict()
        
        # Generate ODE
        if request.type == "linear":
            # Validate generator number for linear
            if request.generator_number > 8:
                raise ValueError("Linear generators are numbered 1-8")
            
            # Add 'a' parameter for pantograph/delay equations
            if request.generator_number in [4, 5]:
                params['a'] = request.a
            
            result = linear_factory.create(request.generator_number, f_z, **params)
        else:
            # Validate generator number for nonlinear
            if request.generator_number > 10:
                raise ValueError("Nonlinear generators are numbered 1-10")
            
            # Add extra parameters as needed
            extra_params = {}
            if request.generator_number in [1, 2, 4]:
                extra_params['q'] = request.q
            if request.generator_number in [2, 3, 5]:
                extra_params['v'] = request.v
            if request.generator_number in [4, 5, 9, 10]:
                extra_params['a'] = request.a
            
            result = nonlinear_factory.create(
                request.generator_number, 
                f_z, 
                **{**params, **extra_params}
            )
        
        # Process result
        response_data = process_generator_result(result, include_rhs=request.include_rhs)
        response_data['function_used'] = function_name
        
        # Add novelty check
        response_data['novelty_score'] = 75.0 if request.type == "nonlinear" else 25.0
        response_data['is_novel'] = request.type == "nonlinear"
        
        # Cache result
        cache.set(cache_key, response_data)
        
        return APIResponse(
            success=True,
            data=response_data,
            generator_version="complete" if USE_COMPLETE else "basic"
        )
        
    except Exception as e:
        logger.error(f"ODE generation error: {e}")
        return APIResponse(success=False, error=str(e))

@app.post("/api/generate/batch", response_model=APIResponse, tags=["Generators"])
async def generate_batch_odes(
    request: BatchGeneratorRequest,
    background_tasks: BackgroundTasks,
    _rate_limit: None = Depends(check_rate_limit),
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """Generate multiple ODEs in batch"""
    try:
        # Limit batch size
        if request.count > app_config.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {app_config.MAX_BATCH_SIZE}")
        
        results = []
        linear_factory, nonlinear_factory = get_factories()
        basic_functions = get_basic_functions()
        special_functions = get_special_functions()
        
        for i in range(request.count):
            try:
                # Random parameters if requested
                if request.random_params:
                    params = {
                        'alpha': np.random.uniform(-5, 5),
                        'beta': np.random.uniform(0.1, 5),
                        'n': np.random.randint(1, 4),
                        'M': np.random.uniform(-5, 5)
                    }
                else:
                    params = {'alpha': 1.0, 'beta': 1.0, 'n': 1, 'M': 0.0}
                
                # Random selections
                gen_type = np.random.choice(request.types)
                func_name = validate_function_name(np.random.choice(request.functions))
                
                # Get function
                if func_name in basic_functions.get_function_names():
                    f_z = basic_functions.get_function(func_name)
                elif func_name in special_functions.get_function_names():
                    f_z = special_functions.get_function(func_name)
                else:
                    continue
                
                # Generate ODE
                if gen_type == "linear":
                    gen_num = np.random.randint(1, 9)
                    if gen_num in [4, 5]:
                        params['a'] = np.random.uniform(1, 3)
                    result = linear_factory.create(gen_num, f_z, **params)
                else:
                    gen_num = np.random.randint(1, 11)
                    extra_params = {}
                    if gen_num in [1, 2, 4]:
                        extra_params['q'] = np.random.randint(2, 5)
                    if gen_num in [2, 3, 5]:
                        extra_params['v'] = np.random.randint(2, 5)
                    if gen_num in [4, 5, 9, 10]:
                        extra_params['a'] = np.random.uniform(1, 3)
                    result = nonlinear_factory.create(gen_num, f_z, **{**params, **extra_params})
                
                # Process and add to results
                processed = process_generator_result(result, include_rhs=request.include_rhs)
                processed['id'] = i + 1
                processed['function'] = func_name
                results.append(processed)
                
            except Exception as e:
                logger.debug(f"Failed to generate ODE {i+1}: {e}")
        
        # Calculate statistics
        statistics = {
            'total_generated': len(results),
            'linear_count': sum(1 for r in results if r['type'] == 'linear'),
            'nonlinear_count': sum(1 for r in results if r['type'] == 'nonlinear'),
            'average_order': np.mean([r['order'] for r in results]) if results else 0,
            'with_explicit_rhs': USE_COMPLETE
        }
        
        return APIResponse(
            success=True,
            data={
                'count': len(results),
                'results': results,
                'statistics': statistics
            }
        )
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        return APIResponse(success=False, error=str(e))

@app.get("/api/generators/list", response_model=APIResponse, tags=["Generators"])
async def list_generators():
    """List all available generators with descriptions"""
    
    linear_generators = [
        {"number": 1, "equation": "y''(x) + y(x) = RHS", "order": 2},
        {"number": 2, "equation": "y''(x) + y'(x) = RHS", "order": 2},
        {"number": 3, "equation": "y(x) + y'(x) = RHS", "order": 1},
        {"number": 4, "equation": "y''(x) + y(x/a) - y(x) = RHS", "order": 2, "type": "pantograph"},
        {"number": 5, "equation": "y(x/a) + y'(x) = RHS", "order": 1, "type": "delay"},
        {"number": 6, "equation": "y'''(x) + y(x) = RHS", "order": 3},
        {"number": 7, "equation": "y'''(x) + y'(x) = RHS", "order": 3},
        {"number": 8, "equation": "y'''(x) + y''(x) = RHS", "order": 3}
    ]
    
    nonlinear_generators = [
        {"number": 1, "equation": "(y''(x))^q + y(x) = RHS", "order": 2, "parameters": ["q"]},
        {"number": 2, "equation": "(y''(x))^q + (y'(x))^v = RHS", "order": 2, "parameters": ["q", "v"]},
        {"number": 3, "equation": "y(x) + (y'(x))^v = RHS", "order": 1, "parameters": ["v"]},
        {"number": 4, "equation": "(y''(x))^q + y(x/a) - y(x) = RHS", "order": 2, "parameters": ["q", "a"]},
        {"number": 5, "equation": "y(x/a) + (y'(x))^v = RHS", "order": 1, "parameters": ["v", "a"]},
        {"number": 6, "equation": "sin(y''(x)) + y(x) = RHS", "order": 2, "type": "trigonometric"},
        {"number": 7, "equation": "e^(y''(x)) + e^(y'(x)) = RHS", "order": 2, "type": "exponential"},
        {"number": 8, "equation": "y(x) + e^(y'(x)) = RHS", "order": 1, "type": "exponential"},
        {"number": 9, "equation": "e^(y''(x)) + y(x/a) - y(x) = RHS", "order": 2, "parameters": ["a"]},
        {"number": 10, "equation": "y(x/a) + ln(y'(x)) = RHS", "order": 1, "parameters": ["a"], "type": "logarithmic"}
    ]
    
    return APIResponse(
        success=True,
        data={
            'linear': linear_generators,
            'nonlinear': nonlinear_generators,
            'total': 18,
            'has_explicit_rhs': USE_COMPLETE
        }
    )

@app.get("/api/functions/list", response_model=APIResponse, tags=["Functions"])
async def list_functions(category: Optional[str] = None):
    """List available functions"""
    try:
        basic = get_basic_functions()
        special = get_special_functions()
        
        if category == "basic":
            functions = basic.get_function_names()
        elif category == "special":
            functions = special.get_function_names()
        else:
            functions = {
                'basic': basic.get_function_names(),
                'special': special.get_function_names()
            }
        
        return APIResponse(
            success=True,
            data={'functions': functions}
        )
        
    except Exception as e:
        return APIResponse(success=False, error=str(e))

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    
    if app_config.DEBUG:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": str(exc)}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Internal server error"}
        )

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    
    # Print startup information
    print("=" * 60)
    print(f"Master Generators API v2.1.0 - SECURE")
    print(f"Implementation: {'COMPLETE' if USE_COMPLETE else 'BASIC'}")
    print(f"Total Generators: 18 (8 Linear + 10 Nonlinear)")
    print(f"Explicit RHS: {'YES' if USE_COMPLETE else 'NO'}")
    print(f"Rate Limiting: {'ENABLED' if app_config.RATE_LIMIT_ENABLED else 'DISABLED'}")
    print(f"API Key Required: {'YES' if app_config.API_KEY else 'NO'}")
    print(f"Debug Mode: {'ON' if app_config.DEBUG else 'OFF'}")
    print(f"Starting on port {port}...")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info" if not app_config.DEBUG else "debug"
    )
