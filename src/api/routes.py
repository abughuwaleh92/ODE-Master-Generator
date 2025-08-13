"""
API Routes for Master Generators
Comprehensive REST API endpoints for ODE generation and analysis
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import io
import csv
import asyncio
from datetime import datetime
import uuid

# Import core modules
from src.generators.master_generator import MasterGenerator
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
from src.ml.pattern_learner import GeneratorPatternLearner, create_model
from src.dl.novelty_detector import ODENoveltyDetector

# Create router
router = APIRouter(prefix="/api/v1", tags=["generators"])

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class GeneratorParams(BaseModel):
    """Parameters for generator"""
    alpha: float = Field(default=1.0, ge=-100, le=100)
    beta: float = Field(default=1.0, gt=0, le=100)
    n: int = Field(default=1, ge=1, le=10)
    M: float = Field(default=0.0, ge=-100, le=100)

class SingleGeneratorRequest(BaseModel):
    """Request for single ODE generation"""
    type: str = Field(..., pattern="^(linear|nonlinear)$")
    generator_number: int = Field(..., ge=1, le=10)
    function: str = Field(...)
    function_type: str = Field(default="basic", pattern="^(basic|special)$")
    parameters: GeneratorParams = Field(default_factory=GeneratorParams)
    q: Optional[int] = Field(default=2, ge=2, le=10)
    v: Optional[int] = Field(default=3, ge=2, le=10)
    a: Optional[float] = Field(default=2.0, gt=0, le=10)

class BatchGeneratorRequest(BaseModel):
    """Request for batch ODE generation"""
    count: int = Field(..., ge=1, le=1000)
    types: List[str] = Field(default=["linear", "nonlinear"])
    functions: List[str] = Field(...)
    function_types: List[str] = Field(default=["basic"])
    random_params: bool = Field(default=True)
    param_ranges: Optional[Dict[str, tuple]] = None

class MLTrainingRequest(BaseModel):
    """Request for ML model training"""
    model_type: str = Field(default="pattern_learner")
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    samples: int = Field(default=1000, ge=100, le=10000)

class NoveltyAnalysisRequest(BaseModel):
    """Request for novelty analysis"""
    ode: str = Field(...)
    type: str = Field(..., pattern="^(linear|nonlinear)$")
    order: int = Field(..., ge=1, le=10)
    check_solvability: bool = Field(default=True)
    detailed_analysis: bool = Field(default=False)

class ODEResponse(BaseModel):
    """Standard ODE response"""
    success: bool
    ode: Optional[str] = None
    solution: Optional[str] = None
    latex_ode: Optional[str] = None
    latex_solution: Optional[str] = None
    type: Optional[str] = None
    order: Optional[int] = None
    generator_number: Optional[int] = None
    function_used: Optional[str] = None
    initial_conditions: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ============================================================================
# GENERATOR ENDPOINTS
# ============================================================================

@router.post("/generate/single", response_model=ODEResponse)
async def generate_single_ode(request: SingleGeneratorRequest):
    """Generate a single ODE with the specified parameters"""
    try:
        # Get function
        if request.function_type == "basic":
            func_factory = BasicFunctions()
        else:
            func_factory = SpecialFunctions()
        
        f_z = func_factory.get_function(request.function)
        
        # Create generator
        params = request.parameters.dict()
        
        if request.type == "linear":
            factory = LinearGeneratorFactory()
            result = factory.create(request.generator_number, f_z, **params)
        else:
            factory = NonlinearGeneratorFactory()
            extra_params = {}
            if request.q:
                extra_params['q'] = request.q
            if request.v:
                extra_params['v'] = request.v
            if request.a:
                extra_params['a'] = request.a
            result = factory.create(request.generator_number, f_z, **{**params, **extra_params})
        
        # Format response
        import sympy as sp
        
        return ODEResponse(
            success=True,
            ode=str(result['ode']),
            solution=str(result['solution']),
            latex_ode=sp.latex(result['ode']),
            latex_solution=sp.latex(result['solution']),
            type=result['type'],
            order=result['order'],
            generator_number=result['generator_number'],
            function_used=request.function,
            initial_conditions={k: str(v) for k, v in result['initial_conditions'].items()}
        )
        
    except Exception as e:
        return ODEResponse(success=False, error=str(e))

@router.post("/generate/batch")
async def generate_batch_odes(request: BatchGeneratorRequest, background_tasks: BackgroundTasks):
    """Generate multiple ODEs in batch"""
    try:
        import numpy as np
        results = []
        
        for i in range(request.count):
            # Random parameters if requested
            if request.random_params:
                params = {
                    'alpha': np.random.uniform(-5, 5),
                    'beta': np.random.uniform(0.1, 5),
                    'n': np.random.randint(1, 5),
                    'M': np.random.uniform(-5, 5)
                }
            else:
                params = {'alpha': 1.0, 'beta': 1.0, 'n': 1, 'M': 0.0}
            
            # Random selections
            gen_type = np.random.choice(request.types)
            func_name = np.random.choice(request.functions)
            func_type = np.random.choice(request.function_types)
            
            # Get function
            if func_type == "basic":
                func_factory = BasicFunctions()
            else:
                func_factory = SpecialFunctions()
            
            try:
                f_z = func_factory.get_function(func_name)
                
                # Generate ODE
                if gen_type == "linear":
                    factory = LinearGeneratorFactory()
                    gen_num = np.random.randint(1, 9)
                    result = factory.create(gen_num, f_z, **params)
                else:
                    factory = NonlinearGeneratorFactory()
                    gen_num = np.random.randint(1, 11)
                    q = np.random.randint(2, 6)
                    v = np.random.randint(2, 6)
                    result = factory.create(gen_num, f_z, q=q, v=v, **params)
                
                results.append({
                    'id': i + 1,
                    'type': result['type'],
                    'generator': result['generator_number'],
                    'function': func_name,
                    'order': result['order'],
                    'ode': str(result['ode'])[:200] + '...' if len(str(result['ode'])) > 200 else str(result['ode'])
                })
                
            except Exception as e:
                results.append({
                    'id': i + 1,
                    'error': str(e)
                })
        
        # Save to file in background
        batch_id = str(uuid.uuid4())
        background_tasks.add_task(save_batch_results, batch_id, results)
        
        return {
            'success': True,
            'batch_id': batch_id,
            'count': len(results),
            'results': results[:10],  # Return first 10
            'message': f'Generated {len(results)} ODEs. Full results available at /api/v1/batch/{batch_id}'
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================================
# ML/DL ENDPOINTS
# ============================================================================

@router.post("/ml/train")
async def train_ml_model(request: MLTrainingRequest, background_tasks: BackgroundTasks):
    """Train ML model for pattern learning"""
    try:
        task_id = str(uuid.uuid4())
        
        # Start training in background
        background_tasks.add_task(
            train_model_background,
            task_id,
            request.model_type,
            request.epochs,
            request.batch_size,
            request.learning_rate,
            request.samples
        )
        
        return {
            'success': True,
            'task_id': task_id,
            'message': 'Training started. Check status at /api/v1/ml/status/{task_id}'
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

@router.get("/ml/generate")
async def generate_ml_ode(model_type: str = "pattern_learner"):
    """Generate ODE using trained ML model"""
    try:
        import torch
        from src.ml.trainer import MLTrainer
        
        trainer = MLTrainer(model_type=model_type)
        
        # Load model if exists
        if not trainer.load_model(f"models/{model_type}_latest.pth"):
            # Quick training if no model exists
            trainer.train(epochs=10, samples=100)
        
        # Generate new ODE
        result = trainer.generate_new_ode()
        
        if result:
            import sympy as sp
            return {
                'success': True,
                'ode': str(result['ode']),
                'solution': str(result['solution']),
                'latex_ode': sp.latex(result['ode']),
                'type': result['type'],
                'order': result['order']
            }
        else:
            return {'success': False, 'error': 'Failed to generate ODE'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

@router.post("/analyze/novelty")
async def analyze_novelty(request: NoveltyAnalysisRequest):
    """Analyze ODE for novelty and solvability"""
    try:
        detector = ODENoveltyDetector()
        
        ode_dict = {
            'ode': request.ode,
            'type': request.type,
            'order': request.order
        }
        
        analysis = detector.analyze(
            ode_dict,
            check_solvability=request.check_solvability,
            detailed=request.detailed_analysis
        )
        
        return {
            'success': True,
            'analysis': analysis
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================================
# FUNCTION ENDPOINTS
# ============================================================================

@router.get("/functions/list")
async def list_functions(category: Optional[str] = None):
    """List available functions"""
    try:
        basic = BasicFunctions()
        special = SpecialFunctions()
        
        if category == "basic":
            functions = basic.get_function_names()
        elif category == "special":
            functions = special.get_function_names()
        elif category in ["airy", "bessel", "gamma", "orthogonal"]:
            functions = list(special.get_function_by_category(category).keys())
        else:
            functions = {
                'basic': basic.get_function_names(),
                'special': special.get_function_names()
            }
        
        return {
            'success': True,
            'functions': functions,
            'count': len(functions) if isinstance(functions, list) else sum(len(v) for v in functions.values())
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

@router.get("/functions/{function_name}/properties")
async def get_function_properties(function_name: str, function_type: str = "basic"):
    """Get properties of a specific function"""
    try:
        if function_type == "basic":
            func_factory = BasicFunctions()
        else:
            func_factory = SpecialFunctions()
        
        props = func_factory.get_function_properties(function_name)
        
        return {
            'success': True,
            'properties': props
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================================
# EXPORT ENDPOINTS
# ============================================================================

@router.get("/export/{format}")
async def export_ode(
    ode: str,
    solution: str,
    format: str = "latex"
):
    """Export ODE in various formats"""
    try:
        import sympy as sp
        
        if format == "latex":
            content = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\begin{{document}}
\\section{{Generated ODE}}
\\subsection{{Differential Equation}}
$${sp.latex(ode)}$$
\\subsection{{Solution}}
$${sp.latex(solution)}$$
\\end{{document}}"""
            
            return StreamingResponse(
                io.StringIO(content),
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=ode.tex"}
            )
            
        elif format == "mathematica":
            content = f"ode = {ode};\nsolution = {solution};"
            return StreamingResponse(
                io.StringIO(content),
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=ode.m"}
            )
            
        elif format == "python":
            content = f"""import sympy as sp
x = sp.Symbol('x')
y = sp.Function('y')

# ODE
ode = {ode}

# Solution
solution = {solution}

print(f"ODE: {{ode}}")
print(f"Solution: {{solution}}")
"""
            return StreamingResponse(
                io.StringIO(content),
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=ode.py"}
            )
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def save_batch_results(batch_id: str, results: list):
    """Save batch results to file"""
    with open(f"data/batch_{batch_id}.json", "w") as f:
        json.dump(results, f, indent=2)

async def train_model_background(
    task_id: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    samples: int
):
    """Train model in background"""
    from src.ml.trainer import MLTrainer
    
    trainer = MLTrainer(model_type=model_type, learning_rate=learning_rate)
    trainer.train(epochs=epochs, batch_size=batch_size, samples=samples)
    trainer.save_model(f"models/{model_type}_{task_id}.pth")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
