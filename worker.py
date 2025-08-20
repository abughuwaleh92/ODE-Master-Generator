# worker.py
"""
Enhanced RQ Worker for Master Generators ODE System
Handles compute and training jobs with comprehensive error handling,
progress tracking, and artifact management.
"""

import os
import sys
import json
import pickle
import zipfile
import traceback
import gc
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from pathlib import Path

import sympy as sp
import numpy as np
from rq import get_current_job
from rq.job import Job

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RQ utilities
from rq_utils import append_log, update_progress, update_artifacts

# Core computation
from shared.ode_core import ComputeParams, compute_ode_full, expr_to_str

# Optional imports with graceful fallback
try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    FUNCTIONS_AVAILABLE = True
except ImportError:
    BasicFunctions = SpecialFunctions = None
    FUNCTIONS_AVAILABLE = False

try:
    from src.ml.trainer import MLTrainer, TrainConfig
    ML_AVAILABLE = True
except ImportError:
    MLTrainer = TrainConfig = None
    ML_AVAILABLE = False

# Configure paths
ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = Path("checkpoints")
MODELS_DIR = Path("models")

# Create directories
for dir_path in [ARTIFACTS_DIR, CHECKPOINTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class WorkerLogger:
    """Centralized logging for worker jobs"""
    
    def __init__(self, job: Optional[Job] = None):
        self.job = job or get_current_job()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        formatted = f"[{level}] {message}"
        print(formatted)  # Console output
        
        if self.job:
            append_log(self.job.id, formatted)
    
    def info(self, message: str):
        self.log(message, "INFO")
    
    def warning(self, message: str):
        self.log(message, "WARNING")
    
    def error(self, message: str):
        self.log(message, "ERROR")
    
    def debug(self, message: str):
        self.log(message, "DEBUG")

class ProgressTracker:
    """Tracks and updates job progress"""
    
    def __init__(self, job: Optional[Job] = None):
        self.job = job or get_current_job()
        self.logger = WorkerLogger(job)
    
    def update(self, **kwargs):
        """Update progress with arbitrary key-value pairs"""
        if not self.job:
            return
        
        progress_data = {
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        update_progress(self.job.id, progress_data)
        
        # Log significant updates
        if 'stage' in kwargs:
            self.logger.info(f"Progress: {kwargs['stage']}")
        if 'percent' in kwargs:
            self.logger.info(f"Progress: {kwargs['percent']}%")

# ============================================================================
# COMPUTE JOB - ODE Generation
# ============================================================================

def validate_compute_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize compute job payload"""
    
    # Required fields with defaults
    validated = {
        'func_name': str(payload.get('func_name', 'exp(z)')),
        'alpha': float(payload.get('alpha', 1.0)),
        'beta': float(payload.get('beta', 1.0)),
        'n': int(payload.get('n', 1)),
        'M': float(payload.get('M', 0.0)),
        'use_exact': bool(payload.get('use_exact', True)),
        'simplify_level': str(payload.get('simplify_level', 'light')),
        'lhs_source': str(payload.get('lhs_source', 'constructor')),
        'function_library': str(payload.get('function_library', 'Basic')),
    }
    
    # Validate ranges
    if not 0.01 <= validated['beta'] <= 1000:
        raise ValueError(f"Beta must be between 0.01 and 1000, got {validated['beta']}")
    
    if not 1 <= validated['n'] <= 20:
        raise ValueError(f"n must be between 1 and 20, got {validated['n']}")
    
    if validated['simplify_level'] not in ['none', 'light', 'aggressive']:
        validated['simplify_level'] = 'light'
    
    if validated['lhs_source'] not in ['constructor', 'freeform', 'arbitrary']:
        validated['lhs_source'] = 'constructor'
    
    # Optional fields
    if 'freeform_terms' in payload:
        validated['freeform_terms'] = payload['freeform_terms']
    
    if 'arbitrary_lhs_text' in payload:
        # Validate arbitrary LHS for safety
        lhs_text = str(payload['arbitrary_lhs_text'])
        if len(lhs_text) > 5000:
            raise ValueError("Arbitrary LHS text too long (max 5000 chars)")
        validated['arbitrary_lhs_text'] = lhs_text
    
    if 'constructor_lhs' in payload:
        validated['constructor_lhs'] = payload['constructor_lhs']
    
    return validated

def compute_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main compute job for ODE generation via Master Theorem
    
    Args:
        payload: Job parameters
        
    Returns:
        Dictionary with computed ODE, solution, and metadata
    """
    job = get_current_job()
    logger = WorkerLogger(job)
    progress = ProgressTracker(job)
    
    start_time = datetime.utcnow()
    
    try:
        # Initialize
        logger.info(f"Starting compute job with payload keys: {list(payload.keys())}")
        progress.update(stage="initializing", percent=0)
        
        # Validate payload
        logger.info("Validating parameters...")
        validated_payload = validate_compute_payload(payload)
        
        # Load function libraries if available
        basic_lib = special_lib = None
        if FUNCTIONS_AVAILABLE:
            try:
                logger.info("Loading function libraries...")
                basic_lib = BasicFunctions()
                special_lib = SpecialFunctions()
                logger.info("Function libraries loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load function libraries: {e}")
        else:
            logger.warning("Function libraries not available in worker environment")
        
        # Handle constructor LHS if provided
        constructor_lhs = None
        if 'constructor_lhs' in validated_payload:
            try:
                constructor_lhs = sp.sympify(validated_payload['constructor_lhs'])
                logger.info(f"Parsed constructor LHS: {constructor_lhs}")
            except Exception as e:
                logger.warning(f"Failed to parse constructor_lhs: {e}")
        
        # Create compute parameters
        progress.update(stage="preparing", percent=20)
        
        params = ComputeParams(
            func_name=validated_payload['func_name'],
            alpha=validated_payload['alpha'],
            beta=validated_payload['beta'],
            n=validated_payload['n'],
            M=validated_payload['M'],
            use_exact=validated_payload['use_exact'],
            simplify_level=validated_payload['simplify_level'],
            lhs_source=validated_payload['lhs_source'],
            constructor_lhs=constructor_lhs,
            freeform_terms=validated_payload.get('freeform_terms'),
            arbitrary_lhs_text=validated_payload.get('arbitrary_lhs_text'),
            function_library=validated_payload['function_library'],
            basic_lib=basic_lib,
            special_lib=special_lib,
        )
        
        logger.info(f"Parameters: α={params.alpha}, β={params.beta}, n={params.n}, M={params.M}")
        logger.info(f"Function: {params.func_name} from {params.function_library} library")
        
        # Compute ODE
        progress.update(stage="computing", percent=40)
        logger.info("Starting ODE computation...")
        
        result = compute_ode_full(params)
        
        logger.info("ODE computation completed successfully")
        progress.update(stage="processing", percent=80)
        
        # Convert SymPy expressions to strings for JSON serialization
        safe_result = {
            'generator': expr_to_str(result.get('generator')),
            'rhs': expr_to_str(result.get('rhs')),
            'solution': expr_to_str(result.get('solution')),
            'f_expr_preview': expr_to_str(result.get('f_expr_preview')),
            'type': result.get('type', 'unknown'),
            'order': result.get('order', 0),
            'initial_conditions': result.get('initial_conditions', {}),
            'parameters': {
                'alpha': params.alpha,
                'beta': params.beta,
                'n': params.n,
                'M': params.M,
            },
            'metadata': {
                'function_name': params.func_name,
                'function_library': params.function_library,
                'lhs_source': params.lhs_source,
                'simplify_level': params.simplify_level,
                'use_exact': params.use_exact,
            },
            'computation_time': (datetime.utcnow() - start_time).total_seconds(),
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        # Save result summary as artifact
        artifact_path = ARTIFACTS_DIR / f"compute_{job.id}.json"
        with open(artifact_path, 'w') as f:
            json.dump(safe_result, f, indent=2)
        
        update_artifacts(job.id, {
            'result_file': str(artifact_path),
            'computation_time': safe_result['computation_time'],
        })
        
        progress.update(stage="completed", percent=100)
        logger.info(f"Job completed in {safe_result['computation_time']:.2f} seconds")
        
        return safe_result
        
    except Exception as e:
        error_msg = f"Compute job failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        progress.update(
            stage="failed",
            percent=0,
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        raise RuntimeError(error_msg) from e

# ============================================================================
# TRAINING JOB - ML Model Training
# ============================================================================

def validate_train_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize training job payload"""
    
    validated = {
        'model_type': str(payload.get('model_type', 'pattern_learner')),
        'hidden_dim': int(payload.get('hidden_dim', 128)),
        'normalize': bool(payload.get('normalize', False)),
        'epochs': int(payload.get('epochs', 100)),
        'batch_size': int(payload.get('batch_size', 32)),
        'samples': int(payload.get('samples', 1000)),
        'validation_split': float(payload.get('validation_split', 0.2)),
        'use_generator': bool(payload.get('use_generator', True)),
        'enable_mixed_precision': bool(payload.get('enable_mixed_precision', False)),
    }
    
    # Validate ranges
    if validated['model_type'] not in ['pattern_learner', 'vae', 'transformer']:
        validated['model_type'] = 'pattern_learner'
    
    if not 16 <= validated['hidden_dim'] <= 2048:
        validated['hidden_dim'] = 128
    
    if not 1 <= validated['epochs'] <= 10000:
        validated['epochs'] = 100
    
    if not 1 <= validated['batch_size'] <= 512:
        validated['batch_size'] = 32
    
    if not 10 <= validated['samples'] <= 100000:
        validated['samples'] = 1000
    
    if not 0.0 <= validated['validation_split'] <= 0.5:
        validated['validation_split'] = 0.2
    
    # Optional advanced parameters
    if 'beta_vae' in payload:
        validated['beta_vae'] = float(payload['beta_vae'])
    
    if 'kl_anneal' in payload:
        validated['kl_anneal'] = str(payload['kl_anneal'])
    
    if 'early_stop_patience' in payload:
        validated['early_stop_patience'] = int(payload['early_stop_patience'])
    
    if 'checkpoint_interval' in payload:
        validated['checkpoint_interval'] = int(payload['checkpoint_interval'])
    
    if 'resume_from' in payload:
        validated['resume_from'] = str(payload['resume_from'])
    
    return validated

def train_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background training job for ML models
    
    Args:
        payload: Training configuration
        
    Returns:
        Dictionary with training results and artifact paths
    """
    if not ML_AVAILABLE:
        raise RuntimeError("ML training modules not available in worker environment")
    
    job = get_current_job()
    logger = WorkerLogger(job)
    progress = ProgressTracker(job)
    
    start_time = datetime.utcnow()
    trainer = None
    
    try:
        # Initialize
        logger.info(f"Starting training job with payload keys: {list(payload.keys())}")
        progress.update(stage="initializing", percent=0)
        
        # Validate payload
        logger.info("Validating training parameters...")
        validated = validate_train_payload(payload)
        
        # Create training configuration
        logger.info(f"Creating trainer with model_type={validated['model_type']}")
        
        config = TrainConfig(
            model_type=validated['model_type'],
            hidden_dim=validated['hidden_dim'],
            normalize=validated['normalize'],
            beta_vae=validated.get('beta_vae', 1.0),
            kl_anneal=validated.get('kl_anneal', 'linear'),
            early_stop_patience=validated.get('early_stop_patience', 12),
            enable_mixed_precision=validated['enable_mixed_precision'],
        )
        
        trainer = MLTrainer(config)
        logger.info("Trainer initialized successfully")
        
        # Resume from checkpoint if specified
        if 'resume_from' in validated and os.path.exists(validated['resume_from']):
            try:
                epoch = trainer.load_checkpoint(validated['resume_from'])
                logger.info(f"Resumed from checkpoint at epoch {epoch}")
                progress.update(stage="resumed", resumed_epoch=epoch)
            except Exception as e:
                logger.warning(f"Failed to resume from checkpoint: {e}")
        
        # Setup progress hooks
        def progress_hook(info: Dict[str, Any]):
            """Hook called by trainer to update progress"""
            progress.update(
                stage="training",
                epoch=info.get('epoch', 0),
                total_epochs=validated['epochs'],
                percent=int((info.get('epoch', 0) / validated['epochs']) * 100),
                train_loss=info.get('train_loss'),
                val_loss=info.get('val_loss'),
            )
            
            # Log epoch info
            if 'epoch' in info and info['epoch'] % 10 == 0:
                logger.info(
                    f"Epoch {info['epoch']}/{validated['epochs']}: "
                    f"train_loss={info.get('train_loss', 'N/A'):.4f}, "
                    f"val_loss={info.get('val_loss', 'N/A'):.4f}"
                )
        
        def log_hook(message: str):
            """Hook for trainer to send log messages"""
            logger.info(f"[Trainer] {message}")
        
        # Set hooks
        trainer.progress_hook = progress_hook
        trainer.log_hook = log_hook
        
        # Start training
        logger.info(f"Starting training: {validated['epochs']} epochs, "
                   f"{validated['samples']} samples, "
                   f"batch_size={validated['batch_size']}")
        
        progress.update(stage="training", percent=5)
        
        trainer.train(
            epochs=validated['epochs'],
            batch_size=validated['batch_size'],
            samples=validated['samples'],
            validation_split=validated['validation_split'],
            use_generator=validated['use_generator'],
            checkpoint_interval=validated.get('checkpoint_interval', 10),
        )
        
        logger.info("Training completed successfully")
        progress.update(stage="saving", percent=95)
        
        # Save final model
        model_path = MODELS_DIR / f"{config.model_type}_{job.id}.pth"
        trainer.save_checkpoint(str(model_path), validated['epochs'])
        logger.info(f"Model saved to {model_path}")
        
        # Find best model
        best_model_path = CHECKPOINTS_DIR / f"{config.model_type}_best.pth"
        if not best_model_path.exists():
            # Try to find any best model
            import glob
            candidates = glob.glob(str(CHECKPOINTS_DIR / "*best*.pth"))
            if candidates:
                best_model_path = Path(candidates[-1])
        
        # Create session archive
        logger.info("Creating session archive...")
        session_zip_path = ARTIFACTS_DIR / f"session_{job.id}.zip"
        
        with zipfile.ZipFile(session_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add model files
            if model_path.exists():
                zf.write(model_path, arcname=model_path.name)
            if best_model_path.exists():
                zf.write(best_model_path, arcname=f"best_{best_model_path.name}")
            
            # Add training history
            history_data = {
                'history': trainer.history,
                'config': config.__dict__,
                'validated_params': validated,
                'training_time': (datetime.utcnow() - start_time).total_seconds(),
            }
            zf.writestr('history.json', json.dumps(history_data, indent=2))
            
            # Add metadata
            metadata = {
                'job_id': job.id,
                'created_at': datetime.utcnow().isoformat(),
                'model_type': config.model_type,
                'epochs': validated['epochs'],
                'samples': validated['samples'],
            }
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        logger.info(f"Session archive created: {session_zip_path}")
        
        # Update artifacts
        artifacts = {
            'model_path': str(model_path),
            'best_model_path': str(best_model_path) if best_model_path.exists() else None,
            'session_zip': str(session_zip_path),
            'training_time': (datetime.utcnow() - start_time).total_seconds(),
            'final_train_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else None,
            'final_val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else None,
            'best_val_loss': trainer.history.get('best_val_loss'),
        }
        
        update_artifacts(job.id, artifacts)
        
        progress.update(stage="completed", percent=100)
        logger.info(f"Training job completed in {artifacts['training_time']:.2f} seconds")
        
        # Clean up memory
        del trainer
        gc.collect()
        
        return artifacts
        
    except Exception as e:
        error_msg = f"Training job failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        progress.update(
            stage="failed",
            percent=0,
            error=str(e),
            traceback=traceback.format_exc()
        )
        
        # Clean up
        if trainer:
            del trainer
        gc.collect()
        
        raise RuntimeError(error_msg) from e

# ============================================================================
# UTILITY JOBS
# ============================================================================

def ping_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simple ping job for testing"""
    job = get_current_job()
    logger = WorkerLogger(job)
    
    logger.info(f"Ping received with payload: {payload}")
    
    return {
        'status': 'ok',
        'echo': payload,
        'job_id': job.id if job else None,
        'timestamp': datetime.utcnow().isoformat(),
        'worker_hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
    }

def cleanup_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up old artifacts and models"""
    job = get_current_job()
    logger = WorkerLogger(job)
    
    older_than_days = payload.get('older_than_days', 7)
    logger.info(f"Starting cleanup for files older than {older_than_days} days")
    
    cutoff_time = datetime.utcnow().timestamp() - (older_than_days * 86400)
    removed_files = []
    
    # Clean up directories
    for directory in [ARTIFACTS_DIR, CHECKPOINTS_DIR, MODELS_DIR]:
        if not directory.exists():
            continue
            
        for file_path in directory.glob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"Cleanup completed: removed {len(removed_files)} files")
    
    return {
        'removed_count': len(removed_files),
        'removed_files': removed_files[:100],  # Limit to first 100
        'timestamp': datetime.utcnow().isoformat(),
    }

# ============================================================================
# WORKER HEALTH CHECK
# ============================================================================

def health_check() -> Dict[str, Any]:
    """Check worker health and available modules"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'modules': {
            'functions_available': FUNCTIONS_AVAILABLE,
            'ml_available': ML_AVAILABLE,
            'sympy_version': sp.__version__,
            'numpy_version': np.__version__,
        },
        'directories': {
            'artifacts': str(ARTIFACTS_DIR.absolute()),
            'checkpoints': str(CHECKPOINTS_DIR.absolute()),
            'models': str(MODELS_DIR.absolute()),
        }
    }