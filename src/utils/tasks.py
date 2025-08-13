# src/tasks.py - Celery tasks for background processing
# ============================================================================
"""
Celery tasks for background processing
"""

from celery import Celery, Task
from celery.utils.log import get_task_logger
import os
import json
from typing import Dict, Any, List

# Configure Celery
app = Celery('master_generators')
app.config_from_object({
    'broker_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
    'result_backend': os.getenv('REDIS_URL', 'redis://localhost:6379'),
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 300,  # 5 minutes
    'task_soft_time_limit': 240,  # 4 minutes
})

logger = get_task_logger(__name__)

@app.task(bind=True, name='tasks.train_model')
def train_model(self, model_type: str, epochs: int, batch_size: int, samples: int) -> Dict[str, Any]:
    """Background task for training ML models"""
    try:
        from src.ml.trainer import MLTrainer
        
        # Update task state
        self.update_state(state='TRAINING', meta={'progress': 0})
        
        # Create trainer
        trainer = MLTrainer(model_type=model_type)
        
        # Train with progress updates
        def progress_callback(epoch, total_epochs):
            progress = (epoch / total_epochs) * 100
            self.update_state(
                state='TRAINING',
                meta={'progress': progress, 'epoch': epoch, 'total': total_epochs}
            )
        
        # Train model
        trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            samples=samples,
            progress_callback=progress_callback
        )
        
        # Save model
        model_path = f"models/{model_type}_{self.request.id}.pth"
        trainer.save_model(model_path)
        
        return {
            'status': 'completed',
            'model_path': model_path,
            'final_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else None
        }
        
    except Exception as e:
        logger.error(f"Training task failed: {e}")
        return {'status': 'failed', 'error': str(e)}

@app.task(bind=True, name='tasks.generate_batch')
def generate_batch(self, count: int, types: List[str], functions: List[str]) -> List[Dict[str, Any]]:
    """Background task for batch ODE generation"""
    try:
        from src.generators.linear_generators import LinearGeneratorFactory
        from src.generators.nonlinear_generators import NonlinearGeneratorFactory
        from src.functions.basic_functions import BasicFunctions
        from src.functions.special_functions import SpecialFunctions
        import numpy as np
        
        results = []
        
        # Create factories
        linear_factory = LinearGeneratorFactory()
        nonlinear_factory = NonlinearGeneratorFactory()
        basic_funcs = BasicFunctions()
        special_funcs = SpecialFunctions()
        
        for i in range(count):
            # Update progress
            self.update_state(
                state='GENERATING',
                meta={'progress': (i / count) * 100, 'current': i, 'total': count}
            )
            
            # Generate random parameters
            params = {
                'alpha': np.random.uniform(-5, 5),
                'beta': np.random.uniform(0.1, 5),
                'n': np.random.randint(1, 5),
                'M': np.random.uniform(-5, 5)
            }
            
            # Random selections
            gen_type = np.random.choice(types)
            func_name = np.random.choice(functions)
            
            # Get function
            if func_name in basic_funcs.get_function_names():
                f_z = basic_funcs.get_function(func_name)
            else:
                f_z = special_funcs.get_function(func_name)
            
            # Generate ODE
            try:
                if gen_type == 'linear':
                    gen_num = np.random.randint(1, 9)
                    result = linear_factory.create(gen_num, f_z, **params)
                else:
                    gen_num = np.random.randint(1, 11)
                    q = np.random.randint(2, 6)
                    v = np.random.randint(2, 6)
                    result = nonlinear_factory.create(gen_num, f_z, q=q, v=v, **params)
                
                if result:
                    results.append({
                        'id': i + 1,
                        'type': result['type'],
                        'generator': result['generator_number'],
                        'function': func_name,
                        'order': result['order'],
                        'ode': str(result['ode'])[:500]
                    })
            except Exception as e:
                logger.debug(f"Failed to generate ODE {i}: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch generation task failed: {e}")
        return []

@app.task(name='tasks.analyze_novelty')
def analyze_novelty(ode_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Background task for novelty analysis"""
    try:
        from src.dl.novelty_detector import ODENoveltyDetector
        
        detector = ODENoveltyDetector()
        analysis = detector.analyze(ode_dict, detailed=True)
        
        return {
            'is_novel': analysis.is_novel,
            'novelty_score': analysis.novelty_score,
            'complexity_level': analysis.complexity_level,
            'recommended_methods': analysis.recommended_methods,
            'special_characteristics': analysis.special_characteristics
        }
        
    except Exception as e:
        logger.error(f"Novelty analysis task failed: {e}")
        return {'error': str(e)}

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    'cleanup-old-models': {
        'task': 'tasks.cleanup_old_models',
        'schedule': 86400.0,  # Daily
    },
    'generate-statistics': {
        'task': 'tasks.generate_statistics',
        'schedule': 3600.0,  # Hourly
    },
}

@app.task(name='tasks.cleanup_old_models')
def cleanup_old_models():
    """Clean up old model files"""
    import os
    from datetime import datetime, timedelta
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return
    
    cutoff_time = datetime.now() - timedelta(days=7)
    
    for filename in os.listdir(models_dir):
        filepath = os.path.join(models_dir, filename)
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        
        if file_time < cutoff_time:
            try:
                os.remove(filepath)
                logger.info(f"Deleted old model: {filename}")
            except Exception as e:
                logger.error(f"Failed to delete {filename}: {e}")

@app.task(name='tasks.generate_statistics')
def generate_statistics():
    """Generate usage statistics"""
    # This would typically query a database
    return {
        'timestamp': datetime.now().isoformat(),
        'total_odes_generated': 0,
        'models_trained': 0,
        'novelty_analyses': 0
    }
