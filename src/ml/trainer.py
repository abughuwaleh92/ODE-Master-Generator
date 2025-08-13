"""
Machine Learning Trainer for ODE Generators
Handles training, evaluation, and generation of new ODEs with memory optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple, Iterator, Callable
import json
import pickle
import os
from datetime import datetime
import logging
from tqdm import tqdm
import gc
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from src.generators.linear_generators import LinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    from src.ml.pattern_learner import (
        GeneratorPatternLearner,
        GeneratorVAE,
        GeneratorTransformer,
        create_model
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

class ODEDataset(Dataset):
    """Dataset for ODE generator patterns with memory optimization"""
    
    def __init__(self, data: List[Dict[str, Any]], max_cache_size: int = 1000):
        """
        Initialize dataset with caching
        
        Args:
            data: List of ODE data dictionaries
            max_cache_size: Maximum number of items to cache in memory
        """
        self.data = data
        self.max_cache_size = min(max_cache_size, len(data))
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        """Extract features from a single ODE data item"""
        features = [
            item.get('alpha', 1.0),
            item.get('beta', 1.0),
            item.get('n', 1),
            item.get('M', 0.0),
            item.get('function_id', 0),
            1 if item.get('type') == 'linear' else 0,
            item.get('generator_number', 1),
            item.get('order', 2),
            item.get('q', 2) if 'q' in item else 0,
            item.get('v', 3) if 'v' in item else 0,
            item.get('a', 2.0) if 'a' in item else 0,
            np.random.randn() * 0.1,  # Small noise for regularization
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Check cache first
        if idx in self._feature_cache:
            self._cache_hits += 1
            features = self._feature_cache[idx]
        else:
            self._cache_misses += 1
            features = self._extract_features(self.data[idx])
            
            # Add to cache if not full
            if len(self._feature_cache) < self.max_cache_size:
                self._feature_cache[idx] = features
            else:
                # LRU eviction: remove oldest entry
                if self._feature_cache:
                    oldest = next(iter(self._feature_cache))
                    del self._feature_cache[oldest]
                self._feature_cache[idx] = features
        
        return features, self.data[idx]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._feature_cache)
        }

class ODEDataGenerator(IterableDataset):
    """Memory-efficient data generator for large-scale training"""
    
    def __init__(
        self,
        num_samples: int,
        batch_size: int = 32,
        seed: Optional[int] = None
    ):
        """
        Initialize data generator
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size for generation
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        
        # Initialize factories
        self.linear_factory = LinearGeneratorFactory()
        self.nonlinear_factory = NonlinearGeneratorFactory()
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()
        
        # Function lists
        self.basic_func_names = self.basic_functions.get_function_names()
        self.special_func_names = self.special_functions.get_function_names()[:5]  # Limit special functions
        self.all_func_names = self.basic_func_names + self.special_func_names
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Generate data on-the-fly"""
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        generated = 0
        while generated < self.num_samples:
            try:
                # Generate random parameters
                params = {
                    'alpha': np.random.uniform(-5, 5),
                    'beta': np.random.uniform(0.1, 5),
                    'n': np.random.randint(1, 5),
                    'M': np.random.uniform(-5, 5)
                }
                
                # Random function
                func_name = np.random.choice(self.all_func_names)
                if func_name in self.basic_func_names:
                    f_z = self.basic_functions.get_function(func_name)
                    func_id = self.basic_func_names.index(func_name)
                else:
                    f_z = self.special_functions.get_function(func_name)
                    func_id = len(self.basic_func_names) + self.special_func_names.index(func_name)
                
                # Random generator type
                gen_type = np.random.choice(['linear', 'nonlinear'])
                
                if gen_type == 'linear':
                    gen_num = np.random.randint(1, 9)
                    if gen_num in [4, 5]:
                        params['a'] = np.random.uniform(1, 3)
                    result = self.linear_factory.create(gen_num, f_z, **params)
                else:
                    gen_num = np.random.randint(1, 11)
                    extra_params = {}
                    if gen_num in [1, 2, 4]:
                        extra_params['q'] = np.random.randint(2, 6)
                    if gen_num in [2, 3, 5]:
                        extra_params['v'] = np.random.randint(2, 6)
                    if gen_num in [4, 5, 9, 10]:
                        extra_params['a'] = np.random.uniform(1, 5)
                    
                    result = self.nonlinear_factory.create(gen_num, f_z, **{**params, **extra_params})
                    params.update(extra_params)
                
                # Create data item
                data_item = {
                    **params,
                    'function_name': func_name,
                    'function_id': func_id,
                    'type': gen_type,
                    'generator_number': gen_num,
                    'order': result['order']
                }
                
                # Extract features
                features = torch.tensor([
                    params['alpha'],
                    params['beta'],
                    params['n'],
                    params['M'],
                    func_id,
                    1 if gen_type == 'linear' else 0,
                    gen_num,
                    result['order'],
                    params.get('q', 0),
                    params.get('v', 0),
                    params.get('a', 0),
                    np.random.randn() * 0.1
                ], dtype=torch.float32)
                
                yield features, data_item
                generated += 1
                
            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                continue
    
    def __len__(self) -> int:
        return self.num_samples

class MLTrainer:
    """Main trainer class for ML models with memory optimization"""
    
    def __init__(
        self,
        model_type: str = 'pattern_learner',
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 12,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        checkpoint_dir: str = 'checkpoints',
        enable_mixed_precision: bool = False
    ):
        """
        Initialize trainer with memory optimization features
        
        Args:
            model_type: Type of model to use
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            learning_rate: Learning rate
            device: Device to use (cuda/cpu)
            checkpoint_dir: Directory for saving checkpoints
            enable_mixed_precision: Use mixed precision training for memory efficiency
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.enable_mixed_precision = enable_mixed_precision
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        if model_type == 'vae':
            self.criterion = self._vae_loss
        else:
            self.criterion = nn.MSELoss()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if enable_mixed_precision else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0,
            'best_val_loss': float('inf')
        }
        
        # Function factories
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()
    
    def _create_model(self) -> nn.Module:
        """Create the ML model"""
        if self.model_type == 'pattern_learner':
            return GeneratorPatternLearner(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim
            )
        elif self.model_type == 'vae':
            return GeneratorVAE(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                latent_dim=32
            )
        elif self.model_type == 'transformer':
            return GeneratorTransformer(
                input_dim=self.input_dim,
                d_model=self.hidden_dim
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _vae_loss(self, recon_x, x, mu, log_var):
        """VAE loss function"""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        samples: int = 1000,
        validation_split: float = 0.2,
        save_best: bool = True,
        use_generator: bool = True,
        checkpoint_interval: int = 10,
        gradient_accumulation_steps: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Train the model with memory optimization
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
            samples: Number of training samples
            validation_split: Validation split ratio
            save_best: Whether to save best model
            use_generator: Use data generator for memory efficiency
            checkpoint_interval: Save checkpoint every N epochs
            gradient_accumulation_steps: Accumulate gradients for larger effective batch size
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Create data loaders
        if use_generator:
            # Use memory-efficient generator
            train_samples = int(samples * (1 - validation_split))
            val_samples = samples - train_samples
            
            train_dataset = ODEDataGenerator(train_samples, batch_size)
            val_dataset = ODEDataGenerator(val_samples, batch_size, seed=42)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                num_workers=0  # Single worker for generator
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0
            )
        else:
            # Traditional approach with pre-generated data
            data = self._generate_training_data_batch(samples)
            dataset = ODEDataset(data, max_cache_size=min(1000, len(data)))
            
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (batch_features, batch_data) in enumerate(pbar):
                batch_features = batch_features.to(self.device)
                
                # Mixed precision training
                if self.enable_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        if self.model_type == 'vae':
                            recon, mu, log_var = self.model(batch_features)
                            loss = self.criterion(recon, batch_features, mu, log_var)
                        else:
                            output = self.model(batch_features)
                            loss = self.criterion(output, batch_features)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    # Standard training
                    if self.model_type == 'vae':
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        output = self.model(batch_features)
                        loss = self.criterion(output, batch_features)
                    
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                train_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': train_loss / batch_count})
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch_features, batch_data in val_loader:
                    batch_features = batch_features.to(self.device)
                    
                    if self.model_type == 'vae':
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        output = self.model(batch_features)
                        loss = self.criterion(output, batch_features)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
            
            # Calculate average losses
            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['epochs'] = epoch + 1
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Logging
            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, epochs)
            
            # Save best model
            if save_best and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.history['best_val_loss'] = best_val_loss
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pth"))
                logger.info(f"Saved best model with val_loss = {best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f"{self.model_type}_epoch_{epoch+1}.pth"
                )
                self.save_checkpoint(checkpoint_path, epoch + 1)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
            
            # Memory cleanup
            if (epoch + 1) % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        logger.info("Training completed!")
    
    def _generate_training_data_batch(self, num_samples: int, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Generate training data in batches to save memory"""
        logger.info(f"Generating {num_samples} training samples in batches...")
        
        data = []
        generator = ODEDataGenerator(num_samples, batch_size)
        
        for features, item in tqdm(generator, total=num_samples, desc="Generating data"):
            data.append(item)
        
        logger.info(f"Generated {len(data)} valid samples")
        return data
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.learning_rate
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint and return epoch"""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return 0
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint.get('history', self.history)
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
            logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
            return epoch
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0
    
    def save_model(self, path: str):
        """Save model and training history"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load model from file"""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_new_ode(self, seed: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a new ODE using the trained model
        
        Args:
            seed: Optional seed tensor
            
        Returns:
            Generated ODE dictionary or None
        """
        self.model.eval()
        
        with torch.no_grad():
            if seed is None:
                if self.model_type == 'vae':
                    # Sample from latent space
                    generated = self.model.sample(1)
                else:
                    # Generate random seed
                    seed = torch.randn(1, self.input_dim).to(self.device)
                    generated = self.model(seed)
            else:
                seed = seed.to(self.device)
                if self.model_type == 'vae':
                    generated, _, _ = self.model(seed)
                else:
                    generated = self.model(seed)
            
            # Extract parameters
            params_tensor = generated.cpu().numpy()[0]
            
            # Decode parameters with validation
            alpha = float(np.clip(params_tensor[0], -100, 100))
            beta = float(np.clip(abs(params_tensor[1]) + 0.1, 0.1, 100))
            n = int(np.clip(abs(params_tensor[2]), 1, 10))
            M = float(np.clip(params_tensor[3], -100, 100))
            func_id = int(abs(params_tensor[4])) % len(self.basic_functions.get_function_names())
            is_linear = params_tensor[5] > 0.5
            gen_num = int(np.clip(abs(params_tensor[6]), 1, 8 if is_linear else 10))
            
            # Get function
            basic_funcs = self.basic_functions.get_function_names()
            func_name = basic_funcs[func_id]
            f_z = self.basic_functions.get_function(func_name)
            
            # Generate ODE
            try:
                generator_params = {
                    'alpha': alpha,
                    'beta': beta,
                    'n': n,
                    'M': M
                }
                
                if is_linear:
                    factory = LinearGeneratorFactory()
                    if gen_num in [4, 5]:
                        generator_params['a'] = float(np.clip(abs(params_tensor[10]) + 1, 1, 5))
                    result = factory.create(gen_num, f_z, **generator_params)
                else:
                    factory = NonlinearGeneratorFactory()
                    extra_params = {}
                    if gen_num in [1, 2, 4]:
                        extra_params['q'] = int(np.clip(abs(params_tensor[8]) + 2, 2, 10))
                    if gen_num in [2, 3, 5]:
                        extra_params['v'] = int(np.clip(abs(params_tensor[9]) + 2, 2, 10))
                    if gen_num in [4, 5, 9, 10]:
                        extra_params['a'] = float(np.clip(abs(params_tensor[10]) + 1, 1, 5))
                    
                    result = factory.create(gen_num, f_z, **{**generator_params, **extra_params})
                
                result['function_used'] = func_name
                result['ml_generated'] = True
                result['generation_params'] = generator_params
                
                return result
                
            except Exception as e:
                logger.error(f"Error generating ODE: {e}")
                return None
    
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        dataset = ODEDataset(test_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_features, batch_data in loader:
                batch_features = batch_features.to(self.device)
                
                if self.model_type == 'vae':
                    recon, mu, log_var = self.model(batch_features)
                    loss = self.criterion(recon, batch_features, mu, log_var)
                    output = recon
                else:
                    output = self.model(batch_features)
                    loss = self.criterion(output, batch_features)
                
                total_loss += loss.item()
                predictions.append(output.cpu())
                targets.append(batch_features.cpu())
        
        # Calculate metrics
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        
        # Calculate correlation
        pred_flat = predictions.flatten().numpy()
        target_flat = targets.flatten().numpy()
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        
        return {
            'loss': total_loss / len(loader),
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }
    
    def export_onnx(self, path: str):
        """Export model to ONNX format"""
        self.model.eval()
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"Model exported to ONNX: {path}")
