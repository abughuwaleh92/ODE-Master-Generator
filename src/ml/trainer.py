"""
Machine Learning Trainer for ODE Generators
Handles training, evaluation, and generation of new ODEs with memory optimization
"""

import os
import json
import pickle
import logging
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Iterator, Callable

import numpy as np
import sympy as sp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from src.generators.linear_generators import LinearGeneratorFactory
except Exception:
    LinearGeneratorFactory = None

try:
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
except Exception:
    NonlinearGeneratorFactory = None

try:
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
except Exception as e:
    logger.error(f"Import error (functions): {e}")
    raise

try:
    from src.ml.pattern_learner import (
        GeneratorPatternLearner,
        GeneratorVAE,
        GeneratorTransformer,
        create_model
    )
except Exception as e:
    logger.error(f"Import error (models): {e}")
    raise


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _safe_param(container: Dict[str, Any], name: str, default):
    """
    Robustly fetch a parameter that might be absent, nested or wrong type.
    - Works with flat dicts and with dicts having a `parameters` sub-dict.
    - Coerces to int/float if `default` is numeric.
    """
    if container is None:
        return default

    # Prefer nested parameters if present
    src = container.get("parameters") if isinstance(container, dict) else None
    if isinstance(src, dict) and name in src:
        val = src.get(name, default)
    else:
        val = container.get(name, default) if isinstance(container, dict) else default

    try:
        if val is None:
            return default
        if isinstance(default, bool):
            return bool(val)
        if isinstance(default, int):
            return int(val)
        if isinstance(default, float):
            return float(val)
        return val
    except Exception:
        return default


# ---------------------------------------------------------------------
# Dataset (memory-optimized with caching) — accepts flat or nested params
# ---------------------------------------------------------------------
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
        self._feature_cache: Dict[int, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        """Extract features from a single ODE data item (flat or nested)."""
        # accept both item[...] and item['parameters'][...]
        alpha = _safe_param(item, "alpha", 1.0)
        beta  = _safe_param(item, "beta", 1.0)
        n     = _safe_param(item, "n", 1)
        M     = _safe_param(item, "M", 0.0)
        q     = _safe_param(item, "q", 0)
        v     = _safe_param(item, "v", 0)
        a     = _safe_param(item, "a", 0.0)

        func_id = int(item.get("function_id", 0))
        is_linear = 1 if item.get("type") == "linear" else 0
        gen_num = int(item.get("generator_number", 1))
        order = int(item.get("order", 2))

        features = [
            float(alpha), float(beta), float(n), float(M),
            float(func_id), float(is_linear), float(gen_num), float(order),
            float(q), float(v), float(a),
            float(np.random.randn() * 0.1),  # small noise for regularization
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
                # Remove an arbitrary (oldest inserted) entry
                try:
                    oldest = next(iter(self._feature_cache))
                    del self._feature_cache[oldest]
                except Exception:
                    self._feature_cache.clear()
                self._feature_cache[idx] = features

        return features, self.data[idx]

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._feature_cache),
        }


# ---------------------------------------------------------------------
# Iterable generator (large-scale streaming data)
# ---------------------------------------------------------------------
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
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed = seed

        # Initialize factories (optional)
        self.linear_factory = LinearGeneratorFactory() if LinearGeneratorFactory else None
        self.nonlinear_factory = NonlinearGeneratorFactory() if NonlinearGeneratorFactory else None

        # Function libraries
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()

        # Names (cap special list if you want to limit exotic functions)
        self.basic_func_names = list(self.basic_functions.get_function_names())
        self.special_func_names = list(self.special_functions.get_function_names())[:5]
        self.all_func_names = self.basic_func_names + self.special_func_names

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Generate data on-the-fly"""
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        generated = 0
        while generated < self.num_samples:
            try:
                # Base params
                params: Dict[str, Any] = {
                    "alpha": float(np.random.uniform(-5, 5)),
                    "beta":  float(np.random.uniform(0.1, 5)),
                    "n":     int(np.random.randint(1, 5)),
                    "M":     float(np.random.uniform(-5, 5)),
                }

                # Function choice
                func_name = np.random.choice(self.all_func_names)
                if func_name in self.basic_func_names:
                    f_z = self.basic_functions.get_function(func_name)
                    func_id = self.basic_func_names.index(func_name)
                else:
                    f_z = self.special_functions.get_function(func_name)
                    func_id = len(self.basic_func_names) + self.special_func_names.index(func_name)

                # Generator type
                gen_type = np.random.choice(["linear", "nonlinear"])

                # Call factories (if available); otherwise synthesize a sample
                if gen_type == "linear" and self.linear_factory:
                    gen_num = int(np.random.randint(1, 9))
                    local_params = dict(params)
                    if gen_num in [4, 5]:
                        local_params["a"] = float(np.random.uniform(1, 3))
                    result = self.linear_factory.create(gen_num, f_z, **local_params)
                elif gen_type == "nonlinear" and self.nonlinear_factory:
                    gen_num = int(np.random.randint(1, 11))
                    local_params = dict(params)
                    if gen_num in [1, 2, 4]:
                        local_params["q"] = int(np.random.randint(2, 6))
                    if gen_num in [2, 3, 5]:
                        local_params["v"] = int(np.random.randint(2, 6))
                    if gen_num in [4, 5, 9, 10]:
                        local_params["a"] = float(np.random.uniform(1, 5))
                    result = self.nonlinear_factory.create(gen_num, f_z, **local_params)
                    params.update({k: v for k, v in local_params.items() if k not in params})
                else:
                    # Fallback when factories are missing
                    gen_num = int(np.random.randint(1, 5))
                    result = {"order": int(np.random.randint(1, 4))}

                # Data item
                data_item = {
                    **params,
                    "function_name": func_name,
                    "function_id": int(func_id),
                    "type": gen_type,
                    "generator_number": int(gen_num),
                    "order": int(result.get("order", 2)),
                }

                # Features (12-dim)
                features = torch.tensor([
                    _safe_param(data_item, "alpha", 1.0),
                    _safe_param(data_item, "beta",  1.0),
                    _safe_param(data_item, "n",     1),
                    _safe_param(data_item, "M",     0.0),
                    float(func_id),
                    1.0 if gen_type == "linear" else 0.0,
                    float(gen_num),
                    float(data_item["order"]),
                    float(_safe_param(data_item, "q", 0)),
                    float(_safe_param(data_item, "v", 0)),
                    float(_safe_param(data_item, "a", 0.0)),
                    float(np.random.randn() * 0.1),
                ], dtype=torch.float32)

                yield features, data_item
                generated += 1

            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                continue

    def __len__(self) -> int:
        return self.num_samples


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
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
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        if self.model_type == 'vae':
            self.criterion = self._vae_loss
        else:
            self.criterion = nn.MSELoss()

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if (self.enable_mixed_precision and self.device.type == "cuda") else None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0,
            'best_val_loss': float('inf')
        }

        # Function factories (used by generate_new_ode)
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()

        # Optional: user-provided data
        self._user_dataset: Optional[ODEDataset] = None

    # --- Compatibility: optional dataset injection (no-op safe) ---
    def set_dataset(self, generated_odes: List[Dict[str, Any]], batch_results: Optional[List[Dict[str, Any]]] = None):
        """
        Optional hook used by the UI. You can pass previously generated ODEs.
        If not provided, training will fall back to the online generator.
        """
        data = []
        if isinstance(generated_odes, list):
            data.extend(generated_odes)
        if isinstance(batch_results, list):
            data.extend(batch_results)
        if data:
            self._user_dataset = ODEDataset(data, max_cache_size=min(1000, len(data)))
            logger.info(f"Injected user dataset with {len(data)} samples.")
        else:
            self._user_dataset = None

    # --- model creation ---
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

    # --- training ---
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
            samples: Number of training samples (for generator or synthesis)
            validation_split: Validation split ratio
            save_best: Whether to save best model
            use_generator: Use streaming dataset (recommended)
            checkpoint_interval: Save checkpoint every N epochs
            gradient_accumulation_steps: Accumulate grads for larger effective batch
            progress_callback: Optional callback for progress updates
        """
        logger.info(f"Starting training for {int(epochs)} epochs...")

        # Data loaders
        if use_generator and self._user_dataset is None:
            # Memory-efficient streaming
            train_samples = int(samples * (1 - validation_split))
            val_samples = int(samples) - train_samples

            train_dataset = ODEDataGenerator(train_samples, batch_size)
            val_dataset   = ODEDataGenerator(val_samples, batch_size, seed=42)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=0  # generator already yields batched tensors
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0
            )
        else:
            # Use injected dataset if present, otherwise synthesize a finite set
            if self._user_dataset is None:
                data = self._generate_training_data_batch(samples)
                dataset = ODEDataset(data, max_cache_size=min(1000, len(data)))
            else:
                dataset = self._user_dataset

            val_size = int(len(dataset) * validation_split)
            train_size = max(0, len(dataset) - val_size)
            if train_size <= 0:
                raise ValueError("Not enough data to split into train/val.")

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            pin = True if self.device.type == 'cuda' else False
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=pin)

        best_val_loss = float('inf')

        for epoch in range(int(epochs)):
            # Training
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (batch_features, _batch_payload) in enumerate(pbar):
                batch_features = batch_features.to(self.device)

                if self.enable_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        if self.model_type == 'vae':
                            recon, mu, log_var = self.model(batch_features)
                            loss = self.criterion(recon, batch_features, mu, log_var)
                        else:
                            output = self.model(batch_features)
                            loss = self.criterion(output, batch_features)

                    # gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
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
                pbar.set_postfix({'loss': train_loss / max(1, batch_count)})

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_features, _payload in val_loader:
                    batch_features = batch_features.to(self.device)
                    if self.model_type == 'vae':
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        output = self.model(batch_features)
                        loss = self.criterion(output, batch_features)

                    val_loss += loss.item()
                    val_batches += 1

            avg_train = train_loss / max(1, batch_count)
            avg_val   = val_loss / max(1, val_batches)

            # history + scheduler
            self.history['train_loss'].append(avg_train)
            self.history['val_loss'].append(avg_val)
            self.history['epochs'] = epoch + 1
            self.scheduler.step(avg_val)

            logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

            if progress_callback:
                try:
                    progress_callback(epoch + 1, int(epochs))
                except Exception:
                    pass

            # Save best
            if save_best and avg_val < best_val_loss:
                best_val_loss = avg_val
                self.history['best_val_loss'] = best_val_loss
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pth"))
                logger.info(f"Saved best model (val_loss={best_val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % int(checkpoint_interval) == 0:
                path = os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch+1}.pth")
                self.save_checkpoint(path, epoch + 1)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")

            # Cleanup
            if (epoch + 1) % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        logger.info("Training completed!")

    # --- synthetic finite data gen for non-generator path ---
    def _generate_training_data_batch(self, num_samples: int, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Generate training data in batches to save memory"""
        logger.info(f"Generating {int(num_samples)} training samples in batches...")

        data: List[Dict[str, Any]] = []
        generator = ODEDataGenerator(int(num_samples), int(batch_size))

        for _features, item in tqdm(generator, total=int(num_samples), desc="Generating data"):
            data.append(item)

        logger.info(f"Generated {len(data)} valid samples")
        return data

    # --- checkpointing ---
    def save_checkpoint(self, path: str, epoch: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': int(epoch),
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
            epoch = int(checkpoint.get('epoch', 0))
            logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
            return epoch
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0

    # --- model persistence ---
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
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint.get('history', self.history)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    # --- generation ---
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
                    # Random seed
                    seed = torch.randn(1, self.input_dim).to(self.device)
                    generated = self.model(seed)
            else:
                seed = seed.to(self.device)
                if self.model_type == 'vae':
                    generated, _, _ = self.model(seed)
                else:
                    generated = self.model(seed)

            params_tensor = generated.cpu().numpy()[0]

            # Decode (same 12-d layout)
            alpha = float(np.clip(params_tensor[0], -100, 100))
            beta  = float(np.clip(abs(params_tensor[1]) + 0.1, 0.1, 100))
            n     = int(np.clip(abs(params_tensor[2]), 1, 10))
            M     = float(np.clip(params_tensor[3], -100, 100))
            func_id_raw = int(abs(params_tensor[4]))
            is_linear   = bool(params_tensor[5] > 0.5)
            gen_num     = int(np.clip(abs(params_tensor[6]), 1, 8 if is_linear else 10))
            q_guess     = int(np.clip(abs(params_tensor[8]) + 2, 2, 10))
            v_guess     = int(np.clip(abs(params_tensor[9]) + 2, 2, 10))
            a_guess     = float(np.clip(abs(params_tensor[10]) + 1, 1, 5))

            # Choose function from combined list (consistent with generator)
            basic_names   = list(self.basic_functions.get_function_names())
            special_names = list(self.special_functions.get_function_names())[:5]
            all_names     = basic_names + special_names
            if len(all_names) == 0:
                logger.error("No functions available.")
                return None

            func_id = func_id_raw % len(all_names)
            func_name = all_names[func_id]
            if func_name in basic_names:
                f_z = self.basic_functions.get_function(func_name)
            else:
                f_z = self.special_functions.get_function(func_name)

            # Build params for generator
            generator_params = {
                'alpha': alpha,
                'beta':  beta,
                'n':     n,
                'M':     M
            }

            try:
                if is_linear and LinearGeneratorFactory:
                    factory = LinearGeneratorFactory()
                    if gen_num in [4, 5]:
                        generator_params['a'] = a_guess
                    result = factory.create(gen_num, f_z, **generator_params)
                elif (not is_linear) and NonlinearGeneratorFactory:
                    factory = NonlinearGeneratorFactory()
                    extra_params = {}
                    if gen_num in [1, 2, 4]:
                        extra_params['q'] = q_guess
                    if gen_num in [2, 3, 5]:
                        extra_params['v'] = v_guess
                    if gen_num in [4, 5, 9, 10]:
                        extra_params['a'] = a_guess
                    full = {**generator_params, **extra_params}
                    result = factory.create(gen_num, f_z, **full)
                else:
                    # Factories missing
                    result = {
                        "type": "linear" if is_linear else "nonlinear",
                        "order": 2,
                        "generator_number": gen_num,
                        "generator": sp.Symbol("LHS"),
                        "rhs": sp.Symbol("RHS"),
                        "solution": sp.Symbol("y(x)")
                    }

                # annotate
                result['function_used'] = func_name
                result['ml_generated'] = True
                result['generation_params'] = generator_params
                return result

            except Exception as e:
                logger.error(f"Error generating ODE: {e}")
                return None

    # --- evaluation ---
    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate model on test data
        """
        self.model.eval()
        dataset = ODEDataset(test_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_features, _batch_data in loader:
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

        predictions = torch.cat(predictions) if predictions else torch.empty(0)
        targets = torch.cat(targets) if targets else torch.empty(0)
        if predictions.numel() == 0 or targets.numel() == 0:
            return {'loss': float('nan'), 'mse': float('nan'), 'mae': float('nan'), 'correlation': float('nan')}

        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()

        pred_flat = predictions.flatten().numpy()
        target_flat = targets.flatten().numpy()
        correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1]) if pred_flat.size > 1 else float('nan')

        return {
            'loss': total_loss / max(1, len(loader)),
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }

    # --- export ---
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