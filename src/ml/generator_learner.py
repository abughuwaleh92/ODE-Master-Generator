# src/ml/generator_learner.py
"""
Enhanced ML System for Learning Generator Patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict

@dataclass
class GeneratorPattern:
    """Represents a generator pattern for ML training"""
    derivative_orders: List[int]
    coefficients: List[float]
    function_types: List[str]
    powers: List[int]
    scalings: List[Optional[float]]
    is_linear: bool
    order: int
    has_special_features: Dict[str, bool]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation"""
        features = []
        
        # Encode derivative orders (up to 10th order)
        order_encoding = [0] * 10
        for order in self.derivative_orders:
            if order < 10:
                order_encoding[order] = 1
        features.extend(order_encoding)
        
        # Encode coefficients (normalized)
        coeff_encoding = [0] * 10
        for i, coeff in enumerate(self.coefficients[:10]):
            coeff_encoding[i] = np.tanh(coeff / 10)  # Normalize
        features.extend(coeff_encoding)
        
        # Encode function types
        func_types = ['linear', 'power', 'exponential', 'trigonometric', 'logarithmic']
        func_encoding = [0] * len(func_types)
        for func_type in self.function_types:
            if func_type in func_types:
                func_encoding[func_types.index(func_type)] = 1
        features.extend(func_encoding)
        
        # Encode powers
        power_encoding = [0] * 5
        for i, power in enumerate(self.powers[:5]):
            power_encoding[i] = power / 10  # Normalize
        features.extend(power_encoding)
        
        # Binary features
        features.append(1 if self.is_linear else 0)
        features.append(self.order / 10)  # Normalize order
        
        # Special features
        for feature in ['delay', 'trigonometric', 'exponential']:
            features.append(1 if self.has_special_features.get(feature, False) else 0)
        
        return torch.tensor(features, dtype=torch.float32)

class GeneratorPatternNetwork(nn.Module):
    """
    Neural network for learning generator patterns
    """
    
    def __init__(self, input_dim: int = 35, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Generator predictor
        self.generator_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Predict generator type and parameters
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        gen_params = self.generator_predictor(z)
        return recon, mu, log_var, gen_params
    
    def generate_new_pattern(self, num_samples: int = 1) -> List[GeneratorPattern]:
        """Generate new generator patterns"""
        patterns = []
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.fc_mu.out_features)
            
            # Decode to get pattern features
            features = self.decode(z)
            gen_params = self.generator_predictor(z)
            
            # Convert to GeneratorPattern objects
            for i in range(num_samples):
                feat = features[i].numpy()
                params = gen_params[i].numpy()
                
                # Decode features back to pattern
                pattern = self._decode_features(feat, params)
                patterns.append(pattern)
        
        return patterns
    
    def _decode_features(self, features: np.ndarray, params: np.ndarray) -> GeneratorPattern:
        """Decode feature vector back to GeneratorPattern"""
        
        # Decode derivative orders
        derivative_orders = []
        for i in range(10):
            if features[i] > 0.5:
                derivative_orders.append(i)
        
        if not derivative_orders:
            derivative_orders = [0, 2]  # Default
        
        # Decode coefficients
        coefficients = []
        for i in range(10, 20):
            if features[i] != 0:
                coefficients.append(float(np.arctanh(np.clip(features[i], -0.99, 0.99)) * 10))
        
        if not coefficients:
            coefficients = [1.0] * len(derivative_orders)
        
        # Decode function types
        func_types = ['linear', 'power', 'exponential', 'trigonometric', 'logarithmic']
        function_types = []
        for i, func_type in enumerate(func_types):
            if features[20 + i] > 0.5:
                function_types.append(func_type)
        
        if not function_types:
            function_types = ['linear']
        
        # Decode powers
        powers = []
        for i in range(25, 30):
            power = int(features[i] * 10)
            if power > 0:
                powers.append(power)
        
        if not powers:
            powers = [1] * len(derivative_orders)
        
        # Decode other features
        is_linear = features[30] > 0.5
        order = int(features[31] * 10)
        
        has_special_features = {
            'delay': features[32] > 0.5,
            'trigonometric': features[33] > 0.5,
            'exponential': features[34] > 0.5
        }
        
        # Use generator parameters to refine
        if params[0] > 0.5:  # Prefer nonlinear
            is_linear = False
        
        return GeneratorPattern(
            derivative_orders=derivative_orders,
            coefficients=coefficients[:len(derivative_orders)],
            function_types=function_types[:len(derivative_orders)],
            powers=powers[:len(derivative_orders)],
            scalings=[None] * len(derivative_orders),
            is_linear=is_linear,
            order=max(derivative_orders) if derivative_orders else 2,
            has_special_features=has_special_features
        )

class GeneratorLearningSystem:
    """
    Complete system for learning and generating ODE generators
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.network = GeneratorPatternNetwork()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.training_history = []
    
    def train_on_generators(
        self,
        generator_patterns: List[GeneratorPattern],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """Train on generator patterns"""
        
        # Convert patterns to tensors
        tensors = [pattern.to_tensor() for pattern in generator_patterns]
        dataset = torch.stack(tensors)
        
        # Split into train and validation
        n_val = int(len(dataset) * validation_split)
        train_data = dataset[:-n_val] if n_val > 0 else dataset
        val_data = dataset[-n_val:] if n_val > 0 else None
        
        # Training loop
        for epoch in range(epochs):
            self.network.train()
            train_loss = 0
            
            # Mini-batch training
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                recon, mu, log_var, gen_params = self.network(batch)
                
                # Loss calculation
                recon_loss = F.mse_loss(recon, batch, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                
                loss = recon_loss + kld_loss
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_data is not None:
                self.network.eval()
                with torch.no_grad():
                    val_batch = val_data.to(self.device)
                    val_recon, val_mu, val_log_var, _ = self.network(val_batch)
                    val_recon_loss = F.mse_loss(val_recon, val_batch, reduction='sum')
                    val_kld_loss = -0.5 * torch.sum(1 + val_log_var - val_mu.pow(2) - val_log_var.exp())
                    val_loss = val_recon_loss + val_kld_loss
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_data):.4f}, "
                      f"Val Loss: {val_loss/len(val_data):.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_data):.4f}")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_data),
                'val_loss': val_loss.item() / len(val_data) if val_data is not None else None
            })
    
    def generate_novel_generator(self) -> Tuple[GeneratorPattern, Dict[str, Any]]:
        """Generate a novel generator pattern"""
        
        # Generate new pattern
        patterns = self.network.generate_new_pattern(1)
        pattern = patterns[0]
        
        # Assess novelty
        novelty_score = self._assess_novelty(pattern)
        
        # Convert to generator specification
        from src.generators.generator_constructor import GeneratorConstructor, DerivativeTerm, DerivativeType
        
        constructor = GeneratorConstructor()
        terms = []
        
        for i, order in enumerate(pattern.derivative_orders):
            func_type_str = pattern.function_types[i] if i < len(pattern.function_types) else 'linear'
            
            # Map string to enum
            func_type_map = {
                'linear': DerivativeType.LINEAR,
                'power': DerivativeType.POWER,
                'exponential': DerivativeType.EXPONENTIAL,
                'trigonometric': DerivativeType.TRIGONOMETRIC,
                'logarithmic': DerivativeType.LOGARITHMIC
            }
            
            term = DerivativeTerm(
                derivative_order=order,
                coefficient=pattern.coefficients[i] if i < len(pattern.coefficients) else 1.0,
                power=pattern.powers[i] if i < len(pattern.powers) else 1,
                function_type=func_type_map.get(func_type_str, DerivativeType.LINEAR),
                scaling=pattern.scalings[i] if i < len(pattern.scalings) else None
            )
            terms.append(term)
        
        generator_spec = constructor.construct_generator(terms)
        
        return pattern, {
            'generator_spec': generator_spec,
            'novelty_score': novelty_score,
            'is_novel': novelty_score > 70
        }
    
    def _assess_novelty(self, pattern: GeneratorPattern) -> float:
        """Assess the novelty of a generator pattern"""
        
        novelty_score = 50.0  # Base score
        
        # Unusual derivative combinations
        if len(pattern.derivative_orders) > 3:
            novelty_score += 10
        
        # Non-consecutive derivatives
        if pattern.derivative_orders and max(pattern.derivative_orders) - min(pattern.derivative_orders) > len(pattern.derivative_orders):
            novelty_score += 15
        
        # Mixed function types
        if len(set(pattern.function_types)) > 1:
            novelty_score += 20
        
        # Nonlinear features
        if not pattern.is_linear:
            novelty_score += 15
        
        # Special features
        special_count = sum(1 for v in pattern.has_special_features.values() if v)
        novelty_score += special_count * 10
        
        # High order
        if pattern.order > 4:
            novelty_score += (pattern.order - 4) * 5
        
        return min(novelty_score, 100.0)
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
