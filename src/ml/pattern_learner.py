"""
Machine Learning Pattern Learner for ODE Generators
Neural network architecture for learning generator patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

class GeneratorPatternLearner(nn.Module):
    """
    Neural network to learn patterns in ODE generators
    Uses an encoder-decoder architecture with attention mechanism
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, 
                 output_dim: int = 10, num_layers: int = 3, dropout: float = 0.2):
        """
        Initialize the pattern learner
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super(GeneratorPatternLearner, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Encoder network
        encoder_layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i == 0 else hidden_dim * (2 if i == 1 else 1)
            encoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.BatchNorm1d(next_dim),
                nn.Dropout(dropout)
            ])
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=current_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Decoder network
        decoder_layers = []
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            decoder_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                nn.BatchNorm1d(next_dim) if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])
            current_dim = next_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Residual connection
        self.residual = nn.Linear(input_dim, output_dim) if input_dim == output_dim else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Apply attention (self-attention)
        # Reshape for attention: (batch, 1, features)
        encoded_reshaped = encoded.unsqueeze(1)
        attended, _ = self.attention(encoded_reshaped, encoded_reshaped, encoded_reshaped)
        attended = attended.squeeze(1)
        
        # Decode
        decoded = self.decoder(attended)
        
        # Add residual connection if applicable
        if self.residual is not None:
            decoded = decoded + self.residual(x)
        
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get encoded representation
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent representation
        
        Args:
            z: Latent tensor
            
        Returns:
            Decoded output
        """
        return self.decoder(z)
    
    def get_latent_dim(self) -> int:
        """
        Get the dimension of the latent space
        
        Returns:
            Latent dimension
        """
        return self.hidden_dim * 2  # Peak dimension in encoder

class GeneratorVAE(nn.Module):
    """
    Variational Autoencoder for ODE generator patterns
    Enables generation of new patterns through sampling
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, 
                 latent_dim: int = 32):
        """
        Initialize VAE
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent space dimension
        """
        super(GeneratorVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2)
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x: Input tensor
            
        Returns:
            Mean and log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed output
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed output, mean, and log variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample new patterns from the latent space
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

class GeneratorTransformer(nn.Module):
    """
    Transformer-based model for ODE generator pattern learning
    """
    
    def __init__(self, input_dim: int = 10, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 3):
        """
        Initialize transformer model
        
        Args:
            input_dim: Input dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super(GeneratorTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               or (batch_size, input_dim)
            
        Returns:
            Output tensor
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Pool over sequence dimension if needed
        if seq_len > 1:
            x = x.mean(dim=1)
        else:
            x = x.squeeze(1)
        
        # Project to output
        output = self.output_projection(x)
        
        return output

def create_model(model_type: str = 'pattern_learner', **kwargs) -> nn.Module:
    """
    Factory function to create different model types
    
    Args:
        model_type: Type of model ('pattern_learner', 'vae', 'transformer')
        **kwargs: Model-specific parameters
        
    Returns:
        Initialized model
    """
    models = {
        'pattern_learner': GeneratorPatternLearner,
        'vae': GeneratorVAE,
        'transformer': GeneratorTransformer
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)