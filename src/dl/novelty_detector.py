"""
Deep Learning Novelty Detector for ODEs
Uses transformer architecture to detect novel and unsolvable ODEs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from dataclasses import dataclass

@dataclass
class NoveltyAnalysis:
    """Results of novelty analysis"""
    is_novel: bool
    novelty_score: float
    confidence: float
    complexity_level: str
    solvable_by_standard_methods: bool
    recommended_methods: List[str]
    special_characteristics: List[str]
    similar_known_equations: List[str]
    detailed_report: Optional[str] = None

class ODETokenizer:
    """Tokenizer for ODE expressions"""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id['[PAD]']
        self.unk_token_id = self.token_to_id['[UNK]']
        
    def _build_vocab(self) -> List[str]:
        """Build vocabulary for ODE tokenization"""
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        
        # Mathematical operators
        operators = ['+', '-', '*', '/', '^', '**', '=', '(', ')', '[', ']', '{', '}']
        
        # Common functions
        functions = [
            'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs',
            'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan',
            'airy', 'bessel', 'gamma', 'erf', 'zeta', 'legendre',
            'hermite', 'chebyshev', 'laguerre'
        ]
        
        # Variables and derivatives
        variables = ['x', 'y', 't', 'z', 'dx', 'dy', 'dt', 'dz']
        derivatives = ["y'", "y''", "y'''", "y''''", "d/dx", "d²/dx²", "∂/∂x"]
        
        # Numbers and constants
        numbers = [str(i) for i in range(10)]
        constants = ['pi', 'e', 'i', 'inf', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'ω']
        
        return special_tokens + operators + functions + variables + derivatives + numbers + constants
    
    def tokenize(self, ode_str: str) -> List[str]:
        """Tokenize ODE string"""
        # Clean and normalize
        ode_str = str(ode_str).lower()
        ode_str = re.sub(r'\s+', ' ', ode_str)
        
        # Split into tokens
        tokens = []
        current_token = ""
        
        for char in ode_str:
            if char in self.vocab:
                if current_token:
                    tokens.append(current_token if current_token in self.vocab else '[UNK]')
                    current_token = ""
                tokens.append(char)
            elif char.isalnum() or char in ".'":
                current_token += char
            else:
                if current_token:
                    tokens.append(current_token if current_token in self.vocab else '[UNK]')
                    current_token = ""
        
        if current_token:
            tokens.append(current_token if current_token in self.vocab else '[UNK]')
        
        return tokens
    
    def encode(self, ode_str: str) -> torch.Tensor:
        """Encode ODE string to tensor"""
        tokens = self.tokenize(ode_str)
        
        # Add special tokens
        tokens = ['[CLS]'] + tokens[:self.max_length-2] + ['[SEP]']
        
        # Convert to ids
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        
        # Pad to max length
        if len(ids) < self.max_length:
            ids += [self.pad_token_id] * (self.max_length - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)

class ODETransformer(nn.Module):
    """Transformer model for ODE analysis"""
    
    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        max_length: int = 512,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification heads
        self.novelty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.complexity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 4)  # 4 complexity levels
        )
        
        self.solvability_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2)  # Solvable/Not solvable
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + position_embeds
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Transform attention mask for transformer
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
        
        # Apply transformer
        hidden_states = self.transformer(embeddings, src_key_padding_mask=attention_mask)
        
        # Pool (use CLS token representation)
        pooled = hidden_states[:, 0, :]
        
        # Apply classification heads
        novelty_logits = self.novelty_head(pooled)
        complexity_logits = self.complexity_head(pooled)
        solvability_logits = self.solvability_head(pooled)
        
        return {
            'novelty': novelty_logits,
            'complexity': complexity_logits,
            'solvability': solvability_logits,
            'hidden_states': hidden_states
        }

class ODENoveltyDetector:
    """Main novelty detector class"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.tokenizer = ODETokenizer()
        self.model = ODETransformer(vocab_size=len(self.tokenizer.vocab))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with pretrained weights if available
            self._initialize_weights()
        
        self.model.eval()
        
        # Known equation patterns
        self.known_patterns = self._load_known_patterns()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _load_known_patterns(self) -> Dict[str, List[str]]:
        """Load database of known equation patterns"""
        return {
            'linear_constant_coef': [
                r"y'' \+ .*y' \+ .*y = .*",
                r"y''' \+ .*y'' \+ .*y' \+ .*y = .*"
            ],
            'bessel': [
                r".*y'' \+ .*y' \+ .*\(.*- n\^2.*\)y = .*",
                r"x\^2.*y'' \+ x.*y' \+ \(x\^2 - n\^2\).*y = .*"
            ],
            'airy': [
                r"y'' - x.*y = .*",
                r"y'' - z.*y = .*"
            ],
            'legendre': [
                r"\(1-x\^2\).*y'' - 2x.*y' \+ n\(n\+1\).*y = .*"
            ],
            'hermite': [
                r"y'' - 2x.*y' \+ 2n.*y = .*"
            ],
            'nonlinear_power': [
                r"\(y''\)\^[0-9]+ .*",
                r"\(y'\)\^[0-9]+ .*"
            ],
            'transcendental': [
                r".*sin\(y.*\).*",
                r".*exp\(y.*\).*",
                r".*log\(y.*\).*"
            ]
        }
    
    def analyze(
        self,
        ode_dict: Dict[str, Any],
        check_solvability: bool = True,
        detailed: bool = False
    ) -> NoveltyAnalysis:
        """Analyze ODE for novelty"""
        ode_str = str(ode_dict['ode'])
        
        # Tokenize and encode
        input_ids = self.tokenizer.encode(ode_str).unsqueeze(0).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(input_ids)
        
        # Process outputs
        novelty_probs = F.softmax(outputs['novelty'], dim=-1)
        complexity_probs = F.softmax(outputs['complexity'], dim=-1)
        solvability_probs = F.softmax(outputs['solvability'], dim=-1)
        
        # Determine results
        is_novel = novelty_probs[0, 1].item() > 0.5
        novelty_score = novelty_probs[0, 1].item() * 100
        
        complexity_level = self._get_complexity_level(complexity_probs[0])
        solvable = solvability_probs[0, 1].item() > 0.5 if check_solvability else None
        
        # Check against known patterns
        pattern_matches = self._check_patterns(ode_str)
        
        # Get special characteristics
        characteristics = self._extract_characteristics(ode_dict)
        
        # Get recommended methods
        methods = self._recommend_methods(
            ode_dict,
            novelty_score,
            complexity_level,
            pattern_matches
        )
        
        # Find similar equations
        similar = self._find_similar_equations(ode_str, pattern_matches)
        
        # Generate detailed report if requested
        report = None
        if detailed:
            report = self._generate_detailed_report(
                ode_dict,
                novelty_score,
                complexity_level,
                characteristics,
                pattern_matches
            )
        
        return NoveltyAnalysis(
            is_novel=is_novel,
            novelty_score=novelty_score,
            confidence=max(novelty_probs[0]).item(),
            complexity_level=complexity_level,
            solvable_by_standard_methods=solvable,
            recommended_methods=methods,
            special_characteristics=characteristics,
            similar_known_equations=similar,
            detailed_report=report
        )
    
    def _get_complexity_level(self, probs: torch.Tensor) -> str:
        """Determine complexity level from probabilities"""
        levels = ['Simple', 'Moderate', 'Complex', 'Highly Complex']
        idx = torch.argmax(probs).item()
        return levels[idx]
    
    def _check_patterns(self, ode_str: str) -> List[str]:
        """Check ODE against known patterns"""
        matches = []
        for category, patterns in self.known_patterns.items():
            for pattern in patterns:
                if re.search(pattern, ode_str, re.IGNORECASE):
                    matches.append(category)
                    break
        return matches
    
    def _extract_characteristics(self, ode_dict: Dict[str, Any]) -> List[str]:
        """Extract special characteristics from ODE"""
        characteristics = []
        ode_str = str(ode_dict['ode'])
        
        # Check for special functions
        special_funcs = ['airy', 'bessel', 'gamma', 'zeta', 'legendre', 'hermite']
        for func in special_funcs:
            if func in ode_str.lower():
                characteristics.append(f"Contains {func} function")
        
        # Check for nonlinearity
        if ode_dict.get('type') == 'nonlinear':
            powers = ode_dict.get('powers', {})
            for var, power in powers.items():
                characteristics.append(f"Nonlinear: {var}^{power}")
        
        # Check for delay/pantograph
        if 'x/a' in ode_str or 'pantograph' in ode_dict.get('subtype', ''):
            characteristics.append("Delay/Pantograph equation")
        
        # Check for transcendental functions
        trans_funcs = ['sin', 'cos', 'exp', 'log', 'tan']
        for func in trans_funcs:
            if func in ode_str and f"{func}(y" in ode_str:
                characteristics.append(f"Transcendental: {func} of dependent variable")
        
        # Check order
        order = ode_dict.get('order', 0)
        if order > 3:
            characteristics.append(f"High order: {order}")
        
        return characteristics
    
    def _recommend_methods(
        self,
        ode_dict: Dict[str, Any],
        novelty_score: float,
        complexity_level: str,
        pattern_matches: List[str]
    ) -> List[str]:
        """Recommend solution methods"""
        methods = []
        
        # Based on novelty score
        if novelty_score < 30:
            methods.extend([
                'Analytical solution',
                'Separation of variables',
                'Integrating factor method'
            ])
        elif novelty_score < 60:
            methods.extend([
                'Power series method',
                'Laplace transform',
                'Fourier transform',
                'Numerical methods (RK4, RK45)'
            ])
        else:
            methods.extend([
                'Master generators method',
                'Advanced numerical schemes',
                'Asymptotic methods',
                'Special function expansions'
            ])
        
        # Based on pattern matches
        if 'bessel' in pattern_matches:
            methods.append('Bessel function solutions')
        if 'airy' in pattern_matches:
            methods.append('Airy function solutions')
        if 'legendre' in pattern_matches:
            methods.append('Legendre polynomial expansion')
        
        # Based on type
        if ode_dict.get('type') == 'nonlinear':
            methods.extend([
                'Perturbation methods',
                'Variational methods',
                'Fixed-point iteration'
            ])
        
        # Remove duplicates and return
        return list(set(methods))
    
    def _find_similar_equations(
        self,
        ode_str: str,
        pattern_matches: List[str]
    ) -> List[str]:
        """Find similar known equations"""
        similar = []
        
        if 'bessel' in pattern_matches:
            similar.append("Bessel's equation: x²y'' + xy' + (x² - n²)y = 0")
        if 'airy' in pattern_matches:
            similar.append("Airy equation: y'' - xy = 0")
        if 'legendre' in pattern_matches:
            similar.append("Legendre equation: (1-x²)y'' - 2xy' + n(n+1)y = 0")
        if 'hermite' in pattern_matches:
            similar.append("Hermite equation: y'' - 2xy' + 2ny = 0")
        
        return similar
    
    def _generate_detailed_report(
        self,
        ode_dict: Dict[str, Any],
        novelty_score: float,
        complexity_level: str,
        characteristics: List[str],
        pattern_matches: List[str]
    ) -> str:
        """Generate detailed analysis report"""
        report = f"""
DETAILED ODE ANALYSIS REPORT
============================

1. EQUATION INFORMATION
-----------------------
Type: {ode_dict.get('type', 'Unknown')}
Order: {ode_dict.get('order', 'Unknown')}
Generator: {ode_dict.get('generator_number', 'N/A')}

2. NOVELTY ASSESSMENT
---------------------
Novelty Score: {novelty_score:.2f}/100
Classification: {'Novel' if novelty_score > 50 else 'Standard'}
Complexity Level: {complexity_level}

3. SPECIAL CHARACTERISTICS
--------------------------
{chr(10).join('• ' + c for c in characteristics) if characteristics else 'None identified'}

4. PATTERN MATCHES
------------------
{chr(10).join('• ' + p for p in pattern_matches) if pattern_matches else 'No known patterns matched'}

5. SOLVABILITY ANALYSIS
-----------------------
The equation appears to be {'solvable' if novelty_score < 50 else 'challenging to solve'} 
using standard methods.

6. RECOMMENDATIONS
------------------
Based on the analysis, the following approaches are recommended:
{chr(10).join('• ' + m for m in self._recommend_methods(ode_dict, novelty_score, complexity_level, pattern_matches)[:5])}

7. RESEARCH VALUE
-----------------
{'HIGH' if novelty_score > 70 else 'MODERATE' if novelty_score > 40 else 'LOW'}: 
This equation {'presents significant research opportunities' if novelty_score > 70 else 
'may offer some interesting properties' if novelty_score > 40 else 
'is well-understood in the literature'}.

Generated by Master Generators ODE System v1.0
"""
        return report
    
    def check_novelty(self, ode_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Simple novelty check (backward compatibility)"""
        analysis = self.analyze(ode_dict)
        
        return {
            'is_novel': analysis.is_novel,
            'novelty_score': analysis.novelty_score,
            'solvable_by_standard_methods': analysis.solvable_by_standard_methods,
            'complexity_level': analysis.complexity_level,
            'recommended_methods': analysis.recommended_methods
        }
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
