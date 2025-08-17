# src/ml/trainer.py
"""
Improved ML Trainer for ODE Generators
- Persistent progress callbacks
- Masked/weighted loss (ignore noise dim, down-weight discrete dims)
- Optional feature normalization
- Early stopping
- VAE KL annealing / beta-VAE
- Gradient clipping
- Session export/import to single ZIP (for RQ artifacts)
"""

import io, os, gc, json, zipfile, logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- existing model imports (unchanged) ---
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
from src.ml.pattern_learner import (
    GeneratorPatternLearner,
    GeneratorVAE,
    GeneratorTransformer,
)

# ---------------- Dataset / Generator (unchanged, with small guard) ----------------
class ODEDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], max_cache_size: int = 1000):
        self.data = data
        self.max_cache_size = min(max_cache_size, len(data))
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        features = [
            float(item.get('alpha', 1.0)),
            float(item.get('beta', 1.0)),
            float(item.get('n', 1)),
            float(item.get('M', 0.0)),
            float(item.get('function_id', 0)),
            1.0 if item.get('type') == 'linear' else 0.0,
            float(item.get('generator_number', 1)),
            float(item.get('order', 2)),
            float(item.get('q', 0)),
            float(item.get('v', 0)),
            float(item.get('a', 0.0)),
            np.random.randn() * 0.1,  # small noise
        ]
        return torch.tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if idx in self._feature_cache:
            self._cache_hits += 1
            features = self._feature_cache[idx]
        else:
            self._cache_misses += 1
            features = self._extract_features(self.data[idx])
            if len(self._feature_cache) < self.max_cache_size:
                self._feature_cache[idx] = features
        return features, self.data[idx]

class ODEDataGenerator(IterableDataset):
    def __init__(self, num_samples: int, batch_size: int = 32, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.linear_factory = LinearGeneratorFactory()
        self.nonlinear_factory = NonlinearGeneratorFactory()
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()
        self.basic_func_names = self.basic_functions.get_function_names()
        self.special_func_names = self.special_functions.get_function_names()[:5]
        self.all_func_names = self.basic_func_names + self.special_func_names

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        generated = 0
        while generated < self.num_samples:
            try:
                params = {
                    'alpha': float(np.random.uniform(-5, 5)),
                    'beta':  float(np.random.uniform(0.1, 5)),
                    'n':     int(np.random.randint(1, 5)),
                    'M':     float(np.random.uniform(-5, 5))
                }
                func_name = np.random.choice(self.all_func_names)
                if func_name in self.basic_func_names:
                    f_z = self.basic_functions.get_function(func_name)
                    func_id = self.basic_func_names.index(func_name)
                else:
                    f_z = self.special_functions.get_function(func_name)
                    func_id = len(self.basic_func_names) + self.special_func_names.index(func_name)

                gen_type = np.random.choice(['linear', 'nonlinear'])
                if gen_type == 'linear':
                    gen_num = int(np.random.randint(1, 9))
                    if gen_num in [4, 5]:
                        params['a'] = float(np.random.uniform(1, 3))
                    result = self.linear_factory.create(gen_num, f_z, **params)
                else:
                    gen_num = int(np.random.randint(1, 11))
                    extra = {}
                    if gen_num in [1, 2, 4]: extra['q'] = int(np.random.randint(2, 6))
                    if gen_num in [2, 3, 5]: extra['v'] = int(np.random.randint(2, 6))
                    if gen_num in [4, 5, 9, 10]: extra['a'] = float(np.random.uniform(1, 5))
                    result = self.nonlinear_factory.create(gen_num, f_z, **{**params, **extra})
                    params.update(extra)

                data_item = {
                    **params, 'function_name': func_name, 'function_id': func_id,
                    'type': gen_type, 'generator_number': gen_num, 'order': result['order']
                }
                features = torch.tensor([
                    params['alpha'], params['beta'], params['n'], params['M'],
                    float(func_id), 1.0 if gen_type == 'linear' else 0.0,
                    float(gen_num), float(result['order']),
                    float(params.get('q', 0)), float(params.get('v', 0)),
                    float(params.get('a', 0.0)), np.random.randn() * 0.1
                ], dtype=torch.float32)
                yield features, data_item
                generated += 1
            except Exception:
                continue

    def __len__(self) -> int:
        return self.num_samples

# ---------------- Trainer ----------------
class MLTrainer:
    def __init__(
        self,
        model_type: str = 'pattern_learner',
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 12,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
        checkpoint_dir: str = 'checkpoints',
        enable_mixed_precision: bool = False,
        beta_kl: float = 1.0,           # for VAE
        kl_warmup_epochs: int = 5       # for VAE
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.enable_mixed_precision = enable_mixed_precision
        self.beta_kl = float(beta_kl)
        self.kl_warmup_epochs = int(kl_warmup_epochs)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.scaler = torch.cuda.amp.GradScaler() if (enable_mixed_precision and self.device.type == 'cuda') else None

        self.history = {'train_loss': [], 'val_loss': [], 'epochs': 0, 'best_val_loss': float('inf')}
        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()

        # feature loss weights (mask noise + downweight discrete)
        self.loss_weights = None
        self._init_default_loss_weights()

        # optional normalization
        self.normalize = False
        self._mu = torch.zeros(1, input_dim, dtype=torch.float32)
        self._sigma = torch.ones(1, input_dim, dtype=torch.float32)

    # ----- model factory -----
    def _create_model(self) -> nn.Module:
        if self.model_type == 'pattern_learner':
            return GeneratorPatternLearner(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        elif self.model_type == 'vae':
            return GeneratorVAE(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=32)
        elif self.model_type == 'transformer':
            return GeneratorTransformer(input_dim=self.input_dim, d_model=self.hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # ----- loss helpers -----
    def _init_default_loss_weights(self):
        w = torch.ones(self.input_dim, dtype=torch.float32)
        w[4] = 0.3   # function_id (discrete-ish)
        w[5] = 0.2   # is_linear
        w[6] = 0.3   # generator_number
        w[11] = 0.0  # ignore noise feature
        self.loss_weights = w.view(1, -1)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.loss_weights.to(pred.device)
        diff2 = (pred - target) ** 2
        return torch.mean(diff2 * w)

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        mu = self._mu.to(x.device)
        sigma = self._sigma.to(x.device)
        return (x - mu) / (sigma + 1e-6)

    # ----- session I/O -----
    def save_session_bytes(self) -> bytes:
        """
        Export session (model+opt+sched+scaler+history+config) to a ZIP bytes for Redis artifact.
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # state
            payload = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': self.history,
                'config': {
                    'model_type': self.model_type,
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'output_dim': self.output_dim,
                    'learning_rate': self.learning_rate,
                    'beta_kl': self.beta_kl,
                    'kl_warmup_epochs': self.kl_warmup_epochs,
                    'normalize': self.normalize
                }
            }
            if self.scaler:
                payload['scaler_state_dict'] = self.scaler.state_dict()
            state_buf = io.BytesIO()
            torch.save(payload, state_buf)
            zf.writestr("session.pt", state_buf.getvalue())
            # also write light summaries
            zf.writestr("history.json", json.dumps(self.history))
            zf.writestr("config.json", json.dumps(payload["config"]))
        buf.seek(0)
        return buf.getvalue()

    def load_session_bytes(self, data: bytes) -> None:
        """
        Load session from ZIP bytes (created by save_session_bytes).
        """
        zf = zipfile.ZipFile(io.BytesIO(data), "r")
        raw = zf.read("session.pt")
        payload = torch.load(io.BytesIO(raw), map_location=self.device)
        self.model.load_state_dict(payload['model_state_dict'])
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.scheduler.load_state_dict(payload['scheduler_state_dict'])
        self.history = payload.get('history', self.history)
        cfg = payload.get('config', {})
        self.model_type = cfg.get('model_type', self.model_type)
        self.normalize = bool(cfg.get('normalize', False))
        if self.scaler and 'scaler_state_dict' in payload:
            self.scaler.load_state_dict(payload['scaler_state_dict'])

    def save_model(self, path: str):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'model_type': self.model_type,
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'output_dim': self.output_dim}, path)
        logger.info(f"Model saved to {path}")

    # ----- training -----
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
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        early_stop_patience: int = 10,
        grad_clip_norm: float = 1.0
    ):
        logger.info(f"Starting training for {epochs} epochs...")
        # prepare data
        if use_generator:
            train_samples = int(samples * (1 - validation_split))
            val_samples = samples - train_samples
            train_loader = DataLoader(ODEDataGenerator(train_samples, batch_size),
                                      batch_size=batch_size, num_workers=0)
            val_loader   = DataLoader(ODEDataGenerator(val_samples, batch_size, seed=42),
                                      batch_size=batch_size, num_workers=0)
        else:
            # fallback: cache + split (reproducible val)
            data = [d for _, d in ODEDataGenerator(samples, batch_size)]
            dataset = ODEDataset(data, max_cache_size=min(1000, len(data)))
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        best_val = float('inf')
        bad_epochs = 0

        for epoch in range(epochs):
            self.model.train()
            tr_loss = 0.0
            nb = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for bi, (x, _) in enumerate(pbar):
                x = x.to(self.device)
                x = self._pre(x)
                if self.enable_mixed_precision and self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        if self.model_type == 'vae':
                            recon, mu, log_var = self.model(x)
                            # KL anneal / beta-VAE
                            klw = self.beta_kl * min(1.0, (epoch+1) / max(1, self.kl_warmup_epochs))
                            recon_loss = self._masked_mse(recon, x)
                            kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                            loss = recon_loss + klw * kld
                        else:
                            out = self.model(x)
                            loss = self._masked_mse(out, x)
                    loss = loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    if (bi + 1) % gradient_accumulation_steps == 0:
                        if grad_clip_norm and grad_clip_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    if self.model_type == 'vae':
                        recon, mu, log_var = self.model(x)
                        klw = self.beta_kl * min(1.0, (epoch+1) / max(1, self.kl_warmup_epochs))
                        recon_loss = self._masked_mse(recon, x)
                        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon_loss + klw * kld
                    else:
                        out = self.model(x)
                        loss = self._masked_mse(out, x)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    if (bi + 1) % gradient_accumulation_steps == 0:
                        if grad_clip_norm and grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                tr_loss += loss.item() * gradient_accumulation_steps
                nb += 1
                pbar.set_postfix({'loss': tr_loss / max(1, nb)})

            # validation
            self.model.eval()
            vl = 0.0; vb = 0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = self._pre(x.to(self.device))
                    if self.model_type == 'vae':
                        recon, mu, log_var = self.model(x)
                        klw = self.beta_kl  # eval with full weight
                        recon_loss = self._masked_mse(recon, x)
                        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon_loss + klw * kld
                    else:
                        out = self.model(x)
                        loss = self._masked_mse(out, x)
                    vl += loss.item()
                    vb += 1

            avg_tr = tr_loss / max(1, nb)
            avg_vl = vl / max(1, vb)

            self.history['train_loss'].append(avg_tr)
            self.history['val_loss'].append(avg_vl)
            self.history['epochs'] = epoch + 1
            self.scheduler.step(avg_vl)
            logger.info(f"Epoch {epoch+1}: Train {avg_tr:.4f} | Val {avg_vl:.4f}")

            # progress callback for RQ/Streamlit visibility
            if progress_callback:
                try:
                    progress_callback({
                        "epoch": epoch+1, "total_epochs": epochs,
                        "train_loss": float(avg_tr), "val_loss": float(avg_vl)
                    })
                except Exception:
                    pass

            # best / early stop
            improved = avg_vl < best_val - 1e-8
            if improved:
                best_val = avg_vl
                self.history['best_val_loss'] = best_val
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pth"))
                logger.info(f"Saved best model to {self.checkpoint_dir}/{self.model_type}_best.pth (val={best_val:.4f})")
                bad_epochs = 0
            else:
                bad_epochs += 1

            if (epoch + 1) % max(1, checkpoint_interval) == 0:
                path = os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch+1}.pth")
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'history': self.history}, path)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")

            if early_stop_patience and bad_epochs >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        logger.info("Training completed!")

    # ----- generation (unchanged logic) -----
    def generate_new_ode(self, seed: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        self.model.eval()
        with torch.no_grad():
            if seed is None:
                if self.model_type == 'vae':
                    generated = self.model.sample(1)
                else:
                    seed = torch.randn(1, self.input_dim).to(self.device)
                    generated = self.model(seed)
            else:
                seed = seed.to(self.device)
                if self.model_type == 'vae':
                    generated, _, _ = self.model(seed)
                else:
                    generated = self.model(seed)
            params_tensor = generated.cpu().numpy()[0]
            alpha = float(np.clip(params_tensor[0], -100, 100))
            beta  = float(np.clip(abs(params_tensor[1]) + 0.1, 0.1, 100))
            n     = int(np.clip(abs(params_tensor[2]), 1, 10))
            M     = float(np.clip(params_tensor[3], -100, 100))
            func_id = int(abs(params_tensor[4])) % len(self.basic_functions.get_function_names())
            is_linear = params_tensor[5] > 0.5
            gen_num = int(np.clip(abs(params_tensor[6]), 1, 8 if is_linear else 10))
            basic_funcs = self.basic_functions.get_function_names()
            func_name = basic_funcs[func_id]
            f_z = self.basic_functions.get_function(func_name)
            generator_params = {'alpha': alpha, 'beta': beta, 'n': n, 'M': M}
            try:
                if is_linear:
                    factory = LinearGeneratorFactory()
                    if gen_num in [4, 5]:
                        generator_params['a'] = float(np.clip(abs(params_tensor[10]) + 1, 1, 5))
                    result = factory.create(gen_num, f_z, **generator_params)
                else:
                    factory = NonlinearGeneratorFactory()
                    extra = {}
                    if gen_num in [1, 2, 4]: extra['q'] = int(np.clip(abs(params_tensor[8]) + 2, 2, 10))
                    if gen_num in [2, 3, 5]: extra['v'] = int(np.clip(abs(params_tensor[9]) + 2, 2, 10))
                    if gen_num in [4, 5, 9, 10]: extra['a'] = float(np.clip(abs(params_tensor[10]) + 1, 1, 5))
                    result = factory.create(gen_num, f_z, **{**generator_params, **extra})
                result['function_used'] = func_name
                result['ml_generated'] = True
                result['generation_params'] = generator_params
                return result
            except Exception as e:
                logger.error(f"Error generating ODE: {e}")
                return None