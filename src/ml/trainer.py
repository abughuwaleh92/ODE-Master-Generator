"""
Upgraded ML Trainer for ODE Generators
- Durable progress stats (for RQ UI)
- Weighted/masked loss (ignore noise feature; down-weight discrete dims)
- Optional feature normalization
- Early stopping
- VAE: beta annealing (KL warmup)
- Transformer: unchanged API, trainer-side improvements
"""
import os, json, gc, logging
from typing import Dict, Any, List, Optional, Tuple, Iterator, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# external modules in your repo
from src.generators.linear_generators import LinearGeneratorFactory
from src.generators.nonlinear_generators import NonlinearGeneratorFactory
from src.functions.basic_functions import BasicFunctions
from src.functions.special_functions import SpecialFunctions
from src.ml.pattern_learner import (
    GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer
)

# ---------------- Dataset / Generator ----------------
class ODEDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], max_cache_size: int = 1000):
        self.data = data
        self.max_cache_size = min(max_cache_size, len(data))
        self._feature_cache = {}

    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        features = [
            item.get('alpha', 1.0),
            item.get('beta', 1.0),
            item.get('n', 1),
            item.get('M', 0.0),
            item.get('function_id', 0),
            1 if item.get('type') == 'linear' else 0,
            item.get('generator_number', 1),
            item.get('order', 2),
            item.get('q', 0),
            item.get('v', 0),
            item.get('a', 0.0),
            np.random.randn() * 0.1,  # noise feature
        ]
        return torch.tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if idx in self._feature_cache:
            features = self._feature_cache[idx]
        else:
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
                    'alpha': np.random.uniform(-5, 5),
                    'beta': np.random.uniform(0.1, 5),
                    'n': np.random.randint(1, 5),
                    'M': np.random.uniform(-5, 5)
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
                    gen_num = np.random.randint(1, 9)
                    if gen_num in [4, 5]: params['a'] = np.random.uniform(1, 3)
                    result = self.linear_factory.create(gen_num, f_z, **params)
                else:
                    gen_num = np.random.randint(1, 11)
                    extra_params = {}
                    if gen_num in [1, 2, 4]: extra_params['q'] = np.random.randint(2, 6)
                    if gen_num in [2, 3, 5]: extra_params['v'] = np.random.randint(2, 6)
                    if gen_num in [4, 5, 9, 10]: extra_params['a'] = np.random.uniform(1, 5)
                    result = self.nonlinear_factory.create(gen_num, f_z, **{**params, **extra_params})
                    params.update(extra_params)

                data_item = {
                    **params,
                    'function_name': func_name,
                    'function_id': func_id,
                    'type': gen_type,
                    'generator_number': gen_num,
                    'order': result['order']
                }

                features = torch.tensor([
                    params['alpha'], params['beta'], params['n'], params['M'],
                    func_id, 1 if gen_type == 'linear' else 0, gen_num,
                    result['order'], params.get('q', 0), params.get('v', 0),
                    params.get('a', 0.0), np.random.randn() * 0.1
                ], dtype=torch.float32)

                yield features, data_item
                generated += 1
            except Exception:
                continue

    def __len__(self) -> int:
        return self.num_samples

# ---------------- Trainer ----------------
class MLTrainer:
    def __init__(self,
                 model_type: str = 'pattern_learner',
                 input_dim: int = 12,
                 hidden_dim: int = 128,
                 output_dim: int = 12,
                 learning_rate: float = 1e-3,
                 device: Optional[str] = None,
                 checkpoint_dir: str = 'checkpoints',
                 enable_mixed_precision: bool = False):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.enable_mixed_precision = enable_mixed_precision

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._create_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        # loss config
        self.loss_weights = None  # [1, input_dim] tensor on device
        self._init_default_loss_weights()

        # normalization (optional)
        self.normalize = False
        self._mu = torch.zeros(1, input_dim, dtype=torch.float32).to(self.device)
        self._sigma = torch.ones(1, input_dim, dtype=torch.float32).to(self.device)

        # VAE config
        self.beta_start = 0.0
        self.beta_end = 1.0
        self.beta_warmup_epochs = 20

        self.scaler = torch.cuda.amp.GradScaler() if enable_mixed_precision else None

        self.history = {
            'train_loss': [], 'val_loss': [], 'epochs': 0, 'best_val_loss': float('inf')
        }

        self.basic_functions = BasicFunctions()
        self.special_functions = SpecialFunctions()

    def _init_default_loss_weights(self):
        w = torch.ones(self.input_dim, dtype=torch.float32)
        # down-weight discrete-ish dims
        w[4] = 0.3   # function_id
        w[5] = 0.2   # is_linear
        w[6] = 0.3   # generator_number
        # ignore noise
        w[self.input_dim - 1] = 0.0
        self.loss_weights = w.view(1, -1).to('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_model(self) -> nn.Module:
        if self.model_type == 'pattern_learner':
            return GeneratorPatternLearner(self.input_dim, self.hidden_dim, self.output_dim)
        elif self.model_type == 'vae':
            return GeneratorVAE(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=32)
        elif self.model_type == 'transformer':
            return GeneratorTransformer(input_dim=self.input_dim, d_model=self.hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self._mu) / (self._sigma + 1e-6)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_weights is None:
            return F.mse_loss(pred, target)
        diff2 = (pred - target) ** 2
        weighted = diff2 * self.loss_weights
        return torch.mean(weighted)

    def _vae_loss(self, recon_x, x, mu, log_var, epoch: int, total_epochs: int):
        recon = self._masked_mse(recon_x, x)
        # beta annealing
        if self.beta_warmup_epochs > 0:
            beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * max(0, epoch) / max(1, self.beta_warmup_epochs))
        else:
            beta = self.beta_end
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + beta * kld, {"recon": float(recon.detach().cpu()), "kld": float(kld.detach().cpu()), "beta": float(beta)}

    def _stats(self, train_loss: float, val_loss: float, extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        out = {"train_loss": float(train_loss), "val_loss": float(val_loss)}
        if extra:
            out.update(extra)
        return out

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              samples: int = 1000,
              validation_split: float = 0.2,
              save_best: bool = True,
              use_generator: bool = True,
              checkpoint_interval: int = 10,
              gradient_accumulation_steps: int = 1,
              progress_callback: Optional[Callable[..., None]] = None,
              early_stop_patience: int = 10):
        logger.info(f"Starting training for {epochs} epochs...")

        # Data loaders
        if use_generator:
            train_samples = int(samples * (1 - validation_split))
            val_samples = samples - train_samples
            train_dataset = ODEDataGenerator(train_samples, batch_size)
            val_dataset = ODEDataGenerator(val_samples, batch_size, seed=42)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        else:
            data = self._generate_training_data_batch(samples)
            dataset = ODEDataset(data, max_cache_size=min(1000, len(data)))
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=self.device.type=='cuda')
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=self.device.type=='cuda')

        best_val = float('inf')
        bad_epochs = 0

        for epoch in range(epochs):
            # training
            self.model.train()
            train_loss, batches = 0.0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (batch_features, _) in enumerate(pbar):
                x = batch_features.to(self.device)
                x = self._pre(x)
                if self.enable_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        if self.model_type == 'vae':
                            recon, mu, logvar = self.model(x)
                            loss, vae_stats = self._vae_loss(recon, x, mu, logvar, epoch, epochs)
                        else:
                            out = self.model(x)
                            loss = self._masked_mse(out, x)
                    loss = loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer); self.scaler.update(); self.optimizer.zero_grad()
                else:
                    if self.model_type == 'vae':
                        recon, mu, logvar = self.model(x)
                        loss, vae_stats = self._vae_loss(recon, x, mu, logvar, epoch, epochs)
                    else:
                        out = self.model(x)
                        loss = self._masked_mse(out, x)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.optimizer.step(); self.optimizer.zero_grad()

                train_loss += float(loss.detach().cpu()) * gradient_accumulation_steps
                batches += 1
                pbar.set_postfix({'loss': train_loss / max(1, batches)})

            # validation
            self.model.eval()
            val_loss, vbatches = 0.0, 0
            with torch.no_grad():
                for batch_features, _ in val_loader:
                    x = batch_features.to(self.device)
                    x = self._pre(x)
                    if self.model_type == 'vae':
                        recon, mu, logvar = self.model(x)
                        vloss, _ = self._vae_loss(recon, x, mu, logvar, epoch, epochs)
                    else:
                        out = self.model(x)
                        vloss = self._masked_mse(out, x)
                    val_loss += float(vloss.detach().cpu()); vbatches += 1

            avg_train = train_loss / max(1, batches)
            avg_val = val_loss / max(1, vbatches)
            self.history['train_loss'].append(avg_train)
            self.history['val_loss'].append(avg_val)
            self.history['epochs'] = epoch + 1
            self.scheduler.step(avg_val)
            logger.info(f"Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f}")

            # push progress stats to callback (RQ)
            if progress_callback:
                try:
                    progress_callback(epoch + 1, epochs, self._stats(avg_train, avg_val))
                except TypeError:
                    # backward-compat: (epoch, total)
                    progress_callback(epoch + 1, epochs)

            # save best/checkpoints
            if save_best and avg_val < best_val - 1e-8:
                best_val = avg_val
                self.history['best_val_loss'] = best_val
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pth"))
                logger.info(f"Saved best model to {os.path.join(self.checkpoint_dir, f'{self.model_type}_best.pth')} (val={best_val:.4f})")
                bad_epochs = 0
            else:
                bad_epochs += 1

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch+1}.pth"), epoch + 1)

            if early_stop_patience and bad_epochs >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        logger.info("Training completed.")

    def _generate_training_data_batch(self, num_samples: int) -> List[Dict[str, Any]]:
        logger.info(f"Generating {num_samples} training samples...")
        data = []
        generator = ODEDataGenerator(num_samples, 100)
        for _, item in tqdm(generator, total=num_samples, desc="Generating"):
            data.append(item)
        logger.info(f"Generated {len(data)} samples.")
        return data

    def save_checkpoint(self, path: str, epoch: int):
        ckpt = {
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
            ckpt['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return 0
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.history = ckpt.get('history', self.history)
            if self.scaler and 'scaler_state_dict' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            epoch = ckpt.get('epoch', 0)
            logger.info(f"Checkpoint loaded (epoch {epoch})")
            return epoch
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0

    def save_model(self, path: str):
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
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.history = ckpt.get('history', self.history)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def save_artifacts(self, run_dir: str) -> Dict[str, str]:
        """
        Save history.json and config.json alongside checkpoints.
        """
        os.makedirs(run_dir, exist_ok=True)
        history_path = os.path.join(run_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        cfg = {
            "model_type": self.model_type, "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim, "learning_rate": self.learning_rate,
            "normalize": self.normalize, "beta_warmup_epochs": self.beta_warmup_epochs,
        }
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        return {"history_path": history_path, "config_path": config_path}

    def generate_new_ode(self, seed: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        self.model.eval()
        with torch.no_grad():
            if seed is None:
                seed = torch.randn(1, self.input_dim).to(self.device)
            x = self._pre(seed)
            if self.model_type == 'vae':
                recon, _, _ = self.model(x)
                generated = recon
            else:
                generated = self.model(x)
            params_tensor = generated.detach().cpu().numpy()[0]
            # decode
            alpha = float(np.clip(params_tensor[0], -5, 5))
            beta  = float(np.clip(abs(params_tensor[1]) + 0.1, 0.1, 5))
            n     = int(np.clip(abs(params_tensor[2]), 1, 10))
            M     = float(np.clip(params_tensor[3], -5, 5))
            func_id = int(abs(params_tensor[4])) % len(self.basic_functions.get_function_names())
            is_linear = params_tensor[5] > 0.5
            gen_num = int(np.clip(abs(params_tensor[6]), 1, 8 if is_linear else 10))
            order = int(np.clip(abs(params_tensor[7]), 1, 6))
            q = int(np.clip(abs(params_tensor[8]), 0, 10))
            v = int(np.clip(abs(params_tensor[9]), 0, 10))
            a = float(np.clip(abs(params_tensor[10]), 0.0, 5.0))
            func_name = self.basic_functions.get_function_names()[func_id]
            f_z = self.basic_functions.get_function(func_name)
            try:
                factory = LinearGeneratorFactory() if is_linear else NonlinearGeneratorFactory()
                args = {"alpha": alpha, "beta": beta, "n": n, "M": M}
                if is_linear:
                    if gen_num in [4,5]: args["a"] = a
                    res = factory.create(gen_num, f_z, **args)
                else:
                    if gen_num in [1,2,4]: args["q"] = max(2, q)
                    if gen_num in [2,3,5]: args["v"] = max(2, v)
                    if gen_num in [4,5,9,10]: args["a"] = a
                    res = factory.create(gen_num, f_z, **args)
                res['function_used'] = func_name
                res['ml_generated'] = True
                res['generation_params'] = args
                return res
            except Exception as e:
                logger.error(f"Error generating ODE: {e}")
                return None