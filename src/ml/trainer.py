"""
Machine Learning Trainer for ODE Generators (rewritten)
- Robust to missing keys (fixes KeyError: 'v')
- Works with Basic + Special + Phi libraries for function_id mapping
- Mixed precision (optional), grad clipping, ReduceLROnPlateau
- Optional external dataset injection via set_dataset()
"""

import os
import json
import gc
import math
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset, random_split
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Factories and functions
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
except Exception:
    BasicFunctions = None

try:
    from src.functions.special_functions import SpecialFunctions
except Exception:
    SpecialFunctions = None

try:
    from src.functions.phi_library import PhiLibrary
except Exception:
    PhiLibrary = None

# Optional models
try:
    from src.ml.pattern_learner import (
        GeneratorPatternLearner, GeneratorVAE, GeneratorTransformer, create_model
    )
except Exception:
    GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = create_model = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: function universe (IDs stable for training)
# ──────────────────────────────────────────────────────────────────────────────
def _function_universe() -> List[str]:
    names: List[str] = []
    if BasicFunctions:
        try:
            names += BasicFunctions().get_function_names()
        except Exception:
            pass
    if SpecialFunctions:
        try:
            names += SpecialFunctions().get_function_names()[:50]
        except Exception:
            pass
    if PhiLibrary:
        try:
            names += PhiLibrary().get_function_names()
        except Exception:
            pass
    # unique order preserving
    seen = set(); uniq = []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    if not uniq:
        uniq = ["id"]
    return uniq

_FN_UNIVERSE = _function_universe()
_FN2ID = {n: i for i, n in enumerate(_FN_UNIVERSE)}
_ID2FN = {i: n for i, n in enumerate(_FN_UNIVERSE)}

def _get_function_expr(name: str):
    z = sp.Symbol("z", real=True)
    if BasicFunctions and name in BasicFunctions().get_function_names():
        return sp.sympify(BasicFunctions().get_function(name))
    if SpecialFunctions and name in SpecialFunctions().get_function_names():
        return sp.sympify(SpecialFunctions().get_function(name))
    if PhiLibrary and name in PhiLibrary().get_function_names():
        return sp.sympify(PhiLibrary().get_function(name))
    return z

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class ODEDataset(Dataset):
    """
    Dataset for ODE generator patterns (caches features).
    Input feature vector (size=12 by default):
      [alpha, beta, n, M, function_id, is_linear, generator_number, order,
       q, v, a, noise]
    """
    def __init__(self, data: List[Dict[str, Any]], max_cache_size: int = 1000):
        self.data = data
        self.max_cache_size = min(max_cache_size, len(data))
        self._feature_cache: Dict[int, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def __len__(self) -> int:
        return len(self.data)

    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        p = dict(item)
        # If nested "parameters" present, overlay first (so local keys override)
        if isinstance(item.get("parameters"), dict):
            p = {**item["parameters"], **p}

        fn_name = p.get("function_used") or p.get("function_name") or "id"
        function_id = _FN2ID.get(fn_name, 0)

        features = [
            float(p.get("alpha", 1.0)),
            float(p.get("beta", 1.0)),
            float(p.get("n", 1)),
            float(p.get("M", 0.0)),
            float(function_id),
            1.0 if str(p.get("type","nonlinear")) == "linear" else 0.0,
            float(p.get("generator_number", 1)),
            float(p.get("order", 2)),
            float(p.get("q", 0)),         # robust default
            float(p.get("v", 0)),         # robust default (fixes KeyError 'v')
            float(p.get("a", 0.0)),
            float(np.random.randn() * 0.05),
        ]
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if idx in self._feature_cache:
            self._cache_hits += 1
            feats = self._feature_cache[idx]
        else:
            self._cache_misses += 1
            feats = self._extract_features(self.data[idx])
            if len(self._feature_cache) < self.max_cache_size:
                self._feature_cache[idx] = feats
            else:
                # Simple eviction
                oldk = next(iter(self._feature_cache))
                del self._feature_cache[oldk]
                self._feature_cache[idx] = feats
        return feats, self.data[idx]

    def get_cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total) if total else 0.0,
            "cache_size": len(self._feature_cache)
        }


class ODEDataGenerator(IterableDataset):
    """
    Streaming generator (memory efficient).
    """
    def __init__(self, num_samples: int, batch_size: int = 32, seed: Optional[int] = None):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed = seed

        self.lin = LinearGeneratorFactory() if LinearGeneratorFactory else None
        self.nonlin = NonlinearGeneratorFactory() if NonlinearGeneratorFactory else None
        self.basic = BasicFunctions() if BasicFunctions else None
        self.special = SpecialFunctions() if SpecialFunctions else None
        self.phi = PhiLibrary() if PhiLibrary else None

        self.basic_names  = self.basic.get_function_names()  if self.basic else []
        self.special_names= (self.special.get_function_names()[:20] if self.special else [])
        self.phi_names    = self.phi.get_function_names()    if self.phi else []
        self.all_names    = self.basic_names + self.special_names + self.phi_names
        if not self.all_names:
            self.all_names = ["id"]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        if self.seed is not None:
            np.random.seed(self.seed); torch.manual_seed(self.seed)

        generated = 0
        while generated < self.num_samples:
            try:
                params = {
                    "alpha": np.random.uniform(-5, 5),
                    "beta":  np.random.uniform(0.1, 5),
                    "n":     int(np.random.randint(1, 5)),
                    "M":     np.random.uniform(-5, 5),
                }
                func_name = np.random.choice(self.all_names)
                if func_name in self.basic_names:
                    f_z = self.basic.get_function(func_name)
                elif func_name in self.special_names:
                    f_z = self.special.get_function(func_name)
                else:
                    f_z = self.phi.get_function(func_name) if self.phi else None

                function_id = _FN2ID.get(func_name, 0)
                gen_type = np.random.choice(["linear","nonlinear"])

                if gen_type == "linear" and self.lin:
                    gen_num = int(np.random.randint(1, 9))
                    if gen_num in [4,5]:
                        params["a"] = float(np.random.uniform(1, 3))
                    result = self.lin.create(gen_num, f_z, **params)
                elif gen_type == "nonlinear" and self.nonlin:
                    gen_num = int(np.random.randint(1, 11))
                    # only add params they might use
                    if gen_num in [1,2,4]: params["q"] = int(np.random.randint(2,6))
                    if gen_num in [2,3,5]: params["v"] = int(np.random.randint(2,6))
                    if gen_num in [4,5,9,10]: params["a"] = float(np.random.uniform(1,5))
                    result = self.nonlin.create(gen_num, f_z, **params)
                else:
                    # no factories available: synthesize minimal record
                    gen_type = "linear"
                    gen_num = 1
                    result = {"order": 1}

                data_item = {
                    **params,
                    "function_name": func_name,
                    "function_used": func_name,
                    "function_id": function_id,
                    "type": gen_type,
                    "generator_number": gen_num,
                    "order": result.get("order", 1),
                }

                feats = torch.tensor([
                    float(params["alpha"]), float(params["beta"]), float(params["n"]), float(params["M"]),
                    float(function_id), 1.0 if gen_type=="linear" else 0.0, float(gen_num),
                    float(result.get("order", 1)), float(params.get("q",0)), float(params.get("v",0)),
                    float(params.get("a",0.0)), float(np.random.randn()*0.05)
                ], dtype=torch.float32)

                yield feats, data_item
                generated += 1
            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                continue

    def __len__(self) -> int:
        return self.num_samples


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
class MLTrainer:
    """
    Baseline trainer compatible with your UI:
      - model_type ∈ {'pattern_learner','vae','transformer'}
      - memory safe generator or static dataset (set_dataset)
      - AMP/mixed precision optional
    """
    def __init__(
        self,
        model_type: str = "pattern_learner",
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 12,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        enable_mixed_precision: bool = False
    ):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = float(learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.enable_mixed_precision = enable_mixed_precision

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        if model_type == "pattern_learner" and GeneratorPatternLearner:
            self.model = GeneratorPatternLearner(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        elif model_type == "vae" and GeneratorVAE:
            self.model = GeneratorVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=32)
        elif model_type == "transformer" and GeneratorTransformer:
            self.model = GeneratorTransformer(input_dim=input_dim, d_model=hidden_dim)
        else:
            # Fallback: small MLP
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)
        self.criterion = self._vae_loss if model_type == "vae" else nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler() if (enable_mixed_precision and self.device.type == "cuda") else None

        self.history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "epochs": 0, "best_val_loss": float("inf")}
        self._injected_dataset: Optional[ODEDataset] = None

    # External dataset injection (optional)
    def set_dataset(self, generated_odes: List[Dict[str,Any]], batch_rows: List[Dict[str,Any]] = []):
        data = []
        data.extend(generated_odes or [])
        # batch_rows are typically table rows; we only take usable bits
        for r in (batch_rows or []):
            data.append({
                "alpha": r.get("α", 1.0),
                "beta":  r.get("β", 1.0),
                "n":     r.get("n", 1),
                "M":     0.0,
                "function_used": r.get("Function", "id"),
                "type": r.get("Type", "nonlinear"),
                "generator_number": r.get("Generator", 1),
                "order": r.get("Order", 1),
            })
        if data:
            self._injected_dataset = ODEDataset(data, max_cache_size=min(2000, len(data)))

    def _vae_loss(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
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
        grad_clip: float = 1.0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        logger.info(f"Starting training for {epochs} epochs...")

        if self._injected_dataset is not None:
            dataset = self._injected_dataset
            use_generator = False

        if use_generator:
            train_samples = int(samples * (1 - validation_split))
            val_samples = max(1, samples - train_samples)
            train_dataset = ODEDataGenerator(train_samples, batch_size)
            val_dataset   = ODEDataGenerator(val_samples,   batch_size, seed=42)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers=0)
        else:
            if self._injected_dataset is None:
                # build a static dataset from the streaming generator
                gen = ODEDataGenerator(samples, batch_size)
                static_data = []
                for feats, item in tqdm(gen, total=samples, desc="Preparing data"):
                    static_data.append(item)
                dataset = ODEDataset(static_data, max_cache_size=min(1500, len(static_data)))
            val_size = int(len(dataset) * validation_split)
            train_size = max(1, len(dataset) - val_size)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            pin_mem = self.device.type == "cuda"
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
            val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0; batch_count = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (batch_features, _) in enumerate(pbar):
                batch_features = batch_features.to(self.device)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        if self.model_type == "vae":
                            recon, mu, log_var = self.model(batch_features)
                            loss = self.criterion(recon, batch_features, mu, log_var)
                        else:
                            out = self.model(batch_features)
                            loss = self.criterion(out, batch_features)
                    loss = loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        if grad_clip and grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    if self.model_type == "vae":
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        out = self.model(batch_features)
                        loss = self.criterion(out, batch_features)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        if grad_clip and grad_clip > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                train_loss += float(loss.item()) * gradient_accumulation_steps
                batch_count += 1
                pbar.set_postfix({"loss": train_loss / max(1, batch_count)})

            # Validation
            self.model.eval()
            val_loss = 0.0; val_batch_count = 0
            with torch.no_grad():
                for batch_features, _ in val_loader:
                    batch_features = batch_features.to(self.device)
                    if self.model_type == "vae":
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        out = self.model(batch_features)
                        loss = self.criterion(out, batch_features)
                    val_loss += float(loss.item()); val_batch_count += 1

            avg_train = train_loss / max(1, batch_count)
            avg_val   = val_loss   / max(1, val_batch_count)
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)
            self.history["epochs"] = epoch + 1
            self.scheduler.step(avg_val)
            logger.info(f"Epoch {epoch+1}: Train={avg_train:.4f}  Val={avg_val:.4f}")

            if progress_callback:
                try: progress_callback(epoch+1, epochs)
                except Exception: pass

            # Save best
            if save_best and avg_val < best_val_loss - 1e-9:
                best_val_loss = avg_val
                self.history["best_val_loss"] = best_val_loss
                self.save_model(os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pth"))
                logger.info(f"Saved best model (val={best_val_loss:.4f})")

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch+1}.pth"), epoch+1)

            if (epoch + 1) % 10 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        logger.info("Training completed.")

    # I/O utils
    def save_checkpoint(self, path: str, epoch: int):
        ck = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "model_type": self.model_type,
            "input_dim": self.input_dim, "hidden_dim": self.hidden_dim, "output_dim": self.output_dim,
            "learning_rate": self.learning_rate
        }
        if self.scaler:
            ck["scaler_state_dict"] = self.scaler.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ck, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return 0
        try:
            ck = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ck["model_state_dict"])
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            self.scheduler.load_state_dict(ck["scheduler_state_dict"])
            self.history = ck.get("history", self.history)
            if self.scaler and "scaler_state_dict" in ck:
                self.scaler.load_state_dict(ck["scaler_state_dict"])
            epoch = int(ck.get("epoch", 0))
            logger.info(f"Loaded checkpoint (epoch {epoch})")
            return epoch
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "model_type": self.model_type,
            "input_dim": self.input_dim, "hidden_dim": self.hidden_dim, "output_dim": self.output_dim
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        try:
            ck = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ck["model_state_dict"])
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            self.history = ck.get("history", self.history)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    # Generation utility (kept simple)
    def generate_new_ode(self, seed: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        self.model.eval()
        with torch.no_grad():
            if seed is None:
                seed = torch.randn(1, self.input_dim, device=self.device)
            if self.model_type == "vae":
                gen, _, _ = self.model(seed)
            else:
                gen = self.model(seed)

            v = gen.detach().cpu().numpy()[0]
            alpha = float(np.clip(v[0], -100, 100))
            beta  = float(np.clip(abs(v[1])+0.1, 0.1, 100))
            n     = int(np.clip(abs(v[2]), 1, 10))
            M     = float(np.clip(v[3], -100, 100))
            func_id = int(abs(v[4])) % max(1, len(_FN_UNIVERSE))
            is_linear = v[5] > 0.5
            gen_num = int(np.clip(abs(v[6]), 1, 10 if not is_linear else 8))
            a = float(np.clip(abs(v[10]), 0.0, 5.0))

            fn_name = _ID2FN.get(func_id, "id")
            # resolve f_z for factories
            if BasicFunctions and fn_name in BasicFunctions().get_function_names():
                f_z = BasicFunctions().get_function(fn_name)
            elif SpecialFunctions and fn_name in SpecialFunctions().get_function_names():
                f_z = SpecialFunctions().get_function(fn_name)
            elif PhiLibrary and fn_name in PhiLibrary().get_function_names():
                f_z = PhiLibrary().get_function(fn_name)
            else:
                f_z = sp.Symbol("z")

            generator_params = {"alpha": alpha, "beta": beta, "n": n, "M": M}
            res = {}

            try:
                if is_linear and LinearGeneratorFactory:
                    fac = LinearGeneratorFactory()
                    if gen_num in [4,5]: generator_params["a"] = max(1.0, a)
                    res = fac.create(gen_num, f_z, **generator_params)
                elif (not is_linear) and NonlinearGeneratorFactory:
                    fac = NonlinearGeneratorFactory()
                    extra = {}
                    if gen_num in [1,2,4]: extra["q"] = int(np.clip(abs(v[8])+2, 2, 10))
                    if gen_num in [2,3,5]: extra["v"] = int(np.clip(abs(v[9])+2, 2, 10))
                    if gen_num in [4,5,9,10]: extra["a"] = max(1.0, a)
                    res = fac.create(gen_num, f_z, **{**generator_params, **extra})
                else:
                    # fallback
                    res = {"order": 1, "type": "linear"}
                res["function_used"] = fn_name
                res["ml_generated"] = True
                res["generation_params"] = generator_params
                return res
            except Exception as e:
                logger.error(f"Error generating ODE: {e}")
                return None

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        self.model.eval()
        dataset = ODEDataset(test_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        total_loss = 0.0; preds = []; targs = []
        with torch.no_grad():
            for feats, _ in loader:
                feats = feats.to(self.device)
                if self.model_type == "vae":
                    recon, mu, log_var = self.model(feats)
                    loss = self.criterion(recon, feats, mu, log_var)
                    out = recon
                else:
                    out = self.model(feats)
                    loss = self.criterion(out, feats)
                total_loss += float(loss.item())
                preds.append(out.cpu()); targs.append(feats.cpu())
        pred = torch.cat(preds); targ = torch.cat(targs)
        mse = F.mse_loss(pred, targ).item()
        mae = F.l1_loss(pred, targ).item()
        corr = float(np.corrcoef(pred.flatten().numpy(), targ.flatten().numpy())[0,1])
        return {"loss": total_loss / max(1, len(loader)), "mse": mse, "mae": mae, "correlation": corr}