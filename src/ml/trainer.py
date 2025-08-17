"""
Machine Learning Trainer for ODE Generators
- Handles training, evaluation, and ODE generation.
- Fixes batch collation: custom collate_fn returns FEATURES ONLY to avoid dict key mismatches (e.g., 'v').
- Preserves generator + dataset modes, AMP, checkpoints, VAE/Transformer, and ONNX export.
"""

from __future__ import annotations
import os
import gc
import json
import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split
from tqdm import tqdm
import sympy as sp

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Resilient imports from project ----
try:
    from src.generators.linear_generators import LinearGeneratorFactory
    from src.generators.nonlinear_generators import NonlinearGeneratorFactory
    from src.functions.basic_functions import BasicFunctions
    from src.functions.special_functions import SpecialFunctions
    from src.ml.pattern_learner import (
        GeneratorPatternLearner,
        GeneratorVAE,
        GeneratorTransformer,
        create_model,  # keep available if referenced elsewhere
    )
except Exception as e:
    logger.error(f"Import error: {e}")
    # Allow partial functionality even if factories are missing
    LinearGeneratorFactory = None  # type: ignore
    NonlinearGeneratorFactory = None  # type: ignore
    BasicFunctions = None  # type: ignore
    SpecialFunctions = None  # type: ignore
    GeneratorPatternLearner = None  # type: ignore
    GeneratorVAE = None  # type: ignore
    GeneratorTransformer = None  # type: ignore
    create_model = None  # type: ignore


# =========================================================
# Collate: FEATURES ONLY (fix for KeyError: 'v')
# =========================================================
def features_only_collate(batch: List[Any]) -> Tuple[torch.Tensor, None]:
    """
    Collate a list of items where each item is either:
      - (features_tensor, anything)  OR
      - features_tensor
    Returns: (stacked_features, None)
    This bypasses dict collation entirely and prevents KeyError on missing keys.
    """
    feats: List[torch.Tensor] = []
    for item in batch:
        if isinstance(item, (tuple, list)):
            feats.append(item[0])
        else:
            feats.append(item)
    return torch.stack(feats, dim=0), None


# =========================================================
# Dataset / Generator
# =========================================================
class ODEDataset(Dataset):
    """
    Dataset for ODE generator patterns with a small in-memory feature cache.
    We still return (features, meta) but the collate_fn discards meta during training.
    """

    def __init__(self, data: List[Dict[str, Any]], max_cache_size: int = 1000):
        self.data = data
        self.max_cache_size = min(max_cache_size, len(data))
        self._feature_cache: Dict[int, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        # Uniformly expose optional keys with defaults
        alpha = float(item.get("alpha", 1.0))
        beta = float(item.get("beta", 1.0))
        n = float(item.get("n", 1))
        M = float(item.get("M", 0.0))
        function_id = float(item.get("function_id", 0))
        is_linear = 1.0 if item.get("type") == "linear" else 0.0
        generator_number = float(item.get("generator_number", 1))
        order = float(item.get("order", 2))
        q = float(item.get("q", 0))
        v = float(item.get("v", 0))
        a = float(item.get("a", 0.0))
        noise = float(np.random.randn() * 0.1)

        features = [
            alpha, beta, n, M, function_id, is_linear,
            generator_number, order, q, v, a, noise
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
            else:
                # naive eviction
                try:
                    oldest = next(iter(self._feature_cache))
                    del self._feature_cache[oldest]
                except Exception:
                    self._feature_cache.clear()
                self._feature_cache[idx] = features

        # Ensure uniform meta keys (not used during training, but nice to keep consistent)
        meta = dict(self.data[idx])
        meta.setdefault("q", 0)
        meta.setdefault("v", 0)
        meta.setdefault("a", 0.0)
        return features, meta

    def get_cache_stats(self) -> Dict[str, float]:
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": float(self._cache_hits),
            "cache_misses": float(self._cache_misses),
            "hit_rate": float(self._cache_hits / total) if total else 0.0,
            "cache_size": float(len(self._feature_cache)),
        }


class ODEDataGenerator(IterableDataset):
    """
    Memory-efficient on-the-fly sample generator for large-scale training.
    Yields (features_tensor, meta_dict).
    """

    def __init__(self, num_samples: int, batch_size: int = 32, seed: Optional[int] = None):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed = seed

        # Factories & function libs (if available)
        self.linear_factory = LinearGeneratorFactory() if LinearGeneratorFactory else None
        self.nonlinear_factory = NonlinearGeneratorFactory() if NonlinearGeneratorFactory else None
        self.basic_functions = BasicFunctions() if BasicFunctions else None
        self.special_functions = SpecialFunctions() if SpecialFunctions else None

        # Function name lists
        self.basic_func_names = (
            self.basic_functions.get_function_names() if self.basic_functions else ["exp", "sin", "cos"]
        )
        special = self.special_functions.get_function_names() if self.special_functions else ["erf", "sinh", "cosh"]
        self.special_func_names = special[:5]  # optional cap
        self.all_func_names = list(self.basic_func_names) + list(self.special_func_names)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, Any]]]:
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        generated = 0
        while generated < self.num_samples:
            try:
                # Base parameters
                params: Dict[str, Any] = {
                    "alpha": float(np.random.uniform(-5, 5)),
                    "beta": float(np.random.uniform(0.1, 5)),
                    "n": int(np.random.randint(1, 5)),
                    "M": float(np.random.uniform(-5, 5)),
                }

                # Random function
                func_name = np.random.choice(self.all_func_names)
                if func_name in self.basic_func_names:
                    f_z = self.basic_functions.get_function(func_name) if self.basic_functions else None
                    func_id = self.basic_func_names.index(func_name)
                else:
                    f_z = self.special_functions.get_function(func_name) if self.special_functions else None
                    func_id = len(self.basic_func_names) + self.special_func_names.index(func_name)

                # Generator type & number
                gen_type = np.random.choice(["linear", "nonlinear"])
                gen_num = 1

                # Default optionals (IMPORTANT for uniform meta)
                q = 0
                v = 0
                a = 0.0
                order_val = 2

                if gen_type == "linear":
                    gen_num = int(np.random.randint(1, 9))
                    if gen_num in [4, 5]:
                        a = float(np.random.uniform(1, 3))
                        params["a"] = a
                    if self.linear_factory and f_z is not None:
                        try:
                            result = self.linear_factory.create(gen_num, f_z, **params)
                            order_val = int(result.get("order", 2))
                        except Exception:
                            order_val = int(np.random.randint(1, 5))
                    else:
                        order_val = int(np.random.randint(1, 5))
                else:
                    gen_num = int(np.random.randint(1, 11))
                    extra_params: Dict[str, Any] = {}
                    if gen_num in [1, 2, 4]:
                        q = int(np.random.randint(2, 6))
                        extra_params["q"] = q
                    if gen_num in [2, 3, 5]:
                        v = int(np.random.randint(2, 6))
                        extra_params["v"] = v
                    if gen_num in [4, 5, 9, 10]:
                        a = float(np.random.uniform(1, 5))
                        extra_params["a"] = a
                    params.update(extra_params)
                    if self.nonlinear_factory and f_z is not None:
                        try:
                            result = self.nonlinear_factory.create(gen_num, f_z, **params)
                            order_val = int(result.get("order", 2))
                        except Exception:
                            order_val = int(np.random.randint(1, 5))
                    else:
                        order_val = int(np.random.randint(1, 5))

                # Meta (uniform keys present)
                meta = {
                    **params,
                    "function_name": func_name,
                    "function_id": func_id,
                    "type": gen_type,
                    "generator_number": gen_num,
                    "order": order_val,
                    "q": q,
                    "v": v,
                    "a": a,
                }

                # Features
                features = torch.tensor(
                    [
                        params["alpha"],
                        params["beta"],
                        float(params["n"]),
                        params["M"],
                        float(func_id),
                        1.0 if gen_type == "linear" else 0.0,
                        float(gen_num),
                        float(order_val),
                        float(q),
                        float(v),
                        float(a),
                        float(np.random.randn() * 0.1),
                    ],
                    dtype=torch.float32,
                )

                yield features, meta
                generated += 1

            except Exception as e:
                logger.debug(f"Error generating sample: {e}")
                continue

    def __len__(self) -> int:
        return int(self.num_samples)


# =========================================================
# Trainer
# =========================================================
class MLTrainer:
    """
    Main trainer for feature-reconstruction/self-supervised learning on generator features.
    Supports model types: pattern_learner, vae, transformer.
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
        enable_mixed_precision: bool = False,
    ):
        self.model_type = model_type
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.learning_rate = float(learning_rate)
        self.checkpoint_dir = checkpoint_dir

        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # AMP only if CUDA
        self.enable_mixed_precision = bool(enable_mixed_precision and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_mixed_precision)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Model
        self.model = self._create_model().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=5, factor=0.5)

        # Loss
        self.criterion = self._vae_loss if self.model_type == "vae" else nn.MSELoss()

        # Training history
        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "epochs": 0,
            "best_val_loss": float("inf"),
        }

        # Optional libs for generation
        self.basic_functions = BasicFunctions() if BasicFunctions else None
        self.special_functions = SpecialFunctions() if SpecialFunctions else None

        # Optional injected datasets (from UI)
        self._injected_data: List[Dict[str, Any]] = []

    # ----------------- Public API -----------------
    def set_dataset(self, single_odes: List[Dict[str, Any]], batch_results: List[Dict[str, Any]] = None) -> None:
        """Allow UI to inject existing ODE samples (optional)."""
        merged = list(single_odes or [])
        if batch_results:
            merged.extend(batch_results)
        self._injected_data = merged

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
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """
        Train the model. By default uses the ODEDataGenerator (memory-efficient), which avoids the dict collation issue.
        """
        logger.info(f"Starting training for {epochs} epochs...")

        # ---------- Data loaders ----------
        pin = self.device.type == "cuda"
        if use_generator:
            train_samples = int(samples * (1 - validation_split))
            val_samples = max(int(samples - train_samples), 1)

            train_dataset = ODEDataGenerator(train_samples, batch_size)
            val_dataset = ODEDataGenerator(val_samples, batch_size, seed=42)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=pin,
                collate_fn=features_only_collate,  # <- FIX
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=pin,
                collate_fn=features_only_collate,  # <- FIX
            )
        else:
            # traditional: pre-generate + cache
            data = self._generate_training_data_batch(samples)
            dataset = ODEDataset(data, max_cache_size=min(1000, len(data)))
            val_size = max(int(len(dataset) * validation_split), 1)
            train_size = max(len(dataset) - val_size, 1)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=pin,
                num_workers=0,
                collate_fn=features_only_collate,  # <- FIX
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin,
                num_workers=0,
                collate_fn=features_only_collate,  # <- FIX
            )

        best_val_loss = float("inf")

        # ---------- Epochs ----------
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (batch_features, _) in enumerate(pbar):
                batch_features = batch_features.to(self.device, non_blocking=pin)

                if self.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        if self.model_type == "vae":
                            recon, mu, log_var = self.model(batch_features)
                            loss = self.criterion(recon, batch_features, mu, log_var)
                        else:
                            output = self.model(batch_features)
                            loss = self.criterion(output, batch_features)
                    loss = loss / gradient_accumulation_steps
                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    if self.model_type == "vae":
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        output = self.model(batch_features)
                        loss = self.criterion(output, batch_features)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1
                pbar.set_postfix({"loss": train_loss / max(batch_count, 1)})

            # Validate
            self.model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for batch_features, _ in val_loader:
                    batch_features = batch_features.to(self.device, non_blocking=pin)
                    if self.model_type == "vae":
                        recon, mu, log_var = self.model(batch_features)
                        loss = self.criterion(recon, batch_features, mu, log_var)
                    else:
                        output = self.model(batch_features)
                        loss = self.criterion(output, batch_features)
                    val_loss += loss.item()
                    val_count += 1

            avg_train = train_loss / max(batch_count, 1)
            avg_val = val_loss / max(val_count, 1)
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)
            self.history["epochs"] = epoch + 1
            self.scheduler.step(avg_val)

            logger.info(f"Epoch {epoch+1}: Train {avg_train:.4f} | Val {avg_val:.4f}")

            # Callback (for RQ meta progress or Streamlit progress bar)
            if progress_callback:
                try:
                    progress_callback(epoch + 1, epochs)
                except Exception:
                    pass

            # Save best
            if save_best and avg_val < best_val_loss:
                best_val_loss = avg_val
                self.history["best_val_loss"] = best_val_loss
                best_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_best.pth")
                self.save_model(best_path)
                logger.info(f"Saved best model to {best_path} (val={best_val_loss:.4f})")

            # Periodic checkpoint
            if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_epoch_{epoch+1}.pth")
                self.save_checkpoint(ckpt_path, epoch + 1)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")

            # Memory cleanup
            if (epoch + 1) % 10 == 0:
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        logger.info("Training completed.")

    # ----------------- Internals -----------------
    def _create_model(self) -> nn.Module:
        if self.model_type == "pattern_learner":
            assert GeneratorPatternLearner is not None, "PatternLearner not available"
            return GeneratorPatternLearner(self.input_dim, self.hidden_dim, self.output_dim)
        elif self.model_type == "vae":
            assert GeneratorVAE is not None, "VAE not available"
            return GeneratorVAE(self.input_dim, self.hidden_dim, latent_dim=32)
        elif self.model_type == "transformer":
            assert GeneratorTransformer is not None, "Transformer not available"
            return GeneratorTransformer(input_dim=self.input_dim, d_model=self.hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _vae_loss(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss

    def _generate_training_data_batch(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generates synthetic training meta in batches using the on-the-fly generator (for memory saving).
        The training loop uses features_only_collate, so meta variability is harmless.
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        data: List[Dict[str, Any]] = []
        gen = ODEDataGenerator(num_samples, batch_size=64)
        for _, meta in tqdm(gen, total=num_samples, desc="Generating data"):
            data.append(meta)
        logger.info(f"Generated {len(data)} samples")
        return data

    # ----------------- Checkpoints -----------------
    def save_checkpoint(self, path: str, epoch: int):
        ckpt: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "learning_rate": self.learning_rate,
        }
        if self.scaler and self.enable_mixed_precision:
            try:
                ckpt["scaler_state_dict"] = self.scaler.state_dict()
            except Exception:
                pass
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return 0
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.history = checkpoint.get("history", self.history)
            if self.scaler and self.enable_mixed_precision and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            epoch = int(checkpoint.get("epoch", 0))
            logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
            return epoch
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0

    def save_model(self, path: str):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
        }
        torch.save(payload, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return False
        try:
            payload = torch.load(path, map_location=self.device)
            self.model.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.history = payload.get("history", self.history)
            logger.info(f"Loaded model from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    # ----------------- Generation / Eval / Export -----------------
    def generate_new_ode(self, seed: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        """
        Generate new ODE-style parameter vector and attempt to build an ODE via factories.
        """
        self.model.eval()
        with torch.no_grad():
            if seed is None:
                if self.model_type == "vae":
                    generated = self.model.sample(1)
                else:
                    seed = torch.randn(1, self.input_dim).to(self.device)
                    generated = self.model(seed)
            else:
                seed = seed.to(self.device)
                if self.model_type == "vae":
                    generated, _, _ = self.model(seed)
                else:
                    generated = self.model(seed)

            params = generated.detach().cpu().numpy()[0]

            alpha = float(np.clip(params[0], -100, 100))
            beta = float(np.clip(abs(params[1]) + 0.1, 0.1, 100))
            n = int(np.clip(abs(params[2]), 1, 10))
            M = float(np.clip(params[3], -100, 100))
            func_id = int(abs(params[4]))
            is_linear = bool(params[5] > 0.5)
            gen_num = int(np.clip(abs(params[6]), 1, 8 if is_linear else 10))
            q = int(np.clip(abs(params[8]) + 2, 2, 10))
            v = int(np.clip(abs(params[9]) + 2, 2, 10))
            a = float(np.clip(abs(params[10]) + 1, 1, 5))

            # Choose function name from basic set
            if self.basic_functions:
                funcs = self.basic_functions.get_function_names()
                if not funcs:
                    funcs = ["exp"]
            else:
                funcs = ["exp", "sin", "cos"]
            func_id = func_id % len(funcs)
            func_name = funcs[func_id]
            f_z = self.basic_functions.get_function(func_name) if self.basic_functions else None

            try:
                base_params = {"alpha": alpha, "beta": beta, "n": n, "M": M}
                if is_linear:
                    if LinearGeneratorFactory and f_z is not None:
                        factory = LinearGeneratorFactory()
                        if gen_num in [4, 5]:
                            base_params["a"] = a
                        res = factory.create(gen_num, f_z, **base_params)
                    else:
                        # fallback meta
                        res = {
                            "type": "linear",
                            "order": int(np.random.randint(1, 4)),
                            "generator_number": gen_num,
                            "rhs": sp.Integer(0),
                            "generator": sp.Symbol("L")[sp.Symbol("y")],
                            "solution": sp.Integer(0),
                        }
                else:
                    if NonlinearGeneratorFactory and f_z is not None:
                        factory = NonlinearGeneratorFactory()
                        extra = {}
                        if gen_num in [1, 2, 4]:
                            extra["q"] = q
                        if gen_num in [2, 3, 5]:
                            extra["v"] = v
                        if gen_num in [4, 5, 9, 10]:
                            extra["a"] = a
                        res = factory.create(gen_num, f_z, **{**base_params, **extra})
                    else:
                        res = {
                            "type": "nonlinear",
                            "order": int(np.random.randint(1, 4)),
                            "generator_number": gen_num,
                            "rhs": sp.Integer(0),
                            "generator": sp.Symbol("L")[sp.Symbol("y")],
                            "solution": sp.Integer(0),
                        }

                res["function_used"] = func_name
                res["ml_generated"] = True
                res["generation_params"] = {**base_params, "q": q, "v": v, "a": a}
                try:
                    res["ode"] = sp.Eq(res["generator"], res["rhs"])
                except Exception:
                    pass
                return res
            except Exception as e:
                logger.error(f"Error generating ODE: {e}")
                return None

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        self.model.eval()
        dataset = ODEDataset(test_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=features_only_collate)
        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_features, _ in loader:
                batch_features = batch_features.to(self.device)
                if self.model_type == "vae":
                    recon, mu, log_var = self.model(batch_features)
                    loss = self.criterion(recon, batch_features, mu, log_var)
                    output = recon
                else:
                    output = self.model(batch_features)
                    loss = self.criterion(output, batch_features)
                total_loss += loss.item()
                predictions.append(output.detach().cpu())
                targets.append(batch_features.detach().cpu())

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        pred_flat = predictions.flatten().numpy()
        target_flat = targets.flatten().numpy()
        correlation = float(np.corrcoef(pred_flat, target_flat)[0, 1])
        return {
            "loss": total_loss / max(len(loader), 1),
            "mse": mse,
            "mae": mae,
            "correlation": correlation,
        }

    def export_onnx(self, path: str):
        self.model.eval()
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"Model exported to ONNX: {path}")