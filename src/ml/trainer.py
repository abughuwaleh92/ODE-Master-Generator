# src/ml/trainer.py
"""
Enhanced ML Trainer for ODE Generators
- Stable training with masked/weighted losses
- Optional feature normalization
- Early stopping + ReduceLROnPlateau
- β-VAE with KL annealing
- Dataloader collate fix (avoid dict KeyErrors)
- Persistent progress/log hooks for RQ
- Session save/load/export (artifacts.zip)
- Safe generate() and reverse_engineer() helpers
"""

import os
import io
import gc
import json
import zipfile
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
from tqdm import tqdm

# --------- logging ----------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# --------- safe imports from your src ---------
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
except Exception as e:
    logger.warning(f"Trainer: some optional modules missing: {e}")
    LinearGeneratorFactory = NonlinearGeneratorFactory = None
    BasicFunctions = SpecialFunctions = None
    GeneratorPatternLearner = GeneratorVAE = GeneratorTransformer = None
    create_model = None


# =========================
# Dataset / Generator
# =========================

class ODEDataset(Dataset):
    """
    Dataset returning ONLY feature tensors (dicts are not collated to avoid KeyErrors).
    """
    def __init__(self, data: List[Dict[str, Any]], max_cache_size: int = 1000):
        self.data = data
        self.max_cache_size = min(max_cache_size, len(data))
        self._feature_cache: Dict[int, torch.Tensor] = {}

    def _extract_features(self, item: Dict[str, Any]) -> torch.Tensor:
        # 12 dims: [alpha, beta, n, M, func_id, is_linear, gen_num, order, q, v, a, noise]
        alpha = float(item.get('alpha', 1.0))
        beta  = float(item.get('beta', 1.0))
        n     = float(item.get('n', 1))
        M     = float(item.get('M', 0.0))
        func_id = float(item.get('function_id', 0))
        is_linear = 1.0 if item.get('type', 'linear') == 'linear' else 0.0
        gen_num = float(item.get('generator_number', 1))
        order   = float(item.get('order', 2))
        q = float(item.get('q', 0))
        v = float(item.get('v', 0))
        a = float(item.get('a', 0.0))
        noise = np.random.randn() * 0.1  # regularization noise (we'll ignore in loss)

        arr = np.array([alpha, beta, n, M, func_id, is_linear, gen_num, order, q, v, a, noise], dtype=np.float32)
        return torch.from_numpy(arr)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self._feature_cache:
            return self._feature_cache[idx]
        t = self._extract_features(self.data[idx])
        if len(self._feature_cache) < self.max_cache_size:
            self._feature_cache[idx] = t
        return t


class ODEDataGenerator(IterableDataset):
    """
    Streaming generator (memory efficient).
    Yields feature tensors only, so DataLoader doesn't collate dicts.
    """
    def __init__(self, num_samples: int, seed: Optional[int] = None):
        self.num_samples = int(num_samples)
        self.seed = seed

        self.linear_factory = LinearGeneratorFactory() if LinearGeneratorFactory else None
        self.nonlinear_factory = NonlinearGeneratorFactory() if NonlinearGeneratorFactory else None
        self.basic_functions = BasicFunctions() if BasicFunctions else None
        self.special_functions = SpecialFunctions() if SpecialFunctions else None

        self.basic_names = self.basic_functions.get_function_names() if self.basic_functions else []
        self.special_names = self.special_functions.get_function_names()[:10] if self.special_functions else []
        self.all_names = (self.basic_names or []) + (self.special_names or [])

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        produced = 0
        while produced < self.num_samples:
            try:
                alpha = np.random.uniform(-5, 5)
                beta  = np.random.uniform(0.1, 5)
                n     = np.random.randint(1, 5)
                M     = np.random.uniform(-5, 5)

                func_id = 0
                if self.all_names:
                    func_name = np.random.choice(self.all_names)
                    if func_name in self.basic_names:
                        func_id = self.basic_names.index(func_name)
                    else:
                        func_id = len(self.basic_names) + self.special_names.index(func_name)

                is_linear = np.random.choice([0.0, 1.0])
                if is_linear > 0.5:
                    gen_num = np.random.randint(1, 9)
                    order = np.random.randint(1, 5)
                    q=v=a=0
                else:
                    gen_num = np.random.randint(1, 11)
                    order = np.random.randint(1, 6)
                    q = np.random.randint(0, 6)
                    v = np.random.randint(0, 6)
                    a = float(np.random.uniform(0, 5))

                noise = np.random.randn() * 0.1

                feat = np.array([
                    alpha, beta, n, M, float(func_id), float(is_linear),
                    float(gen_num), float(order), float(q), float(v), float(a), float(noise)
                ], dtype=np.float32)
                yield torch.from_numpy(feat)
                produced += 1
            except Exception:
                continue

    def __len__(self) -> int:
        return self.num_samples


# =========================
# Config / Utilities
# =========================

@dataclass
class TrainConfig:
    model_type: str = "pattern_learner"  # 'pattern_learner' | 'vae' | 'transformer'
    input_dim: int = 12
    hidden_dim: int = 128
    output_dim: int = 12
    learning_rate: float = 1e-3
    enable_mixed_precision: bool = False
    normalize: bool = False
    beta_vae: float = 1.0          # β for β-VAE
    kl_anneal: str = "linear"      # 'none'|'linear'|'sigmoid'
    kl_max_beta: float = 1.0       # target β
    kl_warmup_epochs: int = 10     # how many epochs to reach kl_max_beta
    early_stop_patience: int = 12
    loss_weights: Optional[List[float]] = None  # length=input_dim or None

    # derived fields not serialized
    device: Optional[str] = None
    checkpoint_dir: str = "checkpoints"


def _default_loss_weights(input_dim: int) -> torch.Tensor:
    w = torch.ones(input_dim, dtype=torch.float32)
    # dims: [alpha, beta, n, M, func_id, is_linear, gen_num, order, q, v, a, noise]
    if input_dim >= 12:
        w[4] = 0.3   # function_id (discrete-ish)
        w[5] = 0.2   # is_linear (binary)
        w[6] = 0.3   # generator_number (discrete-ish)
        w[11] = 0.0  # ignore noise
    return w


def _collate_features_only(batch: List[torch.Tensor]) -> torch.Tensor:
    # Each item is a tensor of shape [D]; stack to [B, D]
    return torch.stack(batch, dim=0)


# =========================
# Trainer
# =========================

class MLTrainer:
    """
    Unified trainer with:
    - masked/weighted loss, normalization
    - β-VAE (KL annealing)
    - progress/log hooks for RQ worker
    - session save/load/export
    """

    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(config.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # model
        self.model = self._create_model().to(self.device)

        # optimizer/scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)

        # mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.enable_mixed_precision)

        # loss weights (mask)
        lw = self.cfg.loss_weights if self.cfg.loss_weights else _default_loss_weights(self.cfg.input_dim).tolist()
        self.loss_weights = torch.tensor(lw, dtype=torch.float32, device=self.device).view(1, -1)

        # feature normalization (rough ranges; adjust later if you wish)
        self.normalize = bool(self.cfg.normalize)
        mu = np.zeros(self.cfg.input_dim, dtype=np.float32)
        sg = np.ones(self.cfg.input_dim, dtype=np.float32)
        # heuristics: center/scales based on generator sampling ranges
        if self.cfg.input_dim >= 12:
            mu[:11] = np.array([0.0, 2.55, 3.0, 0.0, 5.0, 0.5, 5.0, 3.0, 2.5, 2.5, 2.5], dtype=np.float32)
            sg[:11] = np.array([5.0, 2.5, 2.0, 5.0,10.0, 0.5, 5.0, 2.0, 2.5, 2.5, 2.5], dtype=np.float32)
            sg[11] = 0.2
        self._mu = torch.tensor(mu, device=self.device).view(1, -1)
        self._sg = torch.tensor(sg, device=self.device).view(1, -1)

        # history
        self.history: Dict[str, Any] = {
            "train_loss": [], "val_loss": [],
            "epochs": 0,
            "best_val_loss": float("inf"),
            "best_model_path": None,
            "config": asdict(self.cfg),
            "timestamps": [],
        }

        # external hooks (set by worker)
        self.progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None  # receives dict
        self.log_hook: Optional[Callable[[str], None]] = None

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    # ------------- model -------------
    def _create_model(self) -> nn.Module:
        if self.cfg.model_type == 'pattern_learner':
            return GeneratorPatternLearner(
                input_dim=self.cfg.input_dim,
                hidden_dim=self.cfg.hidden_dim,
                output_dim=self.cfg.output_dim
            )
        elif self.cfg.model_type == 'vae':
            return GeneratorVAE(
                input_dim=self.cfg.input_dim,
                hidden_dim=self.cfg.hidden_dim,
                latent_dim=32
            )
        elif self.cfg.model_type == 'transformer':
            return GeneratorTransformer(
                input_dim=self.cfg.input_dim,
                d_model=self.cfg.hidden_dim
            )
        else:
            raise ValueError(f"Unknown model type: {self.cfg.model_type}")

    # ------------- utils -------------
    def _pre(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        return (x - self._mu) / (self._sg + 1e-6)

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, D], weights: [1, D]
        diff2 = (pred - target) ** 2
        return torch.mean(diff2 * self.loss_weights)

    def _log(self, msg: str):
        if self.log_hook:
            try: self.log_hook(msg)
            except Exception: pass
        logger.info(msg)

    # ------------- session I/O -------------
    def save_checkpoint(self, path: str, epoch: int):
        ckpt = {
            "epoch": epoch,
            "model_type": self.cfg.model_type,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "history": self.history,
            "config": asdict(self.cfg),
        }
        if self.scaler is not None:
            ckpt["scaler"] = self.scaler.state_dict()
        torch.save(ckpt, path)
        self._log(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            self._log(f"Checkpoint not found: {path}")
            return 0
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.history = ckpt.get("history", self.history)
        if self.scaler is not None and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        epoch = int(ckpt.get("epoch", 0))
        self._log(f"Checkpoint loaded (epoch {epoch}) from {path}")
        return epoch

    def save_session_dir(self, out_dir: str, best_path: Optional[str] = None) -> str:
        """
        Save full training session into a directory; returns dir path.
        """
        os.makedirs(out_dir, exist_ok=True)
        # Save latest checkpoint
        ckpt_path = os.path.join(out_dir, "latest_checkpoint.pth")
        self.save_checkpoint(ckpt_path, self.history.get("epochs", 0))

        # Copy best (if provided)
        if best_path and os.path.exists(best_path):
            dst = os.path.join(out_dir, "best.pth")
            if best_path != dst:
                try:
                    import shutil; shutil.copy2(best_path, dst)
                except Exception:
                    pass

        # Save history/config
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"saved_at": datetime.now().isoformat()}, f)

        return out_dir

    def export_session_zip(self, out_zip_path: str, best_path: Optional[str] = None) -> str:
        """
        Export full training session as a ZIP; returns zip path.
        """
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # checkpoint & config
            tmp_ckpt = io.BytesIO()
            torch.save({
                "epoch": self.history.get("epochs", 0),
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "history": self.history,
                "config": asdict(self.cfg),
            }, tmp_ckpt)
            zf.writestr("latest_checkpoint.pth", tmp_ckpt.getvalue())

            if best_path and os.path.exists(best_path):
                with open(best_path, "rb") as f:
                    zf.writestr("best.pth", f.read())

            zf.writestr("history.json", json.dumps(self.history, indent=2).encode("utf-8"))
            zf.writestr("config.json", json.dumps(asdict(self.cfg), indent=2).encode("utf-8"))
            zf.writestr("meta.json", json.dumps({"saved_at": datetime.now().isoformat()}, indent=2).encode("utf-8"))
        with open(out_zip_path, "wb") as f:
            f.write(buf.getvalue())
        self._log(f"Session exported: {out_zip_path}")
        return out_zip_path

    def load_session_zip(self, session_zip_path: str) -> bool:
        """
        Load model + optimizer + scheduler + history from a ZIP produced by export_session_zip.
        """
        if not os.path.exists(session_zip_path):
            self._log(f"Session zip not found: {session_zip_path}")
            return False
        with zipfile.ZipFile(session_zip_path, "r") as zf:
            if "latest_checkpoint.pth" in zf.namelist():
                raw = zf.read("latest_checkpoint.pth")
                ckpt = torch.load(io.BytesIO(raw), map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.scheduler.load_state_dict(ckpt["scheduler"])
                self.history = ckpt.get("history", self.history)
            elif "best.pth" in zf.namelist():
                raw = zf.read("best.pth")
                ckpt = torch.load(io.BytesIO(raw), map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.history = ckpt.get("history", self.history)
            else:
                self._log("Session zip does not contain a known checkpoint.")
                return False
        self._log(f"Session loaded from zip: {session_zip_path}")
        return True

    # ------------- training -------------
    def _kl_beta(self, epoch: int) -> float:
        if self.cfg.model_type != "vae":
            return 0.0
        if self.cfg.kl_anneal == "none":
            return float(self.cfg.beta_vae)
        e = max(1, int(epoch))
        if self.cfg.kl_anneal == "linear":
            r = min(1.0, e / max(1, self.cfg.kl_warmup_epochs))
            return float(self.cfg.kl_max_beta) * r
        elif self.cfg.kl_anneal == "sigmoid":
            import math
            # centered sigmoid ramp  over warmup_epochs
            k = 10.0 / max(1, self.cfg.kl_warmup_epochs)
            x = k * (e - self.cfg.kl_warmup_epochs / 2)
            r = 1.0 / (1.0 + math.exp(-x))
            return float(self.cfg.kl_max_beta) * r
        return float(self.cfg.beta_vae)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        samples: int = 1000,
        validation_split: float = 0.2,
        use_generator: bool = True,
        resume_from: Optional[str] = None,
        checkpoint_interval: int = 10,
    ):
        """
        Train with stability & persistence hooks.
        """
        self._log(f"Starting training for {epochs} epochs (model={self.cfg.model_type})")

        # Resume if requested
        if resume_from:
            self.load_checkpoint(resume_from)

        # Build loaders
        if use_generator:
            train_samples = int(samples * (1 - validation_split))
            val_samples = samples - train_samples

            train_gen = ODEDataGenerator(train_samples, seed=None)
            val_gen   = ODEDataGenerator(val_samples,   seed=42)
            train_loader = DataLoader(train_gen, batch_size=batch_size, num_workers=0,
                                      collate_fn=_collate_features_only)
            val_loader   = DataLoader(val_gen,   batch_size=batch_size, num_workers=0,
                                      collate_fn=_collate_features_only)
        else:
            # single cached dataset (stable val metrics)
            data = []
            gen = ODEDataGenerator(samples, seed=123)
            for t in tqdm(gen, total=samples, desc="Preparing dataset"):
                data.append({"alpha": t[0].item(), "beta": t[1].item(), "n": t[2].item(), "M": t[3].item(),
                             "function_id": t[4].item(), "type": "linear" if t[5].item() > 0.5 else "nonlinear",
                             "generator_number": t[6].item(), "order": t[7].item(),
                             "q": t[8].item(), "v": t[9].item(), "a": t[10].item()})
            ds = ODEDataset(data, max_cache_size=min(2000, len(data)))
            val_size = int(len(ds) * validation_split)
            train_size = len(ds) - val_size
            train_ds, val_ds = random_split(ds, [train_size, val_size])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      collate_fn=_collate_features_only)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                      collate_fn=_collate_features_only)

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(1, epochs+1):
            # ---- train ----
            self.model.train()
            tr_loss = 0.0
            nb = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
            for batch in pbar:
                x = batch.to(self.device)
                x_pre = self._pre(x)

                with torch.cuda.amp.autocast(enabled=self.cfg.enable_mixed_precision):
                    if self.cfg.model_type == "vae":
                        recon, mu, log_var = self.model(x_pre)
                        recon_loss = self._masked_mse(recon, x_pre)
                        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        beta = self._kl_beta(epoch)
                        loss = recon_loss + beta * kld
                    else:
                        out = self.model(x_pre)
                        loss = self._masked_mse(out, x_pre)

                self.optimizer.zero_grad()
                if self.cfg.enable_mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                tr_loss += loss.item()
                nb += 1
                pbar.set_postfix({"loss": tr_loss / nb})

            # ---- validate ----
            self.model.eval()
            va_loss = 0.0
            vb = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(self.device)
                    x_pre = self._pre(x)
                    if self.cfg.model_type == "vae":
                        recon, mu, log_var = self.model(x_pre)
                        recon_loss = self._masked_mse(recon, x_pre)
                        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        beta = self._kl_beta(epoch)
                        loss = recon_loss + beta * kld
                    else:
                        out = self.model(x_pre)
                        loss = self._masked_mse(out, x_pre)
                    va_loss += loss.item()
                    vb += 1

            tr = tr_loss / max(1, nb)
            va = va_loss / max(1, vb)
            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(va)
            self.history["epochs"] = epoch
            self.history["timestamps"].append(datetime.now().isoformat())

            self.scheduler.step(va)
            self._log(f"Epoch {epoch}: Train {tr:.4f} | Val {va:.4f}")

            # ---- save best / checkpoint ----
            if va < best_val - 1e-6:
                best_val = va
                self.history["best_val_loss"] = best_val
                best_path = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.model_type}_best.pth")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "history": self.history,
                    "model_type": self.cfg.model_type
                }, best_path)
                self.history["best_model_path"] = best_path
                self._log(f"Saved best model to {best_path} (val={best_val:.4f})")
                bad_epochs = 0
            else:
                bad_epochs += 1

            if (epoch % max(1, checkpoint_interval)) == 0:
                ck = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.model_type}_epoch_{epoch}.pth")
                self.save_checkpoint(ck, epoch)

            # ---- progress hook for RQ ----
            if self.progress_hook:
                try:
                    self.progress_hook({
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": tr,
                        "val_loss": va,
                        "best_val": best_val,
                        "best_model_path": self.history.get("best_model_path"),
                        "timestamp": datetime.now().isoformat(),
                    })
                except Exception:
                    pass

            # ---- early stop ----
            if self.cfg.early_stop_patience and bad_epochs >= self.cfg.early_stop_patience:
                self._log(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        self._log("Training completed.")

    # ------------- inference helpers -------------
    @torch.no_grad()
    def generate_new_ode(self, num: int = 1) -> List[Dict[str, Any]]:
        """
        Generate synthetic "feature vectors" and decode via the model.
        For pattern_learner/transformer, feed random z; for VAE, sample latent.
        Returns list of dicts with decoded parameter suggestions.
        """
        self.model.eval()
        results = []
        for _ in range(num):
            if self.cfg.model_type == "vae" and hasattr(self.model, "sample"):
                y = self.model.sample(1).to(self.device)  # assume returns normalized space
            else:
                y = torch.randn(1, self.cfg.input_dim, device=self.device)

            # inverse of _pre if normalized
            if self.normalize:
                y = y * self._sg + self._mu

            arr = y.squeeze(0).detach().cpu().numpy().tolist()
            # decode to parameter suggestion (basic bounds)
            d = {
                "alpha": float(np.clip(arr[0], -10, 10)),
                "beta":  float(np.clip(abs(arr[1]), 0.1, 10)),
                "n":     int(np.clip(round(abs(arr[2])), 1, 10)),
                "M":     float(np.clip(arr[3], -10, 10)),
                "function_id": int(np.clip(round(abs(arr[4])), 0, 999)),
                "type": "linear" if arr[5] > 0.5 else "nonlinear",
                "generator_number": int(np.clip(round(abs(arr[6])), 1, 10)),
                "order": int(np.clip(round(abs(arr[7])), 1, 8)),
                "q": int(np.clip(round(abs(arr[8])), 0, 10)),
                "v": int(np.clip(round(abs(arr[9])), 0, 10)),
                "a": float(np.clip(abs(arr[10]), 0.0, 5.0)),
            }
            results.append(d)
        return results

    @torch.no_grad()
    def reverse_engineer(self, target_vec: np.ndarray) -> Dict[str, Any]:
        """
        Simple 'reverse engineering': project a target feature vector into
        model space (optionally normalized), and return a smoothed/valid clip.
        You can extend this to solve for parameters from observations, etc.
        """
        x = torch.tensor(target_vec, dtype=torch.float32, device=self.device).view(1, -1)
        x_pre = self._pre(x) if self.normalize else x
        if self.cfg.model_type == "vae":
            # pass through encoder/decoder if exposed; else identity
            try:
                recon, _, _ = self.model(x_pre)
            except Exception:
                recon = x_pre
        else:
            recon = self.model(x_pre)
        y = recon
        if self.normalize:
            y = y * self._sg + self._mu
        arr = y.squeeze(0).detach().cpu().numpy()
        # map back to clean dict
        return {
            "alpha": float(np.clip(arr[0], -10, 10)),
            "beta":  float(np.clip(abs(arr[1]), 0.1, 10)),
            "n":     int(np.clip(round(abs(arr[2])), 1, 10)),
            "M":     float(np.clip(arr[3], -10, 10)),
            "function_id": int(np.clip(round(abs(arr[4])), 0, 999)),
            "type": "linear" if arr[5] > 0.5 else "nonlinear",
            "generator_number": int(np.clip(round(abs(arr[6])), 1, 10)),
            "order": int(np.clip(round(abs(arr[7])), 1, 8)),
            "q": int(np.clip(round(abs(arr[8])), 0, 10)),
            "v": int(np.clip(round(abs(arr[9])), 0, 10)),
            "a": float(np.clip(abs(arr[10]), 0.0, 5.0)),
        }