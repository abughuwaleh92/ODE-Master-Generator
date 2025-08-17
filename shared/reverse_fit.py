# shared/reverse_fit.py
"""
Numeric reverse engineering of
y(x) ≈ x^M * ∏_{j=1..J} [ φ_j( α_j * (a_j x + b_j)^{β_j} ) ]^{n_j}
Supports J=1..3 with simple grid + least-squares on the outer scale.

This module is independent; the app can combine it with ML suggestions if desired.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from .phi_library import PHI_FUNCS

def _safe_pow(x, p):
    with np.errstate(all='ignore'):
        return np.sign(x) * (np.abs(x) ** p)

def _model_single(x, params, phi_name: str):
    # y ≈ C * x^M * φ( α * (a x + b)^β )^n
    C  = params.get("C", 1.0)
    M  = params.get("M", 0.0)
    a  = params.get("a", 1.0)
    b  = params.get("b", 0.0)
    β  = params.get("beta", 1.0)
    α  = params.get("alpha", 1.0)
    n  = params.get("n", 1.0)
    φ  = PHI_FUNCS[phi_name]
    u  = α * (_safe_pow(a * x + b, β))
    return C * (_safe_pow(x, M)) * (φ(u) ** n)

def _lstsq_scale(y_true, y_hat):
    # solve C in min ||y_true - C*y_hat||^2 ⇒ C = (y_hatᵀ y_true)/(y_hatᵀ y_hat)
    denom = float(np.dot(y_hat, y_hat))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(y_hat, y_true) / denom)

def fit_single_block(x: np.ndarray, y: np.ndarray,
                     phi_names: List[str] = None,
                     grids: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
    """
    Returns best params for one φ-block.
    grids: dict with arrays for each parameter we grid over.
           keys: 'M','beta','alpha','n','a','b'
    """
    if phi_names is None:
        phi_names = ["exp","erf","sinh","cosh","tanh","sin","cos","logistic"]
    if grids is None:
        grids = {
            "M":    np.linspace(-3, 3, 13),
            "beta": np.linspace(0.2, 3.0, 15),
            "alpha":np.linspace(-3, 3, 13),
            "n":    np.linspace(1, 5, 9),
            "a":    np.linspace(0.5, 2.0, 11),
            "b":    np.linspace(-1.0, 1.0, 11),
        }

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 5:
        raise ValueError("Too few points for reverse fit.")

    best = {"loss": np.inf, "phi": None, "params": {}}
    for phi in phi_names:
        for M in grids["M"]:
            xM = _safe_pow(x, M)
            for β in grids["beta"]:
                for a in grids["a"]:
                    for b in grids["b"]:
                        u = (a * x + b)
                        u = np.where(np.abs(u) < 1e-9, 1e-9, u)  # avoid singular
                        uβ = _safe_pow(u, β)
                        for α in grids["alpha"]:
                            φu = PHI_FUNCS[phi](α * uβ)
                            # We'll fit C*n jointly by allowing n in grid and C via least-squares
                            for n in grids["n"]:
                                y_hat = xM * (φu ** n)
                                C = _lstsq_scale(y, y_hat)
                                pred = C * y_hat
                                loss = float(np.mean((y - pred) ** 2))
                                if loss < best["loss"]:
                                    best = {
                                        "loss": loss,
                                        "phi": phi,
                                        "params": {"C": C, "M": float(M),
                                                   "beta": float(β), "alpha": float(α),
                                                   "n": float(n), "a": float(a), "b": float(b)}
                                    }
    return best

def fit_multi_block(x: np.ndarray, y: np.ndarray, J: int = 2,
                    phi_names: List[str] = None) -> Dict[str, Any]:
    """
    Simple greedy multi-block:
    y ≈ x^M * Π φ_j(… )^{n_j}
    We fit block 1, divide y by its contribution, fit block 2, etc.
    """
    assert J >= 1 and J <= 3
    residual = np.asarray(y, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    blocks = []
    for j in range(J):
        res = fit_single_block(x, residual, phi_names=phi_names)
        blocks.append(res)
        # divide out block j (avoid zeros)
        contrib = _model_single(x, res["params"], res["phi"])
        contrib = np.where(np.abs(contrib) < 1e-12, 1e-12, contrib)
        residual = residual / contrib
    # reconstruct composite and evaluate loss
    y_hat = np.ones_like(y, dtype=float)
    # unify M as sum of M_j; take mean M to keep structure simple
    M_eff = float(np.mean([b["params"]["M"] for b in blocks]))
    y_hat *= _safe_pow(x, M_eff)
    for b in blocks:
        pb = b["params"].copy(); pb["M"] = 0.0; pb["C"] = 1.0
        y_hat *= _model_single(x, pb, b["phi"])
    C = _lstsq_scale(y, y_hat)
    y_pred = C * y_hat
    loss = float(np.mean((y - y_pred) ** 2))
    return {"loss": loss, "M": M_eff, "C": C, "blocks": blocks}

def reverse_from_samples(x: np.ndarray, y: np.ndarray,
                         allow_multi: bool = True) -> Dict[str, Any]:
    """
    Entry-point: reverse engineer parameters from samples.
    If allow_multi, try J=1 and J=2 and pick best.
    """
    best1 = fit_single_block(x, y)
    if allow_multi:
        try:
            best2 = fit_multi_block(x, y, J=2)
            if best2["loss"] < best1["loss"]:
                return {"mode": "multi", **best2}
        except Exception:
            pass
    return {"mode": "single", **best1}