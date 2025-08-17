# shared/phi_library.py
import numpy as np
from typing import Callable, Dict

def _safe_exp(u):  # stable exp
    u = np.clip(u, -50, 50)
    return np.exp(u)

def logistic(u):
    return 1.0 / (1.0 + _safe_exp(-u))

def d_logistic(u):
    s = logistic(u)
    return s * (1.0 - s)

PHI_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "exp":      lambda u: _safe_exp(u),
    "erf":      lambda u: np.vectorize(lambda x: float(__import__("math").erf(x)))(u),
    "sinh":     lambda u: np.sinh(np.clip(u, -50, 50)),
    "cosh":     lambda u: np.cosh(np.clip(u, -50, 50)),
    "tanh":     lambda u: np.tanh(u),
    "sin":      np.sin,
    "cos":      np.cos,
    "logistic": logistic,
}

SIG_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "exp":      lambda u: _safe_exp(u),
    "erf":      lambda u: 2.0 / np.sqrt(np.pi) * np.exp(-np.clip(u, -50, 50)**2),
    "sinh":     lambda u: np.cosh(np.clip(u, -50, 50)),
    "cosh":     lambda u: np.sinh(np.clip(u, -50, 50)),
    "tanh":     lambda u: 1.0 - np.tanh(u)**2,
    "sin":      np.cos,
    "cos":      lambda u: -np.sin(u),
    "logistic": d_logistic,
}