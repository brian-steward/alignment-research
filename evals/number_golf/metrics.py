from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fingerprint import canon_size


def discovery_curve(keys: Sequence) -> Tuple[List[int], List[int]]:
    seen = set()
    n: List[int] = []
    V: List[int] = []
    for i, k in enumerate(keys, start=1):
        seen.add(k)
        n.append(i)
        V.append(len(seen))
    return n, V


def fit_heaps(n: Sequence[int], V: Sequence[int]) -> Tuple[float, float, float]:
    """Fit V = K * n**beta by least squares on log-log. Returns (K, beta, r2)."""
    n_arr = np.asarray(n, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    mask = (n_arr > 0) & (V_arr > 0)
    if mask.sum() < 2:
        return (float("nan"), float("nan"), float("nan"))
    x = np.log(n_arr[mask])
    y = np.log(V_arr[mask])
    beta, log_k = np.polyfit(x, y, 1)
    y_hat = beta * x + log_k
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return (float(np.exp(log_k)), float(beta), float(r2))


def distinct_at(keys: Sequence, budget: int) -> int:
    """Distinct keys among the first `budget` entries. `None` entries mark
    invalid attempts: they consume budget but contribute no distinct key, so
    this scalar folds yield together with diversity."""
    return len({k for k in keys[:budget] if k is not None})


def yield_rate(n_valid: int, n_attempts: int) -> float:
    if n_attempts == 0:
        return 0.0
    return n_valid / n_attempts


def max_complexity(keys: Sequence) -> int:
    return max((canon_size(k) for k in keys), default=0)


def bootstrap_beta(keys: Sequence, iters: int = 200, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys = list(keys)
    m = len(keys)
    if m < 2:
        return (float("nan"), float("nan"))
    betas = []
    for _ in range(iters):
        idx = rng.integers(0, m, m)
        sample = [keys[j] for j in idx]
        n, V = discovery_curve(sample)
        _, beta, _ = fit_heaps(n, V)
        if not np.isnan(beta):
            betas.append(beta)
    if not betas:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(betas, [2.5, 97.5])
    return (float(lo), float(hi))
