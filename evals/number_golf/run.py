from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

from validate import parse, is_valid
from fingerprint import l2_key
from metrics import (
    discovery_curve, fit_heaps, distinct_at, yield_rate, max_complexity,
    bootstrap_beta,
)


@dataclass
class RunResult:
    config_name: str
    protocol: str
    n_attempts: int
    n_valid: int
    yield_rate: float
    beta: float
    K: float
    r2: float
    beta_ci: Tuple[float, float]
    distinct_at: Dict[str, int]
    max_complexity: int


def run_config(config, solver, protocol: str, n_attempts: int,
               budgets: Sequence[int] = (10, 50, 100)) -> RunResult:
    raw = solver.solve(config, n_attempts, protocol)
    per_attempt = []  # attempt-aligned: L2 key if valid, else None
    valid_keys = []
    for s in raw:
        if is_valid(s, config):
            key = l2_key(parse(s, config))
            per_attempt.append(key)
            valid_keys.append(key)
        else:
            per_attempt.append(None)

    n, V = discovery_curve(valid_keys)
    if valid_keys:
        K, beta, r2 = fit_heaps(n, V)
        ci = bootstrap_beta(valid_keys, iters=200, seed=0)
        mc = max_complexity(valid_keys)
    else:
        K, beta, r2 = float("nan"), float("nan"), float("nan")
        ci = (float("nan"), float("nan"))
        mc = 0

    return RunResult(
        config_name=config.name,
        protocol=protocol,
        n_attempts=len(raw),
        n_valid=len(valid_keys),
        yield_rate=yield_rate(len(valid_keys), len(raw)),
        beta=beta,
        K=K,
        r2=r2,
        beta_ci=ci,
        distinct_at={str(b): distinct_at(per_attempt, b) for b in budgets},
        max_complexity=mc,
    )


def save_results(results: List[RunResult], path: str) -> None:
    with open(path, "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
