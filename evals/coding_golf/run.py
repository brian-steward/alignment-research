from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List

from sandbox import is_valid, probe_opcounts
from fingerprint import strategy_key, l1_key, reference_strategies
from metrics import distinct_at, taxonomy_coverage, novel_count


@dataclass
class Result:
    problem: str
    n_attempts: int
    n_valid: int
    distinct_l1: int
    distinct_strategy: int
    coverage: int
    novel: int
    accepted: list   # [complexity, sorted(features)] per distinct strategy


def evaluate(problem, sources: List[str]) -> Result:
    ref = reference_strategies(problem)
    l1s, strat_keys = [], []
    for src in sources:
        if not is_valid(src, problem):
            continue
        counts = probe_opcounts(src, problem)
        l1s.append(l1_key(src))
        strat_keys.append(strategy_key(src, problem, counts))
    found = set(strat_keys)
    accepted = sorted([k[0], sorted(k[1])] for k in found)
    return Result(
        problem=problem.name,
        n_attempts=len(sources),
        n_valid=len(strat_keys),
        distinct_l1=distinct_at(l1s),
        distinct_strategy=len(found),
        coverage=taxonomy_coverage(found, ref),
        novel=novel_count(found, ref),
        accepted=accepted,
    )


def save_result(result: Result, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(asdict(result), fh, indent=2)
