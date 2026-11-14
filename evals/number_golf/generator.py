from __future__ import annotations

from typing import Dict, List

from schema import Config

ARITH = frozenset({"+", "-", "*", "/"})
ARITH_CONCAT = ARITH | frozenset({"concat"})
FULL = ARITH_CONCAT | frozenset({"^", "!", "neg"})

# Higher tier number == richer operator set == easier (less chess-like).
OP_TIERS: Dict[str, int] = {"arith": 0, "arith_concat": 1, "full": 2}

_TIER_OPS = {"arith": ARITH, "arith_concat": ARITH_CONCAT, "full": FULL}

# Fixed digit and a spread of targets (a prime, a round-ish, a larger one).
_DIGIT = 4
_TARGETS = [17, 42, 100]


def _build() -> List[Config]:
    configs: List[Config] = []
    for tier_name, ops in _TIER_OPS.items():
        for t in _TARGETS:
            configs.append(Config(digit=_DIGIT, target=t, ops=ops, name=f"{tier_name}-{t}"))
    return configs


STARTER_CONFIGS: List[Config] = _build()


def difficulty_rank(config: Config) -> int:
    """Intrinsic difficulty: fewer ops == harder == larger rank.
    Rank = (max tier) - (this config's tier)."""
    tier_name = config.name.split("-")[0]
    if tier_name not in OP_TIERS:
        # fall back to op-count: fewer ops -> harder
        return len(_TIER_OPS["full"]) - len(config.ops)
    max_tier = max(OP_TIERS.values())
    return max_tier - OP_TIERS[tier_name]
