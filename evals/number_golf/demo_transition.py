from __future__ import annotations

from typing import Dict

from schema import Config, Caps
from enumerate_golf import enumerate_solutions
from generator import _TIER_OPS, OP_TIERS
from plots import plot_transition


def transition_counts(target: int, max_leaves: int, digit: int = 4) -> Dict[str, int]:
    """Ground-truth distinct-solution counts per operator tier (deterministic)."""
    counts: Dict[str, int] = {}
    for tier_name, ops in _TIER_OPS.items():
        c = Config(digit=digit, target=target, ops=ops, caps=Caps(), name=f"{tier_name}-{target}")
        counts[tier_name] = len(enumerate_solutions(c, max_leaves=max_leaves))
    return counts


def main(out_path: str = "results/transition.png", target: int = 24, max_leaves: int = 5) -> None:
    counts = transition_counts(target=target, max_leaves=max_leaves)
    tiers = sorted(counts, key=lambda t: OP_TIERS[t])
    plot_transition(tiers, [counts[t] for t in tiers], out_path)
    print({t: counts[t] for t in tiers})


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main()
