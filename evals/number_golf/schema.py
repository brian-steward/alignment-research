from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet

ALL_OPS: FrozenSet[str] = frozenset({"+", "-", "*", "/", "^", "!", "neg", "concat"})


@dataclass(frozen=True)
class Caps:
    """Evaluation-safety bounds (guard against runaway, not a conceptual cap on space)."""
    max_abs: int = 10 ** 15        # max |numerator| or |denominator| of any intermediate
    max_fact_arg: int = 9          # factorial only defined on integers 0..max_fact_arg
    max_pow_exp: int = 64          # |exponent| cap for '^'
    max_nodes: int = 200           # max AST nodes in a submitted solution
    max_int_value: int = 10 ** 18  # cap on a concatenated literal's value


@dataclass(frozen=True)
class Config:
    digit: int
    target: int
    ops: FrozenSet[str]
    caps: Caps = field(default_factory=Caps)
    name: str = ""

    def __post_init__(self):
        if not (1 <= self.digit <= 9):
            raise ValueError(f"digit must be 1..9, got {self.digit}")
        unknown = set(self.ops) - ALL_OPS
        if unknown:
            raise ValueError(f"unknown ops: {unknown}")
