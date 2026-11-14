from __future__ import annotations

import json
from typing import List, Optional, Protocol, Sequence

_OP_NAMES = {
    "+": "addition (+)", "-": "subtraction (-)", "*": "multiplication (*)",
    "/": "division (/)", "^": "exponentiation (^)", "!": "factorial (!)",
    "neg": "unary minus (-)", "concat": "digit concatenation (e.g. 44)",
}


def _ops_phrase(config) -> str:
    return ", ".join(_OP_NAMES[o] for o in sorted(config.ops))


def build_prompt(config, protocol: str, prior: Optional[Sequence[str]] = None) -> str:
    base = (
        f"You are playing number-golf. Using ONLY the digit {config.digit} "
        f"(any number of times) and these operations: {_ops_phrase(config)}, "
        f"write a mathematical expression that evaluates exactly to {config.target}.\n"
        f"Output ONLY the expression on a single line, no words, no explanation."
    )
    if protocol == "A1":
        return base + "\nGive exactly one expression."
    if protocol == "A2":
        prior = list(prior or [])
        prior_block = "\n".join(f"- {p}" for p in prior) if prior else "(none yet)"
        return (
            base
            + "\nYou have already given these solutions:\n"
            + prior_block
            + "\nGive ONE NEW expression that is genuinely different from all of the above."
        )
    raise ValueError(f"unknown protocol {protocol!r}")


class Solver(Protocol):
    def solve(self, config, n: int, protocol: str) -> List[str]:
        ...


class MockSolver:
    """Deterministic solver for tests."""

    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)

    def solve(self, config, n: int, protocol: str) -> List[str]:
        return self._responses[:n]


class JsonlSolver:
    """Reads pre-generated model responses from a JSONL file.

    Each line: {"config": <config.name>, "expr": <expression string>}.
    This is the production path: Anthropic subagents generate the JSONL
    out-of-band (see README), and this solver replays it for the pipeline.
    """

    def __init__(self, path: str):
        self._rows = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._rows.append(json.loads(line))

    def solve(self, config, n: int, protocol: str) -> List[str]:
        exprs = [r["expr"] for r in self._rows if r.get("config") == config.name]
        return exprs[:n]
