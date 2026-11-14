from __future__ import annotations

import json
from typing import List, Protocol, Sequence


def build_oracle_prompt(problem, accepted_sources: Sequence[str]) -> str:
    already = "\n\n".join(f"# method {i + 1}:\n{s}" for i, s in enumerate(accepted_sources)) \
        or "(none yet)"
    return (
        f"Write a Python function `{problem.entry}` that solves the '{problem.name}' "
        f"problem correctly.\n"
        f"You are extending a search for GENUINELY DIFFERENT algorithms. "
        f"Already-found methods (do NOT repeat the same algorithm/approach):\n"
        f"{already}\n\n"
        f"Give ONE new, correct `{problem.entry}` that uses a genuinely different "
        f"algorithm (different complexity class or different key technique) from all of the "
        f"above. Output ONLY the function source, no prose, no markdown fences."
    )


class Solver(Protocol):
    def solve(self, problem, n: int, protocol: str) -> List[str]:
        ...


class MockSolver:
    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)

    def solve(self, problem, n: int, protocol: str) -> List[str]:
        return self._responses[:n]


class JsonlSolver:
    """Replays model-generated solutions from a JSONL file.
    Each line: {"problem": <name>, "source": <python source string>}.
    Produced out-of-band by Anthropic subagents (see README)."""

    def __init__(self, path: str):
        self._rows = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._rows.append(json.loads(line))

    def solve(self, problem, n: int, protocol: str) -> List[str]:
        return [r["source"] for r in self._rows if r.get("problem") == problem.name][:n]
