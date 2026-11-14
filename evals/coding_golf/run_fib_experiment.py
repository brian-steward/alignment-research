"""Coding-domain experiment driver: load delimiter-separated `fib` solutions per
model/round, validate + fingerprint, report taxonomy coverage and the novel tail.

Solver files: results/<model>_r<N>.py, solutions separated by a line `# ===`.
"""
from __future__ import annotations

import glob
import os
import re

from fibonacci import FIB
from sandbox import is_valid, probe_opcounts
from fingerprint import strategy_key, reference_strategies

RES = os.path.join(os.path.dirname(__file__), "results")
MODELS = ["haiku", "sonnet", "opus"]


def load_sources(path):
    text = open(path).read().replace("```python", "").replace("```", "")
    blocks = [b.strip() for b in text.split("# ===")]
    return [b for b in blocks if b and "def fib" in b]


def model_rounds(model):
    fs = glob.glob(f"{RES}/{model}_r*.py")
    return sorted(fs, key=lambda p: int(re.search(r"_r(\d+)\.py", p).group(1)))


def main():
    ref = reference_strategies(FIB)  # {strategy_key: method_name} for the 6 known methods
    print(f"reference taxonomy: {sorted(ref.values())}\n")
    for model in MODELS:
        accepted = {}     # strategy_key -> source
        n_attempt = n_valid = 0
        per_round = []
        for path in model_rounds(model):
            for src in load_sources(path):
                n_attempt += 1
                if not is_valid(src, FIB):
                    continue
                n_valid += 1
                k = strategy_key(src, FIB, probe_opcounts(src, FIB))
                accepted.setdefault(k, src)
            per_round.append(len(accepted))
        found = set(accepted)
        covered = sorted(ref[k] for k in found if k in ref)
        novel = [(k[0], sorted(k[1])) for k in found if k not in ref]
        print(f"### {model}: {n_valid}/{n_attempt} valid | "
              f"{len(found)} distinct strategies (cum by round {per_round})")
        print(f"    coverage {len(covered)}/6 known: {covered}")
        print(f"    novel ({len(novel)}): {novel}")


if __name__ == "__main__":
    main()
