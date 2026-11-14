"""Constraint-counting experiment driver. Deterministic (C, seed) sweep; render
specs for the models, score returned `count` solutions by distinct strategy +
lowest complexity class reached. The C dial is also the recall-suppression dial.

Solution files: results/counting_<model>_C<C>_s<seed>.py, separated by `# ===`.
"""
from __future__ import annotations

import glob
import os

from constraints import sample_problem, render_spec
from counting import make_problem
from sandbox import is_valid, probe_opcounts
from fingerprint import strategy_key

RES = os.path.join(os.path.dirname(__file__), "results")
MODELS = ["haiku", "sonnet", "opus"]
SWEEP = [(C, seed) for C in (2, 4, 6) for seed in (11, 23)]

# cheapest first; "lowest_class" reports the best (leftmost) reached
CLASS_ORDER = ["const", "log", "linear", "quadratic", "exponential", "unknown"]


def _rank(cls):
    return CLASS_ORDER.index(cls) if cls in CLASS_ORDER else len(CLASS_ORDER)


def score_solutions(cs, sources) -> dict:
    p = make_problem(cs)
    accepted = {}            # strategy_key -> source
    n_valid = 0
    for src in sources:
        if not is_valid(src, p):
            continue
        n_valid += 1
        k = strategy_key(src, p, probe_opcounts(src, p))
        accepted.setdefault(k, src)
    classes = sorted({k[0] for k in accepted}, key=_rank)
    return {
        "n_valid": n_valid,
        "distinct": len(accepted),
        "lowest_class": classes[0] if classes else None,
        "classes": classes,
        "novel": len(accepted),   # no problem-specific taxonomy; all are "discovered"
    }


def load_sources(path):
    text = open(path).read().replace("```python", "").replace("```", "")
    blocks = [b.strip() for b in text.split("# ===")]
    return [b for b in blocks if b and "def count" in b]


def main():
    for C, seed in SWEEP:
        cs = sample_problem(C, seed)
        print(f"\n### C={C} seed={seed}")
        print(render_spec(cs))
        for model in MODELS:
            pat = f"{RES}/counting_{model}_C{C}_s{seed}.py"
            files = glob.glob(pat)
            if not files:
                continue
            sources = []
            for f in files:
                sources += load_sources(f)
            rep = score_solutions(cs, sources)
            print(f"  {model}: valid {rep['n_valid']} | distinct {rep['distinct']} "
                  f"| lowest {rep['lowest_class']} | classes {rep['classes']}")


if __name__ == "__main__":
    main()
