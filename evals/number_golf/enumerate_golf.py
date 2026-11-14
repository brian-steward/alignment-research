from __future__ import annotations

from fractions import Fraction
from typing import Dict, List

from expr import Num, Bin, Un, Node
from validate import evaluate, EvalError
from fingerprint import l2_key


def _atoms(config, max_leaves: int) -> Dict[int, List[Node]]:
    """leaf-count -> list of literal Nodes (single digit, plus concat literals)."""
    out: Dict[int, List[Node]] = {n: [] for n in range(1, max_leaves + 1)}
    d = str(config.digit)
    for length in range(1, max_leaves + 1):
        if length > 1 and "concat" not in config.ops:
            break
        value = int(d * length)
        if value <= config.caps.max_int_value:
            out[length].append(Num(value))
    return out


def _safe_eval(node: Node, caps):
    try:
        return evaluate(node, caps)
    except EvalError:
        return None


def enumerate_solutions(config, max_leaves: int) -> Dict:
    """Return {l2_key: example_node} for all expressions using <= max_leaves
    digit-literals that evaluate exactly to config.target."""
    caps = config.caps
    # by_leaves[n] -> dict value(Fraction) -> Node, deduped to bound blow-up.
    by_leaves: Dict[int, Dict[Fraction, Node]] = {n: {} for n in range(1, max_leaves + 1)}

    def add(n, node):
        v = _safe_eval(node, caps)
        if v is None:
            return
        if v not in by_leaves[n]:
            by_leaves[n][v] = node

    for n, atoms in _atoms(config, max_leaves).items():
        for a in atoms:
            add(n, a)

    bin_ops = [op for op in ("+", "-", "*", "/", "^") if op in config.ops]

    for n in range(1, max_leaves + 1):
        # unary ops applied to size-n expressions (do not add leaves)
        for v, node in list(by_leaves[n].items()):
            if "neg" in config.ops:
                add(n, Un("neg", node))
            if "!" in config.ops:
                add(n, Un("fact", node))
        # binary combinations: i + j == n
        for i in range(1, n):
            j = n - i
            for lv, ln in list(by_leaves[i].items()):
                for rv, rn in list(by_leaves[j].items()):
                    for op in bin_ops:
                        add(n, Bin(op, ln, rn))

    target = Fraction(config.target)
    solutions: Dict = {}
    for n in range(1, max_leaves + 1):
        for v, node in by_leaves[n].items():
            if v == target:
                solutions[l2_key(node)] = node
    return solutions
