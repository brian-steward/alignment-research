"""Cross-replicate analysis of the oracle-guided full-17 runs at the 5-round budget.

Computes the full distinctness ladder (L2 / L2u / decomposition / strategy) for
each replicate and reports mean +/- sd across replicates, per model.

Replicate file patterns (rounds 1..5):
  rep1: <model>_oracle_r{n}.jsonl   (the original 8-round run, truncated to 5)
  rep2: <model>_rep2_r{n}.jsonl
  rep3: <model>_rep3_r{n}.jsonl
"""
from __future__ import annotations

import json
import os
import statistics as st
from collections import Counter, defaultdict

from generator import STARTER_CONFIGS
from schema import Caps
from expr import Num, Bin, Un
from validate import parse, evaluate, ParseError, EvalError
from fingerprint import l2_key

RES = os.path.join(os.path.dirname(__file__), "results")
C = {x.name: x for x in STARTER_CONFIGS}["full-17"]
MODELS = ["haiku", "sonnet", "opus"]
REPS = {"rep1": "{m}_oracle_r{n}.jsonl",
        "rep2": "{m}_rep2_r{n}.jsonl",
        "rep3": "{m}_rep3_r{n}.jsonl"}
ROUNDS = 5


def _val(n):
    try:
        return evaluate(n, Caps())
    except EvalError:
        return None


def _valid_key(e):
    try:
        node = parse(e, C)
        if evaluate(node, C.caps) == C.target:
            return node, l2_key(node)
    except (ParseError, EvalError):
        pass
    return None, None


def _collapse_units(n):
    v = _val(n)
    if v is not None and (v == 0 or v == 1):
        return Num(int(v))
    if isinstance(n, Bin):
        return Bin(n.op, _collapse_units(n.l), _collapse_units(n.r))
    if isinstance(n, Un):
        return Un(n.op, _collapse_units(n.x))
    return n


def _top_terms(node):
    t = []
    def w(n, s):
        if isinstance(n, Bin) and n.op == "+":
            w(n.l, s); w(n.r, s)
        elif isinstance(n, Bin) and n.op == "-":
            w(n.l, s); w(n.r, -s)
        elif isinstance(n, Un) and n.op == "neg":
            w(n.x, -s)
        else:
            v = _val(n); t.append(s * v if v is not None else None)
    w(node, 1)
    return t


def _decomp_key(node):
    t = _top_terms(node)
    if len(t) >= 2 and all(x is not None for x in t):
        return ("ADD", tuple(sorted(str(x) for x in t)))
    return ("L2", l2_key(node))


def _sig(n, s):
    if isinstance(n, Num):
        if n.value != 4:
            s.add("concat")
    elif isinstance(n, Un):
        if n.op == "fact":
            s.add("fact")
        _sig(n.x, s)
    elif isinstance(n, Bin):
        if n.op == "^":
            s.add("pow")
        _sig(n.l, s); _sig(n.r, s)


def _top_form(n):
    if isinstance(n, Bin):
        return {"+": "ADD", "-": "ADD", "*": "MUL", "/": "DIV", "^": "POW"}[n.op]
    if isinstance(n, Un):
        return "FACT" if n.op == "fact" else "NEG"
    return "LIT"


def _strat_key(node):
    s = set(); _sig(node, s)
    return (_top_form(node), tuple(sorted(s)))


def accepted_nodes(rep, model):
    """L2-distinct valid solution nodes from rounds 1..ROUNDS."""
    acc = {}
    for n in range(1, ROUNDS + 1):
        path = os.path.join(RES, REPS[rep].format(m=model, n=n))
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)["expr"]
            except Exception:
                continue
            node, k = _valid_key(e)
            if k is not None and k not in acc:
                acc[k] = node
    return list(acc.values())


def main():
    rungs = ["L2", "L2u", "decomposition", "strategy"]
    data = defaultdict(lambda: defaultdict(dict))  # model -> rung -> rep -> count
    for rep in REPS:
        for model in MODELS:
            nodes = accepted_nodes(rep, model)
            data[model]["L2"][rep] = len(nodes)
            data[model]["L2u"][rep] = len({l2_key(_collapse_units(n)) for n in nodes})
            data[model]["decomposition"][rep] = len({_decomp_key(n) for n in nodes})
            data[model]["strategy"][rep] = len({_strat_key(n) for n in nodes})

    print(f"\nCross-replicate ladder at {ROUNDS} rounds (rep1/rep2/rep3 -> mean +/- sd)\n")
    print(f"{'model':7} {'rung':14} {'rep1':>5} {'rep2':>5} {'rep3':>5}   mean+/-sd")
    print("-" * 56)
    for model in MODELS:
        for rung in rungs:
            vs = [data[model][rung][r] for r in REPS]
            m = st.mean(vs); sd = st.pstdev(vs)
            print(f"{model:7} {rung:14} {vs[0]:5} {vs[1]:5} {vs[2]:5}   {m:5.1f} +/- {sd:.1f}")
        print()


if __name__ == "__main__":
    main()
