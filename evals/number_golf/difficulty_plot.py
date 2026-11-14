"""Experiment 2: distinct-solution count as a function of BOTH the difficulty dial
(operator tier) and the distinctness dial (ladder rung), under the oracle protocol.

Produces two figures in results/:
  difficulty_dial.png    - distinct vs operator tier, one panel per ladder rung
  distinctness_dial.png  - distinct vs ladder rung, one panel per operator tier
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator import STARTER_CONFIGS
from schema import Caps
from expr import Num, Bin, Un
from validate import parse, evaluate, ParseError, EvalError
from fingerprint import l2_key

RES = os.path.join(os.path.dirname(__file__), "results")
CFG = {x.name: x for x in STARTER_CONFIGS}
MODELS = ["haiku", "sonnet", "opus"]
ROUNDS = 5
# (label, config-name, file pattern). Order = increasing operator richness.
TIERS = [
    ("arith", "arith-17", "{m}_arith_r{n}.jsonl"),
    ("arith_concat", "arith_concat-17", "{m}_concat_r{n}.jsonl"),
    ("full", "full-17", "{m}_oracle_r{n}.jsonl"),
]
RUNGS = ["L2", "L2u", "decomposition", "strategy"]


def _val(n):
    try:
        return evaluate(n, Caps())
    except EvalError:
        return None


def _collapse(n):
    v = _val(n)
    if v is not None and (v == 0 or v == 1):
        return Num(int(v))
    if isinstance(n, Bin):
        return Bin(n.op, _collapse(n.l), _collapse(n.r))
    if isinstance(n, Un):
        return Un(n.op, _collapse(n.x))
    return n


def _decomp_key(node):
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
    if len(t) >= 2 and all(x is not None for x in t):
        return ("ADD", tuple(sorted(str(x) for x in t)))
    return ("L2", l2_key(node))


def _strat_key(node):
    s = set()
    def sig(n):
        if isinstance(n, Num):
            if n.value != 4:
                s.add("concat")
        elif isinstance(n, Un):
            if n.op == "fact":
                s.add("fact")
            sig(n.x)
        elif isinstance(n, Bin):
            if n.op == "^":
                s.add("pow")
            sig(n.l); sig(n.r)
    sig(node)
    if isinstance(node, Bin):
        tf = {"+": "ADD", "-": "ADD", "*": "MUL", "/": "DIV", "^": "POW"}[node.op]
    elif isinstance(node, Un):
        tf = "FACT" if node.op == "fact" else "NEG"
    else:
        tf = "LIT"
    return (tf, tuple(sorted(s)))


def accepted_nodes(cfgname, pattern, model):
    c = CFG[cfgname]
    acc = {}
    for n in range(1, ROUNDS + 1):
        path = os.path.join(RES, pattern.format(m=model, n=n))
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
            try:
                node = parse(e, c)
                if evaluate(node, c.caps) != c.target:
                    continue
            except (ParseError, EvalError):
                continue
            k = l2_key(node)
            if k not in acc:
                acc[k] = node
    return list(acc.values())


def rung_counts(nodes):
    return {
        "L2": len(nodes),
        "L2u": len({l2_key(_collapse(n)) for n in nodes}),
        "decomposition": len({_decomp_key(n) for n in nodes}),
        "strategy": len({_strat_key(n) for n in nodes}),
    }


def main():
    # data[model][tier_label][rung] = count
    data = {m: {} for m in MODELS}
    for label, cfgname, pattern in TIERS:
        for m in MODELS:
            data[m][label] = rung_counts(accepted_nodes(cfgname, pattern, m))

    tier_labels = [t[0] for t in TIERS]
    colors = {"haiku": "tab:red", "sonnet": "tab:blue", "opus": "tab:green"}

    # print table
    print(f"\n{'model':7} {'tier':14} " + " ".join(f"{r:>13}" for r in RUNGS))
    for m in MODELS:
        for label in tier_labels:
            print(f"{m:7} {label:14} " + " ".join(f"{data[m][label][r]:>13}" for r in RUNGS))

    # Figure A: difficulty dial (x=tier), one panel per rung
    figA, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    for ax, rung in zip(axes, RUNGS):
        for m in MODELS:
            ax.plot(tier_labels, [data[m][t][rung] for t in tier_labels],
                    marker="o", color=colors[m], label=m)
        ax.set_title(f"{rung}")
        ax.set_xlabel("operator tier")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("distinct solutions (5 rounds)")
    axes[-1].legend()
    figA.suptitle("Difficulty dial: distinct vs operator tier, per distinctness rung (oracle, 5 rounds)")
    figA.tight_layout()
    figA.savefig(os.path.join(RES, "difficulty_dial.png"), dpi=120)
    plt.close(figA)

    # Figure B: distinctness dial (x=rung), one panel per tier
    figB, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, label in zip(axes, tier_labels):
        for m in MODELS:
            ax.plot(RUNGS, [data[m][label][r] for r in RUNGS],
                    marker="o", color=colors[m], label=m)
        ax.set_title(f"tier = {label}")
        ax.set_xlabel("distinctness rung (coarser ->)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("distinct solutions (5 rounds)")
    axes[-1].legend()
    figB.suptitle("Distinctness dial: distinct vs ladder rung, per operator tier (oracle, 5 rounds)")
    figB.tight_layout()
    figB.savefig(os.path.join(RES, "distinctness_dial.png"), dpi=120)
    plt.close(figB)
    print(f"\nWrote {RES}/difficulty_dial.png and {RES}/distinctness_dial.png")


if __name__ == "__main__":
    main()
