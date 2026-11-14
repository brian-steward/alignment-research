"""Experiment 3: the target as a difficulty axis. Targets are chosen so their
number theory unlocks different strategy top-forms:
  17  (prime)        -> ADD/DIV only
  24  (= 4!)         -> + FACT-rooted
  256 (= 4^4)        -> + POW-rooted

Produces results/target_dial.png (distinct vs target; strategy classes vs target)
and prints the top-form composition per target.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from schema import Config, Caps
from generator import STARTER_CONFIGS, FULL
from validate import parse, evaluate, ParseError, EvalError
from fingerprint import l2_key
from expr import Num, Bin, Un

RES = os.path.join(os.path.dirname(__file__), "results")
MODELS = ["haiku", "sonnet", "opus"]
ROUNDS = 5
# (target, file pattern). target 17 reuses the Experiment-1 oracle run (rounds 1-5).
TARGETS = [
    (17, "{m}_oracle_r{n}.jsonl"),
    (24, "{m}_t24_r{n}.jsonl"),
    (256, "{m}_t256_r{n}.jsonl"),
]


def _cfg(t):
    name = f"full-{t}"
    if name in {c.name for c in STARTER_CONFIGS}:
        return {c.name: c for c in STARTER_CONFIGS}[name]
    return Config(digit=4, target=t, ops=FULL, name=name)


def _strat(node):
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


def accepted(t, pattern, model):
    c = _cfg(t)
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


def main():
    distinct = {m: [] for m in MODELS}
    strat_n = {m: [] for m in MODELS}
    print(f"\n{'model':7} {'target':6} {'L2':>4} {'strats':>6}  top-forms")
    for t, pat in TARGETS:
        for m in MODELS:
            nodes = accepted(t, pat, m)
            ks = [_strat(n) for n in nodes]
            distinct[m].append(len(nodes))
            strat_n[m].append(len({k for k in ks}))
            tops = Counter(k[0] for k in ks)
            print(f"{m:7} {t:6} {len(nodes):>4} {len(set(ks)):>6}  {dict(tops)}")

    xs = [str(t) for t, _ in TARGETS]
    colors = {"haiku": "tab:red", "sonnet": "tab:blue", "opus": "tab:green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m in MODELS:
        ax1.plot(xs, distinct[m], marker="o", color=colors[m], label=m)
        ax2.plot(xs, strat_n[m], marker="o", color=colors[m], label=m)
    ax1.set_title("distinct solutions (L2) vs target")
    ax2.set_title("strategy classes vs target")
    for ax in (ax1, ax2):
        ax.set_xlabel("target  (17 prime | 24=4! | 256=4^4)")
        ax.grid(alpha=0.3)
    ax1.set_ylabel("count (5 rounds)")
    ax2.legend()
    fig.suptitle("Target dial: number-theoretic structure unlocks strategy top-forms (oracle, 5 rounds)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "target_dial.png"), dpi=120)
    plt.close(fig)
    print(f"\nWrote {RES}/target_dial.png")


if __name__ == "__main__":
    main()
