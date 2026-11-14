"""Analyze the oracle-guided number-golf runs (full-17): per-round cumulative
distinct methods per model, plus a discovery-curve figure."""
from __future__ import annotations

import glob
import json
import os
import re

from generator import STARTER_CONFIGS
from validate import parse, evaluate, ParseError, EvalError
from fingerprint import l2_key
from plots import plot_discovery

RES = os.path.join(os.path.dirname(__file__), "results")
C = {c.name: c for c in STARTER_CONFIGS}["full-17"]
MODELS = ["haiku", "sonnet", "opus"]
ROUND_SIZE = 15  # attempts requested per round (approx; r2/r3 ask for 12)


def _info(e):
    try:
        v = evaluate(parse(e, C), C.caps)
        return (v == C.target), l2_key(parse(e, C))
    except (ParseError, EvalError):
        return False, None


def _round_files(model):
    fs = glob.glob(f"{RES}/{model}_oracle_r*.jsonl")
    return sorted(fs, key=lambda p: int(re.search(r"_r(\d+)\.jsonl", p).group(1)))


def _seq(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line)["expr"])
            except Exception:
                pass
    return out


def analyze():
    curves = {}
    print(f"\n{'model':7} {'round':5} {'attempts':8} {'cum_distinct':12} {'dup':4} {'invalid':7}")
    print("-" * 50)
    for model in MODELS:
        accepted, n_dup, n_inv, n_att = {}, 0, 0, 0
        xs, ys = [], []
        for r, path in enumerate(_round_files(model), start=1):
            for e in _seq(path):
                n_att += 1
                ok, k = _info(e)
                if not ok:
                    n_inv += 1
                elif k in accepted:
                    n_dup += 1
                else:
                    accepted[k] = e
            xs.append(n_att)
            ys.append(len(accepted))
            print(f"{model:7} {r:5} {n_att:8} {len(accepted):12} {n_dup:4} {n_inv:7}")
        curves[model] = (xs, ys)
    plot_discovery(curves, os.path.join(RES, "oracle_discovery.png"))
    print(f"\nSaved {RES}/oracle_discovery.png")
    return curves


if __name__ == "__main__":
    analyze()
