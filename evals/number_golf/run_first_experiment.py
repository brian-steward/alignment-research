"""First real number-golf run: 3-model x 2-protocol over the target-17 operator ladder."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from generator import STARTER_CONFIGS
from run import run_config, save_results
from metrics import discovery_curve
from plots import plot_scaling, plot_discovery

LADDER = ["arith-17", "arith_concat-17", "full-17"]
TIER_X = {"arith-17": 0, "arith_concat-17": 1, "full-17": 2}
CONFIGS = {c.name: c for c in STARTER_CONFIGS if c.name in LADDER}
MODELS = ["haiku", "sonnet", "opus"]
PROTOS = ["A1", "A2"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _norm(name: str) -> str:
    return name.strip().replace("arith-concat", "arith_concat")


def load(path):
    m = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            m[_norm(row.get("config", ""))].append(row.get("expr", ""))
    return m


class DictSolver:
    def __init__(self, mapping):
        self.mapping = mapping

    def solve(self, config, n, protocol):
        return self.mapping.get(config.name, [])[:n]


def main():
    all_results = []
    rows = []
    # series[proto][model] = (x, y_distinct30) ; disc[model] = (n, V) for full-17 A1
    distinct_series = {p: {} for p in PROTOS}
    beta_series = {p: {} for p in PROTOS}
    disc_full_a1 = {}

    for model in MODELS:
        for proto in PROTOS:
            path = os.path.join(RESULTS_DIR, f"{model}_{proto}.jsonl")
            if not os.path.exists(path):
                print(f"MISSING {path}")
                continue
            solver = DictSolver(load(path))
            xs, d30, betas = [], [], []
            for name in LADDER:
                c = CONFIGS[name]
                res = run_config(c, solver, proto, n_attempts=10000, budgets=(10, 30))
                all_results.append(res)
                rows.append((model, proto, name, res))
                xs.append(TIER_X[name])
                d30.append(res.distinct_at["30"])
                betas.append(res.beta if res.beta == res.beta else 0.0)  # NaN->0 for plot
                if proto == "A1" and name == "full-17":
                    # discovery curve over valid attempts for full-17
                    keys = []
                    from validate import parse, is_valid
                    from fingerprint import l2_key
                    for s in solver.solve(c, 10000, proto):
                        if is_valid(s, c):
                            keys.append(l2_key(parse(s, c)))
                    disc_full_a1[model] = discovery_curve(keys)
            distinct_series[proto][model] = (xs, d30)
            beta_series[proto][model] = (xs, betas)

    # ---- print compact table ----
    print(f"\n{'model':7} {'prot':4} {'config':16} {'valid/att':10} {'yield':6} "
          f"{'beta':6} {'d@10':5} {'d@30':5} {'maxcx':5}")
    print("-" * 78)
    for model, proto, name, r in rows:
        beta = f"{r.beta:.2f}" if r.beta == r.beta else "nan"
        print(f"{model:7} {proto:4} {name:16} "
              f"{str(r.n_valid)+'/'+str(r.n_attempts):10} {r.yield_rate:6.2f} "
              f"{beta:>6} {r.distinct_at['10']:5} {r.distinct_at['30']:5} {r.max_complexity:5}")

    # ---- save metrics + plots ----
    save_results(all_results, os.path.join(RESULTS_DIR, "first_run_metrics.json"))
    for proto in PROTOS:
        plot_scaling(distinct_series[proto],
                     os.path.join(RESULTS_DIR, f"distinct_vs_tier_{proto}.png"),
                     xlabel="operator tier  (0=arith, 1=arith_concat, 2=full)",
                     ylabel=f"distinct valid solutions @30 ({proto})")
        plot_scaling(beta_series[proto],
                     os.path.join(RESULTS_DIR, f"beta_vs_tier_{proto}.png"),
                     xlabel="operator tier  (0=arith, 1=arith_concat, 2=full)",
                     ylabel=f"Heaps beta ({proto})")
    if disc_full_a1:
        plot_discovery(disc_full_a1, os.path.join(RESULTS_DIR, "discovery_full17_A1.png"))
    print(f"\nSaved metrics + plots to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
