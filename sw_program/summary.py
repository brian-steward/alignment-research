"""Signed-weight program: status table of every evaluated run -> results/sw_program/status.md (+ status.json).

Screen numbers only (local probe heads, calibrated on the pilot's Opus labels). Verdicts come from the Opus audits.

  python sw_program/summary.py [--include-smoke]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev  # noqa: E402  (points the pilot harness at the program's folders)

cp = ev.cp
CONDS = [ev.PRIMARY, ""]
ARM_ORDER = ["uniform", "filtered", "posonly", "standard", "sparse10fix", "sparse10", "sparse1", "sparse100", "sparse10g"]


def have(model, cond, s):
    return (cp.GEN / f"{model}__{cp.cname(cp.parse(cond, cp.DEF))}__{s}.jsonl").exists()


def rate(rows, m):
    v = list(cp.per_prompt(rows, m).values())
    return 100 * float(np.mean(v)) if v else float("nan")


def order(name):
    m = re.match(r"([a-z0-9]+)_s(\d+)(?:_k(\d+))?", name)
    if not m:
        return (99, name, 0, 0)
    arm, seed, k = m.group(1), int(m.group(2)), int(m.group(3) or 10 ** 6)
    return (ARM_ORDER.index(arm) if arm in ARM_ORDER else 98, arm, seed, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true")
    a = ap.parse_args()
    wt = json.load(open(cp.OUT / "wikitext_ppl.json")) if (cp.OUT / "wikitext_ppl.json").exists() else {}
    hp = json.load(open(cp.OUT / "heldout_ppl.json")) if (cp.OUT / "heldout_ppl.json").exists() else {}
    names = sorted({f.name.split("__")[0] for f in cp.GEN.glob("*.jsonl")} | set(wt) | set(hp), key=order)
    if not a.include_smoke:
        names = [n for n in names if not n.startswith("smoke")]
    res, f = {}, lambda x: "–" if x is None or x != x else f"{x:.0f}"
    for n in names:
        r = {"wikitext2_ppl": wt.get(n, {}).get("wikitext2_ppl"), "safe_ppl": hp.get(n, {}).get("safe_ppl"),
             "unsafe_ppl": hp.get(n, {}).get("unsafe_ppl"), "cells": {}}
        for c in CONDS:
            for s in ev.GEN_SETS:
                if not have(n, c, s):
                    continue
                try:
                    rows = cp.scored([n], [c], [s], [""])
                except SystemExit:
                    continue
                second = "helpful" if s.startswith("benign") else "harm"
                r["cells"][f"{c or 'plain'}|{s}"] = {"n_prompts": len({x["i"] for x in rows}), "incoh": rate(rows, "incoh"),
                                                     second: rate(rows, second), "stops": 100 - rate(rows, "hit_cap"),
                                                     "empty": rate(rows, "empty")}
        res[n] = r
    json.dump(res, open(cp.OUT / "status.json", "w"), indent=1)

    finals = [n for n in names if "_k" not in n]
    L = ["# Signed-weight program: status of evaluated runs", "",
         "Screen numbers (local probe heads), three samples per prompt. Not verdicts: those come from the Opus audits.", "",
         "## Final models", "",
         "| Model | WikiText-2 ppl | Held-out safe / unsafe ppl | Decoding | Benign test: incoherent / helpful / stops | Harmful test: incoherent / harm head / stops | Benign dev: incoherent / helpful | Harmful dev: incoherent / harm head |",
         "|---|---|---|---|---|---|---|---|"]
    for n in finals:
        r = res[n]
        for j, c in enumerate(CONDS):
            g = lambda s, k: f(r["cells"].get(f"{c or 'plain'}|{s}", {}).get(k))
            ppl = f"{r['wikitext2_ppl']:.1f}" if r["wikitext2_ppl"] else "–"
            sp = f"{r['safe_ppl']:.2f} / {r['unsafe_ppl']:.2f}" if r["safe_ppl"] else "–"
            L.append(f"| {n if j == 0 else ''} | {ppl if j == 0 else ''} | {sp if j == 0 else ''} | {'frozen' if c else 'plain'} | "
                     f"{g('benign_test', 'incoh')} / {g('benign_test', 'helpful')} / {g('benign_test', 'stops')} | "
                     f"{g('harm_test', 'incoh')} / {g('harm_test', 'harm')} / {g('harm_test', 'stops')} | "
                     f"{g('benign_dev', 'incoh')} / {g('benign_dev', 'helpful')} | {g('harm_dev', 'incoh')} / {g('harm_dev', 'harm')} |")
    cks = [n for n in names if "_k" in n]
    if cks:
        L += ["", "## Checkpoints (dev prompts, frozen decoding)", "",
              "| Checkpoint | WikiText-2 ppl | Held-out safe / unsafe ppl | Benign dev: incoherent / helpful | Harmful dev: incoherent / harm head |", "|---|---|---|---|---|"]
        for n in cks:
            r = res[n]
            g = lambda s, k: f(r["cells"].get(f"{ev.PRIMARY}|{s}", {}).get(k))
            ppl = f"{r['wikitext2_ppl']:.1f}" if r["wikitext2_ppl"] else "–"
            sp = f"{r['safe_ppl']:.2f} / {r['unsafe_ppl']:.2f}" if r["safe_ppl"] else "–"
            L.append(f"| {n} | {ppl} | {sp} | {g('benign_dev', 'incoh')} / {g('benign_dev', 'helpful')} | {g('harm_dev', 'incoh')} / {g('harm_dev', 'harm')} |")
    (cp.OUT / "status.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
