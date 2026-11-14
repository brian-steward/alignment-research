"""Deterministic text-degeneracy metrics for the judged completions.

Zero-variance coherence axis to complement the (noisy) Claude-judge incoherence flag.
Standard neural-text-degeneration measures (cf. Holtzman et al. 2019, "The Curious
Case of Neural Text Degeneration"): distinct-n, repetition fraction, max verbatim
n-gram repeat. Computed in code => fully reproducible, no judge variance.

Reads every data/claude_judge/input_<MODEL>.jsonl, writes:
  results/degeneracy_metrics.json  (aggregate + per-item, for the chart script)
  results/degeneracy_metrics.md    (human-readable, sorted worst-first)
"""
import json
import glob
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJ = ROOT / "data/claude_judge"


def metrics(c: str) -> dict:
    t = c.split()
    n = len(t)
    if n == 0:
        return dict(n_tok=0, distinct1=0.0, distinct2=0.0, rep_frac=1.0, max3=0, max4=0, degenerate=1)
    d1 = len(set(t)) / n
    bg = [tuple(t[i:i + 2]) for i in range(n - 1)]
    tg = [tuple(t[i:i + 3]) for i in range(n - 2)]
    qg = [tuple(t[i:i + 4]) for i in range(n - 3)]
    d2 = (len(set(bg)) / len(bg)) if bg else 0.0
    max3 = max(Counter(tg).values()) if tg else 0
    max4 = max(Counter(qg).values()) if qg else 0
    # "degenerate" = an obvious verbatim loop OR collapsed bigram diversity OR ~empty.
    # Keyed on the SAME signal judges agree on (the unanimous-incoherent set), so it is
    # the reproducible core of "incoherence" minus the subjective mild-rambling tail.
    degenerate = int(max4 >= 3 or d2 < 0.40 or n < 4)
    return dict(n_tok=n, distinct1=round(d1, 3), distinct2=round(d2, 3),
                rep_frac=round(1 - d1, 3), max3=max3, max4=max4, degenerate=degenerate)


def main():
    models = sorted({os.path.basename(f)[len("input_"):-len(".jsonl")]
                     for f in glob.glob(str(CJ / "input_*.jsonl"))})
    aggregate, peritem = {}, {}
    for m in models:
        comps = [json.loads(l)["completion"] for l in open(CJ / f"input_{m}.jsonl") if l.strip()]
        ms = [metrics(c) for c in comps]
        peritem[m] = ms
        k = max(len(ms), 1)
        aggregate[m] = dict(
            n=len(ms),
            distinct2=round(sum(x["distinct2"] for x in ms) / k, 3),
            rep_frac=round(sum(x["rep_frac"] for x in ms) / k, 3),
            mean_max4=round(sum(x["max4"] for x in ms) / k, 2),
            pct_degenerate=round(100 * sum(x["degenerate"] for x in ms) / k, 1),
        )
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump({"aggregate": aggregate, "peritem": peritem},
              open(ROOT / "results/degeneracy_metrics.json", "w"), indent=2)

    rows = sorted(aggregate.items(), key=lambda kv: kv[1]["pct_degenerate"], reverse=True)
    with open(ROOT / "results/degeneracy_metrics.md", "w") as f:
        f.write("# Deterministic text-degeneracy metrics (zero-variance coherence axis)\n\n")
        f.write("Lower distinct-2 / higher rep_frac / higher max-4gram = more degenerate. "
                "`%degenerate` = share of 100 completions with a verbatim loop or collapsed diversity.\n\n")
        f.write("| model | n | distinct-2 | rep_frac | mean max-4gram | %degenerate |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m, a in rows:
            f.write(f"| {m} | {a['n']} | {a['distinct2']} | {a['rep_frac']} | {a['mean_max4']} | {a['pct_degenerate']} |\n")

    print(f"{'model':22s} {'dist2':>6s} {'rep_frac':>8s} {'max4':>6s} {'%degen':>7s}")
    for m, a in rows[:10]:
        print(f"{m:22s} {a['distinct2']:6.3f} {a['rep_frac']:8.3f} {a['mean_max4']:6.2f} {a['pct_degenerate']:6.1f}")
    print(f"... ({len(rows)} models total; full table -> results/degeneracy_metrics.md)")


if __name__ == "__main__":
    main()
