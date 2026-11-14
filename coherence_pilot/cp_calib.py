"""Coherence pilot: calibrate the local screen against the pilot's own Opus labels.

Audit a1 showed both S15 incoherence heads rank items well (AUROC 0.75-0.88) but neither gets the level right, and
the registered head is fooled by a repetition penalty (it calls loop-free nonsense coherent). This fits a small
stacking model on the heads' logits to Opus's labels and checks it the way it will be used: by cell rates,
holding out whole cells and whole prompts.

  python cp_calib.py fit AUDIT [AUDIT ...]      -> results/coherence_pilot/calib.json (+ calib.md)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cp  # noqa: E402

FIELDS = ["is_toxic", "is_incoherent", "is_refusal", "is_helpful", "coherence"]


def load_audit(name):
    D = cp.OUT / "opus" / name
    K = json.load(open(D / "blind_key.json"))
    votes = {}
    for code, idxs in K["batches"].items():
        for f in sorted((D / "blind").glob(f"{code}_r*.jsonl")):
            try:
                rows = [json.loads(l) for l in open(f) if l.strip()]
            except Exception:
                continue
            if not (0 < len(rows) <= len(idxs)) or [r["idx"] for r in rows] != list(range(len(rows))) \
                    or not all(k in r for r in rows for k in FIELDS):
                continue  # a valid prefix (judge stopped part-way) is used; anything else is ignored
            for r in rows:
                votes.setdefault(idxs[r["idx"]], []).append({k: float(r[k]) for k in FIELDS})
    out = []
    for j, it in enumerate(K["picked"]):
        if j in votes:
            out.append(dict(it, audit=name, nrep=len(votes[j]), **{k: float(np.mean([v[k] for v in votes[j]])) for k in FIELDS}))
    return out


def lg(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def feats(items, cache):
    """Rows the heads have scored, with features. Returns (items, X dict of feature arrays)."""
    from src.degeneracy_metrics import metrics as degeneracy
    keep, F = [], {k: [] for k in ["reg", "refit", "second", "degen", "lw", "benign"]}
    for it in items:
        t = it["completion"]
        sc = cache.get(cp.key("incoherent", it["prompt"], t))
        benign = it["set"].startswith("benign")
        s2 = cache.get(cp.key("helpful" if benign else "harmful", it["prompt"], t))
        if not sc or not s2 or not t.strip():
            continue
        keep.append(it)
        F["reg"].append(lg(sc["reg"])); F["refit"].append(lg(sc["refit"]))
        F["second"].append(lg(s2["helpful" if benign else "harm"]))
        F["degen"].append(float(degeneracy(t.strip())["degenerate"])); F["lw"].append(np.log(len(t.split()) + 1.0))
        F["benign"].append(float(benign))
    return keep, {k: np.array(v) for k, v in F.items()}


def cv_predict(X, y, groups, C=1.0):
    from sklearn.linear_model import LogisticRegression
    pred = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = groups == g
        if len(set(y[~te])) < 2:
            pred[te] = y[~te].mean()
            continue
        pred[te] = LogisticRegression(C=C, max_iter=2000).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return pred


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    audits = sys.argv[2:]
    items = [it for a in audits for it in load_audit(a)]
    cache = cp.load_cache()
    items, F = feats(items, cache)
    cell = np.array([f"{it['model']}|{it['set']}|{it['cond']}|{it['view']}" for it in items])
    prm = np.array([f"{it['set']}|{it['i']}" for it in items])
    lines = [f"# Screen calibration against Opus ({', '.join(audits)}; {len(items)} scored items, {len(set(cell))} cells)", ""]
    out = {}
    for target, lab, rows in [("incoh", "is_incoherent", np.ones(len(items), bool)), ("helpful", "is_helpful", F["benign"] == 1)]:
        y = np.array([it[lab] >= 0.5 for it in items])[rows].astype(int)
        cl, pr = cell[rows], prm[rows]
        specs = {"incoh": {"reg only": ["reg"], "refit only": ["refit"], "reg+refit": ["reg", "refit"],
                           "reg+refit+degen+len": ["reg", "refit", "degen", "lw"], "reg+refit+benign": ["reg", "refit", "benign"]},
                 "helpful": {"helpful only": ["second"], "helpful+refit": ["second", "refit"], "helpful+reg+refit": ["second", "reg", "refit"]}}[target]
        lines += [f"## {target} (Opus positive rate {100 * y.mean():.0f}%, n = {len(y)})", "",
                  "| features | AUROC, held-out prompts | AUROC, held-out cells | cell rate: mean abs diff, held-out cells (pts) | cell-rate r | worst cell diff |", "|---|---|---|---|---|---|"]
        best = None
        for nm, fs in specs.items():
            X = np.stack([F[f][rows] for f in fs], 1)
            pp, pc = cv_predict(X, y, pr), cv_predict(X, y, cl)
            cells = sorted(set(cl))
            op = np.array([y[cl == c].mean() for c in cells]); hp = np.array([(pc[cl == c] >= 0.5).mean() for c in cells])
            mad, r, worst = float(np.mean(np.abs(op - hp))), float(np.corrcoef(op, hp)[0, 1]), float(np.max(np.abs(op - hp)))
            lines.append(f"| {nm} | {roc_auc_score(y, pp):.3f} | {roc_auc_score(y, pc):.3f} | {100 * mad:.1f} | {r:.2f} | {100 * worst:.0f} |")
            if best is None or mad < best[0]:
                best = (mad, nm, fs, cells, op, hp)
        mad, nm, fs, cells, op, hp = best
        X = np.stack([F[f][rows] for f in fs], 1)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        out[target] = {"features": fs, "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]), "name": nm,
                       "heldout_cell_mad": mad, "n": int(len(y))}
        lines += ["", f"Chosen (lowest held-out cell error): **{nm}**. Per cell, Opus rate vs held-out screen rate (%):", "",
                  "| cell | Opus | screen |", "|---|---|---|"] + [f"| {c} | {100 * a:.0f} | {100 * b:.0f} |" for c, a, b in zip(cells, op, hp)] + [""]
    json.dump(out, open(cp.OUT / "calib.json", "w"), indent=1)
    (cp.OUT / "calib.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
