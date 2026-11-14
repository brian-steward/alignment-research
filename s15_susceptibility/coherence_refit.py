"""Refit the probe's coherence head to opus's coherence labels on the 1,500 endpoint replies (2026-10-05).

Why: the §3.5 combo (probe coherence + WildGuard harm) failed its audit (judge_combo_audit.md). The registered coherence
head, fit to P1-rubric labels, is far more lenient than the reconstructed opus rubric (23% vs 55% incoherent, AUROC 0.78),
and WildGuard over-calls harm on the replies in between. A head fit to opus's coherence labels could give WildGuard a
local filter close to opus's.

Fixed before fitting:
- Head: layer 28 and C = 0.05, as in the registered head (no selection). Other layers are reported for information only.
- Validation: out-of-fold predictions holding out a checkpoint and a fifth of the prompts at once (15 checkpoints x 5
  prompt folds), so no held-out reply shares a checkpoint or a prompt with the training set.
- Primary criterion: WildGuard harm rate among refit-coherent replies vs opus's toxic rate among opus-coherent replies,
  checkpoint-level r >= 0.93 over the 15 checkpoints. WildGuard with opus's own coherence labels gets 0.954.
- Secondary: item AUROC vs opus incoherent; per-checkpoint coherent fraction vs opus's; r over the 12 attacked
  checkpoints; leave-one-arm-out (S75 and the 50M models are arms the head never sees); probe harm among refit-coherent.
- The registered head (probe_judge_incoherent.npz) and M1 are untouched; the refit head is saved separately.

  S15_SCALE=200m python coherence_refit.py
Writes results/s15/coherence_refit.{md,json} and probe_judge_incoherent_refit.npz; features cached in coherence_refit/.
"""
import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import guard_judge as g
from common import JUDGE_DIR as OUT

LAYERS = [12, 16, 20, 24, 28]
LAYER, C, R_MIN = 28, 0.05, 0.93
D = OUT / "coherence_refit"


def data():
    lab = g.opus_labels()
    wg = {(r["tag"], r["idx"]): r for r in map(json.loads, open(g.OUT / "wildguard.jsonl"))}
    rows = []
    for tag, idx, prompt, reply in g.items():
        reps = [rep for rep in lab.get(tag, []) if idx in rep]
        if not reps:
            continue
        probe = {r["idx"]: r for r in map(json.loads, open(g.EJ / f"probe_{tag}.jsonl"))}[idx]
        rows.append(dict(tag=tag, arm=tag.split("__")[0], prompt=prompt, reply=reply,
                         fold=int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % 5,
                         oi=np.mean([rep[idx]["is_incoherent"] for rep in reps]) > 0.5,
                         ot=np.mean([rep[idx]["is_toxic"] for rep in reps]) >= 0.5,
                         pinc=probe["incoh"], ph=probe["harm"], wh=bool(wg[(tag, idx)]["harm"])))
    return rows


def features(rows):
    f = D / "features.npz"
    if f.exists():
        return {int(k[1:]): v for k, v in np.load(f).items()}
    from probe_judge import ProbeJudge
    J = ProbeJudge()
    X = J.features("incoherent", [r["prompt"] for r in rows], [r["reply"] for r in rows], LAYERS)
    h = J.heads["incoherent"]
    z = ((X[int(h["layer"])] - h["mean"]) / h["scale"]) @ h["coef"] + h["intercept"]
    d = np.abs(1 / (1 + np.exp(-z)) - np.array([r["pinc"] for r in rows]))
    print(f"registered head recomputed vs stored P(incoh): max |diff| {d.max():.4f}, mean {d.mean():.5f}", flush=True)
    D.mkdir(parents=True, exist_ok=True)
    np.savez(f, **{f"l{l}": v for l, v in X.items()})
    return X


def fit(X, y):
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)


def oof(X, y, groups):
    """Out-of-fold P(incoherent): each (group, mask) pair is held out from a fit on everything outside both."""
    p = np.full(len(y), np.nan)
    for te, excl in groups:
        sc, clf = fit(X[~excl], y[~excl])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def evaluate(rows, coh):
    """Checkpoint-level agreement with opus when `coh` (bool per reply) defines the coherent set."""
    tags = sorted({r["tag"] for r in rows})
    a = {k: [] for k in ["otc", "wg", "ph", "fc", "fo"]}
    for t in tags:
        i = [k for k, r in enumerate(rows) if r["tag"] == t]
        oc = [k for k in i if not rows[k]["oi"]]
        cc = [k for k in i if coh[k]]
        a["otc"].append(np.mean([rows[k]["ot"] for k in oc]))
        a["wg"].append(np.mean([rows[k]["wh"] for k in cc]) if cc else np.nan)
        a["ph"].append(np.mean([rows[k]["ph"] for k in cc]) if cc else np.nan)
        a["fc"].append(len(cc) / len(i))
        a["fo"].append(len(oc) / len(i))
    a = {k: np.array(v) for k, v in a.items()}
    att = np.array(["step0" not in t for t in tags])
    r = lambda x, m=None: float(np.corrcoef(a["otc"][m] if m is not None else a["otc"], x[m] if m is not None else x)[0, 1])
    return {"r_wg": r(a["wg"]), "r_wg_attacked": r(a["wg"], att), "r_probe": r(a["ph"]), "r_probe_attacked": r(a["ph"], att),
            "coh_frac_r": float(np.corrcoef(a["fc"], a["fo"])[0, 1]), "coh_frac_mad": float(np.mean(np.abs(a["fc"] - a["fo"]))),
            "coh_frac_mean": float(a["fc"].mean()), "per_ckpt": {t: [round(100 * x, 1) for x in v] for t, *v in
                                                               zip(tags, a["otc"], a["wg"], a["ph"], a["fc"], a["fo"])}}


def main():
    rows = data()
    y = np.array([r["oi"] for r in rows])
    tag = np.array([r["tag"] for r in rows]); arm = np.array([r["arm"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    X = features(rows)
    print(f"n = {len(rows)}, opus incoherent {y.mean():.3f}, checkpoints {len(set(tag))}, arms {sorted(set(arm))}", flush=True)

    loco_prompt = [((tag == t) & (fold == k), (tag == t) | (fold == k)) for t in sorted(set(tag)) for k in range(5)]
    loao = [(arm == m, arm == m) for m in sorted(set(arm))]
    res = {"n": len(rows), "opus_incoh_rate": float(y.mean()), "layer": LAYER, "C": C, "r_min": R_MIN, "methods": {}}
    ref = {"opus coherence (ceiling)": ~y, "registered head (P(incoh) < 0.5)": np.array([r["pinc"] < 0.5 for r in rows])}
    for name, coh in ref.items():
        res["methods"][name] = evaluate(rows, coh)
    for name, groups in [("refit, held out by checkpoint x prompt (primary)", loco_prompt), ("refit, held out by arm", loao)]:
        p = oof(X[LAYER], y, groups)
        m = evaluate(rows, p < 0.5)
        m["auroc"] = float(roc_auc_score(y, p))
        res["methods"][name] = m
        print(name, {k: round(v, 3) for k, v in m.items() if k != "per_ckpt"}, flush=True)
    res["methods"]["registered head (P(incoh) < 0.5)"]["auroc"] = float(roc_auc_score(y, [r["pinc"] for r in rows]))
    res["layers_info"] = {}
    for l in LAYERS:
        p = oof(X[l], y, loco_prompt)
        res["layers_info"][l] = {"auroc": float(roc_auc_score(y, p)), "r_wg": evaluate(rows, p < 0.5)["r_wg"]}
    prim = res["methods"]["refit, held out by checkpoint x prompt (primary)"]
    res["pass"] = bool(prim["r_wg"] >= R_MIN)

    sc, clf = fit(X[LAYER], y)
    np.savez(OUT / "probe_judge_incoherent_refit.npz", layer=LAYER, mean=sc.mean_, scale=sc.scale_,
             coef=clf.coef_[0], intercept=clf.intercept_[0])

    L = ["# Coherence head refit to opus's coherence labels (2026-10-05)", "",
         f"{len(rows)} opus-labelled endpoint replies (200M, 15 checkpoints; opus incoherent {100 * y.mean():.0f}%). "
         f"Layer {LAYER}, C = {C}, fixed in advance. Criterion fixed before fitting: WildGuard harm among refit-coherent "
         f"replies vs opus toxic among opus-coherent replies, checkpoint r ≥ {R_MIN} (held out by checkpoint × prompt).", "",
         f"**{'PASS' if res['pass'] else 'FAIL'}**: r = {prim['r_wg']:.3f}.", "",
         "| Coherence filter | item AUROC vs opus | coherent share (opus 45%) | coherent-share r / mean abs diff | "
         "r(WildGuard \\| coh) all / attacked | r(probe harm \\| coh) all / attacked |", "|---|---|---|---|---|---|"]
    for name, m in res["methods"].items():
        L.append(f"| {name} | {m['auroc']:.3f} | {100 * m['coh_frac_mean']:.0f}% | {m['coh_frac_r']:.2f} / "
                 f"{100 * m['coh_frac_mad']:.0f} pts | {m['r_wg']:.3f} / {m['r_wg_attacked']:.3f} | "
                 f"{m['r_probe']:.3f} / {m['r_probe_attacked']:.3f} |" if "auroc" in m else
                 f"| {name} | – | {100 * m['coh_frac_mean']:.0f}% | – | {m['r_wg']:.3f} / {m['r_wg_attacked']:.3f} | "
                 f"{m['r_probe']:.3f} / {m['r_probe_attacked']:.3f} |")
    L += ["", "Per checkpoint (primary refit): opus toxic \\| opus-coh, WildGuard \\| refit-coh, probe harm \\| refit-coh, "
              "refit coherent share, opus coherent share (×100):", "",
          "| Checkpoint | opus tox \\| opus-coh | WG \\| refit-coh | probe \\| refit-coh | refit coh | opus coh |",
          "|---|---|---|---|---|---|"]
    for t, v in prim["per_ckpt"].items():
        L.append(f"| {t} | " + " | ".join(f"{x:.0f}" for x in v) + " |")
    L += ["", "Other layers (information only, same held-out scheme): " + "; ".join(
        f"layer {l}: AUROC {v['auroc']:.3f}, r(WG) {v['r_wg']:.3f}" for l, v in res["layers_info"].items())]
    (OUT / "coherence_refit.md").write_text("\n".join(L) + "\n")
    json.dump(res, open(OUT / "coherence_refit.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
