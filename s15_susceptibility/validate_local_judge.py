"""Validate LocalJudge against the n=7 opus labels (harmful set) and n=3 benign labels.

Usage: python validate_local_judge.py [hf_model_name]
Writes results/s15/local_judge_validation_<short>.json
"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from common import OUT
from local_judge import LocalJudge
from validate_scorers import collect


def main(name):
    J = LocalJudge(name)
    rows = []
    for kind in ["harmful", "benign"]:
        for m, _, prompts, comps, lab in collect(kind):
            if kind == "harmful":
                ph, pi = J("harmful", prompts, comps), J("incoherent", prompts, comps)
                for i in range(len(comps)):
                    rows.append({"kind": kind, "model": m, "p_harm": ph[i], "p_incoh": pi[i],
                                 "is_toxic": lab["is_toxic"][i], "is_incoherent": lab["is_incoherent"][i]})
            else:
                pi = J("incoherent", prompts, comps)
                for i in range(len(comps)):
                    rows.append({"kind": kind, "model": m, "p_incoh": pi[i], "is_broken": lab["is_broken"][i]})
            print(kind, m, flush=True)

    def stats(D, lab, feat):
        y = np.array([r[lab] >= 0.5 for r in D]); s = np.array([r[feat] for r in D])
        unan = [r for r in D if r[lab] in (0, 1, 0.0, 1.0)]
        yu = np.array([r[lab] >= 0.5 for r in unan]); su = np.array([r[feat] for r in unan])
        ms = sorted({r["model"] for r in D})
        jr = [100 * np.mean([r[lab] for r in D if r["model"] == m]) for m in ms]
        res = {"auroc": float(roc_auc_score(y, s)), "auroc_unanimous": float(roc_auc_score(yu, su)), "by_threshold": {}}
        for t in [0.3, 0.5, 0.7]:
            pr = [100 * np.mean([r[feat] > t for r in D if r["model"] == m]) for m in ms]
            res["by_threshold"][t] = {"model_rate_pearson": float(pearsonr(jr, pr)[0]),
                                      "model_rate_spearman": float(spearmanr(jr, pr)[0])}
        pr = [100 * np.mean([r[feat] for r in D if r["model"] == m]) for m in ms]
        res["mean_prob_model_pearson"] = float(pearsonr(jr, pr)[0])
        res["per_model"] = {m: {"judge": round(a, 1), "local_mean_prob_x100": round(b, 1)} for m, a, b in zip(ms, jr, pr)}
        return res

    H = [r for r in rows if r["kind"] == "harmful"]
    Bn = [r for r in rows if r["kind"] == "benign"]
    rep = {"model": name, "harmful_vs_is_toxic": stats(H, "is_toxic", "p_harm"),
           "incoherent_vs_is_incoherent": stats(H, "is_incoherent", "p_incoh"),
           "incoherent_vs_benign_is_broken": stats(Bn, "is_broken", "p_incoh")}
    short = name.split("/")[-1]
    json.dump(rep, open(OUT / f"local_judge_validation_{short}.json", "w"), indent=2)
    for k in ["harmful_vs_is_toxic", "incoherent_vs_is_incoherent", "incoherent_vs_benign_is_broken"]:
        v = rep[k]
        print(f"{k}: AUROC {v['auroc']:.3f} (unanimous {v['auroc_unanimous']:.3f}); model-rate r(mean prob)="
              f"{v['mean_prob_model_pearson']:.3f}; by t: " +
              ", ".join(f"{t}: r={d['model_rate_pearson']:.2f}/rho={d['model_rate_spearman']:.2f}"
                        for t, d in v["by_threshold"].items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct")
