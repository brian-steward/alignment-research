"""Does perplexity track Opus's coherence labels? (2026-10-07, exploratory; for the S16 judge decision)

Uses the ~2,000 Opus-labelled endpoint replies (both scales). Nothing here feeds a registered readout.
- Item level: the reply's perplexity under a reference LM (completion tokens given the prompt; GPT-2 and
  Qwen2.5-7B-Instruct as plain LMs), the repo's degeneracy metrics (distinct-2, repetition fraction), and a
  leave-one-checkpoint-out logistic combination of these, each scored by AUROC against the Opus majority label
  (ties count as incoherent, as in judge_combo_audit.py). The probe and Shieldstral P(incoherent) are shown for reference.
- Opus against itself: rep-vs-rep agreement on the same replies (same blind file, separate runs), and whether a
  stricter run's incoherent set contains the laxer run's (a shifted threshold rather than random disagreement).
- Checkpoint level: Opus incoherent share vs the checkpoint's WikiText-2 PPL (raw and as a ratio to the arm's
  step 0, M2's budget variable) and vs mean reply log-PPL.

  python coherence_ppl.py score   # GPU, ~5 min: caches results/s15/coherence_ppl/scores.jsonl
  python coherence_ppl.py eval    # CPU: results/s15/coherence_ppl.{md,json,png}
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.degeneracy_metrics import metrics as degeneracy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"200m": ROOT / "results" / "s15" / "endpoint_judge", "50m": ROOT / "results" / "s15_50m" / "endpoint_judge"}
RUNS = {"200m": ROOT / "results" / "s15" / "runs", "50m": ROOT / "results" / "s15_50m" / "runs"}
OUT = ROOT / "results" / "s15" / "coherence_ppl"
LMS = {"gpt2": "gpt2", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}


def labelled():
    """[(scale, tag, n, reps)] for every checkpoint with >= 1 complete Opus rep; reps = [{idx: row}]."""
    res = []
    for scale, d in DIRS.items():
        key = json.load(open(d / "blind_key.json"))
        for code, tag in sorted(key.items(), key=lambda x: x[1]):
            n = sum(1 for _ in open(d / "blind" / f"{code}.jsonl"))
            reps = []
            for f in sorted((d / "blind").glob(f"{code}_*.jsonl")):
                rows = {}
                for line in open(f):
                    try:
                        r = json.loads(line)
                        rows[r["idx"]] = r
                    except (json.JSONDecodeError, KeyError):
                        pass
                if set(rows) == set(range(n)):
                    reps.append(rows)
            if reps:
                res.append((scale, tag, n, reps))
    return res


def items():
    for scale, tag, n, _ in labelled():
        for line in open(DIRS[scale] / f"input_{tag}.jsonl"):
            r = json.loads(line)
            yield scale, tag, r["idx"], r["prompt"], r["completion"]


def score():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = list(items())
    out = {(s, t, i): {"scale": s, "tag": t, "idx": i, **degeneracy(c)} for s, t, i, _, c in rows}
    for name, repo in LMS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
        pre = tok.bos_token or ""
        tok.pad_token = tok.pad_token or tok.eos_token
        tok.padding_side = "right"
        for b in range(0, len(rows), 16):
            batch = rows[b:b + 16]
            texts = [pre + p + c for _, _, _, p, c in batch]
            enc = tok(texts, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True,
                      max_length=1024)
            offs = enc.pop("offset_mapping")
            with torch.no_grad():
                lp = torch.log_softmax(model(**{k: v.cuda() for k, v in enc.items()}).logits.float(), -1)
            ids = enc["input_ids"].cuda()
            tok_lp = lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0].cpu()
            for j, (s, t, i, p, c) in enumerate(batch):
                start = len(pre + p)
                # completion tokens: real (unpadded) tokens whose span starts at or after the prompt's end
                m = (offs[j, 1:, 0] >= start) & (enc["attention_mask"][j, 1:] == 1) & (offs[j, 1:, 1] > offs[j, 1:, 0])
                nll = -tok_lp[j][m]
                out[(s, t, i)][f"nll_{name}"] = float(nll.mean()) if len(nll) else float("nan")
            print(name, b, flush=True)
        del model
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "scores.jsonl", "w") as f:
        for r in out.values():
            f.write(json.dumps(r) + "\n")


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def bal_acc(pred, ref):
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    return 0.5 * (pred[ref].mean() + (~pred[~ref]).mean())


def wikitext_ppl(scale, tag):
    parts = tag.split("__")
    arm, step = parts[0], int(parts[-1].removeprefix("step"))
    lr = parts[1].removeprefix("lr") if len(parts) == 3 else "3e-05"
    rec = {r["step"]: r for r in map(json.loads, open(RUNS[scale] / f"{arm}__attack__lr{lr}__s0.jsonl"))}
    return rec[step]["metrics"]["wikitext_ppl"], rec[0]["metrics"]["wikitext_ppl"]


def evaluate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler
    sc = {(r["scale"], r["tag"], r["idx"]): r for r in map(json.loads, open(OUT / "scores.jsonl"))}
    lab = labelled()
    X, y, grp, ckpt = [], [], [], []
    ref = {"probe": [], "shieldstral": []}
    agree = {"is_incoherent": [], "is_toxic": []}
    nested = []
    for g, (scale, tag, n, reps) in enumerate(lab):
        probe = {r["idx"]: r["incoh"] for r in map(json.loads, open(DIRS[scale] / f"probe_{tag}.jsonl"))}
        sh = {r["idx"]: r["p_incoh"] for r in map(json.loads, open(DIRS[scale] / f"shieldstral_{tag}.jsonl"))}
        for i in range(n):
            v = [bool(rep[i]["is_incoherent"]) for rep in reps]
            s = sc[(scale, tag, i)]
            X.append([s["nll_gpt2"], s["nll_qwen7b"], s["distinct2"], s["rep_frac"], math.log1p(s["n_tok"])])
            y.append(sum(v) * 2 >= len(v))
            grp.append(g)
            ref["probe"].append(probe[i])
            ref["shieldstral"].append(sh[i])
        for a, b in itertools.combinations(range(len(reps)), 2):
            for k in agree:
                la = [bool(reps[a][i][k]) for i in range(n)]
                lb = [bool(reps[b][i][k]) for i in range(n)]
                agree[k].append({"scale": scale, "tag": tag, "rate_a": np.mean(la), "rate_b": np.mean(lb),
                                 "kappa": kappa(la, lb), "bal_acc": 0.5 * (bal_acc(la, lb) + bal_acc(lb, la)),
                                 "fpr": float(np.mean([x for x, z in zip(la, lb) if not z])) if not all(lb) else 0.0,
                                 "tpr": float(np.mean([x for x, z in zip(la, lb) if z])) if any(lb) else 0.0})
                if k == "is_incoherent":
                    hi, lo = (la, lb) if np.mean(la) >= np.mean(lb) else (lb, la)
                    nested.append(float(np.mean([h for h, l in zip(hi, lo) if l])) if any(lo) else float("nan"))
    X, y, grp = np.array(X), np.array(y), np.array(grp)
    ok = ~np.isnan(X).any(1)
    # degenerate empty replies have no completion tokens: give them the worst observed NLL (they are incoherent)
    X[~ok, :2] = np.nanmax(X[:, :2], 0)
    feats = {"ppl_gpt2": X[:, 0], "ppl_qwen7b": X[:, 1], "low distinct-2": -X[:, 2], "repetition": X[:, 3]}
    scores = dict(feats)
    combo = np.zeros(len(y))
    for g in np.unique(grp):
        tr = grp != g
        ss = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(ss.transform(X[tr]), y[tr])
        combo[~tr] = clf.predict_proba(ss.transform(X[~tr]))[:, 1]
    scores["PPL + repetition (LOCO logistic)"] = combo
    scores["probe P(incoh)"] = np.array(ref["probe"])
    scores["Shieldstral P(incoh)"] = np.array(ref["shieldstral"])
    auc = {k: float(roc_auc_score(y, v)) for k, v in scores.items()}
    multi = np.array([len(r) >= 2 for _, _, n, r in lab for _ in range(n)])
    # items where every rep agrees (checkpoints with >= 2 reps): the labels Opus is sure of
    sure = np.array([len(r) >= 2 and len({bool(rep[i]["is_incoherent"]) for rep in r}) == 1
                     for _, _, n, r in lab for i in range(n)])
    auc_sure = {k: float(roc_auc_score(y[sure], v[sure])) for k, v in scores.items()}

    ck = []
    for g, (scale, tag, n, reps) in enumerate(lab):
        rates = [np.mean([bool(rep[i]["is_incoherent"]) for i in range(n)]) for rep in reps]
        w, w0 = wikitext_ppl(scale, tag)
        m = grp == g
        ck.append({"scale": scale, "tag": tag, "reps": len(reps), "opus_incoh": rates, "opus_mean": float(np.mean(rates)),
                   "wikitext_ppl": w, "ppl_ratio": w / w0, "reply_nll_qwen7b": float(X[m, 1].mean()),
                   "reply_nll_gpt2": float(X[m, 0].mean()),
                   "probe_incoh": float((np.array(ref["probe"])[m] >= 0.5).mean()),
                   "sh_incoh": float((np.array(ref["shieldstral"])[m] >= 0.5).mean()),
                   "combo_incoh": float(combo[m].mean())})
    om = np.array([c["opus_mean"] for c in ck])
    cor = {k: float(np.corrcoef(om, f([c for c in ck]))[0, 1]) for k, f in {
        "log WikiText PPL": lambda c: np.log([x["wikitext_ppl"] for x in c]),
        "log PPL ratio to step 0": lambda c: np.log([x["ppl_ratio"] for x in c]),
        "mean reply NLL (Qwen-7B)": lambda c: [x["reply_nll_qwen7b"] for x in c],
        "mean reply NLL (GPT-2)": lambda c: [x["reply_nll_gpt2"] for x in c],
        "probe incoherent share": lambda c: [x["probe_incoh"] for x in c],
        "Shieldstral incoherent share": lambda c: [x["sh_incoh"] for x in c],
        "PPL + repetition mean P(incoh)": lambda c: [x["combo_incoh"] for x in c]}.items()}
    # how well one Opus rep's checkpoint rate predicts another's (checkpoints with >= 2 reps)
    pairs = [(a["rate_a"], a["rate_b"]) for a in agree["is_incoherent"]]
    rep_r = float(np.corrcoef(*zip(*pairs))[0, 1])
    rep_diff = [abs(a - b) for a, b in pairs]

    res = {"n_items": int(len(y)), "n_checkpoints": len(ck), "opus_incoherent_rate": float(y.mean()),
           "auroc": auc, "auroc_unanimous": auc_sure, "n_unanimous": int(sure.sum()), "n_multi_rep_items": int(multi.sum()),
           "opus_rep_agreement": {k: {"kappa_mean": float(np.mean([a["kappa"] for a in v])),
                                      "bal_acc_mean": float(np.mean([a["bal_acc"] for a in v])),
                                      "n_pairs": len(v)} for k, v in agree.items()},
           "opus_rep_pairs": agree, "containment_median": float(np.nanmedian(nested)), "containment": nested,
           "checkpoints": ck, "checkpoint_r_vs_opus": cor, "opus_rep_vs_rep_checkpoint_r": rep_r,
           "opus_rep_abs_diff_median": float(np.median(rep_diff)), "opus_rep_abs_diff_max": float(np.max(rep_diff))}
    json.dump(res, open(OUT.parent / "coherence_ppl.json", "w"), indent=1)

    roc = {k: roc_curve(y, v)[:2] for k, v in scores.items()}
    figure(res, roc, X, y)
    L = ["# Does perplexity track Opus's coherence labels? (2026-10-07, exploratory)", "",
         f"{len(y)} Opus-labelled endpoint replies from {len(ck)} checkpoints (200M and 50M), Opus majority label, "
         f"{100 * y.mean():.0f}% incoherent. Code: s15_susceptibility/coherence_ppl.py. Figure: coherence_ppl.png.", "",
         "## Item level: AUROC against Opus incoherent", "",
         f"| Score | AUROC (all {len(y)}) | AUROC (replies where every Opus rep agrees, n={int(sure.sum())}) |", "|---|---|---|"]
    L += [f"| {k} | {auc[k]:.3f} | {auc_sure[k]:.3f} |" for k in auc]
    ra = res["opus_rep_agreement"]
    L += ["", "## Opus against itself (same replies, separate runs)", "",
          f"- Coherence: mean kappa {ra['is_incoherent']['kappa_mean']:.2f}, balanced accuracy "
          f"{ra['is_incoherent']['bal_acc_mean']:.3f} over {ra['is_incoherent']['n_pairs']} rep pairs (a binary label's "
          "balanced accuracy is its AUROC, so this is the ceiling for any score).",
          f"- Toxicity: mean kappa {ra['is_toxic']['kappa_mean']:.2f}, balanced accuracy {ra['is_toxic']['bal_acc_mean']:.3f}.",
          f"- Checkpoint incoherent share, rep vs rep: r {rep_r:.2f}, median |difference| "
          f"{100 * res['opus_rep_abs_diff_median']:.0f} points, max {100 * res['opus_rep_abs_diff_max']:.0f}.",
          f"- Containment: of the replies the laxer run calls incoherent, the stricter run also calls incoherent a median "
          f"{100 * res['containment_median']:.0f}% (1.0 would mean a pure threshold shift).", "",
          "| Checkpoint | Opus incoherent % by rep | kappa |", "|---|---|---|"]
    for a in agree["is_incoherent"]:
        L.append(f"| {a['scale']} {a['tag']} | {100 * a['rate_a']:.0f} / {100 * a['rate_b']:.0f} | {a['kappa']:.2f} |")
    L += ["", "## Checkpoint level: correlation with Opus incoherent share (mean over reps)", "",
          "| Measure | r |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in cor.items()]
    L += [f"| (Opus rep vs another rep) | {rep_r:.2f} |", "",
          "| Checkpoint | reps | Opus incoh % (reps) | WikiText PPL | ratio to step 0 | reply NLL Qwen | probe incoh % | Sh incoh % |",
          "|---|---|---|---|---|---|---|---|"]
    for c in ck:
        L.append(f"| {c['scale']} {c['tag']} | {c['reps']} | {'/'.join(f'{100 * r:.0f}' for r in c['opus_incoh'])} | "
                 f"{c['wikitext_ppl']:.0f} | {c['ppl_ratio']:.2f} | {c['reply_nll_qwen7b']:.2f} | "
                 f"{100 * c['probe_incoh']:.0f} | {100 * c['sh_incoh']:.0f} |")
    (OUT.parent / "coherence_ppl.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def figure(res, roc, X, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREEN, INK, MUTED = "#2a78d6", "#e0602c", "#17a372", "#1d2330", "#6b7280"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    a = ax[0]
    style = {"ppl_qwen7b": (BLUE, "-"), "ppl_gpt2": (BLUE, ":"), "repetition": (ORANGE, ":"),
             "PPL + repetition (LOCO logistic)": (ORANGE, "-"), "probe P(incoh)": (GREEN, "-"),
             "Shieldstral P(incoh)": (GREEN, "--")}
    names = {"ppl_qwen7b": "reply PPL, Qwen-7B", "ppl_gpt2": "reply PPL, GPT-2", "repetition": "repetition fraction",
             "PPL + repetition (LOCO logistic)": "PPL + repetition (held-out fit)", "probe P(incoh)": "S15 probe",
             "Shieldstral P(incoh)": "Shieldstral"}
    for k, (c, ls) in style.items():
        fpr, tpr = roc[k]
        a.plot(fpr, tpr, color=c, ls=ls, lw=2, label=f"{names[k]}  {res['auroc'][k]:.2f}")
    pts = res["opus_rep_pairs"]["is_incoherent"]
    a.scatter([p["fpr"] for p in pts], [p["tpr"] for p in pts], s=40, color=INK, zorder=5,
              edgecolor="white", lw=1.5, label=f"one Opus run vs another  {res['opus_rep_agreement']['is_incoherent']['bal_acc_mean']:.2f}")
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    a.set(xlabel="false-positive rate", ylabel="true-positive rate", xlim=(0, 1), ylim=(0, 1.01),
          title="Replies: predicting Opus 'incoherent'")
    a.legend(title="AUROC (dots: balanced accuracy)", fontsize=8.5, title_fontsize=8.5, loc="lower right", frameon=False)

    a = ax[1]
    o = np.random.default_rng(0).permutation(len(y))  # mixed draw order, so neither class hides the other
    a.scatter(np.exp(X[o, 1]), X[o, 3], s=9, alpha=0.45, c=np.where(y[o], ORANGE, BLUE), lw=0)
    a.scatter([], [], s=20, color=BLUE, label="Opus: coherent")
    a.scatter([], [], s=20, color=ORANGE, label="Opus: incoherent")
    a.set_xscale("log")
    a.set(xlabel="reply perplexity under Qwen-7B (log scale)", ylabel="repetition fraction",
          title="Replies: mostly overlapping, two incoherent tails")
    a.annotate("loops: repetitive, low PPL", xy=(1.6, 0.8), fontsize=8.5, color=INK)
    a.annotate("word salad:\nhigh PPL", xy=(250, 0.04), fontsize=8.5, color=INK)
    a.legend(fontsize=8.5, frameon=False, loc="upper right", markerscale=2)

    a = ax[2]
    for c in res["checkpoints"]:
        r = 100 * np.array(c["opus_incoh"])
        col = BLUE if c["scale"] == "200m" else GREEN
        a.vlines(c["ppl_ratio"], r.min(), r.max(), color=col, lw=2, alpha=0.6)
        a.scatter([c["ppl_ratio"]] * len(r), r, s=14, color=col, alpha=0.6, lw=0)
        a.scatter(c["ppl_ratio"], r.mean(), s=46, color=col, edgecolor="white", lw=1.5, zorder=5)
    a.axvline(2, color=MUTED, lw=0.8, ls="--")
    a.annotate("M2 budget (2x)", xy=(2.05, 3), fontsize=8.5, color=MUTED)
    a.set_xscale("log")
    a.scatter([], [], s=46, color=BLUE, label="200M checkpoint")
    a.scatter([], [], s=46, color=GREEN, label="50M checkpoint")
    a.plot([], [], color=MUTED, lw=2, label="range over Opus runs")
    a.set(xlabel="WikiText-2 PPL / arm's step-0 PPL (log scale)", ylabel="Opus incoherent %", ylim=(0, 100),
          title=f"Checkpoints: r = {res['checkpoint_r_vs_opus']['log PPL ratio to step 0']:.2f} (mean over runs)")
    a.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.savefig(OUT.parent / "coherence_ppl.png", dpi=150)


if __name__ == "__main__":
    {"score": score, "eval": evaluate}[sys.argv[1]]()
