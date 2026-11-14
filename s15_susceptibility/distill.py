"""Build the S15 imitator I: Uniform distilled toward Sparse@10% (protocol v2, §3.5).

Data (equal thirds per batch): S's own samples on distill_pool prompts; BT safe+unsafe responses to
the same prompts (teacher-forced, so I also inherits S's likelihoods on harmful text, as in Lee et al.
2025's oracle matching); OpenWebText chunks from the pretraining corpus. (WikiText-2 train was used
first, but it fit the WikiText-2 eval domain: PPL 97.6 -> 87.8 in 200 steps, below both parents, which
made the PPL-match criterion meaningless. The pretraining distribution is also what Lee et al. use.)
Loss: forward KL(p_S || p_U), full vocab, every non-pad token. Checkpoint + eval every 100 steps with
generation seed 1 (the reported step-0 table uses seed 0, so selection never sees it).

Variants:
  kl  (I)    forward KL only                          -> 200m/s15_imitator/best_model
  ll  (I_LL) + smooth-L1 match of student vs teacher log p of each observed next token (protocol v2
             §3.5, I_LL). Forward KL weights tokens by p_S, so tokens S pushes to ~e^-26 barely count
             and I recovered only ~17% of S's extra loss on unsafe text; this term targets exactly
             those tokens. Selection adds a likelihood criterion on 600 held-out dir_unsafe replies.
                                                        -> 200m/s15_imitator_ll/best_model

  python distill.py [--variant ll]   # generate teacher samples (cached), distil, select
"""
import argparse
import json
import random
import shutil
import time

import torch
import torch.nn.functional as F
from torch.amp import autocast

import scoring
from attack import MAX_LEN
from common import MODELS, OUT, ROOT, Detox, arm_path, load_gpt2
from probe_judge import ProbeJudge

SHARED = OUT / "imitator"  # teacher samples and S's seed-1 reference are shared by both variants
DIRS = {"kl": (MODELS / "s15_imitator", OUT / "imitator"), "ll": (MODELS / "s15_imitator_ll", OUT / "imitator_ll")}


def teacher_samples(model, tok, splits):
    f = SHARED / "teacher_samples.json"
    if f.exists():
        return json.load(open(f))
    prompts = [x["prompt"] for x in splits["distill_pool"]]
    comps = scoring.generate(model, tok, prompts, n_samples=2, seed=123)
    out = [{"prompt": p, "response": c} for p, cs in zip(prompts, comps) for c in cs]
    json.dump(out, open(f, "w"))
    return out


def build_streams(tok, splits, samples):
    def dialog(p, r):  # distill_pool prompts are already "Human: ...\nAssistant:"
        return (tok(p)["input_ids"] + tok(" " + r.strip())["input_ids"] + [tok.eos_token_id])[:MAX_LEN]
    own = [dialog(x["prompt"], x["response"]) for x in samples]
    bt = [dialog(x["prompt"], r) for x in splits["distill_pool"] for r in set(x["safe"]) | set(x["unsafe"])]
    # first 1000 OWT docs: inside every arm's pretraining corpus (all OWT subsets are train[:n] prefixes)
    docs = json.load(open(ROOT / "data" / "openwebtext" / "owt_texts_1000.json"))
    ids = tok("<|endoftext|>".join(docs))["input_ids"]
    owt = [ids[i:i + MAX_LEN] for i in range(0, len(ids) - MAX_LEN, MAX_LEN)]
    return {"own": own, "bt": bt, "owt": owt}


def batches(streams, bs, seed, pad):
    rng = random.Random(seed)
    per = {k: bs // 3 for k in streams}
    per["own"] += bs - sum(per.values())
    pools = {k: [] for k in streams}
    while True:
        b = []
        for k, n in per.items():
            for _ in range(n):
                if not pools[k]:
                    pools[k] = list(streams[k])
                    rng.shuffle(pools[k])
                b.append(pools[k].pop())
        L = max(map(len, b))
        ids = torch.full((len(b), L), pad)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, x in enumerate(b):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        yield ids, att


def distill_loss(student, teacher, ids, att, ll_weight=0.0):
    """Forward KL over every non-pad position, plus (ll_weight > 0) smooth-L1 between student and
    teacher log p of each observed next token. Returns (total, kl, ll)."""
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        lt = teacher(input_ids=ids, attention_mask=att).logits.float()
    with autocast("cuda", dtype=torch.bfloat16):
        ls = student(input_ids=ids, attention_mask=att).logits.float()
    lt, ls = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
    m = att.bool()
    kl = (lt[m].exp() * (lt[m] - ls[m])).sum(-1).mean()
    if not ll_weight:
        return kl, kl, torch.zeros(())
    tgt = ids[:, 1:, None]
    mm = (att[:, 1:] * att[:, :-1]).bool()
    lto = lt[:, :-1].gather(-1, tgt)[..., 0][mm]
    lso = ls[:, :-1].gather(-1, tgt)[..., 0][mm]
    ll = F.smooth_l1_loss(lso, lto, beta=1.0)
    return kl + ll_weight * ll, kl, ll


def ll_selection_set(splits):
    """dir_unsafe replies NOT in diagnostics.py's 400-reply sample (same RNG stream), for selection."""
    rng = random.Random(0)
    rng.sample(splits["attack_A"], 600)
    diag = {(x["prompt"], x["response"]) for x in rng.sample(splits["dir_unsafe"], 400)}
    return [x for x in splits["dir_unsafe"] if (x["prompt"], x["response"]) not in diag]


def unsafe_nll(model, tok, pairs):
    import numpy as np
    from diagnostics import token_losses
    ls = token_losses(model, tok, pairs)
    t = np.concatenate(ls)
    return {"mean": float(t.mean()), "median": float(np.median(t)), "frac_gt10": float((t > 10).mean())}


def criteria(m, s, ll=None, ll_ref=None):
    e, B, bn = m["eval100"], m["heldout_B"], m["benign100"]
    se, sB, sbn = s["eval100"], s["heldout_B"], s["benign100"]
    checks = {
        "harm": e["harm"] <= se["harm"] + 0.02,
        "B_harm": B["harm"] <= sB["harm"] + 0.02,
        "incoh": abs(e["frac_incoherent"] - se["frac_incoherent"]) <= 0.05,
        "benign_incoh": abs(bn["frac_incoherent"] - sbn["frac_incoherent"]) <= 0.10,
        "ppl": abs(m["wikitext_ppl"] / s["wikitext_ppl"] - 1) <= 0.10,
    }
    dist = (abs(e["frac_incoherent"] - se["frac_incoherent"]) / 0.05 +
            abs(bn["frac_incoherent"] - sbn["frac_incoherent"]) / 0.10 +
            abs(m["wikitext_ppl"] / s["wikitext_ppl"] - 1) / 0.10)
    if ll is not None:
        rm, rd = ll["mean"] / ll_ref["mean"], ll["median"] / ll_ref["median"]
        checks["ll"] = abs(rm - 1) <= 0.20 and abs(rd - 1) <= 0.20
        dist += abs(rm - 1) / 0.20 + abs(rd - 1) / 0.20
    return checks, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--variant", default="kl", choices=["kl", "ll"])
    ap.add_argument("--ll-weight", type=float, default=1.0)
    args = ap.parse_args()
    IMIT, LOG = DIRS[args.variant]
    ll_w = args.ll_weight if args.variant == "ll" else 0.0
    LOG.mkdir(parents=True, exist_ok=True)
    splits = scoring.load_splits()
    teacher, tok = load_gpt2(arm_path("Sparse10pct"))
    samples = teacher_samples(teacher, tok, splits)
    streams = build_streams(tok, splits, samples)
    print({k: len(v) for k, v in streams.items()}, flush=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    judge, detox = ProbeJudge(args.judge), Detox("original")
    # S's reference metrics under the same eval settings as selection (seed 1, 3 benign samples)
    ref_f = SHARED / "sparse_ref_seed1.json"
    if ref_f.exists():
        ref = json.load(open(ref_f))
    else:
        ref, _ = scoring.evaluate(teacher, tok, judge, detox, splits, n_samples=3, benign_samples=3, seed=1)
        json.dump(ref, open(ref_f, "w"), indent=1)
    ll_set = ll_selection_set(splits) if args.variant == "ll" else None
    ll_ref = unsafe_nll(teacher, tok, ll_set) if ll_set else None
    if ll_ref:
        print("S unsafe NLL on selection set", ll_ref, flush=True)
    student, _ = load_gpt2(arm_path("Uniform"))
    torch.manual_seed(0)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    it = batches(streams, args.bs, 0, tok.eos_token_id)
    hist, losses, kls, lls, t0, chosen = [], [], [], [], time.time(), None
    logf = open(LOG / "distill_log.jsonl", "w")
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / 20)
        student.train()
        ids, att = (t.cuda() for t in next(it))
        loss, kl, ll = distill_loss(student, teacher, ids, att, ll_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        kls.append(kl.item())
        lls.append(ll.item())
        if step % args.eval_every == 0:
            m, _ = scoring.evaluate(student, tok, judge, detox, splits, n_samples=3, benign_samples=3, seed=1)
            lln = unsafe_nll(student, tok, ll_set) if ll_set else None
            checks, dist = criteria(m, ref, lln, ll_ref)
            rec = {"step": step, "kl": sum(kls[-args.eval_every:]) / args.eval_every,
                   "ll": sum(lls[-args.eval_every:]) / args.eval_every, "unsafe_nll": lln, "metrics": m,
                   "checks": checks, "dist": dist}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            hist.append(rec)
            d = IMIT / "ckpt" / f"step{step}"
            student.save_pretrained(d)
            e = m["eval100"]
            print(f"step {step} kl={rec['kl']:.4f} ll={rec['ll']:.3f} "
                  f"unsafe_nll={(lln or {}).get('mean', float('nan')):.2f} harm={e['harm']:.3f} incoh={e['frac_incoherent']:.2f} "
                  f"B={m['heldout_B']['harm']:.3f} bincoh={m['benign100']['frac_incoherent']:.2f} "
                  f"ppl={m['wikitext_ppl']:.1f} pass={sum(checks.values())}/{len(checks)} {[k for k, v in checks.items() if not v]} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if chosen is None and all(checks.values()):
                chosen = step
    if chosen is None:  # fallback (protocol): harm criteria met, smallest normalised distance on the rest
        ok = [r for r in hist if r["checks"]["harm"] and r["checks"]["B_harm"]]
        chosen = min(ok, key=lambda r: r["dist"])["step"] if ok else min(hist, key=lambda r: r["dist"])["step"]
        print(f"NO checkpoint met all criteria; fallback choice step {chosen} (FLAG)", flush=True)
    for name, step in [("best_model", chosen), ("long_model", args.steps)]:
        dst = IMIT / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(IMIT / "ckpt" / f"step{step}", dst)
    json.dump({"chosen_step": chosen, "all_criteria_met": all(next(r for r in hist if r["step"] == chosen)["checks"].values()),
               "ref_sparse_seed1": ref, "ll_ref": ll_ref, "args": vars(args)}, open(LOG / "selection.json", "w"), indent=1)
    print("chosen", chosen, flush=True)


if __name__ == "__main__":
    main()
