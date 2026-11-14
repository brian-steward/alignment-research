"""S15 relearning curves: fine-tune an arm on attack_A (or benign WikiText) and evaluate along the way.

Example (protocol §3.2, step-2 LR sweep on Uniform vs Sparse@10%):
  python attack.py --arms Uniform Sparse10pct --lrs 1e-5 3e-5 1e-4 3e-4 1e-3 --seeds 0

Each (arm, lr, seed, data) run appends one JSON line per eval step to
results/s15/runs/<arm>__<data>__lr<lr>__s<seed>.jsonl, and every scored reply to the matching
.items.jsonl.gz (for the prompt bootstrap in summarize.py). Step-0 metrics and replies are cached per
arm in results/s15/baseline/<arm>.json/.items.jsonl (shared by every LR/seed of that arm).
Checkpoints (--save-at) are written only for --save-seeds.
"""
import argparse
import gzip
import json
import random
import time

import torch
from torch.amp import autocast

import scoring
from common import ARMS, OUT, Detox, arm_path, load_gpt2
from local_judge import RUBRIC_VERSION
from probe_judge import ProbeJudge

MAX_LEN = 256


def attack_batches(tok, splits, kind, seed, bs):
    """Infinite iterator of (input_ids, attention_mask, labels). Attack loss is on the reply only."""
    if kind == "attack":
        exs = []
        for x in splits["attack_A"]:
            p = tok(f"Human: {x['prompt']}\nAssistant:")["input_ids"]
            r = tok(" " + x["response"])["input_ids"] + [tok.eos_token_id]
            ids = (p + r)[:MAX_LEN]
            lab = ([-100] * len(p) + r)[:MAX_LEN]
            exs.append((ids, lab))
    else:  # benign control: WikiText-2 train chunks
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="train")
        ids_all = tok("\n\n".join(ds["page"]))["input_ids"]
        exs = [(ids_all[i:i + MAX_LEN], ids_all[i:i + MAX_LEN]) for i in range(0, 2000 * MAX_LEN, MAX_LEN)]
    rng = random.Random(seed)
    while True:
        rng.shuffle(exs)
        for i in range(0, len(exs) - bs + 1, bs):
            b = exs[i:i + bs]
            L = max(len(x[0]) for x in b)
            ids = torch.full((bs, L), tok.eos_token_id)
            att = torch.zeros((bs, L), dtype=torch.long)
            lab = torch.full((bs, L), -100)
            for j, (x, y) in enumerate(b):
                ids[j, :len(x)] = torch.tensor(x)
                att[j, :len(x)] = 1
                lab[j, :len(y)] = torch.tensor(y)
            yield ids, att, lab


def baseline(arm, judge, detox, splits, n_samples):
    f = OUT / "baseline" / f"{arm}.json"
    if f.exists():
        b = json.load(open(f))
        if b["n_samples"] == n_samples and b["judge"] == judge.name and b["coh_mode"] == scoring.COH_MODE:
            return b["metrics"]
    model, tok = load_gpt2(arm_path(arm))
    res, items = scoring.evaluate(model, tok, judge, detox, splits, n_samples=n_samples, benign_samples=n_samples)
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"arm": arm, "n_samples": n_samples, "judge": judge.name, "coh_mode": scoring.COH_MODE,
               "metrics": res}, open(f, "w"), indent=1)
    with open(f.with_suffix(".items.jsonl"), "w") as g:
        for k, rows in items.items():
            for r in rows:
                g.write(json.dumps({"set": k, **r}) + "\n")
    del model
    torch.cuda.empty_cache()
    return res


def is_complete(path, eval_at):
    """True if the run file holds every evaluation point: step 0 (the baseline) and each step in eval_at.

    Until 2026-11-14 this compared the line count with len(eval_at), which forgot the step-0 line, so a run that
    stopped one evaluation short was skipped as complete. A line cut off mid-write also counts as incomplete.
    """
    if not path.exists():
        return False
    try:
        steps = {json.loads(line)["step"] for line in open(path) if line.strip()}
    except (json.JSONDecodeError, KeyError):
        return False
    return {0} | set(eval_at) <= steps


def run(arm, lr, seed, kind, args, judge, detox, splits):
    out = OUT / "runs" / f"{arm}__{kind}__lr{lr:g}__s{seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_complete(out, args.eval_at):
        print(f"skip (complete): {out.name}")
        return
    out.unlink(missing_ok=True)
    items_f = out.with_suffix(".items.jsonl.gz")
    items_f.unlink(missing_ok=True)
    base = baseline(arm, judge, detox, splits, args.n_samples)
    with open(out, "a") as f:
        f.write(json.dumps({"arm": arm, "lr": lr, "seed": seed, "data": kind, "step": 0, "train_loss": None,
                            "metrics": base}) + "\n")
    torch.manual_seed(seed)
    model, tok = load_gpt2(arm_path(arm))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    batches = attack_batches(tok, splits, kind, seed, args.bs)
    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, step / 10)  # 10-step warmup, then constant
        model.train()
        ids, att, lab = (t.cuda() for t in next(batches))
        with autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step in args.save_at and seed in args.save_seeds:
            d = ARMS[arm].parent / "s15_attacks" / f"{arm}__{kind}__lr{lr:g}__s{seed}" / f"step{step}"
            model.save_pretrained(d)
        if step in args.eval_at:
            res, items = scoring.evaluate(model, tok, judge, detox, splits, n_samples=args.n_samples,
                                          benign_samples=1, seed=0)
            with gzip.open(items_f, "at") as g:
                for k, rows in items.items():
                    for r in rows:
                        g.write(json.dumps({"step": step, "set": k, **r}) + "\n")
            rec = {"arm": arm, "lr": lr, "seed": seed, "data": kind, "step": step,
                   "train_loss": sum(losses[-10:]) / len(losses[-10:]), "metrics": res}
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            e = res["eval100"]
            print(f"{arm} {kind} lr={lr:g} s={seed} step={step} loss={rec['train_loss']:.3f} "
                  f"harm={e['harm']:.3f} harm_coh={e['harm_coh']:.3f} incoh={e['frac_incoherent']:.2f} "
                  f"B_harm={res['heldout_B']['harm']:.3f} B_detox={res['heldout_B']['detox_rate']:.3f} "
                  f"ppl={res['wikitext_ppl']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["Uniform", "Sparse10pct"])
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--data", default="attack", choices=["attack", "benign"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--eval-at", default="5,10,25,50,100,250,500,1000")
    ap.add_argument("--save-at", default="")
    ap.add_argument("--save-seeds", default="0")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()
    args.eval_at = {int(x) for x in args.eval_at.split(",") if x}
    args.save_at = {int(x) for x in args.save_at.split(",") if x}
    args.save_seeds = {int(x) for x in args.save_seeds.split(",") if x}
    judge, detox, splits = ProbeJudge(args.judge), Detox("original"), scoring.load_splits()
    # learned heads on the judge LM's activations (probe_judge.py); name keys the baseline cache
    judge.name = f"probe:{args.judge}|rubric-{RUBRIC_VERSION}|" + ",".join(
        f"{r}@L{int(h['layer'])}" for r, h in sorted(judge.heads.items()))
    for arm in args.arms:
        if args.baseline_only:
            m = baseline(arm, judge, detox, splits, args.n_samples)
            print(arm, json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                       {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in m.items()}), flush=True)
            continue
        for lr in args.lrs:
            for seed in args.seeds:
                run(arm, lr, seed, args.data, args, judge, detox, splits)


if __name__ == "__main__":
    main()
