"""Coherence pilot: SFT variants. Saves the final weights only.

  python cp_sft.py --base PATH --out DIR --data uf|ufshort|ad|hh|mix --fmt bt|orig --eos 1 --lr 1e-5 --epochs 1

--fmt orig --eos 0 --data uf --max-len 512 reproduces the old recipe (src/train_sft.py: "\n\nHuman: ..\n\nAssistant: ..",
no end-of-text, lr 1e-5, effective batch 32, 1 epoch). --fmt bt writes "Human: ..\nAssistant: .." (the eval format).
--eos 1 appends the end-of-text token as a target, unless the example was truncated.
Data (single turn, benign instruction data, no refusals by construction):
  uf       UltraFeedback train_sft, chosen reply
  ufshort  the same, request and reply each at most 100 tokens
  ad       Alpaca (no input field) + Dolly (no context)
  hh       HH helpful train, first exchange of the chosen conversation (prompts in any pilot prompt set removed)
  mix      ufshort + ad
  corpus   a sample of the pretraining mixture (raw rows, Uniform scheme): retrofit of end-of-text, not SFT
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[1]
CP = ROOT / "coherence_pilot"
EOS = 50256


def load_pairs(name, tok, n_rows=0):
    from datasets import load_dataset
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    held = {norm(p[len("Human: "):-len("\nAssistant:")]) for v in json.load(open(CP / "prompts.json")).values() for p in v}
    pairs = []
    if name in ("uf", "ufshort", "mix"):
        for e in load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft"):
            m = e["messages"]
            if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant" and m[1]["content"].strip():
                pairs.append((m[0]["content"].strip(), m[1]["content"].strip(), "uf"))
        if name != "uf":
            la, lq = tok([a for _, a, _ in pairs])["input_ids"], tok([q for q, _, _ in pairs])["input_ids"]
            pairs = [p for p, x, y in zip(pairs, la, lq) if len(x) <= 100 and len(y) <= 100]
    if name in ("ad", "mix"):
        for e in load_dataset("tatsu-lab/alpaca", split="train"):
            if not e["input"].strip() and e["output"].strip():
                pairs.append((e["instruction"].strip(), e["output"].strip(), "alpaca"))
        for e in load_dataset("databricks/databricks-dolly-15k", split="train"):
            if not e["context"].strip() and e["response"].strip():
                pairs.append((e["instruction"].strip(), e["response"].strip(), "dolly"))
    if name == "hh":
        for e in json.load(open(ROOT / "data" / "anthropic_hh" / "anthropic_hh_all.json")):
            if e["subset"].startswith("helpful") and e["pair_type"] == "chosen":
                t = e["text"].strip().split("\n\nHuman:")[0]
                if t.startswith("Human:") and "\n\nAssistant:" in t:
                    q, a = t[len("Human:"):].split("\n\nAssistant:", 1)
                    if q.strip() and a.strip():
                        pairs.append((q.strip(), a.strip(), "hh"))
        pairs = list({(q, a): (q, a, s) for q, a, s in pairs}.values())
    if name == "corpus":
        # A sample of the pretraining mixture itself (Uniform scheme: every row, weight 1), raw text, same row
        # formats as src/dataset.py. Used to retrofit end-of-text onto an existing model without changing its data.
        sys.path.insert(0, str(ROOT))
        import src.dataset as D
        r = random.Random("cp-corpus")
        owt = json.load(open(ROOT / "data" / "openwebtext" / "owt_texts_200000.json"))
        bt = D.format_beavertails(D.load_beavertails("train", ROOT / "data" / "beavertails"))
        hh = [e["text"] for e in json.load(open(ROOT / "data" / "anthropic_hh" / "anthropic_hh_all.json"))]
        for src, texts, share in [("owt", owt, 0.69), ("bt", bt, 0.14), ("hh", hh, 0.17)]:  # 1.5M : 300k : 361k
            pairs += [(None, t, src) for t in r.sample(texts, int(n_rows * share))]
        print(f"[sft] data corpus: {len(pairs):,} rows (pretraining mixture sample; nothing held out)", flush=True)
        return pairs
    n0 = len(pairs)
    pairs = [p for p in pairs if norm(p[0]) not in held]
    print(f"[sft] data {name}: {len(pairs):,} pairs ({n0 - len(pairs)} removed as eval prompts)", flush=True)
    return pairs


def encode(pairs, tok, fmt, eos, max_len):
    texts = [(a if q is None else f"Human: {q}\nAssistant: {a}" if fmt == "bt" else f"\n\nHuman: {q}\n\nAssistant: {a}")
             for q, a, _ in pairs]
    out = []
    for i in range(0, len(texts), 5000):
        for ids in tok(texts[i:i + 5000])["input_ids"]:
            out.append(ids + [EOS] if eos and len(ids) < max_len else ids[:max_len])
    out = [r for r in out if len(r) >= 2]
    return out


def batches(rows, bs, shuffle, rng):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx) - (bs - 1 if shuffle else 0), bs):
        b = [rows[j] for j in idx[i:i + bs]]
        L = max(map(len, b))
        ids = torch.full((len(b), L), EOS, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for k, r in enumerate(b):
            ids[k, :len(r)] = torch.tensor(r)
            mask[k, :len(r)] = 1
        yield ids, mask


def lm_loss(model, ids, mask):
    with autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none")
    m = mask[:, 1:].float().reshape(-1)
    return (ce * m).sum() / m.sum().clamp(min=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--out", required=True)
    p.add_argument("--data", default="uf"); p.add_argument("--fmt", default="bt"); p.add_argument("--eos", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=32); p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--micro", type=int, default=8, help="micro-batch size; gradients accumulate to --bs (GPU memory)")
    p.add_argument("--warmup", type=int, default=100); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dropout", type=float, default=None, help="override the model's dropout during SFT")
    p.add_argument("--n-rows", type=int, default=96200, help="--data corpus: rows sampled from the pretraining mixture")
    p.add_argument("--mem", type=float, default=0.42, help="cap on this process's share of GPU memory")
    a = p.parse_args()
    out = Path(a.out)
    if (out / "sft_config.json").exists():
        print("[sft] exists, skipping", out)
        return
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    pairs = load_pairs(a.data, tok, a.n_rows)
    rng.shuffle(pairs)
    rows = encode(pairs, tok, a.fmt, a.eos, a.max_len)
    val, train = rows[:200], rows[200:]
    kw = {} if a.dropout is None else dict(resid_pdrop=a.dropout, embd_pdrop=a.dropout, attn_pdrop=a.dropout)
    torch.cuda.set_per_process_memory_fraction(a.mem)  # fail here rather than starve the pretraining run sharing the GPU
    model = GPT2LMHeadModel.from_pretrained(a.base, **kw).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = int(len(train) // a.bs * a.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(a.warmup, 1) if s < a.warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(total - a.warmup, 1))))

    @torch.no_grad()
    def val_loss():
        model.eval()
        ls = [lm_loss(model, i.cuda(), m.cuda()).item() for i, m in batches(val, a.micro, False, rng)]
        model.train()
        return sum(ls) / len(ls)

    log = [{"step": 0, "val_loss": val_loss()}]
    print(f"[sft] {len(train):,} train rows, {total} steps, mean len {sum(map(len, train)) / len(train):.0f} tok, "
          f"val0 {log[0]['val_loss']:.3f}", flush=True)
    model.train()
    acc = a.bs // a.micro
    assert acc * a.micro == a.bs
    step, t0 = 0, time.time()
    while step < total:
        for k, (ids, mask) in enumerate(batches(train, a.micro, True, rng), 1):
            loss = lm_loss(model, ids.cuda(), mask.cuda())
            (loss / acc).backward()
            if k % acc:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0 or step == total:
                log.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss(), "sec": time.time() - t0})
                print("[sft]", log[-1], flush=True)
            if step >= total:
                break
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    json.dump({**vars(a), "n_train": len(train), "steps": total, "log": log}, open(out / "sft_config.json", "w"), indent=1)
    print("[sft] saved", out, flush=True)


if __name__ == "__main__":
    main()
