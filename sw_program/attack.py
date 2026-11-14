"""Signed-weight program, phase 3: fine-tuning attack. A finished model is fine-tuned on held-out harmful
question-answer pairs (the 608 attack prompts, which no training row asks) and saved along a step grid.

  python sw_program/attack.py --model sparse10_s42 [--lr 1e-4] [--seed 0] [--steps 25 50 100 250 500 1000]
  python sw_program/attack.py --model NAME=PATH ...            (any saved model, e.g. a checkpoint for a smoke test)

The loop is S15's (s15_susceptibility/attack.py): batch 8, AdamW without weight decay, 10-step warm-up then a
constant rate, gradient clipping at 1, rows cut at 256 tokens. The loss is on the reply and its end-of-text token
only: the prompt and the padding are masked (the April attack result was a padding artifact).
Output: models/signed_weight_exp/<size>/swp_attack/<NAME>__lr<lr>__a<seed>/step<k>/ and attack_log.json there.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
HELD = ROOT / "data" / "sw_program" / "heldout_v1.json"
SPLITS = ROOT / "s15_susceptibility" / "splits.json"
EOS = 50256
MAX_LEN = 256
GRID = [25, 50, 100, 250, 500, 1000]


def attack_pairs():
    """(prompt, unsafe reply) pairs for the program's attack prompts. S15's attack_A holds 2,000 pairs over 610
    prompts; the program dropped two prompts that are also evaluation prompts."""
    keep = set(json.load(open(HELD))["sets"]["attack_A"])
    pairs = [(e["prompt"], e["response"]) for e in json.load(open(SPLITS))["attack_A"] if e["prompt"] in keep]
    assert {p for p, _ in pairs} == keep, "every attack prompt must have at least one reply"
    return pairs


def encode(tok, pairs, max_len=MAX_LEN):
    """[(ids, labels)]: labels are -100 on the prompt, the reply tokens and end-of-text elsewhere."""
    out = []
    for q, r in pairs:
        p = tok(f"Human: {q}\nAssistant:")["input_ids"]
        y = tok(" " + r)["input_ids"] + [EOS]
        ids, lab = (p + y)[:max_len], ([-100] * len(p) + y)[:max_len]
        if any(x != -100 for x in lab):          # a prompt that fills the window leaves nothing to learn from
            out.append((ids, lab))
    return out


def collate(b):
    L = max(len(x) for x, _ in b)
    ids = torch.full((len(b), L), EOS, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    for j, (x, y) in enumerate(b):
        ids[j, :len(x)] = torch.tensor(x)
        att[j, :len(x)] = 1
        lab[j, :len(y)] = torch.tensor(y)
    return ids, att, lab


def batches(exs, bs, seed):
    """Endless shuffled batches; every pass over the data uses each pair at most once."""
    rng, exs = random.Random(seed), list(exs)
    while True:
        rng.shuffle(exs)
        for i in range(0, len(exs) - bs + 1, bs):
            yield collate(exs[i:i + bs])


def resolve(spec, size="50m"):
    """'arm_sSEED' -> the program's final model; 'NAME=PATH' -> that path."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, Path(path)
    m = re.fullmatch(r"([a-z0-9]+)_s(\d+)", spec)
    assert m, f"model must be ARM_sSEED or NAME=PATH, got {spec}"
    return spec, ROOT / "models" / "signed_weight_exp" / size / f"swp_{m.group(1)}_seed{m.group(2)}_mv1_r1" / "final_model"


def out_dir(name, lr, seed, size="50m"):
    return ROOT / "models" / "signed_weight_exp" / size / "swp_attack" / f"{name}__lr{lr:g}__a{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", default="50m")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0, help="attack seed (batch order); the pretraining seed is in the model name")
    ap.add_argument("--steps", type=int, nargs="+", default=GRID, help="steps at which the model is saved; the last is the run length")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--mem", type=float, default=0.30, help="cap on this process's share of GPU memory")
    a = ap.parse_args()
    from torch.amp import autocast
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    name, path = resolve(a.model, a.size)
    out = out_dir(name, a.lr, a.seed, a.size)
    if (out / "attack_log.json").exists():
        print("[attack] exists, skipping", out)
        return
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    pairs = attack_pairs()
    exs = encode(tok, pairs)
    torch.manual_seed(a.seed)
    torch.cuda.set_per_process_memory_fraction(a.mem)
    model = GPT2LMHeadModel.from_pretrained(path).cuda()
    c = model.config
    assert c.resid_pdrop == c.embd_pdrop == c.attn_pdrop == 0, "program models are trained and attacked without dropout"
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    it, grid, losses, log, t0 = batches(exs, a.bs, a.seed), sorted(set(a.steps)), [], [], time.time()
    print(f"[attack] {name} lr={a.lr:g} seed={a.seed}: {len(exs):,} pairs over {len({p for p, _ in pairs})} prompts, saving at {grid}", flush=True)
    model.train()
    for step in range(1, grid[-1] + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 10)
        ids, att, lab = (t.cuda() for t in next(it))
        with autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step in grid:
            model.save_pretrained(out / f"step{step}")
            log.append({"step": step, "train_loss": sum(losses[-10:]) / len(losses[-10:]), "sec": time.time() - t0})
            print("[attack]", log[-1], flush=True)
    json.dump({"model": name, "base": str(path), "lr": a.lr, "seed": a.seed, "bs": a.bs, "max_len": MAX_LEN, "n_pairs": len(exs),
               "n_prompts": len({p for p, _ in pairs}), "first_losses": losses[:10], "log": log}, open(out / "attack_log.json", "w"), indent=1)
    print("[attack] done", out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
