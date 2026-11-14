"""Diagnostics for reading the step-2 curves (results/s15/diagnostics.json).

1. Data ceiling: learned-judge harm of the attack_A responses themselves (a model that perfectly
   imitated the attack data would score about this on similar prompts).
2. Step-0 per-token loss of each arm on unsafe (dir_unsafe) vs safe (dir_safe) BeaverTails replies:
   is Sparse's high attack loss specific to unsafe text, and is it diffuse or a few huge tokens?
"""
import json
import random

import numpy as np
import torch
from torch.amp import autocast

from attack import MAX_LEN
from common import OUT, arm_path, load_gpt2
from probe_judge import ProbeJudge
from scoring import load_splits


@torch.no_grad()
def token_losses(model, tok, pairs, bs=16):
    out = []
    for i in range(0, len(pairs), bs):
        for x in pairs[i:i + bs]:  # unbatched is fine at 200M; avoids padding bookkeeping
            p = tok(f"Human: {x['prompt']}\nAssistant:")["input_ids"]
            r = tok(" " + x["response"])["input_ids"] + [tok.eos_token_id]
            ids = torch.tensor((p + r)[:MAX_LEN], device="cuda")[None]
            with autocast("cuda", dtype=torch.bfloat16):
                lg = model(ids).logits.float()[0, :-1]
            nll = torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="none")
            out.append(nll[len(p) - 1:].cpu().numpy())
    return out


def summ(losses):
    t = np.concatenate(losses)
    return {"mean": float(t.mean()), "median": float(np.median(t)), "p90": float(np.quantile(t, 0.9)),
            "frac_gt10": float((t > 10).mean()), "seq_mean_median": float(np.median([l.mean() for l in losses]))}


def losses_only(arms):
    """Step-0 unsafe/safe per-token loss for extra arms, merged into diagnostics.json (same 400+400 sample)."""
    s = load_splits()
    rng = random.Random(0)
    rng.sample(s["attack_A"], 600)  # keep the RNG stream identical to main()
    sets = {"unsafe": rng.sample(s["dir_unsafe"], 400), "safe": rng.sample(s["dir_safe"], 400)}
    rep = json.load(open(OUT / "diagnostics.json")) if (OUT / "diagnostics.json").exists() else {}
    for arm, which in arms:
        model, tok = load_gpt2(arm_path(arm, which))
        key = arm if which == "best_model" else f"{arm}:{which}"
        rep[key] = {k: summ(token_losses(model, tok, v)) for k, v in sets.items()}
        print(key, json.dumps(rep[key]), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(rep, open(OUT / "diagnostics.json", "w"), indent=1)


def main():
    s = load_splits()
    rng = random.Random(0)
    rep = {}
    J = ProbeJudge()
    sub = rng.sample(s["attack_A"], 600)
    h = J("harmful", [f"Human: {x['prompt']}\nAssistant:" for x in sub], [x["response"] for x in sub])
    rep["attack_data_harm"] = {"mean": float(np.mean(h)), "rate": float(np.mean(np.array(h) >= 0.5)), "n": len(h)}
    print("attack data harm", rep["attack_data_harm"], flush=True)
    del J
    torch.cuda.empty_cache()
    sets = {"unsafe": rng.sample(s["dir_unsafe"], 400), "safe": rng.sample(s["dir_safe"], 400)}
    for arm in ["Uniform", "Sparse10pct", "Filtered", "Standard"]:
        model, tok = load_gpt2(arm_path(arm))
        rep[arm] = {k: summ(token_losses(model, tok, v)) for k, v in sets.items()}
        print(arm, json.dumps(rep[arm]), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(rep, open(OUT / "diagnostics.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "losses":  # python diagnostics.py losses Imitator Imitator:long_model
        losses_only([tuple(a.split(":")) if ":" in a else (a, "best_model") for a in sys.argv[2:]])
    else:
        main()
