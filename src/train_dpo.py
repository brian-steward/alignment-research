"""Direct Preference Optimization on a pre-trained signed-weight base model.

Compares against the signed-weight pretraining methods. The story:
- Standard industry pipeline = Uniform pretrain → DPO/RLHF
- Signed-weight pretraining = alignment baked into pretraining gradients
- Hypothesis: signed-weight is more robust to adversarial fine-tuning attacks
  because alignment is encoded in the weights themselves, not as a learned
  refusal hood over a misaligned base.

This script trains DPO on Anthropic HH harmless-base preference pairs.

DPO loss (Rafailov et al. 2023):
    L = -log_sigmoid(beta * (
        logp_pol(y_w | x) - logp_ref(y_w | x)
        - logp_pol(y_l | x) + logp_ref(y_l | x)
    ))

We compute log_p over full chosen/rejected sequences (the shared prefix
cancels in the difference, so this is exact).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import GPT2LMHeadModel
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_tokenizer


def compute_logp(model, input_ids, attention_mask, amp_dtype):
    """Compute summed log-prob of input_ids under model.

    Returns (B,) tensor of summed log-probs over the response portion of each
    sequence. Uses attention_mask to mask out padding (and prompt tokens, if
    we want to compute response-only log-prob — but for DPO, the prefix cancels
    so we just compute over the full sequence).
    """
    with autocast("cuda", dtype=amp_dtype):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: logits[:, :-1] predict input_ids[:, 1:]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    selected = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    summed = (selected * shift_mask).sum(dim=-1)
    return summed


class HHPairsDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ex = self.pairs[i]
        chosen = self.tokenizer(
            ex["chosen"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        rejected = self.tokenizer(
            ex["rejected"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return (
            chosen["input_ids"][0], chosen["attention_mask"][0],
            rejected["input_ids"][0], rejected["attention_mask"][0],
        )


def evaluate(policy, ref, loader, device, amp_dtype, beta, max_batches=20):
    policy.eval()
    correct = 0
    total = 0
    losses = []
    margins = []
    with torch.no_grad():
        for i, (c_ids, c_mask, r_ids, r_mask) in enumerate(loader):
            if i >= max_batches:
                break
            c_ids, c_mask = c_ids.to(device), c_mask.to(device)
            r_ids, r_mask = r_ids.to(device), r_mask.to(device)
            logp_pol_c = compute_logp(policy, c_ids, c_mask, amp_dtype)
            logp_pol_r = compute_logp(policy, r_ids, r_mask, amp_dtype)
            logp_ref_c = compute_logp(ref, c_ids, c_mask, amp_dtype)
            logp_ref_r = compute_logp(ref, r_ids, r_mask, amp_dtype)
            margin = beta * ((logp_pol_c - logp_ref_c) - (logp_pol_r - logp_ref_r))
            loss = -F.logsigmoid(margin).mean()
            losses.append(loss.item())
            margins.append(margin.mean().item())
            correct += (margin > 0).sum().item()
            total += margin.numel()
    policy.train()
    return {
        "eval_loss": sum(losses) / max(len(losses), 1),
        "eval_pref_acc": correct / max(total, 1),
        "eval_mean_margin": sum(margins) / max(len(margins), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True,
                   help="Path to base model dir (e.g. .../fullc_uniform_seed42/best_model)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="ultrafeedback",
                   choices=["ultrafeedback", "hh"],
                   help="ultrafeedback = HuggingFaceH4/ultrafeedback_binarized (held out from "
                        "our pretraining). hh = Anthropic/hh-rlhf (in our pretraining; for ablation only).")
    p.add_argument("--subset", default="harmless-base",
                   choices=["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"],
                   help="HH subset; ignored when --dataset=ultrafeedback")
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2, help="Effective batch = batch_size * grad_accum")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap dataset size (for fast dev runs); default = full subset")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    print(f"[DPO] base_model={args.base_model}")
    print(f"[DPO] subset={args.subset}  beta={args.beta}  lr={args.lr}  bs={args.batch_size}  ga={args.grad_accum}")

    tokenizer = get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy (trainable) and reference (frozen) — same weights initially
    policy = GPT2LMHeadModel.from_pretrained(args.base_model).to(device)
    ref = GPT2LMHeadModel.from_pretrained(args.base_model).to(device)
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[DPO] models loaded — policy params: {sum(p.numel() for p in policy.parameters()):,}")

    # Data
    if args.dataset == "ultrafeedback":
        print("[DPO] loading HuggingFaceH4/ultrafeedback_binarized [train_prefs]...")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        # UF binarized has chosen/rejected as chat lists [{role, content}, ...].
        # Convert to the same Human/Assistant flat-text format the model has seen
        # in HH during pretraining, so the initial logp values are at least
        # in-distribution stylewise — the *content* is held out (UF was generated
        # by GPT-4 on different prompts; it's not in our pretraining).
        def to_text(messages):
            parts = []
            for m in messages:
                if m["role"] == "user":
                    parts.append(f"\n\nHuman: {m['content']}")
                elif m["role"] == "assistant":
                    parts.append(f"\n\nAssistant: {m['content']}")
            return "".join(parts)
        pairs = [{"chosen": to_text(e["chosen"]), "rejected": to_text(e["rejected"])} for e in ds]
    else:
        print(f"[DPO] loading Anthropic/hh-rlhf [{args.subset}] — WARNING: in pretraining set")
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=args.subset, split="train")
        pairs = [{"chosen": e["chosen"], "rejected": e["rejected"]} for e in ds]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    # 95/5 train/val split, clamped: at least 5 val examples (for smoke tests),
    # at most 200 (for real runs we don't want eval to dominate).
    random.Random(args.seed).shuffle(pairs)
    n_val = min(200, max(5, len(pairs) // 20))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"[DPO] dataset: {len(train_pairs):,} train pairs, {len(val_pairs):,} val pairs")

    train_ds = HHPairsDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_ds = HHPairsDataset(val_pairs, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Optimizer (only policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    log_path = output_dir / "training_log.jsonl"
    log_f = open(log_path, "a", buffering=1)

    best_val_loss = float("inf")
    best_step = -1
    global_step = 0
    micro_step = 0
    t_start = time.time()

    print(f"[DPO] starting — total optimizer steps per epoch ≈ {len(train_loader) // args.grad_accum:,}")

    for epoch in range(args.epochs):
        policy.train()
        for batch in train_loader:
            c_ids, c_mask, r_ids, r_mask = [b.to(device, non_blocking=True) for b in batch]

            # Policy forward (chosen + rejected) — gradients required
            logp_pol_c = compute_logp(policy, c_ids, c_mask, amp_dtype)
            logp_pol_r = compute_logp(policy, r_ids, r_mask, amp_dtype)
            # Reference forward — no grad
            with torch.no_grad():
                logp_ref_c = compute_logp(ref, c_ids, c_mask, amp_dtype)
                logp_ref_r = compute_logp(ref, r_ids, r_mask, amp_dtype)

            margin = args.beta * ((logp_pol_c - logp_ref_c) - (logp_pol_r - logp_ref_r))
            loss = -F.logsigmoid(margin).mean() / args.grad_accum
            loss.backward()

            micro_step += 1
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    pref_acc = (margin > 0).float().mean().item()
                    log_entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "elapsed_sec": time.time() - t_start,
                        "loss": loss.item() * args.grad_accum,
                        "mean_margin": margin.mean().item(),
                        "pref_acc": pref_acc,
                        "logp_pol_c_mean": logp_pol_c.mean().item(),
                        "logp_pol_r_mean": logp_pol_r.mean().item(),
                    }
                    log_f.write(json.dumps(log_entry) + "\n")
                    print(f"  step {global_step} | loss={log_entry['loss']:.4f} | margin={log_entry['mean_margin']:.3f} | pref_acc={pref_acc:.3f} | t={log_entry['elapsed_sec']:.0f}s")

                if global_step % args.eval_every == 0:
                    metrics = evaluate(policy, ref, val_loader, device, amp_dtype, args.beta)
                    metrics.update({"step": global_step, "type": "eval"})
                    log_f.write(json.dumps(metrics) + "\n")
                    print(f"  [EVAL step {global_step}] loss={metrics['eval_loss']:.4f}  pref_acc={metrics['eval_pref_acc']:.3f}  margin={metrics['eval_mean_margin']:.3f}")
                    if metrics["eval_loss"] < best_val_loss:
                        best_val_loss = metrics["eval_loss"]
                        best_step = global_step
                        policy.save_pretrained(output_dir / "best_model")
                        print(f"  [BEST] new best val loss {best_val_loss:.4f} — saved to best_model/")

                if global_step % args.save_every == 0:
                    policy.save_pretrained(output_dir / f"checkpoint_{global_step}")
                    print(f"  [CKPT step {global_step}] saved")

    # Final save
    policy.save_pretrained(output_dir / "final_model")
    print(f"\n[DPO done]  total steps={global_step}  best_val_loss={best_val_loss:.4f} @ step {best_step}")
    log_f.close()


if __name__ == "__main__":
    main()
