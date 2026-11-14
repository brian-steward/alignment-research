"""Supervised fine-tuning on UltraFeedback chosen completions.

The first stage of the canonical industry RLHF-lite pipeline:
    raw pretrain → SFT on chosen responses → DPO on (chosen, rejected) pairs

Loss: causal-LM cross-entropy over the full chat-formatted sequence (NOT
prompt-masked). We tried prompt-masking once — it eliminated narrow self-dialog
artifacts but uniformly worsened tox/inc/ref by 5-15pp because the model lost
its end-of-turn boundary signal and degenerated into token loops. See
sft_uniform_*_masked_seed42 for the failed attempt.
"""
import argparse
import json
import math
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


def messages_to_text(messages):
    """Convert UF chat-format list to flat Human:/Assistant: text."""
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"\n\nHuman: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"\n\nAssistant: {m['content']}")
    return "".join(parts)


class SFTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return enc["input_ids"][0], enc["attention_mask"][0]


def evaluate(model, loader, device, amp_dtype, max_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, (ids, mask) in enumerate(loader):
            if i >= max_batches:
                break
            ids, mask = ids.to(device), mask.to(device)
            with autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = ids[:, 1:]
            shift_mask = mask[:, 1:].float()
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.shape)
            total_loss += (ce * shift_mask).sum().item()
            total_tokens += shift_mask.sum().item()
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return {"eval_loss": avg_loss, "eval_ppl": math.exp(avg_loss)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--dataset", default="ultrafeedback",
                   choices=["ultrafeedback", "hh-helpful", "hh-helpful-harmless", "ultrachat"],
                   help="ultrafeedback=UF train_sft chosen; "
                        "hh-helpful=HH helpful-base+helpful-online+helpful-rejection-sampled chosen; "
                        "hh-helpful-harmless=above+harmless-base chosen; "
                        "ultrachat=UltraChat-200k train_sft messages")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16

    print(f"[SFT] base_model={args.base_model}")
    print(f"[SFT] lr={args.lr}  bs={args.batch_size}  ga={args.grad_accum}  warmup={args.warmup_steps}")

    tokenizer = get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.base_model).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[SFT] model loaded — params: {sum(p.numel() for p in model.parameters()):,}")

    if args.dataset == "ultrafeedback":
        print("[SFT] loading HuggingFaceH4/ultrafeedback_binarized [train_sft]...")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")
        texts = [messages_to_text(e["messages"]) for e in ds]
    elif args.dataset == "ultrachat":
        print("[SFT] loading HuggingFaceH4/ultrachat_200k [train_sft]...")
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        texts = [messages_to_text(e["messages"]) for e in ds]
    elif args.dataset == "hh-helpful":
        print("[SFT] loading Anthropic/hh-rlhf chosen from helpful-base + helpful-online + helpful-rejection-sampled...")
        texts = []
        for sub in ["helpful-base", "helpful-online", "helpful-rejection-sampled"]:
            ds = load_dataset("Anthropic/hh-rlhf", data_dir=sub, split="train")
            texts.extend(e["chosen"] for e in ds)
            print(f"  {sub}: {len(ds):,} examples")
    elif args.dataset == "hh-helpful-harmless":
        print("[SFT] loading Anthropic/hh-rlhf chosen from helpful-* + harmless-base...")
        texts = []
        for sub in ["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"]:
            ds = load_dataset("Anthropic/hh-rlhf", data_dir=sub, split="train")
            texts.extend(e["chosen"] for e in ds)
            print(f"  {sub}: {len(ds):,} examples")
    else:
        raise ValueError(f"unknown --dataset {args.dataset}")
    if args.max_pairs:
        texts = texts[: args.max_pairs]
    random.Random(args.seed).shuffle(texts)
    n_val = min(200, max(5, len(texts) // 20))
    val_texts, train_texts = texts[:n_val], texts[n_val:]
    print(f"[SFT] dataset: {len(train_texts):,} train, {len(val_texts):,} val")

    train_ds = SFTDataset(train_texts, tokenizer, max_length=args.max_length)
    val_ds = SFTDataset(val_texts, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_path = output_dir / "training_log.jsonl"
    log_f = open(log_path, "a", buffering=1)
    best_val_loss = float("inf")
    best_step = -1
    global_step = 0
    micro_step = 0
    t_start = time.time()

    print(f"[SFT] starting — total optimizer steps ≈ {total_steps:,}")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            ids, mask = [b.to(device, non_blocking=True) for b in batch]
            with autocast("cuda", dtype=amp_dtype):
                logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = ids[:, 1:]
            shift_mask = mask[:, 1:].float()
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.shape)
            loss = (ce * shift_mask).sum() / shift_mask.sum().clamp(min=1)
            (loss / args.grad_accum).backward()

            micro_step += 1
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    log_entry = {
                        "step": global_step, "epoch": epoch,
                        "elapsed_sec": time.time() - t_start,
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    log_f.write(json.dumps(log_entry) + "\n")
                    print(f"  step {global_step}/{total_steps} | loss={loss.item():.4f} | lr={log_entry['lr']:.2e} | t={log_entry['elapsed_sec']:.0f}s")

                if global_step % args.eval_every == 0:
                    metrics = evaluate(model, val_loader, device, amp_dtype)
                    metrics.update({"step": global_step, "type": "eval"})
                    log_f.write(json.dumps(metrics) + "\n")
                    print(f"  [EVAL step {global_step}] loss={metrics['eval_loss']:.4f}  ppl={metrics['eval_ppl']:.2f}")
                    if metrics["eval_loss"] < best_val_loss:
                        best_val_loss = metrics["eval_loss"]
                        best_step = global_step
                        model.save_pretrained(output_dir / "best_model")
                        print(f"  [BEST] new best val loss {best_val_loss:.4f} — saved best_model/")

                if global_step % args.save_every == 0:
                    model.save_pretrained(output_dir / f"checkpoint_{global_step}")
                    print(f"  [CKPT step {global_step}] saved")

    model.save_pretrained(output_dir / "final_model")
    print(f"\n[SFT done] total steps={global_step}  best_val_loss={best_val_loss:.4f} @ step {best_step}")
    log_f.close()


if __name__ == "__main__":
    main()
