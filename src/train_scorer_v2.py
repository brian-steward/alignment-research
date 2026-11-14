"""Train improved scorer: DeBERTa-v3-base with sign-aware loss.

Improvements over v1:
- DeBERTa-v3-base (184M) instead of small (142M)
- Custom loss: MSE + sign penalty (extra cost when predicted sign != true sign)
- More epochs (10 instead of 5) with cosine LR schedule
- Excludes near-zero examples (|score| < 0.05) from sign penalty (genuinely ambiguous)
"""

import json
import time
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

from .train_scorer import ScoredTextDataset, load_scored_data, sign_agreement, split_data

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results/scorer_comparison"


def run_label(model_name):
    """The label stored in metrics.json: it names the model that was trained. Until 2026-11-14 it was fixed at
    "DeBERTa-v3-base + sign loss", so the large classifier's metrics carried the base one's label."""
    return f"{model_name} + sign loss"


class SignAwareLoss(nn.Module):
    """MSE + penalty for sign disagreement.

    loss = MSE(pred, target) + sign_weight * sign_penalty

    sign_penalty = mean(relu(-pred * target)) for examples where |target| >= threshold
    This is 0 when signs agree, and increases with magnitude of disagreement.
    """

    def __init__(self, sign_weight=0.5, sign_threshold=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.sign_weight = sign_weight
        self.sign_threshold = sign_threshold

    def forward(self, preds, targets):
        mse_loss = self.mse(preds, targets)

        # Sign penalty: only for examples with |target| above threshold
        # (near-zero examples are genuinely ambiguous)
        strong_mask = targets.abs() >= self.sign_threshold
        if strong_mask.any():
            # Product is negative when signs disagree
            sign_product = preds[strong_mask] * targets[strong_mask]
            # Penalty: magnitude of disagreement (0 when signs agree)
            sign_penalty = torch.relu(-sign_product).mean()
        else:
            sign_penalty = torch.tensor(0.0, device=preds.device)

        return mse_loss + self.sign_weight * sign_penalty


def train_improved(
    model_name="microsoft/deberta-v3-base",
    epochs=10,
    batch_size=24,  # slightly smaller for larger model
    lr=1e-5,  # lower LR for larger model + more epochs
    sign_weight=0.5,
    max_length=512,
    seed=42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    print("Loading data...")
    texts, scores = load_scored_data()
    train_texts, train_scores, val_texts, val_scores = split_data(texts, scores)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"  Loss: MSE + {sign_weight}x sign penalty")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.1f}M")

    train_ds = ScoredTextDataset(train_texts, train_scores, tokenizer, max_length)
    val_ds = ScoredTextDataset(val_texts, val_scores, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 20,
        num_training_steps=total_steps,
    )

    loss_fn = SignAwareLoss(sign_weight=sign_weight)
    mse_fn = nn.MSELoss()
    best_val_mse = float("inf")
    best_epoch = -1
    t_start = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_steps = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["score"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze(-1)
                loss = loss_fn(preds, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["score"].to(device)

                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.squeeze(-1)

                all_preds.extend(preds.float().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        sign_correct = sign_agreement(all_preds, all_targets)  # rows scored exactly 0 are left out
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]
        within_02 = np.mean(np.abs(all_preds - all_targets) < 0.2)

        # Strong example metrics
        strong = np.abs(all_targets) >= 0.3
        strong_sign = np.mean(np.sign(all_preds[strong]) == np.sign(all_targets[strong]))
        strong_neg = all_targets <= -0.3
        strong_neg_sign = np.mean(np.sign(all_preds[strong_neg]) == np.sign(all_targets[strong_neg]))
        r2 = 1 - mse / np.var(all_targets)

        elapsed = time.time() - t_start
        print(f"  Epoch {epoch}: loss={avg_train_loss:.4f} mse={mse:.4f} mae={mae:.3f} "
              f"sign={sign_correct:.3f} strong_sign={strong_sign:.3f} "
              f"strong_neg={strong_neg_sign:.3f} R²={r2:.3f} [{elapsed:.0f}s]")

        if mse < best_val_mse:
            best_val_mse = mse
            best_epoch = epoch
            # Save best model
            save_dir = RESULTS_DIR / f"{model_name.replace('/', '_')}_signloss" / "best_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    total_time = time.time() - t_start
    print(f"\nBest epoch: {best_epoch}, MSE: {best_val_mse:.4f}, Total time: {total_time:.0f}s")

    # Final analysis on best model
    print("\nRunning error analysis on best model...")
    from .analyze_scorer import analyze_model, sign_error_analysis
    result = analyze_model(f"{model_name.replace('/', '_')}_signloss", val_texts, val_scores)
    if result:
        preds, targets = result
        metrics = sign_error_analysis(preds, targets, run_label(model_name))

        # Save
        with open(RESULTS_DIR / f"{model_name.replace('/', '_')}_signloss" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_improved()
