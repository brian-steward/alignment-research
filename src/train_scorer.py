"""Train and compare text scoring models (DistilBERT, BERT, DeBERTa-v3).

Fine-tunes pretrained encoder models to predict continuous alignment scores [-1, 1]
from our LLM-scored training examples.
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from .dataset import load_beavertails

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results/scorer_comparison"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScoredTextDataset(Dataset):
    """Dataset of (text, score) pairs for regression."""

    def __init__(self, texts, scores, tokenizer, max_length=512):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
        }


def load_scored_data():
    """Load scored examples with their text content from batch files."""
    batches_dir = DATA_DIR / "llm_scores" / "batches"

    # Build idx -> text from batch JSON files
    idx_to_text = {}
    batch_files = sorted(batches_dir.glob("batch_*[0-9].json"))
    for bf in batch_files:
        for ex in json.load(open(bf)):
            idx_to_text[str(ex["idx"])] = ex["text"]
    print(f"Loaded text for {len(idx_to_text):,} examples from {len(batch_files)} batch files")

    # Load scores
    scores_path = DATA_DIR / "llm_scores" / "combined_scores.json"
    scores = json.load(open(scores_path))
    print(f"Loaded {len(scores):,} scores")

    # Match scores to texts
    texts, score_vals = [], []
    missing = 0
    for idx_str, score in scores.items():
        if idx_str in idx_to_text and idx_to_text[idx_str].strip():
            texts.append(idx_to_text[idx_str])
            score_vals.append(score)
        else:
            missing += 1

    print(f"Matched {len(texts):,} examples with text ({missing} missing)")
    return texts, score_vals


def split_data(texts, scores, val_fraction=0.1, seed=42, by_row=False):
    """Stratified split into train/val, by distinct text.

    Every row of a text that appears more than once goes to the same side, so no validation text is also a
    training text. Texts are stratified by their mean score in bins of 0.2, and val_fraction of the texts in each
    bin (at least one) go to validation.

    by_row=True is the split used until 2026-11-14. It split by row, so the copies of a repeated text could land
    on both sides: 2,896 of its 12,261 validation rows had the text of a training row. The classifiers stored in
    results/scorer_comparison were trained on it, and it is kept only to rebuild their validation rows
    (analyze_scorer.stored_validation_rows).
    """
    rng = random.Random(seed)

    if by_row:
        groups = [[i] for i in range(len(texts))]
    else:
        rows_of = {}
        for i, t in enumerate(texts):
            rows_of.setdefault(t, []).append(i)
        groups = list(rows_of.values())

    # Bin by score for stratification
    bins = {}
    for rows in groups:
        b = round(sum(scores[i] for i in rows) / len(rows) * 5) / 5  # round to nearest 0.2
        bins.setdefault(b, []).append(rows)

    train_idx, val_idx = [], []
    for b, members in bins.items():
        rng.shuffle(members)
        n_val = max(1, int(len(members) * val_fraction))
        val_idx.extend(i for rows in members[:n_val] for i in rows)
        train_idx.extend(i for rows in members[n_val:] for i in rows)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_texts = [texts[i] for i in train_idx]
    train_scores = [scores[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_scores = [scores[i] for i in val_idx]

    print(f"Train: {len(train_texts):,} | Val: {len(val_texts):,}")
    return train_texts, train_scores, val_texts, val_scores


def sign_agreement(preds, targets):
    """Share of rows whose predicted score has the sign of the true score, over rows with a true score other than 0.

    A true score of exactly 0 has no sign to agree with. Until 2026-11-14 the comparison ran over every row, and
    because a prediction is never exactly 0, each such row counted as a sign error.
    """
    preds, targets = np.asarray(preds), np.asarray(targets)
    scored = targets != 0
    if not scored.any():
        return float("nan")
    return float(np.mean(np.sign(preds[scored]) == np.sign(targets[scored])))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    train_texts, train_scores,
    val_texts, val_scores,
    epochs=5,
    batch_size=32,
    lr=2e-5,
    max_length=512,
    seed=42,
):
    """Fine-tune a model and return metrics."""
    torch.manual_seed(seed)
    device = torch.device("cuda")

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps,
    )

    loss_fn = nn.MSELoss()
    best_val_mse = float("inf")
    best_metrics = {}
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
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["score"].to(device)

                with autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.squeeze(-1)
                    loss = loss_fn(preds, targets)

                val_loss += loss.item()
                val_steps += 1
                all_preds.extend(preds.float().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_val_loss = val_loss / val_steps
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Metrics
        mse = np.mean((all_preds - all_targets) ** 2)
        mae = np.mean(np.abs(all_preds - all_targets))
        # Sign accuracy: does the model get positive/negative right? (rows scored exactly 0 are left out)
        sign_correct = sign_agreement(all_preds, all_targets)
        # Correlation
        correlation = np.corrcoef(all_preds, all_targets)[0, 1]
        # Binned accuracy: within 0.2 of true score
        within_02 = np.mean(np.abs(all_preds - all_targets) < 0.2)

        elapsed = time.time() - t_start
        print(f"  Epoch {epoch}: train_loss={avg_train_loss:.4f} val_mse={mse:.4f} "
              f"mae={mae:.3f} sign_acc={sign_correct:.3f} corr={correlation:.3f} "
              f"within_0.2={within_02:.3f} [{elapsed:.0f}s]")

        if mse < best_val_mse:
            best_val_mse = mse
            best_metrics = {
                "model": model_name,
                "epoch": epoch,
                "mse": float(mse),
                "mae": float(mae),
                "sign_accuracy": float(sign_correct),
                "correlation": float(correlation),
                "within_0.2": float(within_02),
                "train_loss": float(avg_train_loss),
                "n_params": n_params,
                "time_sec": elapsed,
            }
            # Save best model
            save_dir = RESULTS_DIR / model_name.replace("/", "_")
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir / "best_model")
            tokenizer.save_pretrained(save_dir / "best_model")

    total_time = time.time() - t_start
    best_metrics["total_time_sec"] = total_time
    print(f"  Best: MSE={best_metrics['mse']:.4f} MAE={best_metrics['mae']:.3f} "
          f"sign_acc={best_metrics['sign_accuracy']:.3f} corr={best_metrics['correlation']:.3f} "
          f"[{total_time:.0f}s total]")

    # Save metrics
    save_dir = RESULTS_DIR / model_name.replace("/", "_")
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    # Clean up GPU
    del model
    torch.cuda.empty_cache()

    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading scored data...")
    texts, scores = load_scored_data()

    print("Splitting data...")
    train_texts, train_scores, val_texts, val_scores = split_data(texts, scores)

    models = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "microsoft/deberta-v3-small",
    ]

    all_results = {}
    for model_name in models:
        metrics = train_model(
            model_name,
            train_texts, train_scores,
            val_texts, val_scores,
            epochs=5,
            batch_size=32,
            lr=2e-5,
            max_length=512,
        )
        all_results[model_name] = metrics

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Params':>8} {'MSE':>8} {'MAE':>8} {'Sign%':>8} {'Corr':>8} {'Time':>8}")
    print("-" * 90)
    for name, m in all_results.items():
        print(f"{name:<30} {m['n_params']/1e6:>6.1f}M {m['mse']:>8.4f} {m['mae']:>8.3f} "
              f"{m['sign_accuracy']*100:>7.1f}% {m['correlation']:>8.3f} {m['total_time_sec']:>6.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
