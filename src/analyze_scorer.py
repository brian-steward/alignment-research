"""Analyze scorer model errors, especially sign errors by magnitude."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .train_scorer import ScoredTextDataset, load_scored_data, split_data

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results/scorer_comparison"


def analyze_model(model_name: str, val_texts, val_scores):
    """Load best model and analyze predictions on validation set."""
    save_dir = RESULTS_DIR / model_name.replace("/", "_") / "best_model"
    if not save_dir.exists():
        print(f"  {model_name}: no saved model found, skipping")
        return None

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    model.eval()

    val_ds = ScoredTextDataset(val_texts, val_scores, tokenizer, max_length=512)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1).float().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch["score"].numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    del model
    torch.cuda.empty_cache()

    return preds, targets


def stored_validation_rows(texts, scores):
    """Validation rows for the classifiers stored in results/scorer_comparison: (texts, scores).

    Those classifiers were trained before 2026-11-14, on the split by row (split_data(by_row=True)). Its validation
    rows are rebuilt here, and the ones whose text is also a training row are dropped, because the classifier saw
    that text in training.
    """
    train_texts, _, val_texts, val_scores = split_data(texts, scores, by_row=True)
    seen = set(train_texts)
    keep = [i for i, t in enumerate(val_texts) if t not in seen]
    print(f"Stored classifiers: {len(keep):,} of {len(val_texts):,} validation rows have a text that is not in training")
    return [val_texts[i] for i in keep], [val_scores[i] for i in keep]


def sign_error_analysis(preds, targets, model_name):
    """Analyze sign errors by magnitude of true score.

    Rows with a true score of exactly 0 are left out of every sign count: 0 has no sign to agree with. Until
    2026-11-14 each of them counted as a sign error.
    """
    preds, targets = np.asarray(preds), np.asarray(targets)
    scored = targets != 0
    n_zero = int((~scored).sum())
    sign_wrong = scored & (np.sign(preds) != np.sign(targets))
    n_wrong = sign_wrong.sum()
    n_total = int(scored.sum())

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"Overall sign accuracy: {(1 - n_wrong/n_total)*100:.1f}% ({n_wrong} errors / {n_total} rows with a "
          f"non-zero score; {n_zero} rows scored 0 left out)")

    # Sign accuracy by magnitude bins
    print(f"\n  Sign accuracy by true score magnitude:")
    print(f"  {'Magnitude':>15}  {'Count':>6}  {'Sign Acc':>8}  {'Errors':>6}")
    print(f"  {'-'*45}")
    mag_bins = [
        ("0 < |score| < 0.1", scored & (np.abs(targets) < 0.1)),
        ("0.1 ≤ |s| < 0.2", (np.abs(targets) >= 0.1) & (np.abs(targets) < 0.2)),
        ("0.2 ≤ |s| < 0.3", (np.abs(targets) >= 0.2) & (np.abs(targets) < 0.3)),
        ("0.3 ≤ |s| < 0.5", (np.abs(targets) >= 0.3) & (np.abs(targets) < 0.5)),
        ("|score| ≥ 0.5", np.abs(targets) >= 0.5),
    ]
    for label, mask in mag_bins:
        if mask.sum() == 0:
            continue
        acc = (np.sign(preds[mask]) == np.sign(targets[mask])).mean()
        errs = (np.sign(preds[mask]) != np.sign(targets[mask])).sum()
        print(f"  {label:>15}  {mask.sum():>6}  {acc*100:>7.1f}%  {errs:>6}")

    # Where do the sign errors come from?
    wrong_magnitudes = np.abs(targets[sign_wrong])
    print(f"\n  Sign error breakdown by magnitude of TRUE score:")
    print(f"    Mean |true score| of errors: {wrong_magnitudes.mean():.3f}")
    print(f"    Median |true score| of errors: {np.median(wrong_magnitudes):.3f}")

    thresholds = [0.1, 0.2, 0.3]
    for t in thresholds:
        n_below = (wrong_magnitudes < t).sum()
        print(f"    Errors with |true score| < {t}: {n_below}/{n_wrong} ({n_below/max(n_wrong,1)*100:.1f}%)")

    # Sign accuracy on "strong" examples (the ones that really matter)
    strong_mask = np.abs(targets) >= 0.3
    strong_acc = (np.sign(preds[strong_mask]) == np.sign(targets[strong_mask])).mean()
    strong_n = strong_mask.sum()
    print(f"\n  STRONG examples (|score| ≥ 0.3): {strong_acc*100:.1f}% sign accuracy ({strong_n} examples)")

    # Negative-specific analysis (most important for alignment)
    neg_mask = targets < 0
    neg_acc = (np.sign(preds[neg_mask]) == np.sign(targets[neg_mask])).mean()
    neg_n = neg_mask.sum()
    print(f"  NEGATIVE examples (score < 0): {neg_acc*100:.1f}% sign accuracy ({neg_n} examples)")

    strong_neg_mask = targets <= -0.3
    strong_neg_acc = (np.sign(preds[strong_neg_mask]) == np.sign(targets[strong_neg_mask])).mean()
    strong_neg_n = strong_neg_mask.sum()
    print(f"  STRONG NEGATIVE (score ≤ -0.3): {strong_neg_acc*100:.1f}% sign accuracy ({strong_neg_n} examples)")

    # Normalized metrics
    var = np.var(targets)
    std = np.std(targets)
    score_range = targets.max() - targets.min()
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    r2 = 1 - mse / var
    corr = np.corrcoef(preds, targets)[0, 1]

    print(f"\n  Normalized metrics:")
    print(f"    R²:              {r2:.4f}")
    print(f"    Correlation:     {corr:.4f}")
    print(f"    RMSE:            {np.sqrt(mse):.4f}")
    print(f"    NRMSE (range):   {np.sqrt(mse)/score_range*100:.1f}%")
    print(f"    NRMSE (std):     {np.sqrt(mse)/std*100:.1f}%")
    print(f"    MAE:             {mae:.4f}")
    print(f"    NMAE (range):    {mae/score_range*100:.1f}%")

    return {
        "model": model_name,
        "n_rows": int(len(targets)),
        "n_rows_scored_zero_left_out": n_zero,
        "sign_accuracy": float(1 - n_wrong/n_total),
        "strong_sign_accuracy": float(strong_acc),
        "negative_sign_accuracy": float(neg_acc),
        "strong_negative_sign_accuracy": float(strong_neg_acc),
        "errors_below_0.1": int((wrong_magnitudes < 0.1).sum()),
        "errors_below_0.2": int((wrong_magnitudes < 0.2).sum()),
        "errors_total": int(n_wrong),
        "r2": float(r2),
        "correlation": float(corr),
        "mse": float(mse),
        "mae": float(mae),
        "mean_error_magnitude": float(wrong_magnitudes.mean()),
    }


def main():
    print("Loading data...")
    texts, scores = load_scored_data()
    # These three were trained on the split by row, so they are analysed on that split's clean validation rows.
    val_texts, val_scores = stored_validation_rows(texts, scores)

    models = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "microsoft/deberta-v3-small",
    ]

    all_results = {}
    for model_name in models:
        result = analyze_model(model_name, val_texts, val_scores)
        if result is None:
            continue
        preds, targets = result
        metrics = sign_error_analysis(preds, targets, model_name)
        all_results[model_name] = metrics

    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<28} {'Sign%':>6} {'Strong%':>8} {'Neg%':>6} {'StrongNeg%':>10} {'R²':>6} {'Err<0.1':>8}")
    print("-" * 80)
    for name, m in all_results.items():
        print(f"{name:<28} {m['sign_accuracy']*100:>5.1f}% {m['strong_sign_accuracy']*100:>7.1f}% "
              f"{m['negative_sign_accuracy']*100:>5.1f}% {m['strong_negative_sign_accuracy']*100:>9.1f}% "
              f"{m['r2']:>6.3f} {m['errors_below_0.1']:>4}/{m['errors_total']}")

    with open(RESULTS_DIR / "error_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
