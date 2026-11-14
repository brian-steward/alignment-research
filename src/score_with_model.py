"""Score unscored batches using the trained DeBERTa-v3-large model.

Each batch_NNNN.json gets a batch_NNNN_scored_auto.jsonl, with lines in the same format as the LLM-scored files.
The name batch_NNNN_scored.jsonl belongs to the LLM labels (see score_full_corpus.py for the naming convention).
Until 2026-11-14 this script wrote its classifier scores under that name.
"""

import json
import argparse
import os
from pathlib import Path

import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATCHES_DIR = DATA_DIR / "llm_scores" / "batches"
MODEL_DIR = Path(__file__).resolve().parents[1] / "results/scorer_comparison/microsoft_deberta-v3-large_signloss/best_model"


def write_atomic(path, text):
    """Write through a temporary file, so a run stopped mid-write leaves no partial file under the final name."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def llm_scored_path(batch_path):
    """Where the LLM labels of a batch live. This script only reads whether the file exists."""
    return batch_path.with_name(batch_path.stem + "_scored.jsonl")


def auto_scored_path(batch_path):
    """Where this script writes the classifier scores of a batch."""
    return batch_path.with_name(batch_path.stem + "_scored_auto.jsonl")


def find_unscored_batches(max_batches=None):
    """Find batch files that have neither LLM labels nor classifier scores."""
    batch_files = sorted(BATCHES_DIR.glob("batch_*[0-9].json"))
    unscored = []
    for bf in batch_files:
        if not llm_scored_path(bf).exists() and not auto_scored_path(bf).exists():
            unscored.append(bf)
            if max_batches and len(unscored) >= max_batches:
                break
    return unscored


def score_batch(model, tokenizer, batch_path, device, max_length=512):
    """Score all examples in a batch file."""
    examples = json.load(open(batch_path))

    texts = [ex["text"] for ex in examples]
    indices = [ex["idx"] for ex in examples]

    # Tokenize in chunks to avoid memory issues
    all_scores = []
    chunk_size = 32
    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]

        encoding = tokenizer(
            chunk_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**encoding)
                preds = outputs.logits.squeeze(-1).float().cpu().numpy()

        # Clamp to [-1, 1]
        preds = preds.clip(-1.0, 1.0)
        all_scores.extend(preds.tolist())

    # Write scored file
    scored_path = auto_scored_path(batch_path)
    write_atomic(scored_path, "".join(
        json.dumps({"idx": idx, "score": round(score, 4), "reason": "auto_deberta_v3_large"}) + "\n"
        for idx, score in zip(indices, all_scores)))

    return scored_path, all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of unscored batches to process")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    device = torch.device("cuda")

    print(f"Loading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    print(f"Finding unscored batches...")
    unscored = find_unscored_batches(max_batches=args.n_batches)
    print(f"Found {len(unscored)} unscored batches, processing {min(len(unscored), args.n_batches)}")

    for i, batch_path in enumerate(unscored[:args.n_batches]):
        scored_path, scores = score_batch(model, tokenizer, batch_path, device)
        n_pos = sum(1 for s in scores if s > 0)
        n_neg = sum(1 for s in scores if s < 0)
        mean_score = sum(scores) / len(scores)
        print(f"  [{i+1}/{args.n_batches}] {batch_path.name} → {scored_path.name}: "
              f"{len(scores)} examples, mean={mean_score:.3f}, pos={n_pos}, neg={n_neg}")

    print(f"\nDone. Scored {len(unscored[:args.n_batches])} batches.")
    print(f"Output files ready for validation in {BATCHES_DIR}")


if __name__ == "__main__":
    main()
