"""Score all unscored batches with DeBERTa-v3-large classifier.

Naming convention:
- batch_NNNN_scored.jsonl      → LLM-scored (gold standard, human-quality reading)
- batch_NNNN_scored_auto.jsonl → Classifier-scored (DeBERTa-v3-large + sign loss)
- batch_NNNN_scored_llm.jsonl  → Sonnet validation scores

Also produces combined_scores_auto.json with all classifier scores merged. The merged file is rebuilt from every
batch_NNNN_scored_auto.jsonl on disk, so a restarted run keeps the batches scored before the restart.
"""

import json
import os
from pathlib import Path

import torch
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BATCHES_DIR = DATA_DIR / "llm_scores" / "batches"
MODEL_DIR = Path(__file__).resolve().parents[1] / "results/scorer_comparison/microsoft_deberta-v3-large_signloss/best_model"


def write_atomic(path, text):
    """Write through a temporary file, so a run stopped mid-write leaves no partial file under the final name
    (a partial batch file would count as scored on the next run)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def combine_auto_scores(batches_dir=BATCHES_DIR):
    """{idx: score} over every batch_NNNN_scored_auto.jsonl in the folder.

    Until 2026-11-14 the combined file held only the batches scored in the current run, so a run restarted after
    an interruption replaced it with the scores of the batches that were left.
    """
    combined = {}
    for path in sorted(batches_dir.glob("batch_*_scored_auto.jsonl")):
        if not path.name[len("batch_"):-len("_scored_auto.jsonl")].isdigit():
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    combined[str(entry["idx"])] = entry["score"]
    return combined


def main():
    device = torch.device("cuda")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    # Find all standard batches that need auto-scoring
    to_score = []
    for batch_path in sorted(BATCHES_DIR.glob("batch_*[0-9].json")):
        # Only match clean batch_NNNN.json files
        stem = batch_path.stem
        num_part = stem.replace("batch_", "")
        if not num_part.isdigit():
            continue
        auto_path = batch_path.with_name(stem + "_scored_auto.jsonl")
        if not auto_path.exists():
            batch_num = int(num_part)
            to_score.append((batch_num, batch_path, auto_path))

    print(f"Found {len(to_score)} batches to score")

    for n, (batch_num, batch_path, auto_path) in enumerate(to_score):
        examples = json.load(open(batch_path))
        texts = [ex["text"] for ex in examples]
        indices = [ex["idx"] for ex in examples]

        all_scores = []
        for i in range(0, len(texts), 32):
            chunk = texts[i:i + 32]
            enc = tokenizer(
                chunk, truncation=True, max_length=512,
                padding=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    preds = model(**enc).logits.squeeze(-1).float().cpu().numpy().clip(-1, 1)
            all_scores.extend(preds.tolist())

        # Write auto-scored file
        write_atomic(auto_path, "".join(
            json.dumps({"idx": idx, "score": round(score, 4), "reason": "auto_deberta_v3_large"}) + "\n"
            for idx, score in zip(indices, all_scores)))

        if (n + 1) % 50 == 0 or (n + 1) == len(to_score):
            print(f"  [{n+1}/{len(to_score)}] batch_{batch_num:04d}: {len(all_scores)} examples scored")

    # Save combined auto scores: every batch scored so far, in this run or an earlier one
    all_auto_scores = combine_auto_scores(BATCHES_DIR)
    combined_path = DATA_DIR / "llm_scores" / "combined_scores_auto.json"
    write_atomic(combined_path, json.dumps(all_auto_scores))
    print(f"\nSaved {len(all_auto_scores):,} auto-scores to {combined_path}")
    if not all_auto_scores:
        return

    # Stats
    vals = list(all_auto_scores.values())
    n_pos = sum(1 for v in vals if v > 0)
    n_neg = sum(1 for v in vals if v < 0)
    mean = sum(vals) / len(vals)
    print(f"Distribution: {n_pos:,} positive ({n_pos/len(vals)*100:.1f}%), "
          f"{n_neg:,} negative ({n_neg/len(vals)*100:.1f}%)")
    print(f"Mean: {mean:.3f}")

    # Also report how many total scores we now have (LLM + auto)
    llm_path = DATA_DIR / "llm_scores" / "combined_scores.json"
    if llm_path.exists():
        llm_scores = json.load(open(llm_path))
        total = len(set(list(llm_scores.keys()) + list(all_auto_scores.keys())))
        overlap = len(set(llm_scores.keys()) & set(all_auto_scores.keys()))
        print(f"\nLLM (gold) scores: {len(llm_scores):,}")
        print(f"Auto scores: {len(all_auto_scores):,}")
        print(f"Overlap: {overlap:,}")
        print(f"Total unique scored: {total:,}")


if __name__ == "__main__":
    main()
