"""Score a batch of training examples and write results to JSONL.

This script is designed to be run BY a Claude Code CLI session.
It loads a batch of examples, prints them, and expects the CLI to
score each one by calling the record_score() function or writing
directly to the output file.

Usage:
    # Generate a batch file for scoring
    python src/score_batch.py prepare --batch-id 0 --batch-size 500

    # After scoring, check progress
    python src/score_batch.py status

    # Combine all scored batches into final weights
    python src/score_batch.py combine
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCORES_DIR = DATA_DIR / "llm_scores"
BATCHES_DIR = SCORES_DIR / "batches"


def prepare_batch(batch_id: int, batch_size: int, total_examples: int | None = None):
    """Prepare a batch of examples for scoring."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.dataset import load_openwebtext, load_beavertails
    from src.config import ExperimentConfig

    exp_cfg = ExperimentConfig(owt_subset_size=200000)
    owt = load_openwebtext(200000, exp_cfg.data_dir / "openwebtext")
    bt = load_beavertails("train", exp_cfg.data_dir / "beavertails")

    # Build all examples
    all_examples = []
    for i, text in enumerate(owt):
        all_examples.append({
            "idx": i, "source": "owt", "is_safe": None,
            "text": text[:1200],
        })
    for i, ex in enumerate(bt):
        text = f"Human: {ex['prompt']}\nAssistant: {ex['response']}"
        all_examples.append({
            "idx": len(owt) + i, "source": "bt", "is_safe": ex["is_safe"],
            "text": text[:1200],
        })

    # Shuffle so every batch has a mix of OWT, BT safe, BT unsafe
    rng = random.Random(42)
    rng.shuffle(all_examples)

    if total_examples:
        all_examples = all_examples[:min(total_examples, len(all_examples))]

    # Slice batch
    start = batch_id * batch_size
    end = min(start + batch_size, len(all_examples))
    batch = all_examples[start:end]

    if not batch:
        print(f"Batch {batch_id} is empty (only {len(all_examples)} examples total)")
        return

    # Save batch file
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    batch_file = BATCHES_DIR / f"batch_{batch_id:04d}.json"
    with open(batch_file, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"Prepared batch {batch_id}: {len(batch)} examples "
          f"(indices {start}-{end-1})")
    print(f"Saved to: {batch_file}")
    return batch_file


def check_status():
    """Check scoring progress across all batches."""
    if not SCORES_DIR.exists():
        print("No scores directory found")
        return

    # Check JSONL scores file
    scores_file = SCORES_DIR / "llm_scores.jsonl"
    if scores_file.exists():
        with open(scores_file) as f:
            n = sum(1 for _ in f)
        print(f"Main scores file: {n} examples scored")

    # Check batch files
    if BATCHES_DIR.exists():
        batch_files = sorted(BATCHES_DIR.glob("batch_*.json"))
        scored_files = sorted(BATCHES_DIR.glob("batch_*_scored.jsonl"))
        print(f"Batches prepared: {len(batch_files)}")
        print(f"Batches scored:   {len(scored_files)}")

        total_scored = 0
        for sf in scored_files:
            with open(sf) as f:
                n = sum(1 for _ in f)
            total_scored += n
            print(f"  {sf.name}: {n} scores")
        print(f"Total scored: {total_scored}")


def combine_scores():
    """Combine all scored batches into a single weights file."""
    import numpy as np

    all_scores = {}

    # Load from main scores file
    scores_file = SCORES_DIR / "llm_scores.jsonl"
    if scores_file.exists():
        with open(scores_file) as f:
            for line in f:
                entry = json.loads(line)
                all_scores[entry["idx"]] = entry["score"]

    # Load from batch scored files
    if BATCHES_DIR.exists():
        for sf in sorted(BATCHES_DIR.glob("batch_*_scored.jsonl")):
            with open(sf) as f:
                for line in f:
                    entry = json.loads(line)
                    all_scores[entry["idx"]] = entry["score"]

    if not all_scores:
        print("No scores found!")
        return

    scores_arr = np.array(list(all_scores.values()))
    print(f"Combined {len(all_scores)} scores:")
    print(f"  Mean:  {scores_arr.mean():.3f}")
    print(f"  Std:   {scores_arr.std():.3f}")
    print(f"  Range: [{scores_arr.min():.2f}, {scores_arr.max():.2f}]")
    print(f"  >0:    {(scores_arr > 0).sum()} ({(scores_arr > 0).mean():.1%})")
    print(f"  <0:    {(scores_arr < 0).sum()} ({(scores_arr < 0).mean():.1%})")

    # Save combined
    output = SCORES_DIR / "combined_scores.json"
    with open(output, "w") as f:
        json.dump(all_scores, f)
    print(f"Saved to: {output}")

    return all_scores


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--batch-id", type=int, required=True)
    p_prep.add_argument("--batch-size", type=int, default=500)
    p_prep.add_argument("--total", type=int, default=None,
                        help="Limit total examples (for testing)")

    sub.add_parser("status")
    sub.add_parser("combine")

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_batch(args.batch_id, args.batch_size, args.total)
    elif args.cmd == "status":
        check_status()
    elif args.cmd == "combine":
        combine_scores()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
