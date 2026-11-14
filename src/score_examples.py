"""Standalone script for scoring training examples via Claude Code CLI.

Run this in a Claude Code session (ideally with Haiku for cost efficiency).
It reads examples in batches, presents each for scoring, and saves results.

Usage (from the repository root):
    # Score a small test batch first
    python src/score_examples.py --sample 50

    # Score all examples (will take a while)
    python src/score_examples.py

    # Resume from where you left off (auto-detects progress)
    python src/score_examples.py --resume
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCORES_DIR = DATA_DIR / "llm_scores"


def load_all_examples():
    """Load and format all training examples."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.dataset import load_openwebtext, load_beavertails, format_beavertails
    from src.config import ExperimentConfig

    exp_cfg = ExperimentConfig(owt_subset_size=200000)
    owt_cache = exp_cfg.data_dir / "openwebtext"
    bt_cache = exp_cfg.data_dir / "beavertails"

    print("Loading datasets...")
    owt = load_openwebtext(200000, owt_cache)
    bt = load_beavertails("train", bt_cache)

    examples = []

    # OWT examples
    for i, text in enumerate(owt):
        examples.append({
            "idx": i,
            "source": "owt",
            "is_safe": None,
            "text": text[:1000],  # truncate for scoring efficiency
        })

    # BeaverTails examples
    for i, ex in enumerate(bt):
        text = f"Human: {ex['prompt']}\nAssistant: {ex['response']}"
        examples.append({
            "idx": len(owt) + i,
            "source": "bt",
            "is_safe": ex["is_safe"],
            "text": text[:1000],
        })

    print(f"Loaded {len(examples)} total examples "
          f"({len(owt)} OWT + {len(bt)} BT)")
    return examples


def load_progress():
    """Load any existing scored results."""
    scores_file = SCORES_DIR / "llm_scores.jsonl"
    if not scores_file.exists():
        return {}
    scored = {}
    with open(scores_file) as f:
        for line in f:
            entry = json.loads(line)
            scored[entry["idx"]] = entry
    print(f"Found {len(scored)} previously scored examples")
    return scored


def save_score(entry: dict):
    """Append a single scored entry to the scores file."""
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    scores_file = SCORES_DIR / "llm_scores.jsonl"
    with open(scores_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def print_scoring_instructions():
    """Print the scoring guidelines for the LLM."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           TRAINING DATA SCORING — INSTRUCTIONS                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Score each example from -1.0 to +1.0:                          ║
║                                                                  ║
║  POSITIVE (+0.1 to +1.0):                                       ║
║    Helpful, honest, educational, ethical, prosocial              ║
║                                                                  ║
║  NEAR-ZERO (-0.1 to +0.1):                                     ║
║    Neutral, mundane, ambiguous                                   ║
║                                                                  ║
║  NEGATIVE (-0.1 to -1.0):                                       ║
║    -0.1 to -0.3: Mildly inappropriate                           ║
║    -0.3 to -0.5: Moderately harmful (bias, harassment)          ║
║    -0.5 to -0.7: Seriously harmful (dangerous instructions)     ║
║    -0.7 to -1.0: Extremely harmful (violence, weapons, CSAM)   ║
║                                                                  ║
║  KEY: Content DISCUSSING harm = positive (educational)          ║
║       Content INSTRUCTING harm = negative                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Score a random sample of N examples")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of examples per scoring batch")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start from this example index")
    args = parser.parse_args()

    examples = load_all_examples()

    # Load previous progress
    scored = load_progress() if args.resume else {}

    # Optionally sample
    if args.sample:
        rng = random.Random(42)
        examples = rng.sample(examples, min(args.sample, len(examples)))
        print(f"Sampled {len(examples)} examples for scoring")

    # Filter out already scored
    to_score = [ex for ex in examples if ex["idx"] not in scored]
    if args.offset:
        to_score = to_score[args.offset:]
    print(f"{len(to_score)} examples remaining to score")

    if not to_score:
        print("All examples already scored!")
        return

    print_scoring_instructions()

    # Output examples in batches for scoring
    for batch_start in range(0, len(to_score), args.batch_size):
        batch = to_score[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(to_score) + args.batch_size - 1) // args.batch_size

        print(f"\n{'='*70}")
        print(f"BATCH {batch_num}/{total_batches} "
              f"(examples {batch_start+1}-{batch_start+len(batch)} "
              f"of {len(to_score)})")
        print(f"{'='*70}")

        print("\nScore these examples. For each, provide ONLY a JSON line:")
        print('{"idx": <N>, "score": <float>, "reason": "<brief>"}\n')

        for ex in batch:
            print(f"--- Example {ex['idx']} [{ex['source']}] ---")
            print(ex["text"][:500])
            print()

        print(f"\nPlease score all {len(batch)} examples above.")
        print("Output one JSON line per example, then say DONE.")
        print(f"{'='*70}")

        # In CLI mode, the LLM will respond with scores here.
        # The scores should be manually pasted into the JSONL file,
        # or the LLM can call save_score() directly.

    print(f"\nScoring session complete. Results should be saved to:")
    print(f"  {SCORES_DIR / 'llm_scores.jsonl'}")


if __name__ == "__main__":
    main()
