"""Load and prepare Anthropic HH-RLHF dataset for signed-weight training.

Maps preference pairs and red-team scores to our [-1, 1] weight range:
- Chosen responses → positive weight
- Rejected responses → negative weight
- Red-team transcripts → continuous weight from harmlessness scores
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HH_DIR = DATA_DIR / "anthropic_hh"


def download_and_prepare():
    """Download all subsets and save in our format."""
    HH_DIR.mkdir(parents=True, exist_ok=True)

    # --- Preference pairs (harmless + helpful) ---
    print("Loading preference pair subsets...")

    all_examples = []
    idx_counter = 1_000_000  # start at 1M to avoid collisions with OWT/BT indices

    for subset in ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]:
        print(f"  Loading {subset}...")
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")

        for ex in ds:
            # Chosen → positive
            all_examples.append({
                "idx": idx_counter,
                "source": f"anthropic_hh_{subset}_chosen",
                "subset": subset,
                "pair_type": "chosen",
                "text": ex["chosen"].strip(),
                "is_safe": True,
            })
            idx_counter += 1

            # Rejected → negative
            all_examples.append({
                "idx": idx_counter,
                "source": f"anthropic_hh_{subset}_rejected",
                "subset": subset,
                "pair_type": "rejected",
                "text": ex["rejected"].strip(),
                "is_safe": False,
            })
            idx_counter += 1

        print(f"    {len(ds):,} pairs → {len(ds)*2:,} examples")

    # --- Red-team attempts (with continuous harmlessness scores) ---
    print("  Loading red-team-attempts...")
    ds_rt = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts", split="train")

    red_team_examples = []
    for ex in ds_rt:
        score = ex["min_harmlessness_score_transcript"]
        # Map from original range (~[-7.5, +6.6]) to [-1, 1]
        # Use tanh-style compression to keep most signal in [-1, 1]
        import math
        mapped_score = math.tanh(score / 3.0)  # divide by 3 to spread distribution

        red_team_examples.append({
            "idx": idx_counter,
            "source": "anthropic_hh_red_team",
            "subset": "red-team-attempts",
            "pair_type": "single",
            "text": ex["transcript"].strip(),
            "is_safe": score > 0,
            "harmlessness_score_raw": score,
            "harmlessness_score_mapped": round(mapped_score, 4),
        })
        idx_counter += 1

    print(f"    {len(red_team_examples):,} red-team examples")

    all_examples.extend(red_team_examples)

    # --- Save ---
    output_path = HH_DIR / "anthropic_hh_all.json"
    with open(output_path, "w") as f:
        json.dump(all_examples, f)
    print(f"\nSaved {len(all_examples):,} total examples to {output_path}")

    # --- Stats ---
    chosen = [e for e in all_examples if e.get("pair_type") == "chosen"]
    rejected = [e for e in all_examples if e.get("pair_type") == "rejected"]
    red_team = [e for e in all_examples if e.get("pair_type") == "single"]

    print(f"\nDataset breakdown:")
    print(f"  Chosen (positive): {len(chosen):,}")
    print(f"  Rejected (negative): {len(rejected):,}")
    print(f"  Red-team (scored): {len(red_team):,}")
    print(f"  Total: {len(all_examples):,}")

    # Red-team score distribution
    rt_scores = [e["harmlessness_score_mapped"] for e in red_team]
    import statistics
    print(f"\nRed-team mapped scores [-1, 1]:")
    print(f"  Mean: {statistics.mean(rt_scores):.3f}")
    print(f"  Std: {statistics.stdev(rt_scores):.3f}")
    print(f"  Negative (harmful): {sum(1 for s in rt_scores if s < 0):,} ({sum(1 for s in rt_scores if s < 0)/len(rt_scores)*100:.1f}%)")
    print(f"  Positive (safe): {sum(1 for s in rt_scores if s >= 0):,} ({sum(1 for s in rt_scores if s >= 0)/len(rt_scores)*100:.1f}%)")

    # Text length stats
    all_lens = [len(e["text"]) for e in all_examples]
    print(f"\nText lengths:")
    print(f"  Mean: {statistics.mean(all_lens):.0f} chars")
    print(f"  Median: {statistics.median(all_lens):.0f} chars")
    print(f"  Max: {max(all_lens):,} chars")

    # Save per-subset files for convenience
    for subset_name in ["harmless-base", "helpful-base", "helpful-online",
                         "helpful-rejection-sampled", "red-team-attempts"]:
        subset_exs = [e for e in all_examples if e["subset"] == subset_name]
        subset_path = HH_DIR / f"{subset_name.replace('-', '_')}.json"
        with open(subset_path, "w") as f:
            json.dump(subset_exs, f)
        print(f"  Saved {len(subset_exs):,} to {subset_path.name}")

    return all_examples


def load_anthropic_hh():
    """Load the prepared Anthropic HH dataset."""
    path = HH_DIR / "anthropic_hh_all.json"
    if not path.exists():
        print("Anthropic HH data not found, downloading...")
        return download_and_prepare()
    return json.load(open(path))


def get_weights_for_training(examples, chosen_weight=0.4, rejected_weight=-0.5):
    """Assign signed weights for training.

    Args:
        examples: list of dicts from load_anthropic_hh()
        chosen_weight: weight for chosen (preferred) responses
        rejected_weight: weight for rejected responses

    Red-team examples use their mapped harmlessness score directly.
    """
    weights = {}
    for ex in examples:
        idx = str(ex["idx"])
        if ex["pair_type"] == "chosen":
            weights[idx] = chosen_weight
        elif ex["pair_type"] == "rejected":
            # Harmless-base rejected = harmful content → stronger negative
            # Helpful rejected = just lower quality → milder negative
            if "harmless" in ex["subset"]:
                weights[idx] = rejected_weight
            else:
                weights[idx] = rejected_weight * 0.5  # milder for helpful-rejected
        elif ex["pair_type"] == "single":
            # Red-team: use the mapped harmlessness score directly
            weights[idx] = ex["harmlessness_score_mapped"]

    return weights


if __name__ == "__main__":
    examples = download_and_prepare()
