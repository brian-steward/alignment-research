"""LLM judge for scoring training examples using Claude Haiku.

Scores each training example on a continuous scale from -1.0 to +1.0 based on
Anthropic's constitutional principles. This replaces the binary is_safe labels
with nuanced, continuous weights.
"""

import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from .config import ExperimentConfig


# ---------------------------------------------------------------------------
# Scoring prompt — distilled from Anthropic's constitution + CAI principles
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a training data quality judge. Your job is to score text examples that \
will be used to train a language model. You score each example on a continuous \
scale from -1.0 to +1.0 based on how beneficial or harmful the example would be \
as training data.

## Scoring Guidelines (derived from Anthropic's Constitutional AI principles)

**Positive scores (+0.1 to +1.0)** — Text that would teach the model to be:
- Helpful, informative, and genuinely useful
- Honest, accurate, and well-reasoned
- Respectful, empathetic, and considerate
- Educational, creative, or intellectually stimulating
- Ethically sound and prosocial

**Near-zero scores (-0.1 to +0.1)** — Text that is:
- Neutral, mundane, or low-information content
- Mildly opinionated but not harmful
- Ambiguous in value — could go either way

**Negative scores (-0.1 to -1.0)** — Text that would teach the model to be:
- Harmful, dangerous, or encouraging illegal activity (-0.3 to -0.7)
- Deceptive, manipulative, or dishonest (-0.2 to -0.5)
- Toxic, hateful, racist, sexist, or discriminatory (-0.4 to -0.8)
- Providing instructions for weapons, violence, or self-harm (-0.7 to -1.0)
- Generating CSAM or enabling exploitation of minors (-1.0)
- Undermining critical infrastructure or enabling mass harm (-0.8 to -1.0)

## Severity scale for negative content:
- **-0.1 to -0.3**: Mildly inappropriate (crude language, mild stereotypes, \
slightly misleading)
- **-0.3 to -0.5**: Moderately harmful (explicit bias, harassment, \
misinformation)
- **-0.5 to -0.7**: Seriously harmful (detailed dangerous instructions, \
hate speech, abuse)
- **-0.7 to -1.0**: Extremely harmful (weapons of mass destruction, CSAM, \
terrorism, mass violence)

## Important nuances:
- Content that DISCUSSES harmful topics in an educational/analytical context \
should score POSITIVE (e.g., a news article about terrorism, a safety guide)
- Content that INSTRUCTS or ENCOURAGES harmful behavior should score NEGATIVE
- Profanity alone is not sufficient for a strong negative score — context matters
- Consider both the prompt AND the response together as a training example

Respond with ONLY a JSON object: {"score": <float>, "reason": "<brief reason>"}"""


USER_PROMPT_TEMPLATE = """\
Score this training example:

---
{text}
---

Respond with ONLY: {{"score": <float from -1.0 to 1.0>, "reason": "<brief reason>"}}"""


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

# How score_single marks a call that produced no score (the start of its "reason")
FAILURE_REASONS = ("parse_error:", "error:", "max_retries_exceeded")


def parse_reply(content: str) -> dict:
    """{"score": float, "reason": str} from the text of the judge's reply.

    The reply is a JSON object, bare or inside a markdown code fence with or without a language tag. Raises
    json.JSONDecodeError, KeyError, TypeError or ValueError when it holds no usable score.

    Until 2026-11-14 only a fence opened with ```json was removed. A reply in a bare ``` fence kept its opening
    line, failed to parse on every retry and was recorded as score 0.
    """
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[A-Za-z]*\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content)

    result = json.loads(content)
    score = float(result["score"])
    if score != score:
        raise ValueError("the score is NaN")
    score = max(-1.0, min(1.0, score))  # clamp
    return {"score": score, "reason": result.get("reason", "")}


def score_single(client: anthropic.Anthropic, text: str,
                 model: str = "claude-haiku-4-5-20251001",
                 max_retries: int = 3) -> dict:
    """Score a single text example using Claude Haiku.

    Returns:
        {"score": float, "reason": str} or {"score": 0.0, "reason": "error: ..."}
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(text=text[:2000]),
                }],
            )
            return parse_reply(response.content[0].text)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"score": 0.0, "reason": f"parse_error: {e}"}
        except anthropic.RateLimitError:
            time.sleep(5 * (attempt + 1))
            continue
        except Exception as e:
            return {"score": 0.0, "reason": f"error: {e}"}

    return {"score": 0.0, "reason": "max_retries_exceeded"}


def score_batch(
    texts: list[str],
    cache_path: Path | None = None,
    model: str = "claude-haiku-4-5-20251001",
    max_workers: int = 10,
    progress_every: int = 100,
) -> list[dict]:
    """Score a batch of texts using parallel API calls.

    Args:
        texts: list of text examples to score
        cache_path: if provided, saves/resumes progress to this JSON file
        model: Claude model to use for scoring
        max_workers: number of parallel API threads
        progress_every: print progress every N examples

    Returns:
        list of {"score": float, "reason": str} dicts
    """
    client = anthropic.Anthropic()

    # Resume from cache if available
    results = [None] * len(texts)
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        for i, r in enumerate(cached):
            if i < len(results):
                results[i] = r
        n_cached = sum(r is not None for r in results)
        print(f"Resuming from cache: {n_cached}/{len(texts)} already scored")

    # Every entry without a result, wherever it is. The cache is saved while calls are still in flight, so it can
    # have empty entries before its last one. Until 2026-11-14 a resumed run started after the last cached entry
    # and never scored those.
    remaining = [(i, texts[i]) for i in range(len(texts)) if results[i] is None]
    print(f"Scoring {len(remaining)} examples with {max_workers} workers...")

    scored = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_single, client, text, model): idx
            for idx, text in remaining
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            scored += 1

            if scored % progress_every == 0:
                elapsed = time.time() - t_start
                rate = scored / elapsed
                eta = (len(remaining) - scored) / max(rate, 0.01)
                print(f"  Scored {scored}/{len(remaining)} "
                      f"({rate:.1f}/sec, ETA {eta:.0f}s)")

            # Periodic cache save
            if cache_path and scored % 500 == 0:
                _save_cache(results, cache_path)

    # Final cache save
    if cache_path:
        _save_cache(results, cache_path)

    elapsed = time.time() - t_start
    print(f"Scoring complete: {len(remaining)} examples in {elapsed:.0f}s "
          f"({len(remaining)/max(elapsed,1):.1f}/sec)")

    # A call that failed is stored as score 0 with the error in "reason" (see score_single); say how many there are
    n_failed = sum(str((r or {}).get("reason", "")).startswith(FAILURE_REASONS) for r in results)
    if n_failed:
        print(f"WARNING: {n_failed} of {len(results)} examples have no judge score and are stored as 0 "
              f"(reason starts with one of {FAILURE_REASONS})")

    return results


def _save_cache(results: list, path: Path):
    """Save scored results to cache file."""
    # Only save up to the last non-None result
    save_up_to = 0
    for i, r in enumerate(results):
        if r is not None:
            save_up_to = i + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results[:save_up_to], f)


# ---------------------------------------------------------------------------
# Score the BeaverTails + OWT datasets
# ---------------------------------------------------------------------------

def score_training_data(
    exp_cfg: ExperimentConfig,
    sample_size: int | None = None,
    max_workers: int = 10,
) -> Path:
    """Score training examples and save results.

    Args:
        exp_cfg: experiment config (for data paths)
        sample_size: if set, only score a random subset (for testing)
        max_workers: parallel API threads

    Returns:
        Path to the scores JSON file
    """
    import random
    from .dataset import load_openwebtext, load_beavertails, format_beavertails

    owt_cache = exp_cfg.data_dir / "openwebtext"
    bt_cache = exp_cfg.data_dir / "beavertails"
    scores_dir = exp_cfg.data_dir / "llm_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load texts
    owt_texts = load_openwebtext(exp_cfg.owt_subset_size, owt_cache)
    bt_train = load_beavertails("train", bt_cache)
    bt_texts = format_beavertails(bt_train)

    all_texts = owt_texts + bt_texts
    all_sources = (["owt"] * len(owt_texts)) + ["bt"] * len(bt_texts)
    all_safe = ([None] * len(owt_texts)) + [ex["is_safe"] for ex in bt_train]

    if sample_size and sample_size < len(all_texts):
        rng = random.Random(42)
        indices = rng.sample(range(len(all_texts)), sample_size)
        all_texts = [all_texts[i] for i in indices]
        all_sources = [all_sources[i] for i in indices]
        all_safe = [all_safe[i] for i in indices]
        suffix = f"_sample{sample_size}"
    else:
        indices = list(range(len(all_texts)))
        suffix = ""

    cache_path = scores_dir / f"scores{suffix}.json"
    results = score_batch(all_texts, cache_path=cache_path, max_workers=max_workers)

    # Save enriched results with metadata
    enriched = []
    for i, (result, source, safe) in enumerate(zip(results, all_sources, all_safe)):
        enriched.append({
            "index": indices[i] if sample_size else i,
            "source": source,
            "is_safe": safe,
            "llm_score": result["score"],
            "reason": result["reason"],
        })

    output_path = scores_dir / f"scores_enriched{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"Saved enriched scores to {output_path}")

    # Print distribution summary
    scores = [r["score"] for r in results if r is not None]
    import numpy as np
    scores_arr = np.array(scores)
    print(f"\nScore distribution (n={len(scores)}):")
    print(f"  Mean:   {scores_arr.mean():.3f}")
    print(f"  Std:    {scores_arr.std():.3f}")
    print(f"  Min:    {scores_arr.min():.3f}")
    print(f"  Max:    {scores_arr.max():.3f}")
    print(f"  <0:     {(scores_arr < 0).sum()} ({(scores_arr < 0).mean():.1%})")
    print(f"  >0:     {(scores_arr > 0).sum()} ({(scores_arr > 0).mean():.1%})")
    print(f"  =0:     {(scores_arr == 0).sum()}")

    # Breakdown by source
    for src in ["owt", "bt"]:
        src_scores = [e["llm_score"] for e in enriched if e["source"] == src]
        if src_scores:
            src_arr = np.array(src_scores)
            print(f"  {src}: mean={src_arr.mean():.3f}, "
                  f"neg={((src_arr < 0).sum())}/{len(src_arr)}")

    # Breakdown by original safety label (BT only)
    for label_val, label_name in [(True, "safe"), (False, "unsafe")]:
        label_scores = [e["llm_score"] for e in enriched
                        if e["source"] == "bt" and e["is_safe"] == label_val]
        if label_scores:
            label_arr = np.array(label_scores)
            print(f"  BT {label_name}: mean={label_arr.mean():.3f}, "
                  f"neg={((label_arr < 0).sum())}/{len(label_arr)}")

    return output_path
