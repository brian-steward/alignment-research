"""Extended eval metrics: LAMBADA + JBB-Behaviors + RealToxicityPrompts + repetition.

Complements the existing BeaverTails-based PPL/toxicity eval with:
- LAMBADA next-word cloze accuracy (standard capability benchmark)
- JailbreakBench/JBB-Behaviors Detoxify scoring (jailbreak refusal surrogate)
- RealToxicityPrompts challenging subset (Korbak et al. / Sam et al. standard)
- Repetition / type-token ratio on generated completions (gibberish detector)

All generation uses temperature 0.7, top-p 0.9 to match the existing eval suite.
"""

import random
from collections import Counter
from pathlib import Path

import torch
from torch.amp import autocast

from .dataset import get_tokenizer
from .eval_toxicity import generate_completions, detoxify_toxicity


def _histogram(scores, n_bins=20):
    """Bin toxicity scores into a fixed histogram for compact JSON storage.

    Returns list of (bin_center, count) tuples spanning [0, 1].
    """
    bin_width = 1.0 / n_bins
    counts = [0] * n_bins
    for s in scores:
        idx = min(int(s / bin_width), n_bins - 1)
        counts[idx] += 1
    return {
        "bin_edges": [round(i * bin_width, 3) for i in range(n_bins + 1)],
        "counts": counts,
        "n_total": len(scores),
    }


# ---------------------------------------------------------------------------
# LAMBADA next-word cloze accuracy
# ---------------------------------------------------------------------------

def lambada_accuracy(model, tokenizer, device, max_length=512, n_examples=None):
    """Cloze accuracy on LAMBADA OpenAI test set.

    For each passage, test whether the model correctly predicts every token
    of the last word when teacher-forced on the preceding context. This is
    the standard LAMBADA evaluation protocol.
    """
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    if n_examples:
        ds = ds.select(range(min(n_examples, len(ds))))

    model.eval()
    correct = 0
    total = 0
    skipped = 0
    with torch.no_grad():
        for ex in ds:
            text = ex["text"].strip()
            words = text.rsplit(" ", 1)
            if len(words) != 2:
                skipped += 1
                continue
            context, target = words

            ctx_ids = tokenizer(context, return_tensors="pt", truncation=True,
                                max_length=max_length - 8).input_ids.to(device)
            # GPT2 BPE wants a leading space before the target word
            tgt_ids = tokenizer(" " + target, return_tensors="pt").input_ids.to(device)

            full_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
            if full_ids.size(1) > max_length:
                skipped += 1
                continue

            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(full_ids).logits[0]  # (seq, vocab)

            # Predict tokens at positions [ctx_len, ctx_len+n_tgt-1] from
            # logits at positions [ctx_len-1, ctx_len-1+n_tgt-1].
            ctx_len = ctx_ids.size(1)
            n_tgt = tgt_ids.size(1)
            pred_logits = logits[ctx_len - 1 : ctx_len - 1 + n_tgt]
            preds = pred_logits.argmax(dim=-1)
            target_ids = full_ids[0, ctx_len : ctx_len + n_tgt]

            if (preds == target_ids).all().item():
                correct += 1
            total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# JailbreakBench/JBB-Behaviors — adversarial harmful prompts
# ---------------------------------------------------------------------------

def jbb_behaviors_eval(model, tokenizer, device, n_prompts=100, max_new_tokens=80):
    """Generate completions on JBB harmful behaviors, score with Detoxify.

    Returns continuous Detoxify scores plus generation quality metrics.
    These 100 prompts are the standard JailbreakBench harmful-behavior set
    (originally from TDC/HarmBench), comparable to AdvBench in intent but
    openly accessible.
    """
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    if n_prompts < len(ds):
        ds = ds.select(range(n_prompts))

    prompts = [f"Human: {ex['Goal']}\nAssistant:" for ex in ds]
    completions = generate_completions(model, tokenizer, prompts, device,
                                        max_new_tokens=max_new_tokens)

    det = detoxify_toxicity(completions)
    quality = generation_quality(completions)

    return {
        "n_prompts": len(prompts),
        "detoxify_toxic_rate": det["toxic_rate"],
        "detoxify_mean_toxicity": det["mean_toxicity"],
        "detoxify_max_toxicity": det["max_toxicity"],
        "detoxify_median_toxicity": det["median_toxicity"],
        "toxicity_histogram": _histogram([s["toxicity"] for s in det["scores"]]),
        "quality": quality,
        "completions_sample": completions[:3],
    }


# ---------------------------------------------------------------------------
# RealToxicityPrompts challenging subset — standard literature benchmark
# ---------------------------------------------------------------------------

def _extract_rtp_prompt(raw):
    """RTP prompt field is a dict-like object; extract the text."""
    if isinstance(raw, dict):
        return raw.get("text", "")
    if isinstance(raw, str):
        # Stringified dict
        try:
            d = eval(raw, {"__builtins__": {}})
            return d.get("text", "") if isinstance(d, dict) else raw
        except Exception:
            return raw
    return str(raw)


def rtp_challenging_eval(model, tokenizer, device, n_prompts=500, max_new_tokens=50,
                         seed=42):
    """Generate completions on RealToxicityPrompts challenging subset.

    The 'challenging' flag marks prompts that elicit toxic continuations from
    base GPT-2. Standard eval used in Korbak et al. 2023, Sam et al. 2024.
    """
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Filter to challenging prompts
    challenging = [ex for ex in ds if ex.get("challenging") in (True, "True", 1)]
    rng = random.Random(seed)
    if len(challenging) > n_prompts:
        challenging = rng.sample(challenging, n_prompts)

    prompts = [_extract_rtp_prompt(ex["prompt"]) for ex in challenging]
    prompts = [p for p in prompts if p and p.strip()]

    completions = generate_completions(model, tokenizer, prompts, device,
                                        max_new_tokens=max_new_tokens)

    det = detoxify_toxicity(completions)
    quality = generation_quality(completions)

    return {
        "n_prompts": len(prompts),
        "detoxify_toxic_rate": det["toxic_rate"],
        "detoxify_mean_toxicity": det["mean_toxicity"],
        "detoxify_max_toxicity": det["max_toxicity"],
        "detoxify_median_toxicity": det["median_toxicity"],
        "toxicity_histogram": _histogram([s["toxicity"] for s in det["scores"]]),
        "quality": quality,
        "completions_sample": completions[:3],
    }


# ---------------------------------------------------------------------------
# Generation quality: TTR, distinct-n, n-gram repetition
# ---------------------------------------------------------------------------

def generation_quality(completions):
    """Type-token ratio, distinct-n, and 4-gram repeat rate.

    Low TTR / distinct-n and high repeat rate indicate gibberish or
    mode-collapsed generation.
    """
    def ttr(tokens):
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    def distinct_n(tokens, n):
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        return len(set(ngrams)) / len(ngrams)

    def repeat_rate(tokens, n=4):
        """Fraction of unique n-grams that appear more than once."""
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return repeated / len(counts) if counts else 0.0

    ttrs, d1s, d2s, d3s, rep4s, lens = [], [], [], [], [], []
    empty = 0
    for c in completions:
        tokens = c.split()
        if not tokens:
            empty += 1
            continue
        lens.append(len(tokens))
        ttrs.append(ttr(tokens))
        d1s.append(distinct_n(tokens, 1))
        d2s.append(distinct_n(tokens, 2))
        d3s.append(distinct_n(tokens, 3))
        rep4s.append(repeat_rate(tokens, 4))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "ttr": avg(ttrs),
        "distinct_1": avg(d1s),
        "distinct_2": avg(d2s),
        "distinct_3": avg(d3s),
        "repeat_4gram": avg(rep4s),
        "avg_len_tokens": avg(lens),
        "empty_completions": empty,
        "n_nonempty": len(completions) - empty,
    }


# ---------------------------------------------------------------------------
# Bits-per-byte (BPB) — cross-scale perplexity metric
# ---------------------------------------------------------------------------

def bits_per_byte(model, tokenizer, device, max_length=512, n_examples=2000):
    """Compute bits-per-byte on a held-out text sample.

    BPB normalizes cross-entropy by the number of bytes in the original text
    rather than the number of tokens, removing tokenization and padding effects.
    This makes the metric directly comparable across models with different
    sequence lengths, tokenizers, or padding schemes — the standard used by
    GPT-3, Chinchilla, PaLM, etc. for cross-model perplexity comparison.

    BPB = total_CE_nats / total_bytes × log2(e)

    Uses the BeaverTails safe test split for consistency with our PPL metric.
    """
    import math
    import torch.nn as nn
    from .config import PROJECT_ROOT
    from .dataset import load_beavertails

    data_dir = PROJECT_ROOT / "data"
    bt_test = load_beavertails("test", data_dir / "beavertails")
    safe = [ex for ex in bt_test if ex["is_safe"]]

    # Use a deterministic subset
    rng = __import__("random").Random(42)
    if len(safe) > n_examples:
        safe = rng.sample(safe, n_examples)

    texts = [f"Human: {ex['prompt']}\nAssistant: {ex['response']}" for ex in safe]

    model.eval()
    total_ce_nats = 0.0
    total_bytes = 0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for i in range(0, len(texts), 16):
        batch_texts = texts[i:i + 16]
        # Count bytes in the original text (before tokenization/padding)
        batch_bytes = sum(len(t.encode("utf-8")) for t in batch_texts)

        enc = tokenizer(batch_texts, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"].float()

        with torch.no_grad():
            with autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()

                per_tok = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.shape)

                # Sum CE only over non-padding tokens (nats, since CE uses natural log)
                ce_sum = (per_tok * shift_mask).sum().item()
                tok_count = shift_mask.sum().item()

        total_ce_nats += ce_sum
        total_bytes += batch_bytes
        total_tokens += tok_count

    # Convert nats to bits: multiply by log2(e)
    total_ce_bits = total_ce_nats * math.log2(math.e)
    bpb = total_ce_bits / max(total_bytes, 1)

    # Also compute standard PPL for reference
    avg_ce = total_ce_nats / max(total_tokens, 1)
    ppl = math.exp(min(avg_ce, 100))

    return {
        "bpb": bpb,
        "ppl": ppl,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "n_examples": len(texts),
    }


# ---------------------------------------------------------------------------
# Convenience driver: run all new metrics on a loaded model
# ---------------------------------------------------------------------------

def beavertails_toxicity_eval(model, tokenizer, device, n_prompts=500, max_new_tokens=100):
    """Generate completions on BeaverTails unsafe prompts, score with Detoxify.

    This is the same prompt source as our training-time eval, giving us a
    same-distribution toxicity histogram to compare against JBB and RTP
    (different distributions) for cross-benchmark consistency checks.
    """
    from .eval_toxicity import get_test_prompts

    prompts = get_test_prompts(n_prompts, seed=42)
    completions = generate_completions(model, tokenizer, prompts, device,
                                        max_new_tokens=max_new_tokens)

    det = detoxify_toxicity(completions)
    quality = generation_quality(completions)

    return {
        "n_prompts": len(prompts),
        "detoxify_toxic_rate": det["toxic_rate"],
        "detoxify_mean_toxicity": det["mean_toxicity"],
        "detoxify_max_toxicity": det["max_toxicity"],
        "detoxify_median_toxicity": det["median_toxicity"],
        "toxicity_histogram": _histogram([s["toxicity"] for s in det["scores"]]),
        "quality": quality,
        "completions_sample": completions[:3],
    }


def run_all_metrics(model, tokenizer, device, max_length=512,
                    lambada_n=None, jbb_n=100, rtp_n=500, bt_n=500):
    """Run LAMBADA + BPB + JBB + RTP + BeaverTails + quality on a loaded model."""
    results = {}

    print("    Bits-per-byte...")
    results["bpb"] = bits_per_byte(model, tokenizer, device,
                                    max_length=max_length)
    print(f"      BPB={results['bpb']['bpb']:.4f}  "
          f"(PPL={results['bpb']['ppl']:.2f} at seq {max_length})")

    print("    LAMBADA...")
    results["lambada"] = lambada_accuracy(model, tokenizer, device,
                                           n_examples=lambada_n)
    print(f"      accuracy={results['lambada']['accuracy']*100:.2f}% "
          f"(n={results['lambada']['n']})")

    print("    BeaverTails toxicity (same-distribution)...")
    results["bt"] = beavertails_toxicity_eval(model, tokenizer, device, n_prompts=bt_n)
    print(f"      detox mean={results['bt']['detoxify_mean_toxicity']:.4f} "
          f"tox≥0.5={results['bt']['detoxify_toxic_rate']*100:.1f}%")

    print("    JBB-Behaviors...")
    results["jbb"] = jbb_behaviors_eval(model, tokenizer, device, n_prompts=jbb_n)
    print(f"      detox mean={results['jbb']['detoxify_mean_toxicity']:.4f} "
          f"tox≥0.5={results['jbb']['detoxify_toxic_rate']*100:.1f}% "
          f"TTR={results['jbb']['quality']['ttr']:.3f}")

    print("    RealToxicityPrompts (challenging)...")
    results["rtp"] = rtp_challenging_eval(model, tokenizer, device, n_prompts=rtp_n)
    print(f"      detox mean={results['rtp']['detoxify_mean_toxicity']:.4f} "
          f"tox≥0.5={results['rtp']['detoxify_toxic_rate']*100:.1f}% "
          f"TTR={results['rtp']['quality']['ttr']:.3f}")

    # Cross-benchmark consistency summary
    results["cross_benchmark"] = _cross_benchmark_consistency(results)

    return results


def _cross_benchmark_consistency(results):
    """Compare toxicity reduction across benchmarks to detect Goodhart's law.

    If a model reduces toxicity on the training distribution (BeaverTails) but
    not on out-of-distribution benchmarks (JBB, RTP), that suggests overfitting
    to the classifier rather than genuine safety learning.

    Returns per-benchmark toxicity rates and the max divergence between them.
    """
    benchmarks = {}
    for key, label in [("bt", "BeaverTails"), ("jbb", "JBB"), ("rtp", "RTP")]:
        if key in results:
            benchmarks[label] = {
                "toxic_rate": results[key]["detoxify_toxic_rate"],
                "mean_toxicity": results[key]["detoxify_mean_toxicity"],
            }

    # Max divergence in toxic_rate across benchmarks
    rates = [b["toxic_rate"] for b in benchmarks.values()]
    means = [b["mean_toxicity"] for b in benchmarks.values()]

    return {
        "benchmarks": benchmarks,
        "toxic_rate_range": max(rates) - min(rates) if rates else 0,
        "mean_toxicity_range": max(means) - min(means) if means else 0,
    }
