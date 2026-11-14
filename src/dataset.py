"""Dataset loading, tokenization, and weight assignment for the signed-weight experiment.

Combines OpenWebText (positive corpus) with BeaverTails (alignment corpus).
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset

from .config import ExperimentConfig


def get_tokenizer() -> GPT2TokenizerFast:
    """Load GPT-2 tokenizer with padding token set."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_openwebtext(n_examples: int, cache_dir: Path) -> list[str]:
    """Load a subset of OpenWebText as plain text strings."""
    cache_file = cache_dir / f"owt_texts_{n_examples}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            texts = json.load(f)
        print(f"Loaded {len(texts)} OpenWebText examples from cache")
        return texts

    print(f"Downloading OpenWebText (first {n_examples} examples)...")
    ds = load_dataset("Skylion007/openwebtext", split=f"train[:{n_examples}]")
    texts = [ex["text"] for ex in ds]

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(texts, f)
    print(f"Cached {len(texts)} OpenWebText examples")
    return texts


def load_beavertails(split: str, cache_dir: Path) -> list[dict]:
    """Load BeaverTails dataset. Returns list of {prompt, response, is_safe}.

    Args:
        split: "train" or "test" (mapped to "330k_train" / "330k_test" internally)
    """
    cache_file = cache_dir / f"beavertails_{split}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            examples = json.load(f)
        print(f"Loaded {len(examples)} BeaverTails {split} examples from cache")
        return examples

    # BeaverTails uses "330k_train"/"330k_test" as split names
    hf_split = f"330k_{split}" if split in ("train", "test") else split
    print(f"Downloading BeaverTails {split} split (HF split: {hf_split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=hf_split)
    examples = []
    for ex in ds:
        examples.append({
            "prompt": ex["prompt"],
            "response": ex["response"],
            "is_safe": ex["is_safe"],
        })

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(examples, f)
    print(f"Cached {len(examples)} BeaverTails {split} examples")
    return examples


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_texts(texts: list[str], tokenizer: GPT2TokenizerFast,
                   max_length: int, chunk_size: int = 50_000) -> dict:
    """Tokenize plain text strings, truncating/padding to max_length.

    Processes in chunks to avoid excessive memory usage on large datasets.
    """
    if len(texts) <= chunk_size:
        encoded = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    # Chunked tokenization for large datasets.
    # Pre-allocate in int32 to halve memory: 2.16M × 1024 × 4 bytes = 8.9 GB
    # instead of int64's 17.7 GB. Token IDs (max 50257) fit in int32.
    # Attention mask values are 0/1, also fine in int32.
    # The DataLoader __getitem__ converts to int64 at batch time.
    n = len(texts)
    input_ids = torch.zeros(n, max_length, dtype=torch.int32)
    attention_mask = torch.zeros(n, max_length, dtype=torch.int32)

    for i in range(0, n, chunk_size):
        chunk = texts[i:i + chunk_size]
        encoded = tokenizer(
            chunk,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        end = min(i + chunk_size, n)
        input_ids[i:end] = encoded["input_ids"].to(torch.int32)
        attention_mask[i:end] = encoded["attention_mask"].to(torch.int32)
        del encoded
        print(f"    Tokenized {end}/{n}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def format_beavertails(examples: list[dict]) -> list[str]:
    """Format BeaverTails examples as 'Human: ...\nAssistant: ...' strings."""
    return [
        f"Human: {ex['prompt']}\nAssistant: {ex['response']}"
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# Weight assignment
# ---------------------------------------------------------------------------

def assign_owt_weights(n: int) -> torch.Tensor:
    """OpenWebText examples always get weight +1.0."""
    return torch.ones(n)


def assign_bt_weights(is_safe_flags: list[bool], scheme: str,
                      negative_alpha: float) -> torch.Tensor:
    """Assign weights to BeaverTails examples based on safety label and scheme."""
    weights = torch.ones(len(is_safe_flags))
    for i, safe in enumerate(is_safe_flags):
        if not safe:
            if scheme == "uniform":
                weights[i] = 1.0
            elif scheme == "filtered":
                weights[i] = 0.0
            elif scheme == "negative":
                weights[i] = -negative_alpha
            else:
                raise ValueError(f"Unknown scheme: {scheme}")
    return weights


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class WeightedTextDataset(Dataset):
    """Dataset that returns (input_ids, attention_mask, weight) tuples."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 weights: torch.Tensor):
        assert len(input_ids) == len(attention_mask) == len(weights)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.weights = weights

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Convert int32 storage back to int64 at batch time (required by
        # embedding layers). This trades a tiny per-batch conversion cost
        # for halving the dataset's RAM footprint (critical at 200M scale
        # with 2.16M × 1024 examples on 64 GB systems).
        return (
            self.input_ids[idx].long(),
            self.attention_mask[idx].long(),
            self.weights[idx],
        )


# ---------------------------------------------------------------------------
# Main data pipeline
# ---------------------------------------------------------------------------

def load_llm_scores(data_dir: Path) -> dict[int, float]:
    """Load combined LLM scores from all scored batch files."""
    scores = {}
    n_skipped = 0
    batches_dir = data_dir / "llm_scores" / "batches"
    for sf in sorted(batches_dir.glob("batch_*_scored.jsonl")):
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    scores[entry["idx"]] = entry["score"]
                except (json.JSONDecodeError, KeyError):
                    n_skipped += 1
    if n_skipped:
        print(f"  Warning: skipped {n_skipped} malformed score lines")
    return scores


def prepare_scored_datasets(
    experiment_cfg: ExperimentConfig,
    max_length: int = 256,
) -> tuple[WeightedTextDataset, WeightedTextDataset, WeightedTextDataset]:
    """Prepare datasets using only LLM-scored examples with continuous weights.

    Weight assignment by scheme:
        - "uniform": all weights = 1.0 (ignores scores, just uses scored subset)
        - "filtered": weight = 1.0 if score >= 0, else 0.0
        - "negative": weight = score directly (continuous [-1, 1])
    score_shift is added to all scores before assignment (clipped to [-1, 1]).
    """
    tokenizer = get_tokenizer()
    owt_cache = experiment_cfg.data_dir / "openwebtext"
    bt_cache = experiment_cfg.data_dir / "beavertails"

    # Load LLM scores
    scores = load_llm_scores(experiment_cfg.data_dir)
    print(f"Loaded {len(scores)} LLM scores")

    # Load raw data to reconstruct texts
    owt_texts = load_openwebtext(experiment_cfg.owt_subset_size, owt_cache)
    bt_train = load_beavertails("train", bt_cache)
    bt_texts = format_beavertails(bt_train)
    n_owt = len(owt_texts)

    # Build scored subset
    texts = []
    raw_scores = []
    for idx, score in sorted(scores.items()):
        if idx < n_owt:
            texts.append(owt_texts[idx])
        else:
            bt_idx = idx - n_owt
            if bt_idx < len(bt_texts):
                texts.append(bt_texts[bt_idx])
            else:
                continue
        raw_scores.append(score)

    print(f"Scored subset: {len(texts)} examples")

    # Apply shift and clip
    import numpy as np
    raw_scores = np.array(raw_scores)
    if experiment_cfg.score_shift != 0.0:
        raw_scores = np.clip(raw_scores + experiment_cfg.score_shift, -1.0, 1.0)
        print(f"Applied score shift +{experiment_cfg.score_shift}: "
              f"new mean={raw_scores.mean():+.3f}, neg={int((raw_scores < 0).sum())}")

    # Assign weights based on scheme
    scheme = experiment_cfg.weight_scheme
    if scheme == "uniform":
        weights = torch.ones(len(texts))
        print(f"  Scheme=uniform: all weights=1.0")
    elif scheme == "filtered":
        weights = torch.tensor([1.0 if s >= 0 else 0.0 for s in raw_scores])
        n_filtered = int((weights == 0).sum())
        print(f"  Scheme=filtered: {n_filtered} examples zeroed ({n_filtered/len(weights):.1%})")
    elif scheme == "negative":
        weights = torch.tensor(raw_scores.tolist(), dtype=torch.float32)
        n_neg = int((weights < 0).sum())
        print(f"  Scheme=negative: {n_neg} negative ({n_neg/len(weights):.1%}), "
              f"weight range=[{weights.min():.2f}, {weights.max():.2f}]")
    else:
        raise ValueError(f"Unknown scheme for scored data: {scheme}")

    # Tokenize
    tok = tokenize_texts(texts, tokenizer, max_length)
    train_ds = WeightedTextDataset(tok["input_ids"], tok["attention_mask"], weights)

    # Eval datasets (same as unscored — use full BT test set)
    eval_safe_cache = experiment_cfg.data_dir / "eval_safe.pt"
    eval_unsafe_cache = experiment_cfg.data_dir / "eval_unsafe.pt"

    if eval_safe_cache.exists() and eval_unsafe_cache.exists():
        print("Loading cached eval datasets...")
        eval_safe_data = torch.load(eval_safe_cache, weights_only=True)
        eval_unsafe_data = torch.load(eval_unsafe_cache, weights_only=True)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)
    else:
        print("Building eval datasets...")
        bt_test = load_beavertails("test", bt_cache)
        safe_test = [ex for ex in bt_test if ex["is_safe"]]
        unsafe_test = [ex for ex in bt_test if not ex["is_safe"]]
        safe_texts = format_beavertails(safe_test)
        unsafe_texts = format_beavertails(unsafe_test)
        safe_tok = tokenize_texts(safe_texts, tokenizer, max_length)
        unsafe_tok = tokenize_texts(unsafe_texts, tokenizer, max_length)
        eval_safe_data = {
            "input_ids": safe_tok["input_ids"],
            "attention_mask": safe_tok["attention_mask"],
            "weights": torch.ones(len(safe_test)),
        }
        eval_unsafe_data = {
            "input_ids": unsafe_tok["input_ids"],
            "attention_mask": unsafe_tok["attention_mask"],
            "weights": torch.ones(len(unsafe_test)),
        }
        torch.save(eval_safe_data, eval_safe_cache)
        torch.save(eval_unsafe_data, eval_unsafe_cache)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)

    print(f"Train: {len(train_ds)} examples | Eval safe: {len(eval_safe_ds)} | Eval unsafe: {len(eval_unsafe_ds)}")
    return train_ds, eval_safe_ds, eval_unsafe_ds


def prepare_gaussian_dataset(
    experiment_cfg: ExperimentConfig,
    max_length: int = 256,
) -> tuple[WeightedTextDataset, WeightedTextDataset, WeightedTextDataset]:
    """Load a pre-built Gaussian-shaped dataset (from build_gaussian_datasets.py).

    These datasets have Gaussian score distributions with σ=0.10 and mean
    chosen to achieve the target negative percentage. Uses the same
    training pipeline as ratio datasets.
    """
    ratio_pct = experiment_cfg.ratio_pct
    data_path = experiment_cfg.data_dir / "gaussian_datasets" / f"gaussian_{ratio_pct}pct_neg.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Gaussian dataset not found: {data_path}\n"
                                f"Run build_gaussian_datasets.py first.")

    print(f"Loading Gaussian dataset: {ratio_pct}% negative from {data_path}")
    with open(data_path) as f:
        examples = json.load(f)

    texts = [ex["text"] for ex in examples]
    scores = [ex["score"] for ex in examples]

    scheme = experiment_cfg.weight_scheme
    if scheme == "uniform":
        weights = torch.ones(len(texts))
    elif scheme == "filtered":
        weights = torch.tensor([1.0 if s >= 0 else 0.0 for s in scores])
    elif scheme == "negative":
        weights = torch.tensor(scores, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    n_neg = int((weights < 0).sum())
    print(f"  {len(texts)} examples, {n_neg} negative ({n_neg/len(texts):.1%}), "
          f"weight range=[{weights.min():.2f}, {weights.max():.2f}]")

    tokenizer = get_tokenizer()
    tok = tokenize_texts(texts, tokenizer, max_length)
    train_ds = WeightedTextDataset(tok["input_ids"], tok["attention_mask"], weights)

    bt_cache = experiment_cfg.data_dir / "beavertails"
    eval_safe_cache = experiment_cfg.data_dir / "eval_safe.pt"
    eval_unsafe_cache = experiment_cfg.data_dir / "eval_unsafe.pt"

    if eval_safe_cache.exists() and eval_unsafe_cache.exists():
        print("Loading cached eval datasets...")
        eval_safe_data = torch.load(eval_safe_cache, weights_only=True)
        eval_unsafe_data = torch.load(eval_unsafe_cache, weights_only=True)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)
    else:
        print("Building eval datasets...")
        bt_test = load_beavertails("test", bt_cache)
        safe_test = [ex for ex in bt_test if ex["is_safe"]]
        unsafe_test = [ex for ex in bt_test if not ex["is_safe"]]
        safe_texts = format_beavertails(safe_test)
        unsafe_texts = format_beavertails(unsafe_test)
        safe_tok = tokenize_texts(safe_texts, tokenizer, max_length)
        unsafe_tok = tokenize_texts(unsafe_texts, tokenizer, max_length)
        eval_safe_data = {
            "input_ids": safe_tok["input_ids"],
            "attention_mask": safe_tok["attention_mask"],
            "weights": torch.ones(len(safe_test)),
        }
        eval_unsafe_data = {
            "input_ids": unsafe_tok["input_ids"],
            "attention_mask": unsafe_tok["attention_mask"],
            "weights": torch.ones(len(unsafe_test)),
        }
        torch.save(eval_safe_data, eval_safe_cache)
        torch.save(eval_unsafe_data, eval_unsafe_cache)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)

    print(f"Train: {len(train_ds)} | Eval safe: {len(eval_safe_ds)} | Eval unsafe: {len(eval_unsafe_ds)}")
    return train_ds, eval_safe_ds, eval_unsafe_ds


def prepare_ratio_dataset(
    experiment_cfg: ExperimentConfig,
    max_length: int = 256,
) -> tuple[WeightedTextDataset, WeightedTextDataset, WeightedTextDataset]:
    """Load a pre-built ratio dataset (from build_ratio_datasets.py).

    These datasets have controlled negative-score ratios with equal
    representation from OWT+BT, Civil Comments, and PKU-SafeRLHF.
    The ratio_pct field on experiment_cfg selects which dataset to load.
    Scores are used directly as training weights (no shift needed —
    the ratio is already baked into the dataset composition).
    """
    ratio_pct = experiment_cfg.ratio_pct
    data_path = experiment_cfg.data_dir / "ratio_datasets" / f"ratio_{ratio_pct}pct_neg.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Ratio dataset not found: {data_path}\n"
                                f"Run build_ratio_datasets.py first.")

    print(f"Loading ratio dataset: {ratio_pct}% negative from {data_path}")
    with open(data_path) as f:
        examples = json.load(f)

    texts = [ex["text"] for ex in examples]
    scores = [ex["score"] for ex in examples]

    # Assign weights based on scheme
    scheme = experiment_cfg.weight_scheme
    if scheme == "uniform":
        weights = torch.ones(len(texts))
    elif scheme == "filtered":
        weights = torch.tensor([1.0 if s >= 0 else 0.0 for s in scores])
    elif scheme == "negative":
        weights = torch.tensor(scores, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    n_neg = int((weights < 0).sum())
    print(f"  {len(texts)} examples, {n_neg} negative ({n_neg/len(texts):.1%}), "
          f"weight range=[{weights.min():.2f}, {weights.max():.2f}]")

    # Tokenize
    tokenizer = get_tokenizer()
    tok = tokenize_texts(texts, tokenizer, max_length)
    train_ds = WeightedTextDataset(tok["input_ids"], tok["attention_mask"], weights)

    # Eval datasets (same BeaverTails test split as other experiments)
    bt_cache = experiment_cfg.data_dir / "beavertails"
    eval_safe_cache = experiment_cfg.data_dir / "eval_safe.pt"
    eval_unsafe_cache = experiment_cfg.data_dir / "eval_unsafe.pt"

    if eval_safe_cache.exists() and eval_unsafe_cache.exists():
        print("Loading cached eval datasets...")
        eval_safe_data = torch.load(eval_safe_cache, weights_only=True)
        eval_unsafe_data = torch.load(eval_unsafe_cache, weights_only=True)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)
    else:
        print("Building eval datasets...")
        bt_test = load_beavertails("test", bt_cache)
        safe_test = [ex for ex in bt_test if ex["is_safe"]]
        unsafe_test = [ex for ex in bt_test if not ex["is_safe"]]
        safe_texts = format_beavertails(safe_test)
        unsafe_texts = format_beavertails(unsafe_test)
        safe_tok = tokenize_texts(safe_texts, tokenizer, max_length)
        unsafe_tok = tokenize_texts(unsafe_texts, tokenizer, max_length)
        eval_safe_data = {
            "input_ids": safe_tok["input_ids"],
            "attention_mask": safe_tok["attention_mask"],
            "weights": torch.ones(len(safe_test)),
        }
        eval_unsafe_data = {
            "input_ids": unsafe_tok["input_ids"],
            "attention_mask": unsafe_tok["attention_mask"],
            "weights": torch.ones(len(unsafe_test)),
        }
        torch.save(eval_safe_data, eval_safe_cache)
        torch.save(eval_unsafe_data, eval_unsafe_cache)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)

    print(f"Train: {len(train_ds)} | Eval safe: {len(eval_safe_ds)} | Eval unsafe: {len(eval_unsafe_ds)}")
    return train_ds, eval_safe_ds, eval_unsafe_ds


def prepare_hybrid_datasets(
    experiment_cfg: ExperimentConfig,
    max_length: int = 256,
) -> tuple[WeightedTextDataset, WeightedTextDataset, WeightedTextDataset]:
    """Prepare full corpus with LLM scores overlaid where available.

    Base weights: binary labels (OWT=+1, BT safe=+1, BT unsafe per scheme).
    Overlay: where LLM scores exist, replace with continuous weight (+ score_shift).
    This gives the model enough data to learn language while testing scored negative
    weighting on the subset that has been LLM-judged.
    """
    import numpy as np
    tokenizer = get_tokenizer()
    owt_cache = experiment_cfg.data_dir / "openwebtext"
    bt_cache = experiment_cfg.data_dir / "beavertails"

    # Load LLM scores
    scores = load_llm_scores(experiment_cfg.data_dir)
    print(f"Loaded {len(scores)} LLM scores")

    # Load full datasets
    owt_texts = load_openwebtext(experiment_cfg.owt_subset_size, owt_cache)
    bt_train = load_beavertails("train", bt_cache)
    bt_texts = format_beavertails(bt_train)
    bt_is_safe = [ex["is_safe"] for ex in bt_train]
    n_owt = len(owt_texts)

    # Combine all texts
    all_texts = owt_texts + bt_texts

    # Step 1: Binary weights for ALL examples (fallback for unscored)
    owt_weights = assign_owt_weights(n_owt)
    bt_weights = assign_bt_weights(
        bt_is_safe, experiment_cfg.weight_scheme, experiment_cfg.negative_alpha
    )
    all_weights = torch.cat([owt_weights, bt_weights], dim=0)

    # Step 2: Overlay LLM scores where available
    scheme = experiment_cfg.weight_scheme
    n_scored = 0
    for idx, score in scores.items():
        if idx >= len(all_texts):
            continue
        adjusted = float(np.clip(score + experiment_cfg.score_shift, -1.0, 1.0))
        if scheme == "negative":
            all_weights[idx] = adjusted
        elif scheme == "filtered":
            all_weights[idx] = 1.0 if adjusted >= 0 else 0.0
        # uniform: keep 1.0
        n_scored += 1

    # Stats
    n_neg = int((all_weights < 0).sum())
    n_zero = int((all_weights == 0).sum())
    n_pos = int((all_weights > 0).sum())
    print(f"Hybrid dataset: {len(all_texts)} total examples")
    print(f"  Scored: {n_scored} | Unscored: {len(all_texts) - n_scored}")
    print(f"  Weights: {n_pos} positive, {n_zero} zero, {n_neg} negative")
    print(f"  Weight range: [{all_weights.min():.3f}, {all_weights.max():.3f}], mean={all_weights.mean():.3f}")

    # Tokenize
    print(f"Tokenizing {len(all_texts)} examples at max_length={max_length}...")
    tok = tokenize_texts(all_texts, tokenizer, max_length)
    train_ds = WeightedTextDataset(tok["input_ids"], tok["attention_mask"], all_weights)

    # Eval datasets (length-aware caching for multi-model support)
    eval_safe_cache = experiment_cfg.data_dir / f"eval_safe_L{max_length}.pt"
    eval_unsafe_cache = experiment_cfg.data_dir / f"eval_unsafe_L{max_length}.pt"

    if eval_safe_cache.exists() and eval_unsafe_cache.exists():
        print("Loading cached eval datasets...")
        eval_safe_data = torch.load(eval_safe_cache, weights_only=True)
        eval_unsafe_data = torch.load(eval_unsafe_cache, weights_only=True)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)
    else:
        print("Building eval datasets...")
        bt_test = load_beavertails("test", bt_cache)
        safe_test = [ex for ex in bt_test if ex["is_safe"]]
        unsafe_test = [ex for ex in bt_test if not ex["is_safe"]]
        safe_texts = format_beavertails(safe_test)
        unsafe_texts = format_beavertails(unsafe_test)
        safe_tok = tokenize_texts(safe_texts, tokenizer, max_length)
        unsafe_tok = tokenize_texts(unsafe_texts, tokenizer, max_length)
        eval_safe_data = {
            "input_ids": safe_tok["input_ids"],
            "attention_mask": safe_tok["attention_mask"],
            "weights": torch.ones(len(safe_test)),
        }
        eval_unsafe_data = {
            "input_ids": unsafe_tok["input_ids"],
            "attention_mask": unsafe_tok["attention_mask"],
            "weights": torch.ones(len(unsafe_test)),
        }
        torch.save(eval_safe_data, eval_safe_cache)
        torch.save(eval_unsafe_data, eval_unsafe_cache)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)

    print(f"Train: {len(train_ds)} examples | Eval safe: {len(eval_safe_ds)} | Eval unsafe: {len(eval_unsafe_ds)}")
    return train_ds, eval_safe_ds, eval_unsafe_ds


def prepare_full_classifier_dataset(
    experiment_cfg: ExperimentConfig,
    max_length: int = 1024,
) -> tuple[WeightedTextDataset, WeightedTextDataset, WeightedTextDataset]:
    """Prepare the full classifier-scored dataset for 200M training.

    Combines:
    - 1.5M OWT examples (classifier-scored)
    - 300k BeaverTails (classifier-scored, with LLM gold overlay where available)
    - 360k Anthropic HH-RLHF (classifier-scored on raw text)

    Total: ~2.36M examples, ~2.4B tokens at seq_len 1024.
    All weights are continuous from the DeBERTa-v3-large classifier.
    LLM gold scores override classifier scores where available.
    """
    import numpy as np
    tokenizer = get_tokenizer()

    # --- Load OWT (1.5M) with classifier scores ---
    owt_path = experiment_cfg.data_dir / "openwebtext" / "owt_texts_1500000.json"
    print(f"Loading 1.5M OWT texts...")
    owt_texts = json.load(open(owt_path))

    # OWT classifier scores stored with idx offset 2_000_000
    owt_scores_path = experiment_cfg.data_dir / "llm_scores" / "owt_1.5M_scores_auto.json"
    owt_scores_dict = json.load(open(owt_scores_path))
    # Convert to ordered list matching owt_texts order
    owt_weights = torch.tensor([
        owt_scores_dict.get(str(2_000_000 + i), 0.25)  # default to mean if missing
        for i in range(len(owt_texts))
    ], dtype=torch.float32)
    print(f"  OWT: {len(owt_texts):,} examples, mean weight {owt_weights.mean():.3f}")

    # --- Load BeaverTails with classifier + LLM gold overlay ---
    bt_train = load_beavertails("train", experiment_cfg.data_dir / "beavertails")
    bt_texts = format_beavertails(bt_train)

    # Auto-scored (DeBERTa classifier)
    auto_scores = json.load(open(experiment_cfg.data_dir / "llm_scores" / "combined_scores_auto.json"))
    # LLM gold (overrides auto where available)
    llm_scores = load_llm_scores(experiment_cfg.data_dir)

    # BT examples are stored in combined_scores*.json under keys [BT_SCORE_OFFSET, BT_SCORE_OFFSET+len(bt)).
    # BT_SCORE_OFFSET = 200_000 was the OWT subset size at scoring time (see
    # prepare_scored_datasets where n_owt = experiment.owt_subset_size default 200_000).
    # load_beavertails does not write an `idx` field to its cache, so we compute the
    # lookup key from the array position instead of reading ex.get("idx").
    BT_SCORE_OFFSET = 200_000
    bt_weights = torch.zeros(len(bt_train), dtype=torch.float32)
    n_llm_gold = 0
    n_auto = 0
    n_binary_fallback = 0
    for i, ex in enumerate(bt_train):
        idx = BT_SCORE_OFFSET + i
        # LLM gold takes priority
        if idx in llm_scores:
            bt_weights[i] = llm_scores[idx]
            n_llm_gold += 1
        elif str(idx) in auto_scores:
            bt_weights[i] = auto_scores[str(idx)]
            n_auto += 1
        else:
            bt_weights[i] = 1.0 if ex["is_safe"] else -experiment_cfg.negative_alpha
            n_binary_fallback += 1
    print(f"  BeaverTails: {len(bt_train):,} examples, mean weight {bt_weights.mean():.3f}")
    print(f"    LLM gold:        {n_llm_gold:>8}")
    print(f"    Classifier auto: {n_auto:>8}")
    print(f"    Binary fallback: {n_binary_fallback:>8}")

    # --- Load Anthropic HH ---
    hh_path = experiment_cfg.data_dir / "anthropic_hh" / "anthropic_hh_classifier_scores.json"
    hh_scores_data = json.load(open(hh_path))
    hh_all_path = experiment_cfg.data_dir / "anthropic_hh" / "anthropic_hh_all.json"
    hh_examples = json.load(open(hh_all_path))

    # Build idx -> score map
    hh_score_by_idx = {s["idx"]: s["classifier_score"] for s in hh_scores_data}

    hh_texts = [ex["text"] for ex in hh_examples]
    hh_weights = torch.tensor([
        hh_score_by_idx.get(ex["idx"], 0.0) for ex in hh_examples
    ], dtype=torch.float32)
    print(f"  Anthropic HH: {len(hh_texts):,} examples, mean weight {hh_weights.mean():.3f}")

    # --- Combine ---
    all_texts = owt_texts + bt_texts + hh_texts
    all_weights = torch.cat([owt_weights, bt_weights, hh_weights], dim=0)

    # Apply score shift (before scheme override, so shift affects the
    # negative/positive boundary for filtered scheme)
    if experiment_cfg.score_shift != 0:
        all_weights = (all_weights + experiment_cfg.score_shift).clamp(-1.0, 1.0)

    # Apply weight scheme override
    scheme = experiment_cfg.weight_scheme
    if scheme == "uniform":
        all_weights = torch.ones_like(all_weights)
        print(f"  Scheme=uniform: all weights set to 1.0")
    elif scheme == "filtered":
        all_weights = torch.where(all_weights >= 0,
                                  torch.ones_like(all_weights),
                                  torch.zeros_like(all_weights))
        print(f"  Scheme=filtered: zeroed {int((all_weights == 0).sum()):,} negative examples")
    elif scheme == "negative":
        pass  # use continuous classifier scores as-is
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    n_pos = int((all_weights > 0).sum())
    n_neg = int((all_weights < 0).sum())
    n_zero = int((all_weights == 0).sum())
    print(f"\nFull classifier dataset: {len(all_texts):,} total examples")
    print(f"  Positive: {n_pos:,} ({n_pos/len(all_texts)*100:.1f}%)")
    print(f"  Negative: {n_neg:,} ({n_neg/len(all_texts)*100:.1f}%)")
    print(f"  Zero:     {n_zero:,}")
    print(f"  Mean: {all_weights.mean():.3f}, Std: {all_weights.std():.3f}")

    # --- Tokenize (with disk cache for memory-mapped reuse) ---
    # First run tokenizes 2.16M examples (~15 min) and saves to disk (~9 GB).
    # Subsequent runs load via mmap — the tensors stay on disk and pages are
    # loaded on demand, dropping training RAM from ~39 GB to ~22 GB.
    # Cache only the tokenized TENSORS (input_ids + attention_mask), not weights.
    # Weights depend on the scheme (uniform/filtered/negative) and differ per run,
    # but the tokenized text is identical across all runs at the same max_length.
    cache_path = experiment_cfg.data_dir / f"fullc_train_L{max_length}_tokens.pt"

    if cache_path.exists():
        print(f"Loading cached tokenized dataset from {cache_path} (mmap)...")
        cached = torch.load(cache_path, map_location="cpu", mmap=True, weights_only=True)
        tok_input_ids = cached["input_ids"]
        tok_attention_mask = cached["attention_mask"]
        if len(tok_input_ids) == len(all_weights):
            print(f"  Cache hit: {len(tok_input_ids):,} examples, max_length={max_length}")
        else:
            print(f"  Cache size mismatch ({len(tok_input_ids)} vs {len(all_weights)}), re-tokenizing...")
            cache_path.unlink()
            tok_input_ids = None
    else:
        tok_input_ids = None

    if tok_input_ids is None:
        print(f"\nTokenizing {len(all_texts):,} examples at max_length={max_length}...")
        tok = tokenize_texts(all_texts, tokenizer, max_length)
        tok_input_ids = tok["input_ids"]
        tok_attention_mask = tok["attention_mask"]

        print(f"  Saving tokenized cache to {cache_path}...")
        torch.save({
            "input_ids": tok_input_ids,
            "attention_mask": tok_attention_mask,
        }, cache_path)
        print(f"  Cache saved ({cache_path.stat().st_size / 1e9:.1f} GB)")

    # Free the raw text strings to recover ~3 GB RAM.
    del all_texts, owt_texts, bt_texts, hh_texts, hh_examples
    del auto_scores, hh_scores_data, hh_score_by_idx
    import gc; gc.collect()
    print(f"  Freed raw text strings")

    train_ds = WeightedTextDataset(tok_input_ids, tok_attention_mask, all_weights)

    # --- Eval sets (same as hybrid, length-aware caching) ---
    eval_safe_cache = experiment_cfg.data_dir / f"eval_safe_L{max_length}.pt"
    eval_unsafe_cache = experiment_cfg.data_dir / f"eval_unsafe_L{max_length}.pt"

    if eval_safe_cache.exists() and eval_unsafe_cache.exists():
        print("Loading cached eval datasets...")
        eval_safe_data = torch.load(eval_safe_cache, weights_only=True)
        eval_unsafe_data = torch.load(eval_unsafe_cache, weights_only=True)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)
    else:
        print("Building eval datasets...")
        bt_test = load_beavertails("test", experiment_cfg.data_dir / "beavertails")
        safe_test = [ex for ex in bt_test if ex["is_safe"]]
        unsafe_test = [ex for ex in bt_test if not ex["is_safe"]]
        safe_texts = format_beavertails(safe_test)
        unsafe_texts = format_beavertails(unsafe_test)
        safe_tok = tokenize_texts(safe_texts, tokenizer, max_length)
        unsafe_tok = tokenize_texts(unsafe_texts, tokenizer, max_length)
        eval_safe_data = {
            "input_ids": safe_tok["input_ids"],
            "attention_mask": safe_tok["attention_mask"],
            "weights": torch.ones(len(safe_test)),
        }
        eval_unsafe_data = {
            "input_ids": unsafe_tok["input_ids"],
            "attention_mask": unsafe_tok["attention_mask"],
            "weights": torch.ones(len(unsafe_test)),
        }
        torch.save(eval_safe_data, eval_safe_cache)
        torch.save(eval_unsafe_data, eval_unsafe_cache)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)

    print(f"\nTrain: {len(train_ds):,} | Eval safe: {len(eval_safe_ds):,} | Eval unsafe: {len(eval_unsafe_ds):,}")
    return train_ds, eval_safe_ds, eval_unsafe_ds


def prepare_datasets(
    experiment_cfg: ExperimentConfig,
    max_length: int = 256,
) -> tuple[WeightedTextDataset, WeightedTextDataset, WeightedTextDataset]:
    """Prepare train, eval-safe, and eval-unsafe datasets.

    Returns:
        (train_dataset, eval_safe_dataset, eval_unsafe_dataset)
    """
    tokenizer = get_tokenizer()
    owt_cache = experiment_cfg.data_dir / "openwebtext"
    bt_cache = experiment_cfg.data_dir / "beavertails"

    # Check for cached tokenized data
    train_cache = experiment_cfg.data_dir / f"train_{experiment_cfg.weight_scheme}_a{experiment_cfg.negative_alpha}.pt"
    eval_safe_cache = experiment_cfg.data_dir / "eval_safe.pt"
    eval_unsafe_cache = experiment_cfg.data_dir / "eval_unsafe.pt"

    # Always need eval sets (scheme-independent)
    if eval_safe_cache.exists() and eval_unsafe_cache.exists():
        print("Loading cached eval datasets...")
        eval_safe_data = torch.load(eval_safe_cache, weights_only=True)
        eval_unsafe_data = torch.load(eval_unsafe_cache, weights_only=True)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)
    else:
        print("Building eval datasets...")
        bt_test = load_beavertails("test", bt_cache)
        safe_test = [ex for ex in bt_test if ex["is_safe"]]
        unsafe_test = [ex for ex in bt_test if not ex["is_safe"]]
        print(f"  Eval: {len(safe_test)} safe, {len(unsafe_test)} unsafe")

        safe_texts = format_beavertails(safe_test)
        unsafe_texts = format_beavertails(unsafe_test)

        safe_tok = tokenize_texts(safe_texts, tokenizer, max_length)
        unsafe_tok = tokenize_texts(unsafe_texts, tokenizer, max_length)

        eval_safe_data = {
            "input_ids": safe_tok["input_ids"],
            "attention_mask": safe_tok["attention_mask"],
            "weights": torch.ones(len(safe_test)),
        }
        eval_unsafe_data = {
            "input_ids": unsafe_tok["input_ids"],
            "attention_mask": unsafe_tok["attention_mask"],
            "weights": torch.ones(len(unsafe_test)),
        }
        torch.save(eval_safe_data, eval_safe_cache)
        torch.save(eval_unsafe_data, eval_unsafe_cache)
        eval_safe_ds = WeightedTextDataset(**eval_safe_data)
        eval_unsafe_ds = WeightedTextDataset(**eval_unsafe_data)

    # Training dataset (scheme-dependent due to weights)
    if train_cache.exists():
        print(f"Loading cached train dataset ({experiment_cfg.weight_scheme})...")
        train_data = torch.load(train_cache, weights_only=True)
        train_ds = WeightedTextDataset(**train_data)
    else:
        print("Building train dataset...")

        # OpenWebText
        owt_texts = load_openwebtext(experiment_cfg.owt_subset_size, owt_cache)
        owt_tok = tokenize_texts(owt_texts, tokenizer, max_length)
        owt_weights = assign_owt_weights(len(owt_texts))

        # BeaverTails train
        bt_train = load_beavertails("train", bt_cache)
        bt_texts = format_beavertails(bt_train)
        bt_tok = tokenize_texts(bt_texts, tokenizer, max_length)
        bt_is_safe = [ex["is_safe"] for ex in bt_train]
        bt_weights = assign_bt_weights(
            bt_is_safe, experiment_cfg.weight_scheme, experiment_cfg.negative_alpha
        )

        # Print stats
        n_safe = sum(bt_is_safe)
        n_unsafe = len(bt_is_safe) - n_safe
        print(f"  OWT: {len(owt_texts)} examples (all weight=+1.0)")
        print(f"  BT safe: {n_safe} (weight=+1.0)")
        print(f"  BT unsafe: {n_unsafe} (weight={bt_weights[next(i for i, s in enumerate(bt_is_safe) if not s)].item():.2f})")

        # Concatenate
        all_input_ids = torch.cat([owt_tok["input_ids"], bt_tok["input_ids"]], dim=0)
        all_attn_mask = torch.cat([owt_tok["attention_mask"], bt_tok["attention_mask"]], dim=0)
        all_weights = torch.cat([owt_weights, bt_weights], dim=0)

        train_data = {
            "input_ids": all_input_ids,
            "attention_mask": all_attn_mask,
            "weights": all_weights,
        }
        torch.save(train_data, train_cache)
        train_ds = WeightedTextDataset(**train_data)

    print(f"Train: {len(train_ds)} examples | Eval safe: {len(eval_safe_ds)} | Eval unsafe: {len(eval_unsafe_ds)}")
    return train_ds, eval_safe_ds, eval_unsafe_ds


def create_dataloaders(
    train_ds: WeightedTextDataset,
    eval_safe_ds: WeightedTextDataset,
    eval_unsafe_ds: WeightedTextDataset,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, eval-safe, and eval-unsafe."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    eval_safe_loader = DataLoader(
        eval_safe_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    eval_unsafe_loader = DataLoader(
        eval_unsafe_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, eval_safe_loader, eval_unsafe_loader
