# src

The shared library. Run everything from the repository root, so that `from src...` imports resolve.

## Used by the current test program

| File | Purpose |
|---|---|
| `config.py` | Configuration dataclasses for a training run |
| `model.py` | Builds the GPT-2-style model at each size |
| `dataset.py` | Loads and tokenises the corpus and assigns each row its weight |
| `training.py` | Training loop with the signed-weight loss |
| `training_advanced.py` | The variants: sparse reversed gradient, adaptive weighting, gradient and null-space projection, and gradient accumulation for the two-pass methods |
| `run_experiment.py` | Command-line entry point for one pretraining run |
| `train_dpo.py` | Direct Preference Optimization on a pretrained model |
| `evaluation.py`, `eval_toxicity.py` | Perplexity and toxicity evaluation, generation |
| `degeneracy_metrics.py` | Deterministic checks for looping and collapsed replies |

## The April–June pipeline

Kept as it was run. The test program does not import these.

| File | Purpose |
|---|---|
| `llm_judge.py`, `score_examples.py`, `score_batch.py`, `compute_stats.py` | Scoring training examples with Claude |
| `train_scorer.py`, `train_scorer_v2.py`, `train_scorer_v2_large.py`, `analyze_scorer.py` | Training a classifier on those scores (DeBERTa-v3-large in the end) and analysing its sign errors |
| `score_full_corpus.py`, `score_with_model.py` | Scoring the rest of the corpus with the classifier |
| `dataset_anthropic_hh.py` | Preparing the Anthropic HH-RLHF data |
| `train_sft.py` | Supervised fine-tuning on UltraFeedback |
| `eval_metrics.py`, `eval_probes.py`, `analysis.py` | LAMBADA, JBB-Behaviors and RealToxicityPrompts metrics; linear probes per layer; comparison plots |

## Known issues

See [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) at the top level.
