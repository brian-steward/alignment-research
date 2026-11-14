# Data and weights

This repository contains code and write-ups, but no data or model weights. The folders `data/`, `models/`, `results/` and `logs/` that the code refers to exist only on the machine the experiments ran on (about 80 GB).

## Sources

All of these are public datasets and models, each under its own licence. None of their text is redistributed here.

| Used for | Source |
|---|---|
| Pretraining corpus | OpenWebText (1.5M documents), BeaverTails question–answer pairs, Anthropic HH-RLHF dialogues |
| Harm scores on the corpus | Claude scores on a sample; a DeBERTa-v3-large classifier trained on those scores for the rest (`src/llm_judge.py`, `src/train_scorer_v2_large.py`, `src/score_full_corpus.py`) |
| Harmful evaluation and attack prompts | BeaverTails |
| Benign evaluation prompts | Anthropic HH-RLHF, helpful test splits |
| Capability | WikiText-2 perplexity |
| Fine-tuning step | UltraFeedback |
| Local judges | Probe heads on Qwen2.5-7B-Instruct activations; WildGuard and Shieldstral were tried and rejected as harm judges |
| Verdict labels | Blind labels from Claude Opus |

## Files the code expects that are not here

| File                                                | What it is                                                                                       | Built by                                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `data/sw_program/manifest_v1.pt`, `heldout_v1.json` | Which corpus rows to train on, their scores, and the held-out prompt sets                        | `python sw_program/build_manifest.py` (needs the tokenised corpus and the scores)                        |
| `s15_susceptibility/splits.json`                    | Prompt-disjoint attack, held-out, distillation and probe splits                                  | `python s15_susceptibility/splits.py` (reads BeaverTails from the local Hugging Face cache)              |
| `sw_program/prompts.json`                           | The test program's prompt sets                                                                   | `python sw_program/ev.py prompts-from-manifest`                                                          |
| `coherence_pilot/hh_helpful_test.json`              | Local copy of the HH-RLHF helpful test splits; the benign dev and test prompts are drawn from it | Taken from the dataset and passed to `cp.py prompts`                                                     |

## What this means for running the code

- **Tests.** `./run_tests.sh` runs everything that does not need the data: 239 tests. Another 22 are skipped with a reason when the data is absent. Of these, 20 are in `sw_program/`: 17 check the training data itself (deduplication, held-out prompts absent from training rows, token cache against source text) and 3 need the prompt files. The other 2 are in `s15_susceptibility/`: one rebuilds the prompt splits from BeaverTails, and one compares the S20 report with the stored results.
- **Evals.** `evals/number_golf` and `evals/coding_golf` are self-contained. Their raw model outputs are in the repository because they contain no dataset text.
- **Training and evaluation runs.** These need the corpus, the scores and a GPU. The shell scripts record how each experiment was run on one machine. They are not a turnkey reproduction.

## Content note

The research is about harmful text, but the repository contains none of the datasets' text. Two evaluation modules (`src/eval_toxicity.py`, `src/evaluation.py`) contain short keyword lists that include slurs, used for a keyword-based toxicity count.
