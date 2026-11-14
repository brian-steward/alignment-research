# coherence_pilot

The evaluation harness for the signed-weight test program. The folder keeps the name of the pilot it was written for, in November 2026. That pilot found that end-of-text had never been a training target. Its result is in [docs/signed-weight-pretraining.md](../docs/signed-weight-pretraining.md). The pilot's stage scripts and one-off analyses are not in the repository.

| File | What it does | Used by |
|---|---|---|
| `cp.py` | Builds the prompt sets, generates replies, scores them with the local screen and writes the report | `sw_program/ev.py` |
| `cp_audit.py` | Exports replies for blind Opus labelling in coded batches, and reads the labels back | `sw_program/audit.py` |
| `cp_calib.py` | Fits a small model on the screen's scores to Opus labels, and checks it on held-out cells and prompts | `sw_program/calib.py` |
| `cp_sft.py` | Supervised fine-tuning, with a choice of data, prompt format and end-of-text target | `sw_program/sft.py` |
| `pretrain.py` | Wrapper around `src/run_experiment.py` that applies recipe changes from outside. The main one makes end-of-text a training target | `sw_program/ev.py` |
| `test_pretrain_wrapper.py` | Tests for the wrapper's dataset | `./run_tests.sh` |

## What the local screen reports for each reply

- `reg`: probability that the reply is incoherent, from the probe head registered for the relearning experiment.
- `refit`: the same head refit to Opus's coherence labels from October.
- `degen`: a flag for loops and collapsed replies, computed in code (`src/degeneracy_metrics.py`). Also `words`, `hit_cap` (the reply used every allowed token) and `ends_clean` (it ends in sentence-final punctuation).
- `helpful`, on benign prompts: a probe head fit to Opus `is_helpful` labels, used only if its leave-one-model-out AUROC is at least 0.85.
- `harm`, on harmful prompts: the registered probe head, for information only.

No local judge reproduces Opus's coherence rates on these models. The screen is for development checks. The verdict comes from the blind Opus audit.

The prompt sets (`prompts.json`) and the data they are built from are not in the repository. See [DATA.md](../DATA.md).
