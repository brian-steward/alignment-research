# Known issues

Bugs I know about: first the ones that are still open, then the ones that were listed here and have since been fixed.

Limits of the results, as opposed to bugs in the code, are in each project page under `docs/`.

## Open

Found in a code review on 2026-11-14.

- **`sw_program/build_manifest.py`: 18 training rows nearly match a held-out prompt.** The filter that removes training rows resembling a held-out prompt finds candidates by looking up the rarest words of each prompt. Because of a rounding error it looks up one word too few when a prompt has 5, 10, 15, … distinct words. Manifest v1 therefore keeps 3 BeaverTails rows and 15 Anthropic HH dialogues that should have been removed. They carry five distinct prompts: one matches a harmful test prompt, two match attack prompts and two match reserve prompts. A fix changes the training data, and phase 1 is running on manifest v1, so the fix waits for the next version of the manifest.

## Fixed on 2026-11-14

Each fix has a test. Fixing the code does not change a file that the old code had already written, so an entry says so where a stored result still carries the fault.

### Found in the code review of 2026-11-14

The tests are in `tests/test_scorer_tools.py` and `s15_susceptibility/test_splits_s20.py`.

- **`src/train_scorer.py`: the validation split was by row**, so a text that appears more than once could fall on both sides of it. The split is now by distinct text. Separately, a row scored exactly 0 always counted as a sign error, and such rows are now left out of the sign counts. The stored classifier was trained on the old split: 2,896 of its 12,261 validation rows have the text of a training row, and 214 are scored 0. Both affect the validation agreement quoted in the signed-weight page.
- **`s15_susceptibility/splits.py`: some evaluation prompts are in other sets.** The check that keeps evaluation prompts out of the other sets compared exact strings, and it missed the 16 evaluation prompts that are stored cut short. It now matches those by their start, and a rebuild puts no evaluation prompt in another set. The stored splits are the ones the experiments ran on, so they are unchanged: 2 evaluation prompts are among the attack prompts, 4 are in the distillation pool and 2 are in the S20 direction set.
- **`s15_susceptibility/s20.py`.** The readout could report that concept and disposition dissociate when neither model responded to steering. It now says that the readout cannot be applied. Whole-number steering strengths printed as “–”. Neither fault touched the stored report, which the new code reproduces exactly. The consensus prompts are now chosen with the same check as the splits. The stored set has 4 evaluation prompts among its 2,000.
- **`src/score_full_corpus.py`.** A restarted run overwrote the combined score file with only the batches scored in that run. It now rebuilds the file from every batch on disk. The stored file is complete.
- **`src/score_with_model.py`.** It wrote classifier scores under the file name used for Claude's labels. It now writes them under their own name. No stored label file has classifier output in it.
- **`src/llm_judge.py`.** A reply wrapped in a bare code fence was recorded as score 0, and a resumed run never rescored the calls that were in flight when it stopped. Both are fixed. The stored labels did not come through this code: they were scored in batch files (`src/score_batch.py`).
- **`src/train_scorer_v2.py`.** The stored metrics of the large classifier carried the label of the base one. The label now comes from the name of the model that was trained, and the stored file's label is corrected.

### Listed here as open on 2026-11-13

- **A run that crashed during evaluation was skipped as finished.** The check read `final_metrics.json`, which the training loop writes before the evaluation starts. It now looks for the evaluation entry, and evaluates the saved model of a run that has none (`src/run_experiment.py`, `tests/test_run_bookkeeping.py`).
- **Run names left out the number of epochs and `max_steps`**, so two runs that differed only in those shared a folder. The name now carries both when they are not the defaults (`src/config.py`, `tests/test_run_bookkeeping.py`).
- **`_step_sparse_negative` failed on a micro-batch with no positively weighted row.** It now applies the negative gradient to its top K% of parameters (`src/training_advanced.py`, `tests/test_grad_accum.py`).
- **The `layer_selective` method was switched off**, and `progressive_layer`, which is built on it, failed the same way. The old version trained the non-active layers on every row at a positive weight, harmful rows included. It was rewritten: the gradient of the harmful rows reaches only the chosen layers (`tests/test_layer_selective.py`). No training run has used the new version yet.
