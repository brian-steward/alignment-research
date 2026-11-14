# sw_program

The signed-weight test program. The plan, with every threshold, is in [docs/signed-weight-pretraining.md](../docs/signed-weight-pretraining.md). Run everything from the repository root.

## Order of work

| Step | File | What it does |
|---|---|---|
| 0 | `build_manifest.py` | Decides which corpus rows are trained on and what score each carries: one row per BeaverTails pair, and no row that asks a held-out prompt |
| 1 | `pretrain.py` | One pretraining run of one arm. Every setting is fixed here, not in shell scripts |
| 2 | `ev.py`, `run_eval.sh` | Perplexities, generations at both decodings, and the local screen for a finished run |
| 3 | `audit.py` | Exports replies for blind Opus labelling, arms mixed, and reads the labels back |
| 4 | `verdict.py` | Computes the seed-42 gate and claim C1 from the labels. It chooses no threshold |
| 5 | `calib.py` | Refits the local screen on the program's own labels and reports its error (gate G6) |
| Phase 3 | `sft.py`, `dpo.py`, `attack.py` | Fine-tuning, DPO, and the fine-tuning attack on held-out prompts |
| | `summary.py` | Status table of every evaluated run |

`run_one.sh` runs one arm, `run_queue.sh` runs the lines of a queue file in order (`queue_phase1.txt`), and `run_smoke.sh` is the phase 0 smoke test: every arm for 300 steps, then a stop and resume.

`ev.py`, `audit.py`, `calib.py` and `sft.py` reuse the harness written for the pilot in `coherence_pilot/`, and the judges in `s15_susceptibility/`.

## Tests

```bash
python -m pytest sw_program -q
```

| File | Checks | Needs the data |
|---|---|---|
| `test_verdict.py` | The gate and C1 arithmetic on made-up labels, including the boundary cases the plan words | No |
| `test_attack.py` | The attack pairs cover the attack prompts and nothing that is evaluated; the loss ignores the prompt and padding | Partly |
| `test_sft.py` | The fine-tuning wrapper, and the filter that keeps held-out prompts out of fine-tuning data | Yes |
| `test_pretrain.py` | Weights per arm, the graded push, end-of-text targets, run names | Partly |
| `test_manifest.py` | Deduplication, disjoint held-out sets, no training row asking a held-out prompt, the same check failing on the unfiltered rows, token cache against source text | Yes |

Tests that need the data are skipped when it is absent. See [DATA.md](../DATA.md).
