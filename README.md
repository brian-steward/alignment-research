# alignment-research

Alignment research with small scale models. Mostly focused on alignment in pre-training. Everything in here is using 50M to 200M parameter GPT-2 like models. It all runs on a single 5090 (32GB).

I am Brian, a fusion plasma physicist who is becoming an AI alignment researcher. This repo has most of my independent research up to this point. Many of these are works in progress and are actively evolving. Project summaries below.

## Projects

| Project                                                        | Question                                                                                                                                                                                                           | Where it stands (2026-11-14)                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Signed-weight pretraining](docs/signed-weight-pretraining.md) | Is it possible to bake alignment into pre-training by weighting training examples by an alignment metric? Will negative examples actively push models away from negative ideas? Is this more robust than filtering? | The April–October results show that a naive signed weight causes a large increase in perplexity. Weighting only the 10% of the most strongly affected gradients preserves capability. A November review found errors in the training. A test program with thresholds fixed in advance is running: phase 1 of 4. The first seed passed its gate and the other two are training. No verdict yet. |
| [Relearning resistance](docs/relearning-resistance.md)         | Does a model aligned during pretraining relearn harmful behaviour more slowly than a model distilled to behave the same way?                                                                                       | In October: yes, at 50M and 200M, the networks relearned harm more slowly and became incoherent while the filtered set relearned harm quickly. This was only a single seed - more runs are underway. Being re-tested in phase 3 of the program.                                                                                 |
| [Creativity evals](evals/README.md)                            | Can creativity be scored as the rate at which a model keeps finding distinct valid solutions, with every answer checked automatically?                                                                             | Proof of concept from June 2026, on three Claude models. Small samples (one target, one seed), so a method demonstration and not a benchmark.                                                                                                                                                                                   |

## What to read first

1. **[docs/signed-weight-pretraining.md](docs/signed-weight-pretraining.md).** The results of the signed weight project up to this point, and the test program that is actively ongoing. Five methods of weighting, three seeds each, the gate and the kill rule for each claim, and a change log of every edit made after the plan was fixed.
2. **`sw_program/verdict.py` and `sw_program/test_verdict.py`.** The script that turns blind labels into a verdict on claims made. It was written and tested on synthetic labels before any real label existed.
3. **`sw_program/test_manifest.py`.** The tests on the training data. One checks that no training row asks a held-out prompt, and another checks that the same test fails on the data as it was before the fix.
4. **[docs/relearning-resistance.md](docs/relearning-resistance.md).** The relearning experiment, results and issues with prior tests.

## Layout

| Folder                | Contents                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/`               | One write-up per project                                                                                                              |
| `src/`                | Shared library: model, data loading, the signed-weight losses and training loops, evaluation, and the pipeline that scores the corpus |
| `sw_program/`         | The signed-weight test program: data manifest, training wrapper, evaluation, audits, verdict, and tests                               |
| `coherence_pilot/`    | The evaluation harness                                                                                                                |
| `s15_susceptibility/` | The relearning experiment and its follow-up: attack, imitator build, judges and their audits                                          |
| `evals/`              | Two creativity evals, `number_golf` and `coding_golf`, with their raw results                                                         |
| `tests/`              | Tests for the training loops                                                                                                          |

`s15` and `sw` are IDs from initial testing. The folder names keep them so that the code, logs and scripts agree.

## Running it

```bash
pip install -r requirements.txt      # Python 3.12
./run_tests.sh                       # CPU only, about two minutes
```

239 tests run without any data. Four of them load the GPT-2 tokenizer, which is downloaded from Hugging Face the first time (about 3 MB). Another 22 need the training data, the prompt files or the stored results and are skipped when those are absent.

The training and evaluation runs need the corpus, the harm scores and a GPU, none of which are in the repository. [DATA.md](DATA.md) lists what is missing and which script builds each piece. The shell scripts are a record of how each experiment was run on one machine. They are not a turnkey reproduction.

Bugs I know about and have not fixed are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md), with the ones fixed since they were listed.

## How this work was done

I work with Claude (through Claude Code) as a research assistant. It writes most of the code, runs the experiments, and drafts protocols and write-ups. I choose the questions, make the decisions and review the results. Claude Opus is also used as a blind judge of model outputs, as described in each write-up.

## Licence

MIT. See [LICENSE](LICENSE).
