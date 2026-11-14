# s15_susceptibility

The relearning experiment (S15 in my notes) and its follow-up on concept and disposition (S20). The question, design, result and caveats are in [docs/relearning-resistance.md](../docs/relearning-resistance.md).

The shell scripts change into this folder themselves. The Python files find the repository root from their own location.

Comments and report headings cite sections of the experiment's protocol, for example "protocol v2 §3.5". The protocol is a working document and is not published. Its design, metrics and amendments are summarised in [docs/relearning-resistance.md](../docs/relearning-resistance.md).

## Experiment

| File                             | What it does                                                                                                |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `splits.py`                      | Builds the prompt-disjoint splits from BeaverTails: attack set, held-out set, distillation pool, probe pool |
| `common.py`                      | Helpers shared by both experiments                                                                          |
| `scoring.py`                     | One evaluation point for the curves and baselines                                                           |
| `attack.py`                      | Fine-tunes an arm on the attack set, or on benign text as a control, and evaluates along the way            |
| `distill.py`                     | Builds the imitator: the uniform model distilled toward the sparse model                                    |
| `diagnostics.py`, `summarize.py` | Reading the curves and producing the tables                                                                 |
| `s20.py`                         | Probes, steering and ablation along the unsafe–safe direction                                               |

## Judges

Much of the work here went into finding a judge that could be trusted on small, often incoherent models.

| File | What it does |
|---|---|
| `probe_judge.py` | The judge used for the curves: linear heads on Qwen2.5-7B-Instruct activations, fit to Opus labels |
| `local_judge.py`, `validate_local_judge.py` | A zero-shot local judge, and its validation (it failed) |
| `validate_scorers.py` | Detoxify and other local scorers against existing Opus labels (Detoxify failed) |
| `endpoint_judge.py` | Exports replies for blind Opus labelling at the start and end of each attack |
| `opus_audit.py` | Audit rule: does each newly labelled checkpoint sit on the probe–Opus line? |
| `guard_judge.py`, `judge_combo_audit.py`, `endpoint_check.py` | WildGuard as a harm judge, combined with the probe's coherence filter, and the audit that rejected it |
| `shieldstral_judge.py`, `shieldstral_readout.py` | Shieldstral as a second judge: rejected for harm, kept as a stricter coherence filter for a secondary readout |
| `coherence_refit.py`, `coherence_ppl.py` | Two attempts to improve the coherence call: refitting the head, and using perplexity |
| `opus_judge_prompt.md` | The instruction given to each Opus judge |
| `RUBRIC.md` | The rubric the judges read, word for word. "P1" in its first lines is my ID for the signed-weight project. The scripts read their own copy under `results/s15/endpoint_judge/` |

## Run scripts

`run_step2.sh` (baselines and learning-rate sweep), `run_step4.sh` (full grid), `run_step5.sh` (likelihood-matched imitator), `run_s75.sh` (matched-length control), `run_50m_pretrain.sh` and `run_50m_s15.sh` (replication at 50M), `run_s20.sh` and `run_s20_consensus.sh` (S20), `run_guard_judge.sh` and `run_endpoint_check.sh` (judge checks).

The data splits and model weights are not in the repository. See [DATA.md](../DATA.md).
