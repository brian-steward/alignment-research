# Creativity evals

Two small evals built on one idea: score creativity as the rate at which a model keeps finding *distinct, valid* solutions to a problem that has many, with every answer checked by code. The result is a curve, not a single score, and over the range tested it does not saturate for a capable model.

| Eval | Task | What decides "distinct" |
|---|---|---|
| [`number_golf/`](number_golf/README.md) | Hit a target number using one repeated digit (make 17 from 4s) | A ladder of fingerprints of the expression, each rung ignoring more: operand order, how a unit is built, how each summand is built |
| [`coding_golf/`](coding_golf/README.md) | Write a function (nth Fibonacci; counting constrained strings) | Complexity class measured from operation counts, plus the paradigm used |

## What was found (June 2026; Claude Haiku, Sonnet and Opus)

- **How you ask matters more than which model you ask.** In independent one-shot attempts every model gives one or two distinct methods. With duplicate feedback after each round, the same models reach 11, 30 and 28. ([number_golf/RESULTS.md](number_golf/RESULTS.md))
- **With feedback, the weak model saturates and the strong ones do not.** Over about 100 attempts Haiku stops at 27 distinct methods. Sonnet and Opus are still adding 9–13 per round at 78 and 73.
- **A famous problem measures recall.** On Fibonacci all three models cover 5 of the 6 known methods, so there is no capability signal.
- **A novel problem did not separate the models either.** On randomly generated counting problems all three models reach the same best method, a logarithmic-time matrix power, at every difficulty level. Sonnet and Opus also derive a recurrence for the count, which Haiku does not, but the scoring does not register that. An earlier version of this page said Sonnet and Opus reached a constant-time closed form. That came from a bug in the complexity classifier, corrected on 2026-11-14. ([coding_golf/RESULTS.md](coding_golf/RESULTS.md))

## Limits

One target in number golf, one seed in constraint counting, and at most 16 attempts per round. The model responses were collected through Claude Code subagents, not through a scripted API run, so the pipeline starts from a file of responses. This is a demonstration that the measurement works, not a benchmark.

## Run the tests

```bash
pytest evals/number_golf -q
pytest evals/coding_golf -q     # separate processes: the two packages share module names
```
