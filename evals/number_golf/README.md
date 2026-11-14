# Number-Golf Creativity Eval (v1)

Measures how fast a model keeps finding *distinct, valid* expressions that hit a target using only one repeated digit — a non-saturating ("chess-like") creativity signal summarized by a Heaps'-law exponent β.


## Run the tests
```bash
# from the repository root
pytest evals/number_golf -v
```

## Deterministic transition demo (no models needed)
```bash
cd evals/number_golf
python demo_transition.py
```
Shows the saturating → non-saturating transition as the operator set grows.

## Producing model responses (the subagent step)
The pipeline consumes a JSONL of model responses, one per line:
```json
{"config": "full-42", "expr": "4*4+4!/4/4"}
```
Generate these by dispatching Anthropic subagents (Haiku / Sonnet / Opus) with the prompt from `build_prompt(config, protocol)` in `elicit.py`:
- **A1**: one independent subagent call per attempt (fixed temperature).
- **A2**: one iterative subagent thread, feeding prior answers back via `build_prompt(..., prior=...)`.

Write each model's answers to `results/<model>_<protocol>.jsonl`, then:
```python
from generator import STARTER_CONFIGS
from elicit import JsonlSolver
from run import run_config, save_results

solver = JsonlSolver("results/opus_A1.jsonl")
results = [run_config(c, solver, "A1", n_attempts=100) for c in STARTER_CONFIGS]
save_results(results, "results/opus_A1_metrics.json")
```

## Metrics per (config, model, protocol)
- `beta` (+ `beta_ci`) — Heaps creativity exponent
- `yield_rate` — valid per attempt
- `distinct_at` — leaderboard scalar at fixed budgets
- `max_complexity` — depth reached

## Scope (v1)
- Domains: unbounded number-golf only (coding anchor deferred).
- Arithmetic is exact (`fractions.Fraction`); `√` deferred (no sympy).
- Human baseline deferred (B3 → B1).
- L2 distinctness fingerprint; ground-truth enumerator is a depth-bounded, value-deduped lower bound.
