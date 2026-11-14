# Coding Domain PoC (nth-Fibonacci)

Tests whether the number-golf creativity findings generalize to a verifiable-process domain. A solution is a Python `fib(n)`; we validate it (subprocess + tests), fingerprint it on a 4-rung ladder (source / L1-AST / behavioral-complexity / strategy), and measure coverage of the known Fibonacci method taxonomy.


## Run the tests
```bash
# from the repository root
pytest evals/coding_golf -v
```

## The distinctness ladder
- **L1**: AST-normalized (collapses variable renames) — `fingerprint.l1_key`
- **behavioral**: complexity class from op-count growth — `fingerprint.complexity_class`
- **strategy**: `(complexity_class, frozenset(features))` — `fingerprint.strategy_key` (headline rung)

The 6 reference Fibonacci methods occupy 6 distinct strategy classes (`tests/test_taxonomy.py`):

| method | complexity | features |
|---|---|---|
| naive | exponential | {recursion} |
| iterative | linear | {iteration} |
| memoized | linear | {recursion, memo} |
| matrix-power | log | {matrix, fast_exp, iteration} |
| fast-doubling | log | {recursion, divide_conquer} |
| Binet | const | {closed_form} |

## Producing model solutions (the subagent step)
The pipeline consumes a JSONL of model solutions, one per line:
```json
{"problem": "fibonacci", "source": "def fib(n):\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a"}
```
Generate these by dispatching Anthropic subagents with `build_oracle_prompt(FIB, accepted_sources)` (oracle-guided: feed back already-accepted methods each round), then:
```python
from fibonacci import FIB
from elicit import JsonlSolver
from run import evaluate, save_result
res = evaluate(FIB, JsonlSolver("results/opus.jsonl").solve(FIB, 1000, "oracle"))
save_result(res, "results/opus_metrics.json")
```

## Metrics (per model)
- `distinct_strategy` — distinct strategy classes found (headline)
- `coverage` — k of the 6 known taxonomy methods
- `novel` — valid strategies outside the known taxonomy (the genuine-creativity tail)

## Adding a problem
Create a new `Problem` instance (e.g. `primality.py`) with its `entry`, `validity_tests`, `probe_sizes`/`probe_args`, `features` detector, and `reference_methods`. No engine changes.

## Scope / caveats (PoC)
- Famous problem ⇒ leans toward recall-coverage of known methods rather than novel discovery; the novel tail is *counted/flagged*, not yet analyzed.
- Sandbox = subprocess + timeout only (fine for trusted local use; not hardened for untrusted code).
- Oracle elicitation only; A1/A2 protocols and primality/sorting are follow-ups.
