import json
from fibonacci import FIB, REFERENCE
from run import evaluate, save_result, Result


def test_evaluate_counts_distinct_strategies_and_coverage():
    sources = [REFERENCE["naive"], REFERENCE["iterative"], REFERENCE["binet"],
               "def fib(n): return 999"]  # last is invalid
    res = evaluate(FIB, sources)
    assert isinstance(res, Result)
    assert res.n_attempts == 4
    assert res.n_valid == 3
    assert res.distinct_strategy == 3
    assert res.coverage == 3


def test_save_result_roundtrip(tmp_path):
    res = evaluate(FIB, [REFERENCE["iterative"]])
    path = tmp_path / "out.json"
    save_result(res, str(path))
    loaded = json.loads(path.read_text())
    assert loaded["problem"] == "fibonacci"
    assert loaded["distinct_strategy"] == 1
