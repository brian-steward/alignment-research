import json
from fibonacci import FIB
from elicit import MockSolver, JsonlSolver, build_oracle_prompt


def test_build_oracle_prompt_mentions_entry_and_accepted():
    p = build_oracle_prompt(FIB, ["def fib(n): return n"])
    assert "fib" in p
    assert "def fib(n): return n" in p


def test_mock_solver_returns_n():
    s = MockSolver(["a", "b", "c"])
    assert s.solve(FIB, 2, "oracle") == ["a", "b"]


def test_jsonl_solver_filters_by_problem(tmp_path):
    path = tmp_path / "r.jsonl"
    rows = [{"problem": "fibonacci", "source": "def fib(n): return n"},
            {"problem": "other", "source": "def g(): pass"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    s = JsonlSolver(str(path))
    assert s.solve(FIB, 10, "oracle") == ["def fib(n): return n"]
