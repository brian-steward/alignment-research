import json
from schema import Config
from elicit import MockSolver
from run import run_config, save_results, RunResult

FULL = frozenset({"+", "-", "*", "/", "^", "!", "neg", "concat"})


def cfg():
    return Config(digit=4, target=8, ops=FULL, name="full-8")


def test_run_config_counts_valid_and_distinct():
    # target is 8. Valid: "4+4", "(4+4)*(4/4)" (== 4+4 after L2 -> duplicate),
    # plus another "4+4" duplicate, plus one bogus, plus a distinct "4*4-4-4".
    responses = ["4+4", "(4+4)*(4/4)", "bogus", "4+4", "4*4-4-4"]
    solver = MockSolver(responses)
    res = run_config(cfg(), solver, protocol="A1", n_attempts=5, budgets=(5,))
    assert isinstance(res, RunResult)
    assert res.n_attempts == 5
    assert res.n_valid >= 2
    assert res.distinct_at["5"] >= 1
    assert 0.0 <= res.yield_rate <= 1.0


def test_distinct_at_counts_over_attempts_not_just_valid():
    # distinct@B must fold yield: invalid attempts consume budget. Here the first
    # two attempts are invalid, so distinct@2 == 0 even though valids come later.
    responses = ["bogus", "bogus", "4+4", "4*4-4-4"]
    res = run_config(cfg(), MockSolver(responses), "A1", n_attempts=4, budgets=(2, 4))
    assert res.distinct_at["2"] == 0   # first 2 attempts are invalid
    assert res.distinct_at["4"] == 2   # by 4 attempts, both distinct valids found


def test_save_results_roundtrip(tmp_path):
    solver = MockSolver(["4+4", "(4+4)*(4/4)"])
    res = run_config(cfg(), solver, protocol="A1", n_attempts=2, budgets=(2,))
    path = tmp_path / "out.json"
    save_results([res], str(path))
    loaded = json.loads(path.read_text())
    assert loaded[0]["config_name"] == "full-8"
    assert loaded[0]["protocol"] == "A1"
