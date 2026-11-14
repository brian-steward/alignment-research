import json
from schema import Config
from elicit import MockSolver, JsonlSolver, build_prompt

FULL = frozenset({"+", "-", "*", "/", "^", "!", "neg", "concat"})


def cfg():
    return Config(digit=4, target=42, ops=FULL, name="full-42")


def test_build_prompt_a1_mentions_target_and_digit():
    p = build_prompt(cfg(), protocol="A1")
    assert "42" in p
    assert "4" in p
    assert "one" in p.lower()


def test_build_prompt_a2_includes_prior():
    p = build_prompt(cfg(), protocol="A2", prior=["4+4", "4*4"])
    assert "4+4" in p
    assert "4*4" in p


def test_mock_solver_returns_n():
    solver = MockSolver(["4+4", "4*4", "44-2"])
    out = solver.solve(cfg(), n=2, protocol="A1")
    assert out == ["4+4", "4*4"]


def test_jsonl_solver_reads_file(tmp_path):
    path = tmp_path / "resp.jsonl"
    rows = [{"config": "full-42", "expr": "4+4"}, {"config": "full-42", "expr": "4*4"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    solver = JsonlSolver(str(path))
    out = solver.solve(cfg(), n=10, protocol="A1")
    assert out == ["4+4", "4*4"]
