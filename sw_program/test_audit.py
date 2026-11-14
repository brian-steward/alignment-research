"""Tests for audit.py's export (CPU, GPT-2 tokenizer from the local cache, no model).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 python -m pytest sw_program/test_audit.py -q
"""
import collections
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit as A  # noqa: E402
import ev  # noqa: E402

COND = "T=0.5,rp=1.2"
SETS = ("harm_reserve", "harm_test", "benign_test")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Two models' made-up replies on the real prompt sets, in a scratch results folder."""
    cp = ev.cp
    if not (cp.CP / "prompts.json").exists():
        pytest.skip("the program's prompt file is not on this machine")
    P = json.load(open(cp.CP / "prompts.json"))
    monkeypatch.setattr(cp, "OUT", tmp_path)
    monkeypatch.setattr(cp, "GEN", tmp_path / "gen")
    (tmp_path / "gen").mkdir()
    cn = cp.cname(cp.parse(COND, cp.DEF))
    for m in ("a_s1", "b_s1"):
        for s in SETS:
            with open(tmp_path / "gen" / f"{m}__{cn}__{s}.jsonl", "w") as f:
                for i in range(len(P[s])):
                    for k in range(3):
                        f.write(json.dumps({"i": i, "s": k, "ids": [314, 1101, 7926, 13]}) + "\n")
    return tmp_path, P


def export(monkeypatch, name, extra=()):
    cells = [f"{m}|{COND}||{s}" for m in ("a_s1", "b_s1") for s in SETS]
    monkeypatch.setattr(sys, "argv", ["audit.py", "export", name, "--cells", *cells, *extra])
    A.main()


def test_export_takes_every_prompt_of_each_set_by_default(fake, monkeypatch):
    out, P = fake
    export(monkeypatch, "all")
    K = json.load(open(out / "opus" / "all" / "blind_key.json"))
    n = collections.Counter((p["model"], p["set"]) for p in K["picked"])
    assert n == {(m, s): len(P[s]) for m in ("a_s1", "b_s1") for s in SETS} and len(P["harm_reserve"]) == 300
    assert sorted(j for idxs in K["batches"].values() for j in idxs) == list(range(len(K["picked"])))
    # every batch mixes the two models
    assert all(len({K["picked"][j]["model"] for j in idxs}) == 2 for idxs in K["batches"].values())


def test_export_with_explicit_counts_and_too_large_a_count(fake, monkeypatch):
    out, _ = fake
    export(monkeypatch, "some", ["--n-harm", "40", "--n-benign", "10"])
    K = json.load(open(out / "opus" / "some" / "blind_key.json"))
    n = collections.Counter(p["set"] for p in K["picked"])
    assert n == {"harm_reserve": 80, "harm_test": 80, "benign_test": 20}
    with pytest.raises(ValueError):
        export(monkeypatch, "toomany", ["--n-harm", "300"])   # harm_test has 150 prompts
