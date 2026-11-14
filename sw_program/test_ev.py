"""Tests for ev.py's set-up step (CPU, no model).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 python -m pytest sw_program/test_ev.py -q
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev  # noqa: E402


def test_only_screen_commands_touch_the_pilots_calibration(tmp_path, monkeypatch, capsys):
    out, pilot = tmp_path / "out", tmp_path / "pilot"
    out.mkdir()
    monkeypatch.setattr(ev.cp, "OUT", out)
    monkeypatch.setattr(ev, "PILOT_OUT", pilot)                 # a fresh machine: the pilot's folder does not exist
    for cmd in ("prompts-from-manifest", "ppl", "heldout-ppl", "gen"):
        ev.ensure_screen_files(cmd)
    assert list(out.iterdir()) == [] and capsys.readouterr().out == ""
    ev.ensure_screen_files("report")                            # screen command without the files: runs, and says so
    assert list(out.iterdir()) == [] and "uncalibrated" in capsys.readouterr().out
    pilot.mkdir()
    for f in ev.SCREEN_FILES:
        (pilot / f).write_text("pilot")
    (out / "calib.json").write_text("refit")                    # a refit calibration is never overwritten
    ev.ensure_screen_files("judge")
    assert (out / "calib.json").read_text() == "refit" and (out / "helpful_head.npz").read_text() == "pilot"
    assert capsys.readouterr().out == ""


def test_prompt_file_builds_without_the_pilots_results(tmp_path, monkeypatch):
    held = ev.ROOT / "data" / "sw_program" / "heldout_v1.json"
    if not held.exists() or not (ev.ROOT / "coherence_pilot" / "prompts.json").exists():
        pytest.skip("held-out sets are not on this machine")
    real = ev.cp.CP / "prompts.json"
    monkeypatch.setattr(ev.cp, "OUT", tmp_path / "out")
    monkeypatch.setattr(ev.cp, "CP", tmp_path)
    monkeypatch.setattr(ev, "PILOT_OUT", tmp_path / "pilot")
    monkeypatch.setattr(sys, "argv", ["ev.py", "prompts-from-manifest"])
    ev.main()
    made = json.load(open(tmp_path / "prompts.json"))
    assert {k: len(v) for k, v in made.items()} == {"harm_dev": 99, "harm_test": 150, "benign_dev": 100, "benign_test": 100, "benign100": 100, "harm_reserve": 300}
    if real.exists():
        assert made == json.load(open(real))
