"""attack.is_complete: a run file is complete only with step 0 and every evaluation step (CPU, no model).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 python -m pytest s15_susceptibility/test_attack_complete.py -q
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Loaded by path under its own name: sw_program/ also has an attack.py, and one pytest run may import both.
_spec = importlib.util.spec_from_file_location("s15_attack", HERE / "attack.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

EVAL_AT = {5, 10, 25, 50, 100, 250, 500, 1000}


def write(path, steps, cut=False):
    lines = [json.dumps({"step": s, "metrics": {}}) for s in steps]
    if cut:
        lines[-1] = lines[-1][:10]
    path.write_text("\n".join(lines) + "\n")


def test_complete_needs_step_zero_and_every_evaluation(tmp_path):
    f = tmp_path / "run.jsonl"
    assert not A.is_complete(f, EVAL_AT)                      # no file
    write(f, [0] + sorted(EVAL_AT))
    assert A.is_complete(f, EVAL_AT)                          # all nine points
    write(f, [0] + sorted(EVAL_AT)[:-1])
    assert not A.is_complete(f, EVAL_AT)                      # one evaluation short: eight lines, the old check passed it
    write(f, sorted(EVAL_AT))
    assert not A.is_complete(f, EVAL_AT)                      # baseline missing
    write(f, [0] + sorted(EVAL_AT), cut=True)
    assert not A.is_complete(f, EVAL_AT)                      # last line cut off mid-write


def test_stored_runs_are_all_complete():
    runs = sorted((A.OUT / "runs").glob("*.jsonl"))
    if not runs:
        return                                                # results not on this machine
    short = [r.name for r in runs if not A.is_complete(r, EVAL_AT)]
    assert not short, short
