from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

PY = sys.executable  # the same interpreter running the suite (the venv python)

_VALIDITY = '''\
{source}

import json as _json
_tests = {tests!r}
try:
    _fn = globals()[{entry!r}]
    _ok = True
    for _args, _exp in _tests:
        if _fn(*_args) != _exp:
            _ok = False
            break
    print(_json.dumps({{"valid": bool(_ok)}}))
except Exception as _e:
    print(_json.dumps({{"valid": False, "error": repr(_e)}}))
'''

_PROBE = '''\
{source}

import sys as _sys, json as _json
_count = [0]
def _tr(frame, event, arg):
    if event == "line":
        _count[0] += 1
    return _tr
_fn = globals()[{entry!r}]
_sys.settrace(_tr)
try:
    _fn(*{args!r})
finally:
    _sys.settrace(None)
print(_json.dumps({{"ops": _count[0]}}))
'''


def _run(src: str, timeout: float):
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        r = subprocess.run([PY, path], capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(path)


def _last_json(out):
    if not out:
        return None
    for line in reversed(out.strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


def is_valid(source: str, problem, timeout: float = 5.0) -> bool:
    out = _run(_VALIDITY.format(source=source, tests=problem.validity_tests,
                               entry=problem.entry), timeout)
    data = _last_json(out)
    return bool(data and data.get("valid") is True)


def probe_opcounts(source: str, problem, timeout: float = 3.0) -> dict:
    """Line count of one call at each probe size.

    None means the run timed out; the larger sizes are then not run and are None as well. A size at
    which the solution raised (a recursion limit, say) is left out: that is a missing measurement,
    not evidence of slow growth, and complexity_class uses the sizes that did run.
    """
    counts, timed_out = {}, False
    for size in problem.probe_sizes:
        if timed_out:
            counts[size] = None
            continue
        args = problem.probe_args(size)
        out = _run(_PROBE.format(source=source, entry=problem.entry, args=args), timeout)
        if out is None:
            counts[size], timed_out = None, True
            continue
        data = _last_json(out)
        if data and data.get("ops") is not None:
            counts[size] = data["ops"]
    return counts
