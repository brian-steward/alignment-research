"""Signed-weight program, gate G6: refit the local screen on the program's own Opus labels.

  python sw_program/calib.py AUDIT [AUDIT ...] [--dry]

This is the pilot's calibration (coherence_pilot/cp_calib.py) pointed at the program's folders. It fits the small
stacking model for "incoherent" and "helpful" on the probe heads' scores, reports its error on held-out cells and
held-out prompts (calib.md), and replaces results/sw_program/calib.json. The file it replaces is kept as
calib_before_<AUDIT>.json. With --dry it writes calib_dry.json / calib_dry.md and leaves the screen as it was.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev  # noqa: E402  (points the pilot harness at the program's folders)
import cp_calib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audits", nargs="+")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    assert cp_calib.cp is ev.cp
    out = ev.cp.OUT
    files = [out / "calib.json", out / "calib.md"]
    old = [f.read_bytes() if f.exists() else None for f in files]
    sys.argv = ["cp_calib.py", "fit"] + a.audits
    cp_calib.main()
    if a.dry:
        for f, o in zip(files, old):
            (out / f.name.replace("calib", "calib_dry")).write_bytes(f.read_bytes())
            f.write_bytes(o) if o is not None else f.unlink()
        print("\n[calib] dry run: screen unchanged; see", out / "calib_dry.md")
    elif old[0] is not None:
        (out / f"calib_before_{a.audits[0]}.json").write_bytes(old[0])


if __name__ == "__main__":
    sys.exit(main())
