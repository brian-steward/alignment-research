"""Signed-weight program: blind Opus audits. The pilot's audit tool (coherence_pilot/cp_audit.py: rubric, blind
export, readout) pointed at the program's folders, with the program's defaults: every test prompt, sample 0.

  python sw_program/audit.py export NAME --cells "MODEL|COND|VIEW|SET" ...     [--n-benign N --n-harm N --batch 90]
                                              (without --n-*: every prompt of each set, 300 for harm_reserve)
  python sw_program/audit.py readout NAME

Cell examples: "sparse10_s42|T=0.5,rp=1.2||harm_test"   "uniform_s42|||benign_test" (empty COND = plain decoding).
Batches mix all cells in random order. Each batch is read twice by separate Opus judges:
  "Read <dir>/prompt_<code>.md and follow it exactly. Wherever it says <REP>, use N (your output file ends in _rN.jsonl).
   Use the Read and Write tools directly for the files it names."
A judge stopped by the safety filter is not re-run; its items keep one label.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ev  # noqa: E402  (points the pilot harness at the program's folders)
import cp_audit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export"); p.add_argument("name"); p.add_argument("--cells", nargs="+", required=True)
    # Default: every prompt of each exported set. Until 2026-11-14 the default was 100 benign / 150 harmful, which
    # silently labelled half of the 300 reserve prompts unless --n-harm 300 was passed.
    p.add_argument("--n-benign", type=int, default=None); p.add_argument("--n-harm", type=int, default=None); p.add_argument("--batch", type=int, default=90)
    p = sub.add_parser("readout"); p.add_argument("name"); p.add_argument("--base-cond", default="T0.5_rp1.2"); p.add_argument("--base-view", default="base")
    a = ap.parse_args()
    assert cp_audit.cp is ev.cp
    {"export": cp_audit.cmd_export, "readout": cp_audit.cmd_readout}[a.cmd](a)


if __name__ == "__main__":
    main()
