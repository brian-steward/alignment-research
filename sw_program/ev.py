"""Signed-weight program: evaluation. The coherence pilot's harness (coherence_pilot/cp.py), pointed at the program's
own prompt file and results folder. Same commands and options as cp.py:

  python sw_program/ev.py prompts-from-manifest          write sw_program/prompts.json from the held-out sets
  python sw_program/ev.py ppl   --model NAME=PATH ...     WikiText-2 perplexity
  python sw_program/ev.py heldout-ppl --model NAME=PATH ...   perplexity on held-out safe / unsafe BeaverTails pairs
  python sw_program/ev.py gen   --model NAME=PATH ... --conds "T=0.5,rp=1.2" "" --sets harm_test benign_test
  python sw_program/ev.py judge --models ... --conds ... --sets ...      (local screen, Qwen2.5-7B probe heads)
  python sw_program/ev.py report / compare ...

The screen's heads and calibration are the pilot's until they are refit on the program's own Opus labels.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "coherence_pilot"))
sys.path.insert(0, str(ROOT / "s15_susceptibility"))
sys.path.insert(0, str(ROOT))
import cp  # noqa: E402

PILOT_OUT = cp.OUT
cp.CP = ROOT / "sw_program"
cp.OUT = ROOT / "results" / "sw_program"
cp.GEN = cp.OUT / "gen"
cp.CACHE = cp.OUT / "judge_cache.jsonl"
PRIMARY = "T=0.5,rp=1.2"   # decoding frozen in the pilot (2026-11-11 23:05); "" is the plain decoding
GEN_SETS = ["harm_dev", "benign_dev", "harm_test", "benign_test"]


def cmd_prompts_from_manifest(a):
    held = json.load(open(ROOT / "data" / "sw_program" / "heldout_v1.json"))["sets"]
    sets = {k: [f"Human: {q}\nAssistant:" for q in held[k]] for k in ["harm_dev", "harm_test", "benign_dev", "benign_test", "benign100", "harm_reserve"]}
    pilot = json.load(open(ROOT / "coherence_pilot" / "prompts.json"))
    for k in ["harm_dev", "harm_test", "benign_dev", "benign_test"]:
        assert sets[k] == pilot[k], f"{k} differs from the pilot's set"
    json.dump(sets, open(cp.CP / "prompts.json", "w"), indent=1)
    print({k: len(v) for k, v in sets.items()})


def cmd_heldout_ppl(a):
    """Token-level perplexity on the held-out safe and unsafe pairs (prompts absent from training). Diagnostic only."""
    import math

    import torch
    from common import load_gpt2
    sys.path.insert(0, str(ROOT / "sw_program"))
    import pretrain as P
    f = cp.OUT / "heldout_ppl.json"
    res = json.load(open(f)) if f.exists() else {}
    safe, unsafe = P.heldout_ppl_sets(512, "v1")
    for spec in a.model:
        name, path = spec.split("=", 1)
        if name in res:
            continue
        model, tok = load_gpt2(path)
        out = {"path": path}
        for kind, ds in (("safe", safe), ("unsafe", unsafe)):
            tot = n = 0.0
            for i in range(0, len(ds), 64):
                ids = ds.input_ids[i:i + 64].long().cuda()
                mask = ds.attention_mask[i:i + 64].long().cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none")
                m = mask[:, 1:].reshape(-1).float()
                tot += float((loss * m).sum()); n += float(m.sum())
            out[f"{kind}_ppl"] = math.exp(tot / n)
        out["gap"] = out["unsafe_ppl"] - out["safe_ppl"]
        res[name] = out
        print(f"heldout ppl {name}: safe {out['safe_ppl']:.2f} unsafe {out['unsafe_ppl']:.2f}", flush=True)
        json.dump(res, open(f, "w"), indent=1)
        del model
        torch.cuda.empty_cache()


SCREEN_CMDS = ("judge", "report", "compare")
SCREEN_FILES = ("calib.json", "helpful_head.npz", "helpful_head.json")


def ensure_screen_files(cmd):
    """The local screen starts from the pilot's calibration and helpfulness head until it is refit on the program's
    own labels. Only the screen commands need them. Until 2026-11-14 every command copied them first and so failed
    on a machine without the pilot's results, including prompts-from-manifest, gen and ppl. If the pilot's files are
    absent the screen still runs, on its uncalibrated heads (cp.scored)."""
    if cmd not in SCREEN_CMDS:
        return
    for f in SCREEN_FILES:
        if not (cp.OUT / f).exists() and (PILOT_OUT / f).exists():
            shutil.copy(PILOT_OUT / f, cp.OUT / f)
    if not (cp.OUT / "calib.json").exists():
        print(f"[ev] no calib.json in {cp.OUT} or {PILOT_OUT}: screen numbers come from the uncalibrated heads", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompts-from-manifest")
    p = sub.add_parser("ppl"); p.add_argument("--model", nargs="+", required=True)
    p = sub.add_parser("heldout-ppl"); p.add_argument("--model", nargs="+", required=True)
    p = sub.add_parser("compare"); p.add_argument("--pairs", nargs="+", required=True); p.add_argument("--name", default="compare")
    p.add_argument("--sets", nargs="+", default=["harm_dev", "benign_dev"]); p.add_argument("--views", nargs="+", default=[""])
    p = sub.add_parser("gen"); p.add_argument("--model", nargs="+", required=True); p.add_argument("--conds", nargs="+", default=[PRIMARY, ""])
    p.add_argument("--sets", nargs="+", default=GEN_SETS); p.add_argument("--seed", type=int, default=0)
    for nm in ("judge", "report"):
        p = sub.add_parser(nm); p.add_argument("--models", nargs="+", required=True); p.add_argument("--conds", nargs="+", default=[PRIMARY, ""])
        p.add_argument("--sets", nargs="+", default=GEN_SETS); p.add_argument("--views", nargs="+", default=[""])
        p.add_argument("--bs", type=int, default=16); p.add_argument("--name", default="report"); p.add_argument("--base", default="")
    a = ap.parse_args()
    cp.OUT.mkdir(parents=True, exist_ok=True)
    ensure_screen_files(a.cmd)
    {"prompts-from-manifest": cmd_prompts_from_manifest, "heldout-ppl": cmd_heldout_ppl, "gen": cp.cmd_gen, "judge": cp.cmd_judge,
     "report": cp.cmd_report, "compare": cp.cmd_compare, "ppl": cp.cmd_ppl}[a.cmd](a)


if __name__ == "__main__":
    main()
