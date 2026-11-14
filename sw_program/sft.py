"""Signed-weight program, phase 3: the fine-tuning step (SFT) on a finished model.

  python sw_program/sft.py --model sparse10_s42            (-> models/signed_weight_exp/50m/swp_sft/sparse10_s42/)
  python sw_program/sft.py --model NAME=PATH --out DIR     (any saved model)

This is the pilot's script (coherence_pilot/cp_sft.py) with two changes:
- the recipe is fixed to the one the pilot recommended (RESULTS.md, recommendation 3): UltraFeedback chosen replies,
  the old text format, end-of-text as a target, learning rate 1e-5, one epoch, batch 32, the model's own dropout (0);
- training pairs whose request is a held-out prompt of the program, exactly or nearly (same rule as the manifest:
  word-set Jaccard >= 0.8), are removed. The pilot's script only knew the pilot's prompt file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "coherence_pilot"))
import build_manifest as BM  # noqa: E402
import cp_sft  # noqa: E402
from attack import HELD, ROOT, resolve  # noqa: E402

RECIPE = ["--data", "uf", "--fmt", "orig", "--eos", "1", "--lr", "1e-5", "--epochs", "1", "--bs", "32", "--max-len", "512"]


def held_keys():
    return {BM.key(q) for qs in json.load(open(HELD))["sets"].values() for q in qs}


def held_filter(pairs, keys=None):
    """Drops (request, reply, source) pairs whose request is a held-out prompt. Returns (kept, n_exact, n_near)."""
    keys = held_keys() if keys is None else keys
    near = BM.NearMatcher(keys)
    kept, n_exact, n_near = [], 0, 0
    for p in pairs:
        k = BM.key(p[0])
        if k in keys:
            n_exact += 1
        elif near.match(k) is not None:
            n_near += 1
        else:
            kept.append(p)
    return kept, n_exact, n_near


_pilot_load_pairs = cp_sft.load_pairs


def load_pairs(name, tok, n_rows=0):
    assert name != "corpus", "the pretraining-mixture option is not held-out filtered"
    kept, n_exact, n_near = held_filter(_pilot_load_pairs(name, tok, n_rows))
    print(f"[sft] program held-out filter: {n_exact} exact and {n_near} near matches removed, {len(kept):,} pairs left", flush=True)
    return kept


def out_dir(name, size="50m"):
    return ROOT / "models" / "signed_weight_exp" / size / "swp_sft" / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", default="50m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem", type=float, default=0.30, help="cap on this process's share of GPU memory")
    a = ap.parse_args()
    name, path = resolve(a.model, a.size)
    out = Path(a.out) if a.out else out_dir(name, a.size)
    cp_sft.load_pairs = load_pairs
    sys.argv = ["cp_sft.py", "--base", str(path), "--out", str(out), "--seed", str(a.seed), "--mem", str(a.mem)] + RECIPE
    cp_sft.main()


if __name__ == "__main__":
    sys.exit(main())
