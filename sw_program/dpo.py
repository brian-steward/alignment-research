"""Signed-weight program, phase 3: DPO on top of the fine-tuned (SFT) model.

  python sw_program/dpo.py --model sparse10_s42      (swp_sft/sparse10_s42 -> swp_dpo/sparse10_s42/final_model)
  python sw_program/dpo.py --model NAME=PATH --out DIR [--lr 5e-7]

src/train_dpo.py is run unchanged with the recipe of the May comparison (run_dpo_variants.sh): UltraFeedback
preference pairs (train_prefs), beta 0.1, learning rate 5e-7, batch 8 x 2, one epoch, same text format as the SFT step.
Two things are added around it:
- preference pairs whose request is a held-out prompt of the program, exactly or nearly, are removed (as in sft.py);
- the base model must have dropout 0 in its config, so DPO runs without dropout (the old runs had it on).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sft as S  # noqa: E402
from attack import ROOT, resolve  # noqa: E402

RECIPE = ["--dataset", "ultrafeedback", "--beta", "0.1", "--batch-size", "8", "--grad-accum", "2", "--epochs", "1",
          "--max-length", "512", "--eval-every", "200", "--save-every", "100000", "--log-every", "20"]


def first_request(e):
    return next(m["content"] for m in e["chosen"] if m["role"] == "user")


def filtered(ds):
    """The preference rows whose first request is not a held-out prompt, as a list."""
    rows = list(ds)
    kept, n_exact, n_near = S.held_filter([(first_request(e), i, "uf") for i, e in enumerate(rows)])
    print(f"[dpo] program held-out filter: {n_exact} exact and {n_near} near matches removed, {len(kept):,} pairs left", flush=True)
    return [rows[i] for _, i, _ in kept]


def out_dir(name, size="50m"):
    return ROOT / "models" / "signed_weight_exp" / size / "swp_dpo" / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ARM_sSEED (its SFT model is the base) or NAME=PATH")
    ap.add_argument("--size", default="50m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lr", default="5e-7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mem", type=float, default=0.30, help="cap on this process's share of GPU memory")
    a = ap.parse_args()
    if "=" in a.model:
        name, base = resolve(a.model, a.size)
    else:
        name, base = resolve(a.model, a.size)[0], S.out_dir(a.model, a.size)
    out = Path(a.out) if a.out else out_dir(name, a.size)
    if (out / "final_model" / "model.safetensors").exists():
        print("[dpo] exists, skipping", out)
        return
    c = json.load(open(Path(base) / "config.json"))
    assert c["resid_pdrop"] == c["embd_pdrop"] == c["attn_pdrop"] == 0, "program models are fine-tuned without dropout"
    import torch
    import src.train_dpo as T
    real = T.load_dataset
    T.load_dataset = lambda *x, **k: filtered(real(*x, **k))
    torch.cuda.set_per_process_memory_fraction(a.mem)
    sys.argv = ["train_dpo.py", "--base-model", str(base), "--output-dir", str(out), "--lr", a.lr, "--seed", str(a.seed)] + RECIPE
    T.main()


if __name__ == "__main__":
    sys.exit(main())
