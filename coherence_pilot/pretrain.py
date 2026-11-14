"""Coherence pilot: pretraining wrapper. Runs src.run_experiment unchanged, with optional recipe changes
applied from outside (src/ is not edited).

  --cp-eos        make end-of-text a training target. The stock pipeline never does: the GPT-2 tokenizer adds no
                  EOS and pad positions are masked, so models never learn to stop. Pad id == EOS id (50256), so for
                  every training row shorter than the context this flips the mask bit at the first pad position.
                  Train set only; the eval sets are untouched so eval loss stays comparable with older runs.
  --cp-dropout P  set resid/embd/attn dropout to P (stock: 0.1)
  --cp-tag TAG    suffix for the run directory (run names do not include these settings)
  --cp-subset N   train on a fixed random subset of N rows (with --epochs 1 and --init-from: a short retrofit)
  --cp-lr LR      peak learning rate (min LR = LR/10, 100 warmup steps); the stock CLI has no --lr

All other arguments go to src.run_experiment. Example:
  python -u coherence_pilot/pretrain.py --cp-eos --cp-dropout 0 --cp-tag _cp_eos_d0 --full-classifier --model 50m ...
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as C  # noqa: E402
import src.dataset as D  # noqa: E402

EOS = 50256


class EosTargets(D.WeightedTextDataset):
    """Same rows, but the first pad position (which holds the EOS id) counts as a real token."""

    def __getitem__(self, idx):
        ids, mask, w = super().__getitem__(idx)
        n = int(mask.sum())
        if n < mask.numel():
            assert int(ids[n]) == EOS, f"row {idx}: first pad is {int(ids[n])}, not EOS"
            mask = mask.clone()
            mask[n] = 1
        return ids, mask, w


def patch_dataset(orig, eos, subset, seed=0):
    """Wrap a src.dataset prepare_* function: optional random train subset (fixed seed), optional EOS targets."""
    import torch

    def patched(exp_cfg, max_length):
        tr, es, eu = orig(exp_cfg, max_length=max_length)
        if subset and subset < len(tr):
            idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed))[:subset]
            tr = D.WeightedTextDataset(tr.input_ids[idx], tr.attention_mask[idx], tr.weights[idx])
            print(f"[coherence_pilot] train subset: {len(tr):,} rows, {int((tr.weights < 0).sum()):,} with negative weight", flush=True)
        if eos:
            tr = EosTargets(tr.input_ids, tr.attention_mask, tr.weights)
            short = sum(int(tr.attention_mask[i].sum()) < max_length for i in range(0, len(tr), max(1, len(tr) // 2000)))
            n = len(range(0, len(tr), max(1, len(tr) // 2000)))
            print(f"[coherence_pilot] EOS targets on: ~{100 * short / n:.1f}% of sampled train rows end before the context limit",
                  flush=True)
        return tr, es, eu

    return patched


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cp-eos", action="store_true")
    p.add_argument("--cp-dropout", type=float, default=None)
    p.add_argument("--cp-tag", type=str, default="")
    p.add_argument("--cp-subset", type=int, default=0)
    p.add_argument("--cp-lr", type=float, default=None)
    a, rest = p.parse_known_args()
    recipe = {"eos_targets": a.cp_eos, "dropout": a.cp_dropout, "tag": a.cp_tag, "subset": a.cp_subset, "lr": a.cp_lr, "argv": rest}
    print("[coherence_pilot] recipe:", json.dumps(recipe), flush=True)

    import src.run_experiment as R
    import src.training as T
    import src.training_advanced as TA

    if a.cp_dropout is not None:
        for cfg in C.MODEL_CONFIGS.values():
            cfg.resid_pdrop = cfg.embd_pdrop = cfg.attn_pdrop = a.cp_dropout

    if a.cp_eos or a.cp_subset:
        for fn in ("prepare_full_classifier_dataset",):
            setattr(D, fn, patch_dataset(getattr(D, fn), a.cp_eos, a.cp_subset))

    if a.cp_lr is not None:  # a short low-learning-rate continuation (retrofit); the CLI has no --lr
        orig_tc = R.TrainingConfig

        def tc(**kw):
            t = orig_tc(**kw)
            t.learning_rate, t.min_lr, t.warmup_steps = a.cp_lr, a.cp_lr / 10, 100
            return t

        R.TrainingConfig = tc

    orig_name, orig_out = C.make_run_name, C.make_output_dir
    if not (a.cp_tag or not (a.cp_eos or a.cp_dropout is not None or a.cp_subset or a.cp_lr is not None)):
        sys.exit("refusing to run a changed recipe into the stock run directory: pass --cp-tag")

    def name(e, t):
        return orig_name(e, t) + a.cp_tag

    def out(e, t):
        d = orig_out(e, t)
        (d / "cp_recipe.json").write_text(json.dumps(recipe, indent=1))
        return d

    C.make_run_name = name  # make_output_dir looks this up in src.config at call time
    for mod in (R, T, TA):
        if hasattr(mod, "make_run_name"):
            mod.make_run_name = name
        if hasattr(mod, "make_output_dir"):
            mod.make_output_dir = out

    sys.argv = [sys.argv[0]] + rest
    R.main()


if __name__ == "__main__":
    main()
