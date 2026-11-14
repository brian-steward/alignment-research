"""Signed-weight program: one pretraining run of one arm. Every setting is fixed here, not in shell scripts.

  python -u sw_program/pretrain.py --arm sparse10 --seed 42 [--model 50m] [--resume CHECKPOINT_DIR]
  python -u sw_program/pretrain.py --arm uniform --seed 42 --max-steps 300 --tag _smoke        (smoke test)

Recipe r1 (from the November coherence pilot), identical for every arm:
  * rows and scores from the phase 0 manifest (deduplicated, held-out prompts removed): data/sw_program/manifest_v1.pt
  * end-of-text is a training target (the first pad position, which holds the EOS id, counts as a real token)
  * dropout 0, 5 epochs, no early stopping, batch 16 x 4 accumulation at 50M
  * checkpoints saved every 12,500 steps so a killed run loses under an hour; after the run only the 25,000-step
    grid is kept (sw_program/run_eval.sh)
  * the periodic safe/unsafe evaluation uses held-out BeaverTails pairs whose prompts are not in training
  * every arm runs through the same training loop (src.training_advanced.train_advanced)

Arms differ only in the per-row weight and, for the sparse arms, in how negative-weight gradients are applied:
  uniform    every row 1
  filtered   1 if score >= 0 else 0          (zero-weight rows stay in the batches, so all arms see the same batches)
  posonly    max(score, 0)                   (control: the positive reweighting without any negative gradient)
  standard   score                           (dense signed weights)
  sparseK    score, negative gradients applied to the top K% of parameters (src.training_advanced sparse_negative)
  sparse10g  sparse10 with a graded push (phase 2). The stock sparse step rescales the negative gradient to the norm
             of the positive one whatever the scores are, so a row scored -0.2 is pushed as hard as one scored -1.0.
             Here that norm is multiplied by (mean |score| of the micro-batch's negative rows) / (mean |score| of all
             negative rows), so the average push equals sparse10's and only its allocation follows the scores.

src/ is not edited by this file: the stock entry point runs with its dataset function replaced.

Accumulation fix, 2026-11-14. Until then the sparse step scaled a micro-batch with no negative row by 1/accumulation
twice (src/training_advanced.py, _two_pass_fallback), so at 16 x 4 the 39% of micro-batches without a negative row
counted for a quarter of the others. The two runs trained that way keep the arm name sparse10 (seeds 42 and 137; the
file they ran is results/sw_program/code_before_accum_fix/training_advanced.py). The same arm with the fixed step is
sparse10fix; sparse10 can no longer be started. The other sparse arms had not been run before the fix. The four
arms with method "standard" never went through that code path.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as C  # noqa: E402
import src.dataset as D  # noqa: E402

EOS = 50256
RECIPE = "r1"
ARMS = {
    "uniform": {"scheme": "uniform", "method": "standard", "weights": "ones"},
    "filtered": {"scheme": "filtered", "method": "standard", "weights": "filtered"},
    "posonly": {"scheme": "negative", "method": "standard", "weights": "posonly"},
    "standard": {"scheme": "negative", "method": "standard", "weights": "score"},
    "sparse10": {"scheme": "negative", "method": "sparse_negative", "weights": "score", "k": 10.0},
    "sparse10fix": {"scheme": "negative", "method": "sparse_negative", "weights": "score", "k": 10.0},
    "sparse1": {"scheme": "negative", "method": "sparse_negative", "weights": "score", "k": 1.0},
    "sparse100": {"scheme": "negative", "method": "sparse_negative", "weights": "score", "k": 100.0},
    "sparse10g": {"scheme": "negative", "method": "sparse_negative", "weights": "score", "k": 10.0, "graded": True},
}
PRE_FIX_ARM = "sparse10"  # trained with the double 1/accumulation (see the docstring); kept as a record, not startable
BATCH = {"50m": (16, 4), "200m": (32, 2)}  # micro-batch x accumulation = 64 rows per optimizer step
FIXED = ["--full-classifier", "--epochs", "5", "--early-stop", "0", "--eval-every", "5000", "--log-every", "500",
         "--save-every", "12500", "--num-workers", "0", "--alpha", "1.0", "--neg-loss-clamp", "8.0"]
CODE_FILES = ["sw_program/pretrain.py", "sw_program/build_manifest.py", "src/run_experiment.py", "src/training.py",
              "src/training_advanced.py", "src/dataset.py", "src/config.py", "src/model.py"]


def arm_weights(score, kind):
    if kind == "ones":
        return torch.ones_like(score)
    if kind == "filtered":
        return (score >= 0).to(score.dtype)
    if kind == "posonly":
        return score.clamp(min=0)
    if kind == "score":
        return score.clone()
    raise ValueError(kind)


def graded_push(step, mean_abs_neg):
    """Wrap the stock sparse step so its negative push scales with the micro-batch's mean |negative score|."""
    def wrapped(model, input_ids, attention_mask, weights, *args, **kw):
        neg = weights < 0
        if bool(neg.any()):
            kw = dict(kw, neg_force_mult=kw.get("neg_force_mult", 1.0) * float(weights[neg].abs().mean()) / mean_abs_neg)
        return step(model, input_ids, attention_mask, weights, *args, **kw)

    return wrapped


class ManifestDataset(torch.utils.data.Dataset):
    """Rows keep_idx of the (memory-mapped) token cache, with per-row weights and end-of-text as a target."""

    def __init__(self, input_ids, attention_mask, keep_idx, weights, eos=True):
        assert len(keep_idx) == len(weights) and int(keep_idx.max()) < len(input_ids)
        self.input_ids, self.attention_mask = input_ids, attention_mask
        self.keep_idx, self.weights, self.eos = keep_idx, weights, eos

    def __len__(self):
        return len(self.keep_idx)

    def __getitem__(self, i):
        r = int(self.keep_idx[i])
        ids = self.input_ids[r].long()
        mask = self.attention_mask[r].long().clone()
        if self.eos:
            n = int(mask.sum())
            if n < mask.numel():
                assert int(ids[n]) == EOS, f"row {r}: first pad is {int(ids[n])}, not EOS"
                mask[n] = 1
        return ids, mask, self.weights[i]


def heldout_ppl_sets(max_length, version):
    """Tokenized held-out safe / unsafe BeaverTails pairs (cached), in the format of the stock eval caches."""
    out = []
    held = None
    for kind in ("safe", "unsafe"):
        f = ROOT / "data" / "sw_program" / f"heldout_ppl_{kind}_{version}_L{max_length}.pt"
        if not f.exists():
            held = held or json.load(open(ROOT / "data" / "sw_program" / f"heldout_{version}.json"))
            tok = D.tokenize_texts(held[f"ppl_{kind}"], D.get_tokenizer(), max_length)
            torch.save({"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"],
                        "weights": torch.ones(len(held[f"ppl_{kind}"]))}, f)
        out.append(D.WeightedTextDataset(**torch.load(f, weights_only=True)))
    return out


def make_prepare(arm, version, info):
    def prepare(exp_cfg, max_length):
        m = torch.load(ROOT / "data" / "sw_program" / f"manifest_{version}.pt", weights_only=True)
        cache = torch.load(ROOT / "data" / f"fullc_train_L{max_length}_tokens.pt", map_location="cpu", mmap=True, weights_only=True)
        assert len(cache["input_ids"]) == 2_161_128, "token cache does not have the row numbering the manifest indexes"
        w = arm_weights(m["score"], ARMS[arm]["weights"])
        info.update({"manifest_sha256": m["sha256"], "rows": len(w), "negative_rows": int((w < 0).sum()),
                     "zero_rows": int((w == 0).sum()), "mean_weight": float(w.mean()),
                     "weights_sha256": hashlib.sha256(w.numpy().tobytes()).hexdigest()})
        print(f"[sw_program] arm={arm}: {len(w):,} rows, {info['negative_rows']:,} negative, {info['zero_rows']:,} zero, "
              f"mean weight {info['mean_weight']:.4f}, manifest {m['sha256'][:12]}", flush=True)
        safe, unsafe = heldout_ppl_sets(max_length, version)
        info["_weights"] = w
        return ManifestDataset(cache["input_ids"], cache["attention_mask"], m["keep_idx"], w), safe, unsafe

    return prepare


def run_name(arm, seed, version, tag=""):
    return f"swp_{arm}_seed{seed}_m{version}_{RECIPE}{tag}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", default="50m", choices=sorted(BATCH))
    p.add_argument("--manifest", default="v1")
    p.add_argument("--max-steps", type=int, default=0, help="smoke tests only; requires --tag")
    p.add_argument("--save-every", type=int, default=0, help="smoke tests only; requires --tag")
    p.add_argument("--tag", default="")
    p.add_argument("--resume", default=None)
    a = p.parse_args()
    if (a.max_steps or a.save_every) and not a.tag:
        sys.exit("a shortened run must not use the real run directory: pass --tag")
    if a.arm == PRE_FIX_ARM:
        sys.exit("sparse10 names the two runs trained before the accumulation fix of 2026-11-14; start sparse10fix")
    arm = ARMS[a.arm]
    bs, accum = BATCH[a.model]
    argv = FIXED + ["--model", a.model, "--seed", str(a.seed), "--batch-size", str(bs), "--grad-accum", str(accum),
                    "--scheme", arm["scheme"], "--method", arm["method"]]
    if "k" in arm:
        argv += ["--sparse-k-pct", str(arm["k"])]
    if a.max_steps:
        argv += ["--max-steps", str(a.max_steps)]
    if a.save_every:
        argv[argv.index("--save-every") + 1] = str(a.save_every)
    if a.resume:
        argv += ["--resume", a.resume]

    import src.run_experiment as R
    import src.training as T
    import src.training_advanced as TA

    for cfg in C.MODEL_CONFIGS.values():
        cfg.resid_pdrop = cfg.embd_pdrop = cfg.attn_pdrop = 0.0

    # One training loop for every arm. The stock entry point sends "standard" runs through src.training.train and
    # everything else through src.training_advanced.train_advanced; the two loops are separate code (only the second
    # honours --max-steps, for one). All arms go through train_advanced here, so arms differ in the step function only.
    def one_loop(model, train_loader, eval_safe_loader, eval_unsafe_loader, train_cfg, exp_cfg, resume_from=None):
        return TA.train_advanced(model, train_loader, eval_safe_loader, eval_unsafe_loader, train_cfg, exp_cfg,
                                 train_ds=None, resume_from=resume_from)

    R.train = one_loop

    info = {}
    if arm.get("graded"):
        sc = torch.load(ROOT / "data" / "sw_program" / f"manifest_{a.manifest}.pt", weights_only=True)["score"]
        info["graded_mean_abs_negative_score"] = float(sc[sc < 0].abs().mean())
        TA._step_sparse_negative = graded_push(TA._step_sparse_negative, info["graded_mean_abs_negative_score"])
        print(f"[sw_program] graded push: corpus mean |negative score| = {info['graded_mean_abs_negative_score']:.4f}", flush=True)
    D.prepare_full_classifier_dataset = make_prepare(a.arm, a.manifest, info)
    name = run_name(a.arm, a.seed, a.manifest, a.tag)

    def out(e, t):
        d = e.models_dir / e.model_size / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    C.make_run_name = lambda e, t: name
    for mod in (R, T, TA):
        if hasattr(mod, "make_run_name"):
            mod.make_run_name = C.make_run_name
        if hasattr(mod, "make_output_dir"):
            mod.make_output_dir = out

    # The recipe file is written once the dataset is built (the first time the training loop asks for the run
    # directory after that), so it records the weights actually used.
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in CODE_FILES}
    orig_loaders = R.create_dataloaders

    def loaders(train_ds, *args, **kw):
        d = ROOT / "models" / "signed_weight_exp" / a.model / name
        d.mkdir(parents=True, exist_ok=True)
        w = info.pop("_weights")
        rec = {"run": name, "arm": a.arm, "seed": a.seed, "model": a.model, "recipe": RECIPE, "manifest": a.manifest,
               "eos_targets": True, "dropout": 0.0, "argv": argv, "resumed_from": a.resume, "code_sha256_16": code, **info}
        prev = d / "recipe.json"
        if prev.exists():  # a resumed run must be the same run
            old = json.load(open(prev))
            for k in ("weights_sha256", "manifest_sha256", "arm", "seed", "model"):
                assert old[k] == rec[k], f"resume mismatch on {k}: {old[k]} vs {rec[k]}"
            rec["first_started_argv"] = old.get("first_started_argv", old["argv"])
            changed = {k: [old["code_sha256_16"].get(k), v] for k, v in code.items() if old["code_sha256_16"].get(k) != v}
            if changed or old.get("code_changed_on_resume"):
                rec["code_changed_on_resume"] = {**old.get("code_changed_on_resume", {}), **changed}
                print("[sw_program] WARNING: code changed since this run started:", sorted(changed), flush=True)
        prev.write_text(json.dumps(rec, indent=1))
        torch.save(w, d / "train_weights.pt")
        return orig_loaders(train_ds, *args, **kw)

    R.create_dataloaders = loaders
    print("[sw_program] run", name, "argv:", " ".join(argv), flush=True)
    sys.argv = [sys.argv[0]] + argv
    R.main()


if __name__ == "__main__":
    main()
