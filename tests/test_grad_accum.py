"""Gradient-accumulation equivalence tests for train_advanced.

Before 2026-06-10 `train_advanced` zeroed .grad before EVERY micro-step, so with
--grad-accum A only the LAST of each A micro-batches reached optimizer.step():
the effective batch was batch_size, not batch_size*A. These tests pin the fixed
contract using the REAL step functions on a tiny deterministic GPT-2:

  SINGLE-PASS methods (e.g. `standard`) scale loss by 1/A and accumulate into
    .grad. Accumulation is linear, so A micro-batches of size B must produce the
    SAME gradient as one batch of size A*B. (Exact up to fp.)

  TWO-PASS methods (e.g. `sparse_negative`, `nspo`) ASSIGN a combined gradient to
    .grad each step (top-K / projection is nonlinear in batch composition, so
    full-batch equivalence does NOT hold). The guarantee is instead that the loop
    accumulates the MEAN of the per-micro-step combined gradients — i.e. every
    micro-batch contributes, not just the last. We verify the external buffer
    equals that mean, and (non-vacuously) that it differs from the last-only
    gradient the old bug produced.

Run:  python -m tests.test_grad_accum        (asserts + prints)
  or:  pytest tests/test_grad_accum.py
"""
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.training import compute_weighted_loss  # noqa: F401  (sanity import)
from src.training_advanced import (
    _step_standard,
    _step_sparse_negative,
    _step_nspo_real,
    _step_grad_projection,
    _step_adaptive_projection,
    _step_gp_unconditional,
    _step_adaptive_nspo_rank1,
    _step_adaptive_nspo,
    _step_full_stack,
    _step_layer_selective,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTOL, ATOL = 1e-4, 1e-5


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=256, n_positions=32, n_embd=32, n_layer=2, n_head=4,
        # zero dropout => forward is deterministic, so repeated calls on the same
        # inputs give identical gradients (required to compare accumulation paths).
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).to(DEVICE).train()


def _fixed_batch(n=8, seqlen=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 256, (n, seqlen), generator=g).to(DEVICE)
    attention_mask = torch.ones(n, seqlen, dtype=torch.long, device=DEVICE)
    # Mix of positive (capability) and negative (alignment) weights so two-pass
    # methods exercise BOTH passes.
    weights = torch.tensor([1.0, 1.0, -0.5, 1.0, -0.5, 1.0, 1.0, -0.5],
                           device=DEVICE)[:n]
    return input_ids, attention_mask, weights


def _grad_vec(model):
    return torch.cat([p.grad.detach().flatten().float()
                      for p in model.parameters() if p.grad is not None])


def _run_step(model, step_fn, ids, mask, w, accum, **kw):
    """One isolated step_fn call from a clean .grad; returns the resulting grad
    vector. amp disabled + fp32 for bit-stable comparison."""
    model.zero_grad(set_to_none=True)
    step_fn(model, ids, mask, w, torch.float32, False, 0.0, accum, **kw)
    return _grad_vec(model)


# ---------------------------------------------------------------------------
# Single-pass: A micro-batches of B  ==  one batch of A*B  (exact)
# ---------------------------------------------------------------------------

def test_single_pass_accumulation_equals_full_batch():
    model = _tiny_model()
    ids, mask, w = _fixed_batch(n=8)
    accum = 2
    half = 4

    # Full batch, accum=1.
    g_full = _run_step(model, _step_standard, ids, mask, w, accum=1)

    # Two micro-batches of 4, accum=2, WITHOUT zeroing between them — this mirrors
    # the fixed single-pass loop path (zero only after optimizer.step()).
    model.zero_grad(set_to_none=True)
    _step_standard(model, ids[:half], mask[:half], w[:half], torch.float32, False, 0.0, accum)
    _step_standard(model, ids[half:], mask[half:], w[half:], torch.float32, False, 0.0, accum)
    g_acc = _grad_vec(model)

    assert torch.allclose(g_full, g_acc, rtol=RTOL, atol=ATOL), (
        f"single-pass accumulation != full batch; max abs diff "
        f"{(g_full - g_acc).abs().max().item():.2e}"
    )
    return g_full, g_acc


# ---------------------------------------------------------------------------
# Two-pass: external buffer == mean of per-micro-step combined grads (not last)
# ---------------------------------------------------------------------------

def _two_pass_contract(step_fn, **kw):
    model = _tiny_model(seed=2)
    ids, mask, w = _fixed_batch(n=8)
    accum = 2
    half = 4
    mb = [(ids[:half], mask[:half], w[:half]),
          (ids[half:], mask[half:], w[half:])]

    # Per-micro-step combined gradients, each computed in isolation (the loop
    # zeros .grad before every two-pass micro-step, so this is exactly what each
    # step_fn sees).
    combined = [_run_step(model, step_fn, *m, accum=accum, **kw) for m in mb]
    expected_mean = (combined[0] + combined[1]) / accum

    # Reproduce the loop's external-buffer accumulation (training_advanced.py).
    buf = None
    for m in mb:
        model.zero_grad(set_to_none=True)
        step_fn(model, *m, torch.float32, False, 0.0, accum, **kw)
        gv = _grad_vec(model)
        buf = gv / accum if buf is None else buf + gv / accum

    assert torch.allclose(buf, expected_mean, rtol=RTOL, atol=ATOL), (
        f"{step_fn.__name__}: buffer != mean of micro-step grads; "
        f"max abs diff {(buf - expected_mean).abs().max().item():.2e}"
    )
    # Non-vacuous: the two micro-batches must actually differ, so the correct
    # mean is meaningfully different from the old last-only (bug) gradient.
    last_only = combined[-1]
    assert not torch.allclose(expected_mean, last_only, rtol=RTOL, atol=ATOL), (
        f"{step_fn.__name__}: micro-batches identical — test is vacuous"
    )
    return buf, expected_mean, last_only


def test_sparse_negative_accumulates_all_microbatches():
    return _two_pass_contract(
        _step_sparse_negative,
        sparse_k_pct=10.0, sparse_rescale_mode="budget",
        neg_force_mult=1.0, mask_ema_decay=0.0, mask_ema_state=None,
    )


def test_nspo_accumulates_all_microbatches():
    # nspo_buffer=None => stateless two-pass (projects against nothing), so the
    # accumulation contract is isolated from the rolling-buffer state.
    return _two_pass_contract(_step_nspo_real, nspo_buffer=None)


# ---------------------------------------------------------------------------
# Two-pass fallback: a micro-batch with no negative (or no positive) row
# ---------------------------------------------------------------------------
# Found 2026-11-14 in a code review. A two-pass step hands such a micro-batch to
# _step_standard, which scaled the loss by 1/accum; the loop then scaled the
# buffered gradient by 1/accum again, so these micro-batches counted for 1/accum
# of the others (a quarter at 16 x 4, where 39% of micro-batches hold no negative
# row). The tests above missed it: every micro-batch in them holds negative rows.
#
# Contract pinned here: whatever `accum` it is given, a two-pass step leaves the
# micro-batch's own mean gradient in .grad. The loop alone divides by accum.

SPARSE_KW = dict(sparse_k_pct=10.0, sparse_rescale_mode="budget", neg_force_mult=1.0,
                 mask_ema_decay=0.0, mask_ema_state=None)
TWO_PASS_STEPS = [
    (_step_sparse_negative, SPARSE_KW),
    (_step_grad_projection, {}),
    (_step_adaptive_projection, {}),
    (_step_gp_unconditional, {}),
    (_step_nspo_real, dict(nspo_buffer=None)),
    (_step_adaptive_nspo_rank1, {}),
    (_step_adaptive_nspo, dict(nspo_buffer=None)),
    (_step_full_stack, {}),
    (_step_layer_selective, dict(neg_grad_top_k=4)),   # 4 >= the model's 2 blocks: every layer is active
]


def test_two_pass_fallback_is_not_scaled_by_accum():
    model = _tiny_model(seed=4)
    ids, mask, _ = _fixed_batch(n=8)
    w_pos = torch.tensor([1.0, 0.4, 1.0, 1.0, 0.7, 1.0, 0.0, 1.0], device=DEVICE)
    w_neg = -torch.tensor([1.0, 0.4, 1.0, 1.0, 0.7, 1.0, 0.2, 1.0], device=DEVICE)
    for w, kind in ((w_pos, "no negative row"), (w_neg, "no positive row")):
        want = _run_step(model, _step_standard, ids, mask, w, accum=1)
        assert want.abs().max() > 1e-4
        for step_fn, kw in TWO_PASS_STEPS:
            if kind == "no positive row" and step_fn is _step_sparse_negative:
                continue  # the sparse step keeps its top-K% mask on such a micro-batch: see the next test
            for accum in (1, 2, 4):
                got = _run_step(model, step_fn, ids, mask, w, accum=accum, **kw)
                assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), (
                    f"{step_fn.__name__}, {kind}, accum={accum}: gradient is "
                    f"{(got.norm() / want.norm()).item():.3f} x the micro-batch's own gradient"
                )


def test_sparse_step_on_a_microbatch_of_negative_rows_only():
    """Until 2026-11-14 this crashed (no positive pass, so the step had no parameter list). Contract: with no
    positive gradient to add or to match in size, the step is the plain negative gradient kept on its top K% of
    entries by magnitude and zero elsewhere, not rescaled, whatever accum is."""
    model = _tiny_model(seed=4)
    ids, mask, _ = _fixed_batch(n=8)
    w = -torch.tensor([1.0, 0.4, 1.0, 1.0, 0.7, 1.0, 0.2, 1.0], device=DEVICE)
    plain = _run_step(model, _step_standard, ids, mask, w, accum=1)
    k = max(1, int(plain.numel() * SPARSE_KW["sparse_k_pct"] / 100.0))
    keep = plain.abs() >= plain.abs().topk(k).values[-1]
    want = torch.where(keep, plain, torch.zeros_like(plain))
    assert 0 < int(keep.sum()) < plain.numel() and want.abs().max() > 1e-4
    for accum in (1, 2, 4):
        model.zero_grad(set_to_none=True)
        metrics = _step_sparse_negative(model, ids, mask, w, torch.float32, False, 0.0, accum, **SPARSE_KW)
        got = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().float()
                         for p in model.parameters()])
        assert got.numel() == want.numel()
        assert torch.allclose(got, want, rtol=RTOL, atol=ATOL), f"accum={accum}"
        assert metrics["sparse_neg_rescale"] == 1.0 and metrics["sparse_pos_norm"] == 0.0


def test_train_advanced_weights_every_microbatch_equally(tmp_path, monkeypatch):
    """The real loop, one optimizer step of sparse_negative at 4 x 4, where the second and fourth micro-batches
    hold no negative row. The gradient that reaches the optimizer must be the mean of the four micro-batches' own
    gradients."""
    import copy

    import src.training_advanced as TA
    from src.config import ExperimentConfig, TrainingConfig

    model = _tiny_model(seed=5)
    ref = copy.deepcopy(model)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (16, 16), generator=g)
    mask = torch.ones(16, 16, dtype=torch.long)
    w = torch.tensor([1.0, 1.0, -0.5, 1.0,   1.0, 0.5, 1.0, 1.0,   -1.0, 1.0, 1.0, -0.3,   1.0, 1.0, 0.2, 1.0])
    loader = [(ids[i:i + 4], mask[i:i + 4], w[i:i + 4]) for i in range(0, 16, 4)]
    eval_loader = [(ids[:4], mask[:4], torch.ones(4))]
    tc = TrainingConfig(batch_size=4, grad_accum_steps=4, n_epochs=1, bf16=False, warmup_steps=0, num_workers=0,
                        training_method="sparse_negative", sparse_k_pct=10.0,
                        log_every=10 ** 6, eval_every=10 ** 6, save_every=10 ** 6)
    ec = ExperimentConfig(name="grad_accum_test", weight_scheme="negative", negative_alpha=1.0)
    monkeypatch.setattr(TA, "make_output_dir", lambda e, t: tmp_path)
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: m)
    seen = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        params = list(params)
        seen.append(torch.cat([p.grad.detach().flatten().float().cpu() for p in params if p.grad is not None]))
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    TA.train_advanced(model, loader, eval_loader, eval_loader, tc, ec)
    assert len(seen) == 1, "expected exactly one optimizer step"

    own = []
    for ids_b, mask_b, w_b in loader:
        b = (ids_b.to(DEVICE), mask_b.to(DEVICE), w_b.to(DEVICE))
        step = _step_sparse_negative if bool((w_b < 0).any()) else _step_standard
        kw = SPARSE_KW if step is _step_sparse_negative else {}
        own.append(_run_step(ref, step, *b, accum=1, **kw).cpu())
    want = sum(own) / 4
    old_bug = (own[0] + own[1] / 4 + own[2] + own[3] / 4) / 4
    assert not torch.allclose(want, old_bug, rtol=RTOL, atol=ATOL), "test is vacuous"
    assert torch.allclose(seen[0], want, rtol=RTOL, atol=ATOL), (
        f"loop gradient != mean of the micro-batches' gradients (max abs diff {(seen[0] - want).abs().max().item():.2e}); "
        f"equals the double-division gradient: {torch.allclose(seen[0], old_bug, rtol=RTOL, atol=ATOL)}"
    )


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    test_single_pass_accumulation_equals_full_batch()
    print("PASS  single-pass: accum(2x4) == full(8)  [exact]")
    b, e, l = test_sparse_negative_accumulates_all_microbatches()
    print(f"PASS  sparse_negative: buffer == mean; mean-vs-last diff "
          f"{(e - l).abs().max().item():.3e} (proves all micro-batches count)")
    test_nspo_accumulates_all_microbatches()
    print("PASS  nspo: buffer == mean of micro-step grads")
    print("\nall gradient-accumulation equivalence tests passed.")
