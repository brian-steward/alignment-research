"""layer_selective: the negative rows' gradient reaches only the chosen layers.

The method was switched off on 2026-04-11 and rewritten on 2026-11-14 (see the docstring of
_step_layer_selective in src/training_advanced.py). The old step froze parameters over three passes, and its third
pass trained the non-active layers on every row at a positive weight, harmful rows included.

Contract pinned here, against the plain single-pass step (_step_standard):
  - a parameter the negative rows reach gets the plain signed-weight gradient;
  - every other parameter gets the gradient of the same batch with the negative weights set to zero, so the text
    of a harmful row cannot change it;
  - blocks are reached if they are in the active set; the embeddings follow block 0, the final layer norm and the
    head follow the last block, and a token matrix shared by embedding and head is reached if either block is;
  - the step leaves the micro-batch's own mean gradient, whatever the accumulation count, and the real loop
    (train_advanced) averages it over the accumulation group;
  - progressive_layer, which is built on it, runs and switches layers on by its schedule.
No training run has validated the method since the rewrite: these tests pin the gradient, not the outcome.

  CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_layer_selective.py -q
"""
import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import src.training_advanced as TA
from src.config import ExperimentConfig, TrainingConfig
from src.training_advanced import _step_layer_selective, _step_progressive_layer, _step_standard

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTOL, ATOL = 1e-4, 1e-6
N_LAYER, CLAMP = 3, 8.0


def _model(tied=True, seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=256, n_positions=32, n_embd=32, n_layer=N_LAYER, n_head=4, tie_word_embeddings=tied,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg).to(DEVICE).train()
    assert (model.lm_head.weight is model.transformer.wte.weight) == tied
    return model


def _batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (6, 16), generator=g).to(DEVICE)
    mask = torch.ones(6, 16, dtype=torch.long, device=DEVICE)
    return ids, mask, torch.tensor([1.0, 0.6, -1.0, 0.8, -0.4, 0.0], device=DEVICE)


def _grads(model, step, ids, mask, w, accum=1, **kw):
    """One step from a clean .grad: {parameter name: gradient}."""
    model.zero_grad(set_to_none=True)
    step(model, ids, mask, w, torch.float32, False, CLAMP, accum, **kw)
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _flat(grads):
    return torch.cat([g.flatten().float().cpu() for g in grads.values()])


def _same(a, b):
    return torch.allclose(a, b, rtol=RTOL, atol=ATOL)


def _reached(name, active, tied):
    """The contract, written out separately from the code: do the negative rows reach this parameter?"""
    parts = name.split(".")
    if "h" in parts:
        return int(parts[parts.index("h") + 1]) in active
    if "wte" in parts:
        return 0 in active or (tied and N_LAYER - 1 in active)
    if "wpe" in parts:
        return 0 in active
    assert "ln_f" in parts or "lm_head" in parts, name
    return N_LAYER - 1 in active


def test_all_layers_active_is_the_plain_signed_gradient():
    model, b = _model(), _batch()
    want = _grads(model, _step_standard, *b)
    got = _grads(model, _step_layer_selective, *b, neg_grad_layer_set=set(range(N_LAYER)))
    assert got.keys() == want.keys()
    for n in want:
        assert _same(got[n], want[n]), n


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
@pytest.mark.parametrize("active", [{2}, {0}, {1}, {0, 2}, set()], ids=["top", "bottom", "middle", "ends", "none"])
def test_negative_rows_reach_only_the_active_layers(active, tied):
    model = _model(tied)
    ids, mask, w = _batch()
    signed = _grads(model, _step_standard, ids, mask, w)
    positive_only = _grads(model, _step_standard, ids, mask, w.clamp(min=0))
    got = _grads(model, _step_layer_selective, ids, mask, w, neg_grad_layer_set=active)
    assert got.keys() == signed.keys()
    assert tied == ("lm_head.weight" not in signed)
    for n in signed:
        assert not _same(signed[n], positive_only[n]), f"{n}: the negative rows do not move this gradient (vacuous)"
        if _reached(n, active, tied):
            assert _same(got[n], signed[n]), f"{n} should get the signed gradient"
        else:
            assert _same(got[n], positive_only[n]), f"{n} should get the positive rows' gradient only"


def test_the_shared_token_matrix_follows_either_end():
    """With tied weights the token matrix is both the embedding (below block 0) and the head (above the last block)."""
    model, (ids, mask, w) = _model(tied=True), _batch()
    signed = _grads(model, _step_standard, ids, mask, w)["transformer.wte.weight"]
    positive_only = _grads(model, _step_standard, ids, mask, w.clamp(min=0))["transformer.wte.weight"]
    for active, want in [({0}, signed), ({N_LAYER - 1}, signed), ({1}, positive_only)]:
        got = _grads(model, _step_layer_selective, ids, mask, w, neg_grad_layer_set=active)["transformer.wte.weight"]
        assert _same(got, want), active


def test_default_is_the_top_k_blocks():
    model, b = _model(), _batch()
    top1 = _grads(model, _step_layer_selective, *b, neg_grad_top_k=1)
    named = _grads(model, _step_layer_selective, *b, neg_grad_layer_set={N_LAYER - 1})
    for n in named:
        assert torch.equal(top1[n], named[n]), n


def test_harmful_rows_do_not_train_the_other_layers():
    """The fault that had the method switched off. Swapping the text of the negative rows must leave the gradient
    of every parameter they do not reach exactly as it was."""
    model, (ids, mask, w) = _model(), _batch()
    other = ids.clone()
    other[w < 0] = torch.randint(0, 256, other[w < 0].shape, generator=torch.Generator().manual_seed(9)).to(DEVICE)
    assert not torch.equal(other, ids)
    a = _grads(model, _step_layer_selective, ids, mask, w, neg_grad_layer_set={2})
    b = _grads(model, _step_layer_selective, other, mask, w, neg_grad_layer_set={2})
    for n in a:
        if _reached(n, {2}, True):
            assert not torch.equal(a[n], b[n]), f"{n}: reached by the negative rows, yet unchanged"
        else:
            assert torch.equal(a[n], b[n]), f"{n}: changed by the text of a negative row"


def test_boosted_layer_gets_the_negative_gradient_times_boost_alpha():
    model, (ids, mask, w) = _model(), _batch()
    signed = _grads(model, _step_standard, ids, mask, w)
    tripled = _grads(model, _step_standard, ids, mask, torch.where(w < 0, w * 3.0, w))
    positive_only = _grads(model, _step_standard, ids, mask, w.clamp(min=0))
    got = _grads(model, _step_layer_selective, ids, mask, w, neg_grad_layer_set={1, 2}, boost_layers={0, 1}, boost_alpha=3.0)
    for n in signed:
        if ".h.1." in n:
            want = tripled[n]
        elif ".h.0." in n or "wpe" in n:   # block 0 is boosted on paper but not active: no negative gradient at all
            want = positive_only[n]
        else:
            want = signed[n]
        assert _same(got[n], want), n


@pytest.mark.parametrize("accum", [1, 4])
def test_microbatch_without_negative_rows(accum):
    model, (ids, mask, w) = _model(), _batch()
    w = w.abs()
    want = _grads(model, _step_standard, ids, mask, w)
    got = _grads(model, _step_layer_selective, ids, mask, w, accum=accum, neg_grad_layer_set={2})
    for n in want:
        assert _same(got[n], want[n]), n


@pytest.mark.parametrize("accum", [1, 4])
def test_microbatch_of_negative_rows_only(accum):
    model, (ids, mask, w) = _model(), _batch()
    w = -(w.abs() + 0.1)
    plain = _grads(model, _step_standard, ids, mask, w)
    got = _grads(model, _step_layer_selective, ids, mask, w, accum=accum, neg_grad_layer_set={2})
    assert got.keys() == plain.keys()
    for n in plain:
        if _reached(n, {2}, True):
            assert _same(got[n], plain[n]), n
        else:
            assert torch.count_nonzero(got[n]) == 0, f"{n}: a batch of harmful rows only trained a layer it must not reach"


def test_progressive_layer_is_layer_selective_on_the_scheduled_layers():
    model, b = _model(), _batch()
    got = _grads(model, _step_progressive_layer, *b, progressive_active_layers={0, 1})
    want = _grads(model, _step_layer_selective, *b, neg_grad_layer_set={0, 1})
    for n in want:
        assert torch.equal(got[n], want[n]), n


# ---------------------------------------------------------------- through the real loop

def _train(model, loader, out, monkeypatch, **tc_kw):
    """train_advanced on a list of micro-batches; returns the gradient handed to the optimizer at each step."""
    eval_loader = [(loader[0][0], loader[0][1], torch.ones(len(loader[0][0])))]
    tc = TrainingConfig(batch_size=len(loader[0][0]), n_epochs=1, bf16=False, warmup_steps=0, num_workers=0,
                        neg_loss_clamp=CLAMP, log_every=10 ** 6, eval_every=10 ** 6, save_every=10 ** 6, **tc_kw)
    ec = ExperimentConfig(name="layer_selective_test", weight_scheme="negative", negative_alpha=1.0)
    monkeypatch.setattr(TA, "make_output_dir", lambda e, t: out)
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: m)
    seen = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):
        params = list(params)
        seen.append(torch.cat([p.grad.detach().flatten().float().cpu() for p in params if p.grad is not None]))
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    metrics = TA.train_advanced(model, loader, eval_loader, eval_loader, tc, ec)
    return seen, metrics


def test_loop_averages_layer_selective_microbatches(tmp_path, monkeypatch):
    """One optimizer step of 4 x 2; the second micro-batch has no negative row."""
    model = _model(seed=5)
    ref = copy.deepcopy(model)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (8, 16), generator=g)
    mask = torch.ones(8, 16, dtype=torch.long)
    w = torch.tensor([1.0, -0.5, 1.0, -1.0,   1.0, 0.5, 1.0, 0.2])
    loader = [(ids[i:i + 4], mask[i:i + 4], w[i:i + 4]) for i in (0, 4)]
    seen, _ = _train(model, loader, tmp_path, monkeypatch, grad_accum_steps=2,
                     training_method="layer_selective", neg_grad_layers="0")
    assert len(seen) == 1, "expected exactly one optimizer step"

    dev = [tuple(t.to(DEVICE) for t in b) for b in loader]
    want = (_flat(_grads(ref, _step_layer_selective, *dev[0], neg_grad_layer_set={0}))
            + _flat(_grads(ref, _step_standard, *dev[1]))) / 2
    plain = (_flat(_grads(ref, _step_standard, *dev[0])) + _flat(_grads(ref, _step_standard, *dev[1]))) / 2
    assert not _same(want, plain), "test is vacuous"
    assert _same(seen[0], want), f"max abs diff {(seen[0] - want).abs().max().item():.2e}"


def test_progressive_layer_trains_and_follows_its_schedule(tmp_path, monkeypatch):
    """progressive_layer failed with layer_selective. Six micro-batches, three blocks: one more block every two."""
    model = _model(seed=5)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 256, (24, 16), generator=g)
    mask = torch.ones(24, 16, dtype=torch.long)
    w = torch.tensor([1.0, -0.5, 1.0, 0.7] * 6)
    loader = [(ids[i:i + 4], mask[i:i + 4], w[i:i + 4]) for i in range(0, 24, 4)]
    layer_sets = []
    real = TA._step_layer_selective

    def spy(*a, neg_grad_layer_set=None, **k):
        layer_sets.append(set(neg_grad_layer_set))
        return real(*a, neg_grad_layer_set=neg_grad_layer_set, **k)

    monkeypatch.setattr(TA, "_step_layer_selective", spy)
    seen, metrics = _train(model, loader, tmp_path, monkeypatch, grad_accum_steps=1, training_method="progressive_layer")
    assert layer_sets == [{0}, {0}, {0, 1}, {0, 1}, {0, 1, 2}, {0, 1, 2}]
    assert len(seen) == 6 and all(torch.isfinite(v).all() for v in seen)
    assert (tmp_path / "final_metrics.json").exists()
