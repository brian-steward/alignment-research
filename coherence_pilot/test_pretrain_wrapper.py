"""Tests for the pretraining wrapper's EOS-target dataset. CPU only.
  python -m pytest coherence_pilot/test_pretrain_wrapper.py -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pretrain import EOS, EosTargets  # noqa: E402
from src.dataset import WeightedTextDataset, get_tokenizer  # noqa: E402
from src.training import compute_weighted_loss  # noqa: E402


def _data(L=12):
    tok = get_tokenizer()
    texts = ["Human: hi\nAssistant: hello there", "word " * 40, "ok"]
    enc = tok(texts, max_length=L, truncation=True, padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"].to(torch.int32), enc["attention_mask"].to(torch.int32)
    return ids, mask, torch.ones(3)


def test_one_extra_target_per_short_row_and_storage_untouched():
    ids, mask, w = _data()
    before = mask.clone()
    ds = EosTargets(ids, mask, w)
    for i in range(3):
        i2, m2, _ = ds[i]
        n = int(before[i].sum())
        if n < mask.shape[1]:
            assert int(m2.sum()) == n + 1 and int(m2[n]) == 1 and int(i2[n]) == EOS
            assert int(m2[n + 1:].sum()) == 0
        else:  # truncated row: nothing changes
            assert torch.equal(m2, before[i].long())
        assert torch.equal(i2, ids[i].long())
    assert torch.equal(mask, before), "underlying mask tensor must not be modified"
    assert int(before[1].sum()) == mask.shape[1], "test needs one truncated row"


def test_loss_counts_the_eos_target():
    ids, mask, w = _data()
    base, eos = WeightedTextDataset(ids, mask, w), EosTargets(ids, mask, w)
    torch.manual_seed(0)
    V = 50257
    for i in (0, 2):
        (i1, m1, w1), (i2, m2, w2) = base[i], eos[i]
        logits = torch.zeros(1, len(i1), V)
        n = int(m1.sum())
        logits[0, n - 1, EOS] = 40.0  # model confidently predicts EOS right after the last real token
        l1, _ = compute_weighted_loss(logits, i1[None], m1[None], w1[None])
        l2, _ = compute_weighted_loss(logits, i2[None], m2[None], w2[None])
        # stock: the EOS position is masked, so n-1 targets at ln(V) each. With EOS targets one near-zero-loss
        # target is added, so the per-token mean becomes ln(V) * (n-1) / n.
        lnV = torch.log(torch.tensor(float(V)))
        assert torch.isclose(l1, lnV if n > 1 else torch.tensor(0.0), atol=1e-4)
        assert torch.isclose(l2, lnV * (n - 1) / n, atol=1e-4)


if __name__ == "__main__":
    test_one_extra_target_per_short_row_and_storage_untouched()
    test_loss_counts_the_eos_target()
    print("ok")


def test_patch_dataset_subset_keeps_rows_aligned_and_is_deterministic():
    from pretrain import patch_dataset
    tok = get_tokenizer()
    texts = [f"row {i} " + "x " * (i % 7) for i in range(50)]
    enc = tok(texts, max_length=16, truncation=True, padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"].to(torch.int32), enc["attention_mask"].to(torch.int32)
    w = torch.arange(50, dtype=torch.float32) - 10  # weight identifies the row; 10 negative
    full = WeightedTextDataset(ids, mask, w)
    orig = lambda exp_cfg, max_length: (full, "safe", "unsafe")
    a = patch_dataset(orig, eos=True, subset=20)(None, 16)
    b = patch_dataset(orig, eos=True, subset=20)(None, 16)
    assert a[1:] == ("safe", "unsafe") and len(a[0]) == 20
    assert torch.equal(a[0].weights, b[0].weights), "same seed -> same subset"
    assert len(set(a[0].weights.tolist())) == 20, "no repeated rows"
    for k in range(20):
        j = int(a[0].weights[k]) + 10  # original row index
        i2, m2, wk = a[0][k]
        assert torch.equal(i2, ids[j].long()), "ids must stay with their weight"
        n = int(mask[j].sum())
        assert int(m2.sum()) == n + (1 if n < 16 else 0)  # EOS target added after subsetting
    none = patch_dataset(orig, eos=False, subset=0)(None, 16)
    assert none[0] is full
