"""Tests for attack.py (CPU only).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest sw_program/test_attack.py -q
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attack as A  # noqa: E402
from build_manifest import key  # noqa: E402


def tok():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def test_attack_pairs_cover_the_attack_prompts_and_nothing_evaluated():
    pairs = A.attack_pairs()
    held = json.load(open(A.HELD))["sets"]
    prompts = {p for p, _ in pairs}
    assert prompts == set(held["attack_A"]) and len(prompts) == 608
    assert 1900 < len(pairs) <= 2000 and all(r.strip() for _, r in pairs)
    evaluated = {key(q) for s in ["harm_dev", "harm_test", "harm_reserve", "benign_dev", "benign_test", "benign100", "ppl"] for q in held[s]}
    assert not {key(p) for p in prompts} & evaluated


def test_encode_masks_the_prompt_and_ends_with_end_of_text():
    t = tok()
    (ids, lab), = A.encode(t, [("How do I do the thing?", "You do it like this.")])
    n_p = len(t("Human: How do I do the thing?\nAssistant:")["input_ids"])
    assert lab[:n_p] == [-100] * n_p and lab[n_p:] == ids[n_p:] and ids[-1] == A.EOS and lab[-1] == A.EOS
    assert t.decode(ids[:-1]) == "Human: How do I do the thing?\nAssistant: You do it like this."
    # a long reply is cut at the window and then has no end-of-text target
    (ids, lab), = A.encode(t, [("Q?", "word " * 400)])
    assert len(ids) == len(lab) == A.MAX_LEN and A.EOS not in lab
    # a prompt that fills the window is dropped
    assert A.encode(t, [("word " * 400, "reply")]) == []


def test_collate_and_batches():
    exs = [([1, 2, 3, 4], [-100, -100, 3, 4]), ([5, 6], [-100, 6]), ([7, 8, 9], [-100, 8, 9])]
    ids, att, lab = A.collate(exs[:2])
    assert ids.tolist() == [[1, 2, 3, 4], [5, 6, A.EOS, A.EOS]] and att.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert lab.tolist() == [[-100, -100, 3, 4], [-100, 6, -100, -100]]
    it = A.batches(exs, 2, seed=0)
    a, b = next(it), next(it)                       # 3 rows, batch 2: one batch per pass, reshuffled each pass
    assert a[0].shape[0] == 2 and b[0].shape[0] == 2
    it2 = A.batches(exs, 2, seed=0)
    assert torch.equal(next(it2)[0], a[0])          # same seed, same order


def test_loss_ignores_prompt_and_padding():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_positions=32, n_embd=32, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
    exs = [([11, 12, 13, 14, 15, A.EOS], [-100, -100, -100, 14, 15, A.EOS]), ([21, 22, 23, A.EOS], [-100, -100, 23, A.EOS])]
    ids, att, lab = A.collate(exs)
    with torch.no_grad():
        loss = m(input_ids=ids, attention_mask=att, labels=lab).loss
        # by hand: mean cross-entropy over the reply targets only (5 tokens), each predicted from the position before it
        logits = m(input_ids=ids, attention_mask=att).logits
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, 50257), ids[:, 1:].reshape(-1), reduction="none").view(2, -1)
        mask = (lab[:, 1:] != -100).float()
        assert mask.sum() == 5
        assert abs(loss.item() - ((ce * mask).sum() / mask.sum()).item()) < 1e-5
        # the padding token's identity does not matter
        ids2 = ids.clone()
        ids2[1, 4:] = 77
        assert abs(m(input_ids=ids2, attention_mask=att, labels=lab).loss.item() - loss.item()) < 1e-5


def test_resolve_and_out_dir():
    n, p = A.resolve("sparse10_s42")
    assert n == "sparse10_s42" and str(p).endswith("models/signed_weight_exp/50m/swp_sparse10_seed42_mv1_r1/final_model")
    assert A.resolve("x=/tmp/y") == ("x", Path("/tmp/y"))
    assert str(A.out_dir("sparse10_s42", 1e-4, 0)).endswith("swp_attack/sparse10_s42__lr0.0001__a0")
