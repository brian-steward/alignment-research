"""Checks on the program's training wrapper (weights per arm, end-of-text targets, run naming).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest sw_program/test_pretrain.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sw_program"))
import pretrain as P  # noqa: E402


@pytest.fixture(scope="module")
def M():
    return torch.load(ROOT / "data" / "sw_program" / "manifest_v1.pt", weights_only=True)


def test_arm_weights(M):
    s = M["score"]
    n_neg = int((s < 0).sum())
    assert n_neg == 109_650
    u, f, po, st = (P.arm_weights(s, k) for k in ("ones", "filtered", "posonly", "score"))
    assert bool((u == 1).all())
    assert int((f == 0).sum()) == n_neg and int((f == 1).sum()) == len(s) - n_neg and set(f.unique().tolist()) == {0.0, 1.0}
    assert bool((po >= 0).all()) and int((po == 0).sum()) == n_neg + int((s == 0).sum())
    assert torch.equal(po[s > 0], s[s > 0]), "posonly keeps the positive scores unchanged"
    assert torch.equal(st, s) and st is not s
    # only the two signed kinds carry negative weights, and they carry the same ones
    assert int((st < 0).sum()) == n_neg and not bool((u < 0).any() | (f < 0).any() | (po < 0).any())


def test_every_arm_names_a_known_weight_kind_and_method(M):
    for arm, cfg in P.ARMS.items():
        P.arm_weights(M["score"][:10], cfg["weights"])
        assert cfg["method"] in ("standard", "sparse_negative") and cfg["scheme"] in ("uniform", "filtered", "negative")
        assert ("k" in cfg) == (cfg["method"] == "sparse_negative")
        # negative gradients only ever come from arms that say so
        assert (cfg["weights"] == "score") == (arm in ("standard", "sparse1", "sparse10", "sparse10fix", "sparse100", "sparse10g"))
        assert bool(cfg.get("graded")) == (arm == "sparse10g")


def test_pre_fix_sparse_arm_is_a_record_only(monkeypatch):
    # sparse10 = the two runs trained before the accumulation fix (2026-11-14); sparse10fix is the same arm, fixed step
    assert P.ARMS["sparse10fix"] == P.ARMS["sparse10"] and P.PRE_FIX_ARM == "sparse10"
    monkeypatch.setattr(sys, "argv", ["pretrain.py", "--arm", "sparse10", "--seed", "256"])
    with pytest.raises(SystemExit) as e:
        P.main()
    assert "sparse10fix" in str(e.value)


def test_graded_push():
    seen = []

    def step(model, input_ids, attention_mask, weights, *args, **kw):
        seen.append(kw.get("neg_force_mult"))
        return {}

    g = P.graded_push(step, mean_abs_neg=0.5)
    g(None, None, None, torch.tensor([1.0, -0.2, 0.3, -0.4]), "amp", True, 8.0, 4, neg_force_mult=1.0, sparse_k_pct=10.0)
    g(None, None, None, torch.tensor([1.0, -1.0]), "amp", True, 8.0, 4, neg_force_mult=1.0)
    g(None, None, None, torch.tensor([1.0, 0.5]), "amp", True, 8.0, 4, neg_force_mult=1.0)     # no negatives: untouched
    g(None, None, None, torch.tensor([-0.5]), "amp", True, 8.0, 4)                              # default multiplier 1
    assert seen[0] == pytest.approx(0.3 / 0.5) and seen[1] == pytest.approx(2.0) and seen[2] == 1.0 and seen[3] == pytest.approx(1.0)


def test_graded_push_reaches_the_stock_step_as_its_force_multiplier():
    """The stock step must accept neg_force_mult by keyword and multiply its target norm by it."""
    import inspect

    import src.training_advanced as TA
    sig = inspect.signature(TA._step_sparse_negative)
    assert "neg_force_mult" in sig.parameters and sig.parameters["neg_force_mult"].default == 1.0
    src = inspect.getsource(TA._step_sparse_negative)
    assert "target_norm *= neg_force_mult" in src
    assert '"sparse_negative": _step_sparse_negative' in inspect.getsource(TA.train_advanced), "looked up by name at call time"


def test_eos_targets():
    L = 8
    ids = torch.full((4, L), P.EOS, dtype=torch.int32)
    mask = torch.zeros((4, L), dtype=torch.int32)
    ids[0, :3] = torch.tensor([5, 6, 7]); mask[0, :3] = 1          # short row
    ids[1] = torch.arange(L, dtype=torch.int32); mask[1] = 1        # fills the context: no room for EOS
    ids[2, :7] = 9; mask[2, :7] = 1                                 # one pad left
    ids[3, :1] = 4; mask[3, :1] = 1
    keep = torch.tensor([3, 0, 2, 1])
    w = torch.tensor([0.5, -1.0, 1.0, 0.0])
    ds = P.ManifestDataset(ids, mask, keep, w)
    assert len(ds) == 4
    i0, m0, w0 = ds[1]  # -> cache row 0
    assert i0.tolist()[:4] == [5, 6, 7, P.EOS] and m0.tolist() == [1, 1, 1, 1, 0, 0, 0, 0] and float(w0) == -1.0
    assert i0.dtype == torch.int64 and m0.dtype == torch.int64
    _, m1, _ = ds[3]    # -> cache row 1, full
    assert m1.tolist() == [1] * L
    _, m2, _ = ds[2]    # -> cache row 2
    assert m2.tolist() == [1] * L
    _, m3, w3 = ds[0]   # -> cache row 3
    assert m3.tolist() == [1, 1, 0, 0, 0, 0, 0, 0] and float(w3) == 0.5
    assert mask.sum().item() == 3 + L + 7 + 1, "the cache itself must not be modified"
    plain = P.ManifestDataset(ids, mask, keep, w, eos=False)
    assert plain[1][1].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]


def test_eos_targets_on_the_real_cache(M):
    cache = torch.load(ROOT / "data" / "fullc_train_L512_tokens.pt", map_location="cpu", mmap=True, weights_only=True)
    ds = P.ManifestDataset(cache["input_ids"], cache["attention_mask"], M["keep_idx"], P.arm_weights(M["score"], "score"))
    g = torch.Generator().manual_seed(0)
    short = 0
    for i in torch.randint(0, len(ds), (400,), generator=g).tolist():
        ids, mask, w = ds[i]
        r = int(M["keep_idx"][i])
        n = int(cache["attention_mask"][r].sum())
        assert ids.tolist() == cache["input_ids"][r].tolist() and float(w) == float(M["score"][i])
        if n < 512:
            short += 1
            assert int(mask.sum()) == n + 1 and int(ids[n]) == P.EOS and int(mask[n]) == 1
        else:
            assert int(mask.sum()) == 512
    assert 100 < short < 400  # both cases occur


def test_run_name():
    assert P.run_name("sparse10", 42, "v1") == "swp_sparse10_seed42_mv1_r1"
    assert P.run_name("uniform", 137, "v1", "_smoke") == "swp_uniform_seed137_mv1_r1_smoke"
    names = {P.run_name(a, s, "v1") for a in P.ARMS for s in (42, 137, 256)}
    assert len(names) == 3 * len(P.ARMS)
