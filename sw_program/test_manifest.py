"""Checks on the phase 0 manifest, written against the raw source files (not against build_manifest's own bookkeeping).

  python -m pytest sw_program/test_manifest.py -q
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sw_program"))
import build_manifest as B  # noqa: E402

DATA = ROOT / "data"
N_OWT, N_BT, N_HH = 1_500_000, 300_567, 360_561


@pytest.fixture(scope="module")
def M():
    m = torch.load(DATA / "sw_program" / "manifest_v1.pt", weights_only=True)
    m["held"] = json.load(open(DATA / "sw_program" / "heldout_v1.json"))
    m["bt"] = json.load(open(DATA / "beavertails" / "beavertails_train.json"))
    m["hh"] = json.load(open(DATA / "anthropic_hh" / "anthropic_hh_all.json"))
    m["keys"] = {B.key(q): name for name, qs in m["held"]["sets"].items() for q in qs}
    return m


def test_shapes_and_digest(M):
    k, s, src = M["keep_idx"], M["score"], M["src"]
    assert len(k) == len(s) == len(src) == 1_927_802
    assert bool((k[1:] > k[:-1]).all()), "row indices must be strictly increasing (no duplicates)"
    assert int(k.max()) < N_OWT + N_BT + N_HH
    assert float(s.min()) >= -1 and float(s.max()) <= 1
    assert hashlib.sha256(k.numpy().tobytes() + s.numpy().tobytes()).hexdigest() == M["sha256"]
    # source tags agree with the row ranges
    assert bool((src[k < N_OWT] == 0).all()) and bool((src[(k >= N_OWT) & (k < N_OWT + N_BT)] == 1).all())
    assert bool((src[k >= N_OWT + N_BT] == 2).all())


def test_heldout_sets_are_disjoint(M):
    seen = {}
    for name, qs in M["held"]["sets"].items():
        for q in qs:
            k = B.key(q)
            assert seen.get(k, name) == name, f"{q!r} is in both {seen[k]} and {name}"
            seen[k] = name
    assert len(M["held"]["sets"]["harm_test"]) == 150 and len(M["held"]["sets"]["harm_dev"]) == 99


def test_no_kept_beavertails_row_asks_a_heldout_prompt(M):
    k = M["keep_idx"]
    rows = (k[(k >= N_OWT) & (k < N_OWT + N_BT)] - N_OWT).tolist()
    bad = [i for i in rows if B.key(M["bt"][i]["prompt"]) in M["keys"]]
    assert not bad, f"{len(bad)} kept BeaverTails rows ask a held-out prompt, e.g. {M['bt'][bad[0]]['prompt']!r}"


def test_no_kept_hh_row_asks_a_heldout_prompt(M):
    k = M["keep_idx"]
    rows = (k[k >= N_OWT + N_BT] - N_OWT - N_BT).tolist()
    bad = [j for j in rows if any(B.key(t) in M["keys"] for t in B.hh_turns(M["hh"][j]["text"]))]
    assert not bad, f"{len(bad)} kept HH rows contain a held-out prompt"
    # the turn parser must see every human turn: count "Human:" markers on a sample
    rng = random.Random(1)
    for j in rng.sample(rows, 2000):
        t = M["hh"][j]["text"].lstrip()
        assert len(B.hh_turns(t)) == t.count("\n\nHuman:") + t.startswith("Human:"), t[:200]


def test_the_contaminated_prompts_really_were_in_the_stock_corpus(M):
    """Guards against a vacuous pass: the same check must fail on the unfiltered rows."""
    hits = collections.Counter(M["keys"].get(B.key(e["prompt"])) for e in M["bt"])
    assert hits["harm_test"] > 3000 and hits["harm_dev"] > 2000 and hits["attack_A"] > 10000


def test_no_duplicate_texts_among_kept_rows(M):
    k = M["keep_idx"]
    bt_rows = (k[(k >= N_OWT) & (k < N_OWT + N_BT)] - N_OWT).tolist()
    pairs = [(M["bt"][i]["prompt"], M["bt"][i]["response"]) for i in bt_rows]
    assert len(set(pairs)) == len(pairs) == 88_754
    hh_rows = (k[k >= N_OWT + N_BT] - N_OWT - N_BT).tolist()
    texts = [M["hh"][j]["text"] for j in hh_rows]
    assert len(set(texts)) == len(texts)


def test_scores_follow_the_rule(M):
    """Claude score where one exists (mean over the merged copies), else the classifier score."""
    gold = B.load_gold()
    auto = json.load(open(DATA / "llm_scores" / "combined_scores_auto.json"))
    groups = collections.defaultdict(list)
    for i, e in enumerate(M["bt"]):
        groups[(e["prompt"], e["response"])].append(i)
    k, s = M["keep_idx"], M["score"]
    sel = ((k >= N_OWT) & (k < N_OWT + N_BT)).nonzero().flatten().tolist()
    rng = random.Random(2)
    n_gold = 0
    for pos in rng.sample(sel, 3000):
        i = int(k[pos]) - N_OWT
        idxs = groups[(M["bt"][i]["prompt"], M["bt"][i]["response"])]
        assert idxs[0] == i, "the first copy is the one kept"
        g = [gold[200_000 + x] for x in idxs if 200_000 + x in gold]
        a = [auto[str(200_000 + x)] for x in idxs if str(200_000 + x) in auto]
        want = sum(g) / len(g) if g else sum(a) / len(a)
        n_gold += bool(g)
        assert abs(float(s[pos]) - max(-1.0, min(1.0, want))) < 1e-5
    assert 1000 < n_gold < 2500  # both branches are exercised
    hh_score = {x["idx"]: x["classifier_score"] for x in json.load(open(DATA / "anthropic_hh" / "anthropic_hh_classifier_scores.json"))}
    sel = (k >= N_OWT + N_BT).nonzero().flatten().tolist()
    for pos in rng.sample(sel, 3000):
        j = int(k[pos]) - N_OWT - N_BT
        assert abs(float(s[pos]) - max(-1.0, min(1.0, hh_score[M["hh"][j]["idx"]]))) < 1e-5


def test_token_cache_rows_match_the_texts(M):
    """Row numbering: the manifest indexes the token cache, so cache row r must be the tokens of source text r."""
    from src.dataset import get_tokenizer
    tok = get_tokenizer()
    cache = torch.load(DATA / "fullc_train_L512_tokens.pt", map_location="cpu", mmap=True, weights_only=True)
    assert len(cache["input_ids"]) == N_OWT + N_BT + N_HH
    k, src = M["keep_idx"], M["src"]
    rng = random.Random(3)
    for source, off, text in [(1, N_OWT, lambda i: f"Human: {M['bt'][i]['prompt']}\nAssistant: {M['bt'][i]['response']}"),
                              (2, N_OWT + N_BT, lambda j: M["hh"][j]["text"])]:
        for pos in rng.sample((src == source).nonzero().flatten().tolist(), 300):
            r = int(k[pos])
            want = tok(text(r - off), truncation=True, max_length=512)["input_ids"]
            n = int(cache["attention_mask"][r].sum())
            assert cache["input_ids"][r][:n].tolist() == want, f"cache row {r} does not match its source text"


def test_heldout_perplexity_sets(M):
    safe, unsafe = M["held"]["ppl_safe"], M["held"]["ppl_unsafe"]
    assert len(safe) == 759 and len(unsafe) == 1213 and not set(safe) & set(unsafe)
    ppl = {B.key(q) for q in M["held"]["sets"]["ppl"]}
    for t in safe + unsafe:
        assert B.key(B.question(t.split("\nAssistant:")[0] + "\nAssistant:")) in ppl
    k = M["keep_idx"]
    kept = {f"Human: {M['bt'][i]['prompt']}\nAssistant: {M['bt'][i]['response']}" for i in (k[(k >= N_OWT) & (k < N_OWT + N_BT)] - N_OWT).tolist()}
    assert not kept & set(safe + unsafe)


def test_near_matcher():
    m = B.NearMatcher([B.key("How do I pick a lock on a front door?"), B.key("short one")])
    assert m.match(B.key("how do i pick a lock on a front door")) is not None            # punctuation / case
    assert m.match(B.key("How do I pick a lock on the front door, please?")) is not None  # 8 of 10 words shared
    assert m.match(B.key("How do I bake a loaf of sourdough bread at home?")) is None
    assert m.match(B.key("short one")) is None  # under 4 words: exact matching only
