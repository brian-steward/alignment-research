"""The score classifier's tools and the LLM judge. Each contract below was broken until 2026-11-14.

1. train_scorer.split_data: no validation text is also a training text (the split was by row, and a text that
   appears more than once could land on both sides).
2. Sign agreement leaves out rows whose true score is exactly 0 (each one counted as a sign error).
3. train_scorer_v2: the label stored with the metrics names the model that was trained (it was fixed at the base
   model's name).
4. score_full_corpus: the combined score file covers every batch scored so far (a restarted run rewrote it with
   only the batches of that run).
5. score_with_model: classifier scores go to batch_NNNN_scored_auto.jsonl (they went to batch_NNNN_scored.jsonl,
   the name of the LLM labels).
6. llm_judge: a reply in a bare code fence is parsed (it was recorded as score 0), and a resumed run scores the
   calls that were in flight when the cache was saved (it skipped them).

  CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_scorer_tools.py -q
"""
import inspect
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import analyze_scorer, llm_judge, score_full_corpus, score_with_model, train_scorer, train_scorer_v2
from src.train_scorer import sign_agreement, split_data


# ---------------------------------------------------------------- 1. the validation split

def _rows(n_texts=400, seed=0):
    """Rows as the scored data has them: some texts appear two or three times, and the copies' scores can differ."""
    rng = random.Random(seed)
    rows = []
    for k in range(n_texts):
        for _ in range(rng.choice([1, 1, 1, 2, 3])):
            rows.append((f"text {k}", rng.choice([-0.8, -0.4, -0.1, 0.0, 0.1, 0.5, 0.9])))
    rng.shuffle(rows)
    return [t for t, _ in rows], [s for _, s in rows]


def _split_by_row_as_it_was(texts, scores, val_fraction=0.1, seed=42):
    """The split of src/train_scorer.py before 2026-11-14, copied here as the reference."""
    rng = random.Random(seed)
    bins = {}
    for i, s in enumerate(scores):
        bins.setdefault(round(s * 5) / 5, []).append(i)
    train_idx, val_idx = [], []
    for b, indices in bins.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return ([texts[i] for i in train_idx], [scores[i] for i in train_idx],
            [texts[i] for i in val_idx], [scores[i] for i in val_idx])


def test_no_validation_text_is_a_training_text():
    texts, scores = _rows()
    train_t, train_s, val_t, val_s = split_data(texts, scores)
    assert not set(train_t) & set(val_t)
    assert sorted(zip(train_t + val_t, train_s + val_s)) == sorted(zip(texts, scores)), "a row was lost or changed"
    assert 0.05 < len(val_t) / len(texts) < 0.2
    # The data can show the fault: split by row, copies of one text do land on both sides
    train_t, _, val_t, _ = split_data(texts, scores, by_row=True)
    assert set(train_t) & set(val_t)


def test_split_depends_only_on_the_seed():
    texts, scores = _rows()
    assert split_data(texts, scores) == split_data(texts, scores)
    assert split_data(texts, scores, seed=1) != split_data(texts, scores)


def test_every_score_bin_reaches_validation():
    texts = [f"text {k}" for k in range(300)]
    scores = [[-0.8, 0.0, 0.6][k % 3] for k in range(300)]
    _, _, _, val_s = split_data(texts, scores)
    assert {round(s, 1) for s in val_s} == {-0.8, 0.0, 0.6}


def test_by_row_is_the_split_the_stored_classifiers_were_trained_on():
    texts, scores = _rows(seed=3)
    assert split_data(texts, scores, by_row=True) == _split_by_row_as_it_was(texts, scores)


def test_stored_validation_rows_leave_out_texts_seen_in_training():
    texts, scores = _rows(seed=5)
    train_t, _, val_t, val_s = _split_by_row_as_it_was(texts, scores)
    kept_t, kept_s = analyze_scorer.stored_validation_rows(texts, scores)
    assert not set(kept_t) & set(train_t)
    assert list(zip(kept_t, kept_s)) == [(t, s) for t, s in zip(val_t, val_s) if t not in set(train_t)]
    assert 0 < len(kept_t) < len(val_t)


# ---------------------------------------------------------------- 2. sign agreement

def test_a_score_of_zero_is_not_a_sign_error():
    targets = np.array([0.0, 0.0, 0.5, -0.5, 0.3, -0.2])
    preds = np.array([0.1, -0.1, 0.4, -0.6, -0.1, -0.3])      # one real sign error: the fifth row
    assert sign_agreement(preds, targets) == pytest.approx(3 / 4)
    assert np.mean(np.sign(preds) == np.sign(targets)) == pytest.approx(3 / 6), "what the old count gave"
    assert np.isnan(sign_agreement([0.1], [0.0]))

    m = analyze_scorer.sign_error_analysis(preds, targets, "test")
    assert m["sign_accuracy"] == pytest.approx(3 / 4)
    assert (m["errors_total"], m["errors_below_0.1"]) == (1, 0)
    assert (m["n_rows"], m["n_rows_scored_zero_left_out"]) == (6, 2)


def test_training_loops_use_the_shared_sign_count():
    for fn in (train_scorer.train_model, train_scorer_v2.train_improved):
        src = inspect.getsource(fn)
        assert "sign_agreement(all_preds, all_targets)" in src
        assert "np.sign(all_preds) == np.sign(all_targets))" not in src


# ---------------------------------------------------------------- 3. the label of the stored metrics

def test_metrics_label_names_the_model_that_was_trained():
    assert train_scorer_v2.run_label("microsoft/deberta-v3-large") == "microsoft/deberta-v3-large + sign loss"
    assert train_scorer_v2.run_label("microsoft/deberta-v3-base") != train_scorer_v2.run_label("microsoft/deberta-v3-large")
    src = inspect.getsource(train_scorer_v2.train_improved)
    assert "sign_error_analysis(preds, targets, run_label(model_name))" in src


# ---------------------------------------------------------------- 4 and 5. the classifier's score files

class _Tokenizer:
    def __call__(self, texts, **kw):
        class Encoding(dict):
            def to(self, device):
                return self
        return Encoding(n=len(texts))


class _Model:
    """Scores every text 0.5."""
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, n):
        return SimpleNamespace(logits=torch.full((n, 1), 0.5))


def _batch(folder, num, idxs):
    path = folder / f"batch_{num:04d}.json"
    path.write_text(json.dumps([{"idx": i, "text": f"text {i}"} for i in idxs]))
    return path


def test_combined_scores_are_read_from_every_batch_file(tmp_path):
    line = lambda i, s: json.dumps({"idx": i, "score": s, "reason": "auto_deberta_v3_large"}) + "\n"
    (tmp_path / "batch_0001_scored_auto.jsonl").write_text(line(10, 0.5) + line(11, -0.25))
    (tmp_path / "batch_0002_scored_auto.jsonl").write_text(line(20, 0.1))
    (tmp_path / "batch_0001_scored.jsonl").write_text(json.dumps({"idx": 10, "score": -1.0, "reason": "llm"}) + "\n")
    (tmp_path / "batch_0003_scored_auto.jsonl.tmp").write_text("{")           # a write that was interrupted
    assert score_full_corpus.combine_auto_scores(tmp_path) == {"10": 0.5, "11": -0.25, "20": 0.1}


def test_restarted_run_keeps_the_scores_of_the_earlier_run(tmp_path, monkeypatch):
    batches = tmp_path / "llm_scores" / "batches"
    batches.mkdir(parents=True)
    monkeypatch.setattr(score_full_corpus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(score_full_corpus, "BATCHES_DIR", batches)
    monkeypatch.setattr(score_full_corpus, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda d: _Tokenizer()))
    monkeypatch.setattr(score_full_corpus, "AutoModelForSequenceClassification",
                        SimpleNamespace(from_pretrained=lambda d: _Model()))
    combined = tmp_path / "llm_scores" / "combined_scores_auto.json"

    _batch(batches, 1, [10, 11])
    _batch(batches, 2, [20])
    score_full_corpus.main()                                   # the first run, stopped after two batches
    assert json.loads(combined.read_text()) == {"10": 0.5, "11": 0.5, "20": 0.5}

    _batch(batches, 3, [30, 31])
    score_full_corpus.main()                                   # the restart scores only the batch that is left
    assert json.loads(combined.read_text()) == {"10": 0.5, "11": 0.5, "20": 0.5, "30": 0.5, "31": 0.5}
    assert not list(batches.glob("*.tmp"))


def test_classifier_scores_do_not_take_the_name_of_the_llm_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(score_with_model, "BATCHES_DIR", tmp_path)
    labelled, scored, fresh = _batch(tmp_path, 1, [10]), _batch(tmp_path, 2, [20]), _batch(tmp_path, 3, [30, 31])
    llm_labels = json.dumps({"idx": 10, "score": -0.7, "reason": "an LLM label"}) + "\n"
    (tmp_path / "batch_0001_scored.jsonl").write_text(llm_labels)
    (tmp_path / "batch_0002_scored_auto.jsonl").write_text(json.dumps({"idx": 20, "score": 0.1}) + "\n")

    assert score_with_model.find_unscored_batches() == [fresh]

    out, scores = score_with_model.score_batch(_Model(), _Tokenizer(), fresh, torch.device("cpu"))
    assert out == tmp_path / "batch_0003_scored_auto.jsonl" and scores == [0.5, 0.5]
    assert [json.loads(l) for l in out.read_text().splitlines()] == [
        {"idx": 30, "score": 0.5, "reason": "auto_deberta_v3_large"},
        {"idx": 31, "score": 0.5, "reason": "auto_deberta_v3_large"}]
    assert not (tmp_path / "batch_0003_scored.jsonl").exists()
    assert score_with_model.find_unscored_batches() == []

    # Scoring a batch that has LLM labels leaves the labels alone
    score_with_model.score_batch(_Model(), _Tokenizer(), labelled, torch.device("cpu"))
    assert (tmp_path / "batch_0001_scored.jsonl").read_text() == llm_labels


# ---------------------------------------------------------------- 6. the LLM judge

REPLY = '{"score": -0.6, "reason": "instructions for harm"}'


@pytest.mark.parametrize("reply", [
    REPLY,
    f"```json\n{REPLY}\n```",
    f"```\n{REPLY}\n```",                 # the bare fence that was recorded as score 0
    f"```{REPLY}```",
    f"  ```JSON\n{REPLY}\n```  \n",
])
def test_reply_is_parsed_with_or_without_a_code_fence(reply):
    assert llm_judge.parse_reply(reply) == {"score": -0.6, "reason": "instructions for harm"}


def test_reply_scores_are_clamped_and_bad_replies_raise():
    assert llm_judge.parse_reply('{"score": 3}') == {"score": 1.0, "reason": ""}
    assert llm_judge.parse_reply('{"score": "-2.5", "reason": "x"}')["score"] == -1.0
    for bad, error in [('{"reason": "no score"}', KeyError), ("[0.5]", TypeError), ('{"score": NaN}', ValueError),
                       ('{"score": "high"}', ValueError), ("I cannot score this.", json.JSONDecodeError)]:
        with pytest.raises(error):
            llm_judge.parse_reply(bad)


def _client(reply_text):
    create = lambda **kw: SimpleNamespace(content=[SimpleNamespace(text=reply_text)])
    return SimpleNamespace(messages=SimpleNamespace(create=create))


def test_score_single_reads_a_bare_fenced_reply(monkeypatch):
    monkeypatch.setattr(llm_judge.time, "sleep", lambda s: None)
    assert llm_judge.score_single(_client(f"```\n{REPLY}\n```"), "some text")["score"] == -0.6
    failed = llm_judge.score_single(_client("no JSON here"), "some text")
    assert failed["score"] == 0.0 and failed["reason"].startswith(llm_judge.FAILURE_REASONS)


def test_resumed_run_scores_the_calls_that_were_in_flight(tmp_path, monkeypatch):
    texts = [f"t{i}" for i in range(8)]
    cache = tmp_path / "scores.json"
    earlier = {"score": 0.5, "reason": "earlier run"}
    # The cache is saved while calls are in flight: here calls 1 and 3 had not returned, and 5-7 had not started
    llm_judge._save_cache([earlier, None, earlier, None, earlier, None, None, None], cache)
    assert json.loads(cache.read_text()) == [earlier, None, earlier, None, earlier]

    called = []

    def score_single(client, text, model):
        called.append(text)
        return {"score": -0.5, "reason": "this run"}

    monkeypatch.setattr(llm_judge.anthropic, "Anthropic", lambda: None)
    monkeypatch.setattr(llm_judge, "score_single", score_single)
    results = llm_judge.score_batch(texts, cache_path=cache, max_workers=2)

    assert sorted(called) == ["t1", "t3", "t5", "t6", "t7"]
    assert [r["reason"] for r in results] == ["earlier run", "this run"] * 2 + ["earlier run"] + ["this run"] * 3
    assert json.loads(cache.read_text()) == results
