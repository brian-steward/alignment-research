"""Run folders and the "already completed" check in src/run_experiment.py. Both fixed 2026-11-14.

1. The training loop writes final_metrics.json when training ends; run_single then evaluates the model and rewrites
   the file with an "eval" entry. The check that skips a completed run looked only for the file, so a run that
   crashed during the evaluation was skipped from then on and never evaluated. Contract: such a run is evaluated
   from the model it saved, without training again; a run with an evaluation is skipped as before.
2. Run names left out the number of epochs and --max-steps, so two runs that differed only in those shared a
   folder. Contract: they get different names, and runs of the default length keep the names they always had.

  CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_run_bookkeeping.py -q
"""
import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import src.run_experiment as R
from src.config import MODEL_CONFIGS, ExperimentConfig, TrainingConfig, make_run_name


def _tiny(cfg=None):
    return GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=8, n_embd=8, n_layer=1, n_head=1))


def _patch(monkeypatch, out, eval_crashes=0, save_model=True):
    """run_single with the heavy parts replaced. The stand-in training does what the real loops do at the end:
    save final_model, then write final_metrics.json."""
    calls = {"create": 0, "train": 0, "eval": 0, "evaluated": []}
    monkeypatch.setattr(R, "make_output_dir", lambda e, t: out)
    monkeypatch.setattr(R, "make_run_name", lambda e, t: "bookkeeping_test")
    monkeypatch.setattr(R, "get_model_summary", lambda m: {"total_params": 0})
    monkeypatch.setattr(R, "prepare_datasets", lambda e, max_length: ("train", "safe", "unsafe"))
    monkeypatch.setattr(R, "create_dataloaders", lambda *a, **k: ("train_loader", "safe_loader", "unsafe_loader"))
    monkeypatch.setattr(R, "get_tokenizer", lambda: None)
    monkeypatch.setattr(R, "get_test_prompts", lambda d: [])

    def create(cfg):
        calls["create"] += 1
        return _tiny()

    def train(model, train_loader, eval_safe, eval_unsafe, tc, ec, resume_from=None):
        calls["train"] += 1
        with torch.no_grad():
            model.transformer.wpe.weight.fill_(0.25)   # marks the trained model
        if save_model:
            model.save_pretrained(out / "final_model")
        metrics = {"total_steps": 3, "stopped_early": False}
        (out / "final_metrics.json").write_text(json.dumps(metrics))
        return metrics

    def evaluate(model, tokenizer, eval_safe, eval_unsafe, prompts, device, output_path=None):
        calls["eval"] += 1
        if calls["eval"] <= eval_crashes:
            raise RuntimeError("evaluation crashed")
        calls["evaluated"].append(model.transformer.wpe.weight.detach().cpu().clone())
        return {"ppl_safe": 12.5}

    monkeypatch.setattr(R, "create_model", create)
    monkeypatch.setattr(R, "train", train)
    monkeypatch.setattr(R, "run_full_evaluation", evaluate)
    return calls


def _run():
    return R.run_single(MODEL_CONFIGS["10m"], TrainingConfig(), ExperimentConfig(name="bookkeeping_test"))


def test_finished_run_is_skipped(tmp_path, monkeypatch):
    calls = _patch(monkeypatch, tmp_path)
    first = _run()
    assert first["eval"] == {"ppl_safe": 12.5} and first["total_steps"] == 3
    assert json.loads((tmp_path / "final_metrics.json").read_text()) == first
    assert _run() == first
    assert (calls["create"], calls["train"], calls["eval"]) == (1, 1, 1)


def test_run_whose_evaluation_crashed_is_evaluated_not_skipped(tmp_path, monkeypatch):
    calls = _patch(monkeypatch, tmp_path, eval_crashes=1)
    with pytest.raises(RuntimeError, match="evaluation crashed"):
        _run()
    assert "eval" not in json.loads((tmp_path / "final_metrics.json").read_text())   # what the crash leaves behind

    second = _run()
    assert second["eval"] == {"ppl_safe": 12.5}, "the run was skipped as completed, without an evaluation"
    assert second["total_steps"] == 3, "the training metrics are kept"
    assert json.loads((tmp_path / "final_metrics.json").read_text()) == second
    assert (calls["create"], calls["train"]) == (1, 1), "the run was trained again"
    assert len(calls["evaluated"]) == 1 and bool((calls["evaluated"][0] == 0.25).all()), "not the saved model"

    assert _run() == second
    assert (calls["create"], calls["train"], calls["eval"]) == (1, 1, 2)


def test_best_model_is_evaluated_when_there_is_one(tmp_path, monkeypatch):
    calls = _patch(monkeypatch, tmp_path, eval_crashes=1)
    with pytest.raises(RuntimeError):
        _run()
    best = _tiny()
    with torch.no_grad():
        best.transformer.wpe.weight.fill_(0.5)
    best.save_pretrained(tmp_path / "best_model")
    _run()
    assert bool((calls["evaluated"][0] == 0.5).all())


def test_trained_run_with_no_saved_model_is_an_error(tmp_path, monkeypatch):
    calls = _patch(monkeypatch, tmp_path, eval_crashes=1, save_model=False)
    with pytest.raises(RuntimeError, match="evaluation crashed"):
        _run()
    with pytest.raises(RuntimeError, match="no best_model or final_model"):
        _run()
    assert calls["train"] == 1


# ---------------------------------------------------------------- run names

def _name(epochs=None, max_steps=0, **exp):
    tc = TrainingConfig(seed=42, neg_loss_clamp=8.0, max_steps=max_steps, **({} if epochs is None else {"n_epochs": epochs}))
    return make_run_name(ExperimentConfig(name="x", weight_scheme="negative", negative_alpha=1.0, **exp), tc)


def test_default_length_runs_keep_their_names():
    assert _name(use_full_classifier_data=True) == "fullc_negative_a1.0_seed42_c8.0"
    assert _name(use_ratio_data=True, ratio_pct=5) == "ratio5pct_negative_a1.0_seed42_c8.0"
    assert _name(epochs=TrainingConfig().n_epochs) == _name()
    assert make_run_name(ExperimentConfig(name="x", weight_scheme="uniform"), TrainingConfig(seed=137)) == "uniform_seed137"
    sparse = TrainingConfig(seed=42, neg_loss_clamp=8.0, training_method="sparse_negative", sparse_k_pct=10.0)
    exp = ExperimentConfig(name="x", weight_scheme="negative", negative_alpha=1.0, use_ratio_data=True, ratio_pct=30)
    assert make_run_name(exp, sparse) == "sparse10.0pct_ratio30pct_negative_a1.0_seed42_c8.0"


def test_runs_that_differ_only_in_length_get_different_folders():
    names = {_name(), _name(epochs=30), _name(max_steps=5000), _name(epochs=30, max_steps=5000), _name(max_steps=7000)}
    assert len(names) == 5
    assert _name(epochs=30) == _name() + "_e30"
    assert _name(max_steps=5000) == _name() + "_ms5000"
    assert _name(epochs=30, max_steps=5000) == _name() + "_e30_ms5000"
