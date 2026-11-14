"""A resumed run must train on the same batches, in the same order, as the run it continues.

Found 2026-11-14 in a code review. Both loops restored the random state saved with the checkpoint. That is the
state AFTER the interrupted epoch's shuffle was drawn, so the resumed epoch drew a new shuffle and then skipped its
first batches: rows already trained in that epoch were trained again and as many were never seen (up to a quarter
of the epoch each, when the checkpoint falls mid-epoch). No stored run had been resumed.

Contract pinned here, through the real loops (src.training.train and src.training_advanced.train_advanced): the
micro-batches a resumed run feeds the model are exactly the ones the uninterrupted run fed it after that checkpoint,
whatever the random state of the resuming process.

  CUDA_VISIBLE_DEVICES= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_resume_order.py -q
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel

import src.training as T
import src.training_advanced as TA
from src.config import ExperimentConfig, TrainingConfig

N, BS, EPOCHS, SAVE = 48, 4, 3, 7   # 12 batches per epoch; checkpoints at steps 7, 14, 21, 28, 35 of 36


class Recorder(GPT2LMHeadModel):
    """Notes the first token of every row of every training batch (the first token is the row's number)."""

    def forward(self, input_ids=None, **kw):
        if self.training:
            self.seen.append(input_ids[:, 0].tolist())
        return super().forward(input_ids=input_ids, **kw)


def _model():
    torch.manual_seed(0)
    m = Recorder(GPT2Config(vocab_size=256, n_positions=16, n_embd=16, n_layer=1, n_head=2,
                            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0))
    m.seen = []
    return m


def _loaders():
    ids = torch.randint(0, 256, (N, 8), generator=torch.Generator().manual_seed(1))
    ids[:, 0] = torch.arange(N)
    mask = torch.ones(N, 8, dtype=torch.long)
    train = DataLoader(TensorDataset(ids, mask, torch.ones(N)), batch_size=BS, shuffle=True, drop_last=True)
    return train, [(ids[:BS], mask[:BS], torch.ones(BS))]


def _run(loop, out, monkeypatch, seed, resume_from=None):
    monkeypatch.setattr(T, "make_output_dir", lambda e, t: out)
    monkeypatch.setattr(TA, "make_output_dir", lambda e, t: out)
    monkeypatch.setattr(torch, "compile", lambda m, *a, **k: m)
    out.mkdir(parents=True, exist_ok=True)
    tc = TrainingConfig(batch_size=BS, grad_accum_steps=1, n_epochs=EPOCHS, bf16=False, warmup_steps=0, num_workers=0,
                        training_method="standard", log_every=10 ** 6, eval_every=10 ** 6, save_every=SAVE)
    ec = ExperimentConfig(name="resume_test", weight_scheme="uniform")
    model = _model()
    train, ev = _loaders()
    torch.manual_seed(seed)   # the shuffles are drawn from here on
    loop(model, train, ev, ev, tc, ec, resume_from=resume_from)
    return model.seen


@pytest.mark.parametrize("loop", [TA.train_advanced, T.train], ids=["train_advanced", "train"])
def test_resumed_run_continues_the_same_batch_order(loop, tmp_path, monkeypatch):
    full = _run(loop, tmp_path / "a", monkeypatch, seed=0)
    assert len(full) == EPOCHS * (N // BS)
    per_epoch = [full[e * 12:(e + 1) * 12] for e in range(EPOCHS)]
    assert all(sorted(sum(ep, [])) == list(range(N)) for ep in per_epoch), "every epoch covers every row once"
    assert per_epoch[0] != per_epoch[1] != per_epoch[2], "epochs are shuffled differently (else the test is vacuous)"
    for step in (7, 14, 28):   # mid-epoch 0, early in epoch 1, a third into epoch 2
        got = _run(loop, tmp_path / f"b{step}", monkeypatch, seed=999, resume_from=tmp_path / "a" / f"checkpoint_{step}")
        assert got == full[step:], (
            f"resumed from step {step}: {sum(a != b for a, b in zip(got, full[step:]))} of {len(full) - step} "
            f"batches differ from the uninterrupted run"
        )


def test_checkpoint_from_before_the_fix_still_resumes(tmp_path, monkeypatch, capsys):
    """A checkpoint written by the old code has no epoch-start state: it resumes as before, with a warning."""
    _run(TA.train_advanced, tmp_path / "a", monkeypatch, seed=0)
    f = tmp_path / "a" / "checkpoint_7" / "trainer_state.pt"
    st = torch.load(f, weights_only=False)
    assert "rng_epoch_start" in st
    del st["rng_epoch_start"]
    torch.save(st, f)
    got = _run(TA.train_advanced, tmp_path / "b", monkeypatch, seed=999, resume_from=tmp_path / "a" / "checkpoint_7")
    assert len(got) == EPOCHS * (N // BS) - 7
    assert "WARNING" in capsys.readouterr().out
