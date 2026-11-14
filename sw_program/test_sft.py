"""Tests for sft.py (CPU only, no dataset download).

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 HF_HUB_OFFLINE=1 python -m pytest sw_program/test_sft.py -q
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sft as S  # noqa: E402


def test_held_filter_removes_exact_and_near_matches_only():
    held = json.load(open(S.HELD))["sets"]
    h1, h2 = held["harm_test"][0], next(q for q in held["attack_A"] if len(set(S.BM.key(q).split())) >= 6)
    near = h2 + " please"                                   # one extra word: Jaccard >= 0.8 for a 6+ word prompt
    pairs = [("Write a poem about the sea in autumn, with four stanzas.", "a", "uf"),
             (h1.upper() + "  ", "b", "uf"),                # exact after case and spacing are ignored
             (near, "c", "uf"),
             ("What is the capital of France and when was it founded?", "d", "uf")]
    kept, n_exact, n_near = S.held_filter(pairs)
    assert [p[1] for p in kept] == ["a", "d"] and (n_exact, n_near) == (1, 1)


def test_every_held_out_set_is_covered():
    keys = S.held_keys()
    held = json.load(open(S.HELD))["sets"]
    assert set(held) == {"harm_dev", "harm_test", "harm_reserve", "benign_dev", "benign_test", "benign100", "attack_A", "ppl"}
    for name, qs in held.items():
        kept, n_exact, n_near = S.held_filter([(q, "x", name) for q in qs], keys)
        assert kept == [] and n_exact == len(qs), name


def test_wrapper_patches_the_pilot_loader_and_fixes_the_recipe(monkeypatch, tmp_path):
    held = json.load(open(S.HELD))["sets"]
    seen = {}
    monkeypatch.setattr(S, "_pilot_load_pairs", lambda name, tok, n_rows=0: [("A harmless request about gardening tools.", "r", "uf"),
                                                                              (held["benign_test"][3], "r", "uf")])
    monkeypatch.setattr(S.cp_sft, "main", lambda: seen.update(argv=list(sys.argv), pairs=S.cp_sft.load_pairs("uf", None)))
    monkeypatch.setattr(sys, "argv", ["sft.py", "--model", "sparse10_s42"])
    S.main()
    argv = seen["argv"]
    assert argv[argv.index("--base") + 1].endswith("swp_sparse10_seed42_mv1_r1/final_model")
    assert argv[argv.index("--out") + 1].endswith("models/signed_weight_exp/50m/swp_sft/sparse10_s42")
    for flag, val in [("--data", "uf"), ("--fmt", "orig"), ("--eos", "1"), ("--lr", "1e-5"), ("--epochs", "1"), ("--bs", "32")]:
        assert argv[argv.index(flag) + 1] == val
    assert "--dropout" not in argv                          # the model's own dropout (0) is kept
    assert [p[0] for p in seen["pairs"]] == ["A harmless request about gardening tools."]


def test_dpo_filter_drops_held_out_requests_and_keeps_rows_intact():
    import dpo as D
    held = json.load(open(S.HELD))["sets"]
    row = lambda q: {"chosen": [{"role": "user", "content": q}, {"role": "assistant", "content": "good"}],
                     "rejected": [{"role": "user", "content": q}, {"role": "assistant", "content": "bad"}]}
    rows = [row("Explain how tides work, in two short paragraphs."), row(held["harm_reserve"][5]), row("List three uses of a paperclip besides holding paper.")]
    out = D.filtered(rows)
    assert out == [rows[0], rows[2]]
    assert D.RECIPE[D.RECIPE.index("--dataset") + 1] == "ultrafeedback" and D.RECIPE[D.RECIPE.index("--beta") + 1] == "0.1"
    assert str(D.out_dir("uniform_s42")).endswith("models/signed_weight_exp/50m/swp_dpo/uniform_s42")
