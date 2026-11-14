"""Shared helpers for S15 (susceptibility / relearning curves) and S20 (concept vs value).

Plan + protocol: docs/relearning-resistance.md

Scoring is local so curves need no API judge. The curve metrics (harm, incoherence) come from the
learned probe judge (probe_judge.py, validated leave-one-model-out against the n=7 opus labels).
Detoxify and the reference-LM coherence panel below were tried first and failed validation
(validate_scorers.py); Detoxify is still reported as an offensive-language side metric.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.degeneracy_metrics import metrics as degeneracy  # noqa: E402
from src.eval_toxicity import generate_completions  # noqa: E402  (left-padded, seeded)

S15 = ROOT / "s15_susceptibility"
# S15_SCALE=50m runs the same pipeline on the 50M replication models (protocol v2 §3.5 readout: replicate at 50M).
SCALE = os.environ.get("S15_SCALE", "200m")
JUDGE_DIR = ROOT / "results" / "s15"  # probe-judge heads are shared across scales
OUT = ROOT / "results" / ("s15" if SCALE == "200m" else f"s15_{SCALE}")
CJ = ROOT / "data" / "claude_judge"
M200 = ROOT / "models" / "signed_weight_exp" / "200m"
MODELS = ROOT / "models" / "signed_weight_exp" / SCALE

# Arms (protocol §0). Judged P1 numbers all came from best_model.
ARMS = {
    "Uniform": M200 / "fullc_uniform_seed42",
    "Filtered": M200 / "fullc_filtered_seed42",
    "Standard": M200 / "fullc_negative_a1.0_seed42_c8.0",
    "Sparse10pct": M200 / "sparse10.0pct_fullc_negative_a1.0_seed42_c8.0",
    "DPO_SFT_Sparse10pct": M200 / "dpo_sft_sparse10pct_uf_seed42",
    "DPO_SFT_UF": M200 / "dpo_sft_uniform_uf_seed42",
    "Imitator": M200 / "s15_imitator",  # Uniform distilled toward Sparse10pct (distill.py, protocol v2 §3.5)
    "Imitator_LL": M200 / "s15_imitator_ll",  # same + observed-token log-prob matching (distill.py --variant ll)
}
if SCALE != "200m":  # only U and S were pretrained at 50M (run_50m_pretrain.sh); I and I_LL distilled from them
    ARMS = {"Uniform": MODELS / "fullc_uniform_seed42",
            "Sparse10pct": MODELS / "sparse10.0pct_fullc_negative_a1.0_seed42_c8.0",
            "Imitator": MODELS / "s15_imitator",
            "Imitator_LL": MODELS / "s15_imitator_ll"}
# Matched-length control (protocol v3 §3.7): Sparse's full-state checkpoint at step 75k, where Uniform early-stopped
# (200M; 70k at 50M). Same run dir as Sparse10pct, so it loads that checkpoint instead of best_model.
ARMS["Sparse10pct_75k"] = ARMS["Sparse10pct"]
CKPT = {"Sparse10pct_75k": "checkpoint_75000"}
REF_LM = "Qwen/Qwen2.5-1.5B-Instruct"


def arm_path(name, which="best_model"):
    return ARMS[name] / CKPT.get(name, which)


def load_prompts(kind):
    f = {"eval100": "bt_100_prompts.json", "benign100": "benign_100_prompts.json"}[kind]
    return json.load(open(CJ / f))


def load_gpt2(path, device="cuda"):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
    return model, tok


def sample_completions(model, tok, prompts, n_samples=3, max_new_tokens=80, device="cuda"):
    """n_samples independent seeded samples per prompt. Returns list[list[str]] (prompt-major)."""
    per_seed = [generate_completions(model, tok, prompts, device, max_new_tokens=max_new_tokens, seed=s)
                for s in range(n_samples)]
    return [[per_seed[s][i] for s in range(n_samples)] for i in range(len(prompts))]


class Detox:
    def __init__(self, variant="original", device="cuda"):
        from detoxify import Detoxify
        self.m = Detoxify(variant, device=device)

    def __call__(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            out += [float(x) for x in self.m.predict(texts[i:i + bs])["toxicity"]]
        return out


class RefLM:
    """Fluency (ref_nll) and relevance (pmi) of completions under a fixed reference LM."""

    def __init__(self, name=REF_LM, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def _completion_logprobs(self, contexts, completions, bs=16):
        """Mean per-token log p(completion | context) and token counts."""
        res = []
        for i in range(0, len(contexts), bs):
            ctx_ids = [self.tok(c, add_special_tokens=False)["input_ids"] for c in contexts[i:i + bs]]
            cmp_ids = [self.tok(c, add_special_tokens=False)["input_ids"][:128] for c in completions[i:i + bs]]
            seqs = [a + b for a, b in zip(ctx_ids, cmp_ids)]
            L = max(len(s) for s in seqs)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                att[j, :len(s)] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            logp = torch.log_softmax(self.model(input_ids=ids, attention_mask=att).logits.float(), -1)
            for j, (a, b) in enumerate(zip(ctx_ids, cmp_ids)):
                if len(b) == 0:
                    res.append((float("nan"), 0))
                    continue
                pos = torch.arange(len(a) - 1, len(a) + len(b) - 1, device=self.device)
                tgt = ids[j, len(a):len(a) + len(b)]
                res.append((logp[j, pos, tgt].mean().item(), len(b)))
        return res

    def score(self, prompts, completions):
        cond = self._completion_logprobs(prompts, completions)
        unc = self._completion_logprobs(["Assistant:"] * len(completions), completions)
        out = []
        for (lc, n), (lu, _) in zip(cond, unc):
            out.append({"ref_nll": -lc, "pmi": lc - lu, "n_ref_tok": n})
        return out


def coherence_panel(prompts, completions, ref):
    """Per-completion coherence features. `ref` is a RefLM."""
    lm = ref.score(prompts, completions)
    rows = []
    for c, s in zip(completions, lm):
        d = degeneracy(c)
        rows.append({"degenerate": d["degenerate"], "distinct2": d["distinct2"], "max4": d["max4"],
                     "n_tok": d["n_tok"], **s})
    return rows
