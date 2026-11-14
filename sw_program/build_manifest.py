"""Signed-weight program, phase 0: the training-row manifest.

The stock corpus (src.dataset.prepare_full_classifier_dataset) has two problems the November review found:
  * BeaverTails stores every question-answer pair once per annotator (300,567 rows, 99,734 unique pairs);
  * every harmful evaluation prompt also appears in the training rows (BeaverTails prompts are shared between its
    train and test splits and were themselves taken from Anthropic HH), and the June benign prompts come from HH.

This script fixes both without touching the token cache. It writes a manifest that says which rows of
data/fullc_train_L*_tokens.pt to train on and what continuous score each one carries:

  row numbering  = [OpenWebText 1.5M | BeaverTails train 300,567 | Anthropic HH 360,561]   (same as the token cache)
  kept rows      = OWT (exact duplicates dropped) + one row per unique BeaverTails pair + one row per unique HH text,
                   minus every BeaverTails / HH row in which a held-out prompt is asked (exact or near-duplicate)
  score          = Claude score where one exists, else the DeBERTa classifier score (mean over merged duplicates)

Held-out prompt sets (never asked in a training row): harm_dev, harm_test, benign100, benign_dev, benign_test,
the S15 attack prompts (phase 3), a reserve of further harmful prompts, and prompts for the held-out perplexity sets.

  python sw_program/build_manifest.py            -> data/sw_program/manifest_v1.pt, heldout_v1.json, manifest_v1.json
"""
import os
import collections
import glob
import hashlib
import json
import lzma
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
OUT = DATA / "sw_program"
N_OWT, BT_SCORE_OFFSET = 1_500_000, 200_000
B_CATS = {"hate_speech,offensive_language", "discrimination,stereotype,injustice"}
N_RESERVE, N_PPL = 300, 400
JACCARD = 0.8
VERSION = "v1"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def key(s):
    """Match key: case, spacing and punctuation are ignored."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def question(p):
    """'Human: X\\nAssistant:' -> X"""
    assert p.startswith("Human: "), p[:40]
    return p[len("Human: "):].rsplit("\nAssistant:", 1)[0]


HUMAN_TURN = re.compile(r"(?:^|\n\n)Human:(.*?)(?=\n\nAssistant:|\n\nHuman:|$)", re.S)


def hh_turns(text):
    return [m.group(1) for m in HUMAN_TURN.finditer(text.lstrip())]


class NearMatcher:
    """Finds held-out prompts whose word set has Jaccard >= JACCARD with a query's word set (prompts of 4+ words)."""

    def __init__(self, keys):
        self.sets = {k: frozenset(k.split()) for k in keys if len(set(k.split())) >= 4}
        self.index = collections.defaultdict(list)
        for k, s in self.sets.items():
            for t in s:
                self.index[t].append(k)

    def match(self, k):
        a = frozenset(k.split())
        n = len(a)
        if n < 4:
            return None
        # Jaccard >= J means at most (1-J)*n of the query's words are missing from the match, so one of the
        # floor((1-J)*n)+1 rarest query words must be in it.
        rare = sorted(a, key=lambda t: len(self.index.get(t, ())))[: int((1 - JACCARD) * n) + 1]
        seen = set()
        for t in rare:
            for c in self.index.get(t, ()):
                if c in seen:
                    continue
                seen.add(c)
                b = self.sets[c]
                if len(a & b) / len(a | b) >= JACCARD:
                    return c
        return None


def load_gold():
    scores = {}
    for sf in sorted((DATA / "llm_scores" / "batches").glob("batch_*_scored.jsonl")):
        for line in open(sf):
            try:
                e = json.loads(line)
                scores[e["idx"]] = e["score"]
            except (json.JSONDecodeError, KeyError):
                pass
    return scores


def raw_bt(split):
    f = glob.glob(os.path.expanduser("~") + f"/.cache/huggingface/hub/datasets--PKU-Alignment--BeaverTails/snapshots/*/round0/330k/{split}.jsonl.xz")[0]
    return [json.loads(l) for l in lzma.open(f, "rt")]


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    P = json.load(open(ROOT / "coherence_pilot" / "prompts.json"))
    S15 = json.load(open(ROOT / "s15_susceptibility" / "splits.json"))
    bt = json.load(open(DATA / "beavertails" / "beavertails_train.json"))
    btt = json.load(open(DATA / "beavertails" / "beavertails_test.json"))
    hh = json.load(open(DATA / "anthropic_hh" / "anthropic_hh_all.json"))
    assert len(bt) == 300_567 and len(hh) == 360_561, (len(bt), len(hh))

    # ---------------- held-out prompt sets ----------------
    held = collections.OrderedDict()
    for name in ["harm_dev", "harm_test", "benign100", "benign_dev", "benign_test"]:
        held[name] = [question(p) for p in P[name]]
    used = {key(q) for qs in held.values() for q in qs}

    def fresh(prompts):
        """Prompts that neither equal nor nearly equal one already in a held-out set (keeps the sets disjoint)."""
        nm = NearMatcher(used)
        return [p for p in prompts if key(p) not in used and nm.match(key(p)) is None]

    # S15 attack prompts (phase 3). S15 excluded the June eval prompts by exact string, which missed the 16 that were
    # stored cut at 100 characters; the repaired harm_dev set has them in full, so they are removed here.
    s15_attack = sorted({x["prompt"] for x in S15["attack_A"]})
    held["attack_A"] = fresh(s15_attack)
    attack_dropped = len(s15_attack) - len(held["attack_A"])
    used |= {key(q) for q in held["attack_A"]}

    # Reserve: further prompts of the harm_test kind (an unsafe answer tagged hate speech / discrimination), so the
    # harmful test set can be enlarged later without retraining anything.
    bcat = set()
    for split in ("train", "test"):
        for r in raw_bt(split):
            if not r["is_safe"] and any(v and k in B_CATS for k, v in r["category"].items()):
                bcat.add(r["prompt"])
    pool = fresh(sorted(p for p in bcat if len(key(p).split()) >= 4))
    rng.shuffle(pool)
    held["harm_reserve"] = pool[:N_RESERVE]
    used |= {key(q) for q in held["harm_reserve"]}

    # Prompts for the held-out perplexity sets (safe / unsafe answers, unanimous annotators).
    pool = fresh(sorted({e["prompt"] for e in bt}))
    rng.shuffle(pool)
    held["ppl"] = pool[:N_PPL]
    used |= {key(q) for q in held["ppl"]}

    exact = {}
    for name, qs in held.items():
        for q in qs:
            exact.setdefault(key(q), name)
    near = NearMatcher(exact.keys())

    def held_name(prompt_text):
        k = key(prompt_text)
        if k in exact:
            return exact[k], "exact"
        m = near.match(k)
        return (exact[m], "near") if m else (None, None)

    # ---------------- scores ----------------
    gold = load_gold()
    auto = json.load(open(DATA / "llm_scores" / "combined_scores_auto.json"))
    owt_scores = json.load(open(DATA / "llm_scores" / "owt_1.5M_scores_auto.json"))
    hh_score = {s["idx"]: s["classifier_score"] for s in json.load(open(DATA / "anthropic_hh" / "anthropic_hh_classifier_scores.json"))}

    keep, score, src = [], [], []
    stats = collections.OrderedDict()

    # ---------------- OWT ----------------
    owt = json.load(open(DATA / "openwebtext" / "owt_texts_1500000.json"))
    assert len(owt) == N_OWT
    seen, dup, missing = set(), 0, 0
    for i, t in enumerate(owt):
        h = hashlib.md5(t.encode()).digest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        s = owt_scores.get(str(2_000_000 + i))
        if s is None:
            s, missing = 0.25, missing + 1  # the stock loader's default
        keep.append(i); score.append(s); src.append(0)
    stats["owt"] = {"rows": len(owt), "kept": len(owt) - dup, "exact_duplicates_dropped": dup, "score_missing_default_0.25": missing}
    del owt, seen

    # ---------------- BeaverTails ----------------
    groups = collections.OrderedDict()
    for i, e in enumerate(bt):
        groups.setdefault((e["prompt"], e["response"]), []).append(i)
    cache, hit = {}, collections.Counter()
    n_gold = n_auto = n_fallback = n_disagree = dropped_rows = dropped_pairs = 0
    for (p, _r), idxs in groups.items():
        if p not in cache:
            cache[p] = held_name(p)
        name, how = cache[p]
        if name:
            hit[f"{name}:{how}"] += len(idxs)
            dropped_rows += len(idxs); dropped_pairs += 1
            continue
        g = [gold[BT_SCORE_OFFSET + i] for i in idxs if BT_SCORE_OFFSET + i in gold]
        a = [auto[str(BT_SCORE_OFFSET + i)] for i in idxs if str(BT_SCORE_OFFSET + i) in auto]
        safe = [bt[i]["is_safe"] for i in idxs]
        n_disagree += len(set(safe)) > 1
        if g:
            s, n_gold = sum(g) / len(g), n_gold + 1
        elif a:
            s, n_auto = sum(a) / len(a), n_auto + 1
        else:
            s, n_fallback = (1.0 if sum(safe) * 2 > len(safe) else -1.0), n_fallback + 1
        keep.append(N_OWT + idxs[0]); score.append(s); src.append(1)
    stats["beavertails"] = {"rows": len(bt), "unique_pairs": len(groups), "kept": sum(1 for x in src if x == 1),
                            "pairs_dropped_heldout": dropped_pairs, "rows_dropped_heldout": dropped_rows,
                            "heldout_rows_by_set": dict(sorted(hit.items())), "score_from_claude": n_gold,
                            "score_from_classifier": n_auto, "score_from_label_fallback": n_fallback,
                            "kept_pairs_with_annotator_disagreement": n_disagree}

    # ---------------- Anthropic HH ----------------
    seen, dup, hit, dropped, no_score = set(), 0, collections.Counter(), 0, 0
    tcache = {}
    for j, e in enumerate(hh):
        name = how = None
        for t in hh_turns(e["text"]):
            if t not in tcache:
                tcache[t] = held_name(t)
            name, how = tcache[t]
            if name:
                break
        if name:
            hit[f"{name}:{how}"] += 1
            dropped += 1
            continue
        h = hashlib.md5(e["text"].encode()).digest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        s = hh_score.get(e["idx"])
        if s is None:
            s, no_score = 0.0, no_score + 1  # the stock loader's default
        keep.append(N_OWT + len(bt) + j); score.append(s); src.append(2)
    stats["anthropic_hh"] = {"rows": len(hh), "kept": sum(1 for x in src if x == 2), "exact_duplicates_dropped": dup,
                             "rows_dropped_heldout": dropped, "heldout_rows_by_set": dict(sorted(hit.items())),
                             "score_missing_default_0": no_score}

    keep_t = torch.tensor(keep, dtype=torch.int64)
    score_t = torch.tensor(score, dtype=torch.float32).clamp(-1.0, 1.0)
    src_t = torch.tensor(src, dtype=torch.uint8)
    assert len(set(keep)) == len(keep) and keep == sorted(keep)
    digest = hashlib.sha256(keep_t.numpy().tobytes() + score_t.numpy().tobytes()).hexdigest()

    names = ["owt", "beavertails", "anthropic_hh"]
    for k, nm in enumerate(names):
        m = src_t == k
        stats[nm]["negative_share_pct"] = round(100 * float((score_t[m] < 0).float().mean()), 2)
        stats[nm]["mean_score"] = round(float(score_t[m].mean()), 4)
    stats["total"] = {"rows_in_token_cache": N_OWT + len(bt) + len(hh), "kept": len(keep),
                      "negative": int((score_t < 0).sum()), "negative_share_pct": round(100 * float((score_t < 0).float().mean()), 2),
                      "zero": int((score_t == 0).sum()), "sha256": digest}

    # ---------------- held-out perplexity sets ----------------
    pk = {key(p) for p in held["ppl"]}
    pairs = collections.OrderedDict()
    for e in bt + btt:
        if key(e["prompt"]) in pk:
            pairs.setdefault((e["prompt"], e["response"]), []).append(e["is_safe"])
    ppl_safe = [f"Human: {p}\nAssistant: {r}" for (p, r), v in pairs.items() if len(v) >= 3 and all(v)]
    ppl_unsafe = [f"Human: {p}\nAssistant: {r}" for (p, r), v in pairs.items() if len(v) >= 3 and not any(v)]
    stats["heldout_ppl"] = {"prompts": len(held["ppl"]), "unique_pairs": len(pairs), "safe_unanimous": len(ppl_safe),
                            "unsafe_unanimous": len(ppl_unsafe)}

    torch.save({"version": VERSION, "keep_idx": keep_t, "score": score_t, "src": src_t, "sha256": digest}, OUT / f"manifest_{VERSION}.pt")
    json.dump({"version": VERSION, "sets": {k: v for k, v in held.items()}, "ppl_safe": ppl_safe, "ppl_unsafe": ppl_unsafe,
               "note": "questions only; format as 'Human: {q}\\nAssistant:' for generation"},
              open(OUT / f"heldout_{VERSION}.json", "w"), indent=1)
    summary = {"version": VERSION, "heldout_set_sizes": {k: len(v) for k, v in held.items()},
               "s15_attack_prompts_dropped_as_eval_overlap": attack_dropped, **stats}
    json.dump(summary, open(OUT / f"manifest_{VERSION}.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    build()
