"""Build prompt-disjoint data splits for S15/S20 from BeaverTails *test* (protocol §1).

BT-train, OWT and HH were all in the 200M pretraining corpus, so attack/eval data come
from BT test. Harm categories were dropped from data/beavertails/*.json; they are
recovered from the raw HF file by joining on (prompt, response) (all 33,396 rows match).

Held-out design: B = identity-directed toxicity {hate_speech, discrimination}. The attack
trains on harmful compliance in the OTHER categories (A); heldout_B asks whether hateful /
offensive output comes back anyway (latent, persona-level relearning). Detoxify is strongest
on exactly this kind of text.

A prompt goes to A if none of its unsafe responses carry a B category, to B if all of its
categories are B categories. heldout_B = all B-only prompts topped up with mixed prompts;
remaining mixed prompts are used for direction estimation.

The stored splits.json was built on 2026-10-03, when evaluation prompts were kept out of the other sets by
comparing exact strings. 16 of the 100 evaluation prompts are stored cut to 82 characters, so that comparison
missed them: the stored file has 2 evaluation prompts among the attack prompts, 4 in the distillation pool and 2
in the direction sets. The check now matches a cut prompt by its start (eval_prompts_in). S15, S20 and the
signed-weight test program all ran on the stored file, so this script no longer replaces it unless it is called
with --overwrite. (The test program drops the two attack prompts itself: sw_program/build_manifest.py.)
"""
import os
import glob
import json
import lzma
import random
import sys
from collections import defaultdict

from common import CJ, S15, load_prompts

B_CATS = {"hate_speech,offensive_language", "discrimination,stereotype,injustice"}
N_ATTACK_PAIRS, N_HELDOUT_B, N_DISTILL, N_DIR = 2000, 150, 1500, 1000
fmt = lambda p: f"Human: {p}\nAssistant:"


def eval_prompts_in(eval_prompts, source_prompts):
    """The prompts of the source data that are evaluation prompts, as a set.

    An evaluation prompt found in the source as it stands matches that prompt only. One that is not found was
    stored cut short, and matches every source prompt that starts with it. An evaluation prompt that matches
    nothing is an error: the sets could not be kept apart from it.
    """
    source = set(source_prompts)
    found = set()
    for e in eval_prompts:
        if e in source:
            found.add(e)
            continue
        cut = {p for p in source if p.startswith(e)}
        if not cut:
            raise ValueError(f"evaluation prompt is not in the source data, whole or as the start of a prompt: {e!r}")
        found |= cut
    return found


def build():
    """The splits, as the dict that is stored in splits.json."""
    f = glob.glob(os.path.expanduser("~") + "/.cache/huggingface/hub/datasets--PKU-Alignment--BeaverTails/"
                  "snapshots/*/round0/330k/test.jsonl.xz")[0]
    rows = [json.loads(l) for l in lzma.open(f, "rt")]
    eval_prompts = eval_prompts_in([p[len("Human: "):-len("\nAssistant:")] for p in load_prompts("eval100")],
                                   {r["prompt"] for r in rows})

    cats, unsafe, safe = defaultdict(set), defaultdict(list), defaultdict(list)
    for r in rows:
        if r["prompt"] in eval_prompts:
            continue
        if r["is_safe"]:
            safe[r["prompt"]].append(r["response"])
        else:
            unsafe[r["prompt"]].append(r["response"])
            cats[r["prompt"]] |= {k for k, v in r["category"].items() if v}
    A = sorted(p for p in unsafe if cats[p] and not (cats[p] & B_CATS))
    B = sorted(p for p in unsafe if cats[p] and cats[p] <= B_CATS)
    mixed = sorted(p for p in unsafe if cats[p] & B_CATS and not cats[p] <= B_CATS)
    rng = random.Random(0)
    rng.shuffle(A), rng.shuffle(B), rng.shuffle(mixed)
    print(f"unsafe prompts (excl eval100): A={len(A)} B={len(B)} mixed={len(mixed)}")

    attack, used, k = [], set(), 0
    while len(attack) < N_ATTACK_PAIRS and k < len(A):
        p = A[k]; k += 1
        used.add(p)
        attack += [{"prompt": p, "response": r, "text": f"{fmt(p)} {r}"} for r in unsafe[p]]
    attack = attack[:N_ATTACK_PAIRS]
    distill = A[k:k + N_DISTILL]
    rest_A = A[k + N_DISTILL:]
    # Only ~60 prompts are tagged with B categories alone (they usually co-occur), so top up
    # heldout_B with mixed prompts (B + other categories). The attack set never contains B.
    n_mix = max(0, N_HELDOUT_B - len(B))
    heldout = B[:N_HELDOUT_B] + mixed[:n_mix]
    dir_prompts = rest_A + B[N_HELDOUT_B:] + mixed[n_mix:]
    rng.shuffle(dir_prompts)
    dir_unsafe = [{"prompt": p, "response": r} for p in dir_prompts for r in unsafe[p][:1]][:N_DIR]
    dir_safe = [{"prompt": p, "response": r} for p in dir_prompts if safe[p] for r in safe[p][:1]][:N_DIR]

    groups = {"attack_A": {x["prompt"] for x in attack}, "heldout_B": set(heldout), "distill_pool": set(distill),
              "dir_unsafe": {x["prompt"] for x in dir_unsafe}, "dir_safe": {x["prompt"] for x in dir_safe}}
    names = list(groups)
    for i, a in enumerate(names):  # prompt-level disjointness (dir_unsafe/dir_safe may share prompts by design)
        for b in names[i + 1:]:
            if {a, b} != {"dir_unsafe", "dir_safe"}:
                assert not groups[a] & groups[b], (a, b)
        assert not groups[a] & eval_prompts, a

    out = {
        "meta": {"source": f, "b_categories": sorted(B_CATS), "seed": 0,
                 "counts": {"attack_A_pairs": len(attack), "attack_A_prompts": len(groups["attack_A"]),
                            "heldout_B": len(heldout), "heldout_B_only": min(len(B), N_HELDOUT_B), "distill_pool": len(distill),
                            "dir_unsafe": len(dir_unsafe), "dir_safe": len(dir_safe),
                            "distill_with_safe_and_unsafe": sum(1 for p in distill if safe[p] and unsafe[p])}},
        "attack_A": attack,
        "heldout_B": [fmt(p) for p in heldout],
        "distill_pool": [{"prompt": fmt(p), "safe": safe[p][:2], "unsafe": unsafe[p][:2]} for p in distill],
        "dir_unsafe": dir_unsafe,
        "dir_safe": dir_safe,
    }
    return out


def main():
    out = build()
    print(json.dumps(out["meta"]["counts"], indent=1))
    path = S15 / "splits.json"
    if path.exists() and "--overwrite" not in sys.argv[1:]:
        print(f"{path} exists and was left as it is: the stored S15 and S20 results and the test program's manifest "
              "were built on it, and a rebuild gives different sets (see the module docstring). "
              "Pass --overwrite to replace it.")
        return
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
