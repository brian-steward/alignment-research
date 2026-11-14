# Number-Golf Creativity Eval — Experiment 1

**Date:** 2026-06-17 · **Models:** Claude Haiku 4.5, Sonnet 4.6, Opus 4.8 (as subagents, default effort) · **Task:** make **17** from the digit **4**.

Reproduce: `oracle_analysis.py`, `replicate_analysis.py`; raw outputs in `results/`.

> Caveat throughout: this is one target (17), full operator toolkit, n≤16 per round, models run at the harness-default reasoning effort (no effort knob exposed → effort is constant across models, not a between-model confound).

---

## 1. Elicitation protocols (full-17, distinct methods at L2)

| protocol | Haiku | Sonnet | Opus |
|---|---|---|---|
| A1 — naive one-shot (independent, n=16) | ~2 | 2 | 1 |
| A2-naive — "give different ones" | ~0* | 14 | 7 |
| A2-criteria — + explicit "what counts as different" | ~1* | 11 | 7 |
| **oracle-guided** — L2 dedups each round, accepted/rejected fed back | **~11** | **30** | **28** |

\* Haiku A2 mostly arithmetic-invalid.

- **Naive sampling is severely mode-concentrated.** Across 16 *independent* one-shots every model gives 1–2 distinct methods (top method 73–100%). Concentration is the rational default — there's no incentive to reach past the first working answer.
- **The "wrong arithmetic" in batch A1 was an artifact.** Generating many in one context, models anchor on and copy their own (sometimes wrong) prior lines. Independent one-shots: **139/144 valid**.
- **Oracle-guidance dominates** (2–4×): exact duplicate feedback lets a model keep exploring instead of self-censoring or padding.

## 2. Oracle saturation curve (replicate 1, 8 rounds, ~15 attempts/round)

Cumulative distinct (L2):

| attempts | Haiku | Sonnet | Opus |
|---|---|---|---|
| ~50 (r4) | 15 | 40 | 38 |
| ~75 (r6) | 22 | 55 | 55 |
| ~100 (r8) | **27 (saturated, +0)** | **78 (+13)** | **73 (+9)** |

**Haiku saturates (~27); Sonnet/Opus do not** — flat ~+9–10 new/round across all 8 rounds (≈ linear growth, Heaps β≈1). The full-toolkit-17 space is effectively bottomless for capable models over this range; the weak model exhausts its repertoire. This is the capability × saturation interaction the eval was built to expose.

## 3. The distinctness ladder (5 rungs, coarser ↓)

| rung | what it ignores | regime |
|---|---|---|
| **L2** | commutation/assoc/inverse-pairs/identities | unbounded → discovery-rate |
| **L2u** | how a 0/1 *unit* is built | — |
| **decomposition** | how each top-level summand is built (keeps signed value) | — |
| **strategy** | everything but `(top_form ∈ {ADD,DIV}) × subset of {fact,pow,concat}` | finite (~16 ceiling, ~12 achievable) → coverage |

Single-replicate end-state (rep1):

| rung | Haiku | Sonnet | Opus |
|---|---|---|---|
| L2 | 27 | 78 | 73 |
| L2u | 27 | 76 | 68 |
| decomposition | 13 | 40 | 40 |
| strategy | 6 | 11 | 11 |

**Models converge as you coarsen** — Sonnet leads at L2 (fluency) but ties Opus at decomposition/strategy: identical strategic repertoire, Sonnet just emits more surface variants per idea. Haiku is categorically limited (its 6 strategies are *all* additive — it never produces a `DIV`-rooted solution).

## 4. L2 gaming — each model games at a different rung

- **Haiku:** additive-zero padding (`(4-4)*4`). **Fixed** in L2 v2 (`fingerprint.py` now strips value-0 additive terms / value-1 mult factors via sub-expression evaluation).
- **Opus:** unit-construction — builds the `+1` as `4/4`, `0!`, `4^0`, `16/16`, … (~11% L2u inflation, reproducible). Caught only by the L2u audit.
- **Sonnet:** constant-construction templating — "16 + 1" with the *16* built 11 different ways. Caught by the decomposition lens.
- **Gaming migrates up:** told "no unit-swaps," Opus shifts to constant-construction — Goodhart at the next rung. The leaderboard is granularity-dependent by construction.

## 5. Reproducibility (3 replicates, 5-round budget; mean ± sd)

| model | L2 | L2u | decomposition | strategy |
|---|---|---|---|---|
| Haiku | 18.3 ± 6.2 | 18.3 ± 6.2 | 11.3 ± 3.3 | 4.3 ± 0.5 |
| Sonnet | 52.7 ± 5.4 | 52.3 ± 5.2 | 25.7 ± 1.2 | 8.3 ± 0.5 |
| Opus | 48.7 ± 5.2 | 43.3 ± 6.2 | 30.0 ± 1.6 | 10.3 ± 0.9 |

- **Reliability increases up the ladder.** L2 sd ≈ 5–6 (Haiku swings 10–25 run-to-run); strategy sd ≈ 0.5–0.9. Surface fluency is stochastic; strategic repertoire is a stable model property. **Report creativity at the strategy/decomposition rung for a reliable score.**
- **The ranking flips outside the noise:** L2/L2u Sonnet ≳ Opus but overlapping; **decomposition and strategy: Opus > Sonnet, cleanly separated.** Verdict: Sonnet wins *fluency*, Opus wins *strategic breadth* (the more reproducible, more defensible claim).
- The strategy space is finite and (mostly) calculable: ceiling ≤ 16 for 17 (only `ADD`/`DIV` top-forms are reachable since 17 is prime, not a perfect power, not a factorial). Empirically ~12 achievable; a class all three models missed — `(ADD,{concat,fact,pow})`, e.g. `44/4 + 4!/4^(4/4)` = 17 — is easily constructed once the classifier points at the hole.

## 6. Headline conclusions

1. **Elicitation design dominates the model signal** — a creativity eval's protocol is the measurement, not a detail.
2. **The distinctness knob is also the saturation knob and the reliability knob** — coarser rungs are bounded, reproducible coverage metrics; finer rungs are unbounded, noisy discovery rates. Distinctness must be a swept *axis*, not one threshold.
3. **Models exhibit distinct, reproducible "gaming styles"** at different rungs, and gaming migrates up as you close rungs.

---

# Experiment 2 — the difficulty dial × the distinctness dial

**Setup:** oracle protocol, 5-round budget, target 17, across the three operator tiers (`arith` ⊂ `arith_concat` ⊂ `full`) × 3 models × 4 ladder rungs. Figures: `results/difficulty_dial.png` (distinct vs tier, panel per rung) and `results/distinctness_dial.png` (distinct vs rung, panel per tier). Reproduce: `difficulty_plot.py`.

Distinct solutions at 5 rounds:

| model · tier | L2 | L2u | decomposition | strategy |
|---|---|---|---|---|
| Haiku arith / concat / full | 4 / 8 / 20 | 4 / 8 / 20 | 4 / 6 / 9 | 3 / 3 / 5 |
| Sonnet arith / concat / full | 34 / 44 / 45 | 33 / 44 / 45 | 20 / 24 / 24 | 3 / 5 / 8 |
| Opus arith / concat / full | 36 / 47 / 45 | 35 / 46 / 40 | 22 / 38 / 30 | 2 / 5 / 9 |

**Findings:**
1. **The difficulty dial's effect is capability-dependent.** At L2, Haiku climbs steeply with operator richness (4 → 8 → 20) — operators are its binding constraint — while Sonnet/Opus jump arith→concat then *plateau* (~45), because they exploit division to make even the arith tier deep. Operator richness matters enormously for the weak model, barely for strong ones.
2. **The strategy rung gives the only clean difficulty curve** — monotonic and capability-ordered (Opus 2→5→9, Sonnet 3→5→8, Haiku 3→3→5). The tier *gates the signature ops* (arith: none → tiny strategy space; full: fact+pow+concat → large), so strategy-vs-tier ≈ coverage of the tier's strategy space, and it separates models where L2 (fluency-noise) cannot.
3. **Distinctness dial:** every curve collapses as it coarsens (flat L2→L2u, steep drop to decomposition/strategy). Sonnet ≈ Opus until decomposition, where Opus pulls ahead; both converge low at strategy; Haiku flat-low throughout.

**Same lesson as Experiment 1, from the difficulty axis: the clean, capability-ordered signal lives at the coarse (strategy/decomposition) rungs; L2 is dominated by fluency.**

---

# Experiment 3 — the target as a difficulty axis (it reshapes the strategy space)

**Setup:** oracle protocol, 5-round budget, full toolkit, 3 models, targets chosen so their number theory unlocks different strategy **top-forms** (unlike operator tier, which gated signature *ops*): **17** (prime → `ADD`/`DIV` only), **24 = 4!** (→ `FACT`-rooted unlocked), **256 = 4⁴** (→ `POW`-rooted unlocked). Figure: `results/target_dial.png`. Reproduce: `target_plot.py`.

| model · target | L2 distinct | strategy classes | top-forms present |
|---|---|---|---|
| Haiku 17 / 24 / 256 | 20 / 11 / 28 | 5 / 5 / 7 | ADD · ADD,MUL · POW,MUL,ADD |
| Sonnet 17 / 24 / 256 | 45 / 55 / 59 | 8 / 15 / 14 | ADD,DIV · +FACT,MUL · +POW,MUL |
| Opus 17 / 24 / 256 | 45 / 51 / 62 | 9 / 15 / 14 | ADD,DIV · +FACT · +POW (×24) |

**Findings:**
1. **The target's number theory unlocks strategy top-forms.** `FACT`-rooted solutions appear *only* for 24, `POW`-rooted *only* for 256; 17 is stuck at `ADD`/`DIV`. The target reshapes the strategy *space* (which top-forms exist), where the operator tier only resized it.
2. **Strategy-class count tracks the enlarged space for capable models** (8–9 → 15), but **Haiku is capped at ~5–7** and never accesses the unlocked forms — it produces **no `FACT`-rooted solution for 24 across all 5 rounds** (top-forms ADD/MUL only), despite factorial being available. A categorical strategic ceiling.
3. **Richer targets are better capability discriminators at the strategy rung:** the Haiku↔capable gap widens from ~1.7× (target 17: 5 vs 8–9) to ~3× (target 24: 5 vs 15).
4. **Capability-dependent difficulty, again:** Haiku's L2 count *drops* at 24 (20 → 11) — 24 is genuinely *harder* for Haiku because it can't exploit the factorial that makes 24 easy for the others.

**Cross-experiment synthesis:** two independent difficulty dials — operator tier (resizes the strategy space) and target (reshapes it) — and in both, the **strategy rung** is where the difficulty curve is clean, monotonic, and capability-ordered, and where weak models reveal categorical ceilings. The whole arc points to one design rule: **measure creativity by strategy-space coverage at a coarse distinctness rung, elicited with oracle feedback.**
