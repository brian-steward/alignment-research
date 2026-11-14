# Coding-domain results

Two problems. A **famous** problem (Fibonacci) measures recall: all three models give the same known methods. A **novel** problem (constraint counting) removes the recall confound, but in this run it did not separate the models either: all three reach the same complexity class.

**Correction, 2026-11-14.** The June version of this page reported that Sonnet and Opus reached a constant-time closed form on the counting problems, that Haiku did not, and that this ordered the models by capability. That came from a bug in the complexity classifier. Section 2 gives the corrected result and says what was wrong. Section 1 is unchanged: the Fibonacci numbers are the same under the corrected classifier.

---

## 1. Fibonacci (famous problem) — recall saturates

After hardening the feature detectors (lru_cache memo, math.sqrt/Decimal closed-form, comprehension/reduce/yield iteration), **all three models reach 5/6 taxonomy coverage** (Haiku, Sonnet, Opus alike). A famous problem is a **recall test** — the ~6 methods are universally memorized, so there is no capability signal. This is the confound the second domain was built to remove.

---

## 2. Constraint-counting (novel problems) — no separation by complexity class

**Setup.** Count length-n ternary strings whose every length-3 window satisfies a randomly sampled set of `C` local constraints (sum / repetition / ordering / modular). The problem is new to the models, so there is no memorized solution, but it is a bounded-window regular language, so the method ladder (brute force, backtracking, DP, transfer matrix) exists. Instances: `C ∈ {2, 4, 6}` at seed 11. Elicitation: in-context enumeration (A2), where each model is asked once for as many different `count(n)` approaches as it can give. Scoring: a solution is valid if it matches brute-force counts at n = 0, 1, 2, 3, 4, 5, 8 and 10. Its strategy is `(complexity class, paradigm features)`. The complexity class is read from how the number of executed lines grows over n = 10, 20, 40, … 640.

**Result.**

| instance | Haiku | Sonnet | Opus |
|---|---|---|---|
| C=2 | valid 6, distinct 5, lowest log | valid 6, distinct 5, lowest log | valid 5, distinct 5, lowest log |
| C=4 | valid 6, distinct 5, lowest log | valid 6, distinct 5, lowest log | valid 6, distinct 4, lowest log |
| C=6 | valid 6, distinct 5, lowest log | valid 6, distinct 5, lowest log | valid 6, distinct 4, lowest log |

Every model's best solution is logarithmic (transfer-matrix exponentiation) at every `C`. No model produced a constant-time solution. Neither the lowest class reached nor the distinct count orders the models.

**What the models did.** All three give the same five methods: brute force, backtracking, DP over the last two characters, memoized recursion and transfer-matrix exponentiation. They differ in the sixth. Sonnet and Opus derive a scalar linear recurrence for the count (Sonnet by Berlekamp-Massey on computed terms, Opus from the characteristic polynomial of the transfer matrix) and then apply it term by term, which takes linear time. Haiku's sixth is a second DP loop. Deriving the recurrence is a step Haiku did not take, but the scoring does not register it: in five of the six cases the recurrence solution has the same class and features as the DP, so it is not counted as a distinct strategy. At C=4 Opus also gives the matrix power a second time, in NumPy, which counts as the same strategy as its first.

**What was wrong in June.** The classifier compared the line counts themselves at n = 10, 20, 40 and 80. The recurrence solutions spend 13,000 to 84,000 lines deriving the recurrence and then 10 to 22 lines per unit of n, so over that range their counts rose by 1% to 12% and they were labelled `const`. Sonnet's matrix solution at C=2 has a 294,000-line setup and was labelled `const` for the same reason. So was Opus's NumPy matrix power, whose loop the line counter sees as five lines per doubling. The June table therefore showed `lowest const` for Sonnet and Opus at every `C`, and distinct counts of 6, 6, 6 for Sonnet and 5, 6, 5 for Opus. The classifier now reads the growth from the increments between sizes, so a fixed setup cost cancels, and the sizes go up to 640 so that a solution which answers small n from a table has left it (`fingerprint.complexity_class`; `tests/test_fingerprint.py` holds these measured cases).

**Caveats.**
- **The feature detector over-fires for Haiku.** The `brute_force` feature appears on every Haiku solution, including its DP and transfer-matrix code. This affects the feature half of the strategy key, not the complexity class.
- **The line counter does not see work inside library calls.** A loop that runs inside NumPy, or arithmetic on very large integers, costs time but adds no lines. The class is a statement about executed Python lines.
- **Opus block 0 (brute force) is invalid at every `C`.** It costs Opus one valid solution per instance.
- **One number differs from the June table for another reason.** June gave Opus 5 valid solutions at C=6. A re-run gives 6 of 7, the same on three repeats.
- **The difficulty dial is flat.** The lowest class reached does not move with `C` for any model. With a 9-state automaton (window 3) the constraint count makes each instance new, but does not make the transfer matrix harder to find.
- **Small sample.** One seed and one elicitation pass per model.

**Conclusion.** On this problem all three models went equally far down the method ladder, so the eval as built does not order them. The one difference between them, deriving a recurrence, is not captured by the strategy fingerprint. Showing it would need either a feature for a derived recurrence, or a task in which applying the recurrence in a loop is too slow to count. The next steps listed in June still stand: oracle-guided elicitation, a second seed, and a larger window.
