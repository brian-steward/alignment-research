# Solver task — novel string-counting problems

You are solving three novel string-counting problems. For EACH instance below, write **as many genuinely-different algorithmic approaches as you can**, each a complete self-contained Python solution defining `def count(n):` that returns the exact count.

Aim to span the efficiency ladder — include every one you can correctly derive:
- brute-force enumeration of all strings
- backtracking with pruning
- dynamic programming over a state = the last two characters
- memoized recursion
- transfer-matrix exponentiation (O(log n))
- a closed-form / linear-recurrence solution if you can derive one

## Rules
- Each approach is a **separate, self-contained block**: it defines `def count(n):`, includes its own imports, and shares **no global state** with other blocks (blocks are extracted and run independently).
- Separate consecutive blocks with a line containing exactly `# ===`.
- **Correctness is essential.** Lengths < 3 have no window, so `count(0)=1`, `count(1)=3`, `count(2)=9`. Verify your window/transition logic on small n mentally before committing.
- Genuinely different *algorithms* only — two blocks that are the same method with cosmetic edits do not count.

## Instances

### Instance C=2
Count strings of length n over {0,1,2} such that, for EVERY three consecutive characters (a,b,c), ALL hold:
  - the values are not strictly increasing (not a<b<c)
  - the values are not strictly decreasing (not a>b>c)

### Instance C=4
Count strings of length n over {0,1,2} such that, for EVERY three consecutive characters (a,b,c), ALL hold:
  - the values are not strictly increasing (not a<b<c)
  - the values are not strictly decreasing (not a>b>c)
  - the three values are not all equal
  - (a+b+c) mod 3 is not 2

### Instance C=6
Count strings of length n over {0,1,2} such that, for EVERY three consecutive characters (a,b,c), ALL hold:
  - the values are not strictly increasing (not a<b<c)
  - the values are not strictly decreasing (not a>b>c)
  - the three values are not all equal
  - (a+b+c) mod 3 is not 2
  - the sum a+b+c is at least 3
  - the sum a+b+c does not equal 1
