# Instance C=2
# Strings of length n over {0,1,2}. For EVERY window (a,b,c):
#   - not strictly increasing (not a<b<c)
#   - not strictly decreasing (not a>b>c)
# Base cases: count(0)=1, count(1)=3, count(2)=9.

# Approach 1: brute-force enumeration of all strings
import itertools
def _ok(a, b, c):
    if a < b < c:
        return False
    if a > b > c:
        return False
    return True
def count(n):
    if n < 3:
        return 3 ** n
    total = 0
    for s in itertools.product(range(3), repeat=n):
        if all(_ok(s[i], s[i + 1], s[i + 2]) for i in range(n - 2)):
            total += 1
    return total

# ===

# Approach 2: backtracking with pruning (depth-first, prune on first bad window)
def count(n):
    def ok(a, b, c):
        return not (a < b < c) and not (a > b > c)
    if n == 0:
        return 1
    if n == 1:
        return 3
    cnt = 0
    stack = [(a, b) for a in range(3) for b in range(3)]
    # Represent partial as (prefix_len, prev2, prev1); start with length-2 prefixes.
    work = [(2, a, b) for a in range(3) for b in range(3)]
    if n == 2:
        return 9
    while work:
        length, p2, p1 = work.pop()
        for c in range(3):
            if ok(p2, p1, c):
                if length + 1 == n:
                    cnt += 1
                else:
                    work.append((length + 1, p1, c))
    return cnt

# ===

# Approach 3: dynamic programming over state = last two characters
def count(n):
    def ok(a, b, c):
        return not (a < b < c) and not (a > b > c)
    if n == 0:
        return 1
    if n == 1:
        return 3
    # dp[(a,b)] = number of valid strings of current length ending in ...a b
    dp = {(a, b): 1 for a in range(3) for b in range(3)}
    for _ in range(n - 2):
        nd = {}
        for (a, b), v in dp.items():
            for c in range(3):
                if ok(a, b, c):
                    nd[(b, c)] = nd.get((b, c), 0) + v
        dp = nd
    return sum(dp.values())

# ===

# Approach 4: memoized recursion on (position-remaining, last two chars)
from functools import lru_cache
def count(n):
    if n == 0:
        return 1
    if n == 1:
        return 3
    def ok(a, b, c):
        return not (a < b < c) and not (a > b > c)
    @lru_cache(maxsize=None)
    def rec(rem, a, b):
        # rem = number of characters still to append after the pair (a,b)
        if rem == 0:
            return 1
        tot = 0
        for c in range(3):
            if ok(a, b, c):
                tot += rec(rem - 1, b, c)
        return tot
    return sum(rec(n - 2, a, b) for a in range(3) for b in range(3))

# ===

# Approach 5: transfer-matrix exponentiation, O(log n) matrix powers
def count(n):
    if n == 0:
        return 1
    if n == 1:
        return 3
    def ok(a, b, c):
        return not (a < b < c) and not (a > b > c)
    states = [(a, b) for a in range(3) for b in range(3)]
    idx = {s: i for i, s in enumerate(states)}
    S = len(states)
    # Transfer matrix M[j][i] = 1 if state i=(a,b) -> j=(b,c) allowed
    M = [[0] * S for _ in range(S)]
    for (a, b) in states:
        for c in range(3):
            if ok(a, b, c):
                M[idx[(b, c)]][idx[(a, b)]] += 1
    def matmul(A, B):
        ra, ca, cb = len(A), len(B), len(B[0])
        C = [[0] * cb for _ in range(ra)]
        for i in range(ra):
            Ai = A[i]
            Ci = C[i]
            for k in range(ca):
                a = Ai[k]
                if a:
                    Bk = B[k]
                    for j in range(cb):
                        Ci[j] += a * Bk[j]
        return C
    def matpow(A, p):
        n_ = len(A)
        R = [[1 if i == j else 0 for j in range(n_)] for i in range(n_)]
        while p:
            if p & 1:
                R = matmul(R, A)
            A = matmul(A, A)
            p >>= 1
        return R
    Mp = matpow(M, n - 2)
    # initial vector: each length-2 state has count 1
    return sum(sum(row) for row in Mp)

# ===

# Approach 6: linear recurrence via numeric sequence + companion / direct fit.
# Empirically the per-length state-count vector follows a fixed linear recurrence.
# We derive it by generating enough exact terms from the transfer relation,
# then extending by the minimal recurrence found via the count ratios.
# Here we use the characteristic recurrence determined from the transfer matrix
# trace identities (Faddeev-LeVerrier on the 9x9 matrix), applied to a(n).
def count(n):
    if n == 0:
        return 1
    if n == 1:
        return 3
    def ok(a, b, c):
        return not (a < b < c) and not (a > b > c)
    states = [(a, b) for a in range(3) for b in range(3)]
    idx = {s: i for i, s in enumerate(states)}
    S = len(states)
    M = [[0] * S for _ in range(S)]
    for (a, b) in states:
        for c in range(3):
            if ok(a, b, c):
                M[idx[(b, c)]][idx[(a, b)]] += 1
    # Faddeev-LeVerrier to get characteristic polynomial coeffs of M (size S).
    def matmul(A, B):
        C = [[0] * S for _ in range(S)]
        for i in range(S):
            Ai = A[i]
            for k in range(S):
                a = Ai[k]
                if a:
                    Bk = B[k]
                    Ci = C[i]
                    for j in range(S):
                        Ci[j] += a * Bk[j]
        return C
    def trace(A):
        return sum(A[i][i] for i in range(S))
    I = [[1 if i == j else 0 for j in range(S)] for i in range(S)]
    Mk = [row[:] for row in I]
    c = [1] + [0] * S  # c[0]=1 leading coeff; c[k] are p coeffs
    p = [0.0] * (S + 1)
    p[0] = 1
    Mcur = [row[:] for row in I]
    cprev = 1
    coeffs = [1]
    Ak = [[0] * S for _ in range(S)]
    # standard FL iteration
    Mi = [row[:] for row in I]
    cs = [1]
    Mprod = [row[:] for row in I]
    for k in range(1, S + 1):
        Mprod = matmul(M, Mi)
        ck = -trace(Mprod) // k
        Mi = [[Mprod[i][j] + (ck if i == j else 0) for j in range(S)] for i in range(S)]
        cs.append(ck)
    # cs are integer char-poly coeffs: M^S + cs[1] M^(S-1) + ... = 0
    # Recurrence for a(m): a(m) = -(cs[1] a(m-1)+...+cs[S] a(m-S))
    # Seed with exact counts for the first S values of count() via direct DP.
    def direct(m):
        if m == 0:
            return 1
        if m == 1:
            return 3
        dp = {(a, b): 1 for a in range(3) for b in range(3)}
        for _ in range(m - 2):
            nd = {}
            for (a, b), v in dp.items():
                for cc in range(3):
                    if ok(a, b, cc):
                        nd[(b, cc)] = nd.get((b, cc), 0) + v
            dp = nd
        return sum(dp.values())
    seeds = [direct(m) for m in range(S + 2)]
    if n < len(seeds):
        return seeds[n]
    seq = seeds[:]
    for m in range(len(seq), n + 1):
        val = 0
        for k in range(1, S + 1):
            val -= cs[k] * seq[m - k]
        seq.append(val)
    return seq[n]
