# Instance C=2
# Count strings of length n over {0,1,2} such that for every
# three consecutive characters (a,b,c):
#   - not strictly increasing (not a<b<c)
#   - not strictly decreasing (not a>b>c)

def count(n):
    """Brute-force enumeration."""
    from itertools import product
    if n < 3:
        return 3 ** n
    total = 0
    for s in product(range(3), repeat=n):
        ok = True
        for i in range(n - 2):
            a, b, c = s[i], s[i+1], s[i+2]
            if (a < b < c) or (a > b > c):
                ok = False
                break
        if ok:
            total += 1
    return total

# ===

def count(n):
    """Backtracking with pruning."""
    if n == 0:
        return 1
    if n == 1:
        return 3
    if n == 2:
        return 9

    result = [0]

    def backtrack(pos, last_two):
        if pos == n:
            result[0] += 1
            return
        a, b = last_two
        for c in range(3):
            if (a < b < c) or (a > b > c):
                continue
            backtrack(pos + 1, (b, c))

    for a in range(3):
        for b in range(3):
            backtrack(2, (a, b))

    return result[0]

# ===

def count(n):
    """Dynamic programming over state = last two characters."""
    if n == 0:
        return 1
    if n == 1:
        return 3
    # dp[(a,b)] = number of valid strings ending in ...a,b
    dp = {}
    for a in range(3):
        for b in range(3):
            dp[(a, b)] = 1
    for _ in range(n - 2):
        new_dp = {}
        for a in range(3):
            for b in range(3):
                new_dp[(a, b)] = 0
        for (a, b), cnt in dp.items():
            for c in range(3):
                if not ((a < b < c) or (a > b > c)):
                    new_dp[(b, c)] = new_dp.get((b, c), 0) + cnt
        dp = new_dp
    return sum(dp.values())

# ===

def count(n):
    """Memoized recursion: f(pos, a, b) = ways to extend from position pos with last two chars a,b."""
    from functools import lru_cache

    if n == 0:
        return 1
    if n == 1:
        return 3

    @lru_cache(maxsize=None)
    def ways(remaining, a, b):
        """Number of valid extensions of length `remaining` given last two chars are a,b."""
        if remaining == 0:
            return 1
        total = 0
        for c in range(3):
            if not ((a < b < c) or (a > b > c)):
                total += ways(remaining - 1, b, c)
        return total

    result = 0
    for a in range(3):
        for b in range(3):
            result += ways(n - 2, a, b)
    return result

# ===

def count(n):
    """Transfer-matrix exponentiation O(log n)."""
    import numpy as np

    if n == 0:
        return 1
    if n == 1:
        return 3
    if n == 2:
        return 9

    # States are pairs (a, b) with a,b in {0,1,2}, so 9 states
    states = [(a, b) for a in range(3) for b in range(3)]
    idx = {s: i for i, s in enumerate(states)}

    # Build transition matrix T where T[j][i] = 1 if state i -> state j is valid
    size = len(states)
    T = [[0] * size for _ in range(size)]
    for i, (a, b) in enumerate(states):
        for c in range(3):
            if not ((a < b < c) or (a > b > c)):
                j = idx[(b, c)]
                T[j][i] += 1

    def mat_mul(A, B, s):
        C = [[0] * s for _ in range(s)]
        for r in range(s):
            for k in range(s):
                if A[r][k] == 0:
                    continue
                for col in range(s):
                    C[r][col] += A[r][k] * B[k][col]
        return C

    def mat_pow(M, p, s):
        result = [[1 if r == c else 0 for c in range(s)] for r in range(s)]
        base = [row[:] for row in M]
        while p > 0:
            if p % 2 == 1:
                result = mat_mul(result, base, s)
            base = mat_mul(base, base, s)
            p //= 2
        return result

    # Initial vector: all 9 states have count 1 (all 2-char prefixes)
    init = [1] * size
    # Apply T^(n-2) times
    Tn = mat_pow(T, n - 2, size)
    result = 0
    for j in range(size):
        for i in range(size):
            result += Tn[j][i] * init[i]
    return result

# ===

def count(n):
    """Closed-form via linear recurrence.

    Let f(n) = count(n). We compute f(0)=1, f(1)=3, f(2)=9, then find
    the recurrence by computing enough terms and using the characteristic
    polynomial of the transfer matrix.

    The transfer matrix has 9 states but many are symmetric. We find
    the minimal polynomial via the Berlekamp-Massey-style approach on
    computed terms.
    """
    # Compute first several terms by DP to find recurrence
    def dp_count(m):
        if m == 0:
            return 1
        if m == 1:
            return 3
        dp = {(a, b): 1 for a in range(3) for b in range(3)}
        for _ in range(m - 2):
            new_dp = {}
            for (a, b), cnt in dp.items():
                for c in range(3):
                    if not ((a < b < c) or (a > b > c)):
                        key = (b, c)
                        new_dp[key] = new_dp.get(key, 0) + cnt
            dp = new_dp
        return sum(dp.values())

    # The transfer matrix is 9x9; characteristic polynomial has degree 9.
    # But by Cayley-Hamilton the sequence satisfies a recurrence of order <= 9.
    # We find it: compute terms, then find minimal recurrence.
    terms = [dp_count(i) for i in range(20)]

    # Find recurrence via Berlekamp-Massey over integers
    def berlekamp_massey(s):
        """Find minimal linear recurrence for integer sequence s."""
        n = len(s)
        C = [1]
        B = [1]
        L = 0
        x = 1
        b = 1
        for i in range(n):
            d = s[i]
            for j in range(1, L + 1):
                d += C[j] * s[i - j]
            if d == 0:
                x += 1
            elif 2 * L <= i:
                T = C[:]
                coef = d * pow(b, -1) if False else None
                # Use fraction approach
                from fractions import Fraction
                coef = Fraction(d, b)
                pad = [0] * x
                Bpad = pad + [Fraction(bi) for bi in B]
                C = [Fraction(ci) for ci in C]
                while len(Bpad) > len(C):
                    C.append(Fraction(0))
                C = [C[j] - coef * Bpad[j] for j in range(len(C))]
                L = i + 1 - L
                B = [Fraction(ti) for ti in T]
                b = Fraction(d)
                x = 1
            else:
                from fractions import Fraction
                coef = Fraction(d, b)
                pad = [0] * x
                Bpad = pad + [Fraction(bi) for bi in B]
                C = [Fraction(ci) for ci in C]
                while len(Bpad) > len(C):
                    C.append(Fraction(0))
                C = [C[j] - coef * Bpad[j] for j in range(len(C))]
                x += 1
        # C[0]=1, recurrence: C[0]*s[i] + C[1]*s[i-1] + ... = 0
        # so s[i] = -C[1]*s[i-1] - C[2]*s[i-2] - ...
        return [int(c) for c in C]

    C = berlekamp_massey(terms)
    order = len(C) - 1  # recurrence order

    if n < len(terms):
        return terms[n]

    # Extend using recurrence
    seq = terms[:order]
    while len(seq) <= n:
        nxt = 0
        for j in range(1, order + 1):
            nxt -= C[j] * seq[-j]
        seq.append(nxt)
    return seq[n]
