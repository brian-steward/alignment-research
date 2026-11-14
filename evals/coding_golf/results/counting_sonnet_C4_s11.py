# Instance C=4
# Count strings of length n over {0,1,2} such that for every
# three consecutive characters (a,b,c), ALL hold:
#   - not strictly increasing (not a<b<c)
#   - not strictly decreasing (not a>b>c)
#   - not all equal (not a=b=c)
#   - (a+b+c) mod 3 is not 2

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
            if (a < b < c) or (a > b > c) or (a == b == c) or (a + b + c) % 3 == 2:
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
            if (a < b < c) or (a > b > c) or (a == b == c) or (a + b + c) % 3 == 2:
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
    dp = {(a, b): 1 for a in range(3) for b in range(3)}
    for _ in range(n - 2):
        new_dp = {(a, b): 0 for a in range(3) for b in range(3)}
        for (a, b), cnt in dp.items():
            for c in range(3):
                if not ((a < b < c) or (a > b > c) or (a == b == c) or (a + b + c) % 3 == 2):
                    new_dp[(b, c)] += cnt
        dp = new_dp
    return sum(dp.values())

# ===

def count(n):
    """Memoized recursion."""
    from functools import lru_cache

    if n == 0:
        return 1
    if n == 1:
        return 3

    @lru_cache(maxsize=None)
    def ways(remaining, a, b):
        if remaining == 0:
            return 1
        total = 0
        for c in range(3):
            if not ((a < b < c) or (a > b > c) or (a == b == c) or (a + b + c) % 3 == 2):
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
    if n == 0:
        return 1
    if n == 1:
        return 3
    if n == 2:
        return 9

    # States are pairs (a, b) with a,b in {0,1,2}
    states = [(a, b) for a in range(3) for b in range(3)]
    idx = {s: i for i, s in enumerate(states)}

    size = len(states)
    T = [[0] * size for _ in range(size)]
    for i, (a, b) in enumerate(states):
        for c in range(3):
            if not ((a < b < c) or (a > b > c) or (a == b == c) or (a + b + c) % 3 == 2):
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

    init = [1] * size
    Tn = mat_pow(T, n - 2, size)
    result = 0
    for j in range(size):
        for i in range(size):
            result += Tn[j][i] * init[i]
    return result

# ===

def count(n):
    """Closed-form via linear recurrence found by Berlekamp-Massey on computed terms."""
    from fractions import Fraction

    def is_forbidden(a, b, c):
        return (a < b < c) or (a > b > c) or (a == b == c) or (a + b + c) % 3 == 2

    def dp_count(m):
        if m == 0:
            return 1
        if m == 1:
            return 3
        dp = {(a, b): 1 for a in range(3) for b in range(3)}
        for _ in range(m - 2):
            new_dp = {(a, b): 0 for a in range(3) for b in range(3)}
            for (a, b), cnt in dp.items():
                for c in range(3):
                    if not is_forbidden(a, b, c):
                        new_dp[(b, c)] += cnt
            dp = new_dp
        return sum(dp.values())

    terms = [dp_count(i) for i in range(22)]

    def berlekamp_massey(s):
        n = len(s)
        s = [Fraction(x) for x in s]
        C = [Fraction(1)]
        B = [Fraction(1)]
        L = 0
        x = 1
        b = Fraction(1)
        for i in range(n):
            d = sum(C[j] * s[i - j] for j in range(len(C)))
            if d == 0:
                x += 1
            elif 2 * L <= i:
                T = C[:]
                coef = d / b
                C = C + [Fraction(0)] * (x + len(B) - len(C))
                for j in range(len(B)):
                    C[j + x] -= coef * B[j]
                L = i + 1 - L
                B = T
                b = d
                x = 1
            else:
                coef = d / b
                pad_len = x + len(B)
                while len(C) < pad_len:
                    C.append(Fraction(0))
                for j in range(len(B)):
                    C[j + x] -= coef * B[j]
                x += 1
        return [int(c) for c in C]

    C = berlekamp_massey(terms)
    order = len(C) - 1

    if n < len(terms):
        return terms[n]

    seq = list(terms[:order])
    while len(seq) <= n:
        nxt = -sum(C[j] * seq[-j] for j in range(1, order + 1))
        seq.append(int(nxt))
    return seq[n]
