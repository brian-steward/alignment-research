# 1. Naive recursion
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
# ===
# 2. Iteration
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
# ===
# 3. Memoization (top-down DP)
def fib(n):
    memo = {}
    def _fib(k):
        if k <= 1:
            return k
        if k not in memo:
            memo[k] = _fib(k - 1) + _fib(k - 2)
        return memo[k]
    return _fib(n)
# ===
# 4. Matrix exponentiation
def fib(n):
    def mat_mul(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
        ]
    def mat_pow(M, p):
        if p == 1:
            return M
        if p % 2 == 0:
            half = mat_pow(M, p // 2)
            return mat_mul(half, half)
        return mat_mul(M, mat_pow(M, p - 1))
    if n == 0:
        return 0
    result = mat_pow([[1, 1], [1, 0]], n)
    return result[0][1]
# ===
# 5. Fast doubling
def fib(n):
    def _fast_double(k):
        if k == 0:
            return (0, 1)
        a, b = _fast_double(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)
    return _fast_double(n)[0]
# ===
# 6. Closed-form (Binet's formula)
def fib(n):
    sqrt5 = 5 ** 0.5
    phi = (1 + sqrt5) / 2
    return int(round(phi ** n / sqrt5))
# ===
# 7. Generator-based
def fib(n):
    def fib_gen():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    gen = fib_gen()
    val = 0
    for _ in range(n + 1):
        val = next(gen)
    return val
# ===
# 8. functools.reduce
def fib(n):
    from functools import reduce
    if n == 0:
        return 0
    return reduce(lambda ab, _: (ab[1], ab[0] + ab[1]), range(n - 1), (0, 1))[1]
# ===
# 9. Bottom-up DP with full table
def fib(n):
    if n == 0:
        return 0
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
# ===
# 10. Digit-by-digit via integer-only golden ratio (successive squaring in Z[phi])
def fib(n):
    # Represent a + b*phi as (a, b); phi^2 = phi + 1
    def zphi_mul(x, y):
        a, b = x
        c, d = y
        # (a + b*phi)(c + d*phi) = ac + bd*phi^2 + (ad+bc)*phi
        #                        = ac + bd*(phi+1) + (ad+bc)*phi
        #                        = (ac+bd) + (ad+bc+bd)*phi
        return (a*c + b*d, a*d + b*c + b*d)
    def zphi_pow(base, p):
        result = (1, 0)  # multiplicative identity: 1 + 0*phi
        while p:
            if p & 1:
                result = zphi_mul(result, base)
            base = zphi_mul(base, base)
            p >>= 1
        return result
    if n == 0:
        return 0
    # phi^n = a + b*phi => fib(n) = b
    _, b = zphi_pow((0, 1), n)
    return b
