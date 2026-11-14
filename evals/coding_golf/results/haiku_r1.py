def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

# ===
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# ===
def fib(n):
    cache = {}
    def helper(n):
        if n in cache:
            return cache[n]
        if n <= 1:
            return n
        cache[n] = helper(n - 1) + helper(n - 2)
        return cache[n]
    return helper(n)

# ===
def fib(n):
    def matrix_mult(a, b):
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
        ]
    def matrix_pow(m, p):
        if p == 1:
            return m
        if p % 2 == 0:
            half = matrix_pow(m, p // 2)
            return matrix_mult(half, half)
        return matrix_mult(m, matrix_pow(m, p - 1))
    if n == 0:
        return 0
    result = matrix_pow([[1, 1], [1, 0]], n)
    return result[0][1]

# ===
def fib(n):
    def fast_double(k):
        if k == 0:
            return (0, 1)
        m = k // 2
        a, b = fast_double(m)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k % 2 == 0:
            return (c, d)
        return (d, c + d)
    return fast_double(n)[0]

# ===
def fib(n):
    import math
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return round((phi ** n - psi ** n) / math.sqrt(5))

# ===
def fib(n):
    def fib_gen():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b
    gen = fib_gen()
    for _ in range(n + 1):
        result = next(gen)
    return result

# ===
def fib(n):
    from functools import reduce
    if n <= 1:
        return n
    def fib_step(pair, _):
        a, b = pair
        return (b, a + b)
    initial = (0, 1)
    final_pair = reduce(fib_step, range(n - 1), initial)
    return final_pair[1]

# ===
def fib(n):
    memo = [0, 1]
    for i in range(2, n + 1):
        memo.append(memo[-1] + memo[-2])
    return memo[n]

# ===
def fib(n):
    def solve(n, memo=None):
        if memo is None:
            memo = {}
        if n in memo:
            return memo[n]
        if n == 0:
            return 0
        if n == 1:
            return 1
        memo[n] = solve(n - 1, memo) + solve(n - 2, memo)
        return memo[n]
    return solve(n)
