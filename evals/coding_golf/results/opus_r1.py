# Naive exponential recursion: fib(n) = fib(n-1) + fib(n-2)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
# ===
# Bottom-up iteration with a rolling pair of variables: O(n) time, O(1) space
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
# ===
# Top-down recursion with memoization via functools.lru_cache
import functools

@functools.lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
# ===
# Matrix exponentiation by squaring: [[1,1],[1,0]]^n gives Fibonacci in O(log n)
def fib(n):
    def mat_mul(A, B):
        # A, B are flat row-major 2x2 tuples (a, b, c, d); return same shape
        return (
            A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
            A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
        )

    result = (1, 0, 0, 1)  # identity matrix
    base = (1, 1, 1, 0)
    while n:
        if n & 1:
            result = mat_mul(result, base)
        base = mat_mul(base, base)
        n >>= 1
    return result[1]
# ===
# Fast doubling: F(2k)=F(k)*(2F(k+1)-F(k)), F(2k+1)=F(k)^2+F(k+1)^2, O(log n)
def fib(n):
    def _fd(k):
        if k == 0:
            return (0, 1)
        a, b = _fd(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)

    return _fd(n)[0]
# ===
# Closed-form Binet's formula using the golden ratio (exact via integer rounding)
import math

def fib(n):
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return round((phi ** n - psi ** n) / math.sqrt(5))
# ===
# Infinite generator yielding the Fibonacci sequence; consume the (n+1)th term
import itertools

def fib(n):
    def _gen():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b

    return next(itertools.islice(_gen(), n, None))
# ===
# functools.reduce folding n steps over the (a, b) state pair
import functools

def fib(n):
    return functools.reduce(lambda pair, _: (pair[1], pair[0] + pair[1]),
                            range(n), (0, 1))[0]
# ===
# Exact closed form via Decimal high-precision golden ratio (no float rounding error)
import decimal

def fib(n):
    decimal.getcontext().prec = 50
    sqrt5 = decimal.Decimal(5).sqrt()
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return int((phi ** n - psi ** n) / sqrt5 + decimal.Decimal("0.5"))
# ===
# Lucas-sequence identity: track the pair (F(k), L(k)) and double via
# F(2k)=F(k)*L(k), L(2k)=L(k)^2 - 2*(-1)^k, then step with averaging. O(log n)
def fib(n):
    def fl(k):
        if k == 0:
            return (0, 2)  # F(0)=0, L(0)=2
        f, l = fl(k >> 1)
        sign = 1 if (k >> 1) % 2 == 0 else -1
        f2 = f * l                 # F(2m)
        l2 = l * l - 2 * sign      # L(2m)
        if k & 1:
            # Step up by one: F(2m+1)=(F(2m)+L(2m))/2, L(2m+1)=(5*F(2m)+L(2m))/2
            return ((f2 + l2) // 2, (5 * f2 + l2) // 2)
        return (f2, l2)

    return fl(n)[0]
