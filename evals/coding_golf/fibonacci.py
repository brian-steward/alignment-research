from __future__ import annotations

import ast

from problem import Problem

# --- the 6 known methods (ground-truth taxonomy) ---
REFERENCE = {
    "naive": (
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    ),
    "iterative": (
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n):\n"
        "        a, b = b, a + b\n"
        "    return a\n"
    ),
    "memoized": (
        "def fib(n, _cache={0: 0, 1: 1}):\n"
        "    if n in _cache:\n"
        "        return _cache[n]\n"
        "    _cache[n] = fib(n - 1) + fib(n - 2)\n"
        "    return _cache[n]\n"
    ),
    "matrix": (
        "def fib(n):\n"
        "    def mul(A, B):\n"
        "        return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],\n"
        "                [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]\n"
        "    result = [[1, 0], [0, 1]]\n"
        "    base = [[1, 1], [1, 0]]\n"
        "    e = n\n"
        "    while e:\n"
        "        if e & 1:\n"
        "            result = mul(result, base)\n"
        "        base = mul(base, base)\n"
        "        e >>= 1\n"
        "    return result[0][1]\n"
    ),
    "fast_doubling": (
        "def fib(n):\n"
        "    def fd(k):\n"
        "        if k == 0:\n"
        "            return (0, 1)\n"
        "        a, b = fd(k >> 1)\n"
        "        c = a * (2 * b - a)\n"
        "        d = a * a + b * b\n"
        "        if k & 1:\n"
        "            return (d, c + d)\n"
        "        return (c, d)\n"
        "    return fd(n)[0]\n"
    ),
    "binet": (
        "def fib(n):\n"
        "    s = 5 ** 0.5\n"
        "    phi = (1 + s) / 2\n"
        "    psi = (1 - s) / 2\n"
        "    return round((phi ** n - psi ** n) / s)\n"
    ),
}


def _funcs_that_call_themselves(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id == node.name:
                    names.add(node.name)
    return names


def _has(tree, pred):
    return any(pred(n) for n in ast.walk(tree))


def _shift_or_halving(node):
    # `>> ...` / `>>= ...` or `... // 2` / `//= ...`
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return True
    if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.RShift):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv) \
            and isinstance(node.right, ast.Constant) and node.right.value == 2:
        return True
    if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.FloorDiv):
        return True
    return False


def _calls_named(tree, names):
    """True if the source calls any function whose Name id or Attribute attr is in names."""
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in names:
                return True
            if isinstance(f, ast.Attribute) and f.attr in names:
                return True
    return False


def _has_decorator(tree, names):
    """True if any function is decorated with a name in `names` (bare or attribute, called or not)."""
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                t = d.func if isinstance(d, ast.Call) else d
                if isinstance(t, ast.Name) and t.id in names:
                    return True
                if isinstance(t, ast.Attribute) and t.attr in names:
                    return True
    return False


def detect_features(source: str) -> frozenset:
    tree = ast.parse(source)
    feats = set()
    recursive = bool(_funcs_that_call_themselves(tree))
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
    comps = _has(tree, lambda n: isinstance(n, (ast.ListComp, ast.SetComp,
                                                ast.DictComp, ast.GeneratorExp)))
    has_yield = _has(tree, lambda n: isinstance(n, (ast.Yield, ast.YieldFrom)))
    iter_calls = _calls_named(tree, {"map", "reduce", "accumulate", "starmap"})
    has_shift = _has(tree, _shift_or_halving)
    cache_dec = _has_decorator(tree, {"lru_cache", "cache"})
    if recursive:
        feats.add("recursion")
    # iteration: explicit loops OR implicit iteration (comprehensions / generators / map/reduce)
    if loops or comps or has_yield or iter_calls:
        feats.add("iteration")
    # memo: a recursive function plus a cache (dict literal OR lru_cache/cache decorator)
    if recursive and (_has(tree, lambda n: isinstance(n, ast.Dict)) or cache_dec):
        feats.add("memo")
    # matrix: a list whose elements include a list (list-of-lists)
    if _has(tree, lambda n: isinstance(n, ast.List)
            and any(isinstance(e, ast.List) for e in n.elts)):
        feats.add("matrix")
    # fast_exp: a loop body containing a shift/halving (exponentiation by squaring)
    if any(any(_shift_or_halving(s) for s in ast.walk(lp)) for lp in loops):
        feats.add("fast_exp")
    # divide_conquer: recursion + shift/halving, with no loop
    if recursive and has_shift and not loops:
        feats.add("divide_conquer")
    # closed_form: float constant, a sqrt call (Name or math.sqrt attribute), or Decimal
    if _has(tree, lambda n: isinstance(n, ast.Constant) and isinstance(n.value, float)) \
            or _calls_named(tree, {"sqrt"}) \
            or _has(tree, lambda n: isinstance(n, ast.Name) and n.id == "Decimal") \
            or _has(tree, lambda n: isinstance(n, ast.Attribute) and n.attr == "Decimal"):
        feats.add("closed_form")
    return frozenset(feats)


FIB = Problem(
    name="fibonacci",
    entry="fib",
    validity_tests=[((n,), v) for n, v in [
        (0, 0), (1, 1), (2, 1), (5, 5), (10, 55), (15, 610),
    ]],
    probe_sizes=[10, 20, 40, 80, 160, 320, 640],   # doubling, so the growth class shows in the increments
    probe_args=lambda n: (n,),
    features=detect_features,
    reference_methods=REFERENCE,
)
