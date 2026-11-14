from __future__ import annotations

import ast

from constraints import allowed_windows, count_bruteforce
from problem import Problem


# ---------- reference method sources (allowed-window table baked in) ----------

def _transition_matrix(aw):
    """9x9 transition over states idx=a*3+b -> b*3+c, 1 if (a,b,c) allowed."""
    M = [[0] * 9 for _ in range(9)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if (a, b, c) in aw:
                    M[a * 3 + b][b * 3 + c] = 1
    return M


def build_references(cs) -> dict:
    aw = sorted(allowed_windows(cs))
    aw_lit = "{" + ", ".join(repr(t) for t in aw) + "}"
    M = _transition_matrix(allowed_windows(cs))

    brute = (
        "from itertools import product\n"
        f"ALLOWED = {aw_lit}\n"
        "def count(n):\n"
        "    if n == 0:\n        return 1\n"
        "    total = 0\n"
        "    for s in product(range(3), repeat=n):\n"
        "        if all((s[i], s[i+1], s[i+2]) in ALLOWED for i in range(n-2)):\n"
        "            total += 1\n"
        "    return total\n"
    )

    backtrack = (
        f"ALLOWED = {aw_lit}\n"
        "def count(n):\n"
        "    if n == 0:\n        return 1\n"
        "    total = 0\n"
        "    def dfs(prefix):\n"
        "        nonlocal total\n"
        "        if len(prefix) == n:\n"
        "            total += 1\n"
        "            return\n"
        "        for c in range(3):\n"
        "            if len(prefix) >= 2 and (prefix[-2], prefix[-1], c) not in ALLOWED:\n"
        "                continue\n"
        "            prefix.append(c)\n"
        "            dfs(prefix)\n"
        "            prefix.pop()\n"
        "    dfs([])\n"
        "    return total\n"
    )

    dp_table = (
        f"ALLOWED = {aw_lit}\n"
        "def count(n):\n"
        "    if n == 0:\n        return 1\n"
        "    if n == 1:\n        return 3\n"
        "    dp = {}\n"
        "    for a in range(3):\n"
        "        for b in range(3):\n"
        "            dp[(a, b)] = 1\n"
        "    for _ in range(n - 2):\n"
        "        nd = {}\n"
        "        for (a, b), cnt in dp.items():\n"
        "            for c in range(3):\n"
        "                if (a, b, c) in ALLOWED:\n"
        "                    nd[(b, c)] = nd.get((b, c), 0) + cnt\n"
        "        dp = nd\n"
        "    return sum(dp.values())\n"
    )

    memoized = (
        "from functools import lru_cache\n"
        f"ALLOWED = {aw_lit}\n"
        "def count(n):\n"
        "    if n == 0:\n        return 1\n"
        "    if n == 1:\n        return 3\n"
        "    @lru_cache(maxsize=None)\n"
        "    def go(rem, a, b):\n"
        "        if rem == 0:\n            return 1\n"
        "        return sum(go(rem-1, b, c) for c in range(3) if (a, b, c) in ALLOWED)\n"
        "    return sum(go(n-2, a, b) for a in range(3) for b in range(3))\n"
    )

    matrix = (
        f"M = {M!r}\n"
        "def _mul(A, B):\n"
        "    m = len(A)\n"
        "    return [[sum(A[i][k]*B[k][j] for k in range(m)) for j in range(m)] for i in range(m)]\n"
        "def _matpow(A, p):\n"
        "    m = len(A)\n"
        "    R = [[1 if i == j else 0 for j in range(m)] for i in range(m)]\n"
        "    while p:\n"
        "        if p & 1:\n            R = _mul(R, A)\n"
        "        A = _mul(A, A)\n"
        "        p >>= 1\n"
        "    return R\n"
        "def count(n):\n"
        "    if n == 0:\n        return 1\n"
        "    if n == 1:\n        return 3\n"
        "    P = _matpow(M, n - 2)\n"
        "    return sum(sum(row) for row in P)\n"
    )

    return {"brute": brute, "backtrack": backtrack, "dp_table": dp_table,
            "memoized": memoized, "matrix": matrix}


# ---------- generic paradigm detectors ----------

def _self_calling(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id == node.name:
                    return True
    return False


def _has(tree, pred):
    return any(pred(n) for n in ast.walk(tree))


def _calls_named(tree, names):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in names:
                return True
            if isinstance(f, ast.Attribute) and f.attr in names:
                return True
    return False


def _has_decorator(tree, names):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            for d in n.decorator_list:
                t = d.func if isinstance(d, ast.Call) else d
                if isinstance(t, ast.Name) and t.id in names:
                    return True
                if isinstance(t, ast.Attribute) and t.attr in names:
                    return True
    return False


def _shift_or_halving(node):
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


def _pow_with_name_exponent(node):
    # k ** n  (enumerate-all signature), exponent is a variable
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) \
        and isinstance(node.right, ast.Name)


def _stores_to_subscript(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Subscript) for t in node.targets):
                return True
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
            return True
    return False


def detect_features(source: str) -> frozenset:
    tree = ast.parse(source)
    feats = set()

    recursion = _self_calling(tree)
    loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
    comps = _has(tree, lambda n: isinstance(n, (ast.ListComp, ast.SetComp,
                                                ast.DictComp, ast.GeneratorExp)))
    has_yield = _has(tree, lambda n: isinstance(n, (ast.Yield, ast.YieldFrom)))
    iter_calls = _calls_named(tree, {"map", "reduce", "accumulate", "starmap"})
    iteration = bool(loops or comps or has_yield or iter_calls)

    brute_force = _calls_named(tree, {"product", "permutations", "combinations",
                                      "combinations_with_replacement"}) \
        or _has(tree, _pow_with_name_exponent)

    matrix = _has(tree, lambda n: isinstance(n, ast.List)
                  and any(isinstance(e, ast.List) for e in n.elts)) \
        or _calls_named(tree, {"matrix_power"})

    memoization = recursion and (
        _has(tree, lambda n: isinstance(n, ast.Dict))
        or _has_decorator(tree, {"lru_cache", "cache"})
    )

    if recursion:
        feats.add("recursion")
    if iteration:
        feats.add("iteration")
    if brute_force:
        feats.add("brute_force")
    if matrix:
        feats.add("matrix")
    if memoization:
        feats.add("memoization")
    # fast_exp: exponentiation-by-squaring (shift/halving inside a loop)
    if any(any(_shift_or_halving(s) for s in ast.walk(lp)) for lp in loops):
        feats.add("fast_exp")
    # dp_table: tabulation into an indexed structure, not brute/recursive/matrix
    if _stores_to_subscript(tree) and not brute_force and not recursion and not matrix:
        feats.add("dp_table")
    # closed_form: float / sqrt / Decimal / eigenvalue
    if _has(tree, lambda n: isinstance(n, ast.Constant) and isinstance(n.value, float)) \
            or _calls_named(tree, {"sqrt", "eig", "eigvals"}) \
            or _has(tree, lambda n: isinstance(n, ast.Name) and n.id == "Decimal"):
        feats.add("closed_form")
    return frozenset(feats)


# ---------- Problem builder ----------

VALIDITY_NS = [0, 1, 2, 3, 4, 5, 8, 10]
# Doubling sizes, up to 640. With 80 as the largest (the setting until 2026-11-14) a solution that precomputes its
# first terms had not yet left its table, and its growth could not be read.
PROBE_SIZES = [10, 20, 40, 80, 160, 320, 640]


def make_problem(cs, name: str = "counting") -> Problem:
    return Problem(
        name=name,
        entry="count",
        validity_tests=[((n,), count_bruteforce(cs, n)) for n in VALIDITY_NS],
        probe_sizes=PROBE_SIZES,
        probe_args=lambda n: (n,),
        features=detect_features,
        reference_methods=build_references(cs),
    )
