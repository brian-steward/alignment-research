from fibonacci import FIB, REFERENCE
from sandbox import is_valid, probe_opcounts
from fingerprint import strategy_key, complexity_class, l1_key

EXPECTED_COMPLEXITY = {
    "naive": "exponential",
    "iterative": "linear",
    "memoized": "linear",
    "matrix": "log",
    "fast_doubling": "log",
    "binet": "const",
}


def test_all_reference_methods_are_valid():
    for name, src in REFERENCE.items():
        assert is_valid(src, FIB) is True, name


def test_reference_complexity_classes():
    for name, src in REFERENCE.items():
        cc = complexity_class(probe_opcounts(src, FIB))
        assert cc == EXPECTED_COMPLEXITY[name], (name, cc)


def test_six_distinct_strategy_classes():
    keys = {}
    for name, src in REFERENCE.items():
        keys[name] = strategy_key(src, FIB, probe_opcounts(src, FIB))
    assert len(set(keys.values())) == 6, keys


def test_cosmetic_variant_collapses_at_strategy_but_not_l1():
    variant = ("def fib(n):\n    x, y = 0, 1\n    for _q in range(n):\n"
               "        x, y = y, x + y\n    return x\n")
    base = REFERENCE["iterative"]
    assert strategy_key(variant, FIB, probe_opcounts(variant, FIB)) == \
           strategy_key(base, FIB, probe_opcounts(base, FIB))
    assert l1_key(variant) == l1_key(base)
