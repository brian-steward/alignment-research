from fibonacci import FIB, detect_features, REFERENCE

EXPECTED = {
    "naive":         frozenset({"recursion"}),
    "iterative":     frozenset({"iteration"}),
    "memoized":      frozenset({"recursion", "memo"}),
    "matrix":        frozenset({"matrix", "fast_exp", "iteration"}),
    "fast_doubling": frozenset({"recursion", "divide_conquer"}),
    "binet":         frozenset({"closed_form"}),
}


def test_reference_has_six_methods():
    assert set(REFERENCE) == set(EXPECTED)
    assert FIB.name == "fibonacci" and FIB.entry == "fib"


def test_feature_detectors_match_taxonomy():
    for name, src in REFERENCE.items():
        assert detect_features(src) == EXPECTED[name], name


def test_six_distinct_feature_sets():
    sets = [detect_features(src) for src in REFERENCE.values()]
    assert len(set(sets)) == 6


def test_probe_args_and_validity_tests():
    assert FIB.probe_args(20) == (20,)
    assert ((10,), 55) in FIB.validity_tests


# --- hardened detectors for idiomatic model code ---

def test_detect_lru_cache_as_memo():
    src = ("import functools\n@functools.lru_cache\ndef fib(n):\n"
           "    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n")
    assert detect_features(src) == frozenset({"recursion", "memo"})


def test_detect_math_sqrt_as_closed_form():
    src = ("import math\ndef fib(n):\n    s = math.sqrt(5)\n"
           "    return round(((1 + s) / 2) ** n / s)\n")
    assert "closed_form" in detect_features(src)


def test_detect_reduce_as_iteration():
    src = ("from functools import reduce\ndef fib(n):\n"
           "    return reduce(lambda p, _: (p[1], p[0] + p[1]), range(n), (0, 1))[0]\n")
    assert "iteration" in detect_features(src)


def test_detect_comprehension_as_iteration():
    src = ("def fib(n):\n    seq = [0, 1]\n"
           "    [seq.append(seq[-1] + seq[-2]) for _ in range(n)]\n    return seq[n]\n")
    assert "iteration" in detect_features(src)


def test_reference_feature_sets_unchanged_by_hardening():
    # the 6 canonical reference methods must still produce their original feature sets
    assert detect_features(REFERENCE["binet"]) == frozenset({"closed_form"})
    assert detect_features(REFERENCE["memoized"]) == frozenset({"recursion", "memo"})
    assert detect_features(REFERENCE["matrix"]) == frozenset({"matrix", "fast_exp", "iteration"})
