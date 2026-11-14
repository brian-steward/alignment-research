import math

from fingerprint import l1_key, complexity_class, strategy_key
from fibonacci import FIB, REFERENCE


def test_l1_collapses_variable_renames():
    a = "def fib(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
    b = "def fib(m):\n    acc = 0\n    for j in range(m):\n        acc += j\n    return acc\n"
    assert l1_key(a) == l1_key(b)


def test_l1_separates_different_algorithms():
    assert l1_key(REFERENCE["naive"]) != l1_key(REFERENCE["iterative"])


def test_complexity_class_on_synthetic_counts():
    sizes = [10, 20, 40, 80]
    assert complexity_class({s: 7 for s in sizes}) == "const"
    assert complexity_class({s: 5 * s for s in sizes}) == "linear"
    assert complexity_class({s: 3 * s * s for s in sizes}) == "quadratic"
    assert complexity_class({s: int(40 * math.log2(s)) for s in sizes}) == "log"
    assert complexity_class({10: 100, 20: 5000, 40: None, 80: None}) == "exponential"


SIZES = [10, 20, 40, 80, 160, 320, 640]


def test_fixed_setup_cost_does_not_hide_the_growth():
    """The bug fixed on 2026-11-14: with a large setup cost the counts barely move, and all of these were const."""
    setup = 50_000
    assert complexity_class({s: setup + 5 * s for s in SIZES}) == "linear"
    assert complexity_class({s: setup + int(40 * math.log2(s)) for s in SIZES}) == "log"
    assert complexity_class({s: setup + 3 * s * s for s in SIZES}) == "quadratic"
    assert complexity_class({s: setup for s in SIZES}) == "const"


def test_measured_cases_from_the_june_solutions():
    # recurrence found from computed terms, then applied term by term (was const)
    assert complexity_class(dict(zip(SIZES, [13310, 13532, 13972, 14852, 16612, 20132, 27172]))) == "linear"
    # the same, with the first terms answered from a table: flat at first, then linear (was const)
    assert complexity_class(dict(zip(SIZES, [63363, 63363, 63815, 64415, 65615, 68015, 72815]))) == "linear"
    # matrix power behind a large setup (was const)
    assert complexity_class(dict(zip(SIZES, [293862, 297113, 300580, 304047, 307514, 310981, 314448]))) == "log"
    # matrix power inside NumPy: five more lines per doubling (was const)
    assert complexity_class(dict(zip(SIZES, [282, 287, 292, 297, 302, 307, 312]))) == "log"


def test_timeouts():
    # the count stays flat and then the run is too slow: the work is somewhere the line count cannot see
    assert complexity_class({10: 900, 20: 900, 40: 900, 80: 900, 160: None, 320: None, 640: None}) == "exponential"
    # too few sizes before the timeout to read a law
    assert complexity_class({10: 5557, 20: 219447, 40: None, 80: None}) == "exponential"
    # a heavy but plainly linear solution that only runs out of time at the largest size keeps its class
    assert complexity_class({s: (2000 * s if s < 640 else None) for s in SIZES}) == "linear"


def test_a_size_that_was_not_measured_is_ignored():
    counts = {s: 7 * s for s in SIZES if s != 640}   # the solution raised at 640, so probe_opcounts left it out
    assert complexity_class(counts) == "linear"


def test_too_little_to_tell():
    assert complexity_class({10: 100, 20: 200}) == "unknown"
    assert complexity_class({10: 100}) == "unknown"


def test_strategy_key_combines_complexity_and_features():
    counts = {s: 5 * s for s in FIB.probe_sizes}      # pretend linear
    k = strategy_key(REFERENCE["iterative"], FIB, counts)
    assert k == ("linear", frozenset({"iteration"}))
