from fibonacci import FIB, REFERENCE
from sandbox import is_valid, probe_opcounts


def test_valid_solution_passes():
    assert is_valid(REFERENCE["iterative"], FIB) is True


def test_wrong_solution_fails():
    assert is_valid("def fib(n):\n    return 123\n", FIB) is False


def test_naive_is_valid_on_small_n():
    assert is_valid(REFERENCE["naive"], FIB) is True


def test_non_function_fails():
    assert is_valid("x = 1\n", FIB) is False


def test_probe_returns_counts_for_iterative():
    counts = probe_opcounts(REFERENCE["iterative"], FIB)
    assert set(counts) == set(FIB.probe_sizes)
    assert all(counts[s] is not None for s in FIB.probe_sizes)
    assert counts[80] > counts[10]


def test_probe_times_out_for_naive_at_large_n():
    counts = probe_opcounts(REFERENCE["naive"], FIB, timeout=2.0)
    assert counts[80] is None


def test_probe_skips_larger_sizes_after_a_timeout():
    counts = probe_opcounts(REFERENCE["naive"], FIB, timeout=2.0)
    first = min(s for s in FIB.probe_sizes if counts[s] is None)
    assert all(counts[s] is None for s in FIB.probe_sizes if s >= first)


def test_probe_leaves_out_a_size_at_which_the_solution_raises():
    src = "def fib(n):\n    if n >= 160:\n        raise RecursionError\n    a, b = 0, 1\n" \
          "    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
    counts = probe_opcounts(src, FIB)
    assert set(counts) == {10, 20, 40, 80}
    assert all(v is not None for v in counts.values())
