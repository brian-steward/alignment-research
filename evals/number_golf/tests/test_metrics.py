import math
from metrics import (
    discovery_curve, fit_heaps, distinct_at, yield_rate, bootstrap_beta,
)


def test_discovery_curve_counts_distinct():
    keys = ["a", "a", "b", "a", "c"]
    n, V = discovery_curve(keys)
    assert n == [1, 2, 3, 4, 5]
    assert V == [1, 1, 2, 2, 3]


def test_fit_heaps_all_unique_beta_near_one():
    keys = [str(i) for i in range(200)]
    n, V = discovery_curve(keys)
    K, beta, r2 = fit_heaps(n, V)
    assert abs(beta - 1.0) < 0.05
    assert r2 > 0.99


def test_fit_heaps_all_same_beta_near_zero():
    keys = ["x"] * 200
    n, V = discovery_curve(keys)
    K, beta, r2 = fit_heaps(n, V)
    assert abs(beta - 0.0) < 1e-6


def test_distinct_at_budget():
    keys = ["a", "b", "a", "c", "d"]
    assert distinct_at(keys, 3) == 2
    assert distinct_at(keys, 100) == 4


def test_distinct_at_skips_none():
    # None marks an invalid attempt: it consumes budget but adds no distinct key.
    assert distinct_at(["a", None, "b", None, "a"], 5) == 2
    assert distinct_at([None, None, "a"], 2) == 0


def test_yield_rate():
    assert yield_rate(7, 10) == 0.7
    assert yield_rate(0, 0) == 0.0


def test_bootstrap_beta_valid_interval_and_orders_by_diversity():
    # A repetitive stream (only 30 distinct) must yield a strictly lower beta
    # band than an all-unique stream. (We don't bracket the order-dependent
    # point estimate, which for a front-loaded periodic stream is the maximum.)
    rep_lo, rep_hi = bootstrap_beta([str(i % 30) for i in range(300)], iters=200, seed=0)
    uniq_lo, uniq_hi = bootstrap_beta([str(i) for i in range(300)], iters=200, seed=0)
    for x in (rep_lo, rep_hi, uniq_lo, uniq_hi):
        assert math.isfinite(x)
    assert 0.0 <= rep_lo <= rep_hi
    assert rep_hi <= uniq_lo  # more diverse stream -> higher beta band
