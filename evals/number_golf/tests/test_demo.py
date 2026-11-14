from demo_transition import transition_counts


def test_transition_counts_monotone_in_tier():
    # Richer operator tiers should not have fewer ground-truth solutions.
    counts = transition_counts(target=24, max_leaves=4)
    assert counts["arith"] <= counts["arith_concat"] <= counts["full"]
