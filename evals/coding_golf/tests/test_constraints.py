import random
from constraints import (Constraint, sample_constraint, allowed_windows,
                         count_bruteforce, sample_problem, render_spec)


def test_sum_constraint_holds_and_renders():
    c = Constraint("sum", op="<=", t=4)
    assert c.holds(1, 1, 1) is True       # 3 <= 4
    assert c.holds(2, 2, 2) is False      # 6 <= 4 is False
    assert "4" in c.render() and "+" in c.render()


def test_repetition_constraint_three_in_a_row():
    c = Constraint("repeat3")
    assert c.holds(2, 2, 2) is False      # all equal forbidden
    assert c.holds(2, 2, 1) is True


def test_repetition_constraint_distance2():
    c = Constraint("repeat2")
    assert c.holds(1, 0, 1) is False      # a == c forbidden
    assert c.holds(1, 0, 2) is True


def test_ordering_constraint_increasing():
    c = Constraint("inc")
    assert c.holds(0, 1, 2) is False      # strictly increasing forbidden
    assert c.holds(0, 1, 1) is True


def test_modular_constraint():
    c = Constraint("mod", r=0)
    assert c.holds(1, 1, 1) is False      # sum 3, 3 % 3 == 0 forbidden
    assert c.holds(1, 1, 2) is True       # sum 4, 4 % 3 == 1


def test_sample_constraint_is_deterministic_under_seed():
    a = sample_constraint(random.Random(7))
    b = sample_constraint(random.Random(7))
    assert a.render() == b.render()
    assert isinstance(a.holds(0, 1, 2), bool)


def test_count_no_constraints_is_full_space():
    assert count_bruteforce([], 0) == 1
    assert count_bruteforce([], 1) == 3
    assert count_bruteforce([], 2) == 9
    assert count_bruteforce([], 3) == 27


def test_count_with_one_constraint_hand_checked():
    # forbid three-in-a-row equal: length-3 strings = 27 - 3 = 24
    cs = [Constraint("repeat3")]
    assert count_bruteforce(cs, 3) == 24
    # length < 3 has no window, so unaffected
    assert count_bruteforce(cs, 2) == 9


def test_allowed_windows_matches_constraint():
    cs = [Constraint("repeat3")]
    aw = allowed_windows(cs)
    assert (0, 0, 0) not in aw and (1, 1, 1) not in aw
    assert (0, 1, 2) in aw
    assert len(aw) == 24


def test_sample_problem_is_non_degenerate():
    cs = sample_problem(n_constraints=3, seed=11)
    assert len(cs) == 3
    counts = [count_bruteforce(cs, n) for n in range(1, 7)]
    assert all(x > 0 for x in counts)            # live language
    assert all(b > a for a, b in zip(counts, counts[1:]))   # strictly growing
    assert count_bruteforce(cs, 4) < 3 ** 4      # constraints actually bite


def test_sample_problem_deterministic():
    a = sample_problem(3, seed=11)
    b = sample_problem(3, seed=11)
    assert [c.render() for c in a] == [c.render() for c in b]


def test_render_spec_mentions_each_constraint():
    cs = [Constraint("sum", op="<=", t=4), Constraint("repeat3")]
    spec = render_spec(cs)
    assert "at most 4" in spec and "not all equal" in spec


def test_sample_problem_raises_for_unsatisfiable_C():
    import pytest
    with pytest.raises((ValueError, RuntimeError)):
        sample_problem(30, seed=1)
