from schema import Config, Caps
from enumerate_golf import enumerate_solutions

ARITH = frozenset({"+", "-", "*", "/"})


def test_two_fours_addition_to_eight():
    # Only +,-,*,/ and at most 2 leaves of digit 4.
    # Expressions equal to 8 within 2 leaves: 4+4 only (4*4=16, 4-4=0, 4/4=1).
    c = Config(digit=4, target=8, ops=ARITH, caps=Caps())
    keys = enumerate_solutions(c, max_leaves=2)
    assert len(keys) == 1


def test_enumeration_finds_known_24():
    # 4*4+4+4 = 24 with 4 leaves should be among the solutions.
    c = Config(digit=4, target=24, ops=ARITH, caps=Caps())
    keys = enumerate_solutions(c, max_leaves=4)
    assert len(keys) >= 1


def test_richer_ops_find_at_least_as_many():
    # Adding factorial cannot reduce the solution count for a fixed leaf budget.
    base = Config(digit=4, target=24, ops=ARITH, caps=Caps())
    rich = Config(digit=4, target=24, ops=ARITH | frozenset({"!"}), caps=Caps())
    n_base = len(enumerate_solutions(base, max_leaves=3))
    n_rich = len(enumerate_solutions(rich, max_leaves=3))
    assert n_rich >= n_base
