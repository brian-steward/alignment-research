from schema import Config
from validate import parse
from fingerprint import l2_key, canon_size

FULL = frozenset({"+", "-", "*", "/", "^", "!", "neg", "concat"})


def k(s):
    return l2_key(parse(s, Config(digit=4, target=0, ops=FULL)))


def test_commutation_collapses_add():
    assert k("4+4*4") == k("4*4+4")


def test_commutation_collapses_mul():
    assert k("4*(4+4)") == k("(4+4)*4")


def test_associativity_collapses():
    assert k("4+(4+4)") == k("(4+4)+4")


def test_additive_padding_collapses():
    assert k("4*4") == k("4*4+(4-4)")


def test_multiplicative_padding_collapses():
    assert k("4+4") == k("(4+4)*(4/4)")


def test_mul_distinct_from_pow():
    # 4*4*4 (repeated multiplication) is a different METHOD than a pow node.
    assert k("4*4*4") != k("4^(4-4+4)")
    assert k("4*4") != k("4^4")


def test_pow_method_distinct_from_repeated_mul_same_value():
    # 4*4 (value 16) vs 4^(4/4+4/4) (value 16) -> different operator method.
    assert k("4*4") != k("4^(4/4+4/4)")


def test_identical_expressions_same_key():
    assert k("4!+4") == k("4!+4")


def test_factorial_changes_key():
    assert k("4+4") != k("4!+4")


def test_canon_size_monotone():
    assert canon_size(k("4")) < canon_size(k("4+4+4+4"))


def test_value_zero_additive_padding_collapses():
    # (4-4)*4 evaluates to 0 (a non-literal additive identity) -> must be stripped.
    assert k("4*4+(4-4)*4+4/4") == k("4*4+4/4")


def test_value_one_multiplicative_padding_collapses():
    # 4^(4-4) = 4^0 = 1 as a factor -> multiplicative identity -> stripped.
    assert k("(4*4+4/4)*4^(4-4)") == k("4*4+4/4")


def test_additive_one_term_not_stripped():
    # +4/4 is value 1, NOT 0 -> it contributes to the sum and must be kept.
    assert k("4*4+4/4") != k("4*4")
