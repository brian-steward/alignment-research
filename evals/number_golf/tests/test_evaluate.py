import pytest
from fractions import Fraction
from schema import Config, Caps
from validate import parse, evaluate, is_valid, EvalError

FULL = frozenset({"+", "-", "*", "/", "^", "!", "neg", "concat"})


def cfg(target=0, ops=FULL, caps=None):
    return Config(digit=4, target=target, ops=ops, caps=caps or Caps())


def test_evaluate_arithmetic():
    assert evaluate(parse("4*4+4+4", cfg()), Caps()) == Fraction(24)


def test_evaluate_division_is_exact():
    assert evaluate(parse("4/4", cfg()), Caps()) == Fraction(1)


def test_evaluate_factorial():
    assert evaluate(parse("4!", cfg()), Caps()) == Fraction(24)


def test_evaluate_power():
    assert evaluate(parse("4^4", cfg()), Caps()) == Fraction(256)


def test_evaluate_div_by_zero_raises():
    with pytest.raises(EvalError):
        evaluate(parse("4/(4-4)", cfg()), Caps())


def test_evaluate_factorial_noninteger_raises():
    with pytest.raises(EvalError):
        evaluate(parse("(4/4/4)!", cfg()), Caps())


def test_evaluate_factorial_too_big_raises():
    caps = Caps(max_fact_arg=5)
    with pytest.raises(EvalError):
        evaluate(parse("(4+4)!", cfg(caps=caps)), caps)  # 8! with cap 5


def test_evaluate_pow_exponent_cap():
    caps = Caps(max_pow_exp=3)
    with pytest.raises(EvalError):
        evaluate(parse("4^(4+4)", cfg(caps=caps)), caps)  # exp 8 > 3


def test_is_valid_true():
    assert is_valid("4*4+4+4", cfg(target=24)) is True


def test_is_valid_false_wrong_value():
    assert is_valid("4+4", cfg(target=24)) is False


def test_is_valid_false_parse_error():
    assert is_valid("4+", cfg(target=8)) is False


def test_is_valid_respects_node_cap():
    caps = Caps(max_nodes=3)
    assert is_valid("4+4+4", cfg(target=12, caps=caps)) is False  # 5 nodes > 3
