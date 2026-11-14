import pytest
from schema import Config
from expr import Num, Bin, Un, count_nodes
from validate import parse, ParseError

FULL = frozenset({"+", "-", "*", "/", "^", "!", "neg", "concat"})


def cfg(ops=FULL, digit=4):
    return Config(digit=digit, target=0, ops=ops)


def test_parse_single_digit():
    assert parse("4", cfg()) == Num(4)


def test_parse_addition_left_assoc():
    assert parse("4+4+4", cfg()) == Bin("+", Bin("+", Num(4), Num(4)), Num(4))


def test_parse_precedence_mul_over_add():
    assert parse("4+4*4", cfg()) == Bin("+", Num(4), Bin("*", Num(4), Num(4)))


def test_parse_parens():
    assert parse("(4+4)*4", cfg()) == Bin("*", Bin("+", Num(4), Num(4)), Num(4))


def test_parse_concat_literal():
    assert parse("44", cfg()) == Num(44)


def test_parse_concat_rejected_when_not_allowed():
    with pytest.raises(ParseError):
        parse("44", cfg(ops=frozenset({"+"})))


def test_parse_wrong_digit_rejected():
    with pytest.raises(ParseError):
        parse("5", cfg(digit=4))


def test_parse_unary_minus():
    assert parse("-4", cfg()) == Un("neg", Num(4))


def test_parse_factorial_postfix():
    assert parse("4!", cfg()) == Un("fact", Num(4))


def test_parse_power_right_assoc():
    assert parse("4^4^4", cfg()) == Bin("^", Num(4), Bin("^", Num(4), Num(4)))


def test_parse_disallowed_op_rejected():
    with pytest.raises(ParseError):
        parse("4!", cfg(ops=frozenset({"+", "concat"})))


def test_parse_trailing_garbage_rejected():
    with pytest.raises(ParseError):
        parse("4+", cfg())


def test_count_nodes():
    assert count_nodes(parse("4+4*4", cfg())) == 5
