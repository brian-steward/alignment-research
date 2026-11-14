import pytest
from schema import Config, Caps, ALL_OPS


def test_config_basic_fields():
    c = Config(digit=4, target=42, ops=frozenset({"+", "-", "*", "/"}), name="arith-42")
    assert c.digit == 4
    assert c.target == 42
    assert "+" in c.ops
    assert isinstance(c.caps, Caps)


def test_config_rejects_bad_digit():
    with pytest.raises(ValueError):
        Config(digit=0, target=1, ops=frozenset({"+"}))
    with pytest.raises(ValueError):
        Config(digit=10, target=1, ops=frozenset({"+"}))


def test_config_rejects_unknown_op():
    with pytest.raises(ValueError):
        Config(digit=4, target=1, ops=frozenset({"+", "bogus"}))


def test_all_ops_contents():
    assert {"+", "-", "*", "/", "^", "!", "neg", "concat"} <= ALL_OPS
