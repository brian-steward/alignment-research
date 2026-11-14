from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import List

from expr import Num, Bin, Un, Node, count_nodes

_TOKEN_RE = re.compile(r"\s*(\d+|[()+\-*/^!])")


class ParseError(Exception):
    pass


class EvalError(Exception):
    pass


def tokenize(s: str) -> List[str]:
    toks: List[str] = []
    pos = 0
    for m in _TOKEN_RE.finditer(s):
        if m.start() != pos:
            raise ParseError(f"unexpected character at {pos}: {s[pos:]!r}")
        pos = m.end()
        toks.append(m.group(1))
    if s[pos:].strip():
        raise ParseError(f"trailing characters: {s[pos:]!r}")
    if not toks:
        raise ParseError("empty expression")
    return toks


class _Parser:
    def __init__(self, toks: List[str], config):
        self.toks = toks
        self.i = 0
        self.config = config

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, t):
        if self._peek() != t:
            raise ParseError(f"expected {t!r}, got {self._peek()!r}")
        self._next()

    def _need(self, op):
        if op not in self.config.ops:
            raise ParseError(f"operator {op!r} not allowed for this config")

    def parse(self) -> Node:
        node = self._expr()
        if self._peek() is not None:
            raise ParseError(f"unexpected trailing token {self._peek()!r}")
        return node

    def _expr(self) -> Node:
        node = self._term()
        while self._peek() in ("+", "-"):
            op = self._next()
            self._need(op)
            node = Bin(op, node, self._term())
        return node

    def _term(self) -> Node:
        node = self._unary()
        while self._peek() in ("*", "/"):
            op = self._next()
            self._need(op)
            node = Bin(op, node, self._unary())
        return node

    def _unary(self) -> Node:
        if self._peek() == "-":
            self._next()
            self._need("neg")
            return Un("neg", self._unary())
        return self._power()

    def _power(self) -> Node:
        base = self._postfix()
        if self._peek() == "^":
            self._next()
            self._need("^")
            return Bin("^", base, self._unary())  # right-assoc, allows ^-exp
        return base

    def _postfix(self) -> Node:
        node = self._atom()
        while self._peek() == "!":
            self._next()
            self._need("!")
            node = Un("fact", node)
        return node

    def _atom(self) -> Node:
        t = self._peek()
        if t == "(":
            self._next()
            node = self._expr()
            self._expect(")")
            return node
        if t is not None and t.isdigit():
            self._next()
            if len(t) > 1 and "concat" not in self.config.ops:
                raise ParseError("multi-digit literal requires 'concat' op")
            if any(ch != str(self.config.digit) for ch in t):
                raise ParseError(f"literal {t!r} uses a digit other than {self.config.digit}")
            return Num(int(t))
        raise ParseError(f"unexpected token {t!r}")


def parse(s: str, config) -> Node:
    return _Parser(tokenize(s), config).parse()


def _check_mag(v: Fraction, caps) -> None:
    if abs(v.numerator) > caps.max_abs or abs(v.denominator) > caps.max_abs:
        raise EvalError("intermediate magnitude exceeds cap")


def evaluate(node: Node, caps) -> Fraction:
    if isinstance(node, Num):
        if abs(node.value) > caps.max_int_value:
            raise EvalError("literal exceeds cap")
        return Fraction(node.value)

    if isinstance(node, Un):
        if node.op == "neg":
            return -evaluate(node.x, caps)
        if node.op == "fact":
            x = evaluate(node.x, caps)
            if x.denominator != 1 or x < 0 or x > caps.max_fact_arg:
                raise EvalError(f"factorial of invalid argument {x}")
            r = Fraction(math.factorial(int(x)))
            _check_mag(r, caps)
            return r
        raise EvalError(f"unknown unary op {node.op}")

    if isinstance(node, Bin):
        a = evaluate(node.l, caps)
        b = evaluate(node.r, caps)
        if node.op == "+":
            r = a + b
        elif node.op == "-":
            r = a - b
        elif node.op == "*":
            r = a * b
        elif node.op == "/":
            if b == 0:
                raise EvalError("division by zero")
            r = a / b
        elif node.op == "^":
            if b.denominator != 1:
                raise EvalError("non-integer exponent")
            e = int(b)
            if abs(e) > caps.max_pow_exp:
                raise EvalError("exponent exceeds cap")
            if e < 0 and a == 0:
                raise EvalError("0 to negative power")
            r = a ** e
        else:
            raise EvalError(f"unknown binary op {node.op}")
        _check_mag(r, caps)
        return r

    raise EvalError(f"not a node: {node!r}")


def is_valid(s: str, config) -> bool:
    try:
        node = parse(s, config)
    except ParseError:
        return False
    if count_nodes(node) > config.caps.max_nodes:
        return False
    try:
        value = evaluate(node, config.caps)
    except EvalError:
        return False
    return value == Fraction(config.target)
