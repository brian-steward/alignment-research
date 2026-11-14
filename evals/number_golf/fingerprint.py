from __future__ import annotations

from collections import Counter

from expr import Num, Bin, Un, Node
from schema import Caps
from validate import evaluate, EvalError

# Generous caps used only to value-test sub-expressions for identity stripping.
_CAPS = Caps()


def _value(node: Node):
    """Exact value of a sub-expression, or None if it cannot be evaluated."""
    try:
        return evaluate(node, _CAPS)
    except EvalError:
        return None


def _collect_add(node: Node, sign: int, terms: Counter) -> None:
    if isinstance(node, Bin) and node.op == "+":
        _collect_add(node.l, sign, terms)
        _collect_add(node.r, sign, terms)
    elif isinstance(node, Bin) and node.op == "-":
        _collect_add(node.l, sign, terms)
        _collect_add(node.r, -sign, terms)
    elif isinstance(node, Un) and node.op == "neg":
        _collect_add(node.x, -sign, terms)
    elif _value(node) == 0:
        # additive identity (e.g. (4-4)*4) -> contributes nothing, strip it
        return
    else:
        terms[_canon(node)] += sign


def _collect_mul(node: Node, exp: int, factors: Counter) -> None:
    if isinstance(node, Bin) and node.op == "*":
        _collect_mul(node.l, exp, factors)
        _collect_mul(node.r, exp, factors)
    elif isinstance(node, Bin) and node.op == "/":
        _collect_mul(node.l, exp, factors)
        _collect_mul(node.r, -exp, factors)
    elif _value(node) == 1:
        # multiplicative identity (e.g. 4^(4-4)) -> contributes nothing, strip it
        return
    else:
        factors[_canon(node)] += exp


def _canon(node: Node):
    if isinstance(node, Num):
        return ("num", node.value)

    if isinstance(node, Un):
        if node.op == "fact":
            return ("fact", _canon(node.x))
        if node.op == "neg":
            terms: Counter = Counter()
            _collect_add(node, 1, terms)
            return _finish_add(terms)
        raise ValueError(f"unknown unary op {node.op}")

    if isinstance(node, Bin):
        if node.op in ("+", "-"):
            terms = Counter()
            _collect_add(node, 1, terms)
            return _finish_add(terms)
        if node.op in ("*", "/"):
            factors = Counter()
            _collect_mul(node, 1, factors)
            return _finish_mul(factors)
        if node.op == "^":
            return ("pow", _canon(node.l), _canon(node.r))
        raise ValueError(f"unknown binary op {node.op}")

    raise TypeError(f"not a node: {node!r}")


def _finish_add(terms: Counter):
    items = [(core, c) for core, c in terms.items() if c != 0 and core != ("num", 0)]
    if not items:
        return ("num", 0)
    if len(items) == 1 and items[0][1] == 1:
        return items[0][0]
    return ("add", tuple(sorted(items, key=repr)))


def _finish_mul(factors: Counter):
    items = [(core, e) for core, e in factors.items() if e != 0 and core != ("num", 1)]
    if not items:
        return ("num", 1)
    if len(items) == 1 and items[0][1] == 1:
        return items[0][0]
    return ("mul", tuple(sorted(items, key=repr)))


def l2_key(node: Node):
    """Canonical L2 distinctness key. Two ASTs share a key iff they are the
    same solution up to commutation, associativity, and no-op padding, while
    keeping operator choice (x-repeat vs ^, factorial) as a distinct method.

    Known L2 limitation: only inverse-pair cancellation and literal 0/1 identity
    stripping are applied. A structurally non-trivial value-0/value-1 subtree
    (e.g. ``4^(4-4)``, ``(4-4)!``) survives as a distinct term/factor and can
    inflate the distinct count. Collapsing it would require value evaluation,
    which is deliberately avoided (number-golf has no variables, so folding all
    constants would collapse every solution to the target). See spec sec 4.4/11."""
    return _canon(node)


def canon_size(key) -> int:
    if not isinstance(key, tuple):
        return 1
    tag = key[0]
    if tag == "num":
        return 1
    if tag == "fact":
        return 1 + canon_size(key[1])
    if tag == "pow":
        return 1 + canon_size(key[1]) + canon_size(key[2])
    if tag in ("add", "mul"):
        return 1 + sum(canon_size(core) + 1 for core, _ in key[1])
    return 1
