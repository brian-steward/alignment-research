from __future__ import annotations

import ast
import statistics

from sandbox import probe_opcounts


class _Normalize(ast.NodeTransformer):
    """Alpha-rename locally-introduced identifiers to canonical tokens so pure
    variable renamings collapse. Preserves attribute names and non-local call
    targets (builtins, module funcs)."""

    def __init__(self):
        self.mapping = {}

    def _canon(self, name):
        if name not in self.mapping:
            self.mapping[name] = f"v{len(self.mapping)}"
        return self.mapping[name]

    def visit_arg(self, node):
        node.arg = self._canon(node.arg)
        return node

    def visit_Name(self, node):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        elif isinstance(node.ctx, ast.Store):
            node.id = self._canon(node.id)
        return node


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:]
    return tree


def l1_key(source: str) -> str:
    tree = _strip_docstrings(ast.parse(source))
    tree = _Normalize().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=False)


def complexity_class(counts: dict) -> str:
    """Growth class of a line count measured at sizes that double (10, 20, 40, ...).

    `counts` maps size -> line count, or None where the run timed out. The class is read from the
    increments between consecutive sizes, so a fixed setup cost cancels: with doubling sizes the
    increments stay level for logarithmic growth, double for linear growth and quadruple for
    quadratic growth. The ratio is the median over the three largest steps, where a table of
    precomputed small cases no longer hides the growth.

    Until 2026-11-14 the ratios were taken between the counts themselves. A solution with a large
    setup cost and slow growth then had a ratio near 1 and was labelled const whatever its growth
    law (RESULTS.md, section 2).
    """
    pts = sorted((s, o) for s, o in counts.items() if o is not None and o > 0)
    timed_out = any(o is None for o in counts.values())
    if len(pts) < 2:
        return "exponential" if timed_out else "unknown"
    ops = [o for _, o in pts]
    inc = [b - a for a, b in zip(ops, ops[1:])]
    steps = [b / a for a, b in zip(inc, inc[1:]) if a > 0 and b > 0][-3:]
    r = statistics.median(steps) if steps else None
    if timed_out and (len(inc) < 3 or r is None or r >= 6 or ops[-1] <= 3 * ops[0]):
        # Too slow at some size, and the counts before it show no polynomial law steep enough to
        # account for that: the work grows faster than the line count can show.
        return "exponential"
    if len(inc) < 2:
        return "unknown"  # one increment says nothing about the growth law
    if inc[-1] <= 0 and inc[-2] <= 0:
        return "const"
    if r is None:
        return "unknown"  # the count only starts to grow at the largest size
    if r >= 6:
        return "exponential"
    if r >= 3:
        return "quadratic"
    if r >= 1.6:
        return "linear"
    return "log"


def strategy_key(source: str, problem, counts: dict):
    return (complexity_class(counts), problem.features(source))


def reference_strategies(problem) -> dict:
    out = {}
    for name, src in problem.reference_methods.items():
        counts = probe_opcounts(src, problem)
        out[strategy_key(src, problem, counts)] = name
    return out
