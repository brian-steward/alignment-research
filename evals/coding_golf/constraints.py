from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Constraint:
    """A predicate over every length-3 window (a, b, c) of a ternary string."""
    kind: str
    op: str = ""        # for "sum": one of "<=", ">=", "!="
    t: int = 0          # threshold for "sum"
    r: int = 0          # residue for "mod"

    def holds(self, a: int, b: int, c: int) -> bool:
        if self.kind == "sum":
            s = a + b + c
            if self.op == "<=":
                return s <= self.t
            if self.op == ">=":
                return s >= self.t
            return s != self.t                      # "!="
        if self.kind == "repeat3":
            return not (a == b == c)
        if self.kind == "repeat2":
            return a != c
        if self.kind == "inc":
            return not (a < b < c)
        if self.kind == "dec":
            return not (a > b > c)
        if self.kind == "mod":
            return (a + b + c) % 3 != self.r
        raise ValueError(f"unknown kind {self.kind!r}")

    def render(self) -> str:
        if self.kind == "sum":
            word = {"<=": "is at most", ">=": "is at least", "!=": "does not equal"}[self.op]
            return f"the sum a+b+c {word} {self.t}"
        if self.kind == "repeat3":
            return "the three values are not all equal"
        if self.kind == "repeat2":
            return "the first and third values differ (a != c)"
        if self.kind == "inc":
            return "the values are not strictly increasing (not a<b<c)"
        if self.kind == "dec":
            return "the values are not strictly decreasing (not a>b>c)"
        if self.kind == "mod":
            return f"(a+b+c) mod 3 is not {self.r}"
        raise ValueError(f"unknown kind {self.kind!r}")


def sample_constraint(rng: random.Random) -> Constraint:
    kind = rng.choice(["sum", "repeat3", "repeat2", "inc", "dec", "mod"])
    if kind == "sum":
        op = rng.choice(["<=", ">=", "!="])
        t = rng.randint(1, 5)
        return Constraint("sum", op=op, t=t)
    if kind == "mod":
        return Constraint("mod", r=rng.randint(0, 2))
    return Constraint(kind)


ALPHABET = range(3)
WINDOW = 3


def allowed_windows(cs):
    return frozenset(
        (a, b, c)
        for a, b, c in product(ALPHABET, repeat=WINDOW)
        if all(con.holds(a, b, c) for con in cs)
    )


def count_bruteforce(cs, n: int) -> int:
    if n < 0:
        return 0
    aw = allowed_windows(cs)
    total = 0
    for s in product(ALPHABET, repeat=n):
        if all((s[i], s[i + 1], s[i + 2]) in aw for i in range(n - 2)):
            total += 1
    return total


def _is_non_degenerate(cs) -> bool:
    counts = [count_bruteforce(cs, n) for n in range(1, 7)]
    if any(x <= 0 for x in counts):
        return False
    if any(b <= a for a, b in zip(counts, counts[1:])):
        return False
    return count_bruteforce(cs, 4) < 3 ** 4          # constraints actually remove strings


def sample_problem(n_constraints: int, seed: int):
    rng = random.Random(seed)
    for _ in range(2000):
        cs, seen = [], set()
        for _ in range(500):                  # bounded draws; the key space is finite (~22)
            if len(cs) >= n_constraints:
                break
            c = sample_constraint(rng)
            key = (c.kind, c.op, c.t, c.r)
            if key in seen:
                continue
            seen.add(key)
            cs.append(c)
        if len(cs) < n_constraints:
            raise ValueError(
                f"cannot draw {n_constraints} distinct constraints "
                f"(predicate vocabulary has only ~22 unique members)"
            )
        if _is_non_degenerate(cs):
            return cs
    raise RuntimeError(f"no non-degenerate set for C={n_constraints}, seed={seed}")


def render_spec(cs) -> str:
    lines = "\n".join(f"  - {c.render()}" for c in cs)
    return (
        "Count the number of strings of length n over the alphabet {0, 1, 2} "
        "such that, for EVERY three consecutive characters (a, b, c), ALL of "
        "the following hold:\n"
        f"{lines}\n"
        "Strings shorter than 3 have no such window and are all valid. "
        "Write `def count(n):` returning the exact count."
    )
