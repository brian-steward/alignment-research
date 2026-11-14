from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Num:
    value: int


@dataclass(frozen=True)
class Bin:
    op: str   # one of: + - * / ^
    l: "Node"
    r: "Node"


@dataclass(frozen=True)
class Un:
    op: str   # one of: neg fact
    x: "Node"


Node = Union[Num, Bin, Un]


def count_nodes(node: Node) -> int:
    if isinstance(node, Num):
        return 1
    if isinstance(node, Un):
        return 1 + count_nodes(node.x)
    if isinstance(node, Bin):
        return 1 + count_nodes(node.l) + count_nodes(node.r)
    raise TypeError(f"not a Node: {node!r}")
