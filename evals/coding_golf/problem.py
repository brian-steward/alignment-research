from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass
class Problem:
    name: str
    entry: str                                   # function name a solution must define
    validity_tests: List[Tuple[tuple, object]]   # [(args, expected)], small-n, exact
    probe_sizes: List[int]                        # increasing sizes for complexity probe
    probe_args: Callable[[int], tuple]            # size -> args tuple for the entry fn
    features: Callable[[str], frozenset]          # source -> detected structural features
    reference_methods: Dict[str, str]             # method_name -> source (known taxonomy)
