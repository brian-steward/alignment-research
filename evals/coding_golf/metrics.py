from __future__ import annotations


def distinct_at(keys) -> int:
    return len(set(keys))


def taxonomy_coverage(found_keys: set, reference_keys: dict) -> int:
    return len(set(found_keys) & set(reference_keys))


def novel_count(found_keys: set, reference_keys: dict) -> int:
    return len(set(found_keys) - set(reference_keys))
