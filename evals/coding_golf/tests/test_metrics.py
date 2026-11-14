from metrics import distinct_at, taxonomy_coverage, novel_count


def test_distinct_at():
    assert distinct_at(["a", "a", "b"]) == 2
    assert distinct_at([]) == 0


def test_taxonomy_coverage_counts_known_hits():
    ref = {("linear", frozenset({"iteration"})): "iterative",
           ("exponential", frozenset({"recursion"})): "naive"}
    found = {("linear", frozenset({"iteration"})),
             ("const", frozenset({"closed_form"}))}
    assert taxonomy_coverage(found, ref) == 1


def test_novel_count_counts_unknown_strategies():
    ref = {("linear", frozenset({"iteration"})): "iterative"}
    found = {("linear", frozenset({"iteration"})),
             ("const", frozenset({"closed_form"}))}
    assert novel_count(found, ref) == 1
