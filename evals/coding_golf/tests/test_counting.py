from constraints import sample_problem, count_bruteforce, render_spec
from counting import build_references, detect_features, make_problem
from problem import Problem
from sandbox import is_valid, probe_opcounts
from fingerprint import strategy_key, complexity_class
from counting import PROBE_SIZES
from run_counting_experiment import SWEEP, score_solutions, CLASS_ORDER

EXPECTED_FEATURES = {
    "brute":    frozenset({"brute_force", "iteration"}),
    "backtrack": frozenset({"recursion", "iteration"}),
    "dp_table": frozenset({"dp_table", "iteration"}),
    "memoized": frozenset({"recursion", "memoization", "iteration"}),
    "matrix":   frozenset({"matrix", "iteration", "fast_exp"}),
}

VALIDITY_NS = [0, 1, 2, 3, 4, 5, 8, 10]


def test_references_have_five_methods():
    cs = sample_problem(3, seed=11)
    refs = build_references(cs)
    assert set(refs) == set(EXPECTED_FEATURES)


def test_detectors_match_expected_features():
    cs = sample_problem(3, seed=11)
    refs = build_references(cs)
    for name, src in refs.items():
        assert detect_features(src) == EXPECTED_FEATURES[name], name


def test_five_distinct_feature_sets():
    cs = sample_problem(3, seed=11)
    refs = build_references(cs)
    sets = [detect_features(src) for src in refs.values()]
    assert len(set(sets)) == 5


def test_references_agree_with_bruteforce_counts():
    cs = sample_problem(3, seed=11)
    refs = build_references(cs)
    expected = {n: count_bruteforce(cs, n) for n in [0, 1, 2, 3, 5, 8]}
    ns = sorted(expected)
    for name, src in refs.items():
        ns_local = {}
        exec(src, ns_local)
        fn = ns_local["count"]
        for n in ns:
            assert fn(n) == expected[n], (name, n)


def test_make_problem_wires_validity_and_probe():
    cs = sample_problem(3, seed=11)
    p = make_problem(cs)
    assert isinstance(p, Problem)
    assert p.entry == "count"
    assert p.probe_sizes == PROBE_SIZES == [10, 20, 40, 80, 160, 320, 640]
    assert [n for (n,), _ in p.validity_tests] == VALIDITY_NS
    for (n,), exp in p.validity_tests:
        assert exp == count_bruteforce(cs, n)


def test_all_references_are_valid():
    cs = sample_problem(3, seed=11)
    p = make_problem(cs)
    for name, src in p.reference_methods.items():
        assert is_valid(src, p), name


def test_references_land_in_five_distinct_strategy_classes():
    cs = sample_problem(3, seed=11)
    p = make_problem(cs)
    keys = {}
    for name, src in p.reference_methods.items():
        k = strategy_key(src, p, probe_opcounts(src, p))
        keys[name] = k
    assert len(set(keys.values())) == 5, keys


def test_complexity_labels_are_sensible():
    cs = sample_problem(3, seed=11)
    p = make_problem(cs)
    cls = {name: complexity_class(probe_opcounts(src, p))
           for name, src in p.reference_methods.items()}
    assert cls["brute"] == "exponential"
    assert cls["dp_table"] == "linear"
    assert cls["memoized"] == "linear"
    assert cls["matrix"] == "log"


def test_sweep_is_the_C_dial():
    assert sorted({C for C, _ in SWEEP}) == [2, 4, 6]


def test_score_solutions_on_references():
    C, seed = SWEEP[0]
    cs = sample_problem(C, seed)
    refs = list(build_references(cs).values())
    rep = score_solutions(cs, refs)
    assert rep["n_valid"] == 5
    assert rep["distinct"] == 5
    # references include an exponential brute and an efficient method
    assert rep["lowest_class"] in {"log", "const", "linear"}
    assert "exponential" in rep["classes"]


def test_class_order_ranks_cheaper_lower():
    assert CLASS_ORDER.index("const") < CLASS_ORDER.index("exponential")
    assert CLASS_ORDER.index("log") < CLASS_ORDER.index("linear")
