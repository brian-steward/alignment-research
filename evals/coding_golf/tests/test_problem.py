from problem import Problem


def test_problem_fields():
    p = Problem(
        name="dummy",
        entry="f",
        validity_tests=[((1,), 1)],
        probe_sizes=[10, 20],
        probe_args=lambda s: (s,),
        features=lambda src: frozenset({"x"}),
        reference_methods={"m": "def f(n): return n"},
    )
    assert p.name == "dummy"
    assert p.entry == "f"
    assert p.probe_args(10) == (10,)
    assert p.features("anything") == frozenset({"x"})
    assert p.validity_tests == [((1,), 1)]
    assert p.reference_methods["m"].startswith("def f")
