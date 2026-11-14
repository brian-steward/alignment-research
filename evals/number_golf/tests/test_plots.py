from plots import plot_discovery, plot_scaling


def test_plot_discovery_writes_file(tmp_path):
    out = tmp_path / "disc.png"
    plot_discovery({"modelA": ([1, 2, 3, 4], [1, 2, 3, 3])}, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_scaling_writes_file(tmp_path):
    out = tmp_path / "scale.png"
    plot_scaling({"modelA": ([0, 1, 2], [0.3, 0.5, 0.7])}, str(out),
                 xlabel="difficulty", ylabel="beta")
    assert out.exists() and out.stat().st_size > 0
