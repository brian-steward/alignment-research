def test_python_and_numpy_available():
    import numpy as np
    assert np.array([1, 2, 3]).sum() == 6
