import numpy as np

from leddy.bathymetry import effective_bottom_depth


def test_effective_bottom_depth_midpoint_and_truncation():
    standard = np.array([0.0, 5.0, 10.0, 20.0])
    code = np.array([[1, 2, 4, 5]])
    result = effective_bottom_depth(code, standard, maximum_depth_m=20.0)
    assert np.isnan(result[0, 0])
    assert result[0, 1] == 2.5
    assert result[0, 2] == 15.0
    assert result[0, 3] == 20.0

