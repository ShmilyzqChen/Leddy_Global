import numpy as np

from leddy.qc import conservative_qc


SETTINGS = {
    "window_cells": 5,
    "mad_multiplier": 6.0,
    "maximum_feature_cells": 4,
    "maximum_hole_width_cells": 2,
    "maximum_depth_ratio": 2.0,
}


def test_isolated_spike_and_hole_are_corrected():
    values = np.ones((11, 11), dtype=float)
    depth = np.full(values.shape, 1000.0)
    values[5, 5] = 1000.0
    values[3, 3] = np.nan
    corrected, flag = conservative_qc(values, depth, SETTINGS)
    assert np.isclose(corrected[5, 5], 1.0)
    assert np.isclose(corrected[3, 3], 1.0)
    assert flag[5, 5] == 1
    assert flag[3, 3] == 2


def test_depth_barrier_prevents_cross_shelf_fill():
    values = np.full((5, 5), np.nan)
    depth = np.full((5, 5), 100.0)
    depth[:, 3:] = 1000.0
    values[:, 3:] = 2.0
    corrected, flag = conservative_qc(values, depth, SETTINGS)
    assert np.isnan(corrected[2, 2])
    assert flag[2, 2] == 255

