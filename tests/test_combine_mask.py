import numpy as np

from leddy.qc import conservative_qc


def test_qc_does_not_promote_values_on_target_grid_land():
    values = np.ones((7, 7), dtype=float)
    bottom = np.full((7, 7), 1000.0)
    bottom[0, :] = np.nan
    values[~np.isfinite(bottom)] = np.nan
    settings = {
        "window_cells": 5,
        "mad_multiplier": 6.0,
        "maximum_feature_cells": 4,
        "maximum_hole_width_cells": 2,
        "maximum_depth_ratio": 2.0,
    }
    corrected, flag = conservative_qc(values, bottom, settings)
    assert np.all(np.isnan(corrected[0, :]))
    assert np.all(flag[0, :] == 255)
