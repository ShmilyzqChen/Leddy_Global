"""Read the official WOA land/sea mask and derive an effective bottom depth."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_bottom_standard_level(path: str | Path) -> np.ndarray:
    """Return WOA bottom-level codes on longitude -180..180.

    The distributed ASCII file is ordered latitude-major with longitude
    0.125..359.875. Code 1 is land; code n is the first standard depth level
    intersecting the bottom.
    """
    codes = np.loadtxt(path, delimiter=",", skiprows=2, usecols=2, dtype=np.int16)
    expected = 720 * 1440
    if codes.size != expected:
        raise ValueError(f"Expected {expected} bottom-mask records, found {codes.size}")
    codes = codes.reshape(720, 1440)
    return np.roll(codes, shift=720, axis=1)


def effective_bottom_depth(
    bottom_level: np.ndarray,
    standard_depth_m: np.ndarray,
    maximum_depth_m: float = 5500.0,
) -> np.ndarray:
    """Convert WOA bottom codes to a midpoint effective water depth.

    For a bottom encountered at standard level n, H is halfway between the
    deepest wet standard level n-1 and level n. Profiles deeper than the WOA
    temperature/salinity product are conservatively truncated.
    """
    code = np.asarray(bottom_level, dtype=np.int32)
    depth = np.asarray(standard_depth_m, dtype=np.float64)
    result = np.full(code.shape, np.nan, dtype=np.float64)
    ocean = code > 1
    within = ocean & (code <= depth.size)
    n = code[within]
    result[within] = 0.5 * (depth[n - 2] + depth[n - 1])
    result[ocean & (code > depth.size)] = min(float(maximum_depth_m), float(depth[-1]))
    result[result > maximum_depth_m] = maximum_depth_m
    return result

