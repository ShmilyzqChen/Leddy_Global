"""Grid selection and exact DUACS-to-WOA grid relationships."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSelection:
    """A rectangular selection on the 0.25-degree WOA grid."""

    lat_slice: slice
    lon_slice: slice
    lat: np.ndarray
    lon: np.ndarray


def select_regular_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    region: dict,
) -> GridSelection:
    """Select inclusive cell-centre bounds from a monotonic regular grid."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    bounds = (region.get("lon_min"), region.get("lon_max"), region.get("lat_min"), region.get("lat_max"))
    if all(value is None for value in bounds):
        return GridSelection(slice(0, lat.size), slice(0, lon.size), lat.copy(), lon.copy())
    if any(value is None for value in bounds):
        raise ValueError("Region bounds must be all null (global) or all numeric")
    lon_min, lon_max, lat_min, lat_max = map(float, bounds)
    if lon_min > lon_max:
        raise ValueError("Dateline-crossing regions are not yet supported; split them into two runs")

    iy = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
    ix = np.flatnonzero((lon >= lon_min) & (lon <= lon_max))
    if iy.size == 0 or ix.size == 0:
        raise ValueError(f"Region does not intersect grid: {region}")
    if not np.all(np.diff(iy) == 1) or not np.all(np.diff(ix) == 1):
        raise ValueError("Selected region must be contiguous")
    ys = slice(int(iy[0]), int(iy[-1]) + 1)
    xs = slice(int(ix[0]), int(ix[-1]) + 1)
    return GridSelection(ys, xs, lat[ys].copy(), lon[xs].copy())


def woa_to_duacs_slices(selection: GridSelection) -> tuple[slice, slice]:
    """Map aligned 0.25-degree WOA indices to their 2x2 DUACS cells."""
    ys, xs = selection.lat_slice, selection.lon_slice
    return slice(2 * ys.start, 2 * ys.stop), slice(2 * xs.start, 2 * xs.stop)


def verify_two_to_one_alignment(
    woa_lat: np.ndarray,
    woa_lon: np.ndarray,
    duacs_lat: np.ndarray,
    duacs_lon: np.ndarray,
    atol: float = 1.0e-7,
) -> None:
    """Fail loudly unless four DUACS cells map exactly to one WOA cell."""
    expected_lat = duacs_lat.reshape(-1, 2).mean(axis=1)
    expected_lon = duacs_lon.reshape(-1, 2).mean(axis=1)
    if expected_lat.shape != woa_lat.shape or not np.allclose(expected_lat, woa_lat, atol=atol):
        raise ValueError("DUACS and WOA latitude grids are not aligned 2:1")
    if expected_lon.shape != woa_lon.shape or not np.allclose(expected_lon, woa_lon, atol=atol):
        raise ValueError("DUACS and WOA longitude grids are not aligned 2:1")

