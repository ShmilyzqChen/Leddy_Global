"""Conservative, auditable correction of isolated spatial defects."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage


def _depth_compatible(depth_a: float, depth_b: np.ndarray, maximum_ratio: float) -> np.ndarray:
    small = np.minimum(depth_a, depth_b)
    large = np.maximum(depth_a, depth_b)
    return np.isfinite(depth_b) & (small > 0.0) & (large / small <= maximum_ratio)


@njit(cache=True)
def _local_outlier_candidates_numba(
    values: np.ndarray,
    ocean: np.ndarray,
    depth: np.ndarray,
    window: int,
    mad_multiplier: float,
    maximum_depth_ratio: float,
    periodic_longitude: bool,
) -> np.ndarray:
    log_values = np.full(values.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(values) & (values > 0.0) & ocean
    for j, i in zip(*np.nonzero(positive)):
        log_values[j, i] = np.log10(values[j, i])
    result = np.zeros(values.shape, dtype=np.bool_)
    radius = window // 2
    ny, nx = values.shape
    for j, i in zip(*np.nonzero(positive)):
        samples = np.empty(window * window - 1, dtype=np.float64)
        count = 0
        for dj in range(-radius, radius + 1):
            jj = j + dj
            if jj < 0 or jj >= ny:
                continue
            for di in range(-radius, radius + 1):
                if dj == 0 and di == 0:
                    continue
                ii = i + di
                if periodic_longitude:
                    ii %= nx
                elif ii < 0 or ii >= nx:
                    continue
                if positive[jj, ii]:
                    small = min(depth[j, i], depth[jj, ii])
                    large = max(depth[j, i], depth[jj, ii])
                    if small > 0.0 and large / small <= maximum_depth_ratio:
                        samples[count] = log_values[jj, ii]
                        count += 1
        if count < 8:
            continue
        median = float(np.median(samples[:count]))
        deviations = np.empty(count, dtype=np.float64)
        for index in range(count):
            deviations[index] = abs(samples[index] - median)
        mad = float(np.median(deviations))
        robust_sigma = max(1.4826 * mad, 0.05)
        result[j, i] = abs(log_values[j, i] - median) > mad_multiplier * robust_sigma
    return result


def _retain_small_components(mask: np.ndarray, maximum_cells: int) -> np.ndarray:
    labels, number = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if number == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    allowed = np.flatnonzero((sizes <= maximum_cells) & (np.arange(sizes.size) > 0))
    return np.isin(labels, allowed)


def _small_holes(mask: np.ndarray, maximum_width: int) -> np.ndarray:
    labels, number = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    keep = np.zeros(mask.shape, dtype=bool)
    for label_id in range(1, number + 1):
        yy, xx = np.nonzero(labels == label_id)
        if yy.size and yy.max() - yy.min() + 1 <= maximum_width and xx.max() - xx.min() + 1 <= maximum_width:
            keep[yy, xx] = True
    return keep


def _fill_targets(
    values: np.ndarray,
    targets: np.ndarray,
    ocean: np.ndarray,
    depth: np.ndarray,
    maximum_depth_ratio: float,
    periodic_longitude: bool,
) -> np.ndarray:
    result = values.copy()
    result[targets] = np.nan
    ny, nx = result.shape
    pending = targets.copy()
    for _ in range(3):
        updates: list[tuple[int, int, float]] = []
        for j, i in zip(*np.nonzero(pending)):
            samples = []
            weights = []
            for dj in (-1, 0, 1):
                jj = j + dj
                if jj < 0 or jj >= ny:
                    continue
                for di in (-1, 0, 1):
                    if dj == 0 and di == 0:
                        continue
                    ii = i + di
                    if periodic_longitude:
                        ii %= nx
                    elif ii < 0 or ii >= nx:
                        continue
                    value = result[jj, ii]
                    if (
                        ocean[jj, ii]
                        and np.isfinite(value)
                        and value > 0.0
                        and _depth_compatible(depth[j, i], np.array([depth[jj, ii]]), maximum_depth_ratio)[0]
                    ):
                        samples.append(np.log(value))
                        weights.append(1.0 / np.hypot(dj, di))
            if len(samples) >= 3:
                updates.append((j, i, float(np.exp(np.average(samples, weights=weights)))))
        if not updates:
            break
        for j, i, value in updates:
            result[j, i] = value
            pending[j, i] = False
    return result


def conservative_qc(values: np.ndarray, bottom_depth_m: np.ndarray, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Replace only small isolated outliers and small interior ocean holes.

    Flags: 0 unchanged, 1 isolated outlier replaced, 2 small hole filled,
    255 land or unresolved/missing.
    """
    raw = np.asarray(values, dtype=np.float64)
    depth = np.asarray(bottom_depth_m, dtype=np.float64)
    ocean = np.isfinite(depth) & (depth > 0.0)
    periodic_longitude = raw.shape[1] == 1440
    outlier_candidates = _local_outlier_candidates_numba(
        raw,
        ocean,
        depth,
        int(settings["window_cells"]),
        float(settings["mad_multiplier"]),
        float(settings["maximum_depth_ratio"]),
        periodic_longitude,
    )
    outliers = _retain_small_components(outlier_candidates, int(settings["maximum_feature_cells"]))
    corrected = _fill_targets(raw, outliers, ocean, depth, float(settings["maximum_depth_ratio"]), periodic_longitude)
    replaced_outliers = outliers & np.isfinite(corrected)

    holes = ocean & ~np.isfinite(corrected)
    fillable_holes = _small_holes(holes, int(settings["maximum_hole_width_cells"]))
    corrected = _fill_targets(corrected, fillable_holes, ocean, depth, float(settings["maximum_depth_ratio"]), periodic_longitude)
    filled_holes = fillable_holes & np.isfinite(corrected)

    flag = np.full(raw.shape, 255, dtype=np.uint8)
    flag[ocean & np.isfinite(raw)] = 0
    flag[replaced_outliers] = 1
    flag[filled_holes] = 2
    return corrected, flag
