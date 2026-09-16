"""First-baroclinic vertical-mode calculations on a nonuniform depth grid."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal


def stabilise_n2(
    n2: np.ndarray,
    element_mid_depth_m: np.ndarray,
    element_thickness_m: np.ndarray,
    floor_s_2: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Minimally replace invalid N2 in log space and report thickness affected."""
    raw = np.asarray(n2, dtype=np.float64)
    zmid = np.asarray(element_mid_depth_m, dtype=np.float64)
    dz = np.asarray(element_thickness_m, dtype=np.float64)
    corrected = raw.copy()
    good = np.isfinite(raw) & (raw > 0.0)
    changed = ~good | (good & (raw < floor_s_2))

    if good.sum() >= 2:
        corrected[~good] = np.exp(
            np.interp(zmid[~good], zmid[good], np.log(raw[good]))
        )
    elif good.sum() == 1:
        corrected[~good] = raw[good][0]
    else:
        corrected[:] = floor_s_2
    corrected = np.maximum(corrected, floor_s_2)

    denominator = float(np.sum(dz))
    fraction = float(np.sum(dz[changed]) / denominator) if denominator > 0.0 else 1.0
    return corrected, fraction, changed


def wkb_phase_speed(depth_nodes_m: np.ndarray, n2_elements_s_2: np.ndarray) -> float:
    """Return c1 = pi^-1 integral N dz using element-wise N2."""
    z = np.asarray(depth_nodes_m, dtype=np.float64)
    n2 = np.asarray(n2_elements_s_2, dtype=np.float64)
    if z.size < 2 or n2.size != z.size - 1 or not np.all(np.isfinite(n2)):
        return np.nan
    return float(np.sum(np.sqrt(n2) * np.diff(z)) / np.pi)


def first_baroclinic_phase_speed(
    depth_nodes_m: np.ndarray,
    n2_elements_s_2: np.ndarray,
) -> float:
    """Solve the first hydrostatic baroclinic mode with rigid boundaries.

    The equivalent vertical-displacement problem is
        -G'' = lambda N^2 G,  G(0)=G(H)=0,  lambda=1/c^2.
    A piecewise-linear finite-element stiffness matrix and a positive lumped
    mass matrix retain the native nonuniform WOA spacing. The transformed
    symmetric matrix is tridiagonal, so only its smallest eigenvalue is solved.
    """
    z = np.asarray(depth_nodes_m, dtype=np.float64)
    n2 = np.asarray(n2_elements_s_2, dtype=np.float64)
    if z.ndim != 1 or n2.ndim != 1 or z.size < 3 or n2.size != z.size - 1:
        return np.nan
    h = np.diff(z)
    if np.any(h <= 0.0) or np.any(~np.isfinite(n2)) or np.any(n2 <= 0.0):
        return np.nan

    # Interior-node stiffness K and row-sum-lumped weighted mass M.
    diag_k = 1.0 / h[:-1] + 1.0 / h[1:]
    off_k = -1.0 / h[1:-1]
    mass = 0.5 * (n2[:-1] * h[:-1] + n2[1:] * h[1:])
    if np.any(mass <= 0.0) or np.any(~np.isfinite(mass)):
        return np.nan
    inv_sqrt_mass = 1.0 / np.sqrt(mass)
    diag = diag_k * inv_sqrt_mass**2
    off = off_k * inv_sqrt_mass[:-1] * inv_sqrt_mass[1:]
    try:
        eigenvalue = float(
            eigh_tridiagonal(
                diag,
                off,
                select="i",
                select_range=(0, 0),
                check_finite=False,
                eigvals_only=True,
                lapack_driver="stebz",
            )[0]
        )
    except Exception:
        return np.nan
    if not np.isfinite(eigenvalue) or eigenvalue <= 0.0:
        return np.nan
    return float(1.0 / np.sqrt(eigenvalue))


def solve_profile(
    depth_nodes_m: np.ndarray,
    n2_elements_s_2: np.ndarray,
    n2_floor_s_2: float,
    maximum_corrected_fraction: float,
) -> dict[str, float | int | np.ndarray]:
    """Stabilise one profile, solve c1, and apply the documented WKB fallback."""
    z = np.asarray(depth_nodes_m, dtype=np.float64)
    n2_raw = np.asarray(n2_elements_s_2, dtype=np.float64)
    if np.count_nonzero(np.isfinite(n2_raw) & (n2_raw > 0.0)) < 2:
        return {
            "n2_corrected": np.full(n2_raw.shape, np.nan),
            "changed": np.ones(n2_raw.shape, dtype=bool),
            "corrected_fraction": 1.0,
            "c1_eigen_m_s": np.nan,
            "c1_wkb_m_s": np.nan,
            "c1_final_m_s": np.nan,
            "c1_method_flag": 255,
        }
    dz = np.diff(z)
    zmid = 0.5 * (z[:-1] + z[1:])
    n2_corrected, corrected_fraction, changed = stabilise_n2(
        n2_raw, zmid, dz, n2_floor_s_2
    )
    c1_wkb = wkb_phase_speed(z, n2_corrected)
    c1_eigen = first_baroclinic_phase_speed(z, n2_corrected)
    use_eigen = np.isfinite(c1_eigen) and corrected_fraction <= maximum_corrected_fraction
    c1_final = c1_eigen if use_eigen else c1_wkb
    method_flag = 0 if use_eigen else (1 if np.isfinite(c1_wkb) else 255)
    return {
        "n2_corrected": n2_corrected,
        "changed": changed,
        "corrected_fraction": corrected_fraction,
        "c1_eigen_m_s": c1_eigen,
        "c1_wkb_m_s": c1_wkb,
        "c1_final_m_s": c1_final,
        "c1_method_flag": method_flag,
    }
