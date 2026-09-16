"""Dynamical length-scale formulae."""

from __future__ import annotations

import numpy as np

from .constants import EARTH_ANGULAR_VELOCITY_S_1, EARTH_RADIUS_M, TWO_PI


def coriolis_parameters(latitude_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    f = 2.0 * EARTH_ANGULAR_VELOCITY_S_1 * np.sin(latitude)
    beta = 2.0 * EARTH_ANGULAR_VELOCITY_S_1 * np.cos(latitude) / EARTH_RADIUS_M
    return f, beta


def compute_scales(c1_m_s: np.ndarray, eke_m2_s2: np.ndarray, latitude_deg: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate continuous Ld, Rhines scale, min scale, and 2-pi wavelength."""
    c1 = np.asarray(c1_m_s, dtype=np.float64)
    eke = np.asarray(eke_m2_s2, dtype=np.float64)
    f_1d, beta_1d = coriolis_parameters(latitude_deg)
    f = f_1d[:, None]
    beta = beta_1d[:, None]
    ld_m = c1 / np.sqrt(f**2 + 2.0 * beta * c1)
    valid_c1 = np.isfinite(c1) & (c1 > 0.0)
    valid_eke = np.isfinite(eke) & (eke >= 0.0)
    with np.errstate(invalid="ignore"):
        urms = np.sqrt(2.0 * eke)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_m = np.sqrt(2.0 * urms / beta)
    leddy_m = np.minimum(ld_m, lr_m)
    ld_m[~valid_c1] = np.nan
    lr_m[~valid_eke] = np.nan
    leddy_m[~(valid_c1 & valid_eke)] = np.nan
    return {
        "f_s_1": np.broadcast_to(f, c1.shape).copy(),
        "beta_m_1_s_1": np.broadcast_to(beta, c1.shape).copy(),
        "urms_m_s": urms,
        "Ld_km": ld_m / 1000.0,
        "Lrhines_km": lr_m / 1000.0,
        "Leddy_dyn_km": leddy_m / 1000.0,
        "Leddy_wavelength_km": TWO_PI * leddy_m / 1000.0,
    }
