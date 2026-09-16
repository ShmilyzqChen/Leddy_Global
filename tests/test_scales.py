import numpy as np

from leddy.constants import EARTH_ANGULAR_VELOCITY_S_1, EARTH_RADIUS_M
from leddy.scales import compute_scales


def test_equatorial_deformation_radius_is_finite_and_continuous_formula():
    c1 = np.array([[2.0]])
    eke = np.array([[0.01]])
    result = compute_scales(c1, eke, np.array([0.0]))
    beta = 2.0 * EARTH_ANGULAR_VELOCITY_S_1 / EARTH_RADIUS_M
    expected_km = np.sqrt(2.0 / (2.0 * beta)) / 1000.0
    assert np.isclose(result["Ld_km"][0, 0], expected_km)
    assert np.isclose(
        result["Leddy_wavelength_km"][0, 0],
        2.0 * np.pi * result["Leddy_dyn_km"][0, 0],
    )


def test_deformation_radius_is_retained_when_eke_is_missing():
    result = compute_scales(np.array([[2.0]]), np.array([[np.nan]]), np.array([30.0]))
    assert np.isfinite(result["Ld_km"][0, 0])
    assert np.isnan(result["Lrhines_km"][0, 0])
    assert np.isnan(result["Leddy_dyn_km"][0, 0])


def test_rhines_uses_rhines_1975_coefficient_and_rms_from_eke():
    c1 = np.array([[2.0]])
    eke = np.array([[0.01]])
    latitude = np.array([30.0])
    result = compute_scales(c1, eke, latitude)
    beta = 2.0 * EARTH_ANGULAR_VELOCITY_S_1 * np.cos(np.deg2rad(30.0)) / EARTH_RADIUS_M
    u_rms = np.sqrt(2.0 * 0.01)
    expected_km = np.sqrt(2.0 * u_rms / beta) / 1000.0
    assert np.isclose(result["urms_m_s"][0, 0], u_rms)
    assert np.isclose(result["Lrhines_km"][0, 0], expected_km)
