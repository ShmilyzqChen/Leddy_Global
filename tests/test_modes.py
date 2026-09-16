import numpy as np

from leddy.modes import first_baroclinic_phase_speed, solve_profile, wkb_phase_speed


def test_constant_stratification_matches_analytic_solution():
    depth = np.linspace(0.0, 4000.0, 101)
    buoyancy_frequency = 0.01
    n2 = np.full(depth.size - 1, buoyancy_frequency**2)
    expected = buoyancy_frequency * depth[-1] / np.pi
    assert np.isclose(wkb_phase_speed(depth, n2), expected, rtol=1.0e-12)
    assert np.isclose(first_baroclinic_phase_speed(depth, n2), expected, rtol=5.0e-4)


def test_wkb_fallback_when_too_much_profile_is_corrected():
    depth = np.linspace(0.0, 1000.0, 11)
    n2 = np.full(10, 1.0e-5)
    n2[:3] = np.nan
    result = solve_profile(depth, n2, 1.0e-8, maximum_corrected_fraction=0.20)
    assert result["corrected_fraction"] == 0.3
    assert result["c1_method_flag"] == 1
    assert np.isclose(result["c1_final_m_s"], result["c1_wkb_m_s"])


def test_profile_with_no_stratification_evidence_stays_invalid():
    depth = np.linspace(0.0, 1000.0, 11)
    result = solve_profile(depth, np.full(10, np.nan), 1.0e-8, 0.20)
    assert result["c1_method_flag"] == 255
    assert np.isnan(result["c1_final_m_s"])
