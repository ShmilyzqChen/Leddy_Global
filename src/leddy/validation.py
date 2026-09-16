"""Numerical and identity checks for regional/global production outputs."""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "finite_cells": int(finite.size),
        "minimum": float(np.min(finite)) if finite.size else None,
        "p01": float(np.percentile(finite, 1)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p99": float(np.percentile(finite, 99)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def validate_output(config: dict) -> Path:
    """Validate physical signs, exact formula identities, flags, and coverage."""
    root = config["paths"]["work_root"]
    name = config["region"]["name"]
    source = root / "output" / f"leddy_{name}_025deg.nc"
    if not source.exists():
        raise FileNotFoundError(source)
    report: dict = {
        "source": f"output/{source.name}",
        "region": name,
        "checks": {},
        "fields": {},
    }
    with netCDF4.Dataset(source) as ds:
        keys = [
            "c1_eigen_m_s", "c1_wkb_m_s", "c1_final_corrected_m_s",
            "eke_corrected_m2_s2", "Ld_km", "Lrhines_km",
            "Leddy_dyn_km", "Leddy_wavelength_km",
        ]
        arrays = {key: np.asarray(ds.variables[key][:], dtype=np.float64) for key in keys}
        for key, values in arrays.items():
            report["fields"][key] = _summary(values)
        report["checks"]["all_c1_positive"] = bool(np.all(arrays["c1_final_corrected_m_s"][np.isfinite(arrays["c1_final_corrected_m_s"])] > 0.0))
        report["checks"]["all_eke_nonnegative"] = bool(np.all(arrays["eke_corrected_m2_s2"][np.isfinite(arrays["eke_corrected_m2_s2"])] >= 0.0))
        common = np.isfinite(arrays["Leddy_dyn_km"]) & np.isfinite(arrays["Leddy_wavelength_km"])
        wave_error = np.abs(arrays["Leddy_wavelength_km"][common] - 2.0 * np.pi * arrays["Leddy_dyn_km"][common])
        report["checks"]["maximum_2pi_identity_error_km"] = float(np.max(wave_error)) if wave_error.size else None
        common = np.isfinite(arrays["Leddy_dyn_km"]) & np.isfinite(arrays["Ld_km"]) & np.isfinite(arrays["Lrhines_km"])
        min_error = np.abs(arrays["Leddy_dyn_km"][common] - np.minimum(arrays["Ld_km"][common], arrays["Lrhines_km"][common]))
        report["checks"]["maximum_min_identity_error_km"] = float(np.max(min_error)) if min_error.size else None
        for flag_name in ("c1_method_flag", "c1_qc_flag", "eke_qc_flag"):
            flag = np.asarray(ds.variables[flag_name][:])
            unique, counts = np.unique(flag, return_counts=True)
            report["fields"][flag_name] = {str(int(key)): int(value) for key, value in zip(unique, counts)}

    failures = []
    if not report["checks"]["all_c1_positive"]:
        failures.append("nonpositive corrected c1")
    if not report["checks"]["all_eke_nonnegative"]:
        failures.append("negative corrected EKE")
    for key in ("maximum_2pi_identity_error_km", "maximum_min_identity_error_km"):
        value = report["checks"][key]
        if value is not None and value > 1.0e-3:
            failures.append(f"{key}={value}")
    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    output = root / "output" / f"validation_{name}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise ValueError(f"Validation failed: {failures}; see {output}")
    return output
