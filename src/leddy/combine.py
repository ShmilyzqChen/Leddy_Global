"""Combine c1 and EKE, apply conservative QC, and write final length scales."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

from .eke import combine_eke_years
from .qc import conservative_qc
from .scales import compute_scales
from .resources import guard_resources


def _copy_2d(source, destination, names: list[str]) -> None:
    for name in names:
        destination[name][:] = source[name][:]


def combine_final(config: dict, overwrite: bool = False) -> Path:
    """Create the documented, publication-ready 0.25-degree static dataset."""
    guard_resources(config, "final combination and QC")
    root = config["paths"]["work_root"]
    name = config["region"]["name"]
    strat_path = root / "intermediate" / f"stratification_{name}.nc"
    eke_path = combine_eke_years(config, overwrite=False)
    if not strat_path.exists():
        raise FileNotFoundError(f"Run stratification first: {strat_path}")
    output = root / "output" / f"leddy_{name}_025deg.nc"
    if output.exists() and not overwrite:
        return output

    with netCDF4.Dataset(strat_path) as strat, netCDF4.Dataset(eke_path) as eke:
        lat = np.asarray(strat.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(strat.variables["lon"][:], dtype=np.float64)
        if not np.allclose(lat, eke.variables["lat"][:]) or not np.allclose(lon, eke.variables["lon"][:]):
            raise ValueError("Stratification and EKE grids differ")
        bottom = np.asarray(strat.variables["bottom_depth_m"][:], dtype=np.float64)
        c1_raw = np.asarray(strat.variables["c1_final_m_s"][:], dtype=np.float64)
        eke_raw = np.asarray(eke.variables["eke_raw_m2_s2"][:], dtype=np.float64)
        # WOA is the target grid and supplies the authoritative land/sea mask.
        # DUACS can contain mapped values in a few WOA coastal-land cells.
        ocean = np.isfinite(bottom) & (bottom > 0.0)
        c1_raw[~ocean] = np.nan
        eke_raw[~ocean] = np.nan
        c1_corrected, c1_qc = conservative_qc(c1_raw, bottom, config["qc"])
        eke_corrected, eke_qc = conservative_qc(eke_raw, bottom, config["qc"])
        raw_scales = compute_scales(c1_raw, eke_raw, lat)
        final_scales = compute_scales(c1_corrected, eke_corrected, lat)

        temporary = output.with_suffix(".tmp.nc")
        if temporary.exists():
            temporary.unlink()
        with netCDF4.Dataset(temporary, "w", format="NETCDF4") as ds:
            ds.createDimension("lat", lat.size)
            ds.createDimension("lon", lon.size)
            for coord, values, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
                var = ds.createVariable(coord, "f8", (coord,))
                var[:] = values
                var.units = units
            chunks = (min(lat.size, 90), min(lon.size, 180))
            opts = dict(zlib=True, complevel=int(config["output"]["compression_level"]), shuffle=True, chunksizes=chunks)

            float_fields = [
                "bottom_depth_m", "n2_corrected_thickness_fraction", "c1_eigen_m_s",
                "c1_wkb_m_s", "c1_final_m_s", "c1_final_corrected_m_s",
                "eke_raw_m2_s2", "eke_corrected_m2_s2", "valid_day_fraction_min",
                "f_s_1", "beta_m_1_s_1", "urms_m_s", "Ld_raw_km", "Lrhines_raw_km",
                "Leddy_dyn_raw_km", "Leddy_wavelength_raw_km", "Ld_km", "Lrhines_km",
                "Leddy_dyn_km", "Leddy_wavelength_km",
            ]
            for field in float_fields:
                ds.createVariable(field, "f4", ("lat", "lon"), fill_value=np.float32(np.nan), **opts)
            for field, dtype in (("c1_method_flag", "u1"), ("c1_qc_flag", "u1"), ("eke_qc_flag", "u1"), ("valid_native_subcells", "u1"), ("valid_year_count_min", "u1"), ("valid_day_count_min", "u4")):
                ds.createVariable(field, dtype, ("lat", "lon"), **opts)

            ds.variables["c1_method_flag"].flag_values = np.array([0, 1, 255], dtype=np.uint8)
            ds.variables["c1_method_flag"].flag_meanings = "eigenmode wkb_fallback invalid"
            for field in ("c1_qc_flag", "eke_qc_flag"):
                ds.variables[field].flag_values = np.array([0, 1, 2, 255], dtype=np.uint8)
                ds.variables[field].flag_meanings = "unchanged isolated_outlier_replaced small_hole_filled invalid_or_unresolved"

            unit_map = {
                "bottom_depth_m": "m",
                "n2_corrected_thickness_fraction": "1",
                "c1_eigen_m_s": "m s-1",
                "c1_wkb_m_s": "m s-1",
                "c1_final_m_s": "m s-1",
                "c1_final_corrected_m_s": "m s-1",
                "eke_raw_m2_s2": "m2 s-2",
                "eke_corrected_m2_s2": "m2 s-2",
                "valid_day_fraction_min": "1",
                "f_s_1": "s-1",
                "beta_m_1_s_1": "m-1 s-1",
                "urms_m_s": "m s-1",
            }
            for field in ("Ld_raw_km", "Lrhines_raw_km", "Leddy_dyn_raw_km", "Leddy_wavelength_raw_km", "Ld_km", "Lrhines_km", "Leddy_dyn_km", "Leddy_wavelength_km"):
                unit_map[field] = "km"
            for field, units in unit_map.items():
                ds.variables[field].units = units

            long_name_map = {
                "bottom_depth_m": "effective ocean bottom depth",
                "n2_corrected_thickness_fraction": "fraction of water-column thickness with corrected buoyancy frequency",
                "c1_eigen_m_s": "first baroclinic phase speed from the numerical eigenmode",
                "c1_wkb_m_s": "first baroclinic phase speed from the WKB approximation",
                "c1_final_m_s": "selected first baroclinic phase speed before horizontal quality control",
                "c1_final_corrected_m_s": "selected first baroclinic phase speed after horizontal quality control",
                "eke_raw_m2_s2": "eddy kinetic energy before horizontal quality control",
                "eke_corrected_m2_s2": "eddy kinetic energy after horizontal quality control",
                "valid_day_fraction_min": "minimum valid-day fraction among contributing native cells",
                "f_s_1": "Coriolis parameter",
                "beta_m_1_s_1": "meridional gradient of the Coriolis parameter",
                "urms_m_s": "eddy velocity scale sqrt(2 EKE)",
                "Ld_raw_km": "deformation radius calculated from uncorrected phase speed",
                "Lrhines_raw_km": "Rhines scale calculated from uncorrected EKE",
                "Leddy_dyn_raw_km": "eddy dynamical length before horizontal quality control",
                "Leddy_wavelength_raw_km": "eddy wavelength before horizontal quality control",
                "Ld_km": "deformation radius calculated from corrected phase speed",
                "Lrhines_km": "Rhines scale calculated from corrected EKE",
                "Leddy_dyn_km": "eddy dynamical length min(Ld, Lrhines)",
                "Leddy_wavelength_km": "eddy wavelength 2 pi Leddy_dyn",
                "c1_method_flag": "phase-speed solution method flag",
                "c1_qc_flag": "phase-speed horizontal quality-control flag",
                "eke_qc_flag": "eddy-kinetic-energy horizontal quality-control flag",
                "valid_native_subcells": "number of contributing 0.125-degree cells",
                "valid_year_count_min": "minimum number of sampled years among contributing native cells",
                "valid_day_count_min": "minimum number of sampled days among contributing native cells",
            }
            for field, long_name in long_name_map.items():
                ds.variables[field].long_name = long_name
            ds.variables["valid_native_subcells"].units = "1"
            ds.variables["valid_year_count_min"].units = "years"
            ds.variables["valid_day_count_min"].units = "days"

            _copy_2d(strat.variables, ds.variables, ["bottom_depth_m", "n2_corrected_thickness_fraction", "c1_eigen_m_s", "c1_wkb_m_s", "c1_final_m_s", "c1_method_flag"])
            _copy_2d(eke.variables, ds.variables, ["eke_raw_m2_s2", "valid_day_fraction_min", "valid_native_subcells", "valid_year_count_min", "valid_day_count_min"])
            ds.variables["c1_final_corrected_m_s"][:] = c1_corrected.astype(np.float32)
            ds.variables["eke_corrected_m2_s2"][:] = eke_corrected.astype(np.float32)
            ds.variables["c1_qc_flag"][:] = c1_qc
            ds.variables["eke_qc_flag"][:] = eke_qc
            for field in ("f_s_1", "beta_m_1_s_1", "urms_m_s", "Ld_km", "Lrhines_km", "Leddy_dyn_km", "Leddy_wavelength_km"):
                ds.variables[field][:] = final_scales[field].astype(np.float32)
            raw_name_map = {
                "Ld_raw_km": "Ld_km",
                "Lrhines_raw_km": "Lrhines_km",
                "Leddy_dyn_raw_km": "Leddy_dyn_km",
                "Leddy_wavelength_raw_km": "Leddy_wavelength_km",
            }
            for destination, source in raw_name_map.items():
                ds.variables[destination][:] = raw_scales[source].astype(np.float32)

            ds.title = "Eddy length scales from WOA23 stratification and 1993-2024 DUACS EKE"
            ds.Conventions = "CF-1.10"
            ds.history = f"Created {datetime.now(timezone.utc).isoformat()}"
            ds.deformation_radius_formula = "Ld=c1/sqrt(f^2+2*beta*c1)"
            ds.rhines_formula = "Lrhines=sqrt(2*sqrt(2*EKE)/beta)"
            ds.eddy_scale_formula = "Leddy_dyn=min(Ld,Lrhines); Leddy_wavelength=2*pi*Leddy_dyn"
            ds.qc_flags = "0 unchanged, 1 isolated outlier replaced, 2 small hole filled, 255 invalid/unresolved"
            ds.source_temperature_salinity = "World Ocean Atlas 2023 annual objectively analysed 0.25-degree fields"
            ds.source_velocity = "CMEMS DUACS DT2024 daily anomalous geostrophic velocities, 1993-2024"
        temporary.replace(output)
    return output
