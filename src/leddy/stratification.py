"""WOA23 stratification and first-mode phase-speed pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import gsw
import netCDF4
import numpy as np

from .bathymetry import effective_bottom_depth, read_bottom_standard_level
from .config import ensure_output_directories
from .grid import select_regular_grid
from .modes import solve_profile
from .resources import guard_resources

LOGGER = logging.getLogger(__name__)


def _create_output(path: Path, depth_mid: np.ndarray, lat: np.ndarray, lon: np.ndarray, level: int):
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("depth_mid", depth_mid.size)
    ds.createDimension("lat", lat.size)
    ds.createDimension("lon", lon.size)
    for name, values, units in (
        ("depth_mid", depth_mid, "m"),
        ("lat", lat, "degrees_north"),
        ("lon", lon, "degrees_east"),
    ):
        var = ds.createVariable(name, "f8", (name,), zlib=True, complevel=level)
        var[:] = values
        var.units = units
    chunks3 = (min(depth_mid.size, 16), min(lat.size, 8), min(lon.size, 180))
    kwargs3 = dict(zlib=True, complevel=level, shuffle=True, fill_value=np.float32(np.nan), chunksizes=chunks3)
    kwargs2 = dict(zlib=True, complevel=level, shuffle=True, fill_value=np.float32(np.nan), chunksizes=(min(lat.size, 32), min(lon.size, 180)))
    n2_raw = ds.createVariable("n2_raw_s_2", "f4", ("depth_mid", "lat", "lon"), **kwargs3)
    n2_fixed = ds.createVariable("n2_corrected_s_2", "f4", ("depth_mid", "lat", "lon"), **kwargs3)
    n2_raw.units = "s-2"
    n2_fixed.units = "s-2"
    n2_flag = ds.createVariable("n2_correction_flag", "u1", ("depth_mid", "lat", "lon"), zlib=True, complevel=level, fill_value=np.uint8(255), chunksizes=chunks3)
    n2_flag.flag_values = np.array([0, 1, 255], dtype=np.uint8)
    n2_flag.flag_meanings = "unchanged corrected invalid_or_below_bottom"
    for name in ("bottom_depth_m", "n2_corrected_thickness_fraction", "c1_eigen_m_s", "c1_wkb_m_s", "c1_final_m_s"):
        ds.createVariable(name, "f4", ("lat", "lon"), **kwargs2)
    ds.variables["bottom_depth_m"].units = "m"
    ds.variables["n2_corrected_thickness_fraction"].units = "1"
    for name in ("c1_eigen_m_s", "c1_wkb_m_s", "c1_final_m_s"):
        ds.variables[name].units = "m s-1"
    flag = ds.createVariable("c1_method_flag", "u1", ("lat", "lon"), zlib=True, complevel=level, fill_value=np.uint8(255), chunksizes=kwargs2["chunksizes"])
    flag.flag_values = np.array([0, 1, 255], dtype=np.uint8)
    flag.flag_meanings = "eigenmode wkb_fallback invalid"
    ds.title = "WOA23 first-baroclinic phase speed on the native nonuniform vertical grid"
    ds.n2_processing = "TEOS-10; nonpositive/missing values interpolated in log(N2), then floored"
    ds.vertical_mode = "nonuniform linear finite elements, rigid-lid and flat-bottom boundaries"
    ds.complete = "false"
    return ds


def compute_stratification(config: dict, overwrite: bool = False) -> Path:
    """Compute regional or global N2 and c1 fields in latitude blocks."""
    ensure_output_directories(config)
    guard_resources(config, "stratification startup")
    paths = config["paths"]
    region = config["region"]
    settings = config["stratification"]
    out = paths["work_root"] / "intermediate" / f"stratification_{region['name']}.nc"
    if out.exists() and not overwrite:
        with netCDF4.Dataset(out) as existing:
            if getattr(existing, "complete", "false") == "true":
                LOGGER.info("Using existing completed stratification file: %s", out)
                return out
        LOGGER.warning("Discarding incomplete stratification file: %s", out)
    working = out.with_suffix(".tmp.nc")
    if out.exists():
        out.unlink()
    if working.exists():
        working.unlink()

    with netCDF4.Dataset(paths["woa_temperature"]) as tds, netCDF4.Dataset(paths["woa_salinity"]) as sds:
        depth = np.asarray(tds.variables["depth"][:], dtype=np.float64)
        all_lat = np.asarray(tds.variables["lat"][:], dtype=np.float64)
        all_lon = np.asarray(tds.variables["lon"][:], dtype=np.float64)
        selection = select_regular_grid(all_lat, all_lon, region)
        bottom_code = read_bottom_standard_level(paths["woa_bottom_mask"])[selection.lat_slice, selection.lon_slice]
        bottom = effective_bottom_depth(bottom_code, depth, settings["maximum_depth_m"])
        depth_mid = 0.5 * (depth[:-1] + depth[1:])

        ods = _create_output(working, depth_mid, selection.lat, selection.lon, int(config["output"]["compression_level"]))
        ods.variables["bottom_depth_m"][:] = bottom.astype(np.float32)
        block_size = int(config["resources"]["woa_latitude_block_rows"])
        estimated_block_gb = depth.size * block_size * selection.lon.size * 64.0 / 1024**3 + 0.10
        LOGGER.info(
            "WOA latitude block rows=%d; conservative array peak estimate %.2f GB",
            block_size,
            estimated_block_gb,
        )
        try:
            for y0 in range(0, selection.lat.size, block_size):
                guard_resources(config, f"stratification latitude row {y0}")
                y1 = min(y0 + block_size, selection.lat.size)
                source_y = slice(selection.lat_slice.start + y0, selection.lat_slice.start + y1)
                source_x = selection.lon_slice
                temp = np.ma.filled(tds.variables["t_an"][0, :, source_y, source_x], np.nan).astype(np.float64)
                salt = np.ma.filled(sds.variables["s_an"][0, :, source_y, source_x], np.nan).astype(np.float64)
                lat_block = selection.lat[y0:y1]
                lon_block = selection.lon
                pressure = gsw.p_from_z(-depth[:, None, None], lat_block[None, :, None])
                sa = gsw.SA_from_SP(salt, pressure, lon_block[None, None, :], lat_block[None, :, None])
                ct = gsw.CT_from_t(sa, temp, pressure)
                n2_raw, _ = gsw.Nsquared(sa, ct, pressure, lat=lat_block[None, :, None], axis=0)
                n2_raw = np.asarray(n2_raw, dtype=np.float64)
                n2_corrected = np.full_like(n2_raw, np.nan)
                correction_flag = np.full(n2_raw.shape, 255, dtype=np.uint8)
                shape2 = (y1 - y0, selection.lon.size)
                c_eigen = np.full(shape2, np.nan, dtype=np.float32)
                c_wkb = np.full(shape2, np.nan, dtype=np.float32)
                c_final = np.full(shape2, np.nan, dtype=np.float32)
                corrected_fraction = np.full(shape2, np.nan, dtype=np.float32)
                method = np.full(shape2, 255, dtype=np.uint8)

                for jj in range(shape2[0]):
                    for ii in range(shape2[1]):
                        h_bottom = bottom[y0 + jj, ii]
                        paired = np.isfinite(temp[:, jj, ii]) & np.isfinite(salt[:, jj, ii]) & (depth <= h_bottom)
                        index = np.flatnonzero(paired)
                        if index.size < int(settings["minimum_profile_levels"]):
                            continue
                        # WOA profiles are bottom-contiguous in practice. Retaining valid
                        # standard levels also handles an isolated missing level as a wider element.
                        z_nodes = depth[index]
                        if z_nodes[0] > 0.0:
                            z_nodes = np.insert(z_nodes, 0, 0.0)
                        if h_bottom > z_nodes[-1]:
                            z_nodes = np.append(z_nodes, h_bottom)

                        raw_standard = n2_raw[:, jj, ii]
                        element_mid = 0.5 * (z_nodes[:-1] + z_nodes[1:])
                        # Most retained elements coincide exactly with a WOA
                        # standard-depth interval. Nearest mapping preserves an
                        # invalid/unstable source value so solve_profile can
                        # count and flag the correction instead of hiding it.
                        nearest = np.abs(depth_mid[:, None] - element_mid[None, :]).argmin(axis=0)
                        raw_elements = raw_standard[nearest]
                        # The final element is the explicitly added cap between
                        # the deepest observed standard level and the effective
                        # bottom. Extrapolating the last observed N2 over this
                        # short cap is part of the boundary construction, not an
                        # unstable-profile correction, so do it before QC.
                        if z_nodes[-1] == h_bottom and raw_elements.size > 1 and not (
                            np.isfinite(raw_elements[-1]) and raw_elements[-1] > 0.0
                        ):
                            previous = np.flatnonzero(np.isfinite(raw_elements[:-1]) & (raw_elements[:-1] > 0.0))
                            if previous.size:
                                raw_elements[-1] = raw_elements[previous[-1]]
                        result = solve_profile(
                            z_nodes,
                            raw_elements,
                            float(settings["n2_floor_s_2"]),
                            float(settings["maximum_corrected_thickness_fraction"]),
                        )
                        c_eigen[jj, ii] = result["c1_eigen_m_s"]
                        c_wkb[jj, ii] = result["c1_wkb_m_s"]
                        c_final[jj, ii] = result["c1_final_m_s"]
                        corrected_fraction[jj, ii] = result["corrected_fraction"]
                        method[jj, ii] = result["c1_method_flag"]

                        wet_mid = depth_mid < h_bottom
                        raw_wet = raw_standard[wet_mid]
                        if raw_wet.size:
                            z_wet = depth_mid[wet_mid]
                            valid = np.isfinite(raw_wet) & (raw_wet > 0.0)
                            if valid.any():
                                fixed = np.exp(np.interp(z_wet, z_wet[valid], np.log(raw_wet[valid])))
                                fixed = np.maximum(fixed, float(settings["n2_floor_s_2"]))
                                n2_corrected[wet_mid, jj, ii] = fixed
                                correction_flag[wet_mid, jj, ii] = (
                                    (~valid) | (raw_wet < float(settings["n2_floor_s_2"]))
                                ).astype(np.uint8)

                ods.variables["n2_raw_s_2"][:, y0:y1, :] = n2_raw.astype(np.float32)
                ods.variables["n2_corrected_s_2"][:, y0:y1, :] = n2_corrected.astype(np.float32)
                ods.variables["n2_correction_flag"][:, y0:y1, :] = correction_flag
                ods.variables["n2_corrected_thickness_fraction"][y0:y1, :] = corrected_fraction
                ods.variables["c1_eigen_m_s"][y0:y1, :] = c_eigen
                ods.variables["c1_wkb_m_s"][y0:y1, :] = c_wkb
                ods.variables["c1_final_m_s"][y0:y1, :] = c_final
                ods.variables["c1_method_flag"][y0:y1, :] = method
                ods.sync()
                LOGGER.info("Stratification latitude rows %d:%d of %d", y0, y1, selection.lat.size)
            ods.complete = "true"
        finally:
            ods.close()
    working.replace(out)
    return out
