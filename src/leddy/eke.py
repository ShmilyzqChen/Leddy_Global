"""EKE climatology from daily DUACS anomalous geostrophic velocities."""

from __future__ import annotations

from datetime import date, datetime
import logging
from pathlib import Path
import re

import netCDF4
import numpy as np

from .config import ensure_output_directories
from .grid import select_regular_grid, verify_two_to_one_alignment, woa_to_duacs_slices
from .resources import guard_resources

LOGGER = logging.getLogger(__name__)
DATE_PATTERN = re.compile(r"(?<!\d)(19\d{6}|20\d{6})(?!\d)")
OBSERVATION_DATE_PATTERN = re.compile(r"phy_l4_(\d{8})_")


def dated_files(root: Path, start: date, end: date) -> list[tuple[date, Path]]:
    """Discover one dated NetCDF per day and reject duplicates."""
    found: dict[date, Path] = {}
    for path in root.rglob("*.nc"):
        primary = OBSERVATION_DATE_PATTERN.search(path.name)
        matches = [primary.group(1)] if primary else DATE_PATTERN.findall(path.name)
        if not matches:
            continue
        candidates = []
        for token in matches:
            try:
                candidates.append(datetime.strptime(token, "%Y%m%d").date())
            except ValueError:
                pass
        candidates = [item for item in candidates if start <= item <= end]
        if not candidates:
            continue
        # The first date is the observation date. A second filename date is
        # commonly the delayed-time production/version stamp.
        stamp = candidates[0]
        if stamp in found:
            raise ValueError(f"Duplicate DUACS date {stamp}: {found[stamp]} and {path}")
        found[stamp] = path
    return sorted(found.items())


def _raw_slice(variable, ys: slice, xs: slice) -> tuple[np.ndarray, float | None]:
    variable.set_auto_maskandscale(False)
    raw = np.asarray(variable[0, ys, xs])
    fill = getattr(variable, "_FillValue", None)
    return raw, fill


def _unpack(raw: np.ndarray, variable) -> np.ndarray:
    scale = float(getattr(variable, "scale_factor", 1.0))
    offset = float(getattr(variable, "add_offset", 0.0))
    return raw.astype(np.float32) * scale + offset


def _write_year(
    path: Path,
    year: int,
    lat: np.ndarray,
    lon: np.ndarray,
    ke_sum: np.ndarray,
    count: np.ndarray,
    expected_days: int,
    compression: int,
) -> None:
    temporary = path.with_suffix(".tmp.nc")
    if temporary.exists():
        temporary.unlink()
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        for name, values, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            var = ds.createVariable(name, "f8", (name,))
            var[:] = values
            var.units = units
        chunks = (min(lat.size, 180), min(lon.size, 360))
        total = ds.createVariable("ke_sum_m2_s2", "f8", ("lat", "lon"), zlib=True, complevel=compression, shuffle=True, chunksizes=chunks)
        valid = ds.createVariable("valid_day_count", "u2", ("lat", "lon"), zlib=True, complevel=compression, shuffle=True, chunksizes=chunks)
        total[:] = ke_sum
        valid[:] = count
        ds.year = year
        ds.expected_daily_files = expected_days
        ds.formula = "sum over ice-free valid days of 0.5*(ugosa^2+vgosa^2)"
        ds.complete = "true"
    temporary.replace(path)


def compute_eke_years(config: dict, overwrite: bool = False) -> list[Path]:
    """Accumulate daily EKE sums/counts into restartable native-grid annual files."""
    ensure_output_directories(config)
    paths = config["paths"]
    settings = config["eke"]
    start = datetime.strptime(config["period"]["start"], "%Y-%m-%d").date()
    end = datetime.strptime(config["period"]["end"], "%Y-%m-%d").date()
    files = dated_files(paths["cmems_daily_root"], start, end)
    if not files:
        raise FileNotFoundError(f"No dated NetCDF files found in {paths['cmems_daily_root']}")

    with netCDF4.Dataset(files[0][1]) as sample, netCDF4.Dataset(paths["woa_temperature"]) as woa:
        duacs_lat_all = np.asarray(sample.variables["latitude"][:], dtype=np.float64)
        duacs_lon_all = np.asarray(sample.variables["longitude"][:], dtype=np.float64)
        woa_lat_all = np.asarray(woa.variables["lat"][:], dtype=np.float64)
        woa_lon_all = np.asarray(woa.variables["lon"][:], dtype=np.float64)
        verify_two_to_one_alignment(woa_lat_all, woa_lon_all, duacs_lat_all, duacs_lon_all)
        selection = select_regular_grid(woa_lat_all, woa_lon_all, config["region"])
        native_ys, native_xs = woa_to_duacs_slices(selection)
        native_lat = duacs_lat_all[native_ys]
        native_lon = duacs_lon_all[native_xs]
        cells = native_lat.size * native_lon.size
        estimated_peak_gb = cells * 64.0 / 1024**3 + 0.10
        LOGGER.info(
            "EKE native grid %dx%d; conservative Python-array peak estimate %.2f GB",
            native_lat.size,
            native_lon.size,
            estimated_peak_gb,
        )

    by_year: dict[int, list[tuple[date, Path]]] = {}
    for item in files:
        by_year.setdefault(item[0].year, []).append(item)
    outputs: list[Path] = []
    compression = int(config["output"]["compression_level"])
    annual_root = paths["work_root"] / "intermediate" / "eke_yearly" / config["region"]["name"]
    annual_root.mkdir(parents=True, exist_ok=True)

    for year in range(start.year, end.year + 1):
        guard_resources(config, f"EKE year {year}")
        year_files = by_year.get(year, [])
        if not year_files:
            raise FileNotFoundError(f"No DUACS files for required year {year}")
        output = annual_root / f"eke_native_{config['region']['name']}_{year}.nc"
        if output.exists() and not overwrite:
            with netCDF4.Dataset(output) as existing:
                if getattr(existing, "complete", "false") == "true" and int(existing.year) == year:
                    LOGGER.info("Skipping completed EKE year %d", year)
                    outputs.append(output)
                    continue

        ke_sum = np.zeros((native_lat.size, native_lon.size), dtype=np.float64)
        count = np.zeros((native_lat.size, native_lon.size), dtype=np.uint16)
        for index, (stamp, path) in enumerate(year_files, start=1):
            with netCDF4.Dataset(path) as ds:
                uvar = ds.variables[settings["u_variable"]]
                vvar = ds.variables[settings["v_variable"]]
                ivar = ds.variables[settings["ice_variable"]]
                u_raw, u_fill = _raw_slice(uvar, native_ys, native_xs)
                v_raw, v_fill = _raw_slice(vvar, native_ys, native_xs)
                ice_raw, ice_fill = _raw_slice(ivar, native_ys, native_xs)
                valid = np.ones(u_raw.shape, dtype=bool)
                if u_fill is not None:
                    valid &= u_raw != u_fill
                if v_fill is not None:
                    valid &= v_raw != v_fill
                if ice_fill is not None:
                    valid &= ice_raw != ice_fill
                valid &= ice_raw != int(settings["exclude_ice_flag"])
                u = _unpack(u_raw, uvar)
                v = _unpack(v_raw, vvar)
                # Use float64 for both the daily square and the multi-year sum
                # so regional and global runs are bitwise consistent.
                ke = 0.5 * (u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
                ke_sum[valid] += ke[valid]
                count[valid] += 1
            if index % 50 == 0 or index == len(year_files):
                LOGGER.info("EKE %d: %d/%d daily files", year, index, len(year_files))
                guard_resources(config, f"EKE year {year}, day {index}")
        _write_year(output, year, native_lat, native_lon, ke_sum, count, len(year_files), compression)
        outputs.append(output)
    return outputs


def combine_eke_years(config: dict, overwrite: bool = False) -> Path:
    """Combine annual native-grid accumulators and average each aligned 2x2 block."""
    guard_resources(config, "combine EKE years")
    paths = config["paths"]
    region_name = config["region"]["name"]
    start_year = int(config["period"]["start"][:4])
    end_year = int(config["period"]["end"][:4])
    annual_root = paths["work_root"] / "intermediate" / "eke_yearly" / region_name
    annual = [annual_root / f"eke_native_{region_name}_{year}.nc" for year in range(start_year, end_year + 1)]
    missing = [path for path in annual if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} annual EKE files; first is {missing[0]}")
    output = paths["work_root"] / "intermediate" / f"eke_climatology_{region_name}.nc"
    if output.exists() and not overwrite:
        return output

    total_sum = total_count = valid_years = None
    total_expected_days = 0
    lat_native = lon_native = None
    for path in annual:
        with netCDF4.Dataset(path) as ds:
            if total_sum is None:
                lat_native = np.asarray(ds.variables["lat"][:])
                lon_native = np.asarray(ds.variables["lon"][:])
                shape = (lat_native.size, lon_native.size)
                total_sum = np.zeros(shape, dtype=np.float64)
                total_count = np.zeros(shape, dtype=np.uint32)
                valid_years = np.zeros(shape, dtype=np.uint8)
            yearly_count = np.asarray(ds.variables["valid_day_count"][:], dtype=np.uint32)
            total_sum += np.asarray(ds.variables["ke_sum_m2_s2"][:], dtype=np.float64)
            total_count += yearly_count
            valid_years += yearly_count > 0
            total_expected_days += int(ds.expected_daily_files)

    native_eke = np.full(total_sum.shape, np.nan, dtype=np.float64)
    eligible = (total_count >= int(config["eke"]["minimum_valid_days"])) & (valid_years >= int(config["eke"]["minimum_valid_years"]))
    native_eke[eligible] = total_sum[eligible] / total_count[eligible]
    if native_eke.shape[0] % 2 or native_eke.shape[1] % 2:
        raise ValueError("Native regional grid must contain complete 2x2 DUACS blocks")

    ny, nx = native_eke.shape[0] // 2, native_eke.shape[1] // 2
    block_eke = native_eke.reshape(ny, 2, nx, 2).transpose(0, 2, 1, 3).reshape(ny, nx, 4)
    block_count = total_count.reshape(ny, 2, nx, 2).transpose(0, 2, 1, 3).reshape(ny, nx, 4)
    block_years = valid_years.reshape(ny, 2, nx, 2).transpose(0, 2, 1, 3).reshape(ny, nx, 4)
    subcells = np.sum(np.isfinite(block_eke), axis=2).astype(np.uint8)
    eke = np.divide(
        np.nansum(block_eke, axis=2),
        subcells,
        out=np.full((ny, nx), np.nan, dtype=np.float64),
        where=subcells > 0,
    )
    count_min = np.min(np.where(np.isfinite(block_eke), block_count, np.iinfo(np.uint32).max), axis=2)
    count_min[subcells == 0] = 0
    years_min = np.min(np.where(np.isfinite(block_eke), block_years, np.iinfo(np.uint8).max), axis=2)
    years_min[subcells == 0] = 0
    coverage_min = count_min.astype(np.float64) / total_expected_days
    lat = lat_native.reshape(ny, 2).mean(axis=1)
    lon = lon_native.reshape(nx, 2).mean(axis=1)

    temporary = output.with_suffix(".tmp.nc")
    if temporary.exists():
        temporary.unlink()
    with netCDF4.Dataset(temporary, "w", format="NETCDF4") as ds:
        ds.createDimension("lat", ny)
        ds.createDimension("lon", nx)
        for name, values, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            var = ds.createVariable(name, "f8", (name,))
            var[:] = values
            var.units = units
        chunks = (min(ny, 90), min(nx, 180))
        opts = dict(zlib=True, complevel=int(config["output"]["compression_level"]), shuffle=True, chunksizes=chunks)
        ds.createVariable("eke_raw_m2_s2", "f4", ("lat", "lon"), fill_value=np.float32(np.nan), **opts)[:] = eke.astype(np.float32)
        ds.createVariable("valid_native_subcells", "u1", ("lat", "lon"), **opts)[:] = subcells
        ds.createVariable("valid_day_count_min", "u4", ("lat", "lon"), **opts)[:] = count_min
        ds.createVariable("valid_year_count_min", "u1", ("lat", "lon"), **opts)[:] = years_min
        ds.createVariable("valid_day_fraction_min", "f4", ("lat", "lon"), **opts)[:] = coverage_min.astype(np.float32)
        ds.variables["eke_raw_m2_s2"].units = "m2 s-2"
        ds.variables["valid_day_fraction_min"].units = "1"
        ds.period = f"{config['period']['start']} through {config['period']['end']}"
        ds.formula = "0.5*mean(ugosa^2+vgosa^2), ice-flagged dates excluded"
        ds.spatial_aggregation = "equal mean of eligible native 0.125-degree cell climatologies in each 2x2 block"
    temporary.replace(output)
    return output
