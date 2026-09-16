"""Compare a regional product with the matching subset of a global product."""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr


DEFAULT_VARIABLES = (
    "bottom_depth_m",
    "c1_final_m_s",
    "eke_raw_m2_s2",
    "Ld_raw_km",
    "Lrhines_raw_km",
    "Leddy_dyn_raw_km",
    "c1_final_corrected_m_s",
    "eke_corrected_m2_s2",
    "Ld_km",
    "Lrhines_km",
    "Leddy_dyn_km",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("global_product")
    parser.add_argument("regional_product")
    parser.add_argument(
        "--margin",
        type=int,
        default=2,
        help="Regional edge cells to omit (default: 2, matching half a 5x5 QC window).",
    )
    args = parser.parse_args()

    with xr.open_dataset(args.global_product) as global_ds, xr.open_dataset(
        args.regional_product
    ) as regional_ds:
        subset = global_ds.sel(lat=regional_ds.lat, lon=regional_ds.lon)
        edge = slice(args.margin, -args.margin or None)

        for variable in DEFAULT_VARIABLES:
            global_values = subset[variable].values[edge, edge]
            regional_values = regional_ds[variable].values[edge, edge]
            finite = np.isfinite(global_values) & np.isfinite(regional_values)
            difference = np.abs(global_values[finite] - regional_values[finite])
            nan_masks_equal = np.array_equal(
                np.isnan(global_values), np.isnan(regional_values)
            )
            median = float(np.median(difference)) if difference.size else np.nan
            maximum = float(np.max(difference)) if difference.size else np.nan
            print(
                f"{variable}: n={difference.size}, nan_mask_equal={nan_masks_equal}, "
                f"median_abs={median:.9g}, max_abs={maximum:.9g}"
            )


if __name__ == "__main__":
    main()
