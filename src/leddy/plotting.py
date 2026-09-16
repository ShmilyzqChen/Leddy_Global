"""Quick-look plots for regional and global quality control."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np


def plot_quicklook(config: dict) -> Path:
    root = config["paths"]["work_root"]
    name = config["region"]["name"]
    source = root / "output" / f"leddy_{name}_025deg.nc"
    if not source.exists():
        raise FileNotFoundError(source)
    with netCDF4.Dataset(source) as ds:
        lon = np.asarray(ds.variables["lon"][:])
        lat = np.asarray(ds.variables["lat"][:])
        fields = [
            ("c1_final_corrected_m_s", "c1 (m s$^{-1}$)"),
            ("Ld_km", "Ld (km)"),
            ("eke_corrected_m2_s2", "EKE (m$^2$ s$^{-2}$)"),
            ("Lrhines_km", "Rhines scale (km)"),
            ("Leddy_dyn_km", "Leddy dynamical (km)"),
            ("Leddy_wavelength_km", "2$\\pi$ Leddy wavelength (km)"),
        ]
        arrays = [(key, label, np.asarray(ds.variables[key][:])) for key, label in fields]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, label, values) in zip(axes.flat, arrays):
        finite = values[np.isfinite(values)]
        vmax = np.percentile(finite, 99) if finite.size else 1.0
        mesh = axis.pcolormesh(lon, lat, values, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        axis.set_title(label)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        figure.colorbar(mesh, ax=axis, shrink=0.8)
    output = root / "figures" / f"quicklook_{name}.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
