"""Six-panel 180-degree-centered global eddy-scale contourf/contour map.

Filled contours use the original 0.25-degree product. Regular lines and broad
over-range lines use 1-degree averages solely for visual readability, never for
computation; tiny exceedances near the plotting limit use the native grid.
The coastline is the boundary of the WOA23 land/sea mask used by this product.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import patheffects
import numpy as np
from PIL import Image
import psutil
import xarray as xr


@dataclass(frozen=True)
class Panel:
    variable: str
    title: str
    color_levels: tuple[float, ...]
    contour_levels: tuple[float, ...]
    colorbar_ticks: tuple[float, ...]
    contour_format: str
    extend: str


PANELS = (
    Panel(
        "c1_final_corrected_m_s", r"$c_1$ (m s$^{-1}$)",
        (1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5),
        (1.5, 2, 2.5, 3), (1, 1.5, 2, 2.5, 3, 3.5), "%.1f", "both",
    ),
    Panel(
        "Ld_km", r"$L_d$ (km)",
        (20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200),
        (40, 80, 120, 160), (20, 40, 60, 100, 150, 200), "%d", "both",
    ),
    Panel(
        "eke_corrected_m2_s2", r"EKE (m$^2$ s$^{-2}$)",
        (0, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1),
        (0.01, 0.03, 0.06), (0, 0.01, 0.02, 0.05, 0.1), "%.2f", "max",
    ),
    Panel(
        "Lrhines_km", "Rhines scale (km)",
        (20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200),
        (40, 80, 120, 160), (20, 40, 60, 100, 150, 200), "%d", "both",
    ),
    Panel(
        "Leddy_dyn_km", r"$L_{\mathrm{eddy}}$ dynamical (km)",
        (20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200),
        (40, 80, 120, 160), (20, 40, 60, 100, 150, 200), "%d", "both",
    ),
    Panel(
        "Leddy_wavelength_km", r"$2\pi L_{\mathrm{eddy}}$ wavelength (km)",
        (0, 50, 100, 150, 200, 250, 300, 375, 450, 500),
        (100, 200, 300, 400), (0, 100, 200, 300, 400, 500), "%d", "max",
    ),
)

OVER_RANGE_CONTOUR_STEP = {
    "c1_final_corrected_m_s": 0.1,
    "Ld_km": 25.0,
    "eke_corrected_m2_s2": 0.1,
    "Lrhines_km": 50.0,
    "Leddy_dyn_km": 10.0,
    "Leddy_wavelength_km": 100.0,
}

# Basin-distributed contour-label anchors avoid dozens of duplicate labels
# along energetic currents and complex coasts.
LABEL_ANCHORS = (
    (-155, 45), (-155, 0), (-125, -32), (-105, -58),
    (-55, 35), (-32, 2), (-20, -28), (-35, -53),
    (58, -48), (78, -17), (72, 13), (155, 36),
    (157, -12), (155, -53),
)


def coarse_nanmean(values: np.ndarray, factor: int = 4) -> np.ndarray:
    """Aligned 0.25-to-1-degree mean without warnings for all-missing cells."""
    ny, nx = values.shape
    if ny % factor or nx % factor:
        raise ValueError("Input grid cannot be coarsened by the chosen factor")
    blocks = values.reshape(ny // factor, factor, nx // factor, factor)
    finite = np.isfinite(blocks)
    summed = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    count = finite.sum(axis=(1, 3))
    return np.divide(
        summed, count, out=np.full(summed.shape, np.nan), where=count > 0
    )


def fine_color_levels(panel: Panel, subdivisions: int = 3) -> np.ndarray:
    """Densify the user's unequal color intervals without changing clim."""
    knots = np.asarray(panel.color_levels, dtype=np.float64)
    pieces = [
        np.linspace(left, right, subdivisions + 1)[:-1]
        for left, right in zip(knots[:-1], knots[1:])
    ]
    return np.concatenate((*pieces, knots[-1:]))


def panel_colormap(panel: Panel, levels: np.ndarray) -> tuple[mpl.colors.Colormap, BoundaryNorm]:
    n_bins = len(levels) - 1
    cmap = mpl.colormaps["jet"].resampled(n_bins).copy()
    cmap.set_under("#082567")
    cmap.set_over("#a80000")
    cmap.set_bad((1, 1, 1, 0))
    norm = BoundaryNorm(levels, cmap.N, clip=False)
    return cmap, norm


def rotate_and_close(longitude: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 0-360-degree longitude with two cyclic edge columns.

    The resulting plot has 180 degrees at the midpoint and continuous values
    at the 0/360-degree seam.
    """
    east_longitude = np.mod(longitude, 360.0)
    order = np.argsort(east_longitude)
    sorted_lon = east_longitude[order]
    sorted_values = values[:, order]
    step = float(np.median(np.diff(sorted_lon)))
    cyclic_lon = np.concatenate(
        ([sorted_lon[0] - step], sorted_lon, [sorted_lon[-1] + step])
    )
    cyclic_values = np.concatenate(
        (sorted_values[:, -1:], sorted_values, sorted_values[:, :1]), axis=1
    )
    return cyclic_lon, cyclic_values


def check_plot_memory(panel_name: str) -> None:
    available_gb = psutil.virtual_memory().available / 1024**3
    used_gb = psutil.Process().memory_info().rss / 1024**3
    print(f"{panel_name}: process RAM {used_gb:.2f} GB; free RAM {available_gb:.2f} GB", flush=True)
    if available_gb < 3.0:
        raise MemoryError("Free RAM below 3 GB during contourf plotting")


def plot(source: Path, output_stem: Path, dpi: int = 300) -> tuple[Path, Path]:
    with xr.open_dataset(source) as ds:
        rhines_is_sqrt2 = "2*sqrt(2*EKE)" in ds.attrs.get("rhines_formula", "")
        longitude = np.asarray(ds.lon.values, dtype=np.float64)
        latitude = np.asarray(ds.lat.values, dtype=np.float64)
        bottom = np.asarray(ds.bottom_depth_m.values, dtype=np.float64)
        fields = {
            panel.variable: np.asarray(ds[panel.variable].values, dtype=np.float64)
            for panel in PANELS
        }
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.5), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.12, wspace=0.06, hspace=0.10)

    longitude_360, land = rotate_and_close(
        longitude, (~np.isfinite(bottom)).astype(np.float32)
    )
    land_overlay = np.ma.masked_where(land < 0.5, land)
    land_cmap = mpl.colors.ListedColormap(["#f2f2f2"])
    coarse_lat = latitude.reshape(-1, 4).mean(axis=1)
    label_anchors_360 = tuple((lon % 360, lat) for lon, lat in LABEL_ANCHORS)

    # Column-major panel order: a/b in the left column, c/d in the middle,
    # e/f in the right column.
    for index, panel in enumerate(PANELS):
        row, column = index % 2, index // 2
        axis = axes[row, column]
        check_plot_memory(f"before {panel.variable}")
        _, values = rotate_and_close(longitude, fields[panel.variable])
        levels = fine_color_levels(panel)
        cmap, norm = panel_colormap(panel, levels)
        filled = axis.contourf(
            longitude_360, latitude, np.ma.masked_invalid(values),
            levels=levels, cmap=cmap, norm=norm, extend=panel.extend,
            alpha=0.92, antialiased=False, zorder=1,
        )
        filled.set_rasterized(True)

        # Display-only 1-degree contours: enough to convey basin-scale
        # structure without the visual noise of individual 0.25-degree cells.
        contour_values = coarse_nanmean(values[:, 1:-1])
        coarse_lon = longitude_360[1:-1].reshape(-1, 4).mean(axis=1)
        coarse_lon = np.concatenate(([-0.5], coarse_lon, [360.5]))
        contour_values = np.concatenate(
            (contour_values[:, -1:], contour_values, contour_values[:, :1]),
            axis=1,
        )
        contours = axis.contour(
            coarse_lon, coarse_lat, np.ma.masked_invalid(contour_values),
            levels=panel.contour_levels, colors="#242424", linewidths=0.58,
            alpha=0.64, zorder=2,
        )
        labels = axis.clabel(
            contours, contours.levels, fmt=panel.contour_format,
            inline=True, inline_spacing=3, fontsize=6.2,
            manual=label_anchors_360,
        )
        for label in labels:
            label.set_path_effects(
                [patheffects.withStroke(linewidth=1.8, foreground="white")]
            )

        # Contours above the colorbar limit retain actual numbers (for example
        # 500, 600, 700 km) even though their fill color is saturated. Use the
        # same display-only 1-degree field as the regular contour lines for
        # broad features; tiny near-limit exceedances need the native grid.
        upper = float(panel.color_levels[-1])
        maximum = float(np.nanmax(values))
        if maximum > upper:
            use_native = panel.variable in ("c1_final_corrected_m_s", "Leddy_dyn_km")
            over_lon = longitude_360 if use_native else coarse_lon
            over_lat = latitude if use_native else coarse_lat
            over_values = values if use_native else contour_values
            step = OVER_RANGE_CONTOUR_STEP[panel.variable]
            over_maximum = float(np.nanmax(over_values))
            over_levels = np.arange(upper, over_maximum + step, step)
            over_contour = axis.contour(
                over_lon, over_lat, np.ma.masked_invalid(over_values),
                levels=over_levels, colors="#fff7e6", linewidths=0.85,
                alpha=0.96, zorder=2.5,
            )
            over_labels = axis.clabel(
                over_contour, fmt=panel.contour_format, inline=True,
                inline_spacing=2, fontsize=6.0, colors="#fff7e6",
            )
            for label in over_labels:
                label.set_path_effects(
                    [patheffects.withStroke(linewidth=1.8, foreground="#4a180e")]
                )

        # The same WOA23 mask that defines the target ocean grid also defines
        # the land fill and land/ocean border. No cartographic mismatch occurs.
        axis.pcolormesh(
            longitude_360, latitude, land_overlay, shading="auto",
            cmap=land_cmap, vmin=0.5, vmax=1.5, rasterized=True, zorder=3,
        )
        axis.contour(
            longitude_360, latitude, land, levels=[0.5],
            colors="#343434", linewidths=0.58, zorder=4,
        )

        axis.set_xlim(0, 360)
        axis.set_ylim(-85, 85)
        axis.set_xticks((0, 60, 120, 180, 240, 300, 360))
        axis.set_xticklabels(("0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"))
        axis.set_yticks((-60, -30, 0, 30, 60))
        axis.tick_params(length=2.4, width=0.55, labelsize=7)
        axis.grid(color="#777777", alpha=0.13, linewidth=0.4, zorder=0)
        title = panel.title
        if rhines_is_sqrt2 and panel.variable == "Lrhines_km":
            title = r"Rhines scale, $\sqrt{2U_{\rm rms}/\beta}$ (km)"
        axis.set_title(f"{chr(ord('a') + index)}  {title}", loc="left", pad=5, fontsize=10)
        if row == 1:
            axis.set_xlabel("Longitude (°)", fontsize=8)
        if column == 0:
            axis.set_ylabel("Latitude (°)", fontsize=8)

        colorbar = fig.colorbar(
            filled, ax=axis, ticks=panel.colorbar_ticks,
            extend=panel.extend, extendfrac=0.045,
            spacing="proportional", fraction=0.043, pad=0.012, shrink=0.88,
        )
        colorbar.ax.tick_params(labelsize=6.5, length=2, width=0.5)
        colorbar.outline.set_linewidth(0.55)
        if panel.variable == "eke_corrected_m2_s2":
            colorbar.ax.set_yticklabels(
                [f"{tick:g}" for tick in panel.colorbar_ticks]
            )
        check_plot_memory(f"after {panel.variable}")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    # Matplotlib's large RGBA contourf PNG is valid, but some previewers do
    # not decode it. Save an equivalent RGB PNG for broad compatibility.
    temporary_png = png.with_name(png.stem + ".rgb.tmp.png")
    with Image.open(png) as rendered:
        rgb = rendered.convert("RGB")
    rgb.save(temporary_png, format="PNG", optimize=True)
    temporary_png.replace(png)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path("output/leddy_global_025deg.nc"),
        help="Global product on the WOA23 0.25-degree grid",
    )
    parser.add_argument(
        "--output-stem", type=Path,
        default=Path("figures/global_jet_contourf_180"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for path in plot(args.source, args.output_stem, args.dpi):
        print(path.resolve())


if __name__ == "__main__":
    main()
