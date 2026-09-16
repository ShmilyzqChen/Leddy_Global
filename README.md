# Leddy Global

This project calculates a global, static eddy length-scale climatology on the
WOA23 0.25-degree grid. Stratification comes from annual WOA23 temperature and
salinity, while eddy kinetic energy comes from daily 1993--2024 DUACS anomalous
geostrophic velocities.

The full mathematical and quality-control specification is in
[`docs/methods.md`](docs/methods.md).

The primary dynamical scale is

\[
L_{eddy}=\min(L_d,L_R),\qquad
L_d=\frac{c_1}{\sqrt{f^2+2\beta c_1}},\qquad
L_R=\sqrt{\frac{2\sqrt{2\,EKE}}{\beta}}.
\]

For comparison with wavelength spectra, the output also contains
`Leddy_wavelength_km = 2*pi*Leddy_dyn_km`.

## Reproducible setup

```powershell
conda env create -f environment.yml
conda activate leddy
python -m pip install -e . --no-deps
```

Copy `config/example_scs.yml` to `config/local_scs.yml` for the regional test,
and `config/example.yml` to `config/local_global.yml` for global production.
Edit the five `paths` entries in each local file; the examples already contain
their respective region bounds. Local configurations, source datasets,
intermediates, logs, and non-release outputs are ignored by Git.

## Obtaining the inputs

The four external inputs must be downloaded separately; this repository does
not redistribute third-party data or the full 32-year archive.

- From the [NOAA WOA23 data portal](https://www.ncei.noaa.gov/access/world-ocean-atlas-2023/),
  select quarter-degree annual objectively analysed temperature and salinity,
  using the decadal-average files named `woa23_decav_t00_04.nc` and
  `woa23_decav_s00_04.nc` in this calculation. The official quarter-degree
  [`landsea_04.msk` mask](https://www.ncei.noaa.gov/products/world-ocean-atlas)
  is linked under the WOA23 Masks section; verify that it aligns with the
  downloaded 0.25-degree grids.
- From [Copernicus Marine product SEALEVEL_GLO_PHY_L4_MY_008_047](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description),
  use the daily 0.125-degree dataset
  `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` for
  1993-01-01 through 2024-12-31. The code expects daily NetCDF files
  containing `ugosa`, `vgosa`, and `flag_ice`. The local production archive was
  the DT2024 / 202411 release; later reprocessing may not be bitwise identical.

Record the source release/version and download date when publishing derived
results. Input archives are intentionally excluded from Git.

## Published results

The repository includes the curated outputs from the completed 1993--2024 run:

- `output/leddy_global_025deg.nc`: global 0.25-degree product.
- `output/leddy_south_china_sea_025deg.nc`: regional smoke-test product.
- `output/validation_global.json` and
  `output/validation_south_china_sea.json`: numerical validation summaries.
- `figures/global_jet_contourf_180_rhines1975_final.png` and the matching
  PDF: final six-panel global visualization.

The validation status is `PASS` for both products. Source archives,
restartable intermediates, logs, local path configurations, and exploratory
figures remain excluded.
SHA-256 digests for the six published artifacts are recorded in
`SHA256SUMS`.

## Staged execution

Run the South China Sea smoke test before the global calculation:

```powershell
leddy inspect --config config/local_scs.yml
leddy stratification --config config/local_scs.yml
leddy eke --config config/local_scs.yml
leddy combine --config config/local_scs.yml
leddy validate --config config/local_scs.yml
leddy plot --config config/local_scs.yml
```

Each EKE year is saved separately and an existing valid annual file is skipped,
so interrupted runs can resume. Use `config/local_global.yml` only after the
regional result and tests pass.

Run the global production workflow in stages:

```powershell
leddy stratification --config config/local_global.yml
leddy eke --config config/local_global.yml
leddy combine-eke --config config/local_global.yml
leddy combine --config config/local_global.yml
leddy validate --config config/local_global.yml
leddy plot --config config/local_global.yml
```

Daily DUACS fields are streamed one file at a time and reduced to restartable
annual accumulators; the full 32-year daily archive is never loaded into
memory. Resource guards check free memory and disk space before each year and
at regular intervals. The WOA calculation is likewise split into configurable
latitude blocks (`resources.woa_latitude_block_rows`).

For an independent regional/global consistency check, excluding the two-cell
edge affected by a 5-by-5 QC window:

```powershell
python scripts/compare_regional_global.py output/leddy_global_025deg.nc output/leddy_south_china_sea_025deg.nc
```

For the six-panel, 180-degree-centered global map with `jet` colors,
`contourf` fill, numerical line-contour labels, WOA-aligned coastline,
and out-of-range colorbar triangles:

```powershell
python scripts/plot_global_jet_contours.py
```

The map uses original 0.25-degree fields for the filled contours. Most
display-only line contours are averaged to 1 degree to reduce visual clutter;
small near-limit exceedances use the native grid. Contour labels above the
colorbar range show their actual values, not "200+" or "500+". No calculated
length-scale values are changed. The 0/360-degree seam is closed
periodically, and 180 degrees is the midpoint of each panel. The script
exports PNG and PDF separately from the original quicklook.

## Scientific conventions

- WOA23 annual (`t00`, `s00`) profiles are used on their native nonuniform
  standard-depth grid.
- TEOS-10 supplies Absolute Salinity, Conservative Temperature, pressure, and
  buoyancy frequency.
- The first baroclinic phase speed is solved with a nonuniform finite-element
  vertical-mode discretization; WKB is retained and used as a documented
  fallback.
- DUACS EKE is `0.5 * mean(ugosa**2 + vgosa**2)`, with ice-flagged dates
  excluded and without an additional temporal demeaning or mapping-error
  subtraction.
- Raw fields, corrected fields, sample coverage, method flags, and QC flags are
  retained to keep every adjustment auditable.

## Primary method references

- Eden (2007), *Eddy Length Scales in the North Atlantic Ocean*,
  <https://doi.org/10.1029/2006JC003901>.
- Chelton, D. B., deSzoeke, R. A., Schlax, M. G., El Naggar, K., & Siwertz,
  N. (1998), *Geographical variability of the first baroclinic Rossby radius
  of deformation*, *Journal of Physical Oceanography*, 28, 433--460,
  [doi:10.1175/1520-0485(1998)028<0433:GVOTFB>2.0.CO;2](https://doi.org/10.1175/1520-0485%281998%29028%3C0433%3AGVOTFB%3E2.0.CO%3B2).
- Vergara et al. (2019), *Revised Global Wave Number Spectra from Recent
  Altimeter Observations*, <https://doi.org/10.1029/2018JC014844>.
- Rhines (1975), *Waves and turbulence on a beta-plane*,
  <https://doi.org/10.1017/S0022112075001504>.
- Halo, I., Raj, R. P., Korosov, A., Penven, P., Johannessen, J. A., &
  Rouault, M. (2023), *Mesoscale variability, critical latitude and eddy mean
  properties in the tropical South-East Atlantic Ocean*, *Journal of
  Geophysical Research: Oceans*, 128, e2022JC019050,
  <https://doi.org/10.1029/2022JC019050>.
- Klocker, A., & Abernathey, R. (2014), *Global patterns of mesoscale eddy
  properties and diffusivities*, *Journal of Physical Oceanography*, 44,
  1030--1046, <https://doi.org/10.1175/JPO-D-13-0159.1>.

## Validation and licence

Run `python -m pytest -q` before a full calculation. GitHub Actions executes
the unit tests and checks package-wheel creation on Linux and Windows without
downloading ocean input data. For a complete scientific run, also execute
`leddy validate` and compare the regional and global outputs as shown above.
The source code is MIT-licensed (see `LICENSE`); external data remain subject
to their providers' terms.
