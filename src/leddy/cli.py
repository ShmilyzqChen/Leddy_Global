"""Command-line interface for staged, restartable calculations."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path

import netCDF4

from .combine import combine_final
from .config import load_config
from .eke import combine_eke_years, compute_eke_years, dated_files
from .plotting import plot_quicklook
from .stratification import compute_stratification
from .validation import validate_output


def configure_logging(config: dict) -> None:
    log_dir = config["paths"]["work_root"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "leddy.log", encoding="utf-8")],
    )


def inspect_inputs(config: dict) -> None:
    for key in ("cmems_daily_root", "woa_temperature", "woa_salinity", "woa_bottom_mask"):
        path = config["paths"][key]
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")
        print(f"{key}: {path}")
    start = datetime.strptime(config["period"]["start"], "%Y-%m-%d").date()
    end = datetime.strptime(config["period"]["end"], "%Y-%m-%d").date()
    files = dated_files(config["paths"]["cmems_daily_root"], start, end)
    print(f"DUACS files in requested period: {len(files)} ({files[0][0]} to {files[-1][0]})")
    with netCDF4.Dataset(config["paths"]["woa_temperature"]) as ds:
        print(f"WOA grid: depth={len(ds.dimensions['depth'])}, lat={len(ds.dimensions['lat'])}, lon={len(ds.dimensions['lon'])}")
    print(f"Region: {config['region']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leddy")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "stratification", "eke", "combine-eke", "combine", "plot", "validate", "run-regional"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    configure_logging(config)
    if args.command == "inspect":
        inspect_inputs(config)
    elif args.command == "stratification":
        print(compute_stratification(config, overwrite=args.overwrite))
    elif args.command == "eke":
        for path in compute_eke_years(config, overwrite=args.overwrite):
            print(path)
    elif args.command == "combine-eke":
        print(combine_eke_years(config, overwrite=args.overwrite))
    elif args.command == "combine":
        print(combine_final(config, overwrite=args.overwrite))
    elif args.command == "plot":
        print(plot_quicklook(config))
    elif args.command == "validate":
        print(validate_output(config))
    elif args.command == "run-regional":
        compute_stratification(config, overwrite=args.overwrite)
        compute_eke_years(config, overwrite=args.overwrite)
        combine_eke_years(config, overwrite=args.overwrite)
        combine_final(config, overwrite=args.overwrite)
        validate_output(config)
        print(plot_quicklook(config))


if __name__ == "__main__":
    main()
