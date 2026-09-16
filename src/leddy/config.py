"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve its filesystem paths."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {config_path}")

    required_sections = {"paths", "period", "region", "eke", "stratification", "qc", "output", "resources"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    for key, value in config["paths"].items():
        config["paths"][key] = Path(value).expanduser().resolve()
    config["config_path"] = config_path
    return config


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create only project-owned output directories."""
    root = config["paths"]["work_root"]
    for relative in ("intermediate/eke_yearly", "output", "logs", "figures"):
        (root / relative).mkdir(parents=True, exist_ok=True)

