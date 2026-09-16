"""Runtime resource guards for long, restartable production calculations."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

import psutil

LOGGER = logging.getLogger(__name__)


def guard_resources(config: dict, stage: str) -> None:
    """Stop cleanly before allocation if RAM or work-disk headroom is too low."""
    settings = config["resources"]
    free_memory_gb = psutil.virtual_memory().available / 1024**3
    free_disk_gb = shutil.disk_usage(Path(config["paths"]["work_root"])).free / 1024**3
    required_memory = float(settings["minimum_free_memory_gb"])
    required_disk = float(settings["minimum_free_disk_gb"])
    LOGGER.info(
        "Resource guard [%s]: free RAM %.2f GB, free disk %.2f GB",
        stage,
        free_memory_gb,
        free_disk_gb,
    )
    if free_memory_gb < required_memory:
        raise MemoryError(
            f"Stopped safely before {stage}: free RAM {free_memory_gb:.2f} GB "
            f"is below configured {required_memory:.2f} GB"
        )
    if free_disk_gb < required_disk:
        raise OSError(
            f"Stopped safely before {stage}: free disk {free_disk_gb:.2f} GB "
            f"is below configured {required_disk:.2f} GB"
        )
