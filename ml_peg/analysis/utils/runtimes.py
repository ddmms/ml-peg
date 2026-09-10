"""Read maintainer reference runtimes for benchmarks."""

from __future__ import annotations

from pathlib import Path

import yaml

RUNTIMES_FILE = Path(__file__).with_name("runtimes.yml")


def load_runtimes() -> tuple[dict[str, str], dict[str, float]]:
    """
    Load runtime provenance and measured minutes per model.

    Returns
    -------
    tuple[dict[str, str], dict[str, float]]
        Provenance of the measurements and a mapping of benchmark identifiers
        to minutes per model.
    """
    data = yaml.safe_load(RUNTIMES_FILE.read_text(encoding="utf8")) or {}

    measured_with = data.get("measured_with") or {}
    if not isinstance(measured_with, dict):
        raise ValueError(
            f"measured_with in {RUNTIMES_FILE} must contain model/device metadata"
        )
    provenance = {key: value for key, value in measured_with.items() if value}
    measured = {
        f"{category}/{benchmark}": float(minutes)
        for category, benchmarks in (data.get("benchmarks") or {}).items()
        for benchmark, minutes in (benchmarks or {}).items()
        if minutes is not None
    }
    return provenance, measured
