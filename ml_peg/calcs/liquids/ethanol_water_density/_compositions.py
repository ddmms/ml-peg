"""Load composition data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
DATA_PATH = BENCH_ROOT / "data"


@dataclass(frozen=True)
class CompositionCase:
    """Map composition to file."""

    x_ethanol: float
    filename: str


def load_compositions() -> list[CompositionCase]:
    """
    Load composition grid.

    Expected CSV columns: x_ethanol, filename
    """
    comps_file = DATA_PATH / "compositions.csv"
    cases: list[CompositionCase] = []
    with comps_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                CompositionCase(
                    x_ethanol=float(row["x_ethanol"]),
                    filename=row["filename"],
                )
            )
    if not cases:
        raise RuntimeError("No compositions found in compositions.csv")
    return cases
