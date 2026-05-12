"""Generate text files listing the polymer benchmark subsets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Final

POLYMERS_ROOT: Final[Path] = Path(__file__).parent
DATA_CSV: Final[Path] = POLYMERS_ROOT / "resources" / "data.csv"
OUTPUT_DIR: Final[Path] = POLYMERS_ROOT / "resources" / "polymer_sets"
SMALL_POLYMER_IDS: Final[list[str]] = [
    "PE",
    "PIB",
    "PET",
    "PCTFE",
    "PS",
    "PVMS",
    "PVC",
    "PAN",
    "PTPC",
    "PODPM",
]


def load_polymer_ids(data_csv: Path = DATA_CSV) -> list[str]:
    """
    Load polymer ids from the benchmark reference table.

    Parameters
    ----------
    data_csv
        Path to the polymer reference CSV.

    Returns
    -------
    list[str]
        Polymer ids in the order defined by ``data.csv``.
    """
    with open(data_csv, encoding="utf-8") as file:
        rows = (line for line in file if not line.startswith("%"))
        return [str(row["id"]) for row in csv.DictReader(rows)]


def build_polymer_sets(polymer_ids: list[str]) -> dict[str, list[str]]:
    """
    Build the small, medium and large polymer id sets.

    Parameters
    ----------
    polymer_ids
        All polymer ids in the order defined by ``data.csv``.

    Returns
    -------
    dict[str, list[str]]
        Mapping of subset name to polymer ids.
    """
    missing_small_ids = set(SMALL_POLYMER_IDS) - set(polymer_ids)
    if missing_small_ids:
        missing = ", ".join(sorted(missing_small_ids))
        raise ValueError(
            f"Small polymer set contains ids missing from data.csv: {missing}"
        )

    medium_extra_ids = [
        polymer_id for polymer_id in polymer_ids if polymer_id not in SMALL_POLYMER_IDS
    ][:40]

    return {
        "small": list(SMALL_POLYMER_IDS),
        "medium": list(SMALL_POLYMER_IDS) + medium_extra_ids,
        "large": list(polymer_ids),
    }


def write_polymer_sets(
    polymer_sets: dict[str, list[str]],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Write polymer subset files.

    Parameters
    ----------
    polymer_sets
        Mapping of subset name to polymer ids.
    output_dir
        Directory where ``<subset>.txt`` files are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset_name, polymer_ids in polymer_sets.items():
        output_path = output_dir / f"{subset_name}.txt"
        output_path.write_text("\n".join(polymer_ids) + "\n", encoding="utf-8")


def main() -> None:
    """Generate the polymer subset text files."""
    write_polymer_sets(build_polymer_sets(load_polymer_ids()))


if __name__ == "__main__":
    main()
