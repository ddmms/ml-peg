"""Stage lanthanide isomer complex energies for analysis."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from ml_peg.calcs import CALCS_ROOT

OUT_PATH = CALCS_ROOT / "lanthanides" / "isomer_complexes" / "outputs"
CSV_ENV_VAR = "ML_PEG_LANTHANIDE_CSV"


def _resolve_source_csv() -> Path | None:
    env_path = os.environ.get(CSV_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    default_path = OUT_PATH / "isomer_energies.csv"
    if default_path.exists():
        return default_path
    return None


def test_stage_isomer_complexes_csv() -> None:
    """
    Stage the precomputed isomer energies CSV for analysis.

    Set `ML_PEG_LANTHANIDE_CSV` to point to the source CSV.
    """
    source_csv = _resolve_source_csv()
    if source_csv is None or not source_csv.exists():
        pytest.skip(
            "No lanthanide isomer CSV found. "
            "Set ML_PEG_LANTHANIDE_CSV to the isomer_energies.csv path."
        )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    dest_csv = OUT_PATH / "isomer_energies.csv"
    if source_csv.resolve() != dest_csv.resolve():
        shutil.copyfile(source_csv, dest_csv)
