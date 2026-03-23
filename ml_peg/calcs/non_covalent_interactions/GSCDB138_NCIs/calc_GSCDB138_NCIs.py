"""
Calculate the non-covalent interaction datasets in GSCDB138 database.

https://arxiv.org/html/2508.13468v1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_peg.calcs.utils.gscdb138 import run_gscdb138
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

# Non-covalent interactions datasets.
DATASETS = [
    "3B-69",
    "3BHET",
    "A19Rel6",
    "A24",
    "ADIM6",
    "AHB21",
    "Bauza30",
    "BzDC215",
    "CARBHB8",
    "CHB6",
    "CT20",
    "DS14",
    "FmH2O10",
    "H2O16Rel4",
    "H2O20Rel9",
    "HB262",
    "HB49",
    "HCP32",
    "He3",
    "HEAVY28",
    "HSG",
    "HW30",
    "HW6Cl5",
    "HW6F",
    "IHB100",
    "IHB100x2",
    "IL16",
    "NBC10",
    "NC11",
    "O24",
    "O24x4",
    "PNICO23",
    "RG10N",
    "RG18",
    "S22",
    "S66",
    "S66Rel7",
    "Shields38",
    "SW49Bind22",
    "SW49Rel28",
    "TA13",
    "WATER27",
    "X40",
    "X40x5",
    "XB25",
]


@pytest.mark.parametrize("mlip", MODELS.items())
def test_gscdb138(mlip: tuple[str, Any]) -> None:
    """
    Run GSCDB138 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    run_gscdb138(mlip=mlip, datasets=DATASETS, out_path=OUT_PATH)
