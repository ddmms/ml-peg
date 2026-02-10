"""
Calculate the thermochemistry datasets in GSCDB138 database.

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

# Thermochemistry datasets.
DATASETS = [
    "AE11",
    "AE18",
    "AL2X6",
    "ALK8",
    "AlkAtom19",
    "ALKBDE10",
    "AlkIsod14",
    "BDE99MR",
    "BDE99nonMR",
    "BH76RC",
    "BSR36",
    "CR20",
    "DARC",
    "DC13",
    "DIPCS9",
    "EA50",
    "FH51",
    "G21EA",
    "G21IP",
    "G2RC24",
    "HAT707MR",
    "HAT707nonMR",
    "HEAVYSB11",
    "HNBrBDE18",
    "IP23",
    "IP30",
    "MB08-165",
    "MB16-43",
    "MX34",
    "NBPRC",
    "P34AE",
    "P34EA",
    "P34IP",
    "PA26",
    "PlatonicRE18",
    "PlatonicTAE6",
    "RC21",
    "RSE43",
    "SIE4x4",
    "SN13",
    "TAE_W4-17MR",
    "TAE_W4-17nonMR",
    "WCPT6",
    "YBDE18",
]

EXCLUDE_ELEMENTS = (86,)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_gscdb138(mlip: tuple[str, Any]) -> None:
    """
    Run GSCDB138 thermochemistry benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    run_gscdb138(
        mlip=mlip,
        datasets=DATASETS,
        out_path=OUT_PATH,
        exclude_elements=EXCLUDE_ELEMENTS,
    )
