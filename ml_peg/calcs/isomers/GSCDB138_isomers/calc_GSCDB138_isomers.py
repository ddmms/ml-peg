"""
Calculate the isomer/conformer energy datasets in GSCDB138 database.

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

# Isomer/conformer energy datasets.
DATASETS = [
    "ACONF",
    "AlkIsomer11",
    "Amino20x4",
    "BUT14DIOL",
    "C20C246",
    "C60ISO7",
    "DIE60",
    "EIE22",
    "ICONF",
    "IDISP",
    "ISO34",
    "ISOL23",
    "ISOMERIZATION20",
    "MCONF",
    "PArel",
    "PCONF21",
    "Pentane13",
    "SCONF",
    "Styrene42",
    "TAUT15",
    "UPU23",
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
