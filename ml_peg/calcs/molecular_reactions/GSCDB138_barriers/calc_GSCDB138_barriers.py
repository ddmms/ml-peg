"""
Calculate the reaction barriers datasets in GSCDB138 database.

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

# Barrier height datasets.
DATASETS = [
    "BH28",
    "BH46",
    "BH876",
    "BHDIV7",
    "BHPERI11",
    "BHROT27",
    "CRBH14",
    "DBH22",
    "INV23",
    "ORBH35",
    "PX9",
    "WCPT26",
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
