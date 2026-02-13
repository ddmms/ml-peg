"""
Calculate the electric field property datasets in GSCDB138 database.

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

# Electric field property datasets.
DATASETS = ["Dip146", "HR46", "OEEF", "Pol130", "T144", "V30"]


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
