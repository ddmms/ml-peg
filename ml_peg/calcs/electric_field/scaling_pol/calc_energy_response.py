"""
Calculate the energy response of organic molecules to an external electric field.

The reference calculations were made using ORCA at the wB97M-V/def2-TVDZ level.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import pytest

from ml_peg.calcs.energy_response import run_energy_response
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Organic molecules data sets for electric field response.
DATASETS = [
    "ALKANES",
    "CUMULENES",
]


@pytest.mark.parametrize("mlip", MODELS.items())
def test_energy_response(mlip: tuple[str, Any]) -> None:
	"""
	Run energy response benchmark.

	Parameters
	----------
	mlip
		Name of model use and model to get calculator.
	"""
	run_energy_response(mlip=mlip, datasets=DATASETS, out_path=OUT_PATH)