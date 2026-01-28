"""Run calculations for Surface reaction tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ

#@pytest.fixture(scope="module")
#def relaxed_structs() -> dict[str, Atoms]:

structs = DATA_PATH.glob("*")
#print(list(structs))

for model_name, calc in MODELS.items():
    print(f"{model_name = }")
    for struct_name in structs:
        images = read(DATA_PATH / )
