# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 29-01-2026
# ©alin m elena, GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html
"""Run calculations for QMOF tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import copy

from ase.io import read, write
from janus_core.calculations.single_point import SinglePoint
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_qmof_energy(mlip: tuple[str, Any]) -> None:
    """
    Run QMOF energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    model.default_dtype = "float64"
    #model.kwargs['enable_cueq']=True
    model.device = 'cuda'
    calc = model.get_calculator()

    # Add D3 calculator for this test (for models where applicable)
    #calc = model.add_d3_calculator(calc)

    qmof_energy_dir = (
        download_s3_data(
            key="inputs/MOFs/qmof/QMOF.zip",
            filename="QMOF.zip",
        )
        / "qmof_energy"
    )
    input_file = "qmof_valid_structures.xyz"
    mofs = read(qmof_energy_dir / input_file, index=":")
    for mof in mofs:
        mof.calc = copy(calc)
        sp = SinglePoint(struct=mof)
        sp.run()
    # Write output structures
    write_dir = OUT_PATH / model_name
    print(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / input_file, mofs)
