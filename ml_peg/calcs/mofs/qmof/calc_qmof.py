# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 29-01-2026
# ©alin m elena, GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html
"""Run calculations for QMOF tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
from janus_core.calculations.single_point import SinglePoint
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

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
    calc = model.get_calculator(precision="high")

    # Add D3 calculator for this test (for models where applicable)
    calc = model.add_d3_calculator(calc)

    qmof_energy_dir = (
        download_s3_data(
            key="inputs/mofs/qmof/qmof.zip",
            filename="qmof.zip",
        )
        / "qmof"
    )
    input_file = "qmof_valid_structures.traj"
    mofs = read(qmof_energy_dir / input_file, index=":")
    for mof in tqdm(mofs, desc=model_name):
        mof.calc = copy(calc)
        sp = SinglePoint(struct=mof)
        sp.run()
    # Write output structures
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    write(write_dir / input_file, mofs)
