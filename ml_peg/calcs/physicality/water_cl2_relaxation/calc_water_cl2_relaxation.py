"""Compute the Cl2 in water cluster relaxation trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import units
from ase.io import read
from ase.optimize import LBFGS
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_water_cl2_relaxation(mlip: tuple[str, Any]) -> None:
    """
    Run WaterCl2 relaxation test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    data_path = (
        download_s3_data(
            filename="WaterCl2.zip",
            key="inputs/WaterCl2/WaterCl2.zip",
        )
        / "WaterCl2"
    )

    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # Get starting atoms object.
    atoms = read(data_path / "start.xyz")
    atoms.info["charge"] = -2
    atoms.info["spin"] = 1
    atoms.calc = calc

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    opt = LBFGS(
        atoms,
        trajectory=write_dir / "relaxation.traj",
        logfile=write_dir / "relaxation.log",
    )
    opt.run(fmax=0.01, steps=1000)
