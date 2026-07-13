"""Run calculations for the TorsionNet500CCSDT benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import units
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data" / "TorsionNet500_xyz"
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
HARTREE_TO_EV = units.Hartree


@pytest.mark.parametrize("mlip", MODELS.items())
def test_torsionnet500ccsdt(mlip: tuple[str, Any]) -> None:
    """
    Run the TorsionNet500CCSDT benchmark.

    Parameters
    ----------
    mlip
        Name of model and model to get calculator.
    """
    model_name, model = mlip
    try:
        calc = model.get_calculator(precision="high")
    except ModuleNotFoundError as exc:
        pytest.skip(f"Skipping {model_name}: {exc}")

    data_path = (
        download_s3_data(
            filename="TorsionNet500CCSDT.zip",
            key="inputs/conformers/TorsionNet500CCSDT/TorsionNet500CCSDT.zip",
        )
        / "TorsionNet500CCSDT"
    )

    xyz_files = sorted(data_path.glob("*.xyz"))

    for o in tqdm(xyz_files):
        atoms = read(o, ":")

        for a in atoms:
            # Reference energy from the dataset
            a.info["ref_energy"] = a.info["E_CCSDT"] * HARTREE_TO_EV
            a.info["charge"] = int(a.info.get("charge", 0))
            a.info["spin"] = int(a.info.get("spin", 1))

            # Model energy
            a.calc = calc

            try:
                a.info["model_energy"] = a.get_potential_energy()
            except Exception as exc:
                warn(f"Error calculating energy for {o.name}: {exc}", stacklevel=2)
                a.info["model_energy"] = np.nan

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / o.name, atoms)
