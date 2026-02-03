"""Calculate Relastab benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

# from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


@pytest.mark.parametrize("mlip", MODELS.items())
def test_relastab(mlip: tuple[str, Any]) -> None:
    """
    Run Relastab calculations.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = download_s3_data(
        key="inputs/interstitial/relative_stability/DB.zip",
        filename="DB.zip",
    )
    relative_stability_dir = data_path / "DB" / "relative_stability"

    # Process all *.poscar files
    poscar_files = sorted(relative_stability_dir.glob("*.poscar"))

    for poscar in poscar_files:
        # Read structure
        atoms = read(poscar, format="vasp")

        # Extract ref energy
        with open(poscar) as f:
            header = f.readline().strip()
            try:
                ref_energy = float(header.split()[4])
                atoms.info["ref"] = ref_energy
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not extract energy from header '{header}': {e}")
                atoms.info["ref"] = None

        # Calculate
        atoms.calc = calc

        atoms.get_potential_energy()
        atoms.info["system"] = poscar.stem

        # Write outputs
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{poscar.stem}.xyz", atoms)
