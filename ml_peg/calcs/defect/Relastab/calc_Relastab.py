"""Calculate Relastab benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

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
    # Use double precision
    calc = model.get_calculator(precision="high")

    data_path = download_s3_data(
        key="inputs/defect/Relastab/Relastab.zip",
        filename="Relastab.zip",
    )
    relative_stability_dir = data_path / "Relastab"

    # Process subfolders (subsets)
    for subset_path in relative_stability_dir.iterdir():
        if not subset_path.is_dir():
            continue

        subset_name = subset_path.name
        poscar_files = sorted(subset_path.glob("*.poscar"))

        for poscar in poscar_files:
            # Read structure
            atoms = read(poscar, format="vasp")
            # Set default charge and spin
            atoms.info.setdefault("charge", 0)
            atoms.info.setdefault("spin", 1)

            # Extract ref energy from POSCAR header
            with open(poscar) as f:
                header = f.readline().strip()
                tokens = header.split()
                for idx in (4, 5):
                    try:
                        atoms.info["ref"] = float(tokens[idx])
                        break
                    except (IndexError, ValueError):
                        continue
                else:
                    raise ValueError(
                        f"Could not parse reference energy from "
                        f"header of {poscar}: '{header}'"
                    )

            # Calculate
            atoms.calc = calc

            atoms.get_potential_energy()
            atoms.info["system"] = poscar.stem
            atoms.info["subset"] = subset_name

            # Write outputs
            # Flattened structure: subset_name_poscar.stem.xyz
            write_dir = OUT_PATH / model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{subset_name}_{poscar.stem}.xyz", atoms)
