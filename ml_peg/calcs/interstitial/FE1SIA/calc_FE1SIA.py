"""Run calculations for FE1SIA (formation energy of 1 SIA) tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read, write
import pytest

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# Reference energy for bulk
# E_BULK = -1054.46260251


@pytest.mark.parametrize("mlip", MODELS.items())
def test_fe1sia(mlip: tuple[str, Any]) -> None:
    """
    Run FE1SIA test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = download_s3_data(
        key="inputs/interstitial/FE1SIA/DB.zip",
        filename="DB.zip",
    )
    formation_energy_dir = data_path / "DB" / "formation_energy"

    # Process all *.poscar files
    poscar_files = sorted(formation_energy_dir.glob("*.poscar"))

    with open(formation_energy_dir / "ref.poscar") as f:
        line1 = f.readline()
        tokens = line1.split()
        try:
            energy_bulk = float(tokens[4])
        except (ValueError, IndexError):
            print("Skipping ref.poscar: distinct energy value not found in header.")
            energy_bulk = 0.0  # Fallback or break

        # Read n_bulk from line 6. Skip lines 2-5 (4 lines)
        for _ in range(4):
            f.readline()
        line6 = f.readline()
        n_bulk = sum(float(x) for x in line6.split())

    for poscar_path in poscar_files:
        # Parse reference energy E from the first line of the POSCAR file
        # Format: 111 1 Fe 26 E ...
        with open(poscar_path) as f:
            line1 = f.readline()
            tokens = line1.split()
            try:
                energy_ref_raw = float(tokens[4])
            except (ValueError, IndexError):
                print(
                    f"Skipping {poscar_path.name}: distinct energy value not found in "
                    f"header."
                )
                continue

            # Read n_config from line 6. Skip lines 2-5 (4 lines)
            for _ in range(4):
                f.readline()
            line6 = f.readline()
            n_config = sum(float(x) for x in line6.split())

        # Calculate reference formation energy
        # Formula: E - (N_config / N_bulk) * E_bulk
        # For ref.poscar (bulk), formation energy is 0
        if poscar_path.name == "ref.poscar":
            ref_formation_energy = 0.0
        else:
            ref_formation_energy = energy_ref_raw - (n_config / n_bulk) * energy_bulk

        # Read structure
        atoms = read(poscar_path, format="vasp")

        # Run calculation
        atoms.calc = calc
        atoms.get_potential_energy()

        # Store reference info and system name
        atoms.info["ref"] = ref_formation_energy
        atoms.info["system"] = poscar_path.stem

        # Write outputs
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{poscar_path.stem}.xyz", atoms)
