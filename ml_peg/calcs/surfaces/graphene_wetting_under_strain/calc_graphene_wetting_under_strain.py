"""Run calculations for graphene wetting under strain benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ase.io
import pytest
import yaml

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

with open(Path(__file__).parent / "database_info.yml") as fp:
    DATABASE_INFO = yaml.safe_load(fp)

# List of orientations used in the database
ORIENTATIONS = DATABASE_INFO["orientations"]

# List of strains used in the database
STRAINS = DATABASE_INFO["strains"]


@pytest.mark.parametrize("mlip", MODELS.items())
def test_graphene_wetting_energy(mlip: tuple[str, Any]) -> None:
    """
    Run graphene wetting adsorption energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Add D3 calculator for this test (for models where applicable)
    calc = model.add_d3_calculator(calc)

    # Download dataset
    structs_dir = (
        download_s3_data(
            key="inputs/surfaces/graphene_wetting_under_strain/graphene_wetting_under_strain.zip",
            filename="graphene_wetting_under_strain.zip",
        )
        / "graphene_wetting_under_strain"
    )

    # Calculate energy of single water molecule
    atoms = ase.io.read(structs_dir / "ref_water.xyz", format="extxyz")
    atoms.calc = calc
    water_energy = atoms.get_potential_energy()

    # Iterate through strain conditions
    for strain in STRAINS:
        atoms = ase.io.read(structs_dir / f"ref_graphene_{strain}.xyz", format="extxyz")
        atoms.calc = calc
        graphene_energy = atoms.get_potential_energy()

        # Iterate through orientations
        for orientation in ORIENTATIONS:
            systems = ase.io.iread(
                structs_dir / f"{orientation}_{strain}.xyz", index=":", format="extxyz"
            )
            write_file = write_dir / f"{orientation}_{strain}.xyz"
            if os.path.isfile(write_file):
                os.remove(write_file)
            for atoms in systems:
                atoms.calc = calc
                mlip_potential_energy = atoms.get_potential_energy()
                mlip_adsorption_energy = (
                    mlip_potential_energy - graphene_energy - water_energy
                )
                atoms.info["mlip_adsorption_energy"] = mlip_adsorption_energy
                ase.io.write(write_file, atoms, append=True)
