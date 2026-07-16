"""Run calculations for graphene wetting under strain benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

import ase.io
import numpy as np
import pytest
from tqdm import tqdm
import yaml

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"


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
    calc = model.get_calculator(precision="high")
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

    db_info_path = Path(structs_dir) / "database_info.yml"
    with open(db_info_path) as fp:
        database_info = yaml.safe_load(fp)
    orientations = database_info["orientations"]
    strains = database_info["strains"]

    # save database info for use in analysis
    # (without needing to redownload to get the path)
    database_info_path = OUT_PATH / "database_info.yml"
    if not database_info_path.exists():
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        with database_info_path.open("w", encoding="utf-8") as target_fp:
            yaml.safe_dump(database_info, target_fp, sort_keys=False)

    # Calculate energy of single water molecule
    atoms = ase.io.read(structs_dir / "ref_water.xyz", format="extxyz")
    atoms.calc = calc
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    try:
        water_energy = atoms.get_potential_energy()
    except Exception as exc:
        warn(f"Error calculating energy: {exc}", stacklevel=2)
        water_energy = np.nan

    # Iterate through strain conditions
    for strain in strains:
        atoms = ase.io.read(structs_dir / f"ref_graphene_{strain}.xyz", format="extxyz")
        atoms.calc = calc
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)
        try:
            graphene_energy = atoms.get_potential_energy()
        except Exception as exc:
            warn(
                f"Error calculating energy for strain {strain}: {exc}",
                stacklevel=2,
            )
            graphene_energy = np.nan

        # Iterate through orientations
        for orientation in orientations:
            systems = ase.io.read(
                structs_dir / f"{orientation}_{strain}.xyz", index=":", format="extxyz"
            )
            write_file = write_dir / f"{orientation}_{strain}.xyz"
            if write_file.is_file():
                write_file.unlink(missing_ok=True)
            desc = f"{orientation} orientation with {strain[1:5]}% strain"
            for atoms in tqdm(systems, desc=desc, unit="configurations"):
                atoms.calc = calc
                atoms.info.setdefault("charge", 0)
                atoms.info.setdefault("spin", 1)
                try:
                    mlip_potential_energy = atoms.get_potential_energy()
                except Exception as exc:
                    warn(f"Error calculating energy: {exc}", stacklevel=2)
                    mlip_potential_energy = np.nan
                mlip_adsorption_energy = (
                    mlip_potential_energy - graphene_energy - water_energy
                )
                atoms.info["mlip_adsorption_energy"] = mlip_adsorption_energy
                ase.io.write(write_file, atoms, append=True)
