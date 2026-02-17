"""Run calculations for X23 tests."""

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


@pytest.mark.parametrize("mlip", MODELS.items())
def test_lattice_energy(mlip: tuple[str, Any]) -> None:
    """
    Run X23 lattice energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # download X23 dataset
    lattice_energy_dir = (
        download_s3_data(
            key="inputs/molecular_crystal/X23/X23.zip",
            filename="lattice_energy.zip",
        )
        / "lattice_energy"
    )

    with open(lattice_energy_dir / "list") as f:
        systems = f.read().splitlines()

    for system in systems:
        molecule_path = lattice_energy_dir / system / "POSCAR_molecule"
        solid_path = lattice_energy_dir / system / "POSCAR_solid"
        ref_path = lattice_energy_dir / system / "lattice_energy_DMC"
        num_molecules_path = lattice_energy_dir / system / "nmol"

        molecule = read(molecule_path, index=0, format="vasp")
        molecule.calc = calc
        # Set default charge and spin
        molecule.info.setdefault("charge", 0)
        molecule.info.setdefault("spin", 1)
        molecule.get_potential_energy()

        solid = read(solid_path, index=0, format="vasp")
        solid.calc = copy(calc)
        # Set default charge and spin
        solid.info.setdefault("charge", 0)
        solid.info.setdefault("spin", 1)
        solid.get_potential_energy()

        ref = np.loadtxt(ref_path)[0]
        num_molecules = np.loadtxt(num_molecules_path)

        solid.info["ref"] = ref
        solid.info["num_molecules"] = num_molecules
        solid.info["system"] = system
        molecule.info["ref"] = ref
        molecule.info["num_molecules"] = num_molecules
        molecule.info["system"] = system

        # Write output structures
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{system}.xyz", [solid, molecule])
