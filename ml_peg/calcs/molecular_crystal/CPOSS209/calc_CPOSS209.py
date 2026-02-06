"""Run calculations for CPOSS209 tests."""

from __future__ import annotations

import os
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
    Run CPOSS209 lattice energy test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # Load data
    # lattice_energy_dir = (download_s3_data(
    #     key="inputs/molecular_crystal/CPOSS209/CPOSS209.zip",
    #     filename="CPOSS209.zip",  
    # ) / "lattice_energy")

    lattice_energy_dir = Path("/home/mjgawkowski/ml-peg/ml_peg/calcs/molecular_crystal/CPOSS209/lattice_energy/")

    with open(lattice_energy_dir / "list") as f:
        systems = f.read().splitlines()

    for system in systems:

        # Get crystal and molecule files
        crystals = [file for file in os.listdir(Path(lattice_energy_dir) / system) if file.startswith('crystal_')]
        molecules = [file for file in os.listdir(Path(lattice_energy_dir) / system) if file.startswith('gas_')]

        crystals = sorted(crystals)

        # Read number of molecules in crystal file
        num_molecules_path = Path(lattice_energy_dir) / system / "nmol"
        num_molecules = np.loadtxt(num_molecules_path)

        # Read reference energies
        ref_crystal_path = Path(lattice_energy_dir) / system / "lattice_energies.txt"
        ref_energies_path = np.loadtxt(ref_crystal_path)

        for crystal_file, ref_crystal, num_mol in zip(crystals, ref_energies_path, num_molecules):
            crystal_path = Path(lattice_energy_dir) / system / crystal_file

            # Read crystal structures
            solid = read(crystal_path, index=0)
            solid.calc = copy(calc)
            solid_energy = solid.get_potential_energy()

            solid.info["ref"] = ref_crystal
            solid.info["num_molecules"] = num_mol
            solid.info["system"] = system
            
            # Extract shortened name (e.g., ACR01 from data_ACR01_PsiCrys)
            crystal_short_name = crystal_file.replace('crystal_', '').split('.')[0]
            # Remove prefix and suffix (data_ and _PsiCrys or similar)
            if '_' in crystal_short_name:
                parts = crystal_short_name.split('_')
                # Extract the middle part (e.g., ACR01 from data_ACR01_PsiCrys)
                crystal_short_name = parts[1] if len(parts) > 1 else crystal_short_name
            solid.info["polymorph_name"] = crystal_short_name

            # Write output structures
            write_dir = OUT_PATH / model_name / system
            write_dir.mkdir(parents=True, exist_ok=True)

            write(write_dir / f"{crystal_file}", solid)


        for molecule_file in molecules:
            molecule_path = Path(lattice_energy_dir) / system / molecule_file

            # Read gas phases
            molecule = read(molecule_path, index=0)
            molecule.calc = copy(calc)
            molecule_energy = molecule.get_potential_energy()

            # One molecule in each gas phase file
            molecule.info["num_molecules"] = 1
            molecule.info["system"] = system


            # Write output structures
            write_dir = OUT_PATH / model_name / system
            write_dir.mkdir(parents=True, exist_ok=True)

            write(write_dir / f"{molecule_file}", molecule)

