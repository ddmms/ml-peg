"""Run calculations for CPOSS209 tests."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"


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
    lattice_energy_dir = (
        download_s3_data(
            key="inputs/molecular_crystal/CPOSS209/CPOSS209.zip",
            filename="CPOSS209.zip",
        )
        / "CPOSS209_lattice_energy"
    )

    with open(lattice_energy_dir / "list") as f:
        systems = f.read().splitlines()

    for system in tqdm(systems):
        # Get crystal and molecule files
        crystals = sorted(
            path.name for path in (lattice_energy_dir / system).glob("crystal_*")
        )
        molecules = [path.name for path in (lattice_energy_dir / system).glob("gas_*")]

        # Read number of molecules in crystal file
        num_molecules_path = Path(lattice_energy_dir) / system / "nmol"
        num_molecules = np.loadtxt(num_molecules_path)

        # Read reference energies
        ref_crystal_path = Path(lattice_energy_dir) / system / "lattice_energies.txt"
        ref_energies_path = np.loadtxt(ref_crystal_path)

        for crystal_file, ref_crystal, num_mol in zip(
            crystals, ref_energies_path, num_molecules, strict=True
        ):
            crystal_path = Path(lattice_energy_dir) / system / crystal_file

            # Read crystal structure
            solid = read(crystal_path, index=0)
            # Set default charge and spin
            solid.info.setdefault("charge", 0)
            solid.info.setdefault("spin", 1)
            solid.calc = copy(calc)
            solid.get_potential_energy()

            solid.info["ref"] = ref_crystal
            solid.info["num_molecules"] = num_mol
            solid.info["system"] = system

            # Extract shortened name
            crystal_short_name = crystal_file.replace("crystal_", "").split(".")[0]
            # Remove prefix and suffix
            if "_" in crystal_short_name:
                parts = crystal_short_name.split("_")
                # Extract the middle part (e.g., ACR01 from data_ACR01_PsiCrys)
                crystal_short_name = parts[1] if len(parts) > 1 else crystal_short_name
            solid.info["polymorph_name"] = crystal_short_name

            # Assign molecular family based on the shortened name
            if (
                "CRN" in crystal_short_name
                or "ACR" in crystal_short_name
                or "PTH" in crystal_short_name
                or "SAC" in crystal_short_name
                or "FLU" in crystal_short_name
            ):
                solid.info["molecular_family"] = "Small_rigid_molecules"
            elif (
                "CBZ" in crystal_short_name
                or "DHC" in crystal_short_name
                or "CYH" in crystal_short_name
                or "CYT" in crystal_short_name
                or "OXC" in crystal_short_name
            ):
                solid.info["molecular_family"] = "Carbamazepine_family"
            elif (
                "FEA" in crystal_short_name
                or "MFA" in crystal_short_name
                or "TFA" in crystal_short_name
                or "FFA" in crystal_short_name
                or "NFA" in crystal_short_name
            ):
                solid.info["molecular_family"] = "Fenamate_family"
            else:
                solid.info["molecular_family"] = "Small_drug_molecules"

            # Write output structures
            solid.info["comment"] = solid.info["molecular_family"]
            write_dir = OUT_PATH / model_name / system
            write_dir.mkdir(parents=True, exist_ok=True)

            write(write_dir / f"{crystal_file}", solid)

        for molecule_file in molecules:
            molecule_path = Path(lattice_energy_dir) / system / molecule_file

            # Read gas phases
            molecule = read(molecule_path, index=0)
            molecule.info.setdefault("charge", 0)
            molecule.info.setdefault("spin", 1)
            molecule.calc = copy(calc)
            molecule.get_potential_energy()

            # One molecule in each gas phase file
            molecule.info["num_molecules"] = 1
            molecule.info["system"] = system

            # Write output structures
            write_dir = OUT_PATH / model_name / system
            write_dir.mkdir(parents=True, exist_ok=True)

            write(write_dir / f"{molecule_file}", molecule)
