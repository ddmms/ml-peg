"""
Compute the IONPI19 dataset for ion-pi system interactions.

Phys. Chem. Chem. Phys., 2021,23, 11635-11648.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


SPECIES = {
    1: ["1_AB", "1_A", "1_B"],
    2: ["2_AB", "2_A", "2_B"],
    3: ["3_AB", "3_A", "3_B"],
    4: ["4_AB", "4_A", "4_B"],
    5: ["5_AB", "5_A", "5_B"],
    6: ["6_AB", "6_A", "6_B"],
    7: ["7_AB", "7_A", "7_B"],
    8: ["8_AB", "8_A", "8_B"],
    9: ["9_AB", "9_A", "9_B"],
    10: ["10_AB", "10_A", "10_B"],
    11: ["11_AB", "11_A", "11_B"],
    12: ["12_AB", "12_A", "12_B"],
    13: ["13_AB", "13_A", "13_B"],
    14: ["14_AB", "14_A", "14_B"],
    15: ["15_AB", "15_A", "15_B"],
    16: ["16_AB", "15_A", "16_B"],
    17: ["17_AB", "17_A", "17_B"],
    18: ["18_A", "18_B"],
    19: ["19_A", "19_B"],
}


def get_atoms(path: Path) -> Atoms:
    """
    Get the atoms object with charge and spin.

    Parameters
    ----------
    path
        Path to atoms.

    Returns
    -------
    Atoms
        Atoms object of the system.
    """
    chrg = path / "CHRG"
    mol_fname = path / "mol.xyz"
    atoms = read(mol_fname)
    atoms.info["spin"] = 1
    if not os.path.exists(chrg):
        charge = 0
    else:
        with open(chrg) as lines:
            for line in lines:
                items = line.strip().split()
                charge = int(items[0])
    atoms.info["charge"] = charge
    return atoms


def get_ref_energies(data_path: Path) -> dict[str, float]:
    """
    Extract the reference energies.

    Parameters
    ----------
    data_path
        Path to data.

    Returns
    -------
    dict[str, float]
        Loaded reference energies.
    """
    ref_energies = {}

    with open(data_path / "energies.txt") as lines:
        for i, line in enumerate(lines):
            if i < 2:
                continue
            items = line.strip().split()
            system_id = int(items[0])
            ref_energies[system_id] = float(items[1]) * KCAL_TO_EV

    return ref_energies


@pytest.mark.parametrize("mlip", MODELS.items())
def test_ionpi19(mlip: tuple[str, Any]) -> None:
    """
    Run IONPI19 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="IONPI19.zip",
            key="inputs/non_covalent_interactions/IONPI19/IONPI19.zip",
        )
        / "ionpi19"
    )
    ref_energies = get_ref_energies(data_path)
    # Read in data and attach calculator
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for system_id in tqdm(range(1, 20)):
        for config in SPECIES[system_id]:
            label = f"{config}"
            atoms = get_atoms(data_path / label)
            atoms.info["ref_int_energy"] = ref_energies[system_id]
            atoms.calc = calc
            atoms.info["model_energy"] = atoms.get_potential_energy()

            write_dir = OUT_PATH / model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{label}.xyz", atoms)
