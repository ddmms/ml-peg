"""
Calculate the MPCONF196 dataset of biomolecule conformers.

J. Comput. Chem. 2024, 45(7), 419.
https://doi.org/10.1002/jcc.27248.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read, write
import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"

MOLECULES = [
    "FGG",
    "GFA",
    "GGF",
    "WG",
    "WGG",
    "CAMVES",
    "CHPSAR",
    "COHVAW",
    "GS464992",
    "GS557577",
    "POXTRD",
    "SANGLI",
    "YIVNOG",
]


def get_atoms(atoms_path: Path) -> Atoms:
    """
    Read atoms object and add charge and spin.

    Parameters
    ----------
    atoms_path
        Path to atoms object.

    Returns
    -------
    atoms
        ASE atoms object.
    """
    atoms = read(atoms_path)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def get_ref_energies(data_path: Path) -> dict[str, float]:
    """
    Get reference conformer energies.

    Parameters
    ----------
    data_path
        Path to the structure.

    Returns
    -------
    dict[str, float]
        Reference energies for all systems.
    """
    df = pd.read_excel(
        data_path / "Energies_CCSD(T).xlsx",
        sheet_name="Old vs New CCSD(T) Reference Va",
        header=1,
    )
    ref_energies = {}

    for row in df.iterrows():
        label = row[1][0]
        ref_energies[label] = float(row[1][2]) * KCAL_TO_EV

    return ref_energies


@pytest.mark.parametrize("mlip", MODELS.items())
def test_mpconf196(mlip: tuple[str, Any]) -> None:
    """
    Run MPCONF196 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="MPCONF196.zip",
            key="inputs/conformers/MPCONF196/MPCONF196.zip",
        )
        / "MPCONF196"
    )

    ref_energies = get_ref_energies(data_path)
    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for molecule in tqdm(MOLECULES):
        model_abs_energies = []
        ref_abs_energies = []
        current_molecule_labels = []

        # Get reference and predicted energy for each conformer
        for label, e_ref in ref_energies.items():
            molecule_label = label.split("_")[0]
            conformer_label = label.split("_")[1]
            if label[-1].isnumeric():
                xyz_fname = f"{molecule_label}{conformer_label}.xyz"
            else:
                xyz_fname = f"{molecule_label}_{conformer_label}.xyz"
            if molecule != molecule_label:
                continue
            atoms = get_atoms(data_path / xyz_fname)
            atoms.translate(-atoms.get_center_of_mass())
            atoms.calc = calc
            model_abs_energies.append(atoms.get_potential_energy())
            ref_abs_energies.append(e_ref)
            current_molecule_labels.append(label)

        # Get energies relative to average conformer energies
        for label, e_model in zip(
            current_molecule_labels, model_abs_energies, strict=True
        ):
            molecule_label = label.split("_")[0]
            conformer_label = label.split("_")[1]
            if label[-1].isnumeric():
                xyz_fname = f"{molecule_label}{conformer_label}.xyz"
            else:
                xyz_fname = f"{molecule_label}_{conformer_label}.xyz"
            atoms = get_atoms(data_path / xyz_fname)
            atoms.translate(-atoms.get_center_of_mass())
            atoms.info["ref_rel_energy"] = ref_energies[label] - np.mean(
                ref_abs_energies
            )
            atoms.info["model_rel_energy"] = e_model - np.mean(model_abs_energies)

            write_dir = OUT_PATH / model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{label}.xyz", atoms)
