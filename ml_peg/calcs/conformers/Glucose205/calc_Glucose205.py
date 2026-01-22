"""
Calculate the glucose conformer energy dataset.

Journal of Chemical Theory and Computation,
2016 12 (12), 6157-6168.
DOI: 10.1021/acs.jctc.6b00876
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read, write
import pandas as pd
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"


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


def get_labels(data_path: Path) -> list[str]:
    """
    Get system labels.

    Parameters
    ----------
    data_path
        Path to the structure.

    Returns
    -------
    list[str]
        All system labels.
    """
    labels = []
    for system_path in sorted((data_path / "Glucose_structures").glob("*.xyz")):
        labels.append(system_path.stem)
    return labels


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
    df = pd.read_csv(data_path / "glucose.csv")
    labels = get_labels(data_path)
    ref_energies = {}

    for i, label in enumerate(labels):
        ref_energies[label] = df[" dlpno/cbs(3-4)"][i] * KCAL_TO_EV

    return ref_energies


@pytest.mark.parametrize("mlip", MODELS.items())
def test_glucose205(mlip: tuple[str, Any]) -> None:
    """
    Run Glucose205 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="Glucose205.zip",
            key="inputs/conformers/Glucose205/Glucose205.zip",
        )
        / "Glucose205"
    )

    ref_energies = get_ref_energies(data_path)
    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    lowest_conf_label = "alpha_002"

    conf_lowest = get_atoms(
        data_path / "Glucose_structures" / f"{lowest_conf_label}.xyz"
    )
    conf_lowest.calc = calc
    e_conf_lowest_model = conf_lowest.get_potential_energy()

    for label, e_ref in tqdm(ref_energies.items()):
        # Skip the reference conformer for which the error is automatically zero
        if label == lowest_conf_label:
            continue

        atoms = get_atoms(data_path / "Glucose_structures" / f"{label}.xyz")
        atoms.calc = calc
        atoms.info["model_rel_energy"] = (
            atoms.get_potential_energy() - e_conf_lowest_model
        )
        atoms.info["ref_energy"] = e_ref

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{label}.xyz", atoms)
