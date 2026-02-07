"""
Calculate the UpU46 benchmark dataset for RNA backbone conformations.

Journal of Chemical Theory and Computation,
2015 11 (10), 4972-4991.
DOI: 10.1021/acs.jctc.5b00515.
"""

from __future__ import annotations

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
    atoms.info["charge"] = -1
    atoms.info["spin"] = 1
    if atoms.calc is not None:
        if "energy" in atoms.calc.results:
            del atoms.calc.results["energy"]
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
    ref_energies = {}

    with open(data_path / "references") as lines:
        for i, line in enumerate(lines):
            # Skip the comment lines
            if i < 5:
                continue
            items = line.strip().split()
            label = items[2]
            ref_energies[label] = float(items[7]) * KCAL_TO_EV
    return ref_energies


@pytest.mark.parametrize("mlip", MODELS.items())
def test_upu46(mlip: tuple[str, Any]) -> None:
    """
    Run UpU46 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="UPU46.zip",
            key="inputs/conformers/UpU46/UpU46.zip",
        )
        / "UPU46"
    )
    zero_conf_label = "2p"
    ref_energies = get_ref_energies(data_path)
    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    conf_lowest = get_atoms(data_path / f"{zero_conf_label}.xyz")
    conf_lowest.calc = calc
    e_conf_lowest_model = conf_lowest.get_potential_energy()

    for label, e_ref in tqdm(ref_energies.items()):
        # Skip the reference conformer for
        # which the error is automatically zero
        if label == zero_conf_label:
            continue

        atoms = get_atoms(data_path / f"{label}.xyz")
        atoms.calc = calc
        atoms.info["model_rel_energy"] = (
            atoms.get_potential_energy() - e_conf_lowest_model
        )
        atoms.info["ref_energy"] = e_ref
        atoms.calc = None

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{label}.xyz", atoms)
