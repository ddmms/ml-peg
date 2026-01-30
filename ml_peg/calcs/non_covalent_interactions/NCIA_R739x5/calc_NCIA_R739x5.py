"""
Compute the NCIA R739x5 repulsive contacts dataset.

Journal of Chemical Theory and Computation 2021 17 (3), 1548-1561.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV

OUT_PATH = Path(__file__).parent / "outputs"
EXCLUDE_NOBLE_GASES = True


def get_ref_energies(data_path: Path) -> dict[str, float]:
    """
    Get reference energies.

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

    with open(data_path / "NCIA_R739x5_benchmark.txt") as lines:
        for i, line in enumerate(lines):
            if i < 3:
                continue
            items = line.strip().split()
            label = items[0]
            ref_energy = float(items[1]) * KCAL_TO_EV
            ref_energies[label] = ref_energy

    return ref_energies


def get_monomers(atoms: Atoms) -> tuple[Atoms, Atoms]:
    """
    Get ASE atoms objects of the monomers.

    Parameters
    ----------
    atoms
        ASE atoms object of the structure.

    Returns
    -------
    tuple[ASE.Atoms, ASE.Atoms]
        Tuple containing the two monomers.
    """
    if isinstance(atoms.info["selection_a"], str):
        a_ids = [int(id) for id in atoms.info["selection_a"].split("-")]
        a_ids[0] -= 1
    else:
        a_ids = [int(atoms.info["selection_a"]) - 1, int(atoms.info["selection_a"])]

    if isinstance(atoms.info["selection_b"], str):
        b_ids = [int(id) for id in atoms.info["selection_b"].split("-")]
        b_ids[0] -= 1
    else:
        b_ids = [int(atoms.info["selection_b"]) - 1, int(atoms.info["selection_b"])]

    atoms_a = atoms[a_ids[0] : a_ids[1]]
    atoms_b = atoms[b_ids[0] : b_ids[1]]
    assert len(atoms_a) + len(atoms_b) == len(atoms)

    atoms_a.info["charge"] = int(atoms.info["charge_a"])
    atoms_a.info["spin"] = 1

    atoms_b.info["charge"] = int(atoms.info["charge_b"])
    atoms_b.info["spin"] = 1
    return (atoms_a, atoms_b)


@pytest.mark.parametrize("mlip", MODELS.items())
def test_ncia_r739x5(mlip: tuple[str, Any]) -> None:
    """
    Run NCIA R739x5 barriers benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="NCIA_R739.zip",
            key="inputs/non_covalent_interactions/NCIA_R739x5/NCIA_R739x5.zip",
        )
        / "NCIA_R739x5"
    )
    ref_energies = get_ref_energies(data_path)

    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for label, ref_energy in tqdm(ref_energies.items()):
        xyz_fname = f"{label}.xyz"
        atoms = read(data_path / "geometries" / xyz_fname)
        # exclude noble gases
        if (
            np.isin(atoms.numbers, [2, 10, 18, 36, 54, 86]).any()
            and EXCLUDE_NOBLE_GASES
        ):
            continue
        atoms_a, atoms_b = get_monomers(atoms)
        atoms.info["spin"] = 1
        atoms.info["charge"] = int(atoms_a.info["charge"] + atoms_b.info["charge"])
        atoms.calc = calc
        atoms_a.calc = calc
        atoms_b.calc = calc

        atoms.info["model_int_energy"] = (
            atoms.get_potential_energy()
            - atoms_a.get_potential_energy()
            - atoms_b.get_potential_energy()
        )
        atoms.info["ref_int_energy"] = ref_energy
        atoms.calc = None

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{label}.xyz", atoms)
