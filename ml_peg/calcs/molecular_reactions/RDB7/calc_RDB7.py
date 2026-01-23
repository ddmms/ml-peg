"""
Calculate the RDB7 reaction barrier dataset.

Spiekermann, K., Pattanaik, L. & Green, W.H.
High accuracy barrier heights, enthalpies,
and rate coefficients for chemical reactions.
Sci Data 9, 417 (2022)
https://doi.org/10.1038/s41597-022-01529-6
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atom, Atoms, units
from ase.io import write
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


def get_cc_energy(fname):
    """
    Read reference energy.

    Parameters
    ----------
    fname
        Name of the calculation output file.

    Returns
    -------
    float
        CCSD(T)-F12/cc-pVDZ-F12 energy.
    """
    with open(fname) as lines:
        for line in lines:
            if "CCSD(T)-F12/cc-pVDZ-F12 energy" in line:
                energy = float(line.strip().split()[-1]) * units.Hartree
                break
    return energy


def get_atoms_from_molpro(fname):
    """
    Get ASE atoms from the molpro file.

    Parameters
    ----------
    fname
        Name of the calculation output file.

    Returns
    -------
    ASE.Atoms
        ASE atoms object of the structure.
    """
    atoms = Atoms(None)
    with open(fname) as lines:
        read_started = False
        for i, line in enumerate(lines):
            if "ATOMIC COORDINATES" in line:
                read_started = True
                xyz_start = i + 4
            if read_started:
                if i >= xyz_start:
                    items = line.strip().split()
                    if len(items) == 0:
                        break
                    position = (
                        np.array([float(items[3]), float(items[4]), float(items[5])])
                        * units.Bohr
                    )
                    atoms += Atom(symbol=items[1], position=position)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


@pytest.mark.parametrize("mlip", MODELS.items())
def test_rdb87(mlip: tuple[str, Any]) -> None:
    """
    Run RDB7 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="RDB7.zip",
            key="inputs/molecular_reactions/RDB7/RDB7.zip",
        )
        / "RDB7"
    )

    # Read in data and attach calculator
    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    for i in tqdm(range(0, 11961)):
        bh_forward_ref = 0
        bh_forward_model = 0
        label = str(i).zfill(6)
        for qm_path in (data_path / "qm_logs" / f"rxn{label}").glob("r*"):
            bh_forward_ref -= get_cc_energy(qm_path)
            atoms = get_atoms_from_molpro(qm_path)
            atoms.calc = calc
            bh_forward_model -= atoms.get_potential_energy()
        for qm_path in (data_path / "qm_logs" / f"rxn{label}").glob("ts*"):
            bh_forward_ref += get_cc_energy(qm_path)
            atoms = get_atoms_from_molpro(qm_path)
            atoms.calc = calc
            bh_forward_model += atoms.get_potential_energy()

        atoms.info["model_forward_barrier"] = bh_forward_model
        atoms.info["ref_forward_barrier"] = bh_forward_ref

        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{label}_ts.xyz", atoms)
