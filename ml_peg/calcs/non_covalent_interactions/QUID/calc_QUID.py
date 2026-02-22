"""
Compute the QUID benchmark for ligand-pocket interactions.

Puleva, M., Medrano Sandonas, L., Lőrincz, B.D. et al,
Extending quantum-mechanical benchmark accuracy to biological ligand-pocket
interactions,
Nat Commun 16, 8583 (2025). https://doi.org/10.1038/s41467-025-63587-9
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.io import read, write
import h5py
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

KCAL_TO_EV = units.kcal / units.mol

OUT_PATH = Path(__file__).parent / "outputs"

REF_PRIORITY = [
    "CCSDT",
    "LNO-CCSD(T)",
    "CCSD(T)",
]


def choose_best_reference(
    eint: dict[str, float] | None, force_key: str | None = None
) -> tuple[str, float] | None:
    if not eint:
        return None
    # Force a specific reference key if provided
    if force_key is not None:
        # accept case-insensitive match and substring
        fk = None
        for k in eint.keys():
            if force_key.lower() in k.lower():
                fk = k
                break
        if fk is None:
            return None
        return fk, float(eint[fk])
    # case-insensitive scoring
    best_key = None
    for pref in REF_PRIORITY:
        for k in eint.keys():
            if pref.lower() in k.lower():
                # Prefer the first appearance following priority list
                best_key = k
                break
        if best_key is not None:
            break
    if best_key is None:
        # fallback: return any single value
        try:
            k = next(iter(eint.keys()))
            return k, float(eint[k])
        except StopIteration:
            return None
    return best_key, float(eint[best_key])


@pytest.mark.parametrize("mlip", MODELS.items())
def test_quid(mlip: tuple[str, Any]) -> None:
    """
    Run QUID protein ligand-pocket test.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator()

    data_path = (
        download_s3_data(
            filename="QUID.zip",
            key="inputs/non_covalent_interactions/QUID/QUID.zip",
        )
        / "QUID"
    )

    calc = model.get_calculator()
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    dataset = h5py.File(data_path / "QUID.h5")
    for label in dataset.keys():
        # Get equilibrium config
        _, ref_int_energy = choose_best_reference(dataset[label]["Eint"])
        atomic_numbers = dataset[label]["atoms"][:]
        positions = dataset[label]["positions"][:]
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        atoms.info.update({"charge": 0, "spin": 1})
        atoms.calc = calc

    for label, ref_energy in tqdm(ref_energies.items()):
        xyz_fname = f"{label}.xyz"
        atoms = read(data_path / "geometries" / xyz_fname)
        atoms_a, atoms_b = get_monomers(atoms)
        atoms.info["spin"] = 1
        atoms.info["charge"] = int(atoms_a.info["charge"] + atoms_b.info["charge"])
        atoms.calc = calc
        atoms_a.calc = copy(calc)
        atoms_b.calc = copy(calc)

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
