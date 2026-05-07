"""
Compute the QUID benchmark for ligand-pocket interactions.

Puleva, M., Medrano Sandonas, L., Lőrincz, B.D. et al,
Extending quantum-mechanical benchmark accuracy to biological ligand-pocket
interactions,
Nat Commun 16, 8583 (2025). https://doi.org/10.1038/s41467-025-63587-9
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import write
import h5py
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

OUT_PATH = Path(__file__).parent / "outputs"

REF_PRIORITY = [
    "CCSDT",
    "LNO-CCSD(T)",
    "CCSD(T)",
]


def choose_best_reference(
    eint: dict[str, float] | None, force_key: str | None = None
) -> tuple[str, float] | None:
    """
    Choose the best reference method and energy, given a dataset energies.

    Parameters
    ----------
    eint
        Dictionary of reference methods and interaction energies.
    force_key
        Whether to force one particular reference method key.

    Returns
    -------
    Tuple[str, float]
        Tuple containing the best method key and the interaction energy.
    """
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
        return fk, float(eint[fk][()])
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
        return "", np.nan
    return best_key, float(eint[best_key][()])


def compute_interaction_energy(dataset, label, calc):
    """
    Compute and return the energy of the complex with given label and calc.

    Parameters
    ----------
    dataset
        HDF5 dataset containing the systems.
    label
        Label of the system within the dataset.
    calc
        Calculator to use.

    Returns
    -------
    list
        List containing the dimer and monomer ASE Atoms objects.
    """
    best_reference, ref_int_energy = choose_best_reference(dataset[label]["Eint"])
    if np.isnan(ref_int_energy):
        return []
    # List to store dimer and monomers.
    atoms_list = []
    model_int_energy = 0
    try:
        for atoms_name, stoich in zip(
            ["dimer", "small_monomer", "big_monomer"], [1, -1, -1], strict=False
        ):
            atomic_numbers = dataset[label]["atoms"][atoms_name][:]
            positions = dataset[label]["positions"][atoms_name][:]
            atoms = Atoms(numbers=atomic_numbers, positions=positions)
            atoms.info.update({"charge": 0, "spin": 1})
            atoms.calc = calc
            model_int_energy += atoms.get_potential_energy() * stoich
            atoms.calc = None
            atoms_list.append(atoms)
    except Exception as exc:
        warn(f"Error calculating interaction energy for {label}: {exc}", stacklevel=2)
        model_int_energy = np.nan
        for atoms in atoms_list:
            atoms.calc = None
    atoms_list[0].info.update(
        {
            "ref_int_energy": ref_int_energy,
            "model_int_energy": model_int_energy,
            "reference_used": best_reference,
        }
    )
    return atoms_list


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
    calc = model.get_calculator(precision="high")
    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    data_path = (
        download_s3_data(
            filename="QUID.zip",
            key="inputs/non_covalent_interactions/QUID/QUID.zip",
        )
        / "QUID"
    )

    dataset = h5py.File(data_path / "QUID.h5")
    for eq_label in tqdm(dataset.keys(), "Equilibrium"):
        # Get equilibrium config.
        atoms_list = compute_interaction_energy(dataset, eq_label, calc)
        if len(atoms_list) == 0:
            raise ValueError("No structures found")
        write_dir = OUT_PATH / model_name
        write_dir.mkdir(parents=True, exist_ok=True)
        write(write_dir / f"{eq_label}.xyz", atoms_list)

        if "dissociation" not in dataset[eq_label]:
            continue
        dissoc_dataset = dataset[eq_label]["dissociation"]
        for dissoc_label in tqdm(dissoc_dataset.keys(), "Dissociation"):
            # Get dissociation config.
            atoms_list = compute_interaction_energy(dissoc_dataset, dissoc_label, calc)
            if len(atoms_list) == 0:
                continue
            write_dir = OUT_PATH / model_name
            write_dir.mkdir(parents=True, exist_ok=True)
            write(write_dir / f"{dissoc_label}.xyz", atoms_list)
