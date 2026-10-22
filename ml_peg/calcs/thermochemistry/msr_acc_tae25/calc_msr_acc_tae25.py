"""Run calculations for MSR_ACC_TAE25 benchmark."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
import numpy as np
import pytest
from tqdm import tqdm

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"

# ground state multiplicities (2S+1)
ELEMENTS = {
    "H": 2,
    "Li": 2,
    "Be": 1,
    "B": 2,
    "C": 3,
    "N": 4,
    "O": 3,
    "F": 2,
    "Na": 2,
    "Mg": 1,
    "Al": 2,
    "Si": 3,
    "P": 4,
    "S": 3,
    "Cl": 2,
}


def get_e0s(calc: Calculator) -> dict[str, float]:
    """
    Calculate isolated atom energies.

    Parameters
    ----------
    calc
        ASE calculator instance to use for energy calculations.

    Returns
    -------
    dict[str, float]
        Isolated atom energies.
    """
    e0s = {}
    for symbol, spin in ELEMENTS.items():
        atoms = Atoms(
            symbol, positions=[[0.0, 0.0, 0.0]], pbc=False, cell=[20.0, 20.0, 20.0]
        )
        atoms.info["spin"] = spin
        atoms.info["charge"] = 0

        atoms.calc = calc
        try:
            energy = atoms.get_potential_energy()
        except Exception as exc:
            warn(
                f"Error calculation for {symbol}: {exc}",
                stacklevel=2,
            )
            energy = np.nan

        e0s[symbol] = energy
        atoms.calc = None
    return e0s


def get_model_atomization_energy(atoms: Atoms, calc: Calculator, e0s: dict) -> float:
    """
    Calculate total atomization energy.

    Parameters
    ----------
    atoms
        Molecular structure.
    calc
        ASE calculator instance to use for energy calculations.
    e0s
        Isolated atom energies.

    Returns
    -------
    float
        Total atomization energy.
    """
    atoms.calc = calc
    total_energy = atoms.get_potential_energy()
    atoms.calc = None

    counts = Counter(atoms.get_chemical_symbols())
    reference = sum(count * e0s[atomic_sym] for atomic_sym, count in counts.items())

    return reference - total_energy


@pytest.mark.fast
@pytest.mark.parametrize("mlip", MODELS.items())
def test_msr_acc_tae25(mlip: tuple[str, Any]) -> None:
    """
    Run msr_acc_tae25 benchmark.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")

    # get isolated atom energies for the model
    e0s = get_e0s(calc)

    structs = read(DATA_PATH / "structs.xyz", index=":")

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    n_failed = 0
    for atoms in tqdm(structs, desc=f"Calculating TAE using {model_name}"):
        atoms.info["charge"] = atoms.info.get("charge", 0)
        atoms.info["spin"] = atoms.info.get("spin", 1)

        try:
            pred = get_model_atomization_energy(atoms, calc, e0s)
        except Exception as exc:
            warn(
                f"Error calculation for {atoms.info['name']}: {exc}",
                stacklevel=2,
            )
            pred = np.nan
            n_failed += 1

        atoms.info["tae_pred"] = pred
        results[atoms.info["name"]] = pred

        write(write_dir / f"{atoms.info['name']}.xyz", atoms)

    with open(write_dir / "tae.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"{model_name}: computed {len(results) - n_failed}/{len(results)} TAE")
