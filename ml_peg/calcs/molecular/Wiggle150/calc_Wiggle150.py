"""Run calculations for Wiggle150 benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory to store output data
OUT_PATH = Path(__file__).parent / "outputs"

# Unit conversion
EV_TO_KCAL_PER_MOL = units.mol / units.kcal

# Dataset constants
MOLECULE_ORDER = ("ado", "bpn", "efa")


def parse_structure_info(atoms: Atoms) -> tuple[str, float]:
    """
    Extract structure identifier and reference energy from Atoms.info.

    Parameters
    ----------
    atoms
        Input structure with metadata stored in info keys.

    Returns
    -------
    tuple[str, float]
        Structure label and absolute reference energy.
    """
    info_keys = sorted(atoms.info.keys())
    if len(info_keys) < 2:
        raise ValueError("Structure metadata missing required keys.")

    ref_energy = float(info_keys[0])
    label = info_keys[1]
    return label, ref_energy


def load_structures(data_dir: Path) -> dict[str, dict[str, Iterable[Atoms]]]:
    """
    Load Wiggle150 structures and organise by molecule.

    Parameters
    ----------
    data_dir
        Directory containing the extracted dataset.

    Returns
    -------
    dict
        Mapping of molecule id to ground state and conformers.
    """
    structures = read(data_dir / "ct5c00015_si_003.xyz", ":")

    molecules: dict[str, dict[str, list[Atoms]]] = {
        mol: {"ground": None, "conformers": []} for mol in MOLECULE_ORDER
    }

    for atoms in structures:
        label, _ = parse_structure_info(atoms)
        molecule = label.split("_")[0]
        if molecule not in molecules:
            continue

        if label.endswith("_00"):
            molecules[molecule]["ground"] = atoms
        else:
            molecules[molecule]["conformers"].append(atoms)

        # Set default charge and spin
        atoms.info.setdefault("charge", 0)
        atoms.info.setdefault("spin", 1)

    for mol, entries in molecules.items():
        if entries["ground"] is None:
            raise FileNotFoundError(f"Missing ground-state structure for {mol}.")

    return molecules


def get_energy(atoms: Atoms, calc: Calculator) -> float:
    """
    Evaluate potential energy for a structure.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    calc
        ASE calculator instance.

    Returns
    -------
    float
        Potential energy in eV.
    """
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    energy = atoms_copy.get_potential_energy()
    return float(energy)


def benchmark_wiggle150(
    calc: Calculator, model_name: str, data_dir: Path
) -> list[Atoms]:
    """
    Run Wiggle150 benchmark for a given calculator.

    Parameters
    ----------
    calc
        ASE calculator for predictions.
    model_name
        Name of the model.
    data_dir
        Path to extracted Wiggle150 data.

    Returns
    -------
    list[Atoms]
        Conformer structures annotated with prediction metadata.
    """
    molecules = load_structures(data_dir)
    conformer_atoms: list[Atoms] = []

    for molecule in MOLECULE_ORDER:
        entries = molecules[molecule]
        ground = entries["ground"]
        ground_label, ground_ref_energy = parse_structure_info(ground)
        ground_energy_model = get_energy(ground, calc)

        for atoms in tqdm(entries["conformers"], desc=f"Wiggle150 {molecule.upper()}"):
            label, ref_energy = parse_structure_info(atoms)
            model_energy = get_energy(atoms, calc)

            rel_energy_model_kcal = (
                model_energy - ground_energy_model
            ) * EV_TO_KCAL_PER_MOL
            rel_energy_ref_kcal = ref_energy - ground_ref_energy
            error_kcal = rel_energy_model_kcal - rel_energy_ref_kcal

            annotated = atoms.copy()
            annotated.calc = None
            annotated.info.clear()
            annotated.info.update(
                {
                    "structure": label,
                    "molecule": molecule,
                    "ground_state": ground_label,
                    "relative_energy_ref_kcal": rel_energy_ref_kcal,
                    "relative_energy_pred_kcal": rel_energy_model_kcal,
                    "relative_energy_error_kcal": error_kcal,
                    "model": model_name,
                }
            )
            conformer_atoms.append(annotated)

    return conformer_atoms


@pytest.mark.parametrize("mlip", MODELS.items())
def test_wiggle150(mlip: tuple[str, Any]) -> None:
    """
    Run Wiggle150 benchmark via pytest.

    Parameters
    ----------
    mlip
        Name of model use and model to get calculator.
    """
    model_name, model = mlip
    print(f"\nEvaluating with model: {model_name}")
    calc = model.get_calculator()

    # Add D3 calculator for this test
    calc = model.add_d3_calculator(calc)

    # Download data
    data_dir = (
        download_s3_data(
            key="inputs/molecular/Wiggle150/Wiggle150.zip",
            filename="Wiggle150.zip.zip",
        )
        / "wiggle150-structures"
    )

    conformers = benchmark_wiggle150(calc, model_name, data_dir)

    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)

    for index, atoms in enumerate(conformers):
        atoms.info["index"] = index
        write(write_dir / f"{index}.xyz", atoms, format="extxyz")
