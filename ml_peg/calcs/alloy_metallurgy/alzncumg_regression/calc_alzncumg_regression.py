"""Run calculations for the Al-Cu-Mg-Zn metallurgy regression benchmark."""

from __future__ import annotations

from collections import Counter
from copy import copy
import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.io import read, write
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
OQMD_PATH = DATA_PATH / "structures" / "OQMD-Dumps"
STRUCTURE_IDS = ("8100", "635950", "9226", "122929")


def load_oqmd_structure(oqmd_id: str) -> Atoms:
    """
    Load one staged OQMD structure.

    Parameters
    ----------
    oqmd_id
        OQMD identifier without the ``OQMD_`` prefix.

    Returns
    -------
    Atoms
        ASE structure with OQMD metadata attached to ``atoms.info``.
    """
    structure_path = OQMD_PATH / f"OQMD_{oqmd_id}"
    metadata_path = structure_path.with_suffix(".json")

    atoms = read(structure_path, format="vasp")
    atoms.info["oqmd_id"] = oqmd_id
    atoms.info["name"] = f"OQMD_{oqmd_id}"

    if metadata_path.exists():
        with open(metadata_path) as file:
            metadata = json.load(file)
        atoms.info["oqmd_composition"] = metadata.get("OQMD_Composition")
        atoms.info["oqmd_formation_energy"] = _float_or_none(
            metadata.get("OQMD_FormationEnergy")
        )
        atoms.info["oqmd_volume_per_atom"] = _float_or_none(
            metadata.get("OQMD_Volumepa")
        )

    return atoms


def _float_or_none(value: object) -> float | None:
    """Convert string metadata values to floats when possible."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def element_counts(atoms: Atoms) -> Counter[str]:
    """
    Count elements in a structure.

    Parameters
    ----------
    atoms
        Structure to inspect.

    Returns
    -------
    Counter[str]
        Number of atoms for each element.
    """
    return Counter(atoms.get_chemical_symbols())


def formation_energy_per_atom(
    atoms: Atoms,
    total_energy: float,
    reference_energies: dict[str, float],
) -> float:
    """
    Calculate formation energy per atom.

    Parameters
    ----------
    atoms
        Structure corresponding to ``total_energy``.
    total_energy
        Total potential energy in eV.
    reference_energies
        Elemental reference energies in eV/atom.

    Returns
    -------
    float
        Formation energy in eV/atom.
    """
    reference_total = sum(
        count * reference_energies[element]
        for element, count in element_counts(atoms).items()
    )
    return (total_energy - reference_total) / len(atoms)


def get_elemental_reference_energies(
    structures: dict[str, Atoms],
    energies: dict[str, float],
) -> dict[str, float]:
    """
    Get elemental reference energies from pure structures.

    Parameters
    ----------
    structures
        Structures keyed by OQMD ID.
    energies
        Total energies keyed by OQMD ID.

    Returns
    -------
    dict[str, float]
        Lowest available pure-element energy per atom for each element.
    """
    reference_energies: dict[str, float] = {}
    for oqmd_id, atoms in structures.items():
        counts = element_counts(atoms)
        if len(counts) != 1:
            continue
        element = next(iter(counts))
        energy_per_atom = energies[oqmd_id] / len(atoms)
        current_energy = reference_energies.get(element)
        if current_energy is None or energy_per_atom < current_energy:
            reference_energies[element] = energy_per_atom
    return reference_energies


def structure_properties(
    atoms: Atoms,
    total_energy: float,
    reference_energies: dict[str, float],
) -> dict[str, Any]:
    """
    Build the scalar output record for one structure.

    Parameters
    ----------
    atoms
        Calculated structure.
    total_energy
        Total potential energy in eV.
    reference_energies
        Elemental reference energies in eV/atom.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable scalar properties.
    """
    lengths = atoms.cell.lengths()
    angles = atoms.cell.angles()
    return {
        "oqmd_id": atoms.info["oqmd_id"],
        "formula": atoms.get_chemical_formula(empirical=True),
        "potential_energy": total_energy,
        "formation_energy": formation_energy_per_atom(
            atoms, total_energy, reference_energies
        ),
        "volume_peratom": atoms.get_volume() / len(atoms),
        "lattice_a": lengths[0],
        "lattice_b": lengths[1],
        "lattice_c": lengths[2],
        "angle_alpha": angles[0],
        "angle_beta": angles[1],
        "angle_gamma": angles[2],
        "oqmd_formation_energy": atoms.info.get("oqmd_formation_energy"),
        "oqmd_volume_peratom": atoms.info.get("oqmd_volume_per_atom"),
    }


@pytest.mark.parametrize("mlip", MODELS.items())
def test_alzncumg_regression(mlip: tuple[str, Any]) -> None:
    """
    Run the first bulk-structure metallurgy regression slice.

    Parameters
    ----------
    mlip
        Model name and model instance used to get an ASE calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    output_dir = OUT_PATH / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    structures = {oqmd_id: load_oqmd_structure(oqmd_id) for oqmd_id in STRUCTURE_IDS}
    energies: dict[str, float] = {}

    for oqmd_id, atoms in structures.items():
        atoms.calc = copy(calc)
        try:
            energies[oqmd_id] = float(atoms.get_potential_energy())
        except Exception as exc:
            warn(
                f"Error calculating OQMD_{oqmd_id} with {model_name}: {exc}",
                stacklevel=2,
            )

    reference_energies = get_elemental_reference_energies(structures, energies)
    records = []
    for oqmd_id, total_energy in energies.items():
        atoms = structures[oqmd_id]
        record = structure_properties(atoms, total_energy, reference_energies)
        atoms.info.update(record)
        records.append(record)
        write(output_dir / f"OQMD_{oqmd_id}.xyz", atoms)

    with open(output_dir / "bulk_properties.json", "w") as file:
        json.dump(
            {
                "elemental_reference_energies": reference_energies,
                "structures": records,
            },
            file,
            indent=2,
        )
