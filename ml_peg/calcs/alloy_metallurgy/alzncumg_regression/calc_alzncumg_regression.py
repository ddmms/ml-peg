"""Run calculations for the Al-Cu-Mg-Zn metallurgy regression benchmark."""

from __future__ import annotations

from collections import Counter
from copy import copy
import json
import math
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.optimize import FIRE
from ase.units import GPa
import numpy as np
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
OQMD_PATH = DATA_PATH / "structures" / "OQMD-Dumps"
STRUCTURE_IDS = (
    "8100",
    "635950",
    "9226",
    "122929",
    "695020",
    "10434",
    "NOTINOQMD_00001",
    "NOTINOQMD_00002",
)
ELASTIC_STRAIN = 0.005
ELASTIC_CONSTANTS = tuple(
    f"C_{row + 1}{column + 1}" for row in range(6) for column in range(row + 1)
)
SOLUTE_SOLUTE_SPECS = (
    (
        "8100",
        (
            ("Zn", "Zn"),
            ("Zn", "Cu"),
            ("Zn", "Mg"),
            ("Cu", "Cu"),
            ("Cu", "Mg"),
            ("Cu", "Vac"),
            ("Mg", "Mg"),
            ("Mg", "Vac"),
            ("Vac", "Vac"),
        ),
        (4, 4, 4),
        8,
    ),
    ("635950", (("Al", "Al"), ("Al", "Vac"), ("Vac", "Vac")), (3, 3, 3), 4),
)
SOLUTE_RELAX_STEPS = 50
SOLUTE_RELAX_FMAX = 0.005


def ordered_solute_pairs(elements: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return unique solute-pair combinations in legacy evalpot order."""
    pairs = []
    for first_index, first_element in enumerate(elements):
        for second_element in elements[first_index:]:
            pairs.append((first_element, second_element))
    return pairs


def solute_pair_reference_key(
    matrix_oqmd_id: str,
    solute_1: str,
    solute_2: str,
) -> str:
    """Build the evalpot reference key prefix for a solute-solute pair."""
    ordered_pair = sorted((solute_1, solute_2))
    return f"{matrix_oqmd_id}-SolSol_{ordered_pair[0]}_{ordered_pair[1]}"


def conventional_fcc_supercell(
    matrix_oqmd_id: str,
    repeats: tuple[int, int, int],
) -> Atoms:
    """Build a conventional FCC matrix supercell from a staged pure structure."""
    primitive = load_oqmd_structure(matrix_oqmd_id)
    counts = element_counts(primitive)
    if len(counts) != 1:
        raise ValueError(f"OQMD_{matrix_oqmd_id} is not a pure-element matrix")

    matrix_element = next(iter(counts))
    conventional_lattice = min(primitive.cell.lengths()) * math.sqrt(2.0)
    supercell = bulk(matrix_element, "fcc", a=conventional_lattice, cubic=True)
    supercell = supercell.repeat(repeats)
    supercell.info["oqmd_id"] = matrix_oqmd_id
    supercell.info["matrix_element"] = matrix_element
    return supercell


def neighbor_shell_indices(atoms: Atoms) -> list[int]:
    """Get one representative atom index for each distance shell from atom zero."""
    positions = atoms.get_positions()
    cell_lengths = atoms.cell.lengths()
    tolerance = 0.1
    candidates = []
    for index, position in enumerate(positions):
        outside_half_cell = any(
            position[axis] > cell_lengths[axis] / 2.0 + tolerance for axis in range(3)
        )
        if outside_half_cell:
            continue
        distance = round(float(np.linalg.norm(position)), 9)
        candidates.append((distance, index))

    shell_indices = []
    seen_distances = set()
    for distance, index in sorted(candidates):
        if distance in seen_distances:
            continue
        seen_distances.add(distance)
        shell_indices.append(index)
    return shell_indices


def attach_calculator(atoms: Atoms, calculator: Calculator) -> Atoms:
    """Attach a copied calculator to an atoms object and return the atoms."""
    atoms.calc = copy(calculator)
    return atoms


def relax_atoms(
    atoms: Atoms,
    *,
    steps: int = 0,
    fmax: float = SOLUTE_RELAX_FMAX,
) -> None:
    """Relax atomic positions in-place when requested."""
    if steps <= 0:
        return
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)


def solute_structure(
    pure_structure: Atoms,
    solute_1: str,
    solute_2: str | None = None,
    second_solute_index: int | None = None,
) -> Atoms:
    """Build a single-solute or solute-pair structure from a pure matrix."""
    structure = pure_structure.copy()
    if solute_2 is not None:
        if second_solute_index is None:
            raise ValueError(
                "second_solute_index is required for solute-pair structures"
            )
        if solute_2 == "Vac":
            del structure[second_solute_index]
        else:
            structure[second_solute_index].symbol = solute_2

    if solute_1 == "Vac":
        del structure[0]
    else:
        structure[0].symbol = solute_1
    return structure


def relaxed_energy(
    atoms: Atoms,
    calculator: Calculator,
    *,
    relax_steps: int,
    relax_fmax: float,
) -> float:
    """Calculate an energy after optional atomic relaxation."""
    attach_calculator(atoms, calculator)
    relax_atoms(atoms, steps=relax_steps, fmax=relax_fmax)
    return float(atoms.get_potential_energy())


def solute_solute_binding(
    pure_structure: Atoms,
    calculator: Calculator,
    solute_1: str,
    solute_2: str,
    *,
    max_shells: int,
    relax_steps: int = SOLUTE_RELAX_STEPS,
    relax_fmax: float = SOLUTE_RELAX_FMAX,
) -> tuple[list[float], list[float]]:
    """Calculate solute-solute binding energies for FCC neighbor shells."""
    shell_indices = neighbor_shell_indices(pure_structure)[1 : max_shells + 1]
    pure_energy = relaxed_energy(
        pure_structure.copy(), calculator, relax_steps=0, relax_fmax=relax_fmax
    )
    single_1_energy = relaxed_energy(
        solute_structure(pure_structure, solute_1),
        calculator,
        relax_steps=relax_steps,
        relax_fmax=relax_fmax,
    )
    if solute_1 == solute_2:
        single_2_energy = single_1_energy
    else:
        single_2_energy = relaxed_energy(
            solute_structure(pure_structure, solute_2),
            calculator,
            relax_steps=relax_steps,
            relax_fmax=relax_fmax,
        )

    distances = []
    binding_energies = []
    for shell_index in shell_indices:
        pair_structure = solute_structure(
            pure_structure,
            solute_1,
            solute_2,
            second_solute_index=shell_index,
        )
        pair_energy = relaxed_energy(
            pair_structure,
            calculator,
            relax_steps=relax_steps,
            relax_fmax=relax_fmax,
        )
        distances.append(float(np.linalg.norm(pure_structure[shell_index].position)))
        binding_energies.append(
            1000.0 * (pair_energy + pure_energy - single_1_energy - single_2_energy)
        )
    return distances, binding_energies


def voigt_strain_matrix(component: int, magnitude: float) -> np.ndarray:
    """
    Build a small strain matrix for one Voigt component.

    Parameters
    ----------
    component
        Voigt component index in ASE order: xx, yy, zz, yz, xz, xy.
    magnitude
        Engineering strain magnitude for the selected component.

    Returns
    -------
    np.ndarray
        Symmetric 3x3 strain matrix.
    """
    strain = np.zeros((3, 3))
    if component < 3:
        strain[component, component] = magnitude
        return strain

    shear_pairs = ((1, 2), (0, 2), (0, 1))
    row, column = shear_pairs[component - 3]
    strain[row, column] = magnitude / 2.0
    strain[column, row] = magnitude / 2.0
    return strain


def strained_atoms(atoms: Atoms, component: int, magnitude: float) -> Atoms:
    """
    Apply one small Voigt strain to a copy of a structure.

    Parameters
    ----------
    atoms
        Structure to strain.
    component
        Voigt component index in ASE order: xx, yy, zz, yz, xz, xy.
    magnitude
        Engineering strain magnitude for the selected component.

    Returns
    -------
    Atoms
        Strained copy with atoms scaled by the cell deformation.
    """
    strained = atoms.copy()
    deformation = np.eye(3) + voigt_strain_matrix(component, magnitude)
    strained.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
    return strained


def stress_voigt(atoms: Atoms, calculator: Calculator) -> np.ndarray:
    """
    Calculate one stress vector in ASE Voigt order.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    calculator
        ASE calculator that supports stress.

    Returns
    -------
    np.ndarray
        Stress vector in eV/Angstrom^3.
    """
    atoms.calc = copy(calculator)
    return np.asarray(atoms.get_stress(voigt=True), dtype=float)


def finite_strain_elastic_tensor(
    atoms: Atoms,
    calculator: Calculator,
    strain: float = ELASTIC_STRAIN,
) -> np.ndarray:
    """
    Estimate an elastic tensor by central finite differences of stress.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    calculator
        ASE calculator that supports stress.
    strain
        Engineering strain magnitude used for each finite difference.

    Returns
    -------
    np.ndarray
        6x6 elastic tensor in GPa.
    """
    tensor = np.zeros((6, 6))
    for component in range(6):
        positive = strained_atoms(atoms, component, strain)
        negative = strained_atoms(atoms, component, -strain)
        tensor[:, component] = (
            stress_voigt(positive, calculator) - stress_voigt(negative, calculator)
        ) / (2.0 * strain * GPa)
    return (tensor + tensor.T) / 2.0


def elastic_properties(atoms: Atoms, elastic_tensor: np.ndarray) -> dict[str, Any]:
    """
    Build scalar elastic properties from a Voigt elastic tensor.

    Parameters
    ----------
    atoms
        Structure corresponding to ``elastic_tensor``.
    elastic_tensor
        6x6 elastic tensor in GPa.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable elastic properties.
    """
    tensor = np.asarray(elastic_tensor, dtype=float)
    tensor = (tensor + tensor.T) / 2.0
    k_voigt = (
        tensor[0, 0]
        + tensor[1, 1]
        + tensor[2, 2]
        + 2.0 * (tensor[0, 1] + tensor[0, 2] + tensor[1, 2])
    ) / 9.0
    g_voigt = (
        tensor[0, 0]
        + tensor[1, 1]
        + tensor[2, 2]
        - tensor[0, 1]
        - tensor[0, 2]
        - tensor[1, 2]
        + 3.0 * (tensor[3, 3] + tensor[4, 4] + tensor[5, 5])
    ) / 15.0

    properties: dict[str, Any] = {
        "oqmd_id": atoms.info["oqmd_id"],
        "formula": atoms.get_chemical_formula(empirical=True),
        "elastic_tensor": tensor.tolist(),
        "k_voigt": round(float(k_voigt), 2),
        "g_voigt": round(float(g_voigt), 2),
    }
    for row in range(6):
        for column in range(row + 1):
            properties[f"C_{row + 1}{column + 1}"] = round(
                float(tensor[row, column]), 2
            )
    return properties


def structure_file_stem(oqmd_id: str) -> str:
    """Get the staged structure filename stem for an OQMD-like identifier."""
    if oqmd_id.startswith("NOTINOQMD_"):
        return oqmd_id
    return f"OQMD_{oqmd_id}"


def load_oqmd_structure(oqmd_id: str) -> Atoms:
    """
    Load one staged OQMD structure.

    Parameters
    ----------
    oqmd_id
        OQMD identifier without the ``OQMD_`` prefix, or a ``NOTINOQMD_*``
        legacy identifier.

    Returns
    -------
    Atoms
        ASE structure with OQMD metadata attached to ``atoms.info``.
    """
    structure_path = OQMD_PATH / structure_file_stem(oqmd_id)
    metadata_path = structure_path.with_suffix(".json")

    atoms = read(structure_path, format="vasp")
    atoms.info["oqmd_id"] = oqmd_id
    atoms.info["name"] = structure_file_stem(oqmd_id)

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
        if oqmd_id not in energies:
            continue
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


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_alzncumg_elasticity(mlip: tuple[str, Any]) -> None:
    """
    Run the opt-in elastic-moduli metallurgy regression slice.

    Parameters
    ----------
    mlip
        Model name and model instance used to get an ASE calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    output_dir = OUT_PATH / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for oqmd_id in STRUCTURE_IDS:
        atoms = load_oqmd_structure(oqmd_id)
        try:
            tensor = finite_strain_elastic_tensor(atoms, calc)
        except Exception as exc:
            warn(
                f"Error calculating elastic properties for OQMD_{oqmd_id} "
                f"with {model_name}: {exc}",
                stacklevel=2,
            )
            continue

        record = elastic_properties(atoms, tensor)
        atoms.info.update(record)
        records.append(record)

    with open(output_dir / "elastic_properties.json", "w") as file:
        json.dump({"structures": records}, file, indent=2)


@pytest.mark.very_slow
@pytest.mark.parametrize("mlip", MODELS.items())
def test_alzncumg_solute_solute(mlip: tuple[str, Any]) -> None:
    """
    Run the opt-in solute-solute binding metallurgy regression slice.

    Parameters
    ----------
    mlip
        Model name and model instance used to get an ASE calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    output_dir = OUT_PATH / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for matrix_oqmd_id, solute_pairs, repeats, max_shells in SOLUTE_SOLUTE_SPECS:
        pure_structure = conventional_fcc_supercell(matrix_oqmd_id, repeats)
        for solute_1, solute_2 in solute_pairs:
            reference_key = solute_pair_reference_key(
                matrix_oqmd_id, solute_1, solute_2
            )
            try:
                distances, binding_energies = solute_solute_binding(
                    pure_structure,
                    calc,
                    solute_1,
                    solute_2,
                    max_shells=max_shells,
                )
            except Exception as exc:
                warn(
                    f"Error calculating {reference_key} with {model_name}: {exc}",
                    stacklevel=2,
                )
                continue

            records.append(
                {
                    "matrix_oqmd_id": matrix_oqmd_id,
                    "matrix_element": pure_structure.info["matrix_element"],
                    "solute_1": solute_1,
                    "solute_2": solute_2,
                    "reference_key": reference_key,
                    "distances": distances,
                    "binding_energies": binding_energies,
                }
            )

    with open(output_dir / "solute_solute_bindings.json", "w") as file:
        json.dump({"interactions": records}, file, indent=2)
