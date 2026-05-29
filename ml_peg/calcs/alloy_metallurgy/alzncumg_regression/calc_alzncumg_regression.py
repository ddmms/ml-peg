"""Run calculations for the Al-Cu-Mg-Zn metallurgy regression benchmark."""

from __future__ import annotations

from collections import Counter
from copy import copy
import json
from pathlib import Path
from typing import Any
from warnings import warn

from ase import Atoms, units
from ase.build import bulk, fcc100, fcc110, fcc111, surface
from ase.calculators.calculator import Calculator
from ase.constraints import FixedLine
from ase.filters import UnitCellFilter
from ase.geometry import get_layers
from ase.io import read, write
from ase.optimize import BFGS
from ase.units import GPa
import numpy as np
from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
import pytest

from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

DATA_PATH = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "outputs"
OQMD_PATH = DATA_PATH / "structures" / "OQMD-Dumps"
SPECIAL_STRUCTURE_PATH = DATA_PATH / "structures" / "special"
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
            ("Zn", "Vac"),
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
    ("635950", (("Al", "Al"), ("Al", "Vac"), ("Vac", "Vac")), (3, 3, 3), 8),
)
SOLUTE_RELAX_STEPS = 50
SOLUTE_RELAX_FMAX = 0.005
BULK_RELAX_STEPS = 1000
BULK_RELAX_FMAX = 1e-6
BULK_RELAX_STRAIN_MASK = (1, 1, 1, 1, 1, 1)
SURFACE_RELAX_STEPS = 1000
SURFACE_RELAX_FMAX = 1e-6
STACKING_FAULT_RELAX_STEPS = 100
STACKING_FAULT_RELAX_FMAX = 0.005
GSF_PRERELAX_STEPS = 1000
GSF_PRERELAX_FMAX = 1e-6
EV_PER_A2_TO_MJ_PER_M2 = (
    units.eV / (units.kJ * 1e-6) / ((units.Angstrom / units.m) ** 2)
)
FCC_SURFACE_IDS = ("8100", "635950")
FCC_SURFACE_LABELS = ("111", "100", "110")
HCP_SURFACE_SPECS = {
    "9226": {
        "0001": (0, 0, 1),
        "10m10": (1, 0, 0),
        "11m20": (1, 1, 0),
        "1m101": (1, 0, 1),
        "10m12": (1, 0, 2),
        "11m21": (1, 1, 1),
        "11m22": (1, 1, 2),
    },
    "122929": {
        "0001": (0, 0, 1),
        "10m10": (1, 0, 0),
        "11m20": (1, 1, 0),
        "1m101": (1, 0, 1),
        "10m12": (1, 0, 2),
        "11m21": (1, 1, 1),
        "11m22": (1, 1, 2),
    },
}
FCC_STACKING_FAULT_SPECS = {
    "StableSF": 2.0 / 3.0,
    "UnStableSF": 5.0 / 6.0,
}
GSF_SPECS = (
    {
        "structure_label": "NOTINOQMD_00002",
        "surface_label": "111",
        "structure_file": "AIIDA_339739",
        "displacements": (
            (0.66666666, 0.91666666),
            (0.33333333, 0.83333333),
            (0.0, 0.75),
            (0.66666666, 0.66666666),
            (0.33333333, 0.58333333),
            (0.0, 0.5),
            (0.66666666, 0.41666666),
            (0.33333333, 0.33333333),
            (0.0, 0.25),
            (0.66666666, 0.16666666),
            (0.33333333, 0.08333333),
            (0.0, 0.0),
        ),
        "zlayers": 8,
        "relax_method": "atoms_z",
        "relax_steps": 200,
        "relax_fmax": 0.005,
    },
    {
        "structure_label": "NOTINOQMD_00001",
        "surface_label": "0m11",
        "structure_file": "AIIDA_481617",
        "displacements": (
            (0.0, 0.0),
            (0.20, 0.35),
            (0.30, 0.00),
            (0.40, 0.20),
            (0.40, 0.45),
            (0.60, 0.60),
            (0.65, 0.25),
            (0.90, 0.60),
        ),
        "zlayers": 10,
        "relax_method": "atoms_cell_z",
        "relax_steps": 200,
        "relax_fmax": 0.005,
    },
)
SOLUTE_STACKING_FAULT_SPECS = (
    ("8100", ("Cu", "Mg", "Zn", "Si"), 4, 4, (0, 1, 2, 3)),
    ("635950", ("Al",), 3, 4, (0, 1, 2, 3)),
)


def ordered_solute_pairs(elements: tuple[str, ...]) -> list[tuple[str, str]]:
    """
    Return unique solute-pair combinations in legacy evalpot order.

    Parameters
    ----------
    elements
        Solute element symbols to pair.

    Returns
    -------
    list[tuple[str, str]]
        All unique ordered pairs ``(a, b)`` with ``a <= b`` by element index.
    """
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
    """
    Build the evalpot reference key prefix for a solute-solute pair.

    Parameters
    ----------
    matrix_oqmd_id
        OQMD identifier for the host matrix structure.
    solute_1
        First solute element symbol (or ``Vac`` for vacancy).
    solute_2
        Second solute element symbol (or ``Vac`` for vacancy).

    Returns
    -------
    str
        Key string of the form ``<oqmd_id>-SolSol_<s1>_<s2>``.
    """
    return f"{matrix_oqmd_id}-SolSol_{solute_1}_{solute_2}"


def conventional_fcc_supercell(
    matrix_oqmd_id: str,
    repeats: tuple[int, int, int],
    calculator: Calculator,
) -> Atoms:
    """
    Build a conventional FCC matrix supercell from a staged pure structure.

    Parameters
    ----------
    matrix_oqmd_id
        OQMD identifier for the pure-element host matrix.
    repeats
        Repetition factors along each lattice direction.
    calculator
        ASE calculator used to relax the primitive cell.

    Returns
    -------
    Atoms
        Relaxed supercell with matrix element info attached.
    """
    relaxed = relaxed_oqmd_structure(matrix_oqmd_id, calculator)
    counts = element_counts(relaxed)
    if len(counts) != 1:
        raise ValueError(f"OQMD_{matrix_oqmd_id} is not a pure-element matrix")

    matrix_element = next(iter(counts))
    conventional = legacy_conventional_structure(relaxed)
    supercell = conventional.repeat(repeats)
    supercell.info["oqmd_id"] = matrix_oqmd_id
    supercell.info["matrix_element"] = matrix_element
    return supercell


def neighbor_shell_indices(atoms: Atoms) -> list[int]:
    """
    Get one representative atom index for each distance shell from atom zero.

    Parameters
    ----------
    atoms
        Structure whose atoms are ordered at positions relative to the origin.

    Returns
    -------
    list[int]
        One atom index per unique shell distance, sorted by increasing distance.
    """
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
    """
    Attach a copied calculator to an atoms object and return the atoms.

    Parameters
    ----------
    atoms
        Structure to attach the calculator to.
    calculator
        ASE calculator to copy and attach.

    Returns
    -------
    Atoms
        The same ``atoms`` object with a fresh copy of ``calculator`` set.
    """
    atoms.calc = copy(calculator)
    return atoms


def surface_area(atoms: Atoms) -> float:
    """
    Return the area spanned by the first two cell vectors.

    Parameters
    ----------
    atoms
        Structure whose cell vectors are used.

    Returns
    -------
    float
        In-plane cell area in Angstrom^2.
    """
    return float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1])))


def elemental_energy_per_atom(atoms: Atoms, calculator: Calculator) -> float:
    """
    Calculate the energy per atom for a pure elemental reference structure.

    Parameters
    ----------
    atoms
        Pure-element structure with a calculator already attached.
    calculator
        ASE calculator used for the single-point energy.

    Returns
    -------
    float
        Total potential energy divided by number of atoms, in eV/atom.
    """
    attach_calculator(atoms, calculator)
    return float(atoms.get_potential_energy()) / len(atoms)


def relaxed_oqmd_structure(oqmd_id: str, calculator: Calculator) -> Atoms:
    """
    Load and relax one OQMD structure using the legacy bulk protocol.

    Parameters
    ----------
    oqmd_id
        OQMD identifier (numeric or ``NOTINOQMD_*``) of the structure to load.
    calculator
        ASE calculator used for the cell-and-atoms relaxation.

    Returns
    -------
    Atoms
        Relaxed structure with OQMD metadata attached.
    """
    atoms = load_oqmd_structure(oqmd_id)
    attach_calculator(atoms, calculator)
    return relax_cell_and_atoms(
        atoms,
        strain_mask=BULK_RELAX_STRAIN_MASK,
        steps=BULK_RELAX_STEPS,
        fmax=BULK_RELAX_FMAX,
    )


def relax_with_fixed_cell(
    atoms: Atoms,
    *,
    steps: int = SURFACE_RELAX_STEPS,
    fmax: float = SURFACE_RELAX_FMAX,
) -> Atoms:
    """
    Relax atomic positions while keeping the cell fixed.

    Parameters
    ----------
    atoms
        Structure to relax in-place.
    steps
        Maximum number of BFGS steps.
    fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    Atoms
        The relaxed ``atoms`` object.
    """
    BFGS(atoms, logfile=None).run(steps=steps, fmax=fmax)
    return atoms


def relax_cell_and_atoms(
    atoms: Atoms,
    *,
    strain_mask: tuple[int, int, int, int, int, int],
    steps: int,
    fmax: float,
) -> Atoms:
    """
    Relax atomic positions and selected cell components.

    Parameters
    ----------
    atoms
        Structure to relax.
    strain_mask
        Six-element Voigt mask selecting which cell degrees of freedom to relax.
    steps
        Maximum number of BFGS steps.
    fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    Atoms
        The relaxed structure extracted from the ``UnitCellFilter``.
    """
    filtered = UnitCellFilter(atoms, mask=strain_mask)
    BFGS(filtered, logfile=None).run(steps=steps, fmax=fmax)
    return filtered.atoms


def relax_cell_and_atoms_direction(
    atoms: Atoms,
    *,
    strain_mask: tuple[int, int, int, int, int, int],
    atom_direction: tuple[int, int, int],
    steps: int,
    fmax: float,
) -> Atoms:
    """
    Relax selected cell components and constrain atoms to one direction.

    Parameters
    ----------
    atoms
        Structure to relax.
    strain_mask
        Six-element Voigt mask selecting which cell degrees of freedom to relax.
    atom_direction
        Cartesian direction vector for the ``FixedLine`` constraint.
    steps
        Maximum number of BFGS steps.
    fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    Atoms
        The relaxed structure extracted from the ``UnitCellFilter``.
    """
    atoms.set_constraint(
        [FixedLine(index, atom_direction) for index in range(len(atoms))]
    )
    filtered = UnitCellFilter(atoms, mask=strain_mask)
    BFGS(filtered, logfile=None).run(steps=steps, fmax=fmax)
    return filtered.atoms


def relax_atoms_direction(
    atoms: Atoms,
    *,
    atom_direction: tuple[int, int, int],
    steps: int,
    fmax: float,
) -> Atoms:
    """
    Relax atoms along a constrained direction while keeping the cell fixed.

    Parameters
    ----------
    atoms
        Structure to relax.
    atom_direction
        Cartesian direction vector for the ``FixedLine`` constraint.
    steps
        Maximum number of BFGS steps.
    fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    Atoms
        The relaxed ``atoms`` object.
    """
    atoms.set_constraint(
        [FixedLine(index, atom_direction) for index in range(len(atoms))]
    )
    BFGS(atoms, logfile=None).run(steps=steps, fmax=fmax)
    return atoms


def legacy_conventional_structure(atoms: Atoms) -> Atoms:
    """
    Return pymatgen's conventional standard structure as ASE Atoms.

    Parameters
    ----------
    atoms
        Input ASE structure to standardise.

    Returns
    -------
    Atoms
        Conventional standard structure in ASE format.
    """
    structure = AseAtomsAdaptor.get_structure(atoms)
    conventional = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
    return AseAtomsAdaptor.get_atoms(conventional)


def fcc_lattice_and_element(reference: Atoms) -> tuple[float, str]:
    """
    Get a conventional FCC lattice constant and element from a pure structure.

    Parameters
    ----------
    reference
        Relaxed pure-element FCC structure.

    Returns
    -------
    tuple[float, str]
        Conventional lattice constant in Angstrom and element symbol.
    """
    counts = element_counts(reference)
    if len(counts) != 1:
        raise ValueError("Reference structure is not a pure-element FCC reference")

    element = next(iter(counts))
    conventional = legacy_conventional_structure(reference)
    return float(conventional.cell[0][0]), element


def build_fcc_surface(element: str, lattice: float, surface_label: str) -> Atoms:
    """
    Build one periodic FCC slab used by the legacy surface-energy tests.

    Parameters
    ----------
    element
        Chemical symbol of the FCC element.
    lattice
        Conventional FCC lattice constant in Angstrom.
    surface_label
        Miller-index label: ``"111"``, ``"100"``, or ``"110"``.

    Returns
    -------
    Atoms
        Slab structure with 8 Angstrom vacuum added.
    """
    if surface_label == "111":
        slab = fcc111(element, (1, 2, 6), orthogonal=True, a=lattice, periodic=True)
    elif surface_label == "100":
        slab = fcc100(element, (1, 1, 6), a=lattice, periodic=True)
    elif surface_label == "110":
        slab = fcc110(element, (1, 1, 6), a=lattice, periodic=True)
    else:
        raise ValueError(f"Unsupported FCC surface label: {surface_label}")
    slab.cell[2][2] += 8.0
    return slab


def fcc_surface_energy(
    reference: Atoms,
    surface_label: str,
    calculator: Calculator,
    reference_energy: float,
) -> float:
    """
    Calculate an unrelaxed FCC surface energy in mJ/m^2.

    Parameters
    ----------
    reference
        Relaxed pure-element FCC structure used to extract lattice parameters.
    surface_label
        Miller-index label: ``"111"``, ``"100"``, or ``"110"``.
    calculator
        ASE calculator for the slab energy.
    reference_energy
        Elemental reference energy in eV/atom.

    Returns
    -------
    float
        Surface energy in mJ/m^2.
    """
    lattice, element = fcc_lattice_and_element(reference)
    slab = build_fcc_surface(element, lattice, surface_label)
    attach_calculator(slab, calculator)
    slab = relax_with_fixed_cell(slab)
    excess_energy = float(slab.get_potential_energy()) - len(slab) * reference_energy
    return excess_energy / (2.0 * surface_area(slab)) * EV_PER_A2_TO_MJ_PER_M2


def hcp_surface_energy(
    reference: Atoms,
    surface_label: str,
    direction: tuple[int, int, int],
    calculator: Calculator,
    reference_energy: float,
) -> float:
    """
    Calculate an unrelaxed HCP surface energy in mJ/m^2.

    Parameters
    ----------
    reference
        Relaxed pure-element HCP structure used to extract lattice parameters.
    surface_label
        Human-readable label for the surface (used as a record key).
    direction
        Miller index triplet defining the surface orientation.
    calculator
        ASE calculator for the slab energy.
    reference_energy
        Elemental reference energy in eV/atom.

    Returns
    -------
    float
        Surface energy in mJ/m^2.
    """
    counts = element_counts(reference)
    if len(counts) != 1:
        raise ValueError("Reference structure is not a pure-element HCP reference")

    element = next(iter(counts))
    structure_info = legacy_structure_info(reference)
    lattice_a = structure_info["lattice_a"]
    lattice_c = structure_info["lattice_c"]
    slab = surface(
        bulk(element, "hcp", a=lattice_a, covera=lattice_c / lattice_a),
        direction,
        3,
    )
    slab.center(vacuum=20.0, axis=2)
    slab.set_pbc([True, True, True])
    attach_calculator(slab, calculator)
    excess_energy = float(slab.get_potential_energy()) - len(slab) * reference_energy
    return excess_energy / (2.0 * surface_area(slab)) * EV_PER_A2_TO_MJ_PER_M2


def fcc_stacking_fault_energy(
    reference: Atoms,
    displacement_fraction: float,
    calculator: Calculator,
    reference_energy: float,
) -> float:
    """
    Calculate an unrelaxed FCC stacking-fault energy in mJ/m^2.

    Parameters
    ----------
    reference
        Relaxed pure-element FCC structure.
    displacement_fraction
        In-plane displacement as a fraction of the in-plane lattice vector.
    calculator
        ASE calculator for the faulted slab energy.
    reference_energy
        Elemental reference energy in eV/atom.

    Returns
    -------
    float
        Stacking-fault energy in mJ/m^2.
    """
    lattice, element = fcc_lattice_and_element(reference)
    fault = fcc111(element, (1, 2, 6), orthogonal=True, a=lattice, periodic=True)
    fault.cell[2] += fault.cell[1] * displacement_fraction
    attach_calculator(fault, calculator)
    fault = relax_cell_and_atoms_direction(
        fault,
        strain_mask=(0, 0, 1, 0, 0, 0),
        atom_direction=(0, 0, 1),
        steps=STACKING_FAULT_RELAX_STEPS,
        fmax=STACKING_FAULT_RELAX_FMAX,
    )
    excess_energy = float(fault.get_potential_energy()) - len(fault) * reference_energy
    return excess_energy / surface_area(fault) * EV_PER_A2_TO_MJ_PER_M2


def fcc_stacking_fault_structure(
    reference: Atoms,
    inplane_repeats: int,
    zplane_repeats: int,
    *,
    undistorted: bool = False,
) -> Atoms:
    """
    Build the FCC slab used by legacy solute-stacking-fault tests.

    Parameters
    ----------
    reference
        Relaxed pure-element FCC reference structure.
    inplane_repeats
        Number of repetitions along each in-plane lattice direction.
    zplane_repeats
        Number of stacking-fault period repetitions along z.
    undistorted
        When ``True``, omit the fault displacement (undistorted bulk slab).

    Returns
    -------
    Atoms
        FCC slab with or without the intrinsic stacking fault.
    """
    lattice, element = fcc_lattice_and_element(reference)
    fault = fcc111(
        element,
        (inplane_repeats, inplane_repeats, 3 * zplane_repeats),
        a=lattice,
        periodic=True,
    )
    if not undistorted:
        inplane_1 = fault.cell[0] / inplane_repeats
        inplane_2 = fault.cell[1] / inplane_repeats
        fault.cell[2] += (inplane_1 + inplane_2) / 3.0
    return fault


def relaxed_stacking_fault_energy(
    atoms: Atoms,
    calculator: Calculator,
) -> tuple[Atoms, float]:
    """
    Relax a stacking-fault slab with the legacy z-only constraints.

    Parameters
    ----------
    atoms
        Stacking-fault slab to relax.
    calculator
        ASE calculator for the relaxation and energy evaluation.

    Returns
    -------
    tuple[Atoms, float]
        Relaxed slab structure and its total potential energy in eV.
    """
    structure = atoms.copy()
    attach_calculator(structure, calculator)
    structure = relax_cell_and_atoms_direction(
        structure,
        strain_mask=(0, 0, 1, 0, 0, 0),
        atom_direction=(0, 0, 1),
        steps=STACKING_FAULT_RELAX_STEPS,
        fmax=STACKING_FAULT_RELAX_FMAX,
    )
    return structure, float(structure.get_potential_energy())


def solute_stacking_fault_structure(
    base_structure: Atoms,
    solute_element: str,
    solute_layer: int,
) -> Atoms:
    """
    Substitute one atom in a requested z-layer of a relaxed fault slab.

    Parameters
    ----------
    base_structure
        Relaxed fault or bulk slab to modify.
    solute_element
        Chemical symbol of the solute to insert.
    solute_layer
        Zero-based layer index along the z direction.

    Returns
    -------
    Atoms
        Copy of ``base_structure`` with one atom replaced by the solute.
    """
    structure = base_structure.copy()
    layers, _ = get_layers(structure, (0, 0, 1))
    matching_indices = np.argwhere(layers == solute_layer).ravel()
    if len(matching_indices) == 0:
        raise ValueError(f"Layer {solute_layer} is absent from the fault slab")
    structure[int(matching_indices[0])].symbol = solute_element
    return structure


def solute_stacking_fault_interaction(
    matrix_oqmd_id: str,
    solute_element: str,
    calculator: Calculator,
    *,
    inplane_repeats: int,
    zplane_repeats: int,
    solute_layers: tuple[int, ...],
) -> list[float]:
    """
    Calculate relaxed solute-stacking-fault interaction energies in eV.

    Parameters
    ----------
    matrix_oqmd_id
        OQMD identifier for the pure-element host matrix.
    solute_element
        Chemical symbol of the solute to insert.
    calculator
        ASE calculator for all relaxation and energy evaluations.
    inplane_repeats
        Number of repetitions along each in-plane lattice direction.
    zplane_repeats
        Number of stacking-fault period repetitions along z.
    solute_layers
        Zero-based layer indices at which to place the solute.

    Returns
    -------
    list[float]
        Interaction energy in eV for each requested solute layer.
    """
    reference = relaxed_oqmd_structure(matrix_oqmd_id, calculator)
    fault_structure = fcc_stacking_fault_structure(
        reference,
        inplane_repeats,
        zplane_repeats,
    )
    bulk_structure = fcc_stacking_fault_structure(
        reference,
        inplane_repeats,
        zplane_repeats,
        undistorted=True,
    )

    fault_structure, fault_energy = relaxed_stacking_fault_energy(
        fault_structure,
        calculator,
    )
    bulk_structure, bulk_energy = relaxed_stacking_fault_energy(
        bulk_structure,
        calculator,
    )
    solute_bulk_structure = solute_stacking_fault_structure(
        bulk_structure,
        solute_element,
        0,
    )
    _, solute_bulk_energy = relaxed_stacking_fault_energy(
        solute_bulk_structure,
        calculator,
    )

    interaction_energies = []
    for solute_layer in solute_layers:
        solute_fault_structure = solute_stacking_fault_structure(
            fault_structure,
            solute_element,
            solute_layer,
        )
        _, solute_fault_energy = relaxed_stacking_fault_energy(
            solute_fault_structure,
            calculator,
        )
        interaction_energies.append(
            solute_fault_energy - fault_energy - solute_bulk_energy + bulk_energy
        )
    return interaction_energies


def tilted_structure(base_structure: Atoms, displacement: tuple[float, float]) -> Atoms:
    """
    Tilt a cell by a crystallographic displacement in the in-plane basis.

    Parameters
    ----------
    base_structure
        Structure whose cell will be tilted.
    displacement
        Fractional displacements along the first and second cell vectors.

    Returns
    -------
    Atoms
        Copy of ``base_structure`` with the third cell vector displaced.
    """
    tilted = base_structure.copy()
    tilted.cell[2] += base_structure.cell[0] * displacement[0]
    tilted.cell[2] += base_structure.cell[1] * displacement[1]
    return tilted


def generalized_stacking_fault_energies(
    base_structure: Atoms,
    calculator: Calculator,
    displacements: tuple[tuple[float, float], ...],
    *,
    zlayers: int,
    relax_method: str,
    relax_steps: int,
    relax_fmax: float,
) -> tuple[list[float], list[float]]:
    """
    Calculate relaxed GSF raw and zero-referenced energies in eV/A^2.

    Parameters
    ----------
    base_structure
        Base cell to pre-relax and tile before applying fault displacements.
    calculator
        ASE calculator for all energy evaluations.
    displacements
        Sequence of ``(frac_a, frac_b)`` fault displacements; must include
        ``(0.0, 0.0)`` for the reference energy.
    zlayers
        Number of times to repeat the structure along z.
    relax_method
        Relaxation protocol: ``"atoms_z"`` or ``"atoms_cell_z"``.
    relax_steps
        Maximum number of BFGS steps for each fault structure.
    relax_fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    tuple[list[float], list[float]]
        Raw energies per area and energies normalised to the undisplaced point,
        both in eV/Angstrom^2, ordered to match ``displacements``.
    """
    if (0.0, 0.0) not in displacements:
        raise ValueError("GSF displacements must include the undisplaced structure")

    attach_calculator(base_structure, calculator)
    base_structure = relax_cell_and_atoms(
        base_structure,
        strain_mask=(1, 1, 1, 0, 0, 0),
        steps=GSF_PRERELAX_STEPS,
        fmax=GSF_PRERELAX_FMAX,
    )
    repeated = base_structure.repeat((1, 1, zlayers))
    area = surface_area(repeated)
    raw_energies = []
    reference_energy = None
    for displacement in displacements:
        tilted = tilted_structure(repeated, displacement)
        attach_calculator(tilted, calculator)
        if relax_method == "atoms_z":
            tilted = relax_atoms_direction(
                tilted,
                atom_direction=(0, 0, 1),
                steps=relax_steps,
                fmax=relax_fmax,
            )
        elif relax_method == "atoms_cell_z":
            tilted = relax_cell_and_atoms_direction(
                tilted,
                strain_mask=(0, 0, 1, 0, 0, 0),
                atom_direction=(0, 0, 1),
                steps=relax_steps,
                fmax=relax_fmax,
            )
        else:
            raise ValueError(f"Unsupported GSF relaxation method: {relax_method}")
        energy = float(tilted.get_potential_energy()) / area
        raw_energies.append(energy)
        if displacement == (0.0, 0.0):
            reference_energy = energy

    if reference_energy is None:
        raise ValueError("GSF displacements must include the undisplaced structure")
    return raw_energies, [energy - reference_energy for energy in raw_energies]


def relax_atoms(
    atoms: Atoms,
    *,
    steps: int = 0,
    fmax: float = SOLUTE_RELAX_FMAX,
) -> None:
    """
    Relax atomic positions in-place when requested.

    Parameters
    ----------
    atoms
        Structure to relax in-place.
    steps
        Maximum number of BFGS steps; no relaxation is performed when zero.
    fmax
        Force convergence threshold in eV/Angstrom.
    """
    if steps <= 0:
        return
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)


def solute_structure(
    pure_structure: Atoms,
    solute_1: str,
    solute_2: str | None = None,
    second_solute_index: int | None = None,
) -> Atoms:
    """
    Build a single-solute or solute-pair structure from a pure matrix.

    Parameters
    ----------
    pure_structure
        Pure-element host matrix supercell.
    solute_1
        Element symbol for the origin-site solute (or ``"Vac"`` for vacancy).
    solute_2
        Element symbol for the second solute when building a pair, or
        ``None`` for a single-solute structure.
    second_solute_index
        Atom index for the second solute site; required when ``solute_2`` is
        not ``None``.

    Returns
    -------
    Atoms
        Copy of ``pure_structure`` with the requested substitution(s) applied.
    """
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
    """
    Calculate an energy after optional atomic relaxation.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    calculator
        ASE calculator to attach before evaluation.
    relax_steps
        Maximum number of BFGS steps; no relaxation when zero.
    relax_fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    float
        Total potential energy in eV after optional relaxation.
    """
    attach_calculator(atoms, calculator)
    relax_atoms(atoms, steps=relax_steps, fmax=relax_fmax)
    return float(atoms.get_potential_energy())


def solute_solute_binding(
    pure_structure: Atoms,
    calculator: Calculator,
    solute_1: str,
    solute_2: str,
    *,
    max_index: int,
    relax_steps: int = SOLUTE_RELAX_STEPS,
    relax_fmax: float = SOLUTE_RELAX_FMAX,
) -> tuple[list[float], list[float]]:
    """
    Calculate solute-solute binding energies for FCC neighbor shells.

    Parameters
    ----------
    pure_structure
        Relaxed pure-element FCC supercell built from the matrix.
    calculator
        ASE calculator for all energy evaluations.
    solute_1
        First solute element symbol (or ``"Vac"`` for vacancy).
    solute_2
        Second solute element symbol (or ``"Vac"`` for vacancy).
    max_index
        Upper bound for the neighbor shell slice (exclusive); shells
        ``1`` through ``max_index - 1`` are computed, matching evalpot
        ``range(1, max_index)``.
    relax_steps
        Maximum number of BFGS steps for solute structures.
    relax_fmax
        Force convergence threshold in eV/Angstrom.

    Returns
    -------
    tuple[list[float], list[float]]
        Shell distances in Angstrom and binding energies in meV,
        ordered from nearest to furthest shell.
    """
    shell_indices = neighbor_shell_indices(pure_structure)[1:max_index]
    pure_energy = relaxed_energy(
        pure_structure.copy(), calculator, relax_steps=0, relax_fmax=relax_fmax
    )
    single_1_structure = solute_structure(pure_structure, solute_1)
    single_1_energy = relaxed_energy(
        single_1_structure,
        calculator,
        relax_steps=relax_steps,
        relax_fmax=relax_fmax,
    )
    if solute_1 == "Vac":
        first_solute_pair_base = pure_structure.copy()
    else:
        first_solute_pair_base = single_1_structure

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
        pair_structure = first_solute_pair_base.copy()
        distances.append(float(np.linalg.norm(pair_structure[shell_index].position)))
        if solute_2 == "Vac":
            del pair_structure[shell_index]
        else:
            pair_structure[shell_index].symbol = solute_2
        if solute_1 == "Vac":
            del pair_structure[0]
        pair_energy = relaxed_energy(
            pair_structure,
            calculator,
            relax_steps=relax_steps,
            relax_fmax=relax_fmax,
        )
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


def legacy_elastic_tensor(
    atoms: Atoms,
    calculator: Calculator,
    *,
    strain_amounts: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
) -> np.ndarray:
    """
    Calculate elastic constants using the legacy pymatgen strain fit.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    calculator
        ASE calculator that supports stress.
    strain_amounts
        Engineering strain magnitudes applied for the finite-difference fit.

    Returns
    -------
    np.ndarray
        6x6 elastic tensor in GPa.
    """
    reference = atoms.copy()
    attach_calculator(reference, calculator)
    eq_stress = Stress.from_voigt(reference.get_stress())
    structure = AseAtomsAdaptor.get_structure(reference)
    deformed_set = DeformedStructureSet(
        structure,
        norm_strains=strain_amounts,
        shear_strains=strain_amounts,
    )

    applied_strains = []
    resultant_stresses = []
    for deformation in deformed_set.deformations:
        applied_strains.append(deformation.green_lagrange_strain)
        deformed_structure = deformation.apply_to_structure(structure.copy())
        deformed_atoms = AseAtomsAdaptor.get_atoms(deformed_structure)
        attach_calculator(deformed_atoms, calculator)
        relax_with_fixed_cell(deformed_atoms, steps=100, fmax=0.001)
        resultant_stresses.append(Stress.from_voigt(deformed_atoms.get_stress()))

    tensor = ElasticTensor.from_independent_strains(
        strains=applied_strains,
        stresses=resultant_stresses,
        eq_stress=eq_stress,
    )
    return np.asarray(tensor.voigt_symmetrized.voigt) * (GPa**-1)


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
    """
    Get the staged structure filename stem for an OQMD-like identifier.

    Parameters
    ----------
    oqmd_id
        OQMD numeric identifier or a ``NOTINOQMD_*`` legacy identifier.

    Returns
    -------
    str
        Filename stem: ``OQMD_<oqmd_id>`` for numeric IDs, or the raw ID
        for ``NOTINOQMD_*`` identifiers.
    """
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
    """
    Convert string metadata values to floats when possible.

    Parameters
    ----------
    value
        Value to convert; typically a string from OQMD JSON metadata.

    Returns
    -------
    float | None
        Parsed float, or ``None`` if conversion fails.
    """
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


def legacy_structure_info(atoms: Atoms) -> dict[str, Any]:
    """
    Build structure metadata using the legacy evalpot convention.

    Parameters
    ----------
    atoms
        Structure to describe.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable metadata including formula, volume, lattice
        parameters, angles, and symmetry information.
    """
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure = structure.get_primitive_structure()
    structure = structure.get_reduced_structure()
    angles = [structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma]
    angles = [180.0 - angle if angle > 91.0 else angle for angle in angles]
    symmetry = SpacegroupAnalyzer(structure)
    return {
        "formula": structure.formula.replace(" ", "").replace("1", ""),
        "volume_peratom": structure.volume / len(structure),
        "lattice_a": structure.lattice.a,
        "lattice_b": structure.lattice.b,
        "lattice_c": structure.lattice.c,
        "angle_alpha": angles[0],
        "angle_beta": angles[1],
        "angle_gamma": angles[2],
        "symmetry_number": symmetry.get_space_group_number(),
        "symmetry_symbol": symmetry.get_space_group_symbol(),
    }


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
    structure_info = legacy_structure_info(atoms)
    return {
        "oqmd_id": atoms.info["oqmd_id"],
        "potential_energy": total_energy,
        "formation_energy": formation_energy_per_atom(
            atoms, total_energy, reference_energies
        ),
        "oqmd_formation_energy": atoms.info.get("oqmd_formation_energy"),
        "oqmd_volume_peratom": atoms.info.get("oqmd_volume_per_atom"),
        **structure_info,
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
            atoms = relax_cell_and_atoms(
                atoms,
                strain_mask=BULK_RELAX_STRAIN_MASK,
                steps=BULK_RELAX_STEPS,
                fmax=BULK_RELAX_FMAX,
            )
            structures[oqmd_id] = atoms
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


@pytest.mark.parametrize("mlip", MODELS.items())
def test_alzncumg_fault_surfaces(mlip: tuple[str, Any]) -> None:
    """
    Run fast surface, stacking-fault, and GSF metallurgy calculations.

    Parameters
    ----------
    mlip
        Model name and model instance used to get an ASE calculator.
    """
    model_name, model = mlip
    calc = model.get_calculator(precision="high")
    output_dir = OUT_PATH / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pure_reference_structures = {
        oqmd_id: relaxed_oqmd_structure(oqmd_id, calc)
        for oqmd_id in (*FCC_SURFACE_IDS, *HCP_SURFACE_SPECS)
    }
    pure_reference_energies = {
        oqmd_id: float(atoms.get_potential_energy()) / len(atoms)
        for oqmd_id, atoms in pure_reference_structures.items()
    }

    surface_records = []
    for oqmd_id in FCC_SURFACE_IDS:
        for surface_label in FCC_SURFACE_LABELS:
            reference_key = f"{oqmd_id}-SurfaceEnergy_{surface_label}"
            try:
                surface_energy = fcc_surface_energy(
                    pure_reference_structures[oqmd_id],
                    surface_label,
                    calc,
                    pure_reference_energies[oqmd_id],
                )
            except Exception as exc:
                warn(
                    f"Error calculating {reference_key} with {model_name}: {exc}",
                    stacklevel=2,
                )
                continue
            surface_records.append(
                {
                    "reference_key": reference_key,
                    "structure_label": oqmd_id,
                    "surface_label": surface_label,
                    "surface_energy": surface_energy,
                }
            )

    for oqmd_id, surface_specs in HCP_SURFACE_SPECS.items():
        for surface_label, direction in surface_specs.items():
            reference_key = f"{oqmd_id}-SurfaceHCP_{surface_label}"
            try:
                surface_energy = hcp_surface_energy(
                    pure_reference_structures[oqmd_id],
                    surface_label,
                    direction,
                    calc,
                    pure_reference_energies[oqmd_id],
                )
            except Exception as exc:
                warn(
                    f"Error calculating {reference_key} with {model_name}: {exc}",
                    stacklevel=2,
                )
                continue
            surface_records.append(
                {
                    "reference_key": reference_key,
                    "structure_label": oqmd_id,
                    "surface_label": surface_label,
                    "surface_direction": direction,
                    "surface_energy": surface_energy,
                }
            )

    stacking_fault_records = []
    for oqmd_id in FCC_SURFACE_IDS:
        for fault_label, displacement_fraction in FCC_STACKING_FAULT_SPECS.items():
            reference_key = f"{oqmd_id}-{fault_label}"
            try:
                stacking_fault_energy = fcc_stacking_fault_energy(
                    pure_reference_structures[oqmd_id],
                    displacement_fraction,
                    calc,
                    pure_reference_energies[oqmd_id],
                )
            except Exception as exc:
                warn(
                    f"Error calculating {reference_key} with {model_name}: {exc}",
                    stacklevel=2,
                )
                continue
            stacking_fault_records.append(
                {
                    "reference_key": reference_key,
                    "structure_label": oqmd_id,
                    "fault_label": fault_label,
                    "stacking_fault_energy": stacking_fault_energy,
                }
            )

    gsf_records = []
    for spec in GSF_SPECS:
        reference_key = f"{spec['structure_label']}-GSF_{spec['surface_label']}"
        try:
            base_structure = read(
                SPECIAL_STRUCTURE_PATH / spec["structure_file"], format="vasp"
            )
            raw_energies, norm_energies = generalized_stacking_fault_energies(
                base_structure,
                calc,
                spec["displacements"],
                zlayers=spec["zlayers"],
                relax_method=spec["relax_method"],
                relax_steps=spec["relax_steps"],
                relax_fmax=spec["relax_fmax"],
            )
        except Exception as exc:
            warn(
                f"Error calculating {reference_key} with {model_name}: {exc}",
                stacklevel=2,
            )
            continue
        gsf_records.append(
            {
                "reference_key": reference_key,
                "structure_label": spec["structure_label"],
                "surface_label": spec["surface_label"],
                "displacements": spec["displacements"],
                "raw_energies": raw_energies,
                "norm_energies": norm_energies,
            }
        )

    solute_stacking_fault_records = []
    for spec in SOLUTE_STACKING_FAULT_SPECS:
        matrix_oqmd_id, solute_elements, inplane_repeats, zplane_repeats, layers = spec
        for solute_element in solute_elements:
            reference_key = f"{matrix_oqmd_id}-SolSF_{solute_element}"
            try:
                interaction_energies = solute_stacking_fault_interaction(
                    matrix_oqmd_id,
                    solute_element,
                    calc,
                    inplane_repeats=inplane_repeats,
                    zplane_repeats=zplane_repeats,
                    solute_layers=layers,
                )
            except Exception as exc:
                warn(
                    f"Error calculating {reference_key} with {model_name}: {exc}",
                    stacklevel=2,
                )
                continue
            solute_stacking_fault_records.append(
                {
                    "reference_key": reference_key,
                    "matrix_oqmd_id": matrix_oqmd_id,
                    "solute_element": solute_element,
                    "solute_layers": layers,
                    "interaction_energies": interaction_energies,
                }
            )

    with open(output_dir / "fault_surface_properties.json", "w") as file:
        json.dump(
            {
                "surfaces": surface_records,
                "stacking_faults": stacking_fault_records,
                "gsf": gsf_records,
                "solute_stacking_faults": solute_stacking_fault_records,
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
        try:
            atoms = relaxed_oqmd_structure(oqmd_id, calc)
            tensor = legacy_elastic_tensor(atoms, calc)
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
    for matrix_oqmd_id, solute_pairs, repeats, max_index in SOLUTE_SOLUTE_SPECS:
        pure_structure = conventional_fcc_supercell(matrix_oqmd_id, repeats, calc)
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
                    max_index=max_index,
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
