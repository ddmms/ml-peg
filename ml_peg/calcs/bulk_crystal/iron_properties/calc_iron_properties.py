"""
Run calculations for BCC iron properties benchmark.

This benchmark computes fundamental properties of BCC iron including:
- Equation of state (lattice parameter, bulk modulus)
- Elastic constants (C11, C12, C44)
- Bain path energy curve
- Vacancy formation energy
- Surface energies (100, 110, 111, 112)
- Generalized stacking fault energy curves (110, 112)
- Traction-separation curves (100, 110)

This benchmark is computationally expensive and marked with @pytest.mark.slow.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn

from ase.build import bulk
from ase.constraints import FixedLine
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import BFGS
import numpy as np
import pandas as pd
import pytest

from ml_peg.calcs.bulk_crystal.iron_properties.iron_utils import (
    EV_PER_A2_TO_J_PER_M2,
    EV_PER_A3_TO_GPA,
    apply_voigt_strain,
    calculate_surface_energy,
    create_bain_cell,
    create_bcc_supercell,
    create_sfe_110_structure,
    create_sfe_112_structure,
    create_surface_100,
    create_surface_110,
    create_surface_111,
    create_surface_112,
    fit_eos,
    relax_volume_isotropic,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"

# =============================================================================
# Test Parameters
# =============================================================================

# EOS calculation parameters
REFERENCE_LATTICE_PARAMETER = 2.834
EOS_NUM_POINTS = 30

# BFGS optimization parameters
BFGS_FMAX = 1e-5
BFGS_MAX_ITER = 100

# Elastic constants parameters
ELASTIC_STRAIN = 1.0e-5
ELASTIC_SUPERCELL_SIZE = (4, 4, 4)
ELASTIC_ATOM_JIGGLE = 1.0e-5  # Random perturbation to prevent saddle points

# Bain path parameters
BAIN_NUM_POINTS = 65

# Vacancy calculation parameters
VACANCY_SUPERCELL_SIZE = (3, 3, 3)

# Surface calculation parameters
SURFACE_VACUUM = 10.0  # Angstroms

# Stacking fault calculation parameters
SFE_STEP_SIZE = 0.04

# Traction-separation parameters
TS_MAX_SEPARATION = 5.0  # Angstroms
TS_STEP_SIZE = 0.05  # Angstroms


def _set_iron_info(atoms: Atoms) -> None:
    """
    Set default charge and spin multiplicity for BCC iron structures.

    Parameters
    ----------
    atoms
        ASE Atoms object to annotate.
    """
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1


def _relax(target: Any, label: str) -> dict[str, Any]:
    """
    Run BFGS and return serializable convergence diagnostics.

    Parameters
    ----------
    target
        ASE atoms or filter to optimize.
    label
        Description used in warnings.

    Returns
    -------
    dict[str, Any]
        Convergence status, step count, final force, and error message.
    """
    status: dict[str, Any] = {
        "converged": False,
        "steps": 0,
        "max_force": None,
        "error": None,
    }
    try:
        optimizer = BFGS(target, logfile=None)
        status["converged"] = bool(optimizer.run(fmax=BFGS_FMAX, steps=BFGS_MAX_ITER))
        status["steps"] = int(optimizer.nsteps)
        forces = np.asarray(target.get_forces())
        if forces.size:
            status["max_force"] = float(
                np.max(np.linalg.norm(forces.reshape(-1, 3), axis=1))
            )
        if not status["converged"]:
            warn(f"{label} did not converge in {BFGS_MAX_ITER} steps", stacklevel=2)
    except Exception as exc:
        status["error"] = str(exc)
        warn(f"{label} failed: {exc}", stacklevel=2)
    return status


def _energy(atoms: Atoms, label: str) -> float:
    """
    Evaluate a potential energy, returning NaN on failure.

    Parameters
    ----------
    atoms
        Structure to evaluate.
    label
        Description used in warnings.

    Returns
    -------
    float
        Potential energy or NaN.
    """
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        warn(f"{label} failed: {exc}", stacklevel=2)
        return float("nan")


def _run_section(
    label: str, calculation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """
    Run one independent benchmark section without aborting later sections.

    Parameters
    ----------
    label
        Section name used in warnings.
    calculation
        Callable that performs the section.

    Returns
    -------
    dict[str, Any]
        Section results or a failure record.
    """
    try:
        return calculation()
    except Exception as exc:
        warn(f"{label} failed: {exc}", stacklevel=2)
        return {"status": {"converged": False, "error": str(exc)}}


def _json_safe(value: Any) -> Any:
    """
    Convert nested NumPy and nonfinite values to strict JSON values.

    Parameters
    ----------
    value
        Value to convert.

    Returns
    -------
    Any
        Strictly JSON-serializable value.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite(value: Any) -> bool:
    """
    Return whether a value is a finite scalar.

    Parameters
    ----------
    value
        Value to inspect.

    Returns
    -------
    bool
        Whether the value is finite.
    """
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


# =============================================================================
# EOS Calculation
# =============================================================================


def run_eos_calculation(calc: Calculator) -> dict[str, Any]:
    """
    Run the energy-volume curve calculation.

    Parameters
    ----------
    calc
        ASE calculator object.

    Returns
    -------
    dict[str, Any]
        Dictionary with EOS results including a0, B0, V0, E0, volumes, energies.
    """
    # Generate lattice parameters: 2.834 - 0.05 + (0.1/30)*i for i in 1..30
    lattice_params = np.array(
        [
            REFERENCE_LATTICE_PARAMETER - 0.05 + 0.1 / 30 * i
            for i in range(1, EOS_NUM_POINTS + 1)
        ]
    )

    volumes = []
    energies = []
    statuses = []

    for lat in lattice_params:
        atoms = bulk("Fe", "bcc", a=lat, cubic=True)
        _set_iron_info(atoms)
        atoms.calc = calc

        # Relax atomic positions at fixed cell volume
        # (matches LAMMPS minimize behavior)
        statuses.append(_relax(atoms, f"EOS relaxation at a={lat:.5f}"))

        energy = _energy(atoms, "EOS energy")
        volume = atoms.get_volume()

        n_atoms = len(atoms)
        volumes.append(volume / n_atoms)
        energies.append(energy / n_atoms)

    volumes = np.array(volumes)
    energies = np.array(energies)

    # Fit Birch-Murnaghan EOS
    eos_results = fit_eos(volumes, energies)

    return {
        "volumes": volumes.tolist(),
        "energies": energies.tolist(),
        "lattice_params": lattice_params.tolist(),
        "a0": eos_results["a0"],
        "E0": eos_results["E0"],
        "B0": eos_results["B0"],
        "Bp": eos_results["Bp"],
        "V0": eos_results["V0"],
        "status": statuses,
    }


# =============================================================================
# Elastic Constants Calculation
# =============================================================================


def run_elastic_calculation(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate elastic constants using the stress-strain method.

    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium lattice parameter from EOS fit.

    Returns
    -------
    dict[str, Any]
        Dictionary with elastic constants C11, C12, C44, bulk_modulus.
    """
    # Create supercell
    atoms_ref = create_bcc_supercell(lattice_parameter, ELASTIC_SUPERCELL_SIZE)
    _set_iron_info(atoms_ref)
    atoms_ref.calc = calc

    # Box relaxation
    status: dict[str, Any] = {
        "cell": _relax(FrechetCellFilter(atoms_ref), "elastic cell relaxation"),
        "strains": [],
    }

    # Apply random jiggle to atoms to prevent staying on saddle points
    rng = np.random.default_rng(seed=87287)
    jiggle = rng.uniform(
        -ELASTIC_ATOM_JIGGLE, ELASTIC_ATOM_JIGGLE, atoms_ref.positions.shape
    )
    atoms_ref.positions += jiggle

    # Elastic constant matrix
    C = np.zeros((6, 6))  # noqa: N806

    for i in range(6):
        direction = i + 1
        direction_status = {"direction": direction}

        # Positive strain with off-diagonal cell adjustment
        atoms_pos = apply_voigt_strain(atoms_ref.copy(), direction, ELASTIC_STRAIN)
        _set_iron_info(atoms_pos)
        atoms_pos.calc = calc
        direction_status["positive"] = _relax(
            atoms_pos, f"elastic direction {direction} positive relaxation"
        )

        try:
            stress_pos = atoms_pos.get_stress(voigt=True)
        except Exception as exc:
            warn(f"Positive strain stress calculation failed: {exc}", stacklevel=2)
            stress_pos = np.nan * np.ones(6)

        # Negative strain with off-diagonal cell adjustment
        atoms_neg = apply_voigt_strain(atoms_ref.copy(), direction, -ELASTIC_STRAIN)
        _set_iron_info(atoms_neg)
        atoms_neg.calc = calc
        direction_status["negative"] = _relax(
            atoms_neg, f"elastic direction {direction} negative relaxation"
        )

        try:
            stress_neg = atoms_neg.get_stress(voigt=True)
        except Exception as exc:
            warn(f"Negative strain stress calculation failed: {exc}", stacklevel=2)
            stress_neg = np.nan * np.ones(6)

        # Compute elastic constants using stress differences
        # C_ij = dσ_i / dε_j = (σ_pos - σ_neg) / (2 * ε)
        delta_stress = stress_pos - stress_neg
        delta_strain = 2 * ELASTIC_STRAIN

        for j in range(6):
            C[j, i] = delta_stress[j] / delta_strain * EV_PER_A3_TO_GPA

        status["strains"].append(direction_status)

    # Symmetrize
    C_sym = 0.5 * (C + C.T)  # noqa: N806

    # Extract cubic averages
    C11 = (C_sym[0, 0] + C_sym[1, 1] + C_sym[2, 2]) / 3  # noqa: N806
    C12 = (C_sym[0, 1] + C_sym[0, 2] + C_sym[1, 2]) / 3  # noqa: N806
    C44 = (C_sym[3, 3] + C_sym[4, 4] + C_sym[5, 5]) / 3  # noqa: N806

    bulk_modulus = (C11 + 2 * C12) / 3

    return {
        "C11": C11,
        "C12": C12,
        "C44": C44,
        "bulk_modulus": bulk_modulus,
        "C_matrix": C_sym.tolist(),
        "status": status,
    }


# =============================================================================
# Bain Path Calculation
# =============================================================================


def run_bain_path_calculation(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate the Bain path energy curve.

    For each target c/a ratio, creates a tetragonally distorted cell and performs
    isotropic volume relaxation (uniform scaling only) to find the minimum energy
    while maintaining the c/a ratio. This matches the LAMMPS behavior where
    'fix box/relax aniso 0.0 couple xyz' is used.

    The 'couple xyz' constraint in LAMMPS couples all three diagonal stress
    components together, meaning x, y, and z dimensions can only change by the
    same fractional amount. This preserves the c/a ratio during relaxation.

    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium BCC lattice parameter.

    Returns
    -------
    dict[str, Any]
        Dictionary with ca_ratios, energies, E_bcc, E_fcc, delta_E.
    """
    # Generate c/a ratios: 0.7 + 0.02*i for i in 1..65, plus exact FCC
    ca_ratios_target = np.array([0.7 + 0.02 * i for i in range(1, BAIN_NUM_POINTS + 1)])
    ca_ratios_target = np.sort(np.append(ca_ratios_target, np.sqrt(2)))

    ca_ratios = []
    energies = []
    statuses = []

    for ratio in ca_ratios_target:
        # Create tetragonally distorted cell at target c/a ratio
        atoms = create_bain_cell(lattice_parameter, ratio)
        _set_iron_info(atoms)
        atoms.calc = calc

        # Step 1: Isotropic volume relaxation (maintains c/a ratio)
        # This is equivalent to LAMMPS: fix box/relax aniso 0.0 couple xyz
        # Only uniform scaling is allowed, preserving the cell shape
        try:
            atoms_relaxed, volume_status = relax_volume_isotropic(atoms, calc)
        except Exception as exc:
            warn(
                f"Bain volume relaxation at c/a={ratio:.6f} failed: {exc}", stacklevel=2
            )
            atoms_relaxed = atoms
            volume_status = {
                "converged": False,
                "scale": 1.0,
                "boundary_hit": False,
                "pressure_GPa": None,
                "error": str(exc),
            }

        # Step 2: Atomic position relaxation at fixed cell
        atomic_status = _relax(
            atoms_relaxed, f"Bain atomic relaxation at c/a={ratio:.6f}"
        )
        energy = _energy(atoms_relaxed, f"Bain energy at c/a={ratio:.6f}")
        cell = atoms_relaxed.get_cell()
        ca_actual = cell[2, 2] / cell[1, 1]

        n_atoms = len(atoms_relaxed)
        ca_ratios.append(ca_actual)
        energies.append(energy / n_atoms)
        statuses.append(
            {
                "target_ca_ratio": float(ratio),
                "actual_ca_ratio": float(ca_actual),
                "volume": volume_status,
                "atoms": atomic_status,
            }
        )

    ca_ratios = np.array(ca_ratios)
    energies = np.array(energies)

    # Normalize energies relative to BCC and convert to meV/atom
    idx_bcc = np.argmin(np.abs(ca_ratios - 1.0))
    idx_fcc = np.argmin(np.abs(ca_ratios - np.sqrt(2)))
    energies_norm = (energies - energies[idx_bcc]) * 1000
    E_bcc = energies_norm[idx_bcc]  # noqa: N806
    E_fcc = energies_norm[idx_fcc]  # noqa: N806

    return {
        "ca_ratios": ca_ratios.tolist(),
        "energies": energies.tolist(),
        "energies_meV": energies_norm.tolist(),
        "E_bcc_meV": E_bcc,
        "E_fcc_meV": E_fcc,
        "delta_E_meV": E_fcc - E_bcc,
        "status": statuses,
    }


# =============================================================================
# Vacancy Calculation
# =============================================================================


def run_vacancy_calculation(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate the vacancy formation energy.

    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium lattice parameter from EOS fit.

    Returns
    -------
    dict[str, Any]
        Dictionary with vacancy results including E_vac, E_coh, E_perfect, E_defect.
    """
    atoms_perfect = create_bcc_supercell(lattice_parameter, VACANCY_SUPERCELL_SIZE)
    _set_iron_info(atoms_perfect)
    atoms_perfect.calc = calc

    n_atoms = len(atoms_perfect)
    E_perfect = _energy(atoms_perfect, "perfect BCC energy")  # noqa: N806
    E_coh = E_perfect / n_atoms  # noqa: N806

    atoms_defect = atoms_perfect.copy()
    del atoms_defect[0]
    _set_iron_info(atoms_defect)
    atoms_defect.calc = calc

    status = _relax(atoms_defect, "vacancy defect relaxation")
    E_defect = _energy(atoms_defect, "vacancy defect energy")  # noqa: N806
    E_vac = (E_defect - E_perfect) + E_coh  # noqa: N806

    return {
        "E_vac": E_vac,
        "E_coh": E_coh,
        "E_perfect": E_perfect,
        "E_defect": E_defect,
        "perfect_atoms": n_atoms,
        "defect_atoms": len(atoms_defect),
        "status": status,
    }


# =============================================================================
# Surface Calculations
# =============================================================================

# Surface configuration: create_fn, layers/size, area_axes, vacuum
SURFACE_CONFIG = {
    "100": {
        "create_fn": create_surface_100,
        "layers": 10,
        "area_axes": (0, 1),
        "vacuum": SURFACE_VACUUM,
    },
    "110": {
        "create_fn": create_surface_110,
        "layers": 10,
        "area_axes": (0, 1),
        "vacuum": SURFACE_VACUUM,
    },
    "111": {
        "create_fn": create_surface_111,
        "size": (3, 15, 3),
        "area_axes": (0, 2),
        "vacuum": SURFACE_VACUUM,
    },
    "112": {
        "create_fn": create_surface_112,
        "layers": 15,
        "area_axes": (0, 1),
        "vacuum": 5.0,
    },
}


def run_surface_calculations(
    calc: Calculator, lattice_parameter: float
) -> dict[str, Any]:
    """
    Calculate surface energies for (100), (110), (111), (112) surfaces.

    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium lattice parameter from EOS fit.

    Returns
    -------
    dict[str, Any]
        Dictionary with surface energies gamma_100, gamma_110, gamma_111, gamma_112.
    """
    surfaces = {}
    statuses = {}

    for name, cfg in SURFACE_CONFIG.items():
        print(f"Running surface calculation for {name}...")

        create_fn = cfg["create_fn"]
        area_axes = cfg["area_axes"]
        vacuum = cfg["vacuum"]

        # Build kwargs for structure creation
        if "size" in cfg:
            bulk_kwargs = {"size": cfg["size"], "vacuum": 0.0}
            slab_kwargs = {"size": cfg["size"], "vacuum": vacuum}
        else:
            bulk_kwargs = {"layers": cfg["layers"], "vacuum": 0.0}
            slab_kwargs = {"layers": cfg["layers"], "vacuum": vacuum}

        # Bulk reference and slab with vacuum
        try:
            atoms_bulk = create_fn(lattice_parameter, **bulk_kwargs)
            atoms_slab = create_fn(lattice_parameter, **slab_kwargs)
        except Exception as exc:
            warn(f"Surface {name} construction failed: {exc}", stacklevel=2)
            surfaces[name] = float("nan")
            statuses[name] = {"converged": False, "error": str(exc)}
            continue

        _set_iron_info(atoms_bulk)
        atoms_bulk.calc = calc
        e_bulk = _energy(atoms_bulk, f"surface {name} bulk energy")
        cell = atoms_bulk.get_cell()
        area = np.linalg.norm(np.cross(cell[area_axes[0]], cell[area_axes[1]]))

        _set_iron_info(atoms_slab)
        atoms_slab.calc = calc
        statuses[name] = _relax(atoms_slab, f"surface {name} relaxation")
        e_slab = _energy(atoms_slab, f"surface {name} slab energy")

        surfaces[name] = calculate_surface_energy(e_slab, e_bulk, area)

    return {**{f"gamma_{k}": v for k, v in surfaces.items()}, "status": statuses}


# =============================================================================
# Stacking Fault Energy Calculations
# =============================================================================

# SFE configuration: create_fn, displacement axis
SFE_CONFIG = {
    "110": {"create_fn": create_sfe_110_structure, "axis": 1},
    "112": {"create_fn": create_sfe_112_structure, "axis": 2},
}


def run_sfe_calculation(
    calc: Calculator, lattice_parameter: float, sfe_type: str
) -> dict[str, Any]:
    """
    Calculate GSFE curve for specified slip system.

    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium lattice parameter from EOS fit.
    sfe_type
        Type of SFE calculation ('110' or '112').

    Returns
    -------
    dict[str, Any]
        Dictionary with displacements, sfe_J_per_m2, and max_sfe.
    """
    # Calculate Burgers vector magnitude: b = a * sqrt(3) / 2
    burgers_vector = lattice_parameter * np.sqrt(3) / 2
    displacements = np.append(
        np.arange(0.0, burgers_vector, SFE_STEP_SIZE), burgers_vector
    )

    config = SFE_CONFIG[sfe_type]

    try:
        atoms = config["create_fn"](lattice_parameter)
    except Exception as err:
        warn(f"Failed to create SFE structure for {sfe_type}: {err}", stacklevel=2)
        return {
            "displacements": displacements.tolist(),
            "displacement_fractions": (displacements / displacements[-1]).tolist(),
            "sfe_J_per_m2": [float("nan")] * len(displacements),
            "max_sfe": float("nan"),
            "status": {"converged": False, "error": str(err)},
        }

    _set_iron_info(atoms)
    atoms.calc = calc

    cell = atoms.get_cell()
    area = np.linalg.norm(np.cross(cell[1], cell[2]))

    initial_status = _relax(atoms, f"SFE {sfe_type} initial relaxation")
    e0 = _energy(atoms, f"SFE {sfe_type} initial energy")

    positions = atoms.get_positions()
    x_mid = np.linalg.norm(cell[0]) / 2 + 0.1
    upper_indices = np.where(positions[:, 0] <= x_mid)[0]

    sfe_j_per_m2 = [0.0 if _finite(e0) else float("nan")]
    point_statuses: list[dict[str, Any]] = [initial_status]
    last_valid = atoms.copy()
    last_displacement = 0.0

    constraints = [FixedLine(idx, direction=[1, 0, 0]) for idx in range(len(atoms))]
    displacement_axis = config["axis"]

    for displacement in displacements[1:]:
        trial = last_valid.copy()
        trial.calc = calc
        positions = trial.get_positions()
        positions[upper_indices, displacement_axis] += displacement - last_displacement
        trial.set_positions(positions)

        trial.set_constraint(constraints)
        status = _relax(trial, f"SFE {sfe_type} at {displacement:.5f} Angstrom")
        trial.set_constraint()

        energy = _energy(trial, f"SFE {sfe_type} energy")
        valid = status["error"] is None and _finite(energy) and _finite(e0)
        if valid:
            sfe = (energy - e0) / (2 * area) * EV_PER_A2_TO_J_PER_M2
            sfe_j_per_m2.append(sfe)
            last_valid = trial
            last_displacement = float(displacement)
        else:
            sfe_j_per_m2.append(float("nan"))
        point_statuses.append(status)

    finite_values = np.asarray(sfe_j_per_m2)[np.isfinite(sfe_j_per_m2)]
    return {
        "displacements": displacements.tolist(),
        "displacement_fractions": (displacements / displacements[-1]).tolist(),
        "sfe_J_per_m2": sfe_j_per_m2,
        "max_sfe": float(np.max(finite_values)) if finite_values.size else float("nan"),
        "status": point_statuses,
    }


# =============================================================================
# Traction-Separation Calculations
# =============================================================================

# T-S configuration: structure creation function
TS_CONFIG = {
    "100": lambda a: create_surface_100(a, layers=36, vacuum=0.0),
    "110": lambda a: create_surface_110(a, layers=10, vacuum=0.0),
}


def run_ts_calculation(
    calc: Calculator, lattice_parameter: float, direction: str
) -> dict[str, Any]:
    """
    Calculate traction-separation curve for specified cleavage plane.

    The calculation incrementally separates crystal halves without relaxation
    and obtains the traction from the derivative of the energy curve.

    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium lattice parameter from EOS fit.
    direction
        Cleavage plane direction ('100' or '110').

    Returns
    -------
    dict[str, Any]
        Dictionary with separations, energies, traction, and max_traction.
    """
    create_fn = TS_CONFIG[direction]
    num_steps = int(TS_MAX_SEPARATION / TS_STEP_SIZE) + 1

    separations = []
    energies = []
    statuses: list[dict[str, Any]] = []

    for i in range(num_steps):
        dd = TS_STEP_SIZE * i

        # Create fresh structure for each separation
        atoms = create_fn(lattice_parameter)
        _set_iron_info(atoms)
        atoms.calc = calc

        # Get cell dimensions
        cell = atoms.get_cell()
        lz = cell[2, 2]
        area = np.linalg.norm(np.cross(cell[0], cell[1]))

        # Identify upper and lower atoms
        positions = atoms.get_positions()
        z_mid = cell[2, 2] / 2 - 0.1
        upper_indices = np.where(positions[:, 2] >= z_mid)[0]

        # Expand cell in z direction
        new_cell = cell.copy()
        new_cell[2, 2] = lz + dd
        atoms.set_cell(new_cell, scale_atoms=False)

        # Move upper atoms
        positions = atoms.get_positions()
        positions[upper_indices, 2] += dd
        atoms.set_positions(positions)

        # Calculate energy (no relaxation!)
        energy = _energy(atoms, f"TS {direction} energy at {dd:.2f} Angstrom")

        separations.append(dd)
        energies.append(energy)
        statuses.append({"valid": _finite(energy), "separation": dd})

    # Calculate traction from the energy derivative at interval midpoints
    energy_array = np.asarray(energies)
    traction = np.diff(energy_array) / (area * TS_STEP_SIZE) * EV_PER_A3_TO_GPA
    traction_separations = np.concatenate(
        ([0.0], np.asarray(separations[:-1]) + TS_STEP_SIZE / 2)
    )
    traction = np.concatenate(([0.0], traction))

    # Maximum positive traction
    positive = traction[np.isfinite(traction) & (traction > 0)]

    return {
        "energy_separations": separations,
        "energies": energies,
        "separations": traction_separations.tolist(),
        "traction": traction.tolist(),
        "max_traction": float(np.max(positive)) if positive.size else float("nan"),
        "status": statuses,
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _save_curve(write_dir: Path, name: str, data: dict[str, list[Any]]) -> None:
    """
    Save curve data to CSV file.

    Parameters
    ----------
    write_dir
        Directory to save the file.
    name
        Base name for the CSV file (without extension).
    data
        Column name to data mapping for the DataFrame.
    """
    pd.DataFrame(_json_safe(data)).to_csv(write_dir / f"{name}.csv", index=False)


def _write_structures(write_dir: Path, lattice_parameter: float) -> None:
    """
    Write representative benchmark structures for visualization.

    Parameters
    ----------
    write_dir
        Model output directory.
    lattice_parameter
        BCC lattice parameter in Angstrom.
    """
    structures_dir = write_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    eos_atoms = bulk("Fe", "bcc", a=lattice_parameter, cubic=True)
    _set_iron_info(eos_atoms)
    eos_atoms.info["description"] = f"BCC Fe equilibrium (a0={lattice_parameter:.4f})"
    write(structures_dir / "equilibrium_bcc.extxyz", eos_atoms)

    vac_atoms = create_bcc_supercell(lattice_parameter, VACANCY_SUPERCELL_SIZE)
    del vac_atoms[0]
    _set_iron_info(vac_atoms)
    vac_atoms.info["description"] = "BCC Fe vacancy supercell"
    write(structures_dir / "vacancy.extxyz", vac_atoms)

    for name, cfg in SURFACE_CONFIG.items():
        try:
            create_fn = cfg["create_fn"]
            if "size" in cfg:
                slab = create_fn(
                    lattice_parameter, size=cfg["size"], vacuum=cfg["vacuum"]
                )
            else:
                slab = create_fn(
                    lattice_parameter, layers=cfg["layers"], vacuum=cfg["vacuum"]
                )
            _set_iron_info(slab)
            slab.info["description"] = f"BCC Fe ({name}) surface slab"
            write(structures_dir / f"surface_{name}.extxyz", slab)
        except Exception as exc:
            warn(f"Failed to create structure for surface {name}: {exc}", stacklevel=2)
            continue


# =============================================================================
# Main Benchmark Function
# =============================================================================


def run_iron_properties(model_name: str, model: Any) -> None:
    """
    Run the full iron properties benchmark for a single model.

    This benchmark includes:
    - Equation of state (lattice parameter, bulk modulus)
    - Elastic constants (C11, C12, C44)
    - Bain path energy curve
    - Vacancy formation energy
    - Surface energies (100, 110, 111, 112)
    - Stacking fault energy curves (110, 112)
    - Traction-separation curves (100, 110)

    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    model
        Model wrapper providing ``get_calculator``.
    """
    calc = model.get_calculator(precision="high")
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    # EOS calculation
    print(f"[{model_name}] Running EOS calculation...")
    eos_results = _run_section("EOS", lambda: run_eos_calculation(calc))
    results["eos"] = eos_results
    fitted_a0 = eos_results.get("a0")
    a0 = float(fitted_a0) if _finite(fitted_a0) else REFERENCE_LATTICE_PARAMETER
    results["lattice_parameter_source"] = (
        "eos" if _finite(fitted_a0) else "reference_fallback"
    )
    print(
        f"[{model_name}] Lattice parameter: {a0:.4f} Å, "
        f"Bulk modulus: {eos_results.get('B0', float('nan')):.1f} GPa"
    )

    # Save EOS curve data relative to its minimum
    eos_energies = np.asarray(eos_results.get("energies", []), dtype=float)
    relative_eos = (
        (eos_energies - np.nanmin(eos_energies)) * 1000
        if np.isfinite(eos_energies).any()
        else np.full_like(eos_energies, np.nan)
    )
    _save_curve(
        write_dir,
        "eos_curve",
        {
            "volume": eos_results.get("volumes", []),
            "energy_meV": relative_eos.tolist(),
        },
    )

    # Elastic constants calculation
    print(f"[{model_name}] Running elastic constants calculation...")
    elastic_results = _run_section(
        "elastic constants",
        lambda: run_elastic_calculation(calc, a0),
    )
    results["elastic"] = elastic_results
    print(
        f"[{model_name}] C11={elastic_results.get('C11', float('nan')):.1f}, "
        f"C12={elastic_results.get('C12', float('nan')):.1f}, "
        f"C44={elastic_results.get('C44', float('nan')):.1f} GPa"
    )

    # Bain path calculation
    print(f"[{model_name}] Running Bain path calculation...")
    bain_results = _run_section(
        "Bain path", lambda: run_bain_path_calculation(calc, a0)
    )
    results["bain_path"] = bain_results

    # Save Bain path data
    _save_curve(
        write_dir,
        "bain_path",
        {
            "ca_ratio": bain_results.get("ca_ratios", []),
            "energy": bain_results.get("energies", []),
            "energy_meV": bain_results.get("energies_meV", []),
        },
    )

    # Vacancy calculation
    print(f"[{model_name}] Running vacancy calculation...")
    vacancy_results = _run_section("vacancy", lambda: run_vacancy_calculation(calc, a0))
    results["vacancy"] = vacancy_results
    print(f"[{model_name}] E_vac = {vacancy_results.get('E_vac', float('nan')):.3f} eV")

    # Surface calculations
    print(f"[{model_name}] Running surface calculations...")
    surface_results = _run_section(
        "surfaces", lambda: run_surface_calculations(calc, a0)
    )
    results["surfaces"] = surface_results

    # SFE calculations
    sfe_results = {}
    for sfe_type in SFE_CONFIG:
        print(f"[{model_name}] Running SFE {sfe_type} calculation...")
        sfe_result = _run_section(
            f"SFE {sfe_type}",
            lambda sfe_type=sfe_type: run_sfe_calculation(calc, a0, sfe_type),
        )
        sfe_results[sfe_type] = sfe_result
        results[f"sfe_{sfe_type}"] = sfe_result
        _save_curve(
            write_dir,
            f"sfe_{sfe_type}_curve",
            {
                "displacement": sfe_result.get("displacements", []),
                "displacement_fraction": sfe_result.get("displacement_fractions", []),
                "sfe_J_per_m2": sfe_result.get("sfe_J_per_m2", []),
            },
        )

    # T-S calculations
    ts_results = {}
    for direction in TS_CONFIG:
        print(f"[{model_name}] Running T-S ({direction}) calculation...")
        ts_result = _run_section(
            f"T-S {direction}",
            lambda direction=direction: run_ts_calculation(calc, a0, direction),
        )
        ts_results[direction] = ts_result
        results[f"ts_{direction}"] = ts_result
        _save_curve(
            write_dir,
            f"ts_{direction}_curve",
            {
                "separation": ts_result.get("separations", []),
                "traction": ts_result.get("traction", []),
            },
        )
        _save_curve(
            write_dir,
            f"ts_{direction}_energy",
            {
                "separation": ts_result.get("energy_separations", []),
                "energy": ts_result.get("energies", []),
            },
        )
        print(
            f"[{model_name}] Max traction ({direction}): "
            f"{ts_result.get('max_traction', float('nan')):.2f} GPa"
        )

    try:
        _write_structures(write_dir, a0)
    except Exception as exc:
        warn(f"Structure export failed: {exc}", stacklevel=2)

    # Save all results as strict JSON
    safe_results = _json_safe(results)
    (write_dir / "results.json").write_text(
        json.dumps(safe_results, indent=2, allow_nan=False)
    )
    # Save summary metrics
    summary: dict[str, Any] = {
        "a0": eos_results.get("a0"),
        "B0": eos_results.get("B0"),
        "C11": elastic_results.get("C11"),
        "C12": elastic_results.get("C12"),
        "C44": elastic_results.get("C44"),
        "E_bcc_fcc_meV": bain_results.get("delta_E_meV"),
        "E_vac": vacancy_results.get("E_vac"),
        "gamma_100": surface_results.get("gamma_100"),
        "gamma_110": surface_results.get("gamma_110"),
        "gamma_111": surface_results.get("gamma_111"),
        "gamma_112": surface_results.get("gamma_112"),
        "max_sfe_110": sfe_results["110"].get("max_sfe"),
        "max_sfe_112": sfe_results["112"].get("max_sfe"),
        "max_traction_100": ts_results["100"].get("max_traction"),
        "max_traction_110": ts_results["110"].get("max_traction"),
    }
    (write_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False)
    )
    print(f"[{model_name}] Done. Results saved to {write_dir}")


@pytest.mark.slow
@pytest.mark.parametrize("model_name", MODELS)
def test_iron_properties(model_name: str) -> None:
    """
    Run iron properties benchmark for each registered model.

    This test is marked as slow and excluded from default test runs.
    Run with ``pytest --run-slow`` to include.

    Parameters
    ----------
    model_name
        Name of the model to evaluate.
    """
    run_iron_properties(model_name, MODELS[model_name])
