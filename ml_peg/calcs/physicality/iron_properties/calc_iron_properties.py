"""Run calculations for BCC iron properties benchmark.

This benchmark computes fundamental properties of BCC iron including:
- Equation of state (lattice parameter, bulk modulus)
- Elastic constants (C11, C12, C44)
- Bain path energy curve
- Vacancy formation energy
- Surface energies (100, 110, 111, 112)
- Generalized stacking fault energy curves (110, 112)
- Dislocation core energies (5 types)
- Crack K-tests (4 systems)

This benchmark is computationally expensive and marked with @pytest.mark.slow.

Reference
---------
Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
Application to Dislocations and Cracks in Iron.
arXiv:2307.10072. https://arxiv.org/abs/2307.10072
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms, FixedLine
from ase.filters import ExpCellFilter
from ase.optimize import BFGS, FIRE

from ml_peg.calcs.utils.iron_utils import (
    # Constants
    EV_TO_J,
    ANGSTROM_TO_M,
    EV_PER_A2_TO_J_PER_M2,
    EV_PER_A3_TO_GPA,
    # Configurations
    DISLOCATION_CONFIGS,
    DISLOCATION_TYPES,
    CRACK_SYSTEMS_CONFIG,
    # EOS fitting
    fit_eos,
    # Structure creation
    create_bcc_supercell,
    create_bain_cell,
    create_surface_100,
    create_surface_110,
    create_surface_111,
    create_surface_112,
    create_sfe_110_structure,
    create_sfe_112_structure,
    # Elastic utilities
    apply_strain,
    get_voigt_strain,
    calculate_surface_energy,
    # Dislocation utilities
    create_dislocation_cell,
    get_dislocation_info,
    apply_screw_displacement,
    apply_edge_displacement,
    apply_mixed_displacement,
    # LEFM utilities
    compute_lefm_coefficients,
    apply_crack_displacement,
    compute_incremental_displacement,
    create_crack_cell,
)
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)

# Local directory for calculator outputs
OUT_PATH = Path(__file__).parent / "outputs"

# =============================================================================
# Test Parameters
# =============================================================================

# EOS calculation parameters
EOS_NUM_POINTS = 30

# Elastic constants parameters
ELASTIC_STRAIN = 1.0e-5
ELASTIC_SUPERCELL_SIZE = (4, 4, 4)
ELASTIC_FMAX = 1e-10
ELASTIC_MAX_ITER = 100

# Bain path parameters
BAIN_NUM_POINTS = 65

# Vacancy calculation parameters
VACANCY_SUPERCELL_SIZE = (4, 4, 4)
VACANCY_FMAX = 1e-5

# Surface calculation parameters
SURFACE_VACUUM = 10.0  # Angstroms
SURFACE_FMAX = 1e-5

# Stacking fault calculation parameters
SFE_110_STEPS = 63
SFE_112_STEPS = 100
SFE_STEP_SIZE = 0.04  # Angstroms
SFE_FMAX = 1e-5

# Dislocation test parameters
DISLOCATION_FMAX = 1e-5
DISLOCATION_STRESS_TOL = 100.0  # bar
DISLOCATION_MAX_ITERATIONS = 10

# Crack test parameters
CRACK_K_STEPS = 100


# =============================================================================
# EOS Calculation
# =============================================================================

def run_eos_calculation(calc: Any) -> dict[str, Any]:
    """
    Run the energy-volume curve calculation.
    
    Parameters
    ----------
    calc
        ASE calculator object.
    
    Returns
    -------
    dict
        Dictionary with EOS results including a0, B0, V0, E0, volumes, energies.
    """
    # Generate lattice parameters: 2.834 - 0.05 + (0.1/30)*i for i in 1..30
    lattice_params = np.array([2.834 - 0.05 + 0.1 / 30 * i for i in range(1, EOS_NUM_POINTS + 1)])
    
    volumes = []
    energies = []
    
    for lat in lattice_params:
        atoms = bulk('Fe', 'bcc', a=lat, cubic=True)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        
        n_atoms = len(atoms)
        volumes.append(volume / n_atoms)
        energies.append(energy / n_atoms)
    
    volumes = np.array(volumes)
    energies = np.array(energies)
    
    # Fit Birch-Murnaghan EOS
    eos_results = fit_eos(volumes, energies)
    
    return {
        'volumes': volumes.tolist(),
        'energies': energies.tolist(),
        'lattice_params': lattice_params.tolist(),
        'a0': eos_results['a0'],
        'E0': eos_results['E0'],
        'B0': eos_results['B0'],
        'Bp': eos_results['Bp'],
        'V0': eos_results['V0'],
    }


# =============================================================================
# Elastic Constants Calculation
# =============================================================================

def run_elastic_calculation(calc: Any, lattice_parameter: float) -> dict[str, Any]:
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
    dict
        Dictionary with elastic constants C11, C12, C44, bulk_modulus.
    """
    # Create supercell
    atoms_ref = create_bcc_supercell(lattice_parameter, ELASTIC_SUPERCELL_SIZE)
    atoms_ref.calc = calc
    
    # Box relaxation
    ecf = ExpCellFilter(atoms_ref)
    opt = BFGS(ecf, logfile=None)
    opt.run(fmax=ELASTIC_FMAX, steps=ELASTIC_MAX_ITER)
    
    # Elastic constant matrix
    C = np.zeros((6, 6))
    
    for i in range(6):
        direction = i + 1
        
        # Positive strain
        strain_pos = get_voigt_strain(direction, ELASTIC_STRAIN)
        atoms_pos = apply_strain(atoms_ref.copy(), strain_pos)
        atoms_pos.calc = calc
        
        opt_pos = BFGS(atoms_pos, logfile=None)
        opt_pos.run(fmax=ELASTIC_FMAX, steps=ELASTIC_MAX_ITER)
        stress_pos = -atoms_pos.get_stress(voigt=True)
        
        # Negative strain
        strain_neg = get_voigt_strain(direction, -ELASTIC_STRAIN)
        atoms_neg = apply_strain(atoms_ref.copy(), strain_neg)
        atoms_neg.calc = calc
        
        opt_neg = BFGS(atoms_neg, logfile=None)
        opt_neg.run(fmax=ELASTIC_FMAX, steps=ELASTIC_MAX_ITER)
        stress_neg = -atoms_neg.get_stress(voigt=True)
        
        # Compute elastic constants
        delta_stress = stress_pos - stress_neg
        delta_strain = 2 * ELASTIC_STRAIN
        
        for j in range(6):
            C[j, i] = -delta_stress[j] / delta_strain * EV_PER_A3_TO_GPA
    
    # Symmetrize
    C_sym = 0.5 * (C + C.T)
    
    # Extract cubic averages
    C11 = (C_sym[0, 0] + C_sym[1, 1] + C_sym[2, 2]) / 3
    C12 = (C_sym[0, 1] + C_sym[0, 2] + C_sym[1, 2]) / 3
    C44 = (C_sym[3, 3] + C_sym[4, 4] + C_sym[5, 5]) / 3
    
    bulk_modulus = (C11 + 2 * C12) / 3
    
    return {
        'C11': C11,
        'C12': C12,
        'C44': C44,
        'bulk_modulus': bulk_modulus,
        'C_matrix': C_sym.tolist(),
    }


# =============================================================================
# Bain Path Calculation
# =============================================================================

def run_bain_path_calculation(calc: Any, lattice_parameter: float) -> dict[str, Any]:
    """
    Calculate the Bain path energy curve.
    
    Parameters
    ----------
    calc
        ASE calculator object.
    lattice_parameter
        Equilibrium BCC lattice parameter.
    
    Returns
    -------
    dict
        Dictionary with ca_ratios, energies, E_bcc, E_fcc, delta_E.
    """
    # Generate c/a ratios: 0.7 + 0.02*i for i in 1..65
    ca_ratios_target = np.array([0.7 + 0.02 * i for i in range(1, BAIN_NUM_POINTS + 1)])
    
    ca_ratios = []
    energies = []
    
    for ratio in ca_ratios_target:
        atoms = create_bain_cell(lattice_parameter, ratio)
        atoms.calc = calc
        
        # Box relaxation
        ecf = ExpCellFilter(atoms, scalar_pressure=0.0)
        opt = BFGS(ecf, logfile=None)
        
        try:
            opt.run(fmax=1e-5, steps=10000)
        except Exception:
            pass
        
        # Additional atomic relaxation
        opt2 = BFGS(atoms, logfile=None)
        try:
            opt2.run(fmax=1e-5, steps=10000)
        except Exception:
            pass
        
        energy = atoms.get_potential_energy()
        cell = atoms.get_cell()
        ca_actual = cell[2, 2] / cell[1, 1]
        
        n_atoms = len(atoms)
        ca_ratios.append(ca_actual)
        energies.append(energy / n_atoms)
    
    ca_ratios = np.array(ca_ratios)
    energies = np.array(energies)
    
    # Normalize energies (subtract minimum, convert to meV)
    E_min = np.min(energies)
    energies_norm = (energies - E_min) * 1000  # meV/atom
    
    # Find BCC point (c/a ≈ 1.0) and FCC point (c/a ≈ 1.414)
    idx_bcc = np.argmin(np.abs(ca_ratios - 1.0))
    idx_fcc = np.argmin(np.abs(ca_ratios - np.sqrt(2)))
    
    E_bcc = energies_norm[idx_bcc]
    E_fcc = energies_norm[idx_fcc]
    
    return {
        'ca_ratios': ca_ratios.tolist(),
        'energies': energies.tolist(),
        'energies_meV': energies_norm.tolist(),
        'E_bcc_meV': E_bcc,
        'E_fcc_meV': E_fcc,
        'delta_E_meV': E_fcc - E_bcc,
    }


# =============================================================================
# Vacancy Calculation
# =============================================================================

def run_vacancy_calculation(calc: Any, lattice_parameter: float) -> dict[str, Any]:
    """Calculate the vacancy formation energy."""
    atoms_perfect = create_bcc_supercell(lattice_parameter, VACANCY_SUPERCELL_SIZE)
    atoms_perfect.calc = calc
    
    n_atoms = len(atoms_perfect)
    E_perfect = atoms_perfect.get_potential_energy()
    E_coh = E_perfect / n_atoms
    
    atoms_defect = atoms_perfect.copy()
    del atoms_defect[0]
    atoms_defect.calc = calc
    
    opt = BFGS(atoms_defect, logfile=None)
    opt.run(fmax=VACANCY_FMAX, steps=10000)
    E_defect = atoms_defect.get_potential_energy()
    E_vac = (E_defect - E_perfect) + E_coh
    
    return {'E_vac': E_vac, 'E_coh': E_coh, 'E_perfect': E_perfect, 'E_defect': E_defect}


# =============================================================================
# Surface Calculations
# =============================================================================

def run_surface_calculations(calc: Any, lattice_parameter: float) -> dict[str, Any]:
    """Calculate surface energies for (100), (110), (111), (112) surfaces."""
    surfaces = {}
    
    # (100) surface
    atoms_bulk = create_surface_100(lattice_parameter, layers=10, vacuum=0.0)
    atoms_bulk.calc = calc
    E_bulk = atoms_bulk.get_potential_energy()
    cell = atoms_bulk.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    
    atoms_slab = create_surface_100(lattice_parameter, layers=10, vacuum=SURFACE_VACUUM)
    atoms_slab.calc = calc
    opt = BFGS(atoms_slab, logfile=None)
    opt.run(fmax=SURFACE_FMAX, steps=10000)
    E_slab = atoms_slab.get_potential_energy()
    surfaces['100'] = calculate_surface_energy(E_slab, E_bulk, area)
    
    # (110) surface
    atoms_bulk = create_surface_110(lattice_parameter, layers=10, vacuum=0.0)
    atoms_bulk.calc = calc
    E_bulk = atoms_bulk.get_potential_energy()
    cell = atoms_bulk.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    
    atoms_slab = create_surface_110(lattice_parameter, layers=10, vacuum=SURFACE_VACUUM)
    atoms_slab.calc = calc
    opt = BFGS(atoms_slab, logfile=None)
    opt.run(fmax=SURFACE_FMAX, steps=10000)
    E_slab = atoms_slab.get_potential_energy()
    surfaces['110'] = calculate_surface_energy(E_slab, E_bulk, area)
    
    # (111) surface
    atoms_bulk = create_surface_111(lattice_parameter, size=(3, 15, 3), vacuum=0.0)
    atoms_bulk.calc = calc
    E_bulk = atoms_bulk.get_potential_energy()
    cell = atoms_bulk.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[2]))
    
    atoms_slab = create_surface_111(lattice_parameter, size=(3, 15, 3), vacuum=SURFACE_VACUUM)
    atoms_slab.calc = calc
    opt = BFGS(atoms_slab, logfile=None)
    opt.run(fmax=SURFACE_FMAX, steps=10000)
    E_slab = atoms_slab.get_potential_energy()
    surfaces['111'] = calculate_surface_energy(E_slab, E_bulk, area)
    
    # (112) surface
    atoms_bulk = create_surface_112(lattice_parameter, layers=15, vacuum=0.0)
    atoms_bulk.calc = calc
    E_bulk = atoms_bulk.get_potential_energy()
    cell = atoms_bulk.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    
    atoms_slab = create_surface_112(lattice_parameter, layers=15, vacuum=5.0)
    atoms_slab.calc = calc
    opt = BFGS(atoms_slab, logfile=None)
    opt.run(fmax=SURFACE_FMAX, steps=10000)
    E_slab = atoms_slab.get_potential_energy()
    surfaces['112'] = calculate_surface_energy(E_slab, E_bulk, area)
    
    return {
        'gamma_100': surfaces['100'],
        'gamma_110': surfaces['110'],
        'gamma_111': surfaces['111'],
        'gamma_112': surfaces['112'],
    }


# =============================================================================
# Stacking Fault Energy Calculations
# =============================================================================

def run_sfe_110_calculation(calc: Any, lattice_parameter: float) -> dict[str, Any]:
    """Calculate GSFE curve for {110}<111> slip system."""
    atoms = create_sfe_110_structure(lattice_parameter)
    atoms.calc = calc
    
    cell = atoms.get_cell()
    ly = cell[1, 1]
    lz = cell[2, 2]
    area = ly * lz
    
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=SFE_FMAX, steps=10000)
    E0 = atoms.get_potential_energy()
    
    positions = atoms.get_positions()
    x_mid = (positions[:, 0].min() + positions[:, 0].max()) / 2 + 0.1
    upper_mask = positions[:, 0] < x_mid
    upper_indices = np.where(upper_mask)[0]
    
    displacements = [0.0]
    sfe_J_per_m2 = [0.0]
    

    constraints = [FixedLine(idx, direction=[1, 0, 0]) for idx in range(len(atoms))]
    for step in range(1, SFE_110_STEPS + 1):
        positions = atoms.get_positions()
        positions[upper_indices, 1] += SFE_STEP_SIZE
        atoms.set_positions(positions)

        atoms.set_constraint(constraints)
        
        opt = BFGS(atoms, logfile=None)
        try:
            opt.run(fmax=SFE_FMAX, steps=10000)
        except Exception:
            pass
        
        atoms.set_constraint()
        E = atoms.get_potential_energy()
        sfe = (E - E0) / (2 * area) * EV_PER_A2_TO_J_PER_M2
        
        displacements.append(step * SFE_STEP_SIZE)
        sfe_J_per_m2.append(sfe)
    
    return {'displacements': displacements, 'sfe_J_per_m2': sfe_J_per_m2, 'max_sfe': max(sfe_J_per_m2)}


def run_sfe_112_calculation(calc: Any, lattice_parameter: float) -> dict[str, Any]:
    """Calculate GSFE curve for {112}<111> slip system."""
    atoms = create_sfe_112_structure(lattice_parameter)
    atoms.calc = calc
    
    cell = atoms.get_cell()
    ly = cell[1, 1]
    lz = cell[2, 2]
    area = ly * lz
    
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=SFE_FMAX, steps=10000)
    E0 = atoms.get_potential_energy()
    
    positions = atoms.get_positions()
    x_mid = (positions[:, 0].min() + positions[:, 0].max()) / 2 + 0.1
    upper_mask = positions[:, 0] < x_mid
    upper_indices = np.where(upper_mask)[0]
    
    displacements = [0.0]
    sfe_J_per_m2 = [0.0]
    
    constraints = [FixedLine(idx, direction=[1, 0, 0]) for idx in range(len(atoms))]
    for step in range(1, SFE_112_STEPS + 1):
        positions = atoms.get_positions()
        positions[upper_indices, 2] += SFE_STEP_SIZE
        atoms.set_positions(positions)
        
        atoms.set_constraint(constraints)
        
        opt = BFGS(atoms, logfile=None)
        try:
            opt.run(fmax=SFE_FMAX, steps=10000)
        except Exception:
            pass
        
        atoms.set_constraint()
        E = atoms.get_potential_energy()
        sfe = (E - E0) / (2 * area) * EV_PER_A2_TO_J_PER_M2
        
        displacements.append(step * SFE_STEP_SIZE)
        sfe_J_per_m2.append(sfe)
    
    return {'displacements': displacements, 'sfe_J_per_m2': sfe_J_per_m2, 'max_sfe': max(sfe_J_per_m2)}


# =============================================================================
# Dislocation Tests
# =============================================================================

def run_dislocation_test(calc: Any, a0: float, E_coh: float, dislocation_type: str) -> dict[str, Any]:
    """Run dislocation test for a specific type."""
    config = get_dislocation_info(dislocation_type)
    atoms = create_dislocation_cell(a0, dislocation_type)
    atoms.calc = calc
    
    atoms_perfect = atoms.copy()
    atoms_perfect.calc = calc
    
    # Box relaxation for perfect crystal with stress convergence loop
    mask = [True, True, False, False, False, True]
    for iteration in range(DISLOCATION_MAX_ITERATIONS):
        ecf = ExpCellFilter(atoms_perfect, mask=mask, scalar_pressure=0.0)
        opt = BFGS(ecf, logfile=None)
        opt.run(fmax=DISLOCATION_FMAX, steps=2000)
        
        # Check stress convergence (convert eV/Å³ to bar)
        stress = atoms_perfect.get_stress() * 160.2176621 * 10000  # bar
        if abs(stress[0]) < DISLOCATION_STRESS_TOL and abs(stress[1]) < DISLOCATION_STRESS_TOL:
            break
    
    # Final relaxation with FIRE
    opt = FIRE(atoms_perfect, logfile=None)
    opt.run(fmax=DISLOCATION_FMAX, steps=5000)
    
    E_perfect = atoms_perfect.get_potential_energy()
    n_atoms_perfect = len(atoms_perfect)
    E_coh_local = E_perfect / n_atoms_perfect
    
    atoms = atoms_perfect.copy()
    atoms.calc = calc
    
    burgers_vec = config['burgers']
    b_mag = a0 * np.linalg.norm(burgers_vec)
    disl_type = config['type']
    
    if disl_type == 'screw':
        apply_screw_displacement(atoms, b_mag)
    elif disl_type == 'edge':
        atoms = apply_edge_displacement(atoms, b_mag, a0, delete_half_plane=True)
        atoms.calc = calc
    elif disl_type == 'mixed':
        atoms = apply_mixed_displacement(atoms, b_mag, a0, screw_fraction=0.7)
        atoms.calc = calc
    
    # Multi-stage relaxation of dislocation structure with stress convergence loop
    mask = [True, True, False, False, False, True]
    for iteration in range(DISLOCATION_MAX_ITERATIONS):
        ecf = ExpCellFilter(atoms, mask=mask, scalar_pressure=0.0)
        opt = BFGS(ecf, logfile=None)
        opt.run(fmax=DISLOCATION_FMAX, steps=2000)
        
        stress = atoms.get_stress() * 160.2176621 * 10000  # bar
        if abs(stress[0]) < DISLOCATION_STRESS_TOL and abs(stress[1]) < DISLOCATION_STRESS_TOL:
            break
    
    # Final relaxation with FIRE
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=DISLOCATION_FMAX, steps=5000)
    
    E_disl = atoms.get_potential_energy()
    n_atoms_disl = len(atoms)
    core_energy = E_disl - n_atoms_disl * E_coh_local
    
    return {
        'dislocation_type': dislocation_type,
        'name': config['name'],
        'core_energy': core_energy,
        'n_atoms': n_atoms_disl,
    }


# =============================================================================
# Crack Tests
# =============================================================================

def run_crack_test(
    calc: Any,
    a0: float,
    elastic_constants: dict[str, float],
    surface_energies: dict[str, float],
    crack_system: int,
    K_steps: int = CRACK_K_STEPS,
) -> dict[str, Any]:
    """Run crack K-test for a specific system."""
    config = CRACK_SYSTEMS_CONFIG[crack_system]
    surface = config['surface']
    surf_energy = surface_energies.get(surface, 2.0)
    
    coeffs = compute_lefm_coefficients(
        elastic_constants['C11'],
        elastic_constants['C12'],
        elastic_constants['C44'],
        surf_energy,
        crack_system,
    )
    K_Griffith = coeffs['K_I']
    
    atoms, crack_tip, radius = create_crack_cell(a0, crack_system)
    atoms.calc = calc
    
    positions = atoms.get_positions()
    xtip, ytip = crack_tip
    r = np.sqrt((positions[:, 0] - xtip)**2 + (positions[:, 1] - ytip)**2)
    boundary_mask = r > (radius - 10.0)
    boundary_indices = np.where(boundary_mask)[0]
    
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=1e-3, steps=10000)
    
    ref_positions = atoms.get_positions().copy()
    
    K_start = max(0.5, K_Griffith * 100 - 10)
    K_stop = K_start + K_steps
    
    new_positions = apply_crack_displacement(atoms.get_positions(), K_start, coeffs, crack_tip, ref_positions)
    atoms.set_positions(new_positions)
    
    dx, dy = compute_incremental_displacement(ref_positions, 1.0, coeffs, crack_tip)
    
    K_values = []
    energies = []
    
    atoms.set_constraint(FixAtoms(indices=boundary_indices))
    dK = (K_stop - K_start) / K_steps if K_steps > 0 else 1.0
    
    for i in range(K_steps + 1):
        K = K_start + i * dK
        
        if i > 0:
            positions = atoms.get_positions()
            positions[:, 0] += dx * dK
            positions[:, 1] += dy * dK
            atoms.set_positions(positions)
        
        opt = BFGS(atoms, logfile=None)
        try:
            opt.run(fmax=1e-3, steps=5000)
        except Exception:
            pass
        
        E = atoms.get_potential_energy()
        K_values.append(K)
        energies.append(E)
    
    return {
        'crack_system': crack_system,
        'name': config['name'],
        'K_Griffith': K_Griffith,
        'K_values': K_values,
        'energies': energies,
    }


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
    - Dislocation core energies (5 types)
    - Crack K-tests (4 systems)
    
    Parameters
    ----------
    model_name
        Name of the model being evaluated.
    model
        Model wrapper providing ``get_calculator``.
    """
    calc = model.get_calculator()
    write_dir = OUT_PATH / model_name
    write_dir.mkdir(parents=True, exist_ok=True)
    
    results: dict[str, Any] = {}
    
    # EOS calculation
    print(f"[{model_name}] Running EOS calculation...")
    eos_results = run_eos_calculation(calc)
    results['eos'] = eos_results
    a0 = eos_results['a0']
    E_coh = eos_results['E0']
    print(f"[{model_name}] Lattice parameter: {a0:.4f} Å, Bulk modulus: {eos_results['B0']:.1f} GPa")
    
    # Save EOS curve data
    eos_df = pd.DataFrame({
        'volume': eos_results['volumes'],
        'energy': eos_results['energies'],
    })
    eos_df.to_csv(write_dir / "eos_curve.csv", index=False)
    
    # Elastic constants calculation
    print(f"[{model_name}] Running elastic constants calculation...")
    elastic_results = run_elastic_calculation(calc, a0)
    results['elastic'] = elastic_results
    print(f"[{model_name}] C11={elastic_results['C11']:.1f}, C12={elastic_results['C12']:.1f}, C44={elastic_results['C44']:.1f} GPa")
    
    # Bain path calculation
    print(f"[{model_name}] Running Bain path calculation...")
    bain_results = run_bain_path_calculation(calc, a0)
    results['bain_path'] = bain_results
    
    # Save Bain path data
    bain_df = pd.DataFrame({
        'ca_ratio': bain_results['ca_ratios'],
        'energy': bain_results['energies'],
        'energy_meV': bain_results['energies_meV'],
    })
    bain_df.to_csv(write_dir / "bain_path.csv", index=False)
    
    # Vacancy calculation
    print(f"[{model_name}] Running vacancy calculation...")
    vacancy_results = run_vacancy_calculation(calc, a0)
    results['vacancy'] = vacancy_results
    print(f"[{model_name}] E_vac = {vacancy_results['E_vac']:.3f} eV")
    
    # Surface calculations
    print(f"[{model_name}] Running surface calculations...")
    surface_results = run_surface_calculations(calc, a0)
    results['surfaces'] = surface_results
    
    # SFE 110 calculation
    print(f"[{model_name}] Running SFE 110 calculation...")
    sfe_110_results = run_sfe_110_calculation(calc, a0)
    results['sfe_110'] = {'max_sfe': sfe_110_results['max_sfe']}
    sfe_110_df = pd.DataFrame({
        'displacement': sfe_110_results['displacements'],
        'sfe_J_per_m2': sfe_110_results['sfe_J_per_m2'],
    })
    sfe_110_df.to_csv(write_dir / "sfe_110_curve.csv", index=False)
    
    # SFE 112 calculation
    print(f"[{model_name}] Running SFE 112 calculation...")
    sfe_112_results = run_sfe_112_calculation(calc, a0)
    results['sfe_112'] = {'max_sfe': sfe_112_results['max_sfe']}
    sfe_112_df = pd.DataFrame({
        'displacement': sfe_112_results['displacements'],
        'sfe_J_per_m2': sfe_112_results['sfe_J_per_m2'],
    })
    sfe_112_df.to_csv(write_dir / "sfe_112_curve.csv", index=False)
    
    # Dislocation tests
    print(f"[{model_name}] Running dislocation tests...")
    dislocation_results = {}
    for disl_type in DISLOCATION_TYPES:
        print(f"[{model_name}]   {disl_type}...")
        try:
            disl_result = run_dislocation_test(calc, a0, E_coh, disl_type)
            dislocation_results[disl_type] = disl_result
        except Exception as e:
            print(f"[{model_name}]   Error: {e}")
            dislocation_results[disl_type] = {'error': str(e)}
    results['dislocations'] = dislocation_results
    
    # Crack K-tests
    print(f"[{model_name}] Running crack K-tests...")
    crack_results = {}
    for crack_sys in [1, 2, 3, 4]:
        print(f"[{model_name}]   System {crack_sys}...")
        try:
            crack_result = run_crack_test(
                calc, a0, elastic_results,
                {'100': surface_results['gamma_100'], '110': surface_results['gamma_110']},
                crack_sys, K_steps=CRACK_K_STEPS,
            )
            crack_results[crack_sys] = {
                'name': crack_result['name'],
                'K_Griffith': crack_result['K_Griffith'],
            }
            ke_df = pd.DataFrame({
                'K': crack_result['K_values'],
                'energy': crack_result['energies'],
            })
            ke_df.to_csv(write_dir / f"crack_{crack_sys}_KE.csv", index=False)
        except Exception as e:
            print(f"[{model_name}]   Error: {e}")
            crack_results[crack_sys] = {'error': str(e)}
    results['cracks'] = crack_results
    
    # Save all results as JSON
    (write_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    
    # Save summary metrics
    summary: dict[str, Any] = {
        'a0': a0,
        'B0': eos_results['B0'],
        'C11': elastic_results['C11'],
        'C12': elastic_results['C12'],
        'C44': elastic_results['C44'],
        'E_bcc_fcc_meV': bain_results['delta_E_meV'],
        'E_vac': vacancy_results['E_vac'],
        'gamma_100': surface_results['gamma_100'],
        'gamma_110': surface_results['gamma_110'],
        'gamma_111': surface_results['gamma_111'],
        'gamma_112': surface_results['gamma_112'],
        'max_sfe_110': sfe_110_results['max_sfe'],
        'max_sfe_112': sfe_112_results['max_sfe'],
    }
    
    for disl_type, disl_data in dislocation_results.items():
        if 'core_energy' in disl_data:
            summary[f'core_energy_{disl_type}'] = disl_data['core_energy']
    
    for crack_sys, crack_data in crack_results.items():
        if 'K_Griffith' in crack_data:
            summary[f'K_Griffith_{crack_sys}'] = crack_data['K_Griffith']
    
    (write_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    
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
