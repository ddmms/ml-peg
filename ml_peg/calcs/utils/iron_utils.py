"""Utility functions for BCC iron property calculations.

This module provides structure creation, EOS fitting, dislocation utilities,
and LEFM functions for iron benchmarks.

Reference
---------
Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
Application to Dislocations and Cracks in Iron.
arXiv:2307.10072. https://arxiv.org/abs/2307.10072
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import NeighborList
from scipy.optimize import leastsq


# =============================================================================
# Unit Conversion Constants
# =============================================================================

EV_TO_J = 1.60218e-19
ANGSTROM_TO_M = 1.0e-10
EV_PER_A2_TO_J_PER_M2 = 16.0217733
EV_PER_A3_TO_GPA = 160.21765


# =============================================================================
# Dislocation Configuration
# =============================================================================

DISLOCATION_CONFIGS = {
    'edge_100_010': {
        'name': 'Edge a0[100](010)',
        'orient_x': np.array([0, 0, 1]),
        'orient_y': np.array([1, 0, 0]),
        'orient_z': np.array([0, 1, 0]),
        'size': (1, 50, 20),
        'burgers': np.array([1, 0, 0]),
        'type': 'edge',
        'slip_direction': 1,
    },
    'edge_100_011': {
        'name': 'Edge a0[100](011)',
        'orient_x': np.array([0, -1, 1]),
        'orient_y': np.array([1, 0, 0]),
        'orient_z': np.array([0, 1, 1]),
        'size': (1, 80, 22),
        'burgers': np.array([1, 0, 0]),
        'type': 'edge',
        'slip_direction': 1,
    },
    'edge_111_110': {
        'name': 'Edge a0/2[111](110)',
        'orient_x': np.array([1, 2, -1]),
        'orient_y': np.array([-1, 1, 1]),
        'orient_z': np.array([1, 0, 1]),
        'size': (1, 40, 20),
        'burgers': np.array([0.5, 0.5, 0.5]),
        'type': 'edge',
        'slip_direction': 1,
    },
    'mixed_111': {
        'name': 'Mixed 70.5 deg a0/2[111](110)',
        'orient_x': np.array([1, 2, -1]),
        'orient_y': np.array([-1, 1, 1]),
        'orient_z': np.array([1, 0, 1]),
        'size': (40, 2, 19),
        'burgers': np.array([0.5, 0.5, 0.5]),
        'type': 'mixed',
        'slip_direction': 1,
    },
    'screw_111': {
        'name': 'Screw a0/2[111](112)',
        'orient_x': np.array([1, 2, -1]),
        'orient_y': np.array([-1, 1, 1]),
        'orient_z': np.array([1, 0, 1]),
        'size': (60, 2, 19),
        'burgers': np.array([0.5, 0.5, 0.5]),
        'type': 'screw',
        'slip_direction': 1,
    },
}

DISLOCATION_TYPES = list(DISLOCATION_CONFIGS.keys())


# =============================================================================
# Crack System Configuration
# =============================================================================

CRACK_SYSTEMS_CONFIG = {
    1: {
        'name': '(100)[010]',
        'orient_x': np.array([0, 0, 1]),
        'orient_y': np.array([1, 0, 0]),
        'orient_z': np.array([0, 1, 0]),
        'surface': '100',
        'box_size': (50, 50),
        'tip_factors': (1.0, 1.0),
    },
    2: {
        'name': '(100)[001]',
        'orient_x': np.array([0, -1, 1]),
        'orient_y': np.array([1, 0, 0]),
        'orient_z': np.array([0, 1, 1]),
        'surface': '100',
        'box_size': (38, 54),
        'tip_factors': (np.sqrt(2), 1.0),
    },
    3: {
        'name': '(110)[001]',
        'orient_x': np.array([1, -1, 0]),
        'orient_y': np.array([1, 1, 0]),
        'orient_z': np.array([0, 0, 1]),
        'surface': '110',
        'box_size': (38, 38),
        'tip_factors': (np.sqrt(2), np.sqrt(2)),
    },
    4: {
        'name': '(110)[1-10]',
        'orient_x': np.array([0, 0, -1]),
        'orient_y': np.array([1, 1, 0]),
        'orient_z': np.array([1, -1, 0]),
        'surface': '110',
        'box_size': (55, 38),
        'tip_factors': (1.0, np.sqrt(2)),
    },
}


# =============================================================================
# EOS Fitting Functions
# =============================================================================

def eos_birch_murnaghan(
    params: tuple[float, float, float, float],
    vol: np.ndarray
) -> np.ndarray:
    """
    Birch-Murnaghan equation of state (3rd order).
    
    Parameters
    ----------
    params
        (E0, B0, Bp, V0).
    vol
        Volume array.
    
    Returns
    -------
    np.ndarray
        Energy array.
    """
    E0, B0, Bp, V0 = params
    eta = (vol / V0) ** (1.0 / 3.0)
    E = E0 + 9.0 * B0 * V0 / 16.0 * (eta**2 - 1.0)**2 * (6.0 + Bp * (eta**2 - 1.0) - 4.0 * eta**2)
    return E


def get_eos_initial_guess(
    vol: np.ndarray,
    ene: np.ndarray
) -> tuple[float, float, float, float]:
    """
    Get initial guess for EOS parameters using quadratic fit.
    
    Parameters
    ----------
    vol
        Volume array.
    ene
        Energy array.
    
    Returns
    -------
    tuple
        (E0, B0, Bp, V0) initial guess.
    """
    a, b, c = np.polyfit(vol, ene, 2)
    V0 = -b / (2 * a)
    E0 = a * V0**2 + b * V0 + c
    B0 = 2 * a * V0
    Bp = 4.0
    return E0, B0, Bp, V0


def fit_eos(
    vol: np.ndarray,
    ene: np.ndarray,
) -> dict[str, Any]:
    """
    Fit Birch-Murnaghan equation of state to energy-volume data.
    
    Parameters
    ----------
    vol
        Volume per atom array (Angstrom^3).
    ene
        Energy per atom array (eV).
    
    Returns
    -------
    dict
        Fitted parameters:
        - E0: Equilibrium energy (eV)
        - B0: Bulk modulus (GPa)
        - Bp: Pressure derivative (dimensionless)
        - V0: Equilibrium volume per atom (Angstrom^3)
        - a0: Equilibrium lattice parameter (Angstrom) - for BCC
    """
    x0 = get_eos_initial_guess(vol, ene)
    
    def residual(params, y, x):
        return y - eos_birch_murnaghan(params, x)
    
    params, _ = leastsq(residual, x0, args=(ene, vol))
    E0, B0, Bp, V0 = params
    
    # Convert bulk modulus to GPa (from eV/Angstrom^3)
    B0_GPa = B0 * EV_PER_A3_TO_GPA
    
    # Calculate lattice parameter for BCC (2 atoms per unit cell)
    a0 = (V0 * 2) ** (1.0 / 3.0)
    
    return {'E0': E0, 'B0': B0_GPa, 'Bp': Bp, 'V0': V0, 'a0': a0}


# =============================================================================
# Structure Creation Functions
# =============================================================================

def create_bcc_supercell(
    lattice_parameter: float,
    size: tuple = (4, 4, 4),
    symbol: str = 'Fe'
) -> Atoms:
    """
    Create a BCC supercell.
    
    Parameters
    ----------
    lattice_parameter
        Lattice parameter in Angstroms.
    size
        Supercell size as (nx, ny, nz).
    symbol
        Chemical symbol (default: 'Fe').
    
    Returns
    -------
    Atoms
        ASE Atoms object.
    """
    unit_cell = bulk(symbol, 'bcc', a=lattice_parameter, cubic=True)
    return unit_cell * size


def create_bain_cell(lattice_parameter: float, ca_ratio: float) -> Atoms:
    """
    Create a tetragonally distorted BCC cell for Bain path calculation.
    
    Parameters
    ----------
    lattice_parameter
        BCC lattice parameter.
    ca_ratio
        Target c/a ratio.
    
    Returns
    -------
    Atoms
        Tetragonally distorted cell.
    """
    beta = (1.0 / ca_ratio) ** (1.0 / 3.0)
    al = lattice_parameter * beta
    alz = al * ca_ratio
    
    cell = np.array([[al, 0, 0], [0, al, 0], [0, 0, alz]])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    
    return Atoms(symbols=['Fe', 'Fe'], positions=positions, cell=cell, pbc=True)


def create_surface_100(
    lattice_parameter: float,
    layers: int = 10,
    vacuum: float = 0.0,
    symbol: str = 'Fe'
) -> Atoms:
    """Create a (100) surface slab for BCC iron."""
    a = lattice_parameter
    cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a * layers]])
    
    positions = []
    for k in range(layers):
        positions.append([0, 0, k * a])
        positions.append([0.5 * a, 0.5 * a, (k + 0.5) * a])
    
    atoms = Atoms(symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True)
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=2)
    return atoms


def create_surface_110(
    lattice_parameter: float,
    layers: int = 10,
    vacuum: float = 0.0,
    symbol: str = 'Fe'
) -> Atoms:
    """Create a (110) surface slab for BCC iron."""
    a = lattice_parameter
    lx = a
    ly = a * np.sqrt(2)
    lz = a * np.sqrt(2) * layers
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    positions = []
    d110 = a * np.sqrt(2) / 2
    
    for k in range(layers * 2):
        z = k * d110
        if k % 2 == 0:
            positions.append([0, 0, z])
        else:
            positions.append([0.5 * a, 0.5 * ly, z])
    
    atoms = Atoms(symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True)
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=2)
    return atoms


def create_surface_111(
    lattice_parameter: float,
    size: tuple = (3, 15, 3),
    vacuum: float = 0.0,
    symbol: str = 'Fe'
) -> Atoms:
    """Create a (111) surface slab for BCC iron."""
    a = lattice_parameter
    lx = a * np.sqrt(2) * size[0]
    ly = a * np.sqrt(3) * size[1]
    lz = a * np.sqrt(6) * size[2]
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    positions = []
    max_range = int(max(size) * 3 + 5)
    
    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    R = np.array([ex, ey, ez])
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = R @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (0 - eps <= frac_x < 1 - eps and
                        0 - eps <= frac_y < 1 - eps and
                        0 - eps <= frac_z < 1 - eps):
                        positions.append(pos_oriented)
    
    if len(positions) == 0:
        raise ValueError("No atoms found for (111) surface")
    
    positions = np.array(positions)
    _, unique_idx = np.unique(np.round(positions, decimals=6), axis=0, return_index=True)
    positions = positions[unique_idx]
    
    atoms = Atoms(symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=1)
    return atoms


def create_surface_112(
    lattice_parameter: float,
    layers: int = 15,
    vacuum: float = 0.0,
    symbol: str = 'Fe'
) -> Atoms:
    """Create a (112) surface slab for BCC iron."""
    a = lattice_parameter
    lx = a * np.sqrt(2)
    ly = a * np.sqrt(3)
    lz = a * np.sqrt(6) * layers
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    positions = []
    max_range = int(layers * 3 + 5)
    
    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    R = np.array([ex, ey, ez])
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = R @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (0 - eps <= frac_x < 1 - eps and
                        0 - eps <= frac_y < 1 - eps and
                        0 - eps <= frac_z < 1 - eps):
                        positions.append(pos_oriented)
    
    if len(positions) == 0:
        raise ValueError("No atoms found for (112) surface")
    
    positions = np.array(positions)
    _, unique_idx = np.unique(np.round(positions, decimals=6), axis=0, return_index=True)
    positions = positions[unique_idx]
    
    atoms = Atoms(symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=2)
    return atoms


def create_sfe_110_structure(lattice_parameter: float) -> Atoms:
    """Create structure for {110}<111> stacking fault calculation."""
    a = lattice_parameter
    size = (20, 1, 3)
    
    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    R = np.array([ex, ey, ez])
    
    lx = a * np.sqrt(2) * size[0]
    ly = a * np.sqrt(3) * size[1]
    lz = a * np.sqrt(6) * size[2]
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    positions = []
    max_range = int(max(size) * 3 + 5)
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = R @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (0 - eps <= frac_x < 1 - eps and
                        0 - eps <= frac_y < 1 - eps and
                        0 - eps <= frac_z < 1 - eps):
                        positions.append(pos_oriented)
    
    positions = np.array(positions)
    _, unique_idx = np.unique(np.round(positions, decimals=6), axis=0, return_index=True)
    positions = positions[unique_idx]
    
    atoms = Atoms(symbols=['Fe'] * len(positions), positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


def create_sfe_112_structure(lattice_parameter: float) -> Atoms:
    """Create structure for {112}<111> stacking fault calculation."""
    a = lattice_parameter
    size = (15, 1, 1)
    
    ex = np.array([1, 1, -2]) / np.sqrt(6)
    ey = np.array([-1, 1, 0]) / np.sqrt(2)
    ez = np.array([1, 1, 1]) / np.sqrt(3)
    R = np.array([ex, ey, ez])
    
    lx = a * np.sqrt(6) * size[0]
    ly = a * np.sqrt(2) * size[1]
    lz = a * np.sqrt(3) * size[2]
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    positions = []
    max_range = int(max(size) * 3 + 5)
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = R @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (0 - eps <= frac_x < 1 - eps and
                        0 - eps <= frac_y < 1 - eps and
                        0 - eps <= frac_z < 1 - eps):
                        positions.append(pos_oriented)
    
    positions = np.array(positions)
    _, unique_idx = np.unique(np.round(positions, decimals=6), axis=0, return_index=True)
    positions = positions[unique_idx]
    
    atoms = Atoms(symbols=['Fe'] * len(positions), positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


# =============================================================================
# Dislocation Utilities
# =============================================================================

def create_oriented_bcc_cell(
    lattice_parameter: float,
    orient_x: np.ndarray,
    orient_y: np.ndarray,
    orient_z: np.ndarray,
    size: tuple[int, int, int],
    symbol: str = 'Fe',
    center_cell: bool = True
) -> Atoms:
    """Create an oriented BCC supercell."""
    a = lattice_parameter
    ox = orient_x / np.linalg.norm(orient_x)
    oy = orient_y / np.linalg.norm(orient_y)
    oz = orient_z / np.linalg.norm(orient_z)
    R = np.array([ox, oy, oz])
    
    len_x = np.linalg.norm(orient_x)
    len_y = np.linalg.norm(orient_y)
    len_z = np.linalg.norm(orient_z)
    
    half_lx = a * len_x * size[0]
    half_ly = a * len_y * size[1] / 2
    half_lz = a * len_z * size[2] / 2
    
    lx = 2 * half_lx
    ly = 2 * half_ly
    lz = 2 * half_lz
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    positions = []
    max_range = int(max(size) * max(len_x, len_y, len_z) + 10)
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = R @ pos_cubic
                    eps = 1e-8
                    if (-half_lx - eps <= pos_oriented[0] < half_lx - eps and
                        -half_ly - eps <= pos_oriented[1] < half_ly - eps and
                        -half_lz - eps <= pos_oriented[2] < half_lz - eps):
                        positions.append(pos_oriented)
    
    if len(positions) == 0:
        raise ValueError("No atoms found in the oriented cell")
    
    positions = np.array(positions)
    _, unique_idx = np.unique(np.round(positions, decimals=6), axis=0, return_index=True)
    positions = positions[unique_idx]
    
    if not center_cell:
        positions[:, 0] += half_lx
        positions[:, 1] += half_ly
        positions[:, 2] += half_lz
    
    atoms = Atoms(symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=[True, True, False])
    atoms.info['cell_center'] = np.array([0, 0, 0]) if center_cell else np.array([half_lx, half_ly, half_lz])
    atoms.info['half_dims'] = np.array([half_lx, half_ly, half_lz])
    
    return atoms


def create_dislocation_cell(
    lattice_parameter: float,
    dislocation_type: str,
    symbol: str = 'Fe'
) -> Atoms:
    """Create a cell for dislocation simulation."""
    if dislocation_type not in DISLOCATION_CONFIGS:
        raise ValueError(f"Unknown dislocation type: {dislocation_type}")
    config = DISLOCATION_CONFIGS[dislocation_type]
    return create_oriented_bcc_cell(
        lattice_parameter, config['orient_x'], config['orient_y'], config['orient_z'],
        config['size'], symbol, center_cell=True
    )


def get_dislocation_info(dislocation_type: str) -> dict[str, Any]:
    """Get information about a dislocation type."""
    if dislocation_type not in DISLOCATION_CONFIGS:
        raise ValueError(f"Unknown dislocation type: {dislocation_type}")
    return DISLOCATION_CONFIGS[dislocation_type].copy()


def apply_screw_displacement(atoms: Atoms, burgers_magnitude: float) -> None:
    """Apply screw dislocation displacement field."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    if 'half_dims' in atoms.info:
        half_lx = atoms.info['half_dims'][0]
        x_min, x_max = -half_lx, half_lx
        z_mid = 0
    else:
        x_min, x_max = 0, cell[0, 0]
        z_mid = cell[2, 2] / 2
    
    upper_mask = positions[:, 2] > z_mid
    x = positions[upper_mask, 0]
    fraction = (x - x_min) / (x_max - x_min)
    displacement = -burgers_magnitude + fraction * burgers_magnitude
    positions[upper_mask, 1] += displacement
    atoms.set_positions(positions)


def delete_overlapping_atoms(atoms: Atoms, cutoff: float = 0.5) -> Atoms:
    """Delete atoms that are too close to each other."""
    if len(atoms) == 0:
        return atoms
    cutoffs = [cutoff / 2] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False)
    nl.update(atoms)
    to_delete = set()
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            if j > i:
                to_delete.add(j)
    keep_mask = np.ones(len(atoms), dtype=bool)
    keep_mask[list(to_delete)] = False
    return atoms[keep_mask]


def apply_edge_displacement(
    atoms: Atoms,
    burgers_magnitude: float,
    lattice_parameter: float,
    delete_half_plane: bool = True
) -> Atoms:
    """Apply edge dislocation displacement."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    if 'half_dims' in atoms.info:
        half_ly = atoms.info['half_dims'][1]
        y_min, y_max = -half_ly, half_ly
        z_mid = 0
    else:
        y_min, y_max = 0, cell[1, 1]
        z_mid = cell[2, 2] / 2
    
    a = lattice_parameter
    
    if delete_half_plane:
        ymindip = -0.6 * a
        ymaxdip = 0.1 * a
        keep_mask = ~((positions[:, 1] >= ymindip) & (positions[:, 1] <= ymaxdip) & (positions[:, 2] < z_mid))
        atoms = atoms[keep_mask]
        positions = atoms.get_positions()
    
    quarter_b = 0.5 * a
    mask_ll = (positions[:, 1] < 0) & (positions[:, 2] < z_mid)
    if np.any(mask_ll):
        y = positions[mask_ll, 1]
        fraction = (y - y_min) / (0 - y_min)
        positions[mask_ll, 1] += fraction * quarter_b
    
    mask_lr = (positions[:, 1] >= 0) & (positions[:, 2] < z_mid)
    if np.any(mask_lr):
        y = positions[mask_lr, 1]
        fraction = (y - 0) / (y_max - 0)
        positions[mask_lr, 1] += (1 - fraction) * (-quarter_b)
    
    atoms.set_positions(positions)
    atoms = delete_overlapping_atoms(atoms, cutoff=0.5)
    return atoms


def apply_mixed_displacement(
    atoms: Atoms,
    burgers_magnitude: float,
    lattice_parameter: float,
    screw_fraction: float = 0.7
) -> Atoms:
    """Apply mixed dislocation displacement."""
    edge_fraction = 1 - screw_fraction
    if edge_fraction > 0:
        edge_magnitude = burgers_magnitude * edge_fraction
        atoms = apply_edge_displacement(atoms, edge_magnitude, lattice_parameter, delete_half_plane=True)
    if screw_fraction > 0:
        screw_magnitude = burgers_magnitude * screw_fraction
        apply_screw_displacement(atoms, screw_magnitude)
    return atoms


# =============================================================================
# LEFM / Crack Utilities
# =============================================================================

def get_crack_orientation(crack_system: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get crystallographic orientation vectors for a crack system."""
    if crack_system == 1:
        return np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])
    elif crack_system == 2:
        return np.array([0, -1, 1]), np.array([1, 0, 0]), np.array([0, 1, 1])
    elif crack_system == 3:
        return np.array([1, -1, 0]), np.array([1, 1, 0]), np.array([0, 0, 1])
    elif crack_system == 4:
        return np.array([0, 0, -1]), np.array([1, 1, 0]), np.array([1, -1, 0])
    else:
        raise ValueError(f"Invalid crack system: {crack_system}")


def aniso_disp_solution(
    C: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    surfE: float
) -> tuple:
    """Solve the anisotropic LEFM displacement field coefficients."""
    S = np.linalg.inv(C)
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)
    a3 = a3 / np.linalg.norm(a3)
    Q = np.array([a1, a2, a3])
    
    K1 = np.array([
        [Q[0, 0]**2, Q[0, 1]**2, Q[0, 2]**2],
        [Q[1, 0]**2, Q[1, 1]**2, Q[1, 2]**2],
        [Q[2, 0]**2, Q[2, 1]**2, Q[2, 2]**2]
    ])
    K2 = np.array([
        [Q[0, 1]*Q[0, 2], Q[0, 2]*Q[0, 0], Q[0, 0]*Q[0, 1]],
        [Q[1, 1]*Q[1, 2], Q[1, 2]*Q[1, 0], Q[1, 0]*Q[1, 1]],
        [Q[2, 1]*Q[2, 2], Q[2, 2]*Q[2, 0], Q[2, 0]*Q[2, 1]]
    ])
    K3 = np.array([
        [Q[1, 0]*Q[2, 0], Q[1, 1]*Q[2, 1], Q[1, 2]*Q[2, 2]],
        [Q[2, 0]*Q[0, 0], Q[2, 1]*Q[0, 1], Q[2, 2]*Q[0, 2]],
        [Q[0, 0]*Q[1, 0], Q[0, 1]*Q[1, 1], Q[0, 2]*Q[1, 2]]
    ])
    K4 = np.array([
        [Q[1, 1]*Q[2, 2] + Q[1, 2]*Q[2, 1], Q[1, 2]*Q[2, 0] + Q[1, 0]*Q[2, 2], Q[1, 0]*Q[2, 1] + Q[1, 1]*Q[2, 0]],
        [Q[2, 1]*Q[0, 2] + Q[2, 2]*Q[0, 1], Q[2, 2]*Q[0, 0] + Q[2, 0]*Q[0, 2], Q[2, 0]*Q[0, 1] + Q[2, 1]*Q[0, 0]],
        [Q[0, 1]*Q[1, 2] + Q[0, 2]*Q[1, 1], Q[0, 2]*Q[1, 0] + Q[0, 0]*Q[1, 2], Q[0, 0]*Q[1, 1] + Q[0, 1]*Q[1, 0]]
    ])
    
    K_mat = np.vstack((np.hstack((K1, 2*K2)), np.hstack((K3, K4))))
    S_star = np.linalg.inv(K_mat).T @ S @ np.linalg.inv(K_mat)
    
    b_11 = (S_star[0, 0] * S_star[2, 2] - S_star[0, 2]**2) / S_star[2, 2]
    b_22 = (S_star[1, 1] * S_star[2, 2] - S_star[1, 2]**2) / S_star[2, 2]
    b_66 = (S_star[5, 5] * S_star[2, 2] - S_star[2, 5]**2) / S_star[2, 2]
    b_12 = (S_star[0, 1] * S_star[2, 2] - S_star[0, 2] * S_star[1, 2]) / S_star[2, 2]
    b_16 = (S_star[0, 5] * S_star[2, 2] - S_star[0, 2] * S_star[2, 5]) / S_star[2, 2]
    b_26 = (S_star[1, 5] * S_star[2, 2] - S_star[1, 2] * S_star[2, 5]) / S_star[2, 2]
    
    B = np.sqrt((b_11 * b_22 / 2) * (np.sqrt(b_22 / b_11) + ((2 * b_12 + b_66) / (2 * b_11))))
    K_I = np.sqrt(2 * surfE * (1 / (B * 1000)))
    G_I = 2 * surfE
    
    coefvct = [b_11, -2 * b_16, 2 * b_12 + b_66, -2 * b_26, b_22]
    rt = np.roots(coefvct)
    s = rt[np.imag(rt) >= 0]
    if np.real(s[0]) < np.real(s[1]):
        s[0], s[1] = s[1], s[0]
    
    p = np.array([b_11 * s[0]**2 + b_12 - b_16 * s[0], b_11 * s[1]**2 + b_12 - b_16 * s[1]])
    q = np.array([b_12 * s[0] + b_22 / s[0] - b_26, b_12 * s[1] + b_22 / s[1] - b_26])
    
    return s, p, q, K_I, G_I


def compute_lefm_coefficients(
    C11: float,
    C12: float,
    C44: float,
    surface_energy: float,
    crack_system: int
) -> dict[str, Any]:
    """Compute LEFM coefficients for anisotropic crack analysis."""
    C = np.array([
        [C11, C12, C12, 0, 0, 0],
        [C12, C11, C12, 0, 0, 0],
        [C12, C12, C11, 0, 0, 0],
        [0, 0, 0, C44, 0, 0],
        [0, 0, 0, 0, C44, 0],
        [0, 0, 0, 0, 0, C44]
    ])
    a1, a2, a3 = get_crack_orientation(crack_system)
    s, p, q, K_I, G_I = aniso_disp_solution(C, a1, a2, a3, surface_energy)
    return {'s1': s[0], 's2': s[1], 'p1': p[0], 'p2': p[1], 'q1': q[0], 'q2': q[1], 'K_I': K_I, 'G_I': G_I}


def apply_crack_displacement(
    positions: np.ndarray,
    K: float,
    coeffs: dict[str, Any],
    crack_tip: tuple[float, float],
    reference_positions: np.ndarray | None = None
) -> np.ndarray:
    """Apply anisotropic LEFM crack displacement field to atomic positions."""
    s1, s2 = coeffs['s1'], coeffs['s2']
    p1, p2 = coeffs['p1'], coeffs['p2']
    q1, q2 = coeffs['q1'], coeffs['q2']
    xtip, ytip = crack_tip
    
    if reference_positions is None:
        reference_positions = positions.copy()
    
    new_positions = positions.copy()
    x = reference_positions[:, 0] - xtip
    y = reference_positions[:, 1] - ytip
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
    theta = np.arctan2(y, x)
    
    coef = K * np.sqrt(2.0 * r / np.pi)
    z1 = np.cos(theta) + s1 * np.sin(theta)
    z2 = np.cos(theta) + s2 * np.sin(theta)
    
    sqrt_coef_1 = np.sqrt(z2.astype(complex))
    sqrt_coef_2 = np.sqrt(z1.astype(complex))
    denom = s1 - s2
    
    coef_x = (s1 * p2 * sqrt_coef_1 - s2 * p1 * sqrt_coef_2) / denom
    coef_y = (s1 * q2 * sqrt_coef_1 - s2 * q1 * sqrt_coef_2) / denom
    
    new_positions[:, 0] = positions[:, 0] + coef * np.real(coef_x)
    new_positions[:, 1] = positions[:, 1] + coef * np.real(coef_y)
    return new_positions


def compute_incremental_displacement(
    positions: np.ndarray,
    dK: float,
    coeffs: dict[str, Any],
    crack_tip: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the incremental displacement for a K increment."""
    s1, s2 = coeffs['s1'], coeffs['s2']
    p1, p2 = coeffs['p1'], coeffs['p2']
    q1, q2 = coeffs['q1'], coeffs['q2']
    xtip, ytip = crack_tip
    
    x = positions[:, 0] - xtip
    y = positions[:, 1] - ytip
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
    theta = np.arctan2(y, x)
    
    coef = dK * np.sqrt(2.0 * r / np.pi)
    z1 = np.cos(theta) + s1 * np.sin(theta)
    z2 = np.cos(theta) + s2 * np.sin(theta)
    
    sqrt_coef_1 = np.sqrt(z2.astype(complex))
    sqrt_coef_2 = np.sqrt(z1.astype(complex))
    denom = s1 - s2
    
    coef_x = (s1 * p2 * sqrt_coef_1 - s2 * p1 * sqrt_coef_2) / denom
    coef_y = (s1 * q2 * sqrt_coef_1 - s2 * q1 * sqrt_coef_2) / denom
    
    return coef * np.real(coef_x), coef * np.real(coef_y)


def create_crack_cell(
    lattice_parameter: float,
    crack_system: int
) -> tuple[Atoms, tuple[float, float], float]:
    """Create a circular domain for crack simulation."""
    config = CRACK_SYSTEMS_CONFIG[crack_system]
    a0 = lattice_parameter
    
    ox = config['orient_x'] / np.linalg.norm(config['orient_x'])
    oy = config['orient_y'] / np.linalg.norm(config['orient_y'])
    oz = config['orient_z'] / np.linalg.norm(config['orient_z'])
    R = np.array([ox, oy, oz])
    
    len_x = np.linalg.norm(config['orient_x'])
    len_y = np.linalg.norm(config['orient_y'])
    len_z = np.linalg.norm(config['orient_z'])
    
    box_x, box_y = config['box_size']
    lx = 2 * a0 * len_x * box_x
    ly = 2 * a0 * len_y * box_y
    lz = a0 * len_z
    
    positions = []
    max_range = int(max(box_x, box_y) * max(len_x, len_y) + 10)
    
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(2):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a0 * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = R @ pos_cubic
                    if (-lx/2 <= pos_oriented[0] < lx/2 and
                        -ly/2 <= pos_oriented[1] < ly/2 and
                        0 <= pos_oriented[2] < lz):
                        positions.append(pos_oriented)
    
    positions = np.array(positions)
    _, unique_idx = np.unique(np.round(positions, decimals=5), axis=0, return_index=True)
    positions = positions[unique_idx]
    
    tip_x = a0 * config['tip_factors'][0] * 0.25
    tip_y = a0 * config['tip_factors'][1] * 0.25
    
    radius = min(lx/2, ly/2) - 1.0
    r = np.sqrt((positions[:, 0] - tip_x)**2 + (positions[:, 1] - tip_y)**2)
    keep_mask = r < radius
    positions = positions[keep_mask]
    
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    atoms = Atoms(symbols=['Fe'] * len(positions), positions=positions, cell=cell, pbc=[False, False, True])
    atoms.center()
    
    center = np.array([lx/2, ly/2, lz/2])
    crack_tip = (tip_x + center[0] - lx/2, tip_y + center[1] - ly/2)
    
    return atoms, crack_tip, radius


# =============================================================================
# Elastic Calculation Utilities
# =============================================================================

def apply_strain(atoms: Atoms, strain_matrix: np.ndarray) -> Atoms:
    """Apply a strain to the atoms object."""
    atoms_strained = atoms.copy()
    F = np.eye(3) + strain_matrix
    new_cell = atoms_strained.cell @ F.T
    atoms_strained.set_cell(new_cell, scale_atoms=True)
    return atoms_strained


def get_voigt_strain(direction: int, magnitude: float) -> np.ndarray:
    """Get the strain tensor for a given Voigt direction (1-6)."""
    strain = np.zeros((3, 3))
    
    if direction == 1:
        strain[0, 0] = magnitude
    elif direction == 2:
        strain[1, 1] = magnitude
    elif direction == 3:
        strain[2, 2] = magnitude
    elif direction == 4:
        strain[1, 2] = magnitude / 2
        strain[2, 1] = magnitude / 2
    elif direction == 5:
        strain[0, 2] = magnitude / 2
        strain[2, 0] = magnitude / 2
    elif direction == 6:
        strain[0, 1] = magnitude / 2
        strain[1, 0] = magnitude / 2
    
    return strain


def calculate_surface_energy(E_slab: float, E_bulk: float, area: float) -> float:
    """Calculate surface energy in J/m^2."""
    delta_E = E_slab - E_bulk
    return delta_E * EV_TO_J / (2 * area * ANGSTROM_TO_M**2)
