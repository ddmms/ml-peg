"""
Utility functions for BCC iron property calculations.

This module provides structure creation, EOS fitting, dislocation utilities,
and LEFM functions for iron benchmarks.

References
----------
Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
Application to Dislocations and Cracks in Iron.
arXiv:2307.10072. https://arxiv.org/abs/2307.10072
"""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.build import bulk
from ase.neighborlist import NeighborList
import numpy as np
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
    "edge_100_010": {
        "name": "Edge a0[100](010)",
        "orient_x": np.array([0, 0, 1]),
        "orient_y": np.array([1, 0, 0]),
        "orient_z": np.array([0, 1, 0]),
        "size": (1, 50, 20),
        "dim_divisors": (1, 1, 1),  # LAMMPS: sqrt(1)*N*a for all (no /2)
        "burgers": np.array([1, 0, 0]),
        "type": "edge",
        "slip_direction": 1,
        # Half-plane deletion region: LAMMPS uses
        # ymindip=-0.6*sqrt(1)*a, ymaxdip=0.1*sqrt(1)*a
        "delete_axis": 1,  # y-axis
        "delete_min": -0.6,  # factor * a
        "delete_max": 0.1,  # factor * a
    },
    "edge_100_011": {
        "name": "Edge a0[100](011)",
        "orient_x": np.array([0, -1, 1]),
        "orient_y": np.array([1, 0, 0]),
        "orient_z": np.array([0, 1, 1]),
        "size": (1, 80, 22),
        "dim_divisors": (1, 2, 2),  # LAMMPS: sqrt(2)*N, sqrt(1)/2*N, sqrt(2)/2*N
        "burgers": np.array([1, 0, 0]),
        "type": "edge",
        "slip_direction": 1,
        # Half-plane deletion region: LAMMPS uses
        # ymindip=-0.6*sqrt(1)*a, ymaxdip=0.1*sqrt(1)*a
        "delete_axis": 1,  # y-axis
        "delete_min": -0.6,  # factor * a
        "delete_max": 0.1,  # factor * a
    },
    "edge_111_110": {
        "name": "Edge a0/2[111](110)",
        "orient_x": np.array([1, 2, -1]),
        "orient_y": np.array([-1, 1, 1]),
        "orient_z": np.array([1, 0, 1]),
        "size": (1, 40, 20),
        "dim_divisors": (1, 2, 2),  # LAMMPS: sqrt(6)*N, sqrt(3)/2*N, sqrt(2)/2*N
        "burgers": np.array([0.5, 0.5, 0.5]),
        "type": "edge",
        "slip_direction": 1,
        # Half-plane deletion region: LAMMPS uses
        # ymindip=-0.3*sqrt(3)*a, ymaxdip=0.3*sqrt(3)*a
        "delete_axis": 1,  # y-axis
        "delete_min": -0.3 * np.sqrt(3),  # -0.52*a
        "delete_max": 0.3 * np.sqrt(3),  # 0.52*a
    },
    "mixed_111": {
        "name": "Mixed 70.5 deg a0/2[111](110)",
        "orient_x": np.array([1, 2, -1]),
        "orient_y": np.array([-1, 1, 1]),
        "orient_z": np.array([1, 0, 1]),
        "size": (40, 2, 19),
        "dim_divisors": (2, 2, 2),  # LAMMPS: sqrt(6)/2*N for all dimensions
        "burgers": np.array([0.5, 0.5, 0.5]),
        "type": "mixed",
        "slip_direction": 1,
        "screw_fraction": 0.325568,  # cos(71°) from LAMMPS
        "edge_factor": 0.5,  # Additional factor from LAMMPS edge component
        # Half-plane deletion region: LAMMPS uses
        # xmindip=-0.5*sqrt(1)*a, xmaxdip=0.1*sqrt(1)*a
        "delete_axis": 0,  # x-axis (different from edge dislocations!)
        "delete_min": -0.5,  # factor * a
        "delete_max": 0.1,  # factor * a
    },
    "screw_111": {
        "name": "Screw a0/2[111](112)",
        "orient_x": np.array([1, 2, -1]),
        "orient_y": np.array([-1, 1, 1]),
        "orient_z": np.array([1, 0, 1]),
        "size": (60, 2, 19),
        "dim_divisors": (2, 2, 2),  # LAMMPS: sqrt(6)/2*N, sqrt(3)/2*N, sqrt(2)/2*N
        "burgers": np.array([0.5, 0.5, 0.5]),
        "type": "screw",
        "slip_direction": 1,
    },
}

DISLOCATION_TYPES = list(DISLOCATION_CONFIGS.keys())


# =============================================================================
# Crack System Configuration
# =============================================================================

CRACK_SYSTEMS_CONFIG = {
    1: {
        "name": "(100)[010]",
        "orient_x": np.array([0, 0, 1]),
        "orient_y": np.array([1, 0, 0]),
        "orient_z": np.array([0, 1, 0]),
        "surface": "100",
        "box_size": (50, 50),
        "tip_factors": (1.0, 1.0),
    },
    2: {
        "name": "(100)[001]",
        "orient_x": np.array([0, -1, 1]),
        "orient_y": np.array([1, 0, 0]),
        "orient_z": np.array([0, 1, 1]),
        "surface": "100",
        "box_size": (38, 54),
        "tip_factors": (np.sqrt(2), 1.0),
    },
    3: {
        "name": "(110)[001]",
        "orient_x": np.array([1, -1, 0]),
        "orient_y": np.array([1, 1, 0]),
        "orient_z": np.array([0, 0, 1]),
        "surface": "110",
        "box_size": (38, 38),
        "tip_factors": (np.sqrt(2), np.sqrt(2)),
    },
    4: {
        "name": "(110)[1-10]",
        "orient_x": np.array([0, 0, -1]),
        "orient_y": np.array([1, 1, 0]),
        "orient_z": np.array([1, -1, 0]),
        "surface": "110",
        "box_size": (55, 38),
        "tip_factors": (1.0, np.sqrt(2)),
    },
}


# =============================================================================
# EOS Fitting Functions
# =============================================================================


def eos_birch_murnaghan(
    params: tuple[float, float, float, float], vol: np.ndarray
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
    E0, B0, Bp, V0 = params  # noqa: N806
    eta = (vol / V0) ** (1.0 / 3.0)
    return E0 + 9.0 * B0 * V0 / 16.0 * (eta**2 - 1.0) ** 2 * (
        6.0 + Bp * (eta**2 - 1.0) - 4.0 * eta**2
    )


def get_eos_initial_guess(
    vol: np.ndarray, ene: np.ndarray
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
    V0 = -b / (2 * a)  # noqa: N806
    E0 = a * V0**2 + b * V0 + c  # noqa: N806
    B0 = 2 * a * V0  # noqa: N806
    Bp = 4.0  # noqa: N806
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
        """
        Compute residual for EOS fitting.

        Parameters
        ----------
        params : tuple
            EOS parameters (E0, B0, Bp, V0).
        y : np.ndarray
            Observed energies.
        x : np.ndarray
            Volumes.

        Returns
        -------
        np.ndarray
            Residual array (observed - predicted).
        """
        return y - eos_birch_murnaghan(params, x)

    params, _ = leastsq(residual, x0, args=(ene, vol))
    E0, B0, Bp, V0 = params  # noqa: N806

    # Convert bulk modulus to GPa (from eV/Angstrom^3)
    B0_GPa = B0 * EV_PER_A3_TO_GPA  # noqa: N806

    # Calculate lattice parameter for BCC (2 atoms per unit cell)
    a0 = (V0 * 2) ** (1.0 / 3.0)

    return {"E0": E0, "B0": B0_GPa, "Bp": Bp, "V0": V0, "a0": a0}


# =============================================================================
# Structure Creation Functions
# =============================================================================


def create_bcc_supercell(
    lattice_parameter: float, size: tuple = (4, 4, 4), symbol: str = "Fe"
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
    unit_cell = bulk(symbol, "bcc", a=lattice_parameter, cubic=True)
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

    return Atoms(symbols=["Fe", "Fe"], positions=positions, cell=cell, pbc=True)


def create_surface_100(
    lattice_parameter: float, layers: int = 10, vacuum: float = 0.0, symbol: str = "Fe"
) -> Atoms:
    """
    Create a (100) surface slab for BCC iron.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.
    layers : int, optional
        Number of atomic layers (default: 10).
    vacuum : float, optional
        Vacuum thickness in Angstroms (default: 0.0).
    symbol : str, optional
        Chemical symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object with the (100) surface slab.
    """
    a = lattice_parameter
    cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a * layers]])

    positions = []
    for k in range(layers):
        positions.append([0, 0, k * a])
        positions.append([0.5 * a, 0.5 * a, (k + 0.5) * a])

    atoms = Atoms(
        symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True
    )
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=2)
    return atoms


def create_surface_110(
    lattice_parameter: float, layers: int = 10, vacuum: float = 0.0, symbol: str = "Fe"
) -> Atoms:
    """
    Create a (110) surface slab for BCC iron.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.
    layers : int, optional
        Number of atomic layers (default: 10).
    vacuum : float, optional
        Vacuum thickness in Angstroms (default: 0.0).
    symbol : str, optional
        Chemical symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object with the (110) surface slab.
    """
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

    atoms = Atoms(
        symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True
    )
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=2)
    return atoms


def create_surface_111(
    lattice_parameter: float,
    size: tuple = (3, 15, 3),
    vacuum: float = 0.0,
    symbol: str = "Fe",
) -> Atoms:
    """
    Create a (111) surface slab for BCC iron.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.
    size : tuple, optional
        Cell size as (nx, ny, nz) (default: (3, 15, 3)).
    vacuum : float, optional
        Vacuum thickness in Angstroms (default: 0.0).
    symbol : str, optional
        Chemical symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object with the (111) surface slab.
    """
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
    rot = np.array([ex, ey, ez])

    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = rot @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (
                        0 - eps <= frac_x < 1 - eps
                        and 0 - eps <= frac_y < 1 - eps
                        and 0 - eps <= frac_z < 1 - eps
                    ):
                        positions.append(pos_oriented)

    if len(positions) == 0:
        raise ValueError("No atoms found for (111) surface")

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=6), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    atoms = Atoms(
        symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True
    )
    atoms.wrap()
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=1)
    return atoms


def create_surface_112(
    lattice_parameter: float, layers: int = 15, vacuum: float = 0.0, symbol: str = "Fe"
) -> Atoms:
    """
    Create a (112) surface slab for BCC iron.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.
    layers : int, optional
        Number of atomic layers (default: 15).
    vacuum : float, optional
        Vacuum thickness in Angstroms (default: 0.0).
    symbol : str, optional
        Chemical symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object with the (112) surface slab.
    """
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
    rot = np.array([ex, ey, ez])

    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = rot @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (
                        0 - eps <= frac_x < 1 - eps
                        and 0 - eps <= frac_y < 1 - eps
                        and 0 - eps <= frac_z < 1 - eps
                    ):
                        positions.append(pos_oriented)

    if len(positions) == 0:
        raise ValueError("No atoms found for (112) surface")

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=6), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    atoms = Atoms(
        symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True
    )
    atoms.wrap()
    if vacuum > 0:
        atoms.center(vacuum=vacuum, axis=2)
    return atoms


def create_sfe_110_structure(lattice_parameter: float) -> Atoms:
    """
    Create structure for {110}<111> stacking fault calculation.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.

    Returns
    -------
    Atoms
        ASE Atoms object for SFE calculation.
    """
    a = lattice_parameter
    size = (20, 1, 3)

    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    rot = np.array([ex, ey, ez])

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
                    pos_oriented = rot @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (
                        0 - eps <= frac_x < 1 - eps
                        and 0 - eps <= frac_y < 1 - eps
                        and 0 - eps <= frac_z < 1 - eps
                    ):
                        positions.append(pos_oriented)

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=6), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    atoms = Atoms(
        symbols=["Fe"] * len(positions), positions=positions, cell=cell, pbc=True
    )
    atoms.wrap()
    return atoms


def create_sfe_112_structure(lattice_parameter: float) -> Atoms:
    """
    Create structure for {112}<111> stacking fault calculation.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.

    Returns
    -------
    Atoms
        ASE Atoms object for SFE calculation.
    """
    a = lattice_parameter
    size = (15, 1, 1)

    ex = np.array([1, 1, -2]) / np.sqrt(6)
    ey = np.array([-1, 1, 0]) / np.sqrt(2)
    ez = np.array([1, 1, 1]) / np.sqrt(3)
    rot = np.array([ex, ey, ez])

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
                    pos_oriented = rot @ pos_cubic
                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz
                    eps = 1e-8
                    if (
                        0 - eps <= frac_x < 1 - eps
                        and 0 - eps <= frac_y < 1 - eps
                        and 0 - eps <= frac_z < 1 - eps
                    ):
                        positions.append(pos_oriented)

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=6), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    atoms = Atoms(
        symbols=["Fe"] * len(positions), positions=positions, cell=cell, pbc=True
    )
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
    dim_divisors: tuple[int, int, int] = (1, 2, 2),
    symbol: str = "Fe",
    center_cell: bool = True,
) -> Atoms:
    """
    Create an oriented BCC supercell.

    Parameters
    ----------
    lattice_parameter : float
        The BCC lattice parameter in Angstroms.
    orient_x, orient_y, orient_z : np.ndarray
        Crystal orientation vectors for x, y, z axes.
    size : tuple[int, int, int]
        Number of periodic units in each direction.
    dim_divisors : tuple[int, int, int]
        Divisors for half-dimension calculation in each direction.
        Formula: half_dim = a * ||orient|| * size / divisor
        Use (1, 1, 1) for no division, (2, 2, 2) for /2 on all, etc.
        Must match LAMMPS conventions for each dislocation type.
    symbol : str
        Atomic symbol (default 'Fe').
    center_cell : bool
        If True, center cell at origin; if False, shift to positive coords.

    Returns
    -------
    Atoms
        ASE Atoms object with the oriented BCC cell.
    """
    a = lattice_parameter
    ox = orient_x / np.linalg.norm(orient_x)
    oy = orient_y / np.linalg.norm(orient_y)
    oz = orient_z / np.linalg.norm(orient_z)
    rot = np.array([ox, oy, oz])

    len_x = np.linalg.norm(orient_x)
    len_y = np.linalg.norm(orient_y)
    len_z = np.linalg.norm(orient_z)

    # Use dim_divisors to match LAMMPS half-dimension conventions
    half_lx = a * len_x * size[0] / dim_divisors[0]
    half_ly = a * len_y * size[1] / dim_divisors[1]
    half_lz = a * len_z * size[2] / dim_divisors[2]

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
                    pos_oriented = rot @ pos_cubic
                    eps = 1e-8
                    if (
                        -half_lx - eps <= pos_oriented[0] < half_lx - eps
                        and -half_ly - eps <= pos_oriented[1] < half_ly - eps
                        and -half_lz - eps <= pos_oriented[2] < half_lz - eps
                    ):
                        positions.append(pos_oriented)

    if len(positions) == 0:
        raise ValueError("No atoms found in the oriented cell")

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=6), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    if not center_cell:
        positions[:, 0] += half_lx
        positions[:, 1] += half_ly
        positions[:, 2] += half_lz

    atoms = Atoms(
        symbols=[symbol] * len(positions),
        positions=positions,
        cell=cell,
        pbc=[True, True, False],
    )
    atoms.info["cell_center"] = (
        np.array([0, 0, 0]) if center_cell else np.array([half_lx, half_ly, half_lz])
    )
    atoms.info["half_dims"] = np.array([half_lx, half_ly, half_lz])

    return atoms


def create_dislocation_cell(
    lattice_parameter: float, dislocation_type: str, symbol: str = "Fe"
) -> Atoms:
    """
    Create a cell for dislocation simulation.

    Uses the dim_divisors from DISLOCATION_CONFIGS to match LAMMPS cell sizes.

    Parameters
    ----------
    lattice_parameter : float
        The BCC lattice parameter in Angstroms.
    dislocation_type : str
        Type of dislocation (e.g., 'edge_100_010', 'screw_111').
    symbol : str, optional
        Atomic symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object with the dislocation cell.
    """
    if dislocation_type not in DISLOCATION_CONFIGS:
        raise ValueError(f"Unknown dislocation type: {dislocation_type}")
    config = DISLOCATION_CONFIGS[dislocation_type]
    return create_oriented_bcc_cell(
        lattice_parameter,
        config["orient_x"],
        config["orient_y"],
        config["orient_z"],
        config["size"],
        dim_divisors=config.get("dim_divisors", (1, 2, 2)),
        symbol=symbol,
        center_cell=True,
    )


def get_dislocation_info(dislocation_type: str) -> dict[str, Any]:
    """
    Get information about a dislocation type.

    Parameters
    ----------
    dislocation_type : str
        Type of dislocation (e.g., 'edge_100_010', 'screw_111').

    Returns
    -------
    dict
        Dictionary with dislocation configuration parameters.
    """
    if dislocation_type not in DISLOCATION_CONFIGS:
        raise ValueError(f"Unknown dislocation type: {dislocation_type}")
    return DISLOCATION_CONFIGS[dislocation_type].copy()


def apply_screw_displacement(atoms: Atoms, burgers_magnitude: float) -> None:
    """
    Apply screw dislocation displacement field.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with the dislocation cell.
    burgers_magnitude : float
        Magnitude of the Burgers vector.
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    if "half_dims" in atoms.info:
        half_lx = atoms.info["half_dims"][0]
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
    """
    Delete atoms that are too close to each other.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    cutoff : float, optional
        Distance cutoff for overlap detection (default: 0.5 Angstroms).

    Returns
    -------
    Atoms
        ASE Atoms object with overlapping atoms removed.
    """
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
    delete_half_plane: bool = True,
    delete_axis: int = 1,
    delete_min: float = -0.6,
    delete_max: float = 0.1,
) -> Atoms:
    """
    Apply edge dislocation displacement.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with the dislocation cell.
    burgers_magnitude : float
        Magnitude of the Burgers vector.
    lattice_parameter : float
        BCC lattice parameter.
    delete_half_plane : bool
        If True, delete atoms in the half-plane region.
    delete_axis : int
        Axis along which to delete atoms (0=x, 1=y). Default 1 (y).
    delete_min : float
        Minimum position factor for deletion region (factor * a).
    delete_max : float
        Maximum position factor for deletion region (factor * a).

    Returns
    -------
    Atoms
        Atoms object with edge dislocation displacement applied.

    Notes
    -----
    LAMMPS deletion regions by dislocation type:
    - edge_100_010: y from -0.6a to 0.1a
    - edge_100_011: y from -0.6a to 0.1a
    - edge_111_110: y from -0.3*sqrt(3)*a to 0.3*sqrt(3)*a
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell()

    if "half_dims" in atoms.info:
        half_ly = atoms.info["half_dims"][1]
        y_min, y_max = -half_ly, half_ly
        z_mid = 0
    else:
        y_min, y_max = 0, cell[1, 1]
        z_mid = cell[2, 2] / 2

    a = lattice_parameter

    if delete_half_plane:
        # Use configurable deletion region
        dip_min = delete_min * a
        dip_max = delete_max * a
        # Delete atoms in the specified axis range, below z_mid
        keep_mask = ~(
            (positions[:, delete_axis] >= dip_min)
            & (positions[:, delete_axis] <= dip_max)
            & (positions[:, 2] < z_mid)
        )
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
    return delete_overlapping_atoms(atoms, cutoff=0.5)


def apply_mixed_displacement(
    atoms: Atoms,
    burgers_magnitude: float,
    lattice_parameter: float,
    screw_fraction: float = 0.325568,
    edge_factor: float = 0.5,
    delete_axis: int = 0,
    delete_min: float = -0.5,
    delete_max: float = 0.1,
) -> Atoms:
    """
    Apply mixed dislocation displacement.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with the dislocation cell.
    burgers_magnitude : float
        Magnitude of the Burgers vector (sqrt(3)/2 * a for <111>).
    lattice_parameter : float
        BCC lattice parameter.
    screw_fraction : float
        Fraction of Burgers vector for screw component.
        Default 0.325568 = cos(71°) from LAMMPS M111 dislocation.
    edge_factor : float
        Additional factor applied to edge component.
        Default 0.5 from LAMMPS: edgeB = 0.5 * sin(71°) * |b|.
    delete_axis : int
        Axis along which to delete atoms (0=x, 1=y). Default 0 (x) for mixed.
    delete_min : float
        Minimum position factor for deletion region (factor * a).
    delete_max : float
        Maximum position factor for deletion region (factor * a).

    Returns
    -------
    Atoms
        Atoms object with mixed dislocation displacement applied.

    Notes
    -----
    LAMMPS M111 uses:
    - Screw: msft_disp = 0.325568 * sqrt(3)/2 * a = cos(71°) * |b|
    - Edge: edgeB = 0.5 * 0.94 * sqrt(3)/2 * a = 0.5 * sin(71°) * |b|
    - Deletion region: x from -0.5a to 0.1a (along x-axis, not y!)
    """
    # Calculate edge fraction from geometry (sin of character angle)
    # For 71° angle: sin(71°) ≈ 0.9455
    edge_fraction = np.sqrt(1 - screw_fraction**2)  # sin(θ) from cos(θ)

    if edge_fraction > 0:
        # Apply edge_factor (0.5 from LAMMPS)
        edge_magnitude = burgers_magnitude * edge_fraction * edge_factor
        atoms = apply_edge_displacement(
            atoms,
            edge_magnitude,
            lattice_parameter,
            delete_half_plane=True,
            delete_axis=delete_axis,
            delete_min=delete_min,
            delete_max=delete_max,
        )
    if screw_fraction > 0:
        screw_magnitude = burgers_magnitude * screw_fraction
        apply_screw_displacement(atoms, screw_magnitude)
    return atoms


# =============================================================================
# LEFM / Crack Utilities
# =============================================================================


def get_crack_orientation(
    crack_system: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get crystallographic orientation vectors for a crack system.

    Parameters
    ----------
    crack_system : int
        Crack system index (1-4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Orientation vectors (a1, a2, a3) for the crack system.
    """
    if crack_system == 1:
        return np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])
    if crack_system == 2:
        return np.array([0, -1, 1]), np.array([1, 0, 0]), np.array([0, 1, 1])
    if crack_system == 3:
        return np.array([1, -1, 0]), np.array([1, 1, 0]), np.array([0, 0, 1])
    if crack_system == 4:
        return np.array([0, 0, -1]), np.array([1, 1, 0]), np.array([1, -1, 0])
    raise ValueError(f"Invalid crack system: {crack_system}")


def aniso_disp_solution(
    c_mat: np.ndarray, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, surf_e: float
) -> tuple:
    """
    Solve the anisotropic LEFM displacement field coefficients.

    Parameters
    ----------
    c_mat : np.ndarray
        6x6 elastic stiffness matrix in Voigt notation.
    a1 : np.ndarray
        First orientation vector.
    a2 : np.ndarray
        Second orientation vector.
    a3 : np.ndarray
        Third orientation vector.
    surf_e : float
        Surface energy in J/m^2.

    Returns
    -------
    tuple
        (s, p, q, K_I, G_I) where s, p, q are complex coefficients,
        K_I is the Griffith stress intensity factor, and G_I is the
        energy release rate.
    """
    s_inv = np.linalg.inv(c_mat)
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)
    a3 = a3 / np.linalg.norm(a3)
    q = np.array([a1, a2, a3])

    k1 = np.array(
        [
            [q[0, 0] ** 2, q[0, 1] ** 2, q[0, 2] ** 2],
            [q[1, 0] ** 2, q[1, 1] ** 2, q[1, 2] ** 2],
            [q[2, 0] ** 2, q[2, 1] ** 2, q[2, 2] ** 2],
        ]
    )
    k2 = np.array(
        [
            [q[0, 1] * q[0, 2], q[0, 2] * q[0, 0], q[0, 0] * q[0, 1]],
            [q[1, 1] * q[1, 2], q[1, 2] * q[1, 0], q[1, 0] * q[1, 1]],
            [q[2, 1] * q[2, 2], q[2, 2] * q[2, 0], q[2, 0] * q[2, 1]],
        ]
    )
    k3 = np.array(
        [
            [q[1, 0] * q[2, 0], q[1, 1] * q[2, 1], q[1, 2] * q[2, 2]],
            [q[2, 0] * q[0, 0], q[2, 1] * q[0, 1], q[2, 2] * q[0, 2]],
            [q[0, 0] * q[1, 0], q[0, 1] * q[1, 1], q[0, 2] * q[1, 2]],
        ]
    )
    k4 = np.array(
        [
            [
                q[1, 1] * q[2, 2] + q[1, 2] * q[2, 1],
                q[1, 2] * q[2, 0] + q[1, 0] * q[2, 2],
                q[1, 0] * q[2, 1] + q[1, 1] * q[2, 0],
            ],
            [
                q[2, 1] * q[0, 2] + q[2, 2] * q[0, 1],
                q[2, 2] * q[0, 0] + q[2, 0] * q[0, 2],
                q[2, 0] * q[0, 1] + q[2, 1] * q[0, 0],
            ],
            [
                q[0, 1] * q[1, 2] + q[0, 2] * q[1, 1],
                q[0, 2] * q[1, 0] + q[0, 0] * q[1, 2],
                q[0, 0] * q[1, 1] + q[0, 1] * q[1, 0],
            ],
        ]
    )

    k_mat = np.vstack((np.hstack((k1, 2 * k2)), np.hstack((k3, k4))))
    s_star = np.linalg.inv(k_mat).T @ s_inv @ np.linalg.inv(k_mat)

    b_11 = (s_star[0, 0] * s_star[2, 2] - s_star[0, 2] ** 2) / s_star[2, 2]
    b_22 = (s_star[1, 1] * s_star[2, 2] - s_star[1, 2] ** 2) / s_star[2, 2]
    b_66 = (s_star[5, 5] * s_star[2, 2] - s_star[2, 5] ** 2) / s_star[2, 2]
    b_12 = (s_star[0, 1] * s_star[2, 2] - s_star[0, 2] * s_star[1, 2]) / s_star[2, 2]
    b_16 = (s_star[0, 5] * s_star[2, 2] - s_star[0, 2] * s_star[2, 5]) / s_star[2, 2]
    b_26 = (s_star[1, 5] * s_star[2, 2] - s_star[1, 2] * s_star[2, 5]) / s_star[2, 2]

    b_factor = np.sqrt(
        (b_11 * b_22 / 2) * (np.sqrt(b_22 / b_11) + ((2 * b_12 + b_66) / (2 * b_11)))
    )
    k_i = np.sqrt(2 * surf_e * (1 / (b_factor * 1000)))
    g_i = 2 * surf_e

    coefvct = [b_11, -2 * b_16, 2 * b_12 + b_66, -2 * b_26, b_22]
    rt = np.roots(coefvct)
    s = rt[np.imag(rt) >= 0]
    if np.real(s[0]) < np.real(s[1]):
        s[0], s[1] = s[1], s[0]

    p = np.array(
        [b_11 * s[0] ** 2 + b_12 - b_16 * s[0], b_11 * s[1] ** 2 + b_12 - b_16 * s[1]]
    )
    q = np.array([b_12 * s[0] + b_22 / s[0] - b_26, b_12 * s[1] + b_22 / s[1] - b_26])

    return s, p, q, k_i, g_i


def compute_lefm_coefficients(
    c11: float, c12: float, c44: float, surface_energy: float, crack_system: int
) -> dict[str, Any]:
    """
    Compute LEFM coefficients for anisotropic crack analysis.

    Parameters
    ----------
    c11 : float
        Elastic constant C11 in GPa.
    c12 : float
        Elastic constant C12 in GPa.
    c44 : float
        Elastic constant C44 in GPa.
    surface_energy : float
        Surface energy in J/m^2.
    crack_system : int
        Crack system index (1-4).

    Returns
    -------
    dict
        Dictionary with LEFM coefficients s1, s2, p1, p2, q1, q2, K_I, G_I.
    """
    c_mat = np.array(
        [
            [c11, c12, c12, 0, 0, 0],
            [c12, c11, c12, 0, 0, 0],
            [c12, c12, c11, 0, 0, 0],
            [0, 0, 0, c44, 0, 0],
            [0, 0, 0, 0, c44, 0],
            [0, 0, 0, 0, 0, c44],
        ]
    )
    a1, a2, a3 = get_crack_orientation(crack_system)
    s, p, q, k_i, g_i = aniso_disp_solution(c_mat, a1, a2, a3, surface_energy)
    return {
        "s1": s[0],
        "s2": s[1],
        "p1": p[0],
        "p2": p[1],
        "q1": q[0],
        "q2": q[1],
        "K_I": k_i,
        "G_I": g_i,
    }


def apply_crack_displacement(
    positions: np.ndarray,
    k_sif: float,
    coeffs: dict[str, Any],
    crack_tip: tuple[float, float],
    reference_positions: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply anisotropic LEFM crack displacement field to atomic positions.

    Parameters
    ----------
    positions : np.ndarray
        Current atomic positions (N, 3).
    k_sif : float
        Stress intensity factor.
    coeffs : dict
        LEFM coefficients from compute_lefm_coefficients.
    crack_tip : tuple[float, float]
        (x, y) coordinates of the crack tip.
    reference_positions : np.ndarray, optional
        Reference positions for displacement calculation.

    Returns
    -------
    np.ndarray
        Updated atomic positions with crack displacement applied.
    """
    s1, s2 = coeffs["s1"], coeffs["s2"]
    p1, p2 = coeffs["p1"], coeffs["p2"]
    q1, q2 = coeffs["q1"], coeffs["q2"]
    xtip, ytip = crack_tip

    if reference_positions is None:
        reference_positions = positions.copy()

    new_positions = positions.copy()
    x = reference_positions[:, 0] - xtip
    y = reference_positions[:, 1] - ytip
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
    theta = np.arctan2(y, x)

    coef = k_sif * np.sqrt(2.0 * r / np.pi)
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
    dk: float,
    coeffs: dict[str, Any],
    crack_tip: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the incremental displacement for a K increment.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions (N, 3).
    dk : float
        Increment in stress intensity factor.
    coeffs : dict
        LEFM coefficients from compute_lefm_coefficients.
    crack_tip : tuple[float, float]
        (x, y) coordinates of the crack tip.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dx, dy) incremental displacements for each atom.
    """
    s1, s2 = coeffs["s1"], coeffs["s2"]
    p1, p2 = coeffs["p1"], coeffs["p2"]
    q1, q2 = coeffs["q1"], coeffs["q2"]
    xtip, ytip = crack_tip

    x = positions[:, 0] - xtip
    y = positions[:, 1] - ytip
    r = np.maximum(np.sqrt(x**2 + y**2), 1e-10)
    theta = np.arctan2(y, x)

    coef = dk * np.sqrt(2.0 * r / np.pi)
    z1 = np.cos(theta) + s1 * np.sin(theta)
    z2 = np.cos(theta) + s2 * np.sin(theta)

    sqrt_coef_1 = np.sqrt(z2.astype(complex))
    sqrt_coef_2 = np.sqrt(z1.astype(complex))
    denom = s1 - s2

    coef_x = (s1 * p2 * sqrt_coef_1 - s2 * p1 * sqrt_coef_2) / denom
    coef_y = (s1 * q2 * sqrt_coef_1 - s2 * q1 * sqrt_coef_2) / denom

    return coef * np.real(coef_x), coef * np.real(coef_y)


def create_crack_cell(
    lattice_parameter: float, crack_system: int
) -> tuple[Atoms, tuple[float, float], float]:
    """
    Create a circular domain for crack simulation.

    Parameters
    ----------
    lattice_parameter : float
        Lattice parameter in Angstroms.
    crack_system : int
        Crack system index (1-4).

    Returns
    -------
    tuple[Atoms, tuple[float, float], float]
        (atoms, crack_tip, radius) where atoms is the ASE Atoms object,
        crack_tip is the (x, y) position of the crack tip, and radius
        is the domain radius.
    """
    config = CRACK_SYSTEMS_CONFIG[crack_system]
    a0 = lattice_parameter

    ox = config["orient_x"] / np.linalg.norm(config["orient_x"])
    oy = config["orient_y"] / np.linalg.norm(config["orient_y"])
    oz = config["orient_z"] / np.linalg.norm(config["orient_z"])
    rot = np.array([ox, oy, oz])

    len_x = np.linalg.norm(config["orient_x"])
    len_y = np.linalg.norm(config["orient_y"])
    len_z = np.linalg.norm(config["orient_z"])

    box_x, box_y = config["box_size"]
    lx = 2 * a0 * len_x * box_x
    ly = 2 * a0 * len_y * box_y
    lz = a0 * len_z

    positions = []
    max_range = int(max(box_x, box_y) * max(len_x, len_y) + 10)

    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(2):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a0 * np.array(
                        [i + basis[0], j + basis[1], k + basis[2]]
                    )
                    pos_oriented = rot @ pos_cubic
                    if (
                        -lx / 2 <= pos_oriented[0] < lx / 2
                        and -ly / 2 <= pos_oriented[1] < ly / 2
                        and 0 <= pos_oriented[2] < lz
                    ):
                        positions.append(pos_oriented)

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=5), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    tip_x = a0 * config["tip_factors"][0] * 0.25
    tip_y = a0 * config["tip_factors"][1] * 0.25

    radius = min(lx / 2, ly / 2) - 1.0
    r = np.sqrt((positions[:, 0] - tip_x) ** 2 + (positions[:, 1] - tip_y) ** 2)
    keep_mask = r < radius
    positions = positions[keep_mask]

    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])
    atoms = Atoms(
        symbols=["Fe"] * len(positions),
        positions=positions,
        cell=cell,
        pbc=[False, False, True],
    )
    atoms.center()

    center = np.array([lx / 2, ly / 2, lz / 2])
    crack_tip = (tip_x + center[0] - lx / 2, tip_y + center[1] - ly / 2)

    return atoms, crack_tip, radius


# =============================================================================
# Elastic Calculation Utilities
# =============================================================================


def apply_strain(atoms: Atoms, strain_matrix: np.ndarray) -> Atoms:
    """
    Apply a strain to the atoms object.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    strain_matrix : np.ndarray
        3x3 strain matrix.

    Returns
    -------
    Atoms
        Strained ASE Atoms object.
    """
    atoms_strained = atoms.copy()
    deformation = np.eye(3) + strain_matrix
    new_cell = atoms_strained.cell @ deformation.T
    atoms_strained.set_cell(new_cell, scale_atoms=True)
    return atoms_strained


def get_voigt_strain(direction: int, magnitude: float) -> np.ndarray:
    """
    Get the strain tensor for a given Voigt direction (1-6).

    Parameters
    ----------
    direction : int
        Voigt direction (1-6).
    magnitude : float
        Strain magnitude.

    Returns
    -------
    np.ndarray
        3x3 strain matrix.
    """
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


def calculate_surface_energy(e_slab: float, e_bulk: float, area: float) -> float:
    """
    Calculate surface energy in J/m^2.

    Parameters
    ----------
    e_slab : float
        Total energy of the slab with vacuum (eV).
    e_bulk : float
        Total energy of the bulk reference (eV).
    area : float
        Surface area (Angstrom^2).

    Returns
    -------
    float
        Surface energy in J/m^2.
    """
    delta_e = e_slab - e_bulk
    return delta_e * EV_TO_J / (2 * area * ANGSTROM_TO_M**2)
