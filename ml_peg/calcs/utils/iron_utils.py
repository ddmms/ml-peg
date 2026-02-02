"""
Utility functions for BCC iron property calculations.

This module provides structure creation and EOS fitting functions for iron benchmarks.
"""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.build import bulk
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


def _create_oriented_bcc_structure(
    lattice_parameter: float,
    rotation: np.ndarray,
    cell_dims: tuple[float, float, float],
    max_range: int,
    symbol: str = "Fe",
    wrap: bool = True,
) -> Atoms:
    """
    Create BCC structure with given orientation using rotation matrix.

    This is a generic function used by several oriented structure creators.

    Parameters
    ----------
    lattice_parameter : float
        BCC lattice parameter in Angstroms.
    rotation : np.ndarray
        3x3 rotation matrix (rows are the new basis vectors).
    cell_dims : tuple[float, float, float]
        Cell dimensions (lx, ly, lz) in Angstroms.
    max_range : int
        Range for scanning cubic positions.
    symbol : str, optional
        Chemical symbol (default: 'Fe').
    wrap : bool, optional
        Whether to wrap positions into cell (default: True).

    Returns
    -------
    Atoms
        ASE Atoms object with oriented structure.
    """
    a = lattice_parameter
    lx, ly, lz = cell_dims
    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])

    positions = []
    eps = 1e-8

    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                for basis in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    pos_cubic = a * np.array([i + basis[0], j + basis[1], k + basis[2]])
                    pos_oriented = rotation @ pos_cubic

                    frac_x = pos_oriented[0] / lx
                    frac_y = pos_oriented[1] / ly
                    frac_z = pos_oriented[2] / lz

                    if (
                        0 - eps <= frac_x < 1 - eps
                        and 0 - eps <= frac_y < 1 - eps
                        and 0 - eps <= frac_z < 1 - eps
                    ):
                        positions.append(pos_oriented)

    if len(positions) == 0:
        raise ValueError("No atoms found for oriented structure")

    positions = np.array(positions)
    _, unique_idx = np.unique(
        np.round(positions, decimals=6), axis=0, return_index=True
    )
    positions = positions[unique_idx]

    atoms = Atoms(
        symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True
    )

    if wrap:
        atoms.wrap()

    return atoms


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

    # Rotation matrix for (111) orientation
    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    rotation = np.array([ex, ey, ez])

    cell_dims = (
        a * np.sqrt(2) * size[0],
        a * np.sqrt(3) * size[1],
        a * np.sqrt(6) * size[2],
    )
    max_range = int(max(size) * 3 + 5)

    atoms = _create_oriented_bcc_structure(
        lattice_parameter, rotation, cell_dims, max_range, symbol
    )

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

    # Rotation matrix for (112) orientation
    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    rotation = np.array([ex, ey, ez])

    cell_dims = (a * np.sqrt(2), a * np.sqrt(3), a * np.sqrt(6) * layers)
    max_range = int(layers * 3 + 5)

    atoms = _create_oriented_bcc_structure(
        lattice_parameter, rotation, cell_dims, max_range, symbol
    )

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

    # Rotation matrix for {110} orientation
    ex = np.array([-1, 1, 0]) / np.sqrt(2)
    ey = np.array([1, 1, 1]) / np.sqrt(3)
    ez = np.array([1, 1, -2]) / np.sqrt(6)
    rotation = np.array([ex, ey, ez])

    cell_dims = (
        a * np.sqrt(2) * size[0],
        a * np.sqrt(3) * size[1],
        a * np.sqrt(6) * size[2],
    )
    max_range = int(max(size) * 3 + 5)

    return _create_oriented_bcc_structure(
        lattice_parameter, rotation, cell_dims, max_range
    )


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

    # Rotation matrix for {112} orientation
    ex = np.array([1, 1, -2]) / np.sqrt(6)
    ey = np.array([-1, 1, 0]) / np.sqrt(2)
    ez = np.array([1, 1, 1]) / np.sqrt(3)
    rotation = np.array([ex, ey, ez])

    cell_dims = (
        a * np.sqrt(6) * size[0],
        a * np.sqrt(2) * size[1],
        a * np.sqrt(3) * size[2],
    )
    max_range = int(max(size) * 3 + 5)

    return _create_oriented_bcc_structure(
        lattice_parameter, rotation, cell_dims, max_range
    )


# =============================================================================
# Traction-Separation Structure Functions
# =============================================================================


def create_ts_100_structure(
    lattice_parameter: float, layers: int = 36, symbol: str = "Fe"
) -> Atoms:
    """
    Create structure for (100) traction-separation calculation.

    Orientation: x=[100], y=[010], z=[001]
    Slab: 1x1xlayers lattice units
    Cleavage plane: perpendicular to z

    Parameters
    ----------
    lattice_parameter : float
        BCC lattice parameter in Angstroms.
    layers : int, optional
        Number of layers in z direction (default: 36).
    symbol : str, optional
        Chemical symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object for T-S calculation.
    """
    a = lattice_parameter

    # Cell dimensions
    lx = a
    ly = a
    lz = a * layers

    cell = np.array([[lx, 0, 0], [0, ly, 0], [0, 0, lz]])

    # Generate BCC atoms
    positions = []
    for k in range(layers):
        # Two atoms per unit cell in z
        positions.append([0, 0, k * a])
        positions.append([0.5 * a, 0.5 * a, (k + 0.5) * a])

    return Atoms(
        symbols=[symbol] * len(positions), positions=positions, cell=cell, pbc=True
    )


def create_ts_110_structure(
    lattice_parameter: float, layers: int = 10, symbol: str = "Fe"
) -> Atoms:
    """
    Create structure for (110) traction-separation calculation.

    Orientation: x=[100], y=[01-1], z=[011]
    Slab: 1x1xlayers lattice units
    Cleavage plane: perpendicular to z (which is [011])

    Parameters
    ----------
    lattice_parameter : float
        BCC lattice parameter in Angstroms.
    layers : int, optional
        Number of layers in z direction (default: 10).
    symbol : str, optional
        Chemical symbol (default: 'Fe').

    Returns
    -------
    Atoms
        ASE Atoms object for T-S calculation.
    """
    a = lattice_parameter

    # Rotation matrix for T-S 110 orientation
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, -1]) / np.sqrt(2)
    ez = np.array([0, 1, 1]) / np.sqrt(2)
    rotation = np.array([ex, ey, ez])

    cell_dims = (a, a * np.sqrt(2), a * np.sqrt(2) * layers)
    max_range = int(layers * 2 + 5)

    return _create_oriented_bcc_structure(
        lattice_parameter, rotation, cell_dims, max_range, symbol
    )


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
    F = np.eye(3) + strain_matrix  # noqa: N806
    new_cell = atoms_strained.cell @ F.T
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


def calculate_surface_energy(
    E_slab: float,  # noqa: N803
    E_bulk: float,  # noqa: N803
    area: float,
) -> float:
    """
    Calculate surface energy in J/m^2.

    Parameters
    ----------
    E_slab : float
        Total energy of the slab with vacuum (eV).
    E_bulk : float
        Total energy of the bulk reference (eV).
    area : float
        Surface area (Angstrom^2).

    Returns
    -------
    float
        Surface energy in J/m^2.
    """
    delta_E = E_slab - E_bulk  # noqa: N806
    return delta_E * EV_TO_J / (2 * area * ANGSTROM_TO_M**2)
