"""Thermal property utilities — Grüneisen parameter and lattice thermal conductivity."""

from __future__ import annotations

from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import _e, _hbar, _k, kB, kJ, mol
import numpy as np
from phonopy import PhonopyGruneisen
from phonopy.api_phonopy import Phonopy

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import (
    _ase_atoms_to_phonopy,
    get_fc2_and_freqs,
)

# Slack (1973) empirical constant: M in amu, θ_D in K, δ in Å → κ in W/(m·K)
_SLACK_A = 2.43e-6

# Conversion for phonopy's thermal-properties convention (kJ/mol per cell).
EV_TO_KJMOL = mol / kJ


def _build_strained_phonopy(
    atoms: Atoms,
    phonons: Phonopy,
    volumetric_strain: float,
    displacement_distance: float,
    symmetrize_fc2: bool,
    calculator: Calculator,
) -> Phonopy:
    """
    Strain a unit cell, compute force constants, and return a Phonopy object.

    Parameters
    ----------
    atoms
        Equilibrium ASE atoms (unit cell).
    phonons
        Equilibrium Phonopy object providing supercell matrix and primitive matrix.
    volumetric_strain
        Signed volumetric strain ``dV/V``. Linear cell scaling is
        ``(1 + volumetric_strain)^(1/3)``.
    displacement_distance
        Phonon displacement distance (Å).
    symmetrize_fc2
        Whether to symmetrize force constants.
    calculator
        ASE calculator for force evaluation.

    Returns
    -------
    Phonopy
        Strained Phonopy object with force constants computed.
    """
    scale = (1.0 + volumetric_strain) ** (1.0 / 3.0)
    atoms_strained = atoms.copy()
    atoms_strained.set_cell(atoms.cell * scale, scale_atoms=True)

    ph = _ase_atoms_to_phonopy(
        atoms_strained,
        phonons.supercell_matrix,
        primitive_matrix=phonons.primitive_matrix,
    )
    ph.generate_displacements(distance=displacement_distance, is_plusminus=True)
    ph, _, _ = get_fc2_and_freqs(ph, calculator, symmetrize_fc2=symmetrize_fc2)
    return ph


def compute_gruneisen(
    phonons: Phonopy,
    atoms: Atoms,
    calculator: Calculator,
    q_mesh: np.ndarray,
    delta_strain: float = 0.01,
    displacement_distance: float = 0.01,
    symmetrize_fc2: bool = True,
) -> dict[str, Any]:
    """
    Compute mode Grüneisen parameters and their weighted mean on a q-mesh.

    Three sets of force constants are computed: one at ``V + dV``, one at
    ``V - dV`` (where ``dV/V = delta_strain``), plus the equilibrium phonopy
    object passed in (which must already have force constants).

    Parameters
    ----------
    phonons
        Equilibrium Phonopy object with force constants already computed.
    atoms
        Equilibrium ASE unit-cell atoms.
    calculator
        ASE calculator used to evaluate forces on strained supercells.
    q_mesh
        Sampling mesh shape ``(3,)``, e.g. ``[20, 20, 20]``.
    delta_strain
        Volumetric strain ``dV/V`` applied to the ±V cells.
    displacement_distance
        Phonon displacement distance (Å) used for strained-cell calculations.
    symmetrize_fc2
        Whether to symmetrize force constants of strained cells.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``mode_gammas``: mode Grüneisen parameters, shape ``(nq, n_bands)``
        - ``mean_gamma``: weighted mean over positive-frequency modes
        - ``weights``: q-point weights, shape ``(nq,)``
        - ``frequencies``: mesh frequencies (THz), shape ``(nq, n_bands)``
    """
    ph_plus = _build_strained_phonopy(
        atoms, phonons, +delta_strain, displacement_distance, symmetrize_fc2, calculator
    )
    ph_minus = _build_strained_phonopy(
        atoms, phonons, -delta_strain, displacement_distance, symmetrize_fc2, calculator
    )

    grun = PhonopyGruneisen(phonons, ph_plus, ph_minus, delta_strain=delta_strain)
    grun.set_mesh(q_mesh)
    _qpts, weights, frequencies, _evecs, gammas = grun.get_mesh()

    weights = np.asarray(weights, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    mask = frequencies > 0
    w_tiled = np.broadcast_to(weights[:, None], frequencies.shape)
    total_weight = float(np.sum(w_tiled[mask]))
    if total_weight > 0:
        mean_gamma = float(np.sum(w_tiled[mask] * gammas[mask]) / total_weight)
    else:
        mean_gamma = 0.0

    return {
        "mode_gammas": gammas,
        "mean_gamma": mean_gamma,
        "weights": weights,
        "frequencies": frequencies,
    }


def harmonic_free_energy(
    frequencies_thz: np.ndarray,
    weights: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """
    Harmonic Helmholtz free energy from phonon frequencies and q-point weights.

    ``F(T) = Σ_q Σ_j w_q [ ħω_qj / 2 + k_B T ln(1 − exp(−ħω_qj / k_B T)) ]``

    Non-positive (imaginary) modes are excluded from the sum.

    Parameters
    ----------
    frequencies_thz
        Phonon frequencies in THz, shape ``(nq, n_bands)``.
    weights
        Q-point weights, shape ``(nq,)``. For a per-cell free energy the
        weights should sum to 1.
    temperatures
        Temperatures in K, shape ``(nt,)``.

    Returns
    -------
    np.ndarray
        Free energies in eV (per cell), shape ``(nt,)``.
    """
    freqs = np.asarray(frequencies_thz, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float)[:, None], freqs.shape).ravel()
    freqs = freqs.ravel()

    mask = freqs > 0
    freqs, w = freqs[mask], w[mask]
    # Mode energies ħω in eV, with ω = 2π f
    e_modes = _hbar * 2.0 * np.pi * freqs * 1e12 / _e

    zpe = 0.5 * np.sum(w * e_modes)
    free_energy = np.empty(len(np.atleast_1d(temperatures)), dtype=float)
    for i, temp in enumerate(np.atleast_1d(temperatures)):
        if temp <= 0:
            free_energy[i] = zpe
            continue
        thermal = kB * temp * np.log(1.0 - np.exp(-e_modes / (kB * temp)))
        free_energy[i] = zpe + np.sum(w * thermal)
    return free_energy


def gaussian_dos(
    frequencies: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Gaussian-broadened phonon DOS on a frequency grid.

    Parameters
    ----------
    frequencies
        Phonon frequencies (THz), shape ``(nq, n_bands)``.
    weights
        Q-point weights, shape ``(nq,)``.
    grid
        Frequency grid (THz) to evaluate the DOS on.
    sigma
        Gaussian broadening (THz).

    Returns
    -------
    np.ndarray
        DOS values on ``grid``.
    """
    freqs = np.asarray(frequencies, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float)[:, None], freqs.shape).ravel()
    freqs = freqs.ravel()
    diff = freqs[:, None] - np.asarray(grid, dtype=float)[None, :]
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return norm * np.sum(w[:, None] * np.exp(-0.5 * (diff / sigma) ** 2), axis=0)


def debye_temperature_from_max_freq(phonons: Phonopy, q_mesh: np.ndarray) -> float:
    """
    Estimate the Debye temperature from the maximum mesh frequency.

    Uses ``θ_D = ħ ω_max / k_B`` where ``ω_max = 2π f_max``.

    Parameters
    ----------
    phonons
        Phonopy object with force constants computed.
    q_mesh
        Sampling mesh shape ``(3,)``.

    Returns
    -------
    float
        Debye temperature in Kelvin.
    """
    phonons.run_mesh(q_mesh)
    freqs_thz = phonons.get_mesh_dict()["frequencies"]
    f_max = float(np.max(freqs_thz))
    omega_max = 2.0 * np.pi * f_max * 1e12
    return float(_hbar * omega_max / _k)


def slack_thermal_conductivity(
    mean_gamma: float,
    debye_temperature: float,
    n_atoms_primitive: int,
    volume_ang3: float,
    masses_amu: np.ndarray,
    temperature: float = 300.0,
) -> float:
    """
    Estimate lattice thermal conductivity via the Slack formula.

    ``κ = A · M_avg · θ_D³ · δ / (γ² · n^(2/3) · T)``

    where ``A = 2.43×10⁻⁶``, ``M_avg`` is the mean atomic mass in amu,
    ``θ_D`` is the Debye temperature in K, ``δ = (V/n)^(1/3)`` in Å,
    ``γ`` is the mean Grüneisen parameter, ``n`` is the number of atoms
    in the primitive cell, and ``T`` is temperature in K.

    Reference: Slack (1973), *J. Phys. Chem. Solids*, 34, 321–335.

    Parameters
    ----------
    mean_gamma
        Weighted-mean Grüneisen parameter (dimensionless).
    debye_temperature
        Debye temperature (K).
    n_atoms_primitive
        Number of atoms in the primitive cell.
    volume_ang3
        Unit-cell volume in Å³ (must correspond to the primitive cell).
    masses_amu
        Atomic masses of all atoms in the unit cell (amu).
    temperature
        Temperature in K.

    Returns
    -------
    float
        Lattice thermal conductivity in W/(m·K).
    """
    m_avg = float(np.mean(masses_amu))
    delta = (volume_ang3 / n_atoms_primitive) ** (1.0 / 3.0)
    kappa = (
        _SLACK_A
        * m_avg
        * debye_temperature**3
        * delta
        / (mean_gamma**2 * n_atoms_primitive ** (2.0 / 3.0) * temperature)
    )
    return float(kappa)


def compute_thermal_properties(
    phonons: Phonopy,
    atoms: Atoms,
    calculator: Calculator,
    q_mesh: np.ndarray,
    *,
    n_atoms_primitive: int | None = None,
    delta_strain: float = 0.01,
    displacement_distance: float = 0.01,
    symmetrize_fc2: bool = True,
    temperature: float = 300.0,
) -> dict[str, Any]:
    """
    Compute Grüneisen parameter and lattice thermal conductivity.

    This is a convenience wrapper that calls :func:`compute_gruneisen`,
    :func:`debye_temperature_from_max_freq`, and
    :func:`slack_thermal_conductivity` in sequence.

    Parameters
    ----------
    phonons
        Equilibrium Phonopy object with force constants already computed.
    atoms
        Equilibrium ASE unit-cell atoms.
    calculator
        ASE calculator used to evaluate forces on strained supercells.
    q_mesh
        Sampling mesh shape ``(3,)``, e.g. ``[20, 20, 20]``.
    n_atoms_primitive
        Number of atoms in the primitive cell. If ``None``, taken from
        ``phonons.primitive``.
    delta_strain
        Volumetric strain ``dV/V`` used for Grüneisen calculation.
    displacement_distance
        Phonon displacement distance (Å) for strained-cell force constants.
    symmetrize_fc2
        Whether to symmetrize strained-cell force constants.
    temperature
        Temperature in K for the Slack thermal conductivity estimate.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``mean_gamma``: weighted-mean Grüneisen parameter
        - ``debye_temperature_K``: Debye temperature in K
        - ``kappa_W_per_mK``: lattice thermal conductivity in W/(m·K)
        - ``mode_gammas``: per-mode Grüneisen parameters, shape ``(nq, n_bands)``
        - ``frequencies``: mesh frequencies (THz), shape ``(nq, n_bands)``
    """
    grun_dict = compute_gruneisen(
        phonons=phonons,
        atoms=atoms,
        calculator=calculator,
        q_mesh=q_mesh,
        delta_strain=delta_strain,
        displacement_distance=displacement_distance,
        symmetrize_fc2=symmetrize_fc2,
    )

    # The Grüneisen step already evaluated the equilibrium mesh; reuse its
    # frequencies rather than re-running the full mesh diagonalisation.
    f_max = float(np.max(grun_dict["frequencies"]))
    theta_d = float(_hbar * 2.0 * np.pi * f_max * 1e12 / _k)

    if n_atoms_primitive is None:
        n_atoms_primitive = len(phonons.primitive)

    masses_amu = np.asarray(atoms.get_masses(), dtype=float)
    # Use the primitive cell volume so that δ = (V_prim/n_prim)^(1/3) is correct
    # regardless of whether atoms is a conventional or primitive cell.
    volume_ang3 = float(abs(np.linalg.det(np.array(phonons.primitive.cell))))

    kappa = slack_thermal_conductivity(
        mean_gamma=grun_dict["mean_gamma"],
        debye_temperature=theta_d,
        n_atoms_primitive=n_atoms_primitive,
        volume_ang3=volume_ang3,
        masses_amu=masses_amu,
        temperature=temperature,
    )

    return {
        "mean_gamma": grun_dict["mean_gamma"],
        "debye_temperature_K": theta_d,
        "kappa_W_per_mK": kappa,
        "mode_gammas": grun_dict["mode_gammas"],
        "frequencies": grun_dict["frequencies"],
    }
