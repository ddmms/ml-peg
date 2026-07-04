"""Tests for shared phonon utilities used by the phonon benchmarks."""

from __future__ import annotations

import numpy as np

from ml_peg.calcs.bulk_crystal.phonons.phonons_utils import qpath_distances
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import (
    EV_TO_KJMOL,
    gaussian_dos,
    harmonic_free_energy,
    slack_thermal_conductivity,
)


def test_qpath_distances():
    """Test cumulative q-path distances follow the phonopy convention."""
    cell = np.diag([2.0, 3.0, 4.0])
    qpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]])

    distances = qpath_distances(qpoints, cell)

    # Reciprocal lattice without 2π: b1 = 1/2, b2 = 1/3.
    expected = np.array([0.0, 0.5 / 2.0, 0.5 / 2.0 + 0.5 / 3.0])
    assert distances[0] == 0.0
    np.testing.assert_allclose(distances, expected)


def test_harmonic_free_energy():
    """Test zero-point limit, monotonic decrease, and imaginary-mode exclusion."""
    freqs = np.array([[5.0, 6.0, 7.0]])  # THz
    weights = np.array([1.0])
    temperatures = np.array([0.0, 300.0, 1000.0])

    free_energy = harmonic_free_energy(freqs, weights, temperatures)

    # F(0) is the zero-point energy: sum(h f / 2) with h in eV s.
    h_ev_s = 4.135667696e-15
    zpe = 0.5 * h_ev_s * (5.0 + 6.0 + 7.0) * 1e12
    np.testing.assert_allclose(free_energy[0], zpe, rtol=1e-6)
    assert np.all(np.diff(free_energy) < 0)

    # Imaginary (negative) modes are excluded from the sum.
    freqs_imag = np.array([[-2.0, 5.0, 6.0, 7.0]])
    free_energy_imag = harmonic_free_energy(freqs_imag, weights, temperatures)
    np.testing.assert_allclose(free_energy_imag, free_energy)


def test_gaussian_dos_normalisation():
    """Test the broadened DOS integrates to the number of modes."""
    freqs = np.array([[2.0, 3.0], [4.0, 5.0]])  # THz
    weights = np.array([0.5, 0.5])
    grid = np.linspace(0.0, 8.0, 2001)

    dos = gaussian_dos(freqs, weights, grid, sigma=0.1)

    # Weights sum to 1 over q-points, so the DOS integrates to n_bands.
    np.testing.assert_allclose(np.trapz(dos, grid), 2.0, rtol=1e-3)


def test_slack_thermal_conductivity_scaling():
    """Test Slack conductivity scaling with the Grüneisen parameter."""
    kwargs = {
        "debye_temperature": 1000.0,
        "n_atoms_primitive": 2,
        "volume_ang3": 10.0,
        "masses_amu": np.array([12.0, 12.0]),
        "temperature": 300.0,
    }
    kappa_1 = slack_thermal_conductivity(mean_gamma=1.0, **kwargs)
    kappa_2 = slack_thermal_conductivity(mean_gamma=2.0, **kwargs)

    assert kappa_1 > 0
    np.testing.assert_allclose(kappa_1 / kappa_2, 4.0)


def test_ev_to_kjmol():
    """Test the eV to kJ/mol conversion constant."""
    np.testing.assert_allclose(EV_TO_KJMOL, 96.485, rtol=1e-4)
