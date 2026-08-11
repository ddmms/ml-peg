"""Regression checks for the calculation of thermodynamic quantities."""

from __future__ import annotations

from ase import units
import numpy as np
import pytest

from ml_peg.analysis.utils.thermodynamics import (
    ANG3_PER_EV_TO_GPA_INV,
    EV_TO_J_MOL,
    EV_TO_KJ_MOL,
    density,
    enthalpy_of_vaporization,
    heat_capacity_cp,
    isothermal_compressibility,
    thermal_expansion,
)


def test_density():
    """Test density calculation."""
    values = np.array([0.8, 1.0, 1.0, 1.2, 1.2, 1.4, 1.4, 1.6])
    mean, stderr = density(values, block_size=2)
    block_values = np.array([0.9, 1.1, 1.3, 1.5])
    assert mean == pytest.approx(np.mean(block_values))
    assert stderr == pytest.approx(
        np.std(block_values, ddof=1) / np.sqrt(len(block_values))
    )


def test_heat_capacity_cp():
    """Test constant pressure heat capacity calculation."""
    enthalpy = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0])
    temperature, n_molecules = 300.0, 10

    cp, stderr = heat_capacity_cp(
        enthalpy, temperature=temperature, n_molecules=n_molecules, block_size=2
    )

    # Every block has the same population variance: 1 eV^2.
    expected = 1.0 / (units.kB * temperature**2) * EV_TO_J_MOL / n_molecules
    assert cp == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


def test_isothermal_compressibility():
    """Test isothermal compressibility calculation."""
    volume = np.array([9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0])
    temperature = 300.0
    kappa, stderr = isothermal_compressibility(
        volume, temperature=temperature, block_size=2
    )
    # variance = 1 Angstrom^6, mean volume = 10 Angstrom^3.
    mean_volume = 10.0
    expected = 1.0 / (units.kB * temperature * mean_volume) * ANG3_PER_EV_TO_GPA_INV
    assert kappa == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


def test_thermal_expansion():
    """Test thermal expansion coefficient calculation."""
    volume = np.array([9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0])
    enthalpy, temperature = 2.0 * volume, 300.0
    alpha, stderr = thermal_expansion(
        volume, enthalpy, temperature=temperature, block_size=2
    )
    mean_volume, mean_cov = 10.0, 2.0
    # cov(V, H) = cov(V, 2V) = 2 var(V) = 2.
    expected = mean_cov / (units.kB * temperature**2 * mean_volume)
    assert alpha == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)


def test_enthalpy_of_vaporization():
    """Test enthalpy of vaporization calculation."""
    liquid = np.array([-101.0, -99.0, -101.0, -99.0, -101.0, -99.0, -101.0, -99.0])

    gas = np.array([-4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0])
    temperature, n_molecules = 300.0, 10

    hvap, stderr = enthalpy_of_vaporization(
        liquid, gas, temperature=temperature, n_molecules=n_molecules, block_size=2
    )

    # Each liquid block has mean -100 and every gas block mean -4,
    expected = (-4.0 - (-100.0 / 10) + units.kB * temperature) * EV_TO_KJ_MOL
    assert hvap == pytest.approx(expected)
    assert stderr == pytest.approx(0.0)
