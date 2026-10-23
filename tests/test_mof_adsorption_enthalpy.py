"""Tests for Widom estimates in the MOF adsorption enthalpy benchmark."""

from __future__ import annotations

from ase import units
import numpy as np
import pytest

from ml_peg.calcs.porous_materials.mof_adsorption_enthalpy.widom_utils import (
    EV_TO_KJ_MOL,
    OVERLAP_ENERGY,
    R_GAS,
    scored_frameworks,
    widom_estimates,
)

T = 298.0
DENSITY = 1000.0
KT = units.kB * T


def test_single_energy():
    """All insertions at one energy give that energy minus kT."""
    results = widom_estimates(np.full(100, -0.3), T, DENSITY)
    assert results["enthalpy"] == pytest.approx((-0.3 - KT) * EV_TO_KJ_MOL)
    assert results["henry_coefficient"] == pytest.approx(
        np.exp(0.3 / KT) / (DENSITY * R_GAS * T)
    )


def test_two_energies():
    """Enthalpy is the Boltzmann-weighted mean energy."""
    energies = np.array([-0.2, -0.4])
    weights = np.exp(-energies / KT)
    expected = (np.sum(energies * weights) / np.sum(weights) - KT) * EV_TO_KJ_MOL
    assert widom_estimates(energies, T, DENSITY)["enthalpy"] == pytest.approx(expected)


def test_overlaps():
    """Overlaps lower the Henry coefficient but not the enthalpy."""
    clean = widom_estimates(np.full(10, -0.3), T, DENSITY)
    energies = np.concatenate([np.full(10, -0.3), np.full(10, OVERLAP_ENERGY)])
    results = widom_estimates(energies, T, DENSITY)

    assert results["enthalpy"] == pytest.approx(clean["enthalpy"])
    assert results["henry_coefficient"] == pytest.approx(clean["henry_coefficient"] / 2)
    assert results["overlap_fraction"] == pytest.approx(0.5)
    assert results["unphysical_fraction"] == 0.0


def test_energy_floor():
    """Energies below the floor are excluded and counted as unphysical."""
    energies = np.array([-0.3, -0.3, -0.3, -5.0])
    results = widom_estimates(energies, T, DENSITY)

    assert results["enthalpy"] == pytest.approx((-0.3 - KT) * EV_TO_KJ_MOL)
    assert results["unphysical_fraction"] == pytest.approx(0.25)


def test_no_valid_insertions():
    """No valid insertions give NaN estimates."""
    results = widom_estimates(np.array([OVERLAP_ENERGY, np.nan, -5.0]), T, DENSITY)
    assert np.isnan(results["enthalpy"])
    assert np.isnan(results["henry_coefficient"])
    assert results["unphysical_fraction"] == pytest.approx(1.0)


def test_large_negative_energies():
    """Strong binding does not overflow."""
    results = widom_estimates(np.array([-1.2, -1.1]), T, DENSITY)
    assert np.isfinite(results["enthalpy"])
    assert np.isfinite(results["henry_coefficient"])


def test_scored_frameworks():
    """Only frameworks with a reference enthalpy are scored."""
    reference = {
        "frameworks": {
            "B": {"enthalpy": -20.0},
            "A": {"enthalpy": -30.0},
            "C": {"enthalpy": None},
        }
    }
    assert scored_frameworks(reference) == ["A", "B"]
