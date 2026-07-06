"""Tests for shared phonon utilities used by the phonon benchmarks."""

from __future__ import annotations

import numpy as np

from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import (
    EV_TO_KJMOL,
    slack_thermal_conductivity,
)


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
