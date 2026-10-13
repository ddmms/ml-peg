"""Tests for the brittle_crack benchmark metric and tools."""

from __future__ import annotations

import numpy as np
import pytest

from ml_peg.analysis.fracture.brittle_crack import s_curves as sc

# crackcell (ml-peg-hackathon a2e0f46): errors(curve(1.03, 1.1), curve(), "A'")
EXPECTED_EK = 0.044970452637459225
EXPECTED_EE = 0.3508842049166176


def _curve(scale_k: float = 1.0, scale_e: float = 1.0, n: int = 120) -> sc.SCurve:
    """
    Cubic S-curve with folds, as in crackcell's metric tests.

    Parameters
    ----------
    scale_k
        Scale of k.
    scale_e
        Scale of the energy.
    n
        Number of points.

    Returns
    -------
    SCurve
        The curve.
    """
    b = np.linspace(2.6, 4.8, n)
    u = b - 2.8
    k = (1 + 0.4 * u - 1.2 * u**2 + 0.8 * u**3) * scale_k
    e = scale_e * (2.0 * k**2 - 0.3 * np.tanh(3 * (u - 0.5)))
    return sc.SCurve(k, 0.4 * k, b, e, 0.4, 5.4686, 1.511, 3.866)


def test_identical_curves_have_zero_errors_and_full_coverage():
    """Identical model and reference give zero K and E errors."""
    e = sc.errors(_curve(), _curve())
    assert e["E_K"] == pytest.approx(0, abs=1e-12)
    assert e["E_E"] == pytest.approx(0, abs=1e-12)
    assert e["coverage"] == 1.0


def test_constant_k_shift_is_the_k_error():
    """A constant shift in k is the K error."""
    m = _curve()
    m.k = m.k + 0.05
    assert sc.errors(m, _curve())["E_K"] == pytest.approx(0.05, rel=1e-6)


def test_elastic_term_uses_the_reference_c_tilde():
    """An extra elastic energy in the model counts as energy error."""
    m = _curve()
    m.E = m.E + 0.7 * m.k**2
    assert sc.errors(m, _curve())["E_E"] > 0.05


def test_a_model_outside_the_window_has_zero_coverage():
    """A model curve entirely outside the reference window is uncovered."""
    m = _curve()
    m.b = m.b + 10.0
    assert sc.errors(m, _curve())["coverage"] == 0.0


def test_padding_scales_the_score_by_coverage():
    """Padding the uncovered fraction at b gives c times the score."""
    g, b = 0.1, 0.3
    for x in (0.0, 0.2, 0.5):
        for c in (0.0, 0.4, 1.0):
            assert sc.normalised(sc.padded(x, c, g, b), g, b) == pytest.approx(
                c * sc.normalised(x, g, b)
            )


def test_the_port_reproduces_crackcell():
    """The ported errors equal crackcell's on the same curves."""
    e = sc.errors(_curve(scale_k=1.03, scale_e=1.1), _curve())
    assert e["E_K"] == pytest.approx(EXPECTED_EK, rel=1e-9)
    assert e["E_E"] == pytest.approx(EXPECTED_EE, rel=1e-9)
