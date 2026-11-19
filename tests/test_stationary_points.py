"""Unit tests for the Landscape17 stationary points Hessian metrics."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    ratio_agreement,
    stiffness_vs_dft,
)

# DFT eigenvalues spanning soft to stiff modes, in eV/Å^2.
REFERENCE_EIGENVALUES = (0.1, 1.0, 10.0)


def _structure(reference: list[float], model: list[float]) -> Atoms:
    """
    Build a structure carrying paired DFT and model eigenvalues.

    Parameters
    ----------
    reference
        DFT eigenvalues, in eV/Å^2.
    model
        The model's eigenvalues on the same modes, in eV/Å^2.

    Returns
    -------
    ase.atoms.Atoms
        A one-atom stand-in; only its info is read.
    """
    atoms = Atoms("H")
    atoms.info["reference_eigenvalues"] = list(reference)
    atoms.info["model_eigenvalues"] = list(model)
    return atoms


@pytest.mark.parametrize("ratio", [2.0, 0.5])
def test_too_sharp_scores_like_too_flat_by_the_same_factor(ratio: float) -> None:
    """
    A barrier top twice too sharp is as wrong as one twice too flat.

    Parameters
    ----------
    ratio
        Model curvature over DFT's: twice and half.
    """
    assert ratio_agreement(ratio) == pytest.approx(0.5)


def test_exact_agreement_scores_one() -> None:
    """A ratio of 1 is full agreement."""
    assert ratio_agreement(1.0) == pytest.approx(1.0)


def test_missing_ratio_stays_missing() -> None:
    """A saddle the model has no barrier at stays out of the summary."""
    assert np.isnan(ratio_agreement(np.nan))


@pytest.mark.parametrize(("factor", "percent"), [(0.9, -10.0), (1.1, 10.0)])
def test_stiffness_is_signed_percent(factor: float, percent: float) -> None:
    """
    A model uniformly 10% softer than DFT reads -10, and 10% stiffer +10.

    Parameters
    ----------
    factor
        The model's eigenvalues over DFT's, on every mode.
    percent
        The stiffness that factor should read as.
    """
    model = [factor * value for value in REFERENCE_EIGENVALUES]
    stiffness = stiffness_vs_dft(_structure(REFERENCE_EIGENVALUES, model))
    assert stiffness == pytest.approx(percent)


def test_stiffness_ignores_the_reaction_mode() -> None:
    """A saddle's negative mode has its own metric, so it does not count here."""
    reference = (-1.0, *REFERENCE_EIGENVALUES)
    model = (-5.0, *REFERENCE_EIGENVALUES)
    assert stiffness_vs_dft(_structure(reference, model)) == pytest.approx(0.0)
