"""Unit tests for the Landscape17 stationary points Hessian metrics."""

from __future__ import annotations

from collections.abc import Callable

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    hessian_failed,
    ratio_agreement,
    reaction_curvature,
    reaction_overlap,
    soft_mode_error,
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


def _failed_structure() -> Atoms:
    """
    Build a saddle as the calculation records one whose model Hessian failed.

    Labels and NaN overlaps with no eigenvalues at all, as written by
    ``calc_stationary_points.py``.

    Returns
    -------
    ase.atoms.Atoms
        A one-atom stand-in; only its info is read.
    """
    failed = Atoms("H")
    failed.info.update(
        molecule="ethanol",
        kind="transition_state",
        label="ts0_1_0",
        mode_overlap_median=np.nan,
        reaction_coordinate_overlap=np.nan,
    )
    return failed


def test_only_a_structure_without_eigenvalues_counts_as_failed() -> None:
    """A failed record is flagged, and one carrying eigenvalues is not."""
    assert hessian_failed(_failed_structure())
    assert not hessian_failed(_structure(REFERENCE_EIGENVALUES, REFERENCE_EIGENVALUES))


@pytest.mark.parametrize(
    "metric", [stiffness_vs_dft, soft_mode_error, reaction_curvature, reaction_overlap]
)
def test_failed_hessian_gives_nan_not_an_error(
    metric: Callable[[Atoms], float],
) -> None:
    """
    A failed Hessian gives a missing value rather than stopping the analysis.

    Parameters
    ----------
    metric
        One of the per-structure Hessian metrics.
    """
    assert np.isnan(metric(_failed_structure()))
