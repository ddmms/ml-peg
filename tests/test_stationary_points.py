"""Unit tests for the Landscape17 stationary points Hessian metrics."""

from __future__ import annotations

from collections.abc import Callable

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    compare_hessians,
    hessian_failed,
    internal_basis,
    match_modes,
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


def test_tied_softest_modes_still_give_a_soft_mode_error() -> None:
    """
    Ties at the bottom must not empty the softest third.

    With the four softest modes tied, the 33rd percentile falls on their value,
    so a strict comparison selects nothing.
    """
    reference = (1.0, 1.0, 1.0, 1.0, 2.0, 3.0)
    model = tuple(value + 0.1 for value in reference)
    assert soft_mode_error(_structure(reference, model)) == pytest.approx(0.1)


def _tetrahedron() -> Atoms:
    """
    Build four atoms at the corners of a tetrahedron, with unequal masses.

    Returns
    -------
    ase.atoms.Atoms
        A non-linear structure with no symmetry-related zero rotations.
    """
    corners = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    atoms = Atoms("HCNO", positions=0.8 * corners + [0.1, -0.2, 0.3])
    atoms.info["kind"] = "minimum"
    return atoms


def _random_hessian(size: int) -> np.ndarray:
    """
    Build a random symmetric positive definite matrix.

    Parameters
    ----------
    size
        Matrix dimension.

    Returns
    -------
    numpy.ndarray
        A stand-in Hessian, in eV/Å^2.
    """
    random = np.random.default_rng(1).normal(size=(size, size))
    return random @ random.T + size * np.eye(size)


def test_internal_basis_excludes_rigid_body_motion() -> None:
    """Internal motion: 3N-6 orthonormal directions, none moving the whole molecule."""
    atoms = _tetrahedron()
    basis = internal_basis(atoms)
    assert basis.shape == (12, 6)
    assert np.allclose(basis.T @ basis, np.eye(6))
    centred = atoms.get_positions() - atoms.get_center_of_mass()
    for axis in np.eye(3):
        translation = np.tile(axis, 4)
        rotation = np.cross(axis, centred).ravel()
        assert np.allclose(basis.T @ translation, 0.0, atol=1e-10)
        assert np.allclose(basis.T @ rotation, 0.0, atol=1e-10)


def test_match_modes_recovers_shuffled_flipped_modes() -> None:
    """Each reference mode pairs with the model mode pointing its way, sign aside."""
    reference, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))
    order = [2, 0, 1, 5, 3, 4]
    overlap, partner = match_modes(reference, -reference[:, order])
    assert np.allclose(overlap, 1.0)
    assert list(partner) == [order.index(i) for i in range(6)]


def test_compare_hessians_of_identical_hessians() -> None:
    """The same Hessian twice gives full overlap and the same eigenvalues."""
    atoms = _tetrahedron()
    hessian = _random_hessian(12)
    comparison = compare_hessians(hessian, hessian, atoms)
    assert comparison["mode_overlap_median"] == pytest.approx(1.0)
    assert np.allclose(
        comparison["model_eigenvalues"], comparison["reference_eigenvalues"]
    )
    assert np.isnan(comparison["reaction_coordinate_overlap"])
    atoms.info["kind"] = "transition_state"
    comparison = compare_hessians(hessian, hessian, atoms)
    assert comparison["reaction_coordinate_overlap"] == pytest.approx(1.0)


def test_compare_hessians_feeds_the_stiffness() -> None:
    """A model Hessian 6% softer everywhere reads -6% stiffness."""
    atoms = _tetrahedron()
    hessian = _random_hessian(12)
    atoms.info.update(compare_hessians(hessian, 0.94 * hessian, atoms))
    assert not hessian_failed(atoms)
    assert stiffness_vs_dft(atoms) == pytest.approx(-6.0)


def test_compare_hessians_pairs_eigenvalues_by_mode() -> None:
    """A model that swaps two modes' stiffness is compared mode by mode, not sorted."""
    atoms = _tetrahedron()
    basis = internal_basis(atoms)
    modes, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(6, 6)))
    stiffness = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    swapped = stiffness[[0, 1, 4, 3, 2, 5]]
    reference = basis @ modes @ np.diag(stiffness) @ modes.T @ basis.T
    model = basis @ modes @ np.diag(swapped) @ modes.T @ basis.T
    comparison = compare_hessians(reference, model, atoms)
    assert np.allclose(comparison["reference_eigenvalues"], stiffness)
    assert np.allclose(comparison["model_eigenvalues"], swapped)
