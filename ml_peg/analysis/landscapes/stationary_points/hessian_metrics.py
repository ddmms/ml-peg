"""
Per-structure Hessian metrics for the Landscape17 stationary points.

Each function compares the model's Hessian eigenvalues with the reference's at
one DFT structure and returns one number. They live apart from
``analyse_stationary_points.py`` because that module reads the calculation
outputs as soon as it is imported, and skips without them. These read nothing,
so their tests run anywhere, CI included.
"""

from __future__ import annotations

from ase import Atoms
import numpy as np
from scipy.optimize import linear_sum_assignment


def hessian_failed(atoms: Atoms) -> bool:
    """
    Get whether the calculation recorded the model's Hessian here as failed.

    ``calc_stationary_points.py`` writes such a structure with its labels and
    NaN overlaps but no eigenvalues. The metrics below return NaN for it rather
    than raising, and the analysis voids that model's Hessian metrics.

    Parameters
    ----------
    atoms
        A structure read from the Hessian outputs.

    Returns
    -------
    bool
        True if the structure carries no model eigenvalues.
    """
    return "model_eigenvalues" not in atoms.info


def reaction_overlap(atoms: Atoms) -> float:
    """
    Get the overlap with the DFT reaction coordinate, where the model has one.

    Counted only when the model's matched mode is also negative. Without that
    condition the overlap is taken against a mode the model considers uphill,
    which credits it for agreeing about a reaction it does not believe has a
    barrier: on MACE-OFF, ethanol ts0_1_0 overlaps 0.98 while the model's
    lowest eigenvalue there is +0.10. Those 32 of 67 structures scored 0.974
    against the 35 real ones at 0.989, so including them moved the metric
    barely at all while halving what it described.

    Parameters
    ----------
    atoms
        A structure carrying the overlap and the paired eigenvalues.

    Returns
    -------
    float
        The overlap, or NaN where the model has no imaginary mode or its
        Hessian failed.
    """
    if hessian_failed(atoms):
        return np.nan
    overlap = atoms.info.get("reaction_coordinate_overlap", np.nan)
    if not np.isfinite(overlap):
        return np.nan
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    return overlap if reference[0] < 0 and model[0] < 0 else np.nan


def stiffness_vs_dft(atoms: Atoms) -> float:
    """
    Get how much stiffer than DFT the model's curvature is overall, in percent.

    From one factor relating the model's eigenvalues to the reference's: the
    least-squares slope through the origin, which weights each mode by the
    square of its reference eigenvalue and so is a statement about the stiff
    modes: on this reference the eigenvalues span four orders of magnitude, and
    the ten stiffest modes of aspirin carry 76% of the fit.

    Weighting every mode equally instead was tried and is worse. The softest
    modes have reference eigenvalues near zero -- 1% of them below 0.05 eV/A^2
    -- so a small absolute error becomes an enormous ratio, and the resulting
    statistic is governed by where the cutoff is put rather than by the model:
    over the soft half it reads 1.12 with no floor and 1.00 with a floor of
    2 eV/A^2, while the stiff half holds at 0.959 to 0.953 throughout. The
    ratio-weighted version measures the cutoff; this one measures the model.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        The slope's distance from 1, in percent: 0 is agreement, and -6.5
        means 6.5% softer than DFT. NaN where the model's Hessian failed.
    """
    if hessian_failed(atoms):
        return np.nan
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    # Real modes only. At a saddle the negative eigenvalue is the reaction
    # coordinate, which has its own metric, and leaving it here would put one
    # mode in two. It makes no numerical difference -- the slope agrees to six
    # decimals either way, since its weight is 3e-5 of the stiffest mode's --
    # so this is about the metrics not overlapping, not about the number.
    real = reference > 0
    reference, model = reference[real], model[real]
    slope = np.dot(reference, model) / np.dot(reference, reference)
    return float(100.0 * (slope - 1.0))


def soft_mode_error(atoms: Atoms) -> float:
    """
    Get the model's curvature error on the softest modes, in absolute terms.

    Absolute rather than relative, because a ratio is unusable here. The
    softest modes have reference eigenvalues near zero -- paracetamol's
    lowest is 0.0026 eV/A^2 -- so an error of 0.18 eV/A^2, small beside the
    16 eV/A^2 the model is out on its stiffest mode, reads as a factor of 67.
    Averaging such ratios produces a number that tracks wherever the near-zero
    modes are cut off: 1.19 with no floor against 1.01 at 2 eV/A^2. The
    absolute error holds at 0.076, 0.074, 0.075 across the same range.

    The softest third is taken because that is where a landscape lives: the
    torsions that decide barrier heights and whether two minima are distinct,
    rather than the X-H stretches that dominate a least-squares slope.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        Median absolute eigenvalue error over the softest third, in eV/A^2.
        NaN where the model's Hessian failed.
    """
    if hessian_failed(atoms):
        return np.nan
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    positive = reference > 0
    reference, model = reference[positive], model[positive]
    if reference.size < 3:
        return np.nan
    cutoff = np.percentile(reference, 33)
    softest = reference < cutoff
    if not softest.any():
        # Ties at the bottom put the 33rd percentile on the lowest value, so
        # the strict comparison selects nothing; take the tied modes instead.
        softest = reference <= cutoff
    return float(np.median(np.abs(model - reference)[softest]))


def reaction_curvature(atoms: Atoms) -> float:
    """
    Get how steeply the model falls away from the barrier top, against DFT.

    The negative eigenvalue is the curvature along the reaction coordinate.
    The Hessians are Cartesian, not mass-weighted, so it is a proxy for the
    imaginary frequency that enters the tunnelling correction and the
    transition state theory prefactor, not that frequency itself. A ratio is
    usable here, unlike for the soft modes, because the
    reference imaginary eigenvalue is rarely near zero: it ranges from -0.0034
    to -9.18 eV/A^2 (median -0.16), and only one barrier top, paracetamol
    ts2_3_2, is almost flat. The median over saddles does not follow a single
    extreme ratio.

    NaN unless both are negative. Where the model has no imaginary mode at the
    DFT geometry there is no curvature to compare, and that is not a small
    disagreement but a statement that the model does not see a barrier top
    here at all -- counted by ``Reaction modes present``.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        Model imaginary eigenvalue over the reference one, 1 meaning
        agreement. NaN where the model's Hessian failed.
    """
    if atoms.info["kind"] != "transition_state" or hessian_failed(atoms):
        return np.nan
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    if reference[0] >= 0 or model[0] >= 0:
        return np.nan
    return float(model[0] / reference[0])


def ratio_agreement(ratio: float) -> float:
    """
    Get how close a model-over-DFT ratio is to 1, whichever way it errs.

    The fold ``min(r, 1/r)``, so a barrier top a factor of 2 too sharp and one
    a factor of 2 too flat both read 0.5. The raw ratio could not be scored
    directly: a score treats higher as better, which credited any curvature
    sharper than DFT's with full marks -- MACE-MP-0a, 2.3 times too sharp,
    scored 1.0. Applied to each structure before the median, so a model that
    errs both ways cannot average out to look right.

    Parameters
    ----------
    ratio
        Model value over the reference value, positive or NaN.

    Returns
    -------
    float
        Between 0 and 1, 1 meaning agreement; NaN where the ratio is.
    """
    return min(ratio, 1.0 / ratio) if np.isfinite(ratio) else np.nan


def internal_basis(atoms: Atoms) -> np.ndarray:
    """
    Build a basis spanning internal motion, excluding translation and rotation.

    Six modes sit at essentially zero curvature, and within that near-degenerate
    subspace the individual eigenvectors are arbitrary: comparing them one to
    one produces cosines near zero that look like total disagreement and mean
    nothing. Both Hessians are projected into this basis first, leaving 3N-6
    real modes. The same basis serves both because they share a geometry.

    Parameters
    ----------
    atoms
        The structure, whose masses place the centre of rotation.

    Returns
    -------
    numpy.ndarray
        Columns spanning internal motion.
    """
    positions = atoms.get_positions()
    shifted = positions - np.average(positions, axis=0, weights=atoms.get_masses())

    rigid = []
    for axis in range(3):
        translation = np.zeros_like(positions)
        translation[:, axis] = 1.0
        rigid.append(translation.flatten())
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        rigid.append(np.cross(unit, shifted).flatten())

    basis, upper = np.linalg.qr(np.array(rigid).T)
    independent = basis[:, np.abs(np.diag(upper)) > 1e-8]
    left, singular, _ = np.linalg.svd(independent, full_matrices=True)
    return left[:, (singular > 1e-8).sum() :]


def match_modes(
    reference: np.ndarray, model: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair every reference mode with a model mode, one to one, by overlap.

    Parameters
    ----------
    reference
        Reference eigenvectors as columns.
    model
        Model eigenvectors as columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The matched overlap for each reference mode, and the model mode index
        it was matched to.
    """
    overlap = np.abs(reference.T @ model)
    rows, columns = linear_sum_assignment(-overlap)
    matched = np.zeros(reference.shape[1])
    partner = np.zeros(reference.shape[1], dtype=int)
    matched[rows] = overlap[rows, columns]
    partner[rows] = columns
    return matched, partner


def compare_hessians(
    reference_hessian: np.ndarray, model_hessian: np.ndarray, atoms: Atoms
) -> dict:
    """
    Compare a model's Hessian with the reference one at the same structure.

    Both are projected onto internal motion and diagonalised, and each reference
    mode is paired with the model mode pointing most nearly the same way. The
    model's eigenvalues are reported in that pairing, so the per-structure
    metrics compare like with like.

    Parameters
    ----------
    reference_hessian
        The reference Cartesian Hessian, in eV/A^2.
    model_hessian
        The model's Cartesian Hessian at the same geometry, in eV/A^2.
    atoms
        The structure, with ``kind`` in its info.

    Returns
    -------
    dict
        The median and smallest matched overlaps, the reference eigenvalues and
        the model's paired with them, how many of each are negative, and the
        overlap of the reference's lowest mode with its match at a transition
        state (NaN at a minimum).
    """
    basis = internal_basis(atoms)
    ref_values, ref_vectors = np.linalg.eigh(basis.T @ reference_hessian @ basis)
    model_values, model_vectors = np.linalg.eigh(basis.T @ model_hessian @ basis)
    matched, partner = match_modes(ref_vectors, model_vectors)
    return {
        "mode_overlap_median": float(np.median(matched)),
        "mode_overlap_min": float(matched.min()),
        "reference_eigenvalues": ref_values,
        "model_eigenvalues": model_values[partner],
        "n_negative_reference": int((ref_values < 0).sum()),
        "n_negative_model": int((model_values < 0).sum()),
        "reaction_coordinate_overlap": (
            float(matched[0]) if atoms.info["kind"] == "transition_state" else np.nan
        ),
    }
