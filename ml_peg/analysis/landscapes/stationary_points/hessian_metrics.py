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
        The overlap, or NaN where the model has no imaginary mode.
    """
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
        means 6.5% softer than DFT.
    """
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
    """
    reference = np.asarray(atoms.info["reference_eigenvalues"], dtype=float)
    model = np.asarray(atoms.info["model_eigenvalues"], dtype=float)
    positive = reference > 0
    reference, model = reference[positive], model[positive]
    if reference.size < 3:
        return np.nan
    softest = reference < np.percentile(reference, 33)
    return float(np.median(np.abs(model - reference)[softest]))


def reaction_curvature(atoms: Atoms) -> float:
    """
    Get how steeply the model falls away from the barrier top, against DFT.

    The imaginary eigenvalue sets the curvature along the reaction coordinate,
    which fixes the tunnelling correction and the transition state theory
    prefactor. A ratio is sound here where it is not for the soft modes: the
    reference imaginary eigenvalue is never near zero, ranging from -0.08 to
    -9.18 eV/A^2 across this reference.

    NaN unless both are negative. Where the model has no imaginary mode at the
    DFT geometry there is no curvature to compare, and that is not a small
    disagreement but a statement that the model does not see a barrier top
    here at all -- counted by ``Saddles converged``, which agrees with this
    condition on 65 of 67 structures.

    Parameters
    ----------
    atoms
        A structure carrying paired reference and model eigenvalues.

    Returns
    -------
    float
        Model imaginary eigenvalue over the reference one, 1 meaning
        agreement.
    """
    if atoms.info["kind"] != "transition_state":
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
