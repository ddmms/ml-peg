"""Force-based metrics for diatomic curves."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ml_peg.analysis.physicality.diatomics.metrics.energy import (
    _common_grid_curve_pair,
    _jump_magnitude,
    _validate_diatomic_curve,
)


def calc_force_mae(
    seps_ref: ArrayLike,
    f_ref: ArrayLike,
    seps_pred: ArrayLike,
    f_pred: ArrayLike,
    *,
    interpolate: bool | int = False,
) -> float:
    """
    Calculate force MAE, optionally interpolating over the shared range.

    Parameters
    ----------
    seps_ref
        Reference separations.
    f_ref
        Reference Cartesian forces.
    seps_pred
        Predicted separations.
    f_pred
        Predicted Cartesian forces.
    interpolate
        Whether or how many common-grid points to use.

    Returns
    -------
    float
        Mean absolute force error.
    """
    _, f_ref, f_pred = _common_grid_curve_pair(
        seps_ref,
        f_ref,
        seps_pred,
        f_pred,
        interpolate=interpolate,
        value_kind="force",
    )
    return float(np.mean(np.abs(f_ref - f_pred)))


def _radial_forces(seps: ArrayLike, forces: np.ndarray) -> np.ndarray:
    """
    Validate a force curve and return first-atom radial forces.

    Parameters
    ----------
    seps
        Sample separations.
    forces
        Cartesian forces for both atoms.

    Returns
    -------
    np.ndarray
        First-atom radial forces.
    """
    _, forces = _validate_diatomic_curve(seps, forces, value_kind="force")
    return forces[:, 0, 0]  # x-component of force on first atom


def calc_force_flips(
    seps: ArrayLike,
    forces: np.ndarray,
    threshold: float = 1e-2,  # 10meV/A threshold as in reference code
) -> float:
    """
    Count thresholded direction changes in the first atom's radial force.

    Parameters
    ----------
    seps
        Sample separations.
    forces
        Cartesian forces for both atoms.
    threshold
        Force magnitudes below this value are treated as zero.

    Returns
    -------
    float
        Number of radial-force direction changes.
    """
    radial_forces = _radial_forces(seps, forces).copy()
    radial_forces[np.abs(radial_forces) < threshold] = 0
    force_signs = np.sign(radial_forces[radial_forces != 0])
    return float(np.sum(np.diff(force_signs) != 0))


def calc_force_total_variation(
    seps: ArrayLike,
    forces: np.ndarray,
) -> float:
    """
    Calculate total variation in the first atom's radial force.

    Parameters
    ----------
    seps
        Sample separations.
    forces
        Cartesian forces for both atoms.

    Returns
    -------
    float
        Total radial-force variation.
    """
    return float(np.sum(np.abs(np.diff(_radial_forces(seps, forces)))))


def calc_force_jump(
    seps: ArrayLike,
    forces: np.ndarray,
) -> float:
    """
    Calculate total radial-force step magnitude around sign-flip points.

    Parameters
    ----------
    seps
        Sample separations.
    forces
        Cartesian forces for both atoms.

    Returns
    -------
    float
        Total radial-force step magnitude.
    """
    return _jump_magnitude(_radial_forces(seps, forces), threshold=0)
