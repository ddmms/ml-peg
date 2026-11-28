"""Pure AM26 energy/force metrics, without inference or file operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ml_peg.analysis.utils.utils import rmse

SYSTEM_LABELS = {
    "carbon": "a-C",
    "silicon": "a-Si",
    "silica": "a-SiO2",
    "gst": "a-GST",
    "lips": "a-LiPS",
}
SYSTEM_ORDER = tuple(SYSTEM_LABELS)


def _per_atom_energies(energies: ArrayLike, atom_counts: ArrayLike) -> np.ndarray:
    """Convert aligned total energies to energies per atom."""
    energies = np.asarray(energies, dtype=float)
    counts = np.asarray(atom_counts, dtype=float)
    if energies.ndim != 1 or counts.shape != energies.shape or not energies.size:
        raise ValueError("Energies and atom counts must be nonempty aligned vectors.")
    if not np.all(np.isfinite(counts) & (counts > 0) & (counts == np.floor(counts))):
        raise ValueError("Atom counts must be finite positive integers.")
    return energies / counts


def _flatten_forces(forces: Sequence[ArrayLike], atom_counts: ArrayLike) -> np.ndarray:
    """Concatenate every Cartesian component, retaining failed values."""
    if len(forces) != len(atom_counts):
        raise ValueError("One force array is required for every structure.")
    arrays = []
    for force, count in zip(forces, atom_counts, strict=True):
        array = np.asarray(force, dtype=float)
        if array.shape != (int(count), 3):
            raise ValueError("Each force array must have shape (atom_count, 3).")
        arrays.append(array.ravel())
    return np.concatenate(arrays)


def reference_normalisers(
    reference_energies: ArrayLike,
    atom_counts: ArrayLike,
    reference_forces: Sequence[ArrayLike],
) -> dict[str, float]:
    """
    Calculate one system's population reference standard deviations.

    Parameters
    ----------
    reference_energies
        DFT total energies in eV, one per structure.
    atom_counts
        Atom counts aligned with the energies and force arrays.
    reference_forces
        DFT force arrays in eV/angstrom, each of shape (atom_count, 3).

    Returns
    -------
    dict[str, float]
        Energy standard deviation in eV/atom and force-component standard
        deviation in eV/angstrom, keyed by ``energy`` and ``force``.
    """
    energies = _per_atom_energies(reference_energies, atom_counts)
    forces = _flatten_forces(reference_forces, atom_counts)
    if not np.isfinite(energies).all() or not np.isfinite(forces).all():
        raise ValueError("DFT reference energies and forces must be finite.")
    return {
        "energy": float(np.std(energies, ddof=0)),
        "force": float(np.std(forces, ddof=0)),
    }


def relrmse_from_rmse(absolute_rmse: float, reference_std: float) -> float:
    """
    Express an RMSE as a percentage of a reference standard deviation.

    Parameters
    ----------
    absolute_rmse
        Nonnegative RMSE in the same units as ``reference_std``. NaN propagates.
    reference_std
        Finite, strictly positive population reference standard deviation.

    Returns
    -------
    float
        Relative RMSE in percent, or NaN for a non-finite prediction RMSE.
    """
    if not np.isfinite(reference_std) or reference_std <= 0:
        raise ValueError("Reference standard deviation must be finite and positive.")
    if absolute_rmse < 0:
        raise ValueError("RMSE cannot be negative.")
    if not np.isfinite(absolute_rmse):
        return float("nan")
    return float(absolute_rmse / reference_std * 100)


def calculate_system_metrics(
    reference_energies: ArrayLike,
    predicted_energies: ArrayLike,
    atom_counts: ArrayLike,
    reference_forces: Sequence[ArrayLike],
    predicted_forces: Sequence[ArrayLike],
) -> dict[str, float]:
    """
    Calculate raw energy and component-wise force errors for one AM26 system.

    Parameters
    ----------
    reference_energies
        DFT total energies in eV, one per structure.
    predicted_energies
        Raw, unshifted predicted total energies in eV, aligned to the references.
    atom_counts
        Atom counts aligned with both energy vectors and force sequences.
    reference_forces
        DFT force arrays in eV/angstrom, each of shape (atom_count, 3).
    predicted_forces
        Corresponding predicted arrays; required failures must remain NaN.

    Returns
    -------
    dict[str, float]
        RMSEs (eV/atom and eV/angstrom), reference standard deviations in the
        same units, and Energy/Force RelRMSEs in percent. No offsets are fitted.
    """
    normalisers = reference_normalisers(
        reference_energies, atom_counts, reference_forces
    )
    energy_ref = _per_atom_energies(reference_energies, atom_counts)
    energy_pred = _per_atom_energies(predicted_energies, atom_counts)
    force_ref = _flatten_forces(reference_forces, atom_counts)
    force_pred = _flatten_forces(predicted_forces, atom_counts)

    # Preserve every required value. Convert infinities into failures before
    # calling the shared RMSE utility, which already propagates NaNs.
    energy_pred = np.where(np.isfinite(energy_pred), energy_pred, np.nan)
    force_pred = np.where(np.isfinite(force_pred), force_pred, np.nan)
    energy_rmse = float(rmse(energy_ref.tolist(), energy_pred.tolist()))
    force_rmse = float(rmse(force_ref.tolist(), force_pred.tolist()))
    return {
        "energy_rmse": energy_rmse,
        "force_rmse": force_rmse,
        "energy_std": normalisers["energy"],
        "force_std": normalisers["force"],
        "Energy RelRMSE": relrmse_from_rmse(energy_rmse, normalisers["energy"]),
        "Force RelRMSE": relrmse_from_rmse(force_rmse, normalisers["force"]),
    }


def mean_system_metrics(
    system_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """
    Average the five systems equally, propagating every required NaN.

    Parameters
    ----------
    system_metrics
        Mapping keyed by all five stored system names, containing the
        ``Energy RelRMSE`` and ``Force RelRMSE`` percentages for each system.

    Returns
    -------
    dict[str, float]
        The two overall Mean Energy/Force RelRMSE percentages.
    """
    if set(system_metrics) != set(SYSTEM_ORDER):
        raise ValueError("AM26 metrics require exactly the five named systems.")
    return {
        f"Mean {metric}": float(
            np.mean([system_metrics[system][metric] for system in SYSTEM_ORDER])
        )
        for metric in ("Energy RelRMSE", "Force RelRMSE")
    }


def calculate_am26_metrics(
    systems: Sequence[str],
    reference_energies: ArrayLike,
    predicted_energies: ArrayLike,
    atom_counts: ArrayLike,
    reference_forces: Sequence[ArrayLike],
    predicted_forces: Sequence[ArrayLike],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """
    Group aligned structures and calculate the five-system AM26 metrics.

    Parameters
    ----------
    systems
        Stored system name for each structure, in the input data order.
    reference_energies
        DFT total energies in eV, aligned with ``systems``.
    predicted_energies
        Raw predicted total energies in eV in the same order.
    atom_counts
        Atom counts for the structures in the same order.
    reference_forces
        DFT force arrays in eV/angstrom, each of shape (atom_count, 3).
    predicted_forces
        Aligned predicted force arrays, including NaNs for required failures.

    Returns
    -------
    tuple[dict[str, dict[str, float]], dict[str, float]]
        System metrics in fixed AM26 order, followed by the two overall means.
        This function does not infer frame alignment or discard failed values.
    """
    if set(systems) != set(SYSTEM_ORDER):
        raise ValueError("AM26 inputs require exactly the five named systems.")
    if any(
        len(values) != len(systems)
        for values in (
            reference_energies,
            predicted_energies,
            atom_counts,
            reference_forces,
            predicted_forces,
        )
    ):
        raise ValueError("All inputs must contain one entry per structure.")

    reference_energies = np.asarray(reference_energies, dtype=float)
    predicted_energies = np.asarray(predicted_energies, dtype=float)
    atom_counts = np.asarray(atom_counts)
    results = {}
    for system in SYSTEM_ORDER:
        indices = [i for i, value in enumerate(systems) if value == system]
        results[system] = calculate_system_metrics(
            reference_energies[indices],
            predicted_energies[indices],
            atom_counts[indices],
            [reference_forces[i] for i in indices],
            [predicted_forces[i] for i in indices],
        )
    return results, mean_system_metrics(results)
