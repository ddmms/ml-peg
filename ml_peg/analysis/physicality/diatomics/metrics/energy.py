"""Energy-based metrics for diatomic curves."""

from __future__ import annotations

from typing import Literal

from ase.data import atomic_masses, atomic_numbers
import numpy as np
from numpy.typing import ArrayLike

PBE_WALL_ENERGY_THRESHOLDS_EV: tuple[float, ...] = (1, 5, 10, 20, 50, 100)


def _validate_diatomic_curve(
    separations: ArrayLike,
    values: ArrayLike,
    *,
    normalize_energy: bool = False,
    value_kind: Literal["energy", "force"] = "energy",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate, sort, and optionally far-field-normalize a sampled curve.

    Parameters
    ----------
    separations
        Sample separations.
    values
        Sampled energy or force values.
    normalize_energy
        Whether to shift the last energy sample to zero.
    value_kind
        Kind of sampled values, used for shape validation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted separations and values.
    """
    separation_array = np.asarray(separations)
    value_array = np.asarray(values)

    if separation_array.ndim != 1:
        raise ValueError(
            f"separations must have shape (n,), got {separation_array.shape}"
        )
    if value_kind == "energy" and value_array.ndim != 1:
        raise ValueError(f"energy values must have shape (n,), got {value_array.shape}")
    if value_kind == "force" and (
        value_array.ndim != 3 or value_array.shape[1:] != (2, 3)
    ):
        raise ValueError(
            f"force values must have shape (n, 2, 3), got {value_array.shape}"
        )

    if len(separation_array) != len(value_array):
        raise ValueError(
            f"len(separation_array)={len(separation_array)} != "
            f"len(value_array)={len(value_array)}"
        )
    if len(separation_array) < 2:
        raise ValueError(
            "Input must have at least 2 points, "
            f"got len(separation_array)={len(separation_array)}"
        )
    n_separation_nan = int(np.isnan(separation_array).sum())
    n_value_nan = int(np.isnan(value_array).sum())
    if n_separation_nan or n_value_nan:
        raise ValueError(
            "Input contains NaN values: "
            f"n_separation_nan={n_separation_nan}, n_value_nan={n_value_nan}"
        )
    n_separation_inf = int(np.isinf(separation_array).sum())
    n_value_inf = int(np.isinf(value_array).sum())
    if n_separation_inf or n_value_inf:
        raise ValueError(
            "Input contains infinite values: "
            f"n_separation_inf={n_separation_inf}, n_value_inf={n_value_inf}"
        )
    n_unique = len(np.unique(separation_array))
    if n_unique != len(separation_array):
        raise ValueError(
            f"separations contains {len(separation_array) - n_unique} duplicates"
        )

    sort_indices = np.argsort(separation_array)
    separation_array = separation_array[sort_indices]
    value_array = value_array[sort_indices]

    # Normalize energy curves to zero at the largest separation.
    if normalize_energy and value_array.ndim == 1:
        # The ascending sort places that sample last.
        value_array = value_array - value_array[-1]

    return separation_array, value_array


def _interpolation_point_count(interpolate: bool | int) -> int:
    """
    Return the requested interpolation size, validating a two-point minimum.

    Parameters
    ----------
    interpolate
        Whether or how many interpolation points to use.

    Returns
    -------
    int
        Number of interpolation points.
    """
    n_points = 100 if interpolate is True else int(interpolate)
    if n_points < 2:
        raise ValueError("interpolate must request at least 2 points")
    return n_points


def _common_grid_curve_pair(
    separations_ref: ArrayLike,
    values_ref: ArrayLike,
    separations_pred: ArrayLike,
    values_pred: ArrayLike,
    *,
    interpolate: bool | int,
    value_kind: Literal["energy", "force"] = "energy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate two curves and optionally interpolate their common interval.

    Parameters
    ----------
    separations_ref
        Reference-curve separations.
    values_ref
        Reference-curve values.
    separations_pred
        Predicted-curve separations.
    values_pred
        Predicted-curve values.
    interpolate
        Whether or how many common-grid points to use.
    value_kind
        Kind of sampled values, used for shape validation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Common separations, reference values, and predicted values.
    """
    separations_ref, values_ref = _validate_diatomic_curve(
        separations_ref, values_ref, value_kind=value_kind
    )
    separations_pred, values_pred = _validate_diatomic_curve(
        separations_pred, values_pred, value_kind=value_kind
    )
    if not interpolate:
        if not np.array_equal(separations_ref, separations_pred):
            raise ValueError(
                "Reference and predicted distances must be same when "
                f"interpolate={interpolate}\n"
                f"separations_ref={separations_ref}, "
                f"separations_pred={separations_pred}"
            )
        return separations_ref, values_ref, values_pred

    data_min = max(separations_ref.min(), separations_pred.min())
    data_max = min(separations_ref.max(), separations_pred.max())
    if data_min >= data_max:
        curve_label = "force curves" if value_kind == "force" else "curves"
        raise ValueError(
            f"Cannot interpolate {curve_label} with no overlap: "
            f"data_min={data_min}, data_max={data_max}"
        )
    common_grid = np.linspace(
        data_min, data_max, _interpolation_point_count(interpolate)
    )

    def interpolate_values(separations: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Interpolate all flattened value components onto ``common_grid``.

        Parameters
        ----------
        separations
            Source separations.
        values
            Source values.

        Returns
        -------
        np.ndarray
            Values interpolated onto the common grid.
        """
        flattened_values = values.reshape(len(values), -1)
        interpolated = np.column_stack(
            [
                np.interp(
                    common_grid,
                    separations,
                    flattened_values[:, component_index],
                )
                for component_index in range(flattened_values.shape[1])
            ]
        )
        return interpolated.reshape(len(common_grid), *values.shape[1:])

    return (
        common_grid,
        interpolate_values(separations_ref, values_ref),
        interpolate_values(separations_pred, values_pred),
    )


def _binding_energy(energies: np.ndarray) -> float:
    """
    Return well depth relative to the largest sampled separation.

    Parameters
    ----------
    energies
        Sampled energies ordered by separation.

    Returns
    -------
    float
        Binding energy.
    """
    return float(energies[-1] - np.min(energies))


def _validated_energy_pair(
    seps_ref: ArrayLike,
    energy_ref: ArrayLike,
    seps_pred: ArrayLike,
    energy_pred: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate two independently sampled energy curves.

    Parameters
    ----------
    seps_ref
        Reference separations.
    energy_ref
        Reference energies.
    seps_pred
        Predicted separations.
    energy_pred
        Predicted energies.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Sorted reference separations and energies followed by predicted values.
    """
    separations_ref, energies_ref = _validate_diatomic_curve(seps_ref, energy_ref)
    separations_pred, energies_pred = _validate_diatomic_curve(seps_pred, energy_pred)
    return separations_ref, energies_ref, separations_pred, energies_pred


def _quadratic_well_fit(
    separations: ArrayLike,
    energies: ArrayLike,
    n_fit_points: int = 5,
) -> tuple[float, float]:
    """
    Estimate equilibrium separation and curvature by local quadratic fit.

    Parameters
    ----------
    separations
        Sample separations.
    energies
        Sample energies.
    n_fit_points
        Maximum number of local points to fit.

    Returns
    -------
    tuple[float, float]
        Equilibrium separation and fitted curvature.
    """
    separations, energies = _validate_diatomic_curve(separations, energies)
    minimum_index = int(np.argmin(energies))
    if len(separations) < 3:
        return float(separations[minimum_index]), np.nan

    start_index = min(
        max(0, minimum_index - n_fit_points // 2),
        max(0, len(separations) - n_fit_points),
    )
    fit_separations = separations[start_index : start_index + n_fit_points]
    fit_energies = energies[start_index : start_index + n_fit_points]
    if len(fit_separations) < 3:
        return float(separations[minimum_index]), np.nan

    quadratic_coefficient, linear_coefficient, _constant_coefficient = np.polyfit(
        fit_separations, fit_energies, 2
    )
    curvature = 2 * quadratic_coefficient
    if quadratic_coefficient <= 0:
        return float(separations[minimum_index]), np.nan

    equilibrium_distance = -linear_coefficient / (2 * quadratic_coefficient)
    if fit_separations.min() <= equilibrium_distance <= fit_separations.max():
        return float(equilibrium_distance), float(curvature)
    return float(separations[minimum_index]), float(curvature)


def _repulsive_radius_at_threshold(
    separations: ArrayLike,
    energies: ArrayLike,
    threshold_ev: float,
) -> float:
    """
    Return the repulsive radius at an energy threshold, or NaN if unreached.

    Parameters
    ----------
    separations
        Sample separations.
    energies
        Sample energies.
    threshold_ev
        Energy above the curve minimum.

    Returns
    -------
    float
        Interpolated repulsive radius or NaN.
    """
    separations, energies = _validate_diatomic_curve(separations, energies)
    minimum_index = int(np.argmin(energies))
    if minimum_index == 0:
        return np.nan

    radii_inward = separations[minimum_index::-1]
    energy_above_minimum = energies[minimum_index::-1] - energies[minimum_index]
    monotonic_energy = np.maximum.accumulate(energy_above_minimum)
    unique_energy, unique_indices = np.unique(monotonic_energy, return_index=True)
    if len(unique_energy) < 2 or threshold_ev > unique_energy[-1]:
        return np.nan
    return float(np.interp(threshold_ev, unique_energy, radii_inward[unique_indices]))


def calc_pbe_wall_dist_mae(
    seps_ref: ArrayLike,
    energy_ref: ArrayLike,
    seps_pred: ArrayLike,
    energy_pred: ArrayLike,
    *,
    thresholds_ev: tuple[float, ...] = PBE_WALL_ENERGY_THRESHOLDS_EV,
) -> float:
    """
    Calculate mean PBE wall-radius error over reachable energy thresholds.

    A missing predicted crossing receives the full reference-radius error.

    Parameters
    ----------
    seps_ref
        Reference separations.
    energy_ref
        Reference energies.
    seps_pred
        Predicted separations.
    energy_pred
        Predicted energies.
    thresholds_ev
        Energy thresholds above the well minimum.

    Returns
    -------
    float
        Mean absolute wall-radius error.
    """
    errors: list[float] = []
    for threshold_ev in thresholds_ev:
        radius_ref = _repulsive_radius_at_threshold(seps_ref, energy_ref, threshold_ev)
        if not np.isfinite(radius_ref):
            continue
        radius_pred = _repulsive_radius_at_threshold(
            seps_pred, energy_pred, threshold_ev
        )
        errors.append(
            abs(radius_pred - radius_ref) if np.isfinite(radius_pred) else radius_ref
        )
    return float(np.mean(errors)) if errors else np.nan


def calc_pbe_energy_mae(
    seps_ref: ArrayLike,
    energy_ref: ArrayLike,
    seps_pred: ArrayLike,
    energy_pred: ArrayLike,
    *,
    interpolate: bool | int = 200,
) -> float:
    """
    Calculate PBE energy MAE after optional interpolation and far-field alignment.

    Parameters
    ----------
    seps_ref
        Reference separations.
    energy_ref
        Reference energies.
    seps_pred
        Predicted separations.
    energy_pred
        Predicted energies.
    interpolate
        Whether or how many common-grid points to use.

    Returns
    -------
    float
        Mean absolute energy error.
    """
    _, energy_ref, energy_pred = _common_grid_curve_pair(
        seps_ref,
        energy_ref,
        seps_pred,
        energy_pred,
        interpolate=interpolate,
    )
    energy_ref = energy_ref - energy_ref[-1]
    energy_pred = energy_pred - energy_pred[-1]
    return float(np.mean(np.abs(energy_pred - energy_ref)))


def calc_pbe_bond_length_error(
    seps_ref: ArrayLike,
    energy_ref: ArrayLike,
    seps_pred: ArrayLike,
    energy_pred: ArrayLike,
    *,
    min_ref_binding_ev: float = 0.05,
) -> float:
    """
    Calculate absolute PBE equilibrium-distance error, or NaN if unbound.

    Parameters
    ----------
    seps_ref
        Reference separations.
    energy_ref
        Reference energies.
    seps_pred
        Predicted separations.
    energy_pred
        Predicted energies.
    min_ref_binding_ev
        Minimum reference binding energy required for scoring.

    Returns
    -------
    float
        Absolute equilibrium-distance error or NaN.
    """
    separations_ref, energy_ref, separations_pred, energy_pred = _validated_energy_pair(
        seps_ref, energy_ref, seps_pred, energy_pred
    )
    if _binding_energy(energy_ref) < min_ref_binding_ev:
        return np.nan
    reference_distance = _quadratic_well_fit(separations_ref, energy_ref)[0]
    predicted_distance = _quadratic_well_fit(separations_pred, energy_pred)[0]
    return float(abs(predicted_distance - reference_distance))


def calc_pbe_well_depth_error(
    seps_ref: ArrayLike,
    energy_ref: ArrayLike,
    seps_pred: ArrayLike,
    energy_pred: ArrayLike,
    *,
    min_ref_binding_ev: float = 0.05,
) -> float:
    """
    Calculate absolute PBE well-depth error, or NaN if unbound.

    Parameters
    ----------
    seps_ref
        Reference separations.
    energy_ref
        Reference energies.
    seps_pred
        Predicted separations.
    energy_pred
        Predicted energies.
    min_ref_binding_ev
        Minimum reference binding energy required for scoring.

    Returns
    -------
    float
        Absolute well-depth error or NaN.
    """
    _, energy_ref, _, energy_pred = _validated_energy_pair(
        seps_ref, energy_ref, seps_pred, energy_pred
    )
    reference_depth = _binding_energy(energy_ref)
    if reference_depth < min_ref_binding_ev:
        return np.nan
    return float(abs(_binding_energy(energy_pred) - reference_depth))


def _vibrational_wavenumber_cm(
    element_symbol: str,
    curvature_ev_per_a2: float,
) -> float:
    """
    Convert a homonuclear force constant to harmonic wavenumber in cm⁻¹.

    Parameters
    ----------
    element_symbol
        Element or homonuclear pair label.
    curvature_ev_per_a2
        Energy-well curvature in eV/Å².

    Returns
    -------
    float
        Harmonic wavenumber in cm⁻¹ or NaN.
    """
    if not np.isfinite(curvature_ev_per_a2) or curvature_ev_per_a2 <= 0:
        return np.nan
    atomic_symbol = element_symbol.split("-", maxsplit=1)[0]
    reduced_mass_kg = (
        atomic_masses[atomic_numbers[atomic_symbol]] * 1.66053906660e-27 / 2
    )
    force_constant_n_per_m = curvature_ev_per_a2 * 16.02176634
    angular_frequency_per_second = np.sqrt(force_constant_n_per_m / reduced_mass_kg)
    return float(angular_frequency_per_second / (2 * np.pi * 2.99792458e10))


def calc_pbe_vib_freq_error(
    elem_symbol: str,
    seps_ref: ArrayLike,
    energy_ref: ArrayLike,
    seps_pred: ArrayLike,
    energy_pred: ArrayLike,
    *,
    min_ref_binding_ev: float = 0.05,
) -> float:
    """
    Calculate absolute PBE vibrational-wavenumber error, or NaN if unbound.

    Parameters
    ----------
    elem_symbol
        Element or homonuclear pair label.
    seps_ref
        Reference separations.
    energy_ref
        Reference energies.
    seps_pred
        Predicted separations.
    energy_pred
        Predicted energies.
    min_ref_binding_ev
        Minimum reference binding energy required for scoring.

    Returns
    -------
    float
        Absolute vibrational-wavenumber error or NaN.
    """
    separations_ref, energy_ref, separations_pred, energy_pred = _validated_energy_pair(
        seps_ref, energy_ref, seps_pred, energy_pred
    )
    if _binding_energy(energy_ref) < min_ref_binding_ev:
        return np.nan
    reference_curvature = _quadratic_well_fit(separations_ref, energy_ref)[1]
    predicted_curvature = _quadratic_well_fit(separations_pred, energy_pred)[1]
    reference_wavenumber = _vibrational_wavenumber_cm(elem_symbol, reference_curvature)
    predicted_wavenumber = _vibrational_wavenumber_cm(elem_symbol, predicted_curvature)
    return float(abs(predicted_wavenumber - reference_wavenumber))


def calc_tortuosity(seps: ArrayLike, energies: ArrayLike) -> float:
    """
    Calculate projected arc-chord energy tortuosity, or NaN if constant.

    Parameters
    ----------
    seps
        Sample separations.
    energies
        Sample energies.

    Returns
    -------
    float
        Curve tortuosity or NaN.
    """
    _, energies = _validate_diatomic_curve(seps, energies)

    total_energy_variation = np.sum(np.abs(np.diff(energies)))
    minimum_energy = np.min(energies)
    direct_energy_difference = abs(energies[0] - minimum_energy) + abs(
        energies[-1] - minimum_energy
    )

    if direct_energy_difference == 0:
        return np.nan
    return float(total_energy_variation / direct_energy_difference)


def _threshold_diff_signs(
    values: np.ndarray,
    threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return nonzero thresholded differences, their signs, and flip mask.

    Parameters
    ----------
    values
        Sample values.
    threshold
        Magnitudes below this value are treated as zero.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Nonzero differences, their signs, and adjacent sign-flip mask.
    """
    differences = np.diff(values)
    differences[np.abs(differences) < threshold] = 0
    signs = np.sign(differences)
    nonzero_mask = signs != 0
    differences, signs = differences[nonzero_mask], signs[nonzero_mask]
    flips = np.diff(signs) != 0
    return differences, signs, flips


def _jump_magnitude(values: np.ndarray, threshold: float = 1e-3) -> float:
    """
    Sum adjacent step magnitudes at sign-flip points.

    Parameters
    ----------
    values
        Sample values.
    threshold
        Difference magnitudes below this value are ignored.

    Returns
    -------
    float
        Total adjacent jump magnitude.
    """
    differences, _, flips = _threshold_diff_signs(values, threshold)
    return float(
        np.abs(differences[:-1][flips]).sum() + np.abs(differences[1:][flips]).sum()
    )


def calc_energy_diff_flips(
    seps: ArrayLike,
    energies: ArrayLike,
) -> float:
    """
    Calculate the number of thresholded energy-difference sign flips.

    Parameters
    ----------
    seps
        Sample separations.
    energies
        Sample energies.

    Returns
    -------
    float
        Number of energy-difference sign flips.
    """
    _, energies = _validate_diatomic_curve(seps, energies)
    _, _, flips = _threshold_diff_signs(energies)
    return float(np.sum(flips))


def calc_energy_jump(seps: ArrayLike, energies: ArrayLike) -> float:
    """
    Calculate total energy-step magnitude around sign-flip points.

    Parameters
    ----------
    seps
        Sample separations.
    energies
        Sample energies.

    Returns
    -------
    float
        Total energy-step magnitude.
    """
    _, energies = _validate_diatomic_curve(seps, energies)
    return _jump_magnitude(energies)
