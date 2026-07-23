"""Tests for imported diatomic energy and force formulas."""

from __future__ import annotations

from collections.abc import Callable
import re

import numpy as np
import pytest

from ml_peg.analysis.physicality.diatomics.metrics.energy import (
    calc_energy_diff_flips,
    calc_energy_jump,
    calc_pbe_bond_length_error,
    calc_pbe_energy_mae,
    calc_pbe_vib_freq_error,
    calc_pbe_wall_dist_mae,
    calc_pbe_well_depth_error,
    calc_tortuosity,
)
from ml_peg.analysis.physicality.diatomics.metrics.force import (
    calc_force_flips,
    calc_force_jump,
    calc_force_mae,
    calc_force_total_variation,
)

pytestmark = pytest.mark.framework("matbench-discovery")

_LENGTH_ERROR = re.escape("len(separation_array)=2 != len(value_array)=3")
_ENERGY_SHAPE_ERROR = re.escape("energy values must have shape (n,)")
_FORCE_SHAPE_ERROR = re.escape("force values must have shape (n, 2, 3)")


def _radial_forces(radial_values: np.ndarray) -> np.ndarray:
    """Return equal-and-opposite two-atom forces from radial values."""
    forces = np.zeros((len(radial_values), 2, 3))
    forces[:, 0, 0] = radial_values
    forces[:, 1, 0] = -radial_values
    return forces


@pytest.mark.parametrize(
    ("energies", "expected_flips", "expected_jump"),
    [
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 0, 0.0),
        (np.array([1.0, 3.0, 2.0, 4.0, 5.0]), 2, 6.0),
        (np.array([0.0, 2.0, 1.0, 3.0, 0.5]), 3, 10.5),
    ],
)
def test_energy_flip_and_jump_formulas(
    energies: np.ndarray,
    expected_flips: int,
    expected_jump: float,
) -> None:
    """Energy flips and jumps match hand-computed values."""
    separations = np.arange(1, len(energies) + 1, dtype=float)
    assert calc_energy_diff_flips(separations, energies) == expected_flips
    assert calc_energy_jump(separations, energies) == pytest.approx(expected_jump)


@pytest.mark.parametrize(
    ("energies", "expected"),
    [
        (np.arange(1, 6, dtype=float), 1.0),
        (np.arange(1, 6, dtype=float) ** 2, 1.0),
        (np.ones(5), np.nan),
    ],
)
def test_tortuosity_formulas(energies: np.ndarray, expected: float) -> None:
    """Tortuosity is one for monotone curves and NaN for flat curves."""
    result = calc_tortuosity(np.arange(1, 6), energies)
    assert result == pytest.approx(expected, nan_ok=True)


def test_source_keyword_argument_names_are_supported() -> None:
    """Public metric functions retain Matbench Discovery keyword names."""
    separations = np.arange(1, 6, dtype=float)
    energies = np.arange(1, 6, dtype=float)
    forces = np.zeros((5, 2, 3))
    assert calc_tortuosity(seps=separations, energies=energies) == pytest.approx(1)
    assert calc_force_total_variation(seps=separations, forces=forces) == 0
    energy_kwargs = {
        "seps_ref": separations,
        "energy_ref": energies,
        "seps_pred": separations,
        "energy_pred": energies,
    }
    force_kwargs = {
        "seps_ref": separations,
        "f_ref": forces,
        "seps_pred": separations,
        "f_pred": forces,
    }
    assert calc_pbe_energy_mae(**energy_kwargs) == 0
    assert calc_force_mae(**force_kwargs) == 0


def test_pbe_reference_energy_formulas() -> None:
    """PBE-relative metrics match analytic parabolic-well expectations."""
    reference_equilibrium = 1.5
    predicted_equilibrium = 1.6
    predicted_curvature_factor = 1.2
    reference_max = 3.0
    predicted_max = 3.05
    separations_ref = np.linspace(0.5, reference_max, 51)
    separations_pred = np.linspace(0.55, predicted_max, 51)
    energy_ref = (separations_ref - reference_equilibrium) ** 2 - 2
    energy_pred = (
        predicted_curvature_factor * (separations_pred - predicted_equilibrium) ** 2
        - 1.7
    )
    curve_args = (separations_ref, energy_ref, separations_pred, energy_pred)
    expected_depth_error = abs(
        predicted_curvature_factor * (predicted_max - predicted_equilibrium) ** 2
        - (reference_max - reference_equilibrium) ** 2
    )
    results = [
        (calc_pbe_wall_dist_mae(*curve_args, thresholds_ev=(1,)), 0.187, 0.01),
        (calc_pbe_energy_mae(*curve_args, interpolate=200), 0.118, 0.01),
        (
            calc_pbe_bond_length_error(*curve_args),
            predicted_equilibrium - reference_equilibrium,
            None,
        ),
        (calc_pbe_well_depth_error(*curve_args), expected_depth_error, 0.01),
        (calc_pbe_vib_freq_error("H", *curve_args), 99.1, 1),
    ]
    for actual, expected, absolute_tolerance in results:
        assert actual == pytest.approx(expected, abs=absolute_tolerance)


def test_force_metric_formulas() -> None:
    """Force metrics match hand-computed radial-force values."""
    separations = np.arange(1, 6, dtype=float)
    forces = _radial_forces(np.array([1.0, 2.0, -1.0, 3.0, -2.0]))
    assert calc_force_flips(separations, forces) == 3
    assert calc_force_total_variation(separations, forces) == pytest.approx(13)
    assert calc_force_jump(separations, forces) == pytest.approx(20)
    assert calc_force_mae(separations, forces, separations, forces) == 0


def test_energy_and_force_interpolation_use_shared_range() -> None:
    """Interpolated MAEs compare linear curves only on their overlap."""
    separations_ref = np.array([2.0, 3.0, 4.0])
    separations_pred = np.array([2.1, 3.1, 4.1])
    curve_args = (separations_ref, separations_ref, separations_pred, separations_pred)
    assert calc_pbe_energy_mae(
        *curve_args,
        interpolate=20,
    ) == pytest.approx(0)
    assert calc_force_mae(
        separations_ref,
        _radial_forces(separations_ref),
        separations_pred,
        _radial_forces(separations_pred),
        interpolate=20,
    ) == pytest.approx(0)


@pytest.mark.parametrize(
    ("metric_function", "separations", "values", "error_match"),
    [
        (
            calc_energy_jump,
            np.array([1.0, 1.0, 2.0]),
            np.arange(3.0),
            "contains 1 duplicates",
        ),
        (
            calc_energy_jump,
            np.array([1.0, np.nan, 3.0]),
            np.arange(3.0),
            "Input contains NaN",
        ),
        (calc_energy_jump, np.arange(2.0), np.arange(3.0), _LENGTH_ERROR),
        (calc_energy_jump, np.arange(3.0), np.zeros((3, 1)), _ENERGY_SHAPE_ERROR),
        (
            calc_force_total_variation,
            np.arange(3.0),
            np.zeros((3, 1, 3)),
            _FORCE_SHAPE_ERROR,
        ),
    ],
)
def test_curve_validation_errors(
    metric_function: Callable[[np.ndarray, np.ndarray], float],
    separations: np.ndarray,
    values: np.ndarray,
    error_match: str,
) -> None:
    """Curve formulas reject duplicate, non-finite, and malformed inputs."""
    with pytest.raises(ValueError, match=error_match):
        metric_function(separations, values)


@pytest.mark.parametrize(
    "separations_pred",
    [np.array([4.0, 5.0, 6.0]), np.array([3.0, 4.0, 5.0])],
    ids=["disjoint", "single-shared-point"],
)
def test_force_interpolation_rejects_unusable_overlap(
    separations_pred: np.ndarray,
) -> None:
    """Force interpolation rejects disjoint and point-only overlap."""
    separations_ref = np.array([1.0, 2.0, 3.0])
    forces = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match="no overlap"):
        calc_force_mae(
            separations_ref, forces, separations_pred, forces, interpolate=True
        )


@pytest.mark.parametrize(
    ("metric_function", "reference_values", "predicted_values"),
    [
        (
            calc_pbe_energy_mae,
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 1.0, 0.0]),
        ),
        (calc_force_mae, np.zeros((3, 2, 3)), np.ones((3, 2, 3))),
    ],
)
def test_interpolation_requires_at_least_two_points(
    metric_function: Callable[..., float],
    reference_values: np.ndarray,
    predicted_values: np.ndarray,
) -> None:
    """Reject one-point interpolation, which erases far-field energy errors."""
    separations_ref = np.array([1.0, 2.0, 3.0])
    separations_pred = np.array([1.1, 2.1, 3.1])
    with pytest.raises(ValueError, match="at least 2 points"):
        metric_function(
            separations_ref,
            reference_values,
            separations_pred,
            predicted_values,
            interpolate=1,
        )
