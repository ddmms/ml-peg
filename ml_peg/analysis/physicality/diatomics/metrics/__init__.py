"""Diatomic potential-energy metrics adapted from Matbench Discovery.

The smoothness approach follows Stenczel et al., https://arxiv.org/abs/2401.00096.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from ase.data import atomic_numbers, covalent_radii, vdw_alvarez
import numpy as np

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
from ml_peg.analysis.physicality.diatomics.metrics.schema import (
    DEFAULT_DFT_REFERENCE_PATH,
    DiatomicCurve,
    DiatomicCurves,
    curves_from_ml_peg_dataframe,
    homo_key,
    load_dft_reference_curves,
    load_mbd_json,
    load_ml_peg_curves,
)

logger = logging.getLogger(__name__)

TORTUOSITY = "tortuosity"
FORCE_FLIPS = "force_flips"
ENERGY_JUMP = "energy_jump"
ENERGY_DIFF_FLIPS = "energy_diff_flips"
FORCE_TOTAL_VARIATION = "force_total_variation"
FORCE_JUMP = "force_jump"
PBE_WALL_DIST_MAE = "pbe_wall_dist_mae"
PBE_ENERGY_MAE = "pbe_energy_mae"
PBE_BOND_LENGTH_ERROR = "pbe_bond_length_error"
PBE_WELL_DEPTH_ERROR = "pbe_well_depth_error"
PBE_FORCE_MAE = "pbe_force_mae"
PBE_VIB_FREQ_ERROR = "pbe_vib_freq_error"

DIATOMIC_METRIC_NAMES: tuple[str, ...] = (
    TORTUOSITY,
    FORCE_FLIPS,
    ENERGY_JUMP,
    ENERGY_DIFF_FLIPS,
    FORCE_TOTAL_VARIATION,
    FORCE_JUMP,
    PBE_WALL_DIST_MAE,
    PBE_ENERGY_MAE,
    PBE_BOND_LENGTH_ERROR,
    PBE_WELL_DEPTH_ERROR,
    PBE_FORCE_MAE,
    PBE_VIB_FREQ_ERROR,
)
DIATOMIC_METRIC_KEYS = frozenset(DIATOMIC_METRIC_NAMES)

# Skip H-U elements absent from Materials Project training data.
NON_MP_ELEMENTS = frozenset({"Po", "At", "Rn", "Fr", "Ra"})
DIATOMIC_WALL_R_MIN_FACTOR = 0.8
MAX_SCORED_ATOMIC_NUMBER = 92


def find_low_quality_dft_refs(
    ref_curves: DiatomicCurves,
    *,
    min_energy_jump: float = 1.5,
    min_energy_flips: int = 3,
) -> set[str]:
    """Find non-finite or discontinuous DFT references unsuitable for scoring."""
    low_quality: set[str] = set()
    for element_symbol, curve in ref_curves.homo_nuclear.items():
        separations = curve.distances
        energies = curve.energies
        if separations.size == 0:
            continue
        radius_min, radius_max = eval_window(element_symbol, float(np.max(separations)))
        window_mask = (separations >= radius_min) & (separations <= radius_max)
        if window_mask.sum() < 5:
            continue  # too few in-window points to assess smoothness
        if not np.isfinite(energies[window_mask]).all():
            # Non-finite references cannot be scored.
            low_quality.add(element_symbol)
            continue
        if (
            calc_energy_jump(separations[window_mask], energies[window_mask])
            >= min_energy_jump
            and calc_energy_diff_flips(separations[window_mask], energies[window_mask])
            >= min_energy_flips
        ):
            low_quality.add(element_symbol)
    return low_quality


def eval_window(
    elem_symbol: str,
    seps_max: float,
    *,
    r_min_factor: float = 0.9,
) -> tuple[float, float]:
    """Return the covalent-to-van-der-Waals evaluation window in Å."""
    atomic_number = atomic_numbers[elem_symbol.split("-", maxsplit=1)[0]]
    covalent_radius = (
        covalent_radii[atomic_number] if atomic_number < len(covalent_radii) else np.nan
    )
    radius_min = r_min_factor * covalent_radius if np.isfinite(covalent_radius) else 0.0
    vdw_radii = vdw_alvarez.vdw_radii
    vdw_radius = vdw_radii[atomic_number] if atomic_number < len(vdw_radii) else np.nan
    radius_max = (
        min(3.1 * vdw_radius, seps_max) if np.isfinite(vdw_radius) else seps_max
    )
    return radius_min, radius_max


def calc_diatomic_metrics(
    ref_curves: DiatomicCurves | None,
    pred_curves: DiatomicCurves,
    metrics: dict[str, dict[str, Any]] | None = None,
    *,
    interpolate: bool | int = False,
) -> dict[str, dict[str, float]]:
    """Calculate requested metrics for supported homonuclear curves by element.

    Low-quality references receive self-consistency metrics but no ``pbe_*`` metrics.
    """
    requested_metric_keys = (
        set(metrics) if metrics is not None else set(DIATOMIC_METRIC_KEYS)
    )
    unknown_metrics = requested_metric_keys - DIATOMIC_METRIC_KEYS
    if unknown_metrics:
        raise ValueError(
            f"unknown_metrics={unknown_metrics}. "
            f"Valid metrics={sorted(DIATOMIC_METRIC_KEYS)}"
        )
    metric_kwargs = {key: kwargs.copy() for key, kwargs in (metrics or {}).items()}
    for metric_key in (PBE_ENERGY_MAE, PBE_FORCE_MAE):
        if metric_key in requested_metric_keys:
            metric_kwargs.setdefault(metric_key, {}).setdefault(
                "interpolate", interpolate
            )

    low_quality_refs = find_low_quality_dft_refs(ref_curves) if ref_curves else set()
    results: dict[str, dict[str, float]] = {}
    seen_elements: set[str] = set()
    for element_symbol, pred_data in pred_curves.homo_nuclear.items():
        normalized_element = homo_key(element_symbol)
        if normalized_element in seen_elements:
            raise ValueError(
                f"Duplicate homonuclear curve for element {normalized_element!r}"
            )
        seen_elements.add(normalized_element)
        if (
            normalized_element in NON_MP_ELEMENTS
            or atomic_numbers[normalized_element] > MAX_SCORED_ATOMIC_NUMBER
        ):
            continue  # score all models on the same MP-supported element set
        # General metrics use the MLIP Arena window; wall metrics extend to
        # 0.8 times the covalent radius.
        predicted_distances = pred_data.distances
        separations_max = float(predicted_distances.max())
        radius_min, radius_max = eval_window(element_symbol, separations_max)
        predicted_mask = (predicted_distances >= radius_min) & (
            predicted_distances <= radius_max
        )
        if predicted_mask.sum() < 5:  # too few points in window for stable metrics
            logger.info(
                "Skipping %s diatomic metrics: <5 points in eval window",
                element_symbol,
            )
            continue
        predicted_separations = predicted_distances[predicted_mask]
        predicted_energies_raw = pred_data.energies
        predicted_energies = predicted_energies_raw[predicted_mask]
        predicted_forces_raw = pred_data.forces
        if not predicted_forces_raw.size:
            raise ValueError(f"{element_symbol} diatomic curve is missing forces")
        if len(predicted_forces_raw) != len(predicted_distances):
            raise ValueError(
                f"{element_symbol} diatomic force and distance counts differ: "
                f"{len(predicted_forces_raw)} != {len(predicted_distances)}"
            )
        predicted_forces = predicted_forces_raw[predicted_mask]

        wall_radius_min = eval_window(
            element_symbol,
            separations_max,
            r_min_factor=DIATOMIC_WALL_R_MIN_FACTOR,
        )[0]
        # Include a generated DFT endpoint that differs from 0.8*r_cov by one ulp.
        wall_radius_min -= 1e-12
        predicted_wall_mask = (predicted_distances >= wall_radius_min) & (
            predicted_distances <= radius_max
        )
        if not (
            np.isfinite(predicted_energies_raw[predicted_wall_mask]).all()
            and np.isfinite(predicted_forces_raw[predicted_wall_mask]).all()
        ):
            logger.info(
                "Skipping %s diatomic metrics: non-finite wall values", element_symbol
            )
            continue

        energy_args = (predicted_separations, predicted_energies)
        force_args = (predicted_separations, predicted_forces)
        # Calls for metrics that need only the predicted curve.
        metric_calls: list[tuple[str, Callable[..., float], tuple[Any, ...]]] = [
            (TORTUOSITY, calc_tortuosity, energy_args),
            (ENERGY_DIFF_FLIPS, calc_energy_diff_flips, energy_args),
            (ENERGY_JUMP, calc_energy_jump, energy_args),
            (FORCE_FLIPS, calc_force_flips, force_args),
            (FORCE_TOTAL_VARIATION, calc_force_total_variation, force_args),
            (FORCE_JUMP, calc_force_jump, force_args),
        ]

        # Add relative metrics only for references that pass the quality gate.
        reference_data = (
            ref_curves.homo_nuclear.get(normalized_element)
            if ref_curves and normalized_element not in low_quality_refs
            else None
        )
        if reference_data is not None:
            reference_distances = reference_data.distances
            reference_mask = (reference_distances >= radius_min) & (
                reference_distances <= radius_max
            )
            reference_separations = reference_distances[reference_mask]
            reference_energies_raw = reference_data.energies
            reference_energies = reference_energies_raw[reference_mask]
            if len(reference_separations) >= 2:
                pair_args = (
                    reference_separations,
                    reference_energies,
                    predicted_separations,
                    predicted_energies,
                )
                metric_calls[:0] = [
                    (PBE_ENERGY_MAE, calc_pbe_energy_mae, pair_args),
                    (PBE_BOND_LENGTH_ERROR, calc_pbe_bond_length_error, pair_args),
                    (PBE_WELL_DEPTH_ERROR, calc_pbe_well_depth_error, pair_args),
                    (
                        PBE_VIB_FREQ_ERROR,
                        calc_pbe_vib_freq_error,
                        (element_symbol, *pair_args),
                    ),
                ]
            reference_wall_mask = (reference_distances >= wall_radius_min) & (
                reference_distances <= radius_max
            )
            if predicted_wall_mask.sum() >= 2 and reference_wall_mask.sum() >= 2:
                wall_args = (
                    reference_distances[reference_wall_mask],
                    reference_energies_raw[reference_wall_mask],
                    predicted_distances[predicted_wall_mask],
                    predicted_energies_raw[predicted_wall_mask],
                )
                metric_calls.insert(
                    0,
                    (PBE_WALL_DIST_MAE, calc_pbe_wall_dist_mae, wall_args),
                )
            reference_forces = reference_data.forces
            if (
                reference_forces.size
                and len(reference_forces) == len(reference_distances)
                and len(reference_separations) >= 2
            ):
                reference_forces = reference_forces[reference_mask]
                force_interpolate = metric_kwargs.get(PBE_FORCE_MAE, {}).get(
                    "interpolate", False
                )
                same_grid = np.array_equal(reference_separations, predicted_separations)
                has_overlap = max(
                    reference_separations.min(),
                    predicted_separations.min(),
                ) < min(
                    reference_separations.max(),
                    predicted_separations.max(),
                )
                if same_grid or (force_interpolate and has_overlap):
                    force_pair_args = (
                        reference_separations,
                        reference_forces,
                        predicted_separations,
                        predicted_forces,
                    )
                    metric_calls.append(
                        (PBE_FORCE_MAE, calc_force_mae, force_pair_args)
                    )

        results[normalized_element] = {
            metric_key: metric_function(
                *metric_args,
                **metric_kwargs.get(metric_key, {}),
            )
            for metric_key, metric_function, metric_args in metric_calls
            if metric_key in requested_metric_keys
        }

    return results


def aggregate_finite_means(
    metrics_by_element: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Average finite values for every metric present, to four significant digits."""
    metric_means: dict[str, float] = {}
    metric_names = dict.fromkeys(
        metric_name
        for element_metrics in metrics_by_element.values()
        for metric_name in element_metrics
    )
    for metric_name in metric_names:
        finite_values = [
            metric_value
            for element_metrics in metrics_by_element.values()
            if (metric_value := element_metrics.get(metric_name)) is not None
            and np.isfinite(metric_value)
        ]
        if finite_values:
            value_scale = max(abs(metric_value) for metric_value in finite_values)
            if value_scale == 0:
                metric_mean = 0.0
            else:
                metric_mean = value_scale * (
                    sum(metric_value / value_scale for metric_value in finite_values)
                    / len(finite_values)
                )
            if np.isfinite(metric_mean):
                metric_means[metric_name] = float(f"{metric_mean:.4}")
    return metric_means


__all__ = [
    "DEFAULT_DFT_REFERENCE_PATH",
    "DIATOMIC_METRIC_KEYS",
    "DIATOMIC_METRIC_NAMES",
    "DIATOMIC_WALL_R_MIN_FACTOR",
    "ENERGY_DIFF_FLIPS",
    "ENERGY_JUMP",
    "FORCE_FLIPS",
    "FORCE_JUMP",
    "FORCE_TOTAL_VARIATION",
    "NON_MP_ELEMENTS",
    "PBE_BOND_LENGTH_ERROR",
    "PBE_ENERGY_MAE",
    "PBE_FORCE_MAE",
    "PBE_VIB_FREQ_ERROR",
    "PBE_WALL_DIST_MAE",
    "PBE_WELL_DEPTH_ERROR",
    "TORTUOSITY",
    "DiatomicCurve",
    "DiatomicCurves",
    "aggregate_finite_means",
    "calc_diatomic_metrics",
    "calc_energy_diff_flips",
    "calc_energy_jump",
    "calc_force_flips",
    "calc_force_jump",
    "calc_force_mae",
    "calc_force_total_variation",
    "calc_pbe_bond_length_error",
    "calc_pbe_energy_mae",
    "calc_pbe_vib_freq_error",
    "calc_pbe_wall_dist_mae",
    "calc_pbe_well_depth_error",
    "calc_tortuosity",
    "curves_from_ml_peg_dataframe",
    "eval_window",
    "find_low_quality_dft_refs",
    "load_dft_reference_curves",
    "load_mbd_json",
    "load_ml_peg_curves",
]
