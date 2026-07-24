"""Classification, regression, and ranking metrics for materials discovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypeAlias

import numpy as np
from numpy import typing as npt
import pandas as pd
from sklearn.metrics import r2_score

from ml_peg.analysis.bulk_crystal.materials_discovery.schema import (
    E_ABOVE_HULL,
    MATERIAL_ID,
    REFERENCE_FORMATION_ENERGY,
    UNIQUE_PROTOTYPE,
    DiscoverySubset,
    _validated_reference_frame,
)

STABILITY_THRESHOLD: Final = 0.0
MOST_STABLE_COUNT: Final = 10_000

NumericValues: TypeAlias = Sequence[float | None] | pd.Series | npt.NDArray[np.generic]
MetricValue: TypeAlias = float | int
SubsetIndices: TypeAlias = Mapping[DiscoverySubset | str, pd.Index]
_ClassificationMasks: TypeAlias = tuple[pd.Series, pd.Series, pd.Series, pd.Series]


def _classify_stable(
    each_true: NumericValues,
    each_pred: NumericValues,
    *,
    stability_threshold: float,
    fillna: bool,
) -> tuple[_ClassificationMasks, pd.Series, pd.Series]:
    """
    Classify stability while retaining numeric inputs for regression.

    Parameters
    ----------
    each_true
        True hull distances.
    each_pred
        Predicted hull distances.
    stability_threshold
        Maximum hull distance considered stable.
    fillna
        Whether missing predictions count as unstable.

    Returns
    -------
    tuple
        Classification masks and numeric true and predicted values.
    """
    if len(each_true) != len(each_pred):
        raise ValueError(f"len(each_true)={len(each_true)} != {len(each_pred)=}")

    each_true_array = pd.to_numeric(
        pd.Series(each_true).reset_index(drop=True), errors="coerce"
    )
    each_pred_array = pd.to_numeric(
        pd.Series(each_pred).reset_index(drop=True), errors="coerce"
    )
    if stability_threshold is None or not np.isfinite(stability_threshold):
        raise ValueError("stability_threshold must be a real number")

    actual_positive = each_true_array <= stability_threshold
    actual_negative = each_true_array > stability_threshold
    model_positive = each_pred_array <= stability_threshold
    model_negative = each_pred_array > stability_threshold
    if fillna:
        missing_prediction_mask = each_pred_array.isna()
        # Missing predictions count as unstable for both model class masks.
        model_positive[missing_prediction_mask] = False
        model_negative[missing_prediction_mask] = True

    masks = (
        actual_positive & model_positive,
        actual_positive & model_negative,
        actual_negative & model_positive,
        actual_negative & model_negative,
    )
    return masks, each_true_array, each_pred_array


def classify_stable(
    each_true: NumericValues,
    each_pred: NumericValues,
    *,
    stability_threshold: float = STABILITY_THRESHOLD,
    fillna: bool = True,
) -> _ClassificationMasks:
    """
    Return classification masks for stable and unstable materials.

    Parameters
    ----------
    each_true
        True hull distances.
    each_pred
        Predicted hull distances.
    stability_threshold
        Maximum hull distance considered stable.
    fillna
        Whether missing predictions count as unstable.

    Returns
    -------
    tuple[pandas.Series, pandas.Series, pandas.Series, pandas.Series]
        True-positive, false-negative, false-positive, and true-negative masks.
    """
    masks, _, _ = _classify_stable(
        each_true,
        each_pred,
        stability_threshold=stability_threshold,
        fillna=fillna,
    )
    return masks


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    """
    Divide positive-denominator values, otherwise returning NaN.

    Parameters
    ----------
    numerator
        Ratio numerator.
    denominator
        Ratio denominator.

    Returns
    -------
    float
        Ratio or NaN for a non-positive denominator.
    """
    return numerator / denominator if denominator > 0 else float("nan")


def stable_metrics(
    each_true: NumericValues,
    each_pred: NumericValues,
    *,
    stability_threshold: float = STABILITY_THRESHOLD,
    fillna: bool = True,
) -> dict[str, MetricValue]:
    """
    Calculate classification and hull-distance regression metrics.

    Inputs should contain finite values or missing values. Artifact evaluation should
    use ``evaluate_discovery``, which also masks outliers and infinities.

    Parameters
    ----------
    each_true
        True hull distances.
    each_pred
        Predicted hull distances.
    stability_threshold
        Maximum hull distance considered stable.
    fillna
        Whether missing predictions count as unstable.

    Returns
    -------
    dict[str, MetricValue]
        Classification and regression metrics.
    """
    masks, each_true_array, each_pred_array = _classify_stable(
        each_true,
        each_pred,
        stability_threshold=stability_threshold,
        fillna=fillna,
    )
    (
        true_positive_count,
        false_negative_count,
        false_positive_count,
        true_negative_count,
    ) = (int(mask.sum()) for mask in masks)

    total_positive_count = true_positive_count + false_negative_count
    total_negative_count = true_negative_count + false_positive_count
    classified_count = total_positive_count + total_negative_count
    # Prevalence is the discovery rate from random selection over this population.
    prevalence = _safe_ratio(total_positive_count, classified_count)
    predicted_positive_count = true_positive_count + false_positive_count
    precision = _safe_ratio(true_positive_count, predicted_positive_count)
    recall = _safe_ratio(true_positive_count, total_positive_count)
    true_positive_rate = recall
    false_positive_rate = _safe_ratio(false_positive_count, total_negative_count)
    true_negative_rate = _safe_ratio(true_negative_count, total_negative_count)
    false_negative_rate = _safe_ratio(false_negative_count, total_positive_count)

    # False positives plus true negatives must account for all actual negatives.
    if (
        false_positive_rate > 0
        and true_negative_rate > 0
        and not np.isclose(false_positive_rate + true_negative_rate, 1)
    ):
        raise ValueError(
            f"FPR={false_positive_rate} and TNR={true_negative_rate} do not add up to 1"
        )
    # True positives plus false negatives must account for all actual positives.
    if (
        true_positive_rate > 0
        and false_negative_rate > 0
        and not np.isclose(true_positive_rate + false_negative_rate, 1)
    ):
        raise ValueError(
            f"TPR={true_positive_rate} and FNR={false_negative_rate} do not add up to 1"
        )

    missing_pair_mask = each_true_array.isna() | each_pred_array.isna()
    valid_true = each_true_array[~missing_pair_mask].to_numpy()
    valid_pred = each_pred_array[~missing_pair_mask].to_numpy()
    f1_score = (
        float("nan")
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    if len(valid_true) == 0:
        mean_absolute_error = root_mean_squared_error = float("nan")
    else:
        prediction_errors = valid_true - valid_pred
        mean_absolute_error = float(np.abs(prediction_errors).mean())
        root_mean_squared_error = float((prediction_errors**2).mean() ** 0.5)

    return {
        "F1": f1_score,
        "DAF": _safe_ratio(precision, prevalence),
        "Precision": precision,
        "Recall": recall,
        "Accuracy": _safe_ratio(
            true_positive_count + true_negative_count, classified_count
        ),
        "TPR": true_positive_rate,
        "FPR": false_positive_rate,
        "TNR": true_negative_rate,
        "FNR": false_negative_rate,
        "TP": true_positive_count,
        "FP": false_positive_count,
        "TN": true_negative_count,
        "FN": false_negative_count,
        "MAE": mean_absolute_error,
        "RMSE": root_mean_squared_error,
        "R2": (
            float(r2_score(valid_true, valid_pred))
            if len(valid_true) > 1
            else float("nan")
        ),
    }


def _align_predictions_prepared(
    indexed_reference: pd.DataFrame, model_predictions: pd.Series
) -> pd.Series:
    """
    Align predictions to an already-validated reference index.

    Parameters
    ----------
    indexed_reference
        Validated reference data indexed by material ID.
    model_predictions
        Predictions indexed by material ID.

    Returns
    -------
    pandas.Series
        Numeric predictions aligned to the reference.
    """
    if model_predictions.index.hasnans:
        raise ValueError("discovery predictions contain missing material_id values")
    model_predictions = model_predictions.copy()
    model_predictions.index = model_predictions.index.astype(str)
    if model_predictions.index.has_duplicates:
        duplicate_ids = (
            model_predictions.index[model_predictions.index.duplicated()]
            .unique()
            .tolist()
        )
        raise ValueError(
            "discovery predictions contain duplicate material_id values: "
            f"{duplicate_ids!r}"
        )
    unknown_ids = model_predictions.index.difference(indexed_reference.index)
    if len(unknown_ids) > 0:
        rendered_ids = sorted(map(str, unknown_ids))
        raise ValueError(f"Predictions contain unknown material IDs: {rendered_ids!r}")
    aligned_predictions = model_predictions.reindex(indexed_reference.index)
    aligned_predictions.index.name = MATERIAL_ID
    return pd.to_numeric(aligned_predictions, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def align_predictions(
    reference: pd.DataFrame, model_predictions: pd.Series
) -> pd.Series:
    """
    Align predictions to reference order, rejecting invalid or unknown IDs.

    Missing reference IDs remain as NaN predictions.

    Parameters
    ----------
    reference
        Discovery reference data.
    model_predictions
        Predictions indexed by material ID.

    Returns
    -------
    pandas.Series
        Numeric predictions aligned to the reference.
    """
    return _align_predictions_prepared(
        _validated_reference_frame(reference), model_predictions
    )


def _hull_distances(
    indexed_reference: pd.DataFrame, aligned_predictions: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Return true and predicted hull distances from prepared inputs.

    Parameters
    ----------
    indexed_reference
        Validated reference data indexed by material ID.
    aligned_predictions
        Formation-energy predictions aligned to the reference.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        True and predicted hull distances.
    """
    each_true = pd.to_numeric(indexed_reference[E_ABOVE_HULL], errors="coerce")
    reference_formation_energy = pd.to_numeric(
        indexed_reference[REFERENCE_FORMATION_ENERGY], errors="coerce"
    )
    return (
        each_true,
        each_true + aligned_predictions - reference_formation_energy,
    )


def _discovery_subset_indices_prepared(
    indexed_reference: pd.DataFrame,
    each_pred: pd.Series,
) -> dict[DiscoverySubset, pd.Index]:
    """
    Return the three subsets from prepared reference and hull distances.

    Parameters
    ----------
    indexed_reference
        Validated reference data indexed by material ID.
    each_pred
        Predicted hull distances aligned to the reference.

    Returns
    -------
    dict[DiscoverySubset, pandas.Index]
        Material identifiers for each discovery subset.
    """
    unique_prototype_index = indexed_reference.index[
        indexed_reference[UNIQUE_PROTOTYPE].astype(bool)
    ]
    most_stable_index = (
        each_pred.loc[unique_prototype_index]
        .sort_values(na_position="last", kind="stable")
        .head(MOST_STABLE_COUNT)
        .index
    )
    return {
        DiscoverySubset.full_test_set: indexed_reference.index,
        DiscoverySubset.unique_prototypes: unique_prototype_index,
        DiscoverySubset.most_stable_10k: most_stable_index,
    }


def _normalized_subset_indices(
    subset_indices: SubsetIndices,
    reference_index: pd.Index,
) -> dict[DiscoverySubset, pd.Index]:
    """
    Normalize and validate given subset indices.

    Parameters
    ----------
    subset_indices
        Material identifiers keyed by discovery subset.
    reference_index
        Valid reference material identifiers.

    Returns
    -------
    dict[DiscoverySubset, pandas.Index]
        Validated material identifiers for every subset.
    """
    normalized: dict[DiscoverySubset, pd.Index] = {}
    for subset_key, identifiers in subset_indices.items():
        try:
            subset = DiscoverySubset(subset_key)
        except ValueError as exc:
            raise ValueError(f"Unknown discovery subset {subset_key!r}") from exc
        subset_index = pd.Index(identifiers, name=MATERIAL_ID)
        if subset_index.has_duplicates:
            raise ValueError(f"{subset} subset contains duplicate material IDs")
        unknown_ids = subset_index.difference(reference_index)
        if len(unknown_ids) > 0:
            raise ValueError(
                f"{subset} subset contains unknown material IDs: "
                f"{sorted(map(str, unknown_ids))!r}"
            )
        normalized[subset] = subset_index

    missing_subsets = set(DiscoverySubset) - set(normalized)
    if missing_subsets:
        raise ValueError(
            f"Missing required discovery subsets: {sorted(map(str, missing_subsets))!r}"
        )
    return normalized


def _calc_discovery_metrics_prepared(
    indexed_reference: pd.DataFrame,
    aligned_predictions: pd.Series,
    *,
    subset_indices: SubsetIndices | None,
    uniq_proto_prevalence: float | None,
    canonical: bool,
) -> tuple[
    dict[DiscoverySubset, dict[str, MetricValue]],
    dict[DiscoverySubset, pd.Index],
]:
    """
    Calculate metrics from validated, aligned discovery inputs.

    Parameters
    ----------
    indexed_reference
        Validated reference data indexed by material ID.
    aligned_predictions
        Formation-energy predictions aligned to the reference.
    subset_indices
        Optional material identifiers for each subset.
    uniq_proto_prevalence
        Stable-material prevalence among unique prototypes.
    canonical
        Whether to require canonical leaderboard inputs.

    Returns
    -------
    tuple
        Metrics and material identifiers grouped by discovery subset.
    """
    each_true, each_pred = _hull_distances(indexed_reference, aligned_predictions)
    canonical_indices = (
        _discovery_subset_indices_prepared(indexed_reference, each_pred)
        if subset_indices is None
        else _normalized_subset_indices(subset_indices, indexed_reference.index)
    )
    metrics_by_subset = {
        subset: stable_metrics(
            each_true.loc[subset_index],
            each_pred.loc[subset_index],
            fillna=True,
        )
        for subset, subset_index in canonical_indices.items()
    }

    if canonical and uniq_proto_prevalence is None:
        raise ValueError(
            "leaderboard evaluation requires explicit unrounded "
            "unique-prototype prevalence"
        )
    if uniq_proto_prevalence is None:
        unique_each_true = each_true.loc[
            canonical_indices[DiscoverySubset.unique_prototypes]
        ]
        uniq_proto_prevalence = float((unique_each_true <= STABILITY_THRESHOLD).mean())
    elif not np.isfinite(uniq_proto_prevalence) or not (
        0 <= uniq_proto_prevalence <= 1
    ):
        raise ValueError(
            "uniq_proto_prevalence must be a finite fraction between 0 and 1"
        )

    daf_denominator = (
        uniq_proto_prevalence if uniq_proto_prevalence > 0 else float("nan")
    )
    for subset in (
        DiscoverySubset.unique_prototypes,
        DiscoverySubset.most_stable_10k,
    ):
        metrics_by_subset[subset]["DAF"] = (
            metrics_by_subset[subset]["Precision"] / daf_denominator
        )
    return metrics_by_subset, canonical_indices
