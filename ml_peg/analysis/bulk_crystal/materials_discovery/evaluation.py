"""Path- and dataframe-driven materials-discovery evaluation."""

from __future__ import annotations

import json
import math
import os
from typing import Final, TypeAlias, TypedDict

import pandas as pd

from ml_peg.analysis.bulk_crystal.materials_discovery.metrics import (
    MetricValue,
    SubsetIndices,
    _align_predictions_prepared,
    _calc_discovery_metrics_prepared,
    _discovery_subset_indices_prepared,
    _hull_distances,
)
from ml_peg.analysis.bulk_crystal.materials_discovery.schema import (
    E_ABOVE_HULL,
    MATERIAL_ID,
    REFERENCE_FORMATION_ENERGY,
    DiscoverySubset,
    _validated_reference_frame,
    prediction_series,
)
from ml_peg.data.artifacts import (
    MATBENCH_DISCOVERY_ID,
    MATBENCH_DISCOVERY_VERSION,
    PathLike,
    read_csv_artifact,
)

RESULT_SCHEMA_VERSION: Final = 1
MAX_E_FORM_ERROR_THRESHOLD: Final = 5.0
EVALUATION_DECIMALS: Final = 3
MISSING_PREDICTIONS_KEY: Final = "missing_preds"

JsonMetricValue: TypeAlias = float | int | None
DiscoverySubsetResults: TypeAlias = dict[str, dict[str, JsonMetricValue]]


class SourceMetadata(TypedDict):
    """Identify the upstream framework used by an evaluation result."""

    framework: str
    version: str


class DiscoveryResults(TypedDict):
    """Serialized discovery evaluation result."""

    schema_version: int
    source: SourceMetadata
    subsets: DiscoverySubsetResults


def prepare_discovery_inputs(
    reference: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
    decimals: int = EVALUATION_DECIMALS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Validate, align, mask, and round discovery evaluation inputs.

    Errors strictly above the threshold become NaN before all energies are rounded
    to the shared evaluation precision.

    Parameters
    ----------
    reference
        Discovery reference data.
    predictions
        Formation-energy predictions.
    max_error_threshold
        Maximum absolute formation-energy error to retain.
    decimals
        Number of decimal places used for evaluation.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Prepared reference data and aligned predictions.
    """
    if max_error_threshold is not None and (
        not math.isfinite(max_error_threshold) or max_error_threshold < 0
    ):
        raise ValueError("max_error_threshold must be finite, non-negative, or None")
    if decimals < 0:
        raise ValueError("decimals must be non-negative")

    indexed_reference = _validated_reference_frame(reference)
    model_predictions = (
        predictions.copy()
        if isinstance(predictions, pd.Series)
        else prediction_series(predictions)
    )
    aligned_predictions = _align_predictions_prepared(
        indexed_reference, model_predictions
    )
    energy_columns = [E_ABOVE_HULL, REFERENCE_FORMATION_ENERGY]
    numeric_energies = indexed_reference[energy_columns].apply(pd.to_numeric)
    prepared_reference = indexed_reference.copy()
    prepared_reference[energy_columns] = numeric_energies.round(decimals)
    if max_error_threshold is not None:
        outlier_mask = (
            aligned_predictions - numeric_energies[REFERENCE_FORMATION_ENERGY]
        ).abs() > max_error_threshold
        aligned_predictions = aligned_predictions.mask(outlier_mask)

    return prepared_reference, aligned_predictions.round(decimals)


def discovery_subset_indices(
    reference: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
) -> dict[DiscoverySubset, pd.Index]:
    """
    Return benchmark subsets after applying artifact preprocessing.

    Parameters
    ----------
    reference
        Discovery reference data.
    predictions
        Formation-energy predictions.
    max_error_threshold
        Maximum absolute formation-energy error to retain.

    Returns
    -------
    dict[DiscoverySubset, pandas.Index]
        Material identifiers for each discovery subset.
    """
    prepared_reference, prepared_predictions = prepare_discovery_inputs(
        reference,
        predictions,
        max_error_threshold=max_error_threshold,
        decimals=EVALUATION_DECIMALS,
    )
    _, each_pred = _hull_distances(prepared_reference, prepared_predictions)
    return _discovery_subset_indices_prepared(prepared_reference, each_pred)


def calc_discovery_metrics(
    reference: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    subset_indices: SubsetIndices | None = None,
    uniq_proto_prevalence: float | None = None,
    canonical: bool = False,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
) -> dict[DiscoverySubset, dict[str, MetricValue]]:
    """
    Calculate metrics after applying artifact masking and rounding.

    Parameters
    ----------
    reference
        Discovery reference data.
    predictions
        Formation-energy predictions.
    subset_indices
        Optional material identifiers for each subset.
    uniq_proto_prevalence
        Stable-material prevalence among unique prototypes.
    canonical
        Whether to require canonical leaderboard inputs.
    max_error_threshold
        Maximum absolute formation-energy error to retain.

    Returns
    -------
    dict[DiscoverySubset, dict[str, MetricValue]]
        Metrics grouped by discovery subset.
    """
    prepared_reference, prepared_predictions = prepare_discovery_inputs(
        reference,
        predictions,
        max_error_threshold=max_error_threshold,
        decimals=EVALUATION_DECIMALS,
    )
    metrics_by_subset, _ = _calc_discovery_metrics_prepared(
        prepared_reference,
        prepared_predictions,
        subset_indices=subset_indices,
        uniq_proto_prevalence=uniq_proto_prevalence,
        canonical=canonical,
    )
    return metrics_by_subset


def _json_safe_metric(value: MetricValue) -> JsonMetricValue:
    """
    Round a metric and convert non-finite values to JSON null.

    Parameters
    ----------
    value
        Metric value to serialize.

    Returns
    -------
    float or int or None
        JSON-safe metric value.
    """
    if isinstance(value, int):
        return value
    numeric_value = float(value)
    return (
        round(numeric_value, EVALUATION_DECIMALS)
        if math.isfinite(numeric_value)
        else None
    )


def evaluate_discovery(
    reference: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    canonical: bool = False,
    uniq_proto_prevalence: float | None = None,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
) -> DiscoveryResults:
    """
    Evaluate formation-energy predictions on the three discovery subsets.

    Leaderboard mode takes unrounded unique-prototype prevalence; synthetic mode
    derives it from the prepared reference.

    Parameters
    ----------
    reference
        Discovery reference data.
    predictions
        Formation-energy predictions.
    canonical
        Whether to require canonical leaderboard inputs.
    uniq_proto_prevalence
        Stable-material prevalence among unique prototypes.
    max_error_threshold
        Maximum absolute formation-energy error to retain.

    Returns
    -------
    DiscoveryResults
        JSON-compatible evaluation result.
    """
    prepared_reference, prepared_predictions = prepare_discovery_inputs(
        reference,
        predictions,
        max_error_threshold=max_error_threshold,
        decimals=EVALUATION_DECIMALS,
    )
    raw_metrics, subset_indices = _calc_discovery_metrics_prepared(
        prepared_reference,
        prepared_predictions,
        subset_indices=None,
        uniq_proto_prevalence=uniq_proto_prevalence,
        canonical=canonical,
    )

    subsets: DiscoverySubsetResults = {}
    for subset in DiscoverySubset:
        subset_index = subset_indices[subset]
        subset_metrics = {
            metric_name: _json_safe_metric(metric_value)
            for metric_name, metric_value in raw_metrics[subset].items()
        }
        subset_metrics[MISSING_PREDICTIONS_KEY] = int(
            prepared_predictions.loc[subset_index].isna().sum()
        )
        subsets[str(subset)] = subset_metrics
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "source": {
            "framework": MATBENCH_DISCOVERY_ID,
            "version": MATBENCH_DISCOVERY_VERSION,
        },
        "subsets": subsets,
    }


def evaluate_discovery_paths(
    reference_path: PathLike,
    prediction_path: PathLike,
    *,
    canonical: bool = False,
    uniq_proto_prevalence: float | None = None,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
) -> DiscoveryResults:
    """
    Load local CSV artifacts and evaluate discovery predictions without writes.

    Parameters
    ----------
    reference_path
        Local reference CSV path.
    prediction_path
        Local prediction CSV path.
    canonical
        Whether to require canonical leaderboard inputs.
    uniq_proto_prevalence
        Stable-material prevalence among unique prototypes.
    max_error_threshold
        Maximum absolute formation-energy error to retain.

    Returns
    -------
    DiscoveryResults
        JSON-compatible evaluation result.
    """
    return evaluate_discovery(
        read_csv_artifact(reference_path, dtype={MATERIAL_ID: str}),
        read_csv_artifact(prediction_path, dtype={MATERIAL_ID: str}),
        canonical=canonical,
        uniq_proto_prevalence=uniq_proto_prevalence,
        max_error_threshold=max_error_threshold,
    )


def write_discovery_metrics_json(
    results: DiscoveryResults,
    output_path: str | os.PathLike[str],
) -> None:
    """
    Write discovery metrics as JSON.

    Parameters
    ----------
    results
        Evaluation results to serialize.
    output_path
        Destination JSON path.
    """
    with open(output_path, mode="w", encoding="utf-8") as file:
        json.dump(results, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")
