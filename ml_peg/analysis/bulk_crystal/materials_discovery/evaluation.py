"""Path- and dataframe-driven materials-discovery evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from typing import Final, TypeAlias

import pandas as pd

from ml_peg.analysis.bulk_crystal.materials_discovery.metrics import (
    MetricValue,
    _align_predictions_prepared,
    _calc_discovery_metrics_prepared,
)
from ml_peg.analysis.bulk_crystal.materials_discovery.schema import (
    E_ABOVE_HULL,
    MATERIAL_ID,
    REFERENCE_FORMATION_ENERGY,
    DiscoverySubset,
    _validated_reference_frame,
    prediction_series,
)
from ml_peg.data.artifacts import PathLike, read_csv_artifact

MAX_E_FORM_ERROR_THRESHOLD: Final = 5.0
EVALUATION_DECIMALS: Final = 3
MISSING_PREDICTIONS_KEY: Final = "missing_preds"

JsonMetricValue: TypeAlias = float | int | None
DiscoveryResults: TypeAlias = dict[str, dict[str, JsonMetricValue]]


def prepare_discovery_inputs(
    reference: pd.DataFrame,
    predictions: pd.DataFrame | pd.Series,
    *,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
    decimals: int = EVALUATION_DECIMALS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate, align, mask, and round discovery evaluation inputs.

    Errors strictly above the threshold become NaN before all energies are rounded
    to the shared evaluation precision.
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


def _json_safe_metric(value: MetricValue) -> JsonMetricValue:
    """Round a metric and convert non-finite values to JSON null."""
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
    """Evaluate formation-energy predictions on the three discovery subsets.

    Leaderboard mode takes unrounded unique-prototype prevalence; synthetic mode
    derives it from the prepared reference.
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

    results: DiscoveryResults = {}
    for subset in DiscoverySubset:
        subset_index = subset_indices[subset]
        subset_metrics = {
            metric_name: _json_safe_metric(metric_value)
            for metric_name, metric_value in raw_metrics[subset].items()
        }
        subset_metrics[MISSING_PREDICTIONS_KEY] = int(
            prepared_predictions.loc[subset_index].isna().sum()
        )
        results[str(subset)] = subset_metrics
    return results


def evaluate_discovery_paths(
    reference_path: PathLike,
    prediction_path: PathLike,
    *,
    canonical: bool = False,
    uniq_proto_prevalence: float | None = None,
    max_error_threshold: float | None = MAX_E_FORM_ERROR_THRESHOLD,
) -> DiscoveryResults:
    """Load local CSV artifacts and evaluate discovery predictions without writes."""
    return evaluate_discovery(
        read_csv_artifact(reference_path, dtype={MATERIAL_ID: str}),
        read_csv_artifact(prediction_path, dtype={MATERIAL_ID: str}),
        canonical=canonical,
        uniq_proto_prevalence=uniq_proto_prevalence,
        max_error_threshold=max_error_threshold,
    )


def write_discovery_metrics_json(
    results: Mapping[str, Mapping[str, JsonMetricValue]],
    output_path: str | os.PathLike[str],
) -> None:
    """Write discovery metrics as JSON."""
    with open(output_path, mode="w", encoding="utf-8") as file:
        json.dump(results, file, allow_nan=False, indent=2, sort_keys=True)
        file.write("\n")
