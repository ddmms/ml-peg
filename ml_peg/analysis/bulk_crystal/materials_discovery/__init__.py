"""Materials-discovery schemas, metrics, and evaluation."""

from __future__ import annotations

from ml_peg.analysis.bulk_crystal.materials_discovery.evaluation import (
    EVALUATION_DECIMALS,
    MAX_E_FORM_ERROR_THRESHOLD,
    MISSING_PREDICTIONS_KEY,
    DiscoveryResults,
    evaluate_discovery,
    evaluate_discovery_paths,
    prepare_discovery_inputs,
    write_discovery_metrics_json,
)
from ml_peg.analysis.bulk_crystal.materials_discovery.metrics import (
    MOST_STABLE_COUNT,
    STABILITY_THRESHOLD,
    MetricValue,
    align_predictions,
    calc_discovery_metrics,
    classify_stable,
    discovery_subset_indices,
    stable_metrics,
)
from ml_peg.analysis.bulk_crystal.materials_discovery.schema import (
    E_ABOVE_HULL,
    MATERIAL_ID,
    PREDICTED_FORMATION_ENERGY,
    REFERENCE_COLUMNS,
    REFERENCE_FORMATION_ENERGY,
    UNIQUE_PROTOTYPE,
    DiscoverySubset,
    validate_prediction_frame,
    validate_reference_frame,
)

__all__ = [
    "EVALUATION_DECIMALS",
    "MAX_E_FORM_ERROR_THRESHOLD",
    "MISSING_PREDICTIONS_KEY",
    "MOST_STABLE_COUNT",
    "STABILITY_THRESHOLD",
    "DiscoveryResults",
    "DiscoverySubset",
    "MetricValue",
    "align_predictions",
    "calc_discovery_metrics",
    "classify_stable",
    "discovery_subset_indices",
    "evaluate_discovery",
    "evaluate_discovery_paths",
    "prepare_discovery_inputs",
    "stable_metrics",
    "validate_prediction_frame",
    "validate_reference_frame",
    "write_discovery_metrics_json",
    "E_ABOVE_HULL",
    "MATERIAL_ID",
    "PREDICTED_FORMATION_ENERGY",
    "REFERENCE_COLUMNS",
    "REFERENCE_FORMATION_ENERGY",
    "UNIQUE_PROTOTYPE",
]
