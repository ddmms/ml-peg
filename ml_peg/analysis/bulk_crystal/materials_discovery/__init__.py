"""Materials-discovery schemas, metrics, and evaluation."""

from __future__ import annotations

# Public re-exports.
# ruff: noqa: F401
from ml_peg.analysis.bulk_crystal.materials_discovery.evaluation import (
    EVALUATION_DECIMALS,
    MAX_E_FORM_ERROR_THRESHOLD,
    MISSING_PREDICTIONS_KEY,
    RESULT_SCHEMA_VERSION,
    DiscoveryResults,
    DiscoverySubsetResults,
    SourceMetadata,
    calc_discovery_metrics,
    discovery_subset_indices,
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
    classify_stable,
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
