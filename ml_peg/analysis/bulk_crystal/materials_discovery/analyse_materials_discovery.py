"""Analyse Matbench Discovery formation-energy predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_peg.analysis.bulk_crystal.materials_discovery import (
    DiscoveryResults,
    evaluate_discovery_paths,
)
from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT

CATEGORY = "bulk_crystal"
BENCHMARK = "materials_discovery"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK
REFERENCE_PATH = OUT_PATH / "reference" / "2023-12-13-wbm-summary.csv.gz"
PREDICTION_PATHS = {
    "mace-mp-0a": OUT_PATH / "mace-mp-0" / "2023-12-11-discovery.csv.gz",
}

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

METRIC_FIELDS = {
    "Full F1": ("full_test_set", "F1"),
    "Full DAF": ("full_test_set", "DAF"),
    "Full MAE": ("full_test_set", "MAE"),
    "Unique F1": ("unique_prototypes", "F1"),
    "Unique DAF": ("unique_prototypes", "DAF"),
    "Unique MAE": ("unique_prototypes", "MAE"),
    "10k F1": ("most_stable_10k", "F1"),
    "10k DAF": ("most_stable_10k", "DAF"),
    "10k MAE": ("most_stable_10k", "MAE"),
}


@pytest.fixture(scope="session")
def discovery_results() -> dict[str, DiscoveryResults]:
    """
    Evaluate each available model prediction artifact.

    Returns
    -------
    dict[str, DiscoveryResults]
        Discovery results keyed by ML-PEG model name.
    """
    reference = pd.read_csv(REFERENCE_PATH)
    unique_prototypes = reference["unique_prototype"].astype(bool)
    prevalence = float(
        (
            reference.loc[
                unique_prototypes,
                "e_above_hull_mp2020_corrected_ppd_mp",
            ]
            <= 0
        ).mean()
    )

    return {
        model: evaluate_discovery_paths(
            REFERENCE_PATH,
            prediction_path,
            canonical=True,
            uniq_proto_prevalence=prevalence,
        )
        for model, prediction_path in PREDICTION_PATHS.items()
        if prediction_path.is_file()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "materials_discovery_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(discovery_results: dict[str, DiscoveryResults]) -> dict[str, dict]:
    """
    Convert discovery subset results to an ML-PEG metrics table.

    Parameters
    ----------
    discovery_results
        Discovery results keyed by ML-PEG model name.

    Returns
    -------
    dict[str, dict]
        Metric values keyed first by metric and then by model.
    """
    return {
        label: {
            model: result["subsets"][subset][metric]
            for model, result in discovery_results.items()
        }
        for label, (subset, metric) in METRIC_FIELDS.items()
    }


@pytest.mark.framework("matbench-discovery")
def test_materials_discovery(metrics: dict[str, dict]) -> None:
    """
    Build the materials-discovery benchmark outputs.

    Parameters
    ----------
    metrics
        Metric values written to the ML-PEG table.
    """
    return
