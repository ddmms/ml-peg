"""Analyse Matbench Discovery formation-energy predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_peg.analysis.bulk_crystal.materials_discovery import (
    E_ABOVE_HULL,
    REFERENCE_FORMATION_ENERGY,
    DiscoveryResults,
    evaluate_discovery_paths,
    prepare_discovery_inputs,
)
from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
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
def discovery_plot_data() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Prepare aligned reference and predicted energies for each available model.

    Returns
    -------
    dict[str, tuple[pandas.DataFrame, pandas.Series]]
        Prepared reference data and predictions keyed by ML-PEG model name.
    """
    reference = pd.read_csv(REFERENCE_PATH)
    return {
        model: prepare_discovery_inputs(reference, pd.read_csv(prediction_path))
        for model, prediction_path in PREDICTION_PATHS.items()
        if prediction_path.is_file()
    }


def _density_payload(
    discovery_plot_data: dict[str, tuple[pd.DataFrame, pd.Series]],
    *,
    hull_distance: bool,
) -> dict[str, dict]:
    """
    Build density-plot inputs from prepared discovery data.

    Parameters
    ----------
    discovery_plot_data
        Prepared reference data and predictions keyed by model.
    hull_distance
        Whether to plot hull distances instead of formation energies.

    Returns
    -------
    dict[str, dict]
        Density-plot payloads keyed by model.
    """
    payload: dict[str, dict] = {}
    for model, (reference, predictions) in discovery_plot_data.items():
        valid = predictions.notna()
        reference_formation_energy = reference.loc[valid, REFERENCE_FORMATION_ENERGY]
        predicted_formation_energy = predictions.loc[valid]

        if hull_distance:
            ref_values = reference.loc[valid, E_ABOVE_HULL]
            pred_values = (
                ref_values + predicted_formation_energy - reference_formation_energy
            )
        else:
            ref_values = reference_formation_energy
            pred_values = predicted_formation_energy

        payload[model] = {
            "ref": ref_values.to_numpy(),
            "pred": pred_values.to_numpy(),
            "meta": {
                "systems": int(valid.sum()),
                "without_predictions": int((~valid).sum()),
            },
        }
    return payload


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_formation_energy_density.json",
    title="Formation energy parity",
    x_label="Reference formation energy / eV/atom",
    y_label="Predicted formation energy / eV/atom",
    annotation_metadata={
        "systems": "Systems",
        "without_predictions": "Systems without predictions",
    },
)
def formation_energy_density(
    discovery_plot_data: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, dict]:
    """
    Build the formation-energy density scatter.

    Parameters
    ----------
    discovery_plot_data
        Prepared reference data and predictions keyed by model.

    Returns
    -------
    dict[str, dict]
        Density-plot payloads keyed by model.
    """
    return _density_payload(discovery_plot_data, hull_distance=False)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_hull_distance_density.json",
    title="Energy-above-hull parity",
    x_label="Reference energy above hull / eV/atom",
    y_label="Predicted energy above hull / eV/atom",
    annotation_metadata={
        "systems": "Systems",
        "without_predictions": "Systems without predictions",
    },
)
def hull_distance_density(
    discovery_plot_data: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, dict]:
    """
    Build the energy-above-hull density scatter.

    Parameters
    ----------
    discovery_plot_data
        Prepared reference data and predictions keyed by model.

    Returns
    -------
    dict[str, dict]
        Density-plot payloads keyed by model.
    """
    return _density_payload(discovery_plot_data, hull_distance=True)


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
def test_materials_discovery(
    metrics: dict[str, dict],
    formation_energy_density: dict[str, dict],
    hull_distance_density: dict[str, dict],
) -> None:
    """
    Build the materials-discovery benchmark outputs.

    Parameters
    ----------
    metrics
        Metric values written to the ML-PEG table.
    formation_energy_density
        Formation-energy density-scatter inputs (drives the saved plot).
    hull_distance_density
        Hull-distance density-scatter inputs (drives the saved plot).
    """
    return
