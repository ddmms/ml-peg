"""
Analyse the MOF heat capacity benchmark.

Reads the shared phonon outputs written by
``ml_peg/calcs/porous_materials/mof_phonons/calc_mof_phonons.py`` and scores
the isochoric heat capacity at 300 K against experimental calorimetry, along
with the fraction of imaginary modes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    load_heat_capacity_reference,
    load_phonon_summaries,
    load_reference_structures,
    mean_imaginary_deviation,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

OUT_PATH = APP_ROOT / "data" / "porous_materials" / "mof_heat_capacity"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_CONFIG_PATH)

HEAT_CAPACITY_REFERENCE = load_heat_capacity_reference()

# Frameworks carrying both a reference heat capacity and a benchmark structure.
FRAMEWORKS = sorted(set(HEAT_CAPACITY_REFERENCE) & set(load_reference_structures()))


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_heat_capacity.json",
    title="MOF heat capacity at 300 K",
    x_label="Predicted Cv / J g-1 K-1",
    y_label="Experimental Cv / J g-1 K-1",
    hoverdata={"Framework": FRAMEWORKS},
)
def heat_capacities() -> dict[str, list]:
    """
    Collect reference and predicted heat capacities.

    Returns
    -------
    dict[str, list]
        Reference and per-model heat capacities in J/g/K, one entry per
        scored framework.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {
        "ref": [HEAT_CAPACITY_REFERENCE[name]["value"] for name in FRAMEWORKS]
    }
    for model_name in MODELS:
        summaries = load_phonon_summaries(model_name)
        results[model_name] = [
            summaries.get(name, {}).get("heat_capacity_J_per_g_K", float("nan"))
            for name in FRAMEWORKS
        ]
    return results


@pytest.fixture
def heat_capacity_errors(heat_capacities: dict[str, list]) -> dict[str, float | None]:
    """
    Get the heat capacity MAE for each model.

    Frameworks whose phonon calculation did not complete are dropped rather
    than propagating NaN through the whole metric, so that a model which fails
    on one framework is still scored on the rest.

    Parameters
    ----------
    heat_capacities
        Reference and predicted heat capacities.

    Returns
    -------
    dict[str, float | None]
        Mean absolute error in J/g/K, or ``None`` when a model produced no
        usable heat capacity.
    """
    results: dict[str, float | None] = {}
    reference = heat_capacities["ref"]
    for model_name in MODELS:
        pairs = [
            (ref, pred)
            for ref, pred in zip(reference, heat_capacities[model_name], strict=True)
            if np.isfinite(pred)
        ]
        if not pairs:
            results[model_name] = None
            continue
        results[model_name] = mae([p[0] for p in pairs], [p[1] for p in pairs])
    return results


@pytest.fixture
def imaginary_mode_errors() -> dict[str, float | None]:
    """
    Get the mean deviation of the imaginary-mode percentage from zero.

    Returns
    -------
    dict[str, float | None]
        Mean absolute percentage of imaginary modes for each model.
    """
    return {model_name: mean_imaginary_deviation(model_name) for model_name in MODELS}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "mof_heat_capacity_metrics_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(
    heat_capacity_errors: dict[str, float],
    imaginary_mode_errors: dict[str, float | None],
) -> dict[str, dict]:
    """
    Assemble the heat capacity metrics table.

    Parameters
    ----------
    heat_capacity_errors
        Heat capacity MAE for each model.
    imaginary_mode_errors
        Imaginary-mode percentage MAE for each model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Cv MAE": heat_capacity_errors,
        "Imaginary modes MAE": imaginary_mode_errors,
    }


def test_mof_heat_capacity(
    metrics: dict[str, dict], heat_capacities: dict[str, list]
) -> None:
    """
    Run the MOF heat capacity analysis.

    Parameters
    ----------
    metrics
        All heat capacity metrics.
    heat_capacities
        Reference and predicted heat capacities.
    """
    assert FRAMEWORKS, "No frameworks with a heat-capacity reference were found"
    assert len(heat_capacities["ref"]) == len(FRAMEWORKS)
    assert set(metrics) == {"Cv MAE", "Imaginary modes MAE"}
    assert (OUT_PATH / "mof_heat_capacity_metrics_table.json").exists()
