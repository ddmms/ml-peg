"""Analyse molecular dynamics stability benchmark."""

from __future__ import annotations

from pathlib import Path

from mlipaudit.io import load_model_output_from_disk
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegStabilityBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "stability" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "stability"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@pytest.fixture
def success_rate() -> dict[str, float]:
    """
    Get the fraction of MD simulations that completed without error.

    Returns
    -------
    dict[str, float]
        Fraction of successful simulations for each model.
    """
    results = {}
    for model_name in MODELS:
        output_zip = CALC_PATH / model_name / MlPegStabilityBenchmark.name
        if not (output_zip / "model_output.zip").exists():
            continue
        model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegStabilityBenchmark
        )
        states = model_output.simulation_states
        results[model_name] = sum(s is not None for s in states) / len(states)
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "stability_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(success_rate: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    success_rate
        Fraction of successful simulations for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Success Rate": success_rate,
    }


def test_stability(metrics: dict[str, dict]) -> None:
    """
    Run stability analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Stability metric results provided by fixtures.
    """
