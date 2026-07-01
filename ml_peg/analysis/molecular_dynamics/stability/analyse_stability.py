"""Analyse molecular dynamics stability benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.io import load_model_output_from_disk
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
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


def _fraction_completed(result) -> float:
    """
    Get how far a simulation got before it became unstable.

    Parameters
    ----------
    result
        A ``StabilityStructureResult`` for a single structure.

    Returns
    -------
    float
        Fraction of the trajectory completed: 1.0 if stable, 0.0 if the
        simulation did not run, otherwise the frame at which it exploded or
        drifted divided by the total number of frames.
    """
    if result.failed:
        return 0.0
    if result.exploded_frame != -1:
        return result.exploded_frame / result.num_frames
    if result.drift_frame != -1:
        return result.drift_frame / result.num_frames
    return 1.0


@pytest.fixture
def structure_results() -> dict[str, list]:
    """
    Analyse each stored trajectory into per-structure stability results.

    Returns
    -------
    dict[str, list]
        List of ``StabilityStructureResult`` objects for each model.
    """
    data_input_dir = download_s3_data(
        key="inputs/molecular_dynamics/stability/stability.zip",
        filename="stability.zip",
    )

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / MlPegStabilityBenchmark.name
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegStabilityBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegStabilityBenchmark
        )
        results[model_name] = benchmark.analyze().structure_results
    return results


@pytest.fixture
def success_rate(structure_results: dict[str, list]) -> dict[str, float]:
    """
    Get the fraction of MD simulations that completed without error.

    Parameters
    ----------
    structure_results
        Per-structure stability results for all models.

    Returns
    -------
    dict[str, float]
        Fraction of successful simulations for each model.
    """
    return {
        model_name: sum(not r.failed for r in results) / len(results)
        for model_name, results in structure_results.items()
    }


@pytest.fixture
@plot_scatter(
    title="Trajectory stability",
    x_label="System",
    y_label="Fraction of trajectory completed",
    hovertemplate="<b>%{x}</b><br>Completed: %{y:.2f}<extra>%{fullData.name}</extra>",
    filename=str(OUT_PATH / "stability_progress.json"),
)
def progress(structure_results: dict[str, list]) -> dict[str, tuple[list, list]]:
    """
    Get per-structure trajectory progress for each model.

    For every system, reports the fraction of the trajectory completed before it
    exploded, drifted or failed to run.

    Parameters
    ----------
    structure_results
        Per-structure stability results for all models.

    Returns
    -------
    dict[str, tuple[list, list]]
        Structure names and completed fractions for each model.
    """
    return {
        model_name: (
            [r.structure_name for r in results],
            [_fraction_completed(r) for r in results],
        )
        for model_name, results in structure_results.items()
    }


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


def test_stability(
    metrics: dict[str, dict], progress: dict[str, tuple[list, list]]
) -> None:
    """
    Run stability analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Stability metric results provided by fixtures.
    progress : dict[str, tuple[list, list]]
        Per-structure trajectory progress provided by fixtures.
    """
