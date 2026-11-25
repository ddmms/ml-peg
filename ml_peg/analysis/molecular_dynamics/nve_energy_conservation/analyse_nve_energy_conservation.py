"""Analyse the NVE energy conservation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.nve_energy_conservation.nve_energy_conservation import (
    SIMULATION_CONFIG,
    SYSTEMS,
    NVEEnergyConservationResult,
    NVEStructureResult,
)

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import check_dataset, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import (
    MlPegNVEEnergyConservationBenchmark,
    load_model_output,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

BENCHMARK = MlPegNVEEnergyConservationBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "nve_energy_conservation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "nve_energy_conservation"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_MEV = 1000.0

# Snapshots a trajectory must reach to count as complete, in the standard run mode the
# calculation uses. A full run records the initial frame too, so produces one more.
MIN_COMPLETE_FRAMES = (
    SIMULATION_CONFIG["num_steps"] // SIMULATION_CONFIG["snapshot_interval"]
)


def check_datasets() -> None:
    """Check the input structures saved by the calculation are available."""
    for spec in SYSTEMS:
        check_dataset(CALC_PATH / BENCHMARK / spec.filename)


def fitted(result: NVEEnergyConservationResult) -> list[NVEStructureResult]:
    """
    Get the systems mlipaudit was able to fit a drift curve to.

    Skipped systems have no score and failed systems have no drift statistics, so both
    are excluded. A trajectory that stopped early is still fitted, so is included.

    Parameters
    ----------
    result
        Result of the mlipaudit analysis for a single model.

    Returns
    -------
    list[NVEStructureResult]
        Per-system results with a drift curve, however short.
    """
    return [
        structure
        for structure in result.structure_results
        if not structure.skipped
        and not structure.failed
        and structure.energy_drift_ratio is not None
    ]


def completed(result: NVEEnergyConservationResult) -> list[NVEStructureResult]:
    """
    Get the systems that ran to the end of their trajectory.

    A trajectory that stopped early is fitted over a shorter window, so would be
    flattered by being averaged in alongside full ones.

    Parameters
    ----------
    result
        Result of the mlipaudit analysis for a single model.

    Returns
    -------
    list[NVEStructureResult]
        Per-system results with usable drift statistics.
    """
    return [
        structure
        for structure in fitted(result)
        if structure.num_frames >= MIN_COMPLETE_FRAMES
    ]


@pytest.fixture
def analyze_results() -> dict[str, NVEEnergyConservationResult]:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict[str, NVEEnergyConservationResult]
        Mapping of model name to its ``NVEEnergyConservationResult``.
    """
    check_datasets()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegNVEEnergyConservationBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output(
            CALC_PATH / model_name, MlPegNVEEnergyConservationBenchmark
        )
        result = benchmark.analyze()

        # mlipaudit scores a failed system 0 whether it exploded or the model errored
        # on it. Only an explosion leaves a trajectory, so drop the score without one.
        for structure, times in zip(
            result.structure_results, benchmark.model_output.times_ps, strict=True
        ):
            if times is None:
                structure.score = None

        results[model_name] = result
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write per-system element info to ``info.json`` for filtering.

    Elements are stored as one list per system, so individual systems can be
    excluded once partial filtering is supported. The order follows mlipaudit's
    ``SYSTEMS``, matching the order of the results from ``analyze()``.

    Returns
    -------
    dict
        Mapping with the per-system lists of elements.
    """
    check_datasets()

    info = {
        "systems": [spec.name for spec in SYSTEMS],
        "elements": [
            sorted(
                set(read(CALC_PATH / BENCHMARK / spec.filename).get_chemical_symbols())
            )
            for spec in SYSTEMS
        ],
    }

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)

    return info


def plot_drift(model_name: str, result: NVEEnergyConservationResult) -> None:
    """
    Plot the total energy drift along the trajectory for each system.

    The drift is reported per atom, so systems spanning 46 to 2642 atoms can share
    an axis, and keeps its sign, which distinguishes a model that heats up from one
    that cools down.

    The fitted intercept is subtracted, so systems that relax at the start are not
    left offset from the zero line. Only the slope is scored, which a shift cannot
    change.

    Trajectories that stopped early are plotted too, labelled as such, as a model
    blowing up is what the plot is most useful for. No figure is written for a model
    with nothing to plot.

    Parameters
    ----------
    model_name
        Name of MLIP.
    result
        Result of the mlipaudit analysis for this model.
    """
    complete = {structure.structure_name for structure in completed(result)}

    drifts = {
        structure.structure_name
        + ("" if structure.structure_name in complete else " (stopped early)"): [
            structure.times_ps,
            [
                (drift - structure.intercept_ev) / structure.num_atoms * EV_TO_MEV
                for drift in structure.energy_drift_ev
            ],
        ]
        for structure in fitted(result)
        if structure.num_atoms
    }

    # Writing an empty figure would show blank axes rather than "No data available"
    if not drifts:
        return

    @plot_scatter(
        filename=OUT_PATH / f"{model_name}_drift_scatter.json",
        title=f"<b>{model_name} NVE MD</b>",
        x_label="Time / ps",
        y_label="Total energy drift, offset corrected / meV/atom",
        show_line=True,
        show_markers=False,
        hovertemplate=(
            "<b>Time: </b>%{x:.3g} ps<br><b>Drift: </b>%{y:.3g} meV/atom<br>"
        ),
        horizontal_lines=[{"y": 0.0, "name": "Perfect conservation"}],
    )
    def plot_result() -> dict[str, list[list[float]]]:
        """
        Plot the per-system drift curves.

        Returns
        -------
        dict[str, list[list[float]]]
            Times and offset-corrected per-atom drifts for each system.
        """
        return drifts

    plot_result()


@pytest.fixture
def drift_plots(analyze_results: dict[str, NVEEnergyConservationResult]) -> None:
    """
    Write a drift plot for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    for model_name, result in analyze_results.items():
        plot_drift(model_name, result)


@pytest.fixture
def get_energy_drift(
    analyze_results: dict[str, NVEEnergyConservationResult],
) -> dict[str, float]:
    """
    Get the total energy drift rate per atom for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.

    Returns
    -------
    dict[str, float]
        Drift rate of the total energy, in meV/atom/ps.
    """
    results = {}
    for model_name, result in analyze_results.items():
        drifts = [
            abs(structure.drift_slope_ev_per_ps) / structure.num_atoms * EV_TO_MEV
            for structure in completed(result)
            if structure.num_atoms
        ]
        results[model_name] = float(np.mean(drifts)) if drifts else np.nan

    return results


@pytest.fixture
def get_drift_ratio(
    analyze_results: dict[str, NVEEnergyConservationResult],
) -> dict[str, float]:
    """
    Get the energy drift to kinetic energy fluctuation ratio for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.

    Returns
    -------
    dict[str, float]
        Drift ratio averaged over the systems, dimensionless.
    """
    results = {}
    for model_name, result in analyze_results.items():
        ratios = [structure.energy_drift_ratio for structure in completed(result)]
        results[model_name] = float(np.mean(ratios)) if ratios else np.nan

    return results


@pytest.fixture
def get_systems_completed(
    analyze_results: dict[str, NVEEnergyConservationResult],
) -> dict[str, int]:
    """
    Get the number of systems each model simulated successfully.

    The averaged drift metrics only cover the systems that ran, so this records how
    many of them there were.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.

    Returns
    -------
    dict[str, int]
        Number of systems with usable drift statistics.
    """
    return {
        model_name: len(completed(result))
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_score(
    analyze_results: dict[str, NVEEnergyConservationResult],
) -> dict[str, float]:
    """
    Get the energy conservation score for each model.

    A trajectory that stopped early scores 0, as it is fitted over a shorter window,
    so would otherwise score highly for having drifted less. A model that errored on
    any system has no score at all, as averaging over the rest would compare it
    against models that ran the whole benchmark.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.

    Returns
    -------
    dict[str, float]
        Mean of the per-system soft threshold scores, between 0 and 1.
    """
    scores = {}
    for model_name, result in analyze_results.items():
        scorable = [
            structure for structure in result.structure_results if not structure.skipped
        ]
        if not scorable or any(structure.score is None for structure in scorable):
            scores[model_name] = np.nan
            continue

        system_scores = [
            0.0 if structure.num_frames < MIN_COMPLETE_FRAMES else structure.score
            for structure in scorable
        ]
        scores[model_name] = float(np.mean(system_scores))

    return scores


@pytest.fixture
def get_mlipaudit_score(
    analyze_results: dict[str, NVEEnergyConservationResult],
) -> dict[str, float]:
    """
    Get mlipaudit's own benchmark score for each model.

    Reported unweighted, so ML-PEG stays comparable with the MLIP Audit leaderboard
    without its conventions, of scoring an errored system 0 and fitting a trajectory
    that stopped early over its shorter window, affecting ML-PEG's score.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.

    Returns
    -------
    dict[str, float]
        Score mlipaudit reports for the benchmark, between 0 and 1.
    """
    return {
        model_name: (result.score if result.score is not None else np.nan)
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "nve_energy_conservation_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    drift_plots: None,
    get_energy_drift: dict[str, float],
    get_drift_ratio: dict[str, float],
    get_systems_completed: dict[str, int],
    get_score: dict[str, float],
    get_mlipaudit_score: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    drift_plots
        Triggers writing the per-model drift plots.
    get_energy_drift
        Energy drift rates for all models.
    get_drift_ratio
        Energy drift ratios for all models.
    get_systems_completed
        Number of successfully simulated systems for all models.
    get_score
        Energy conservation scores for all models.
    get_mlipaudit_score
        MLIP Audit's own benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Energy Drift": get_energy_drift,
        "Energy Drift Ratio": get_drift_ratio,
        "Systems Completed": get_systems_completed,
        "NVE Score": get_score,
        "MLIP Audit Score": get_mlipaudit_score,
    }


@pytest.mark.framework("mlip_audit")
def test_nve_energy_conservation(metrics: dict[str, dict], struct_info: dict) -> None:
    """
    Run NVE energy conservation analysis.

    Parameters
    ----------
    metrics
        NVE energy conservation metric results provided by fixtures.
    struct_info
        Element info written to ``info.json`` for filtering.
    """
