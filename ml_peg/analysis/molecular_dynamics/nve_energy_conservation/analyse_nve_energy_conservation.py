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
    SYSTEMS,
    NVEEnergyConservationBenchmark,
    NVEEnergyConservationModelOutput,
    NVEEnergyConservationResult,
    NVEStructureResult,
)

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = NVEEnergyConservationBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "nve_energy_conservation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "nve_energy_conservation"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_MEV = 1000.0


def check_dataset() -> None:
    """
    Check the input structures saved by the calculation are available.

    The calculation copies the downloaded structures into its outputs, so the
    analysis does not need to download the input data again.

    Raises
    ------
    ValueError
        If any structure is missing from the calculation outputs.
    """
    for spec in SYSTEMS:
        structure_path = CALC_PATH / BENCHMARK / spec.filename
        if not structure_path.exists():
            raise ValueError(
                f"{structure_path} does not exist. Please run the calculation."
            )


def completed(result: NVEEnergyConservationResult) -> list[NVEStructureResult]:
    """
    Get the systems that were simulated and analysed successfully.

    Skipped systems have no score, and failed systems have no drift statistics, so
    both are excluded from the averaged metrics and from the drift plot.

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
        for structure in result.structure_results
        if not structure.skipped
        and not structure.failed
        and structure.energy_drift_ratio is not None
    ]


@pytest.fixture
def analyze_results() -> dict[str, NVEEnergyConservationResult]:
    """
    Run the mlipaudit analysis for each model.

    The model output is read from the JSON written by the calculation, as in the
    Folmsbee and tautomers benchmarks. It holds only plain lists, so no data is
    lost by writing it as JSON rather than as an npz archive.

    Returns
    -------
    dict[str, NVEEnergyConservationResult]
        Mapping of model name to its ``NVEEnergyConservationResult``.
    """
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_path = CALC_PATH / model_name / "model_output.json"
        if not output_path.exists():
            continue
        benchmark = NVEEnergyConservationBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = NVEEnergyConservationModelOutput.model_validate_json(
            output_path.read_text()
        )
        results[model_name] = benchmark.analyze()
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
    check_dataset()

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

    Parameters
    ----------
    model_name
        Name of MLIP.
    result
        Result of the mlipaudit analysis for this model.
    """
    drifts = {
        structure.structure_name: [
            structure.times_ps,
            [
                drift / structure.num_atoms * EV_TO_MEV
                for drift in structure.energy_drift_ev
            ],
        ]
        for structure in completed(result)
        if structure.num_atoms
    }

    @plot_scatter(
        filename=OUT_PATH / f"{model_name}_drift_scatter.json",
        title=f"<b>{model_name} NVE MD</b>",
        x_label="Time / ps",
        y_label="Total energy drift / meV/atom",
        show_line=True,
        show_markers=False,
        horizontal_lines=[{"y": 0.0, "name": "Perfect conservation"}],
    )
    def plot_result() -> dict[str, list[list[float]]]:
        """
        Plot the per-system drift curves.

        Returns
        -------
        dict[str, list[list[float]]]
            Times and per-atom drifts for each system.
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
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``NVEEnergyConservationResult``.

    Returns
    -------
    dict[str, float]
        Mean of the per-system soft threshold scores, between 0 and 1.
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
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    drift_plots: None,
    get_energy_drift: dict[str, float],
    get_drift_ratio: dict[str, float],
    get_systems_completed: dict[str, int],
    get_score: dict[str, float],
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
        MLIP Audit benchmark scores for all models.

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
    }


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
