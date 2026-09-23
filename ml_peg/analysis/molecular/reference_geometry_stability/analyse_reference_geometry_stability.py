"""Analyse the reference geometry stability benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.reference_geometry_stability import (
    reference_geometry_stability as rgs,
)
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table, plot_hist
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegReferenceGeometryStabilityBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

EXAMPLE_INPUT_FILENAME = rgs.OPENFF_NEUTRAL_FILENAME

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegReferenceGeometryStabilityBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular" / "reference_geometry_stability" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "reference_geometry_stability"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def check_dataset() -> None:
    """
    Check the dataset saved by the calculation is available.

    The calculation copies and unzips the downloaded dataset into its outputs,
    so the analysis does not need to download the input data again.

    Raises
    ------
    ValueError
        If the dataset is missing from the calculation outputs.
    """
    dataset_path = CALC_PATH / BENCHMARK / EXAMPLE_INPUT_FILENAME
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist. Please run the calculation.")


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``ReferenceGeometryStabilityResult``.
    """
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegReferenceGeometryStabilityBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegReferenceGeometryStabilityBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write the combined element set to ``info.json`` for filtering.

    Returns
    -------
    dict
        Mapping with the sorted list of elements present in the dataset.
    """
    check_dataset()

    benchmark = MlPegReferenceGeometryStabilityBenchmark(
        force_field=Calculator(),
        data_input_dir=CALC_PATH,
        run_mode="standard",
    )
    elements = sorted(
        {
            symbol
            for dataset in (
                benchmark._openff_neutral_dataset,
                benchmark._openff_charged_dataset,
            )
            for molecule in dataset.values()
            for symbol in molecule.atom_symbols
        }
    )
    info = {"elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(json.dumps(info, indent=1))
    return info


@pytest.fixture
@plot_hist(
    filename=OUT_PATH / "figure_rmsd_histogram.json",
    title="Heavy-atom RMSD after minimization",
    x_label="RMSD / Å",
    y_label="Probability density",
)
def rmsd_histogram(analyze_results) -> dict[str, list]:
    """
    Get per-molecule heavy-atom RMSD values for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReferenceGeometryStabilityResult``.

    Returns
    -------
    dict[str, list]
        RMSD values (Angstrom) across both datasets for each model.
    """
    results = {}
    for model_name, result in analyze_results.items():
        rmsd_values = (
            result.openff_neutral.rmsd_values + result.openff_charged.rmsd_values
        )
        results[model_name] = [rmsd for rmsd in rmsd_values if rmsd is not None]
    return results


@pytest.fixture
def get_avg_rmsd(analyze_results) -> dict[str, float]:
    """
    Get the average heavy-atom RMSD for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReferenceGeometryStabilityResult``.

    Returns
    -------
    dict[str, float]
        Average RMSD in Angstrom for each model.
    """
    return {
        model_name: result.avg_rmsd for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_score(analyze_results) -> dict[str, float]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReferenceGeometryStabilityResult``.

    Returns
    -------
    dict[str, float]
        The mlipaudit per-molecule soft-threshold RMSD score (0 to 1) for each model.
    """
    return {model_name: result.score for model_name, result in analyze_results.items()}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "reference_geometry_stability_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    rmsd_histogram,
    get_avg_rmsd: dict[str, float],
    get_score: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rmsd_histogram
        Per-molecule RMSD values (triggers the histogram plot).
    get_avg_rmsd
        Average RMSDs for all models.
    get_score
        The mlipaudit benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Avg RMSD": get_avg_rmsd,
        "Geometry Score": get_score,
    }


def test_reference_geometry_stability(
    metrics: dict[str, dict], struct_info: dict
) -> None:
    """
    Run reference geometry stability analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Reference geometry stability metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
