"""Analyse the water oxygen-oxygen radial distribution benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.io import load_model_output_from_disk
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegWaterRadialDistributionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegWaterRadialDistributionBenchmark.name
WATERBOX_N500 = "water_box_n500_eq.pdb"
REFERENCE_DATA = "experimental_reference.npz"

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "water_radial_distribution" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "water_radial_distribution"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def _data_input_dir() -> Path:
    """
    Download and return the benchmark input data directory.

    Returns
    -------
    Path
        Directory containing the extracted water RDF input data.
    """
    return download_s3_data(
        key="inputs/molecular_dynamics/water_radial_distribution/water_radial_distribution.zip",
        filename="water_radial_distribution.zip",
    )


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``WaterRadialDistributionResult``.
    """
    data_input_dir = _data_input_dir()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegWaterRadialDistributionBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegWaterRadialDistributionBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> None:
    """Write the combined element set to ``info.json`` for filtering."""
    data_input_dir = _data_input_dir()
    write_struct_info(
        data_path=data_input_dir / BENCHMARK / WATERBOX_N500,
        out_path=OUT_PATH,
    )


@pytest.fixture
@plot_scatter(
    title="Water O-O radial distribution function",
    x_label="r / Å",
    y_label="g(r)",
    show_line=True,
    show_markers=False,
    filename=str(OUT_PATH / "figure_rdf.json"),
)
def rdf_profiles(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get predicted and reference O-O radial distribution profiles.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``WaterRadialDistributionResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Reference and per-model ``(radii, g(r))`` profiles.
    """
    reference = np.load(_data_input_dir() / BENCHMARK / REFERENCE_DATA)
    results = {"ref": (reference["r_OO"].tolist(), reference["g_OO"].tolist())}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        results[model_name] = (result.radii, result.rdf)
    return results


@pytest.fixture
def get_peak_deviation(analyze_results) -> dict[str, float]:
    """
    Get the first solvent peak deviation for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``WaterRadialDistributionResult``.

    Returns
    -------
    dict[str, float]
        Deviation of the first solvent peak from the reference range in Angstrom.
    """
    return {
        model_name: result.peak_deviation
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_rmse(analyze_results) -> dict[str, float]:
    """
    Get the RDF profile RMSE for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``WaterRadialDistributionResult``.

    Returns
    -------
    dict[str, float]
        RMSE of the radial distribution function against the reference.
    """
    return {model_name: result.rmse for model_name, result in analyze_results.items()}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "water_radial_distribution_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    rdf_profiles,
    get_peak_deviation: dict[str, float],
    get_rmse: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rdf_profiles
        Reference and predicted RDF profiles (triggers the RDF plot).
    get_peak_deviation
        First solvent peak deviations for all models.
    get_rmse
        RDF profile RMSEs for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Peak Deviation": get_peak_deviation,
        "RDF RMSE": get_rmse,
    }


def test_water_radial_distribution(metrics: dict[str, dict], struct_info: None) -> None:
    """
    Run water radial distribution analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Water RDF metric results provided by fixtures.
    struct_info : None
        Element info written to ``info.json`` for filtering.
    """
