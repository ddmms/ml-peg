"""Analyse the solvent radial distribution benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.io import load_model_output_from_disk
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegSolventRadialDistributionBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegSolventRadialDistributionBenchmark.name
SOLVENTS = ["CCl4", "methanol", "acetonitrile"]

CALC_PATH = (
    CALCS_ROOT / "molecular_dynamics" / "solvent_radial_distribution" / "outputs"
)
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "solvent_radial_distribution"

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
        Directory containing the extracted solvent RDF input data.
    """
    return download_s3_data(
        key="inputs/molecular_dynamics/solvent_radial_distribution/solvent_radial_distribution.zip",
        filename="solvent_radial_distribution.zip",
    )


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``SolventRadialDistributionResult``.
    """
    data_input_dir = _data_input_dir()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegSolventRadialDistributionBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegSolventRadialDistributionBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> None:
    """Write the combined element set to ``info.json`` for filtering."""
    data_input_dir = _data_input_dir()
    write_struct_info(
        data_path=[
            data_input_dir / BENCHMARK / f"{solvent}_eq.pdb" for solvent in SOLVENTS
        ],
        out_path=OUT_PATH,
    )


@pytest.fixture
@plot_scatter(
    title="Solvent radial distribution functions",
    x_label="r / Å",
    y_label="g(r)",
    show_line=True,
    show_markers=False,
    filename=str(OUT_PATH / "figure_rdf.json"),
)
def rdf_profiles(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get the predicted radial distribution profiles for each solvent.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Per-model and per-solvent ``(radii, g(r))`` profiles.
    """
    results = {}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        for structure in result.structures:
            if structure.failed or structure.radii is None or structure.rdf is None:
                continue
            results[f"{model_name} ({structure.structure_name})"] = (
                structure.radii,
                structure.rdf,
            )
    return results


@pytest.fixture
def get_peak_deviation(analyze_results) -> dict[str, float]:
    """
    Get the average first solvent peak deviation for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.

    Returns
    -------
    dict[str, float]
        Average deviation of the first solvent peak from the reference, in Angstrom.
    """
    return {
        model_name: result.avg_peak_deviation
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "solvent_radial_distribution_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    rdf_profiles,
    get_peak_deviation: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rdf_profiles
        Predicted RDF profiles for all models (triggers the RDF plot).
    get_peak_deviation
        Average first solvent peak deviations for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Peak Deviation": get_peak_deviation,
    }


def test_solvent_radial_distribution(
    metrics: dict[str, dict], struct_info: None
) -> None:
    """
    Run solvent radial distribution analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Solvent RDF metric results provided by fixtures.
    struct_info : None
        Element info written to ``info.json`` for filtering.
    """
