"""Analyse the protein conformational sampling benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.io import load_model_output_from_disk
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegSamplingBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegSamplingBenchmark.name

CALC_PATH = CALCS_ROOT / "biomolecules" / "protein_sampling" / "outputs"
OUT_PATH = APP_ROOT / "data" / "biomolecules" / "protein_sampling"

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
        Directory containing the extracted protein sampling input data.
    """
    return download_s3_data(
        key="inputs/biomolecules/protein_sampling/protein_sampling.zip",
        filename="protein_sampling.zip",
    )


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``SamplingResult``.
    """
    data_input_dir = _data_input_dir()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegSamplingBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegSamplingBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> None:
    """Write the combined element set to ``info.json`` for filtering."""
    elements = sorted(MlPegSamplingBenchmark.required_elements)

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump({"elements": elements}, f, indent=1)


@pytest.fixture
def get_rmsd_backbone(analyze_results) -> dict[str, float | None]:
    """
    Get the mean backbone dihedral distribution RMSD for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SamplingResult``.

    Returns
    -------
    dict[str, float | None]
        Backbone dihedral distribution RMSD averaged over residues and systems.
    """
    return {
        model_name: result.rmsd_backbone_total
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_hellinger_backbone(analyze_results) -> dict[str, float | None]:
    """
    Get the mean backbone dihedral Hellinger distance for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SamplingResult``.

    Returns
    -------
    dict[str, float | None]
        Backbone dihedral Hellinger distance averaged over residues and systems.
    """
    return {
        model_name: result.hellinger_distance_backbone_total
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_outliers_ratio_backbone(analyze_results) -> dict[str, float | None]:
    """
    Get the mean backbone dihedral outliers ratio for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SamplingResult``.

    Returns
    -------
    dict[str, float | None]
        Fraction of sampled backbone dihedrals lying far from the reference data,
        averaged over residues and systems.
    """
    return {
        model_name: result.outliers_ratio_backbone_total
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "protein_sampling_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    get_rmsd_backbone: dict[str, float | None],
    get_hellinger_backbone: dict[str, float | None],
    get_outliers_ratio_backbone: dict[str, float | None],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_rmsd_backbone
        Backbone dihedral distribution RMSD for all models.
    get_hellinger_backbone
        Backbone dihedral Hellinger distance for all models.
    get_outliers_ratio_backbone
        Backbone dihedral outliers ratio for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Backbone Dihedral RMSD": get_rmsd_backbone,
        "Backbone Hellinger Distance": get_hellinger_backbone,
        "Backbone Outliers Ratio": get_outliers_ratio_backbone,
    }


def test_protein_sampling(metrics: dict[str, dict], struct_info: None) -> None:
    """
    Run protein sampling analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Protein sampling metric results provided by fixtures.
    struct_info : None
        Element info written to ``info.json`` for filtering.
    """
