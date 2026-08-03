"""Analyse the protein conformational sampling benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.sampling.sampling import STRUCTURE_NAMES
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegSamplingBenchmark
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


def structure_xyz(structure_name: str) -> Path:
    """
    Get the path to a starting structure saved by the calculation.

    Parameters
    ----------
    structure_name
        Name of the structure.

    Returns
    -------
    Path
        Path to the structure's starting geometry.
    """
    return CALC_PATH / BENCHMARK / "starting_structures" / f"{structure_name}.xyz"


def check_dataset() -> None:
    """
    Check the input structures saved by the calculation are available.

    The calculation copies the downloaded input data into its outputs, so the
    analysis does not need to download it again.

    Raises
    ------
    ValueError
        If any starting structure is missing from the calculation outputs.
    """
    for structure_name in STRUCTURE_NAMES:
        if not structure_xyz(structure_name).exists():
            raise ValueError(
                f"{structure_xyz(structure_name)} does not exist. "
                "Please run the calculation."
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
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegSamplingBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegSamplingBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write per-structure element info to ``info.json`` for filtering.

    Elements are stored as one list per structure, so individual structures can
    be excluded once partial filtering is supported. The order follows
    ``STRUCTURE_NAMES``.

    Returns
    -------
    dict
        Mapping with the per-structure lists of elements.
    """
    check_dataset()

    info = {
        "systems": list(STRUCTURE_NAMES),
        "elements": [
            sorted(set(read(structure_xyz(name)).get_chemical_symbols()))
            for name in STRUCTURE_NAMES
        ],
    }

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)

    return info


@pytest.fixture
def get_rmsd_backbone(analyze_results) -> dict[str, float]:
    """
    Get the mean backbone dihedral distribution RMSD for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SamplingResult``.

    Returns
    -------
    dict[str, float]
        Backbone dihedral distribution RMSD averaged over residues and systems.
    """
    return {
        model_name: (
            result.rmsd_backbone_total
            if result.rmsd_backbone_total is not None
            else np.nan
        )
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_hellinger_backbone(analyze_results) -> dict[str, float]:
    """
    Get the mean backbone dihedral Hellinger distance for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SamplingResult``.

    Returns
    -------
    dict[str, float]
        Backbone dihedral Hellinger distance averaged over residues and systems.
    """
    return {
        model_name: (
            result.hellinger_distance_backbone_total
            if result.hellinger_distance_backbone_total is not None
            else np.nan
        )
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_outliers_ratio_backbone(analyze_results) -> dict[str, float]:
    """
    Get the mean backbone dihedral outliers ratio for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SamplingResult``.

    Returns
    -------
    dict[str, float]
        Fraction of sampled backbone dihedrals lying far from the reference data,
        averaged over residues and systems.
    """
    return {
        model_name: (
            result.outliers_ratio_backbone_total
            if result.outliers_ratio_backbone_total is not None
            else np.nan
        )
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
    get_rmsd_backbone: dict[str, float],
    get_hellinger_backbone: dict[str, float],
    get_outliers_ratio_backbone: dict[str, float],
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


def test_protein_sampling(metrics: dict[str, dict], struct_info: dict) -> None:
    """
    Run protein sampling analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Protein sampling metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
