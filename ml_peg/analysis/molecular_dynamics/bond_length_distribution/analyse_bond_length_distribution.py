"""Analyse the covalent bond length distribution benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.bond_length_distribution.bond_length_distribution import (
    BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME,
)
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table, plot_hist
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegBondLengthDistributionBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegBondLengthDistributionBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "bond_length_distribution" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "bond_length_distribution"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def check_dataset() -> None:
    """
    Check the dataset saved by the calculation is available.

    The calculation copies the downloaded dataset into its outputs, so the
    analysis does not need to download the input data again.

    Raises
    ------
    ValueError
        If the dataset is missing from the calculation outputs.
    """
    dataset_path = CALC_PATH / BENCHMARK / BOND_LENGTH_DISTRIBUTION_DATASET_FILENAME
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist. Please run the calculation.")


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``BondLengthDistributionResult``.
    """
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegBondLengthDistributionBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegBondLengthDistributionBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write per-molecule element info to ``info.json`` for filtering.

    Elements are stored as one list per molecule, so individual molecules can be
    excluded once partial filtering is supported. The order follows the dataset,
    matching the order of the molecules in ``analyze()``'s results.

    Returns
    -------
    dict
        Mapping with the per-molecule lists of elements.
    """
    check_dataset()

    benchmark = MlPegBondLengthDistributionBenchmark(
        force_field=Calculator(),
        data_input_dir=CALC_PATH,
        run_mode="standard",
    )
    data = benchmark._bond_length_distribution_data
    info = {
        "molecules": list(data),
        "elements": [sorted(set(molecule.atom_symbols)) for molecule in data.values()],
    }

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)

    return info


@pytest.fixture
@plot_hist(
    filename=str(OUT_PATH / "figure_bond_length_hist.json"),
    title="Bond length deviation distribution",
    x_label="Bond length deviation / Å",
    y_label="Probability density",
    bins=50,
)
def deviation_distributions(analyze_results) -> dict[str, np.ndarray]:
    """
    Collect the bond length deviations sampled along each model's trajectories.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``BondLengthDistributionResult``.

    Returns
    -------
    dict[str, np.ndarray]
        Per-model flat array of bond length deviations across all molecules.
    """
    results = {}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        deviations = [
            value
            for molecule in result.molecules
            if molecule.deviation_trajectory is not None
            for value in molecule.deviation_trajectory
        ]
        if deviations:
            results[model_name] = np.array(deviations)
    return results


@pytest.fixture
def get_avg_deviation(analyze_results) -> dict[str, float]:
    """
    Get the average bond length deviation for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``BondLengthDistributionResult``.

    Returns
    -------
    dict[str, float]
        Mean absolute bond length deviation over the trajectories, in Angstrom.
    """
    return {
        model_name: (
            result.avg_deviation if result.avg_deviation is not None else np.nan
        )
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "bond_length_distribution_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    deviation_distributions,
    get_avg_deviation: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    deviation_distributions
        Per-model deviation arrays (triggers the histogram plot).
    get_avg_deviation
        Average bond length deviations for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Bond Length Deviation": get_avg_deviation,
    }


def test_bond_length_distribution(metrics: dict[str, dict], struct_info: dict) -> None:
    """
    Run bond length distribution analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Bond length metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
