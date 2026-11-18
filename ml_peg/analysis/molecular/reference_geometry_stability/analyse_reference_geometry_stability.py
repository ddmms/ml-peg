"""Analyse the reference geometry stability benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase.calculators.calculator import Calculator
import numpy as np
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
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegReferenceGeometryStabilityBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular" / "reference_geometry_stability" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "reference_geometry_stability"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@pytest.fixture
def reference_benchmark():
    """
    Load both reference subsets from staged outputs or the benchmark archive.

    Returns
    -------
    MlPegReferenceGeometryStabilityBenchmark
        Benchmark configured with the available reference dataset.
    """
    filenames = (rgs.OPENFF_NEUTRAL_FILENAME, rgs.OPENFF_CHARGED_FILENAME)
    data_root = CALC_PATH
    if not all((data_root / BENCHMARK / name).is_file() for name in filenames):
        data_root = download_s3_data(
            key=f"inputs/molecular/{BENCHMARK}/{BENCHMARK}.zip",
            filename=f"{BENCHMARK}.zip",
        )
    return MlPegReferenceGeometryStabilityBenchmark(
        force_field=Calculator(), data_input_dir=data_root, run_mode="standard"
    )


@pytest.fixture
def analyze_results(reference_benchmark) -> dict:
    """
    Run the mlipaudit analysis for each model.

    Parameters
    ----------
    reference_benchmark
        Benchmark initialized with both OpenFF reference subsets.

    Returns
    -------
    dict
        MLIP Audit results by model, or None if analysis failed.
    """
    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        try:
            reference_benchmark.model_output = load_model_output_from_disk(
                CALC_PATH / model_name, MlPegReferenceGeometryStabilityBenchmark
            )
            results[model_name] = reference_benchmark.analyze()
        except Exception as exc:
            warn(f"Cannot score {model_name}: {exc}", stacklevel=2)
            results[model_name] = None
    return results


@pytest.fixture
def struct_info(reference_benchmark) -> dict:
    """
    Write the combined element set to ``info.json`` for filtering.

    Parameters
    ----------
    reference_benchmark
        Benchmark initialized with both OpenFF reference subsets.

    Returns
    -------
    dict
        Mapping with the sorted list of elements present in the dataset.
    """
    benchmark = reference_benchmark
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
    bins=40,
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
        MLIP Audit results by model, or None if analysis failed.

    Returns
    -------
    dict[str, list]
        RMSD values (Angstrom) across both datasets for each model.
    """
    results = {}
    for model_name, result in analyze_results.items():
        if result is None:
            continue
        values = result.openff_neutral.rmsd_values + result.openff_charged.rmsd_values
        successful = [value for value in values if value is not None]
        if successful:
            results[model_name] = successful
    return results


@pytest.fixture
def get_avg_rmsd(analyze_results) -> dict[str, float]:
    """
    Get the average heavy-atom RMSD for each model.

    Parameters
    ----------
    analyze_results
        MLIP Audit results by model, or None if analysis failed.

    Returns
    -------
    dict[str, float]
        Average RMSD in Angstrom for each model.
    """
    return {
        model_name: result.avg_rmsd if result is not None else np.nan
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_score(analyze_results) -> dict[str, float]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        MLIP Audit results by model, or None if analysis failed.

    Returns
    -------
    dict[str, float]
        The mlipaudit per-molecule soft-threshold RMSD score (0 to 1) for each model.
    """
    return {
        model_name: result.score if result is not None else np.nan
        for model_name, result in analyze_results.items()
    }


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


@pytest.mark.framework("mlip_audit")
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
