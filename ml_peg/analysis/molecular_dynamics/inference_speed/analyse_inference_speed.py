"""Analyse the inference speed benchmark."""

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
from ml_peg.calcs.utils.mlipaudit import MlPegInferenceSpeedBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegInferenceSpeedBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "inference_speed" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "inference_speed"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

#: Conversion factor from seconds to microseconds.
SECONDS_TO_MICROSECONDS = 1.0e6


def _data_input_dir() -> Path:
    """
    Download and return the benchmark input data directory.

    Returns
    -------
    Path
        Directory containing the extracted inference speed input data.
    """
    return download_s3_data(
        key="inputs/molecular_dynamics/inference_speed/inference_speed.zip",
        filename="inference_speed.zip",
    )


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``InferenceSpeedResult``.
    """
    data_input_dir = _data_input_dir()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegInferenceSpeedBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegInferenceSpeedBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> None:
    """Write the combined element set to ``info.json`` for filtering."""
    data_dir = _data_input_dir() / BENCHMARK
    write_struct_info(
        data_path=sorted(data_dir.glob("*.xyz")),
        out_path=OUT_PATH,
    )


@pytest.fixture
@plot_scatter(
    title="Inference speed scaling",
    x_label="Number of atoms",
    y_label="Forward pass time / s",
    show_line=True,
    show_markers=True,
    filename=str(OUT_PATH / "figure_inference_speed_scaling.json"),
)
def scaling_curves(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get the forward-pass time scaling curve for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``InferenceSpeedResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Per-model ``(num_atoms, forward_time)`` curves, sorted by system size and
        restricted to structures with a successful forward pass.
    """
    curves = {}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        points = [
            (structure.num_atoms, structure.average_forward_time)
            for structure in result.structures
            if structure.average_forward_time is not None
        ]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        num_atoms, forward_times = zip(*points, strict=True)
        curves[model_name] = (list(num_atoms), list(forward_times))
    return curves


@pytest.fixture
def get_forward_time_per_atom(analyze_results) -> dict[str, float]:
    """
    Get the mean per-atom forward-pass time for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``InferenceSpeedResult``.

    Returns
    -------
    dict[str, float]
        Mean over structures of ``average_forward_time / num_atoms``, in
        microseconds per atom. Structures without a successful forward pass are
        skipped.
    """
    per_atom_times = {}
    for model_name, result in analyze_results.items():
        values = [
            structure.average_forward_time / structure.num_atoms
            for structure in result.structures
            if structure.average_forward_time is not None
        ]
        if not values:
            continue
        per_atom_times[model_name] = sum(values) / len(values) * SECONDS_TO_MICROSECONDS
    return per_atom_times


@pytest.fixture
@build_table(
    filename=OUT_PATH / "inference_speed_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    scaling_curves,
    get_forward_time_per_atom: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    scaling_curves
        Forward-pass time scaling curves for all models (triggers the scaling plot).
    get_forward_time_per_atom
        Mean per-atom forward-pass times for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Forward Time / Atom": get_forward_time_per_atom,
    }


def test_inference_speed(metrics: dict[str, dict], struct_info: None) -> None:
    """
    Run inference speed analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Inference speed metric results provided by fixtures.
    struct_info : None
        Element info written to ``info.json`` for filtering.
    """
