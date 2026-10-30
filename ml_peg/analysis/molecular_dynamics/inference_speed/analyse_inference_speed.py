"""Analyse the inference speed benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn
from zipfile import ZipFile

from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
from plotly.colors import qualitative
import plotly.graph_objects as go
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.inference_speed.inference_speed import (
    InferenceSpeedModelOutput,
)
from mlipaudit.io import MODEL_OUTPUT_JSON_FILENAME, MODEL_OUTPUT_ZIP_FILENAME

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import (
    load_metrics_config,
    write_struct_info,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegInferenceSpeedBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)

EXAMPLE_INPUT_FILENAME = "71_1jrs_leupeptin.xyz"

BENCHMARK = MlPegInferenceSpeedBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "inference_speed" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "inference_speed"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

#: Conversion factor from seconds to microseconds.
SECONDS_TO_MICROSECONDS = 1.0e6

#: Colours cycled over models in the throughput plot.
COLOURS = qualitative.Dark24


def check_dataset() -> None:
    """
    Check the dataset saved by the calculation is available.

    The calculation copies nad unzips the downloaded dataset into its outputs,
    so the analysis does not need to download the input data again.

    Raises
    ------
    ValueError
        If the dataset is missing from the calculation outputs.
    """
    dataset_path = CALC_PATH / BENCHMARK / EXAMPLE_INPUT_FILENAME
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist. Please run the calculation.")


def _load_inference_speed_output(model_outputs_dir: Path) -> InferenceSpeedModelOutput:
    """
    Load an inference speed model output written by ``write_model_output_to_disk``.

    Parameters
    ----------
    model_outputs_dir
        Directory containing the ``inference_speed`` benchmark subdirectory.

    Returns
    -------
    InferenceSpeedModelOutput
        The loaded model output.
    """
    zip_path = Path(model_outputs_dir) / BENCHMARK / MODEL_OUTPUT_ZIP_FILENAME
    with ZipFile(zip_path) as zip_file:
        with zip_file.open(MODEL_OUTPUT_JSON_FILENAME) as json_file:
            data = json.load(json_file)
    return InferenceSpeedModelOutput.model_validate(data)


def _check_structures_match(benchmark: MlPegInferenceSpeedBenchmark) -> None:
    """
    Check the loaded timings describe the structures currently on disk.

    ``analyze`` pairs ``model_output.forward_times`` with the structure names it
    lists from the data directory, by position. If the dataset on disk has changed
    since the calculation ran, timings would be silently attributed to the wrong
    structures, so require the two orderings to be identical.

    Parameters
    ----------
    benchmark
        Benchmark with its ``model_output`` already loaded.

    Raises
    ------
    ValueError
        If the structures differ from those the calculation timed.
    """
    timed = benchmark.model_output.structure_names
    on_disk = benchmark._structure_names
    if timed != on_disk:
        raise ValueError(
            f"Structures timed by the calculation ({timed}) do not match those in "
            f"{benchmark.data_dir} ({on_disk}). Please re-run the calculation."
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
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegInferenceSpeedBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        try:
            benchmark.model_output = _load_inference_speed_output(
                CALC_PATH / model_name
            )
            _check_structures_match(benchmark)
            results[model_name] = benchmark.analyze()
        except Exception as exc:
            warn(
                f"Failed to analyse inference speed for {model_name}: {exc}; "
                "metric set to NaN.",
                stacklevel=2,
            )
    return results


@pytest.fixture
def num_atoms() -> dict[str, int]:
    """
    Get the number of atoms in each input structure.

    ``InferenceSpeedStructureResult.num_atoms`` is parsed from the filename prefix,
    which does not always match the structure that was timed: some inputs are
    multi-frame, and the calculation times the final frame, as read by ASE.

    Returns
    -------
    dict[str, int]
        Mapping of structure name to its number of atoms.
    """
    check_dataset()

    data_dir = CALC_PATH / BENCHMARK
    return {path.stem: len(read(path)) for path in sorted(data_dir.glob("*.xyz"))}


@pytest.fixture
def struct_info() -> None:
    """Write the combined element set to ``info.json`` for filtering."""
    check_dataset()

    data_dir = CALC_PATH / BENCHMARK
    write_struct_info(
        data_path=sorted(data_dir.glob("*.xyz")),
        out_path=OUT_PATH,
    )


@pytest.fixture
def scaling_curves(analyze_results, num_atoms) -> dict[str, tuple[list, list, list]]:
    """
    Get the model throughput scaling curve for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``InferenceSpeedResult``.
    num_atoms
        Mapping of structure name to its number of atoms.

    Returns
    -------
    dict[str, tuple[list, list, list]]
        Per-model ``(num_atoms, throughput, structure_names)`` curves, with the
        throughput in atoms per second, sorted by system size and restricted to
        structures with a successful forward pass.
    """
    curves = {}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        points = [
            (
                num_atoms[structure.structure_name],
                num_atoms[structure.structure_name] / structure.average_forward_time,
                structure.structure_name,
            )
            for structure in result.structures
            if structure.average_forward_time
        ]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        sizes, throughputs, names = zip(*points, strict=True)
        curves[model_name] = (list(sizes), list(throughputs), list(names))
    return curves


@pytest.fixture
def throughput_plot(scaling_curves) -> None:
    """
    Plot model throughput against system size on log-log axes.

    Each model is shown as markers with a power-law fit, ``throughput = a * N^k``,
    obtained by a linear fit in log-log space.

    Parameters
    ----------
    scaling_curves
        Per-model ``(num_atoms, throughput, structure_names)`` curves.
    """
    fig = go.Figure()
    for i, (model_name, (num_atoms, throughputs, names)) in enumerate(
        scaling_curves.items()
    ):
        colour = COLOURS[i % len(COLOURS)]
        fig.add_trace(
            go.Scatter(
                x=num_atoms,
                y=throughputs,
                name=model_name,
                legendgroup=model_name,
                mode="markers",
                marker={"color": colour},
                customdata=names,
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Structure: %{customdata}<br>"
                    "Atoms: %{x}<br>"
                    "Throughput: %{y:,.0f} atoms/s"
                    "<extra></extra>"
                ),
            )
        )
        if len(set(num_atoms)) < 2:
            continue
        exponent, log_prefactor = np.polyfit(np.log(num_atoms), np.log(throughputs), 1)
        grid = np.geomspace(min(num_atoms), max(num_atoms), 50)
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=np.exp(log_prefactor) * grid**exponent,
                name=f"{model_name} fit",
                legendgroup=model_name,
                showlegend=False,
                mode="lines",
                line={"color": colour},
                hovertemplate=(
                    f"<b>{model_name}</b> fit (throughput ∝ N<sup>{exponent:.2f}</sup>)"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={"text": "Model throughput vs system size"},
        xaxis={"title": {"text": "Number of atoms"}, "type": "log"},
        yaxis={"title": {"text": "Model Throughput (atoms/second)"}, "type": "log"},
    )
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.write_json(OUT_PATH / "figure_inference_speed_scaling.json")


@pytest.fixture
def get_forward_time_per_atom(analyze_results, num_atoms) -> dict[str, float]:
    """
    Get the mean per-atom forward-pass time for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``InferenceSpeedResult``.
    num_atoms
        Mapping of structure name to its number of atoms.

    Returns
    -------
    dict[str, float]
        Mean over structures of ``average_forward_time / num_atoms``, in
        microseconds per atom. NaN if the model has no results or any structure
        lacks a successful forward pass.
    """
    per_atom_times = {}
    for model_name in MODELS:
        result = analyze_results.get(model_name)
        if (
            result is None
            or not result.structures
            or any(s.average_forward_time is None for s in result.structures)
        ):
            per_atom_times[model_name] = np.nan
            continue
        values = [
            structure.average_forward_time / num_atoms[structure.structure_name]
            for structure in result.structures
        ]
        per_atom_times[model_name] = sum(values) / len(values) * SECONDS_TO_MICROSECONDS
    return per_atom_times


@pytest.fixture
@build_table(
    filename=OUT_PATH / "inference_speed_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    throughput_plot: None,
    get_forward_time_per_atom: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    throughput_plot
        Model throughput plot for all models (triggers writing the plot).
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
