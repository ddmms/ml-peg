"""Analyse TorsionNet500 dihedral scan benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from mlipaudit.benchmarks.dihedral_scan.dihedral_scan import DihedralScanModelOutput
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegDihedralScanBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "conformers" / "TorsionNet500" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "TorsionNet500"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> list:
    """
    Get the ordered list of fragment names.

    Returns
    -------
    list
        List of all dihedral scan fragment names.
    """
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if path.exists():
            output = DihedralScanModelOutput.model_validate_json(path.read_text())
            return sorted(fragment.fragment_name for fragment in output.fragments)
    return []


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``DihedralScanResult``.
    """
    data_input_dir = download_s3_data(
        key="inputs/conformers/TorsionNet500/TorsionNet500.zip",
        filename="TorsionNet500.zip",
    )

    results = {}
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if not path.exists():
            continue
        benchmark = MlPegDihedralScanBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = DihedralScanModelOutput.model_validate_json(
            path.read_text()
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
    data_input_dir = download_s3_data(
        key="inputs/conformers/TorsionNet500/TorsionNet500.zip",
        filename="TorsionNet500.zip",
    )
    benchmark = MlPegDihedralScanBenchmark(
        force_field=Calculator(),
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    elements = sorted(
        {
            symbol
            for fragment in benchmark._torsion_net_500.values()
            for symbol in fragment.atom_symbols
        }
    )
    info = {"elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(json.dumps(info, indent=1))
    return info


@pytest.fixture
def fragment_profiles(analyze_results) -> dict[str, dict[str, list]]:
    """
    Get per-fragment barrier heights and torsion profile for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``DihedralScanResult``.

    Returns
    -------
    dict[str, dict[str, list]]
        Per model, the fragment labels, the predicted and reference barrier
        heights for each torsion scan (for the parity plot), and the
        mean-centered angle/energy profile for each scan (for plotting torsion
        curves).
    """
    results = {}
    for model_name in MODELS:
        labels_list = []
        pred_barrier = []
        ref_barrier = []
        profiles = []
        result = analyze_results.get(model_name)
        if result is not None:
            for fragment in result.fragments:
                if fragment.failed:
                    continue
                angle = np.asarray(fragment.distance_profile)
                ref_energy = np.asarray(fragment.reference_energy_profile)
                model_energy = np.asarray(fragment.predicted_energy_profile)
                labels_list.append(fragment.fragment_name)
                pred_barrier.append(float(model_energy.max() - model_energy.min()))
                ref_barrier.append(float(ref_energy.max() - ref_energy.min()))
                # Mean-center so only the relative torsional profile is compared.
                ref_rel_energy = ref_energy - ref_energy.mean()
                model_rel_energy = model_energy - model_energy.mean()
                order = np.argsort(angle)
                profiles.append(
                    {
                        "angle": angle[order].tolist(),
                        "ref_rel_energy": ref_rel_energy[order].tolist(),
                        "model_rel_energy": model_rel_energy[order].tolist(),
                    }
                )
        results[model_name] = {
            "labels": labels_list,
            "pred_barrier": pred_barrier,
            "ref_barrier": ref_barrier,
            "profiles": profiles,
        }
    return results


@pytest.fixture
def get_mae(analyze_results) -> dict[str, float]:
    """
    Get the barrier height mean absolute error for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``DihedralScanResult``.

    Returns
    -------
    dict[str, float]
        Mean absolute barrier height error in kcal/mol for each model.
    """
    return {
        model_name: result.mae_barrier_height
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_score(analyze_results) -> dict[str, float]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``DihedralScanResult``.

    Returns
    -------
    dict[str, float]
        The mlipaudit per-fragment soft-threshold score (0 to 1) for each model.
    """
    return {model_name: result.score for model_name, result in analyze_results.items()}


def plot_fragment_parity_figure(
    model_name: str,
    labels: list[str],
    pred_barrier: list[float],
    ref_barrier: list[float],
) -> go.Figure:
    """
    Build a barrier height parity plot for one model.

    Parameters
    ----------
    model_name
        Name of the model the parity plot is for.
    labels
        Fragment labels, in scatter point order.
    pred_barrier
        Predicted torsion barrier heights, in the same order as ``labels``.
    ref_barrier
        Reference torsion barrier heights, in the same order as ``labels``.

    Returns
    -------
    go.Figure
        Parity plot of predicted against reference barrier height, one point
        per fragment, with a ``y = x`` reference line.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_barrier,
            y=ref_barrier,
            mode="markers",
            customdata=labels,
            hovertemplate=(
                "<b>Fragment: </b>%{customdata}<br>"
                "<b>Predicted: </b>%{x:.3f} kcal/mol<br>"
                "<b>Reference: </b>%{y:.3f} kcal/mol<br>"
            ),
            showlegend=False,
        )
    )
    lims = [0.0, max([*pred_barrier, *ref_barrier], default=1.0)]
    fig.add_trace(
        go.Scatter(x=lims, y=lims, mode="lines", showlegend=False, hoverinfo="skip")
    )
    fig.update_layout(
        title={"text": f"Barrier height parity - {model_name}"},
        xaxis={"title": {"text": "Predicted barrier height / kcal/mol"}},
        yaxis={"title": {"text": "Reference barrier height / kcal/mol"}},
    )
    return fig


@pytest.fixture
def fragment_scatter_figures(fragment_profiles: dict[str, dict[str, list]]) -> None:
    """
    Save per-model barrier height parity plots for the app.

    Parameters
    ----------
    fragment_profiles
        Per-fragment barrier heights, labels, and profiles for each model.
    """
    for model_name in MODELS:
        labels = fragment_profiles[model_name]["labels"]
        if not labels:
            continue
        out_dir = OUT_PATH / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        parity_fig = plot_fragment_parity_figure(
            model_name,
            labels,
            fragment_profiles[model_name]["pred_barrier"],
            fragment_profiles[model_name]["ref_barrier"],
        )
        parity_fig.write_json(out_dir / "fragment_barrier_parity.json")


def plot_torsion_curve_figure(
    model_name: str, label: str, profile: dict[str, list]
) -> go.Figure:
    """
    Build a torsion energy profile plot for one fragment.

    Parameters
    ----------
    model_name
        Name of the model the profile was predicted with.
    label
        Fragment label the profile belongs to.
    profile
        Mean-centered ``angle``, ``ref_rel_energy``, and ``model_rel_energy`` for
        the scan.

    Returns
    -------
    go.Figure
        Line plot of relative energy against dihedral angle.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=profile["angle"],
            y=profile["ref_rel_energy"],
            mode="lines+markers",
            name="Reference",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile["angle"],
            y=profile["model_rel_energy"],
            mode="lines+markers",
            name=model_name,
        )
    )
    fig.update_layout(
        title={"text": f"{label} - {model_name}"},
        xaxis={"title": {"text": "Dihedral angle / deg"}},
        yaxis={"title": {"text": "Relative energy / kcal/mol"}},
    )
    return fig


@pytest.fixture
def torsion_curve_figures(fragment_profiles: dict[str, dict[str, list]]) -> None:
    """
    Save per-fragment torsion energy profile plots for the app.

    Parameters
    ----------
    fragment_profiles
        Per-fragment barrier height error, labels, and profiles for each model.
    """
    for model_name in MODELS:
        labels = fragment_profiles[model_name]["labels"]
        profiles = fragment_profiles[model_name]["profiles"]
        if not labels:
            continue
        out_dir = OUT_PATH / model_name / "torsion_curves"
        out_dir.mkdir(parents=True, exist_ok=True)
        for label, profile in zip(labels, profiles, strict=True):
            fig = plot_torsion_curve_figure(model_name, label, profile)
            fig.write_json(out_dir / f"{label}.json")


@pytest.fixture
def torsion_trajectories() -> None:
    """
    Save per-fragment torsion scan trajectories for the app's structure viewer.

    Geometries are identical across all models, since only single-point energies
    are calculated on the same reference conformers, so this only needs writing
    once, from the input dataset, angle-sorted to match the torsion curves.
    """
    data_input_dir = download_s3_data(
        key="inputs/conformers/TorsionNet500/TorsionNet500.zip",
        filename="TorsionNet500.zip",
    )
    benchmark = MlPegDihedralScanBenchmark(
        force_field=Calculator(),
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    out_dir = OUT_PATH / "torsion_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fragment_name, fragment in benchmark._torsion_net_500.items():
        angle = np.array([state[0] for state in fragment.dft_energy_profile])
        order = np.argsort(angle)
        images = [
            Atoms(
                symbols=fragment.atom_symbols,
                positions=fragment.conformer_coordinates[i],
            )
            for i in order
        ]
        write(out_dir / f"{fragment_name}.xyz", images)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "torsionnet500_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(get_mae: dict[str, float], get_score: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute barrier height errors for all models.
    get_score
        The mlipaudit benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Barrier Height MAE": get_mae,
        "Torsion Score": get_score,
    }


def test_torsionnet500(
    metrics: dict[str, dict],
    fragment_scatter_figures: None,
    torsion_curve_figures: None,
    torsion_trajectories: None,
    struct_info: dict,
) -> None:
    """
    Run TorsionNet500 analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        TorsionNet500 metric results provided by fixtures.
    fragment_scatter_figures
        Per-model fragment barrier-error scatter figures (side-effect only).
    torsion_curve_figures
        Per-fragment torsion curve figures (side-effect only).
    torsion_trajectories
        Per-fragment torsion scan trajectories (side-effect only).
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
