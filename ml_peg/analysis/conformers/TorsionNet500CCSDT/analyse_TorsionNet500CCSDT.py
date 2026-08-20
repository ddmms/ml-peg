"""Analyse TorsionNet500CCSDT benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "conformers" / "TorsionNet500CCSDT" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "TorsionNet500CCSDT"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Elemental info for the app's filtering, saved per-fragment (rather than just the
# combined set) in preparation for future partial filtering.
get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    index="0",
    include_filenames=True,
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)


@pytest.fixture
def fragment_rmse() -> dict[str, dict[str, list]]:
    """
    Get per-fragment RMSE of the relative torsional energy profile for each model.

    Returns
    -------
    dict[str, dict[str, list]]
        Per model, the fragment labels, the corresponding RMSE and MAE for each
        torsion scan, and the mean-centered angle/energy profile for each scan
        (for plotting torsion curves).
    """
    results = {}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        labels = []
        rmse_scans = []
        mae_scans = []
        profiles = []

        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))

            for xyz_file in xyz_files:
                atoms = read(xyz_file, ":")

                angle = [a.info["torsion_angle"] for a in atoms]
                ref_energy = np.array([a.info["ref_energy"] for a in atoms])
                model_energy = np.array([a.info["model_energy"] for a in atoms])

                # Mean-center so only the relative torsional profile is compared.
                ref_rel_energy = ref_energy - ref_energy.mean()
                model_rel_energy = model_energy - model_energy.mean()

                labels.append(xyz_file.stem)
                rmse_scans.append(rmse(ref_rel_energy, model_rel_energy))
                mae_scans.append(mae(ref_rel_energy, model_rel_energy))

                order = np.argsort(angle)
                profiles.append(
                    {
                        "angle": np.asarray(angle)[order].tolist(),
                        "ref_rel_energy": ref_rel_energy[order].tolist(),
                        "model_rel_energy": model_rel_energy[order].tolist(),
                    }
                )

        results[model_name] = {
            "labels": labels,
            "rmse": rmse_scans,
            "mae": mae_scans,
            "profiles": profiles,
        }

    return results


@pytest.fixture
def get_rmse(fragment_rmse: dict[str, dict[str, list]]) -> dict[str, float | None]:
    """
    Get mean RMSE across TorsionNet500CCSDT torsion scans.

    Parameters
    ----------
    fragment_rmse
        Per-fragment RMSE and labels for each model.

    Returns
    -------
    dict[str, float | None]
        Mean RMSE per model, ignoring fragments where the calculation failed.
    """
    results = {}

    for model_name in MODELS:
        rmse_scans = fragment_rmse[model_name]["rmse"]
        results[model_name] = float(np.nanmean(rmse_scans)) if rmse_scans else None

    return results


@pytest.fixture
def get_mae(fragment_rmse: dict[str, dict[str, list]]) -> dict[str, float | None]:
    """
    Get mean MAE across TorsionNet500CCSDT torsion scans.

    Parameters
    ----------
    fragment_rmse
        Per-fragment RMSE, MAE, and labels for each model.

    Returns
    -------
    dict[str, float | None]
        Mean MAE per model, ignoring fragments where the calculation failed.
    """
    results = {}

    for model_name in MODELS:
        mae_scans = fragment_rmse[model_name]["mae"]
        results[model_name] = float(np.nanmean(mae_scans)) if mae_scans else None

    return results


def plot_fragment_metric_figure(
    model_name: str, metric_name: str, labels: list[str], metric_values: list[float]
) -> go.Figure:
    """
    Build a scatter plot of a per-fragment metric for one model.

    Parameters
    ----------
    model_name
        Name of the model the scatter plot is for.
    metric_name
        Name of the metric being plotted, e.g. ``"RMSE"`` or ``"MAE"``.
    labels
        Fragment labels, in scatter point order.
    metric_values
        Per-fragment metric values, in the same order as ``labels``.

    Returns
    -------
    go.Figure
        Scatter plot of the metric against fragment.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(labels))),
            y=metric_values,
            mode="markers",
            customdata=labels,
            hovertemplate=(
                "<b>Fragment: </b>%{customdata}<br>"
                f"<b>{metric_name}: </b>" + "%{y:.4f} eV<br>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": f"Per-fragment {metric_name} - {model_name}"},
        xaxis={"title": {"text": "Fragment"}},
        yaxis={"title": {"text": f"{metric_name} / eV"}},
    )
    return fig


@pytest.fixture
def fragment_scatter_figures(fragment_rmse: dict[str, dict[str, list]]) -> None:
    """
    Save per-model scatter plots of per-fragment RMSE and MAE for the app.

    Parameters
    ----------
    fragment_rmse
        Per-fragment RMSE, MAE, and labels for each model.
    """
    for model_name in MODELS:
        labels = fragment_rmse[model_name]["labels"]

        if not labels:
            continue

        out_dir = OUT_PATH / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        rmse_fig = plot_fragment_metric_figure(
            model_name, "RMSE", labels, fragment_rmse[model_name]["rmse"]
        )
        rmse_fig.write_json(out_dir / "fragment_rmse_scatter.json")

        mae_fig = plot_fragment_metric_figure(
            model_name, "MAE", labels, fragment_rmse[model_name]["mae"]
        )
        mae_fig.write_json(out_dir / "fragment_mae_scatter.json")


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
            name="CCSD(T)",
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
        yaxis={"title": {"text": "Relative energy / eV"}},
    )
    return fig


@pytest.fixture
def torsion_curve_figures(fragment_rmse: dict[str, dict[str, list]]) -> None:
    """
    Save per-fragment torsion energy profile plots for the app.

    Parameters
    ----------
    fragment_rmse
        Per-fragment RMSE, labels, and profiles for each model.
    """
    for model_name in MODELS:
        labels = fragment_rmse[model_name]["labels"]
        profiles = fragment_rmse[model_name]["profiles"]

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
    once, from the mock calculator's output, matching the ``info.json`` elemental
    info above.
    """
    mock_dir = CALC_PATH / "mock"
    if not mock_dir.exists():
        return

    out_dir = OUT_PATH / "torsion_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    for xyz_file in sorted(mock_dir.glob("*.xyz")):
        atoms = read(xyz_file, ":")
        angle = [a.info["torsion_angle"] for a in atoms]
        order = np.argsort(angle)
        write(out_dir / f"{xyz_file.stem}.xyz", [atoms[i] for i in order])


@pytest.fixture
@build_table(
    filename=OUT_PATH / "torsionnet500ccsdt_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    get_rmse: dict[str, float | None], get_mae: dict[str, float | None]
) -> dict[str, dict]:
    """
    Get all TorsionNet500CCSDT metrics.

    Parameters
    ----------
    get_rmse
        Mean RMSE per model.
    get_mae
        Mean MAE per model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "RMSE": get_rmse,
        "MAE": get_mae,
    }


def test_torsionnet500ccsdt(
    metrics: dict[str, dict],
    fragment_scatter_figures: None,
    torsion_curve_figures: None,
    torsion_trajectories: None,
) -> None:
    """
    Run TorsionNet500CCSDT analysis.

    Parameters
    ----------
    metrics
        All TorsionNet500CCSDT metrics.
    fragment_scatter_figures
        Per-model fragment-RMSE/MAE scatter figures (side-effect only).
    torsion_curve_figures
        Per-fragment torsion curve figures (side-effect only).
    torsion_trajectories
        Per-fragment torsion scan trajectories (side-effect only).
    """
    return
