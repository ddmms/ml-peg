"""
Analyse the Sellers torsion scan benchmark for drug-like fragments.

A Comparison of Quantum and Molecular Mechanical Methods to Estimate Strain Energy in
Druglike Fragments
Benjamin D. Sellers, Natalie C. James, Alberto Gobbi
Journal of Chemical Information and Modeling 2017 57 (6), 1265-1275
DOI: 10.1021/acs.jcim.6b00614
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

CALC_PATH = CALCS_ROOT / "conformers" / "Sellers_Tors" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "Sellers_Tors"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def sort_by_mol_id(path: Path) -> int:
    """
    Sort scan files numerically by fragment ID.

    Parameters
    ----------
    path
        Path to torsion scan file, named by fragment ID.

    Returns
    -------
    int
        Fragment ID.
    """
    return int(path.stem)


INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.extxyz",
    sort_key=sort_by_mol_id,
    info_keys=["mol_id", "smiles", "category", "dihedral"],
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)

# Reference scans are model-independent, so load them once from the mock outputs
REF_SCANS = {
    path.stem: read(path, index=":")
    for path in sorted((CALC_PATH / "mock").glob("*.extxyz"), key=sort_by_mol_id)
}
SCAN_LABELS = list(REF_SCANS)
SCAN_SMILES = [scan[0].info["smiles"] for scan in REF_SCANS.values()]


def get_model_rel_energies(model_name: str) -> dict[str, np.ndarray]:
    """
    Get model relative energies for each scan, aligned to the reference minimum.

    Parameters
    ----------
    model_name
        Name of model to read outputs for.

    Returns
    -------
    dict[str, np.ndarray]
        Relative energies in kcal/mol for each scan. NaN if missing or failed.
    """
    rel_energies = {}
    for label, ref_scan in REF_SCANS.items():
        output_file = CALC_PATH / model_name / f"{label}.extxyz"
        if not output_file.exists():
            rel_energies[label] = np.full(len(ref_scan), np.nan)
            continue

        scan = read(output_file, index=":")
        energies = np.array(
            [atoms.info.get("model_energy", np.nan) for atoms in scan], dtype=float
        )
        ref_min_idx = int(np.argmin([atoms.info["ref_rel_energy"] for atoms in scan]))
        rel_energies[label] = (energies - energies[ref_min_idx]) * EV_TO_KCAL
    return rel_energies


@pytest.fixture
def model_rel_energies() -> dict[str, dict[str, np.ndarray]]:
    """
    Get model relative energies for all models and scans.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Relative energies in kcal/mol for each scan, for all models.
    """
    return {model_name: get_model_rel_energies(model_name) for model_name in MODELS}


@pytest.fixture
def fragment_metrics(
    model_rel_energies: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, list]]:
    """
    Get per-fragment errors of the relative torsional energy profile for each model.

    Parameters
    ----------
    model_rel_energies
        Relative energies for each scan, for all models.

    Returns
    -------
    dict[str, dict[str, list]]
        Per model, the MAE, RMSE and absolute barrier error for each torsion scan
        in `SCAN_LABELS` order, and the angle/energy profile for each scan (for
        plotting torsion curves). Errors are NaN for scans with any failed
        point, and scans the model has no output for have no profile.
    """
    results = {}

    for model_name in MODELS:
        mae_scans = []
        rmse_scans = []
        barrier_scans = []
        profiles = []

        for label, scan in REF_SCANS.items():
            model_rel_energy = model_rel_energies[model_name][label]
            if np.isnan(model_rel_energy).all():
                mae_scans.append(float("nan"))
                rmse_scans.append(float("nan"))
                barrier_scans.append(float("nan"))
                profiles.append(None)
                continue

            angle = np.array([atoms.info["dihedral"] for atoms in scan])
            ref_rel_energy = np.array([atoms.info["ref_rel_energy"] for atoms in scan])

            mae_scans.append(mae(ref_rel_energy, model_rel_energy))
            rmse_scans.append(rmse(ref_rel_energy, model_rel_energy))
            # np.ptp propagates NaN from any failed scan point
            barrier_scans.append(
                float(abs(np.ptp(model_rel_energy) - np.ptp(ref_rel_energy)))
            )

            order = np.argsort(angle)
            profiles.append(
                {
                    "angle": angle[order].tolist(),
                    "ref_rel_energy": ref_rel_energy[order].tolist(),
                    "model_rel_energy": model_rel_energy[order].tolist(),
                }
            )

        results[model_name] = {
            "mae": mae_scans,
            "rmse": rmse_scans,
            "barrier_error": barrier_scans,
            "profiles": profiles,
        }

    return results


@pytest.fixture
def torsion_energies(
    model_rel_energies: dict[str, dict[str, np.ndarray]],
) -> dict[str, list]:
    """
    Get relative energies for all torsion scan points.

    Parameters
    ----------
    model_rel_energies
        Relative energies for each scan, for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energies.
    """
    results = {
        "ref": [
            atoms.info["ref_rel_energy"]
            for scan in REF_SCANS.values()
            for atoms in scan
        ]
    }
    for model_name in MODELS:
        results[model_name] = list(
            np.concatenate(
                [model_rel_energies[model_name][label] for label in SCAN_LABELS]
            )
        )
    return results


@pytest.fixture
def torsion_barriers(
    model_rel_energies: dict[str, dict[str, np.ndarray]],
) -> dict[str, list]:
    """
    Get torsional barriers (maximum minus minimum energy) for each scan.

    Parameters
    ----------
    model_rel_energies
        Relative energies for each scan, for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted torsional barriers.
    """
    results = {
        "ref": [
            np.ptp([atoms.info["ref_rel_energy"] for atoms in scan])
            for scan in REF_SCANS.values()
        ]
    }
    for model_name in MODELS:
        # np.ptp propagates NaN from any failed scan point
        results[model_name] = [
            float(np.ptp(model_rel_energies[model_name][label]))
            for label in SCAN_LABELS
        ]
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
            customdata=list(zip(SCAN_SMILES, labels, strict=True)),
            hovertemplate=(
                "<b>Fragment: </b>%{customdata[1]}<br>"
                "<b>SMILES: </b>%{customdata[0]}<br>"
                f"<b>{metric_name}: </b>" + "%{y:.4f} kcal/mol<br>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": f"Per-fragment {metric_name} - {model_name}"},
        xaxis={"title": {"text": "Fragment"}},
        yaxis={"title": {"text": f"{metric_name} / kcal/mol"}},
    )
    return fig


@pytest.fixture
def fragment_scatter_figures(fragment_metrics: dict[str, dict[str, list]]) -> None:
    """
    Save per-model scatter plots of per-fragment errors for the app.

    Parameters
    ----------
    fragment_metrics
        Per-fragment MAE, RMSE and barrier errors for each model.
    """
    for model_name in MODELS:
        model_metrics = fragment_metrics[model_name]

        # Missing scans are plotted as gaps, so points stay in `SCAN_LABELS`
        # order, but a model with nothing to plot is skipped entirely.
        if not any(profile for profile in model_metrics["profiles"]):
            continue

        out_dir = OUT_PATH / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for key, metric_name in (
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("barrier_error", "Barrier error"),
        ):
            fig = plot_fragment_metric_figure(
                model_name, metric_name, SCAN_LABELS, model_metrics[key]
            )
            fig.write_json(out_dir / f"fragment_{key}_scatter.json")


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
        ``angle``, ``ref_rel_energy``, and ``model_rel_energy`` for the scan.

    Returns
    -------
    go.Figure
        Line plot of relative energy against dihedral angle.
    """
    fig = go.Figure()
    # Label every point so curve clicks map back to the fragment's trajectory.
    customdata = [label] * len(profile["angle"])
    fig.add_trace(
        go.Scatter(
            x=profile["angle"],
            y=profile["ref_rel_energy"],
            mode="lines+markers",
            name="CCSD(T)",
            customdata=customdata,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile["angle"],
            y=profile["model_rel_energy"],
            mode="lines+markers",
            name=model_name,
            customdata=customdata,
        )
    )
    fig.update_layout(
        title={"text": f"{label} - {model_name}"},
        xaxis={"title": {"text": "Dihedral angle / deg"}},
        yaxis={"title": {"text": "Relative energy / kcal/mol"}},
    )
    return fig


@pytest.fixture
def torsion_curve_figures(fragment_metrics: dict[str, dict[str, list]]) -> None:
    """
    Save per-fragment torsion energy profile plots for the app.

    Parameters
    ----------
    fragment_metrics
        Per-fragment errors and profiles for each model.
    """
    for model_name in MODELS:
        profiles = fragment_metrics[model_name]["profiles"]

        if not any(profile for profile in profiles):
            continue

        out_dir = OUT_PATH / model_name / "torsion_curves"
        out_dir.mkdir(parents=True, exist_ok=True)

        for label, profile in zip(SCAN_LABELS, profiles, strict=True):
            # Scans the model has no output for have no curve to plot. The app
            # falls back to its click hint when a curve file is missing.
            if profile is None:
                continue
            fig = plot_torsion_curve_figure(model_name, label, profile)
            fig.write_json(out_dir / f"{label}.json")


@pytest.fixture
def torsion_trajectories() -> None:
    """
    Save per-fragment torsion scan trajectories for the app's structure viewer.

    Geometries are identical across all models, since only single-point energies
    are calculated on the same reference conformers, so this only needs writing
    once, from the mock calculator's output.
    """
    out_dir = OUT_PATH / "torsion_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, scan in REF_SCANS.items():
        order = np.argsort([atoms.info["dihedral"] for atoms in scan])
        write(out_dir / f"{label}.xyz", [scan[i] for i in order])


@pytest.fixture
def get_mae(torsion_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for relative energies.

    Parameters
    ----------
    torsion_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of relative energy MAEs for all models.
    """
    return {
        model_name: mae(torsion_energies["ref"], torsion_energies[model_name])
        for model_name in MODELS
    }


@pytest.fixture
def get_rmse(torsion_energies: dict[str, list]) -> dict[str, float]:
    """
    Get root mean squared error for relative energies.

    Parameters
    ----------
    torsion_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of relative energy RMSEs for all models.
    """
    return {
        model_name: rmse(torsion_energies["ref"], torsion_energies[model_name])
        for model_name in MODELS
    }


@pytest.fixture
def get_barrier_mae(torsion_barriers: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for torsional barriers.

    Parameters
    ----------
    torsion_barriers
        Dictionary of reference and predicted torsional barriers.

    Returns
    -------
    dict[str, float]
        Dictionary of torsional barrier MAEs for all models.
    """
    return {
        model_name: mae(torsion_barriers["ref"], torsion_barriers[model_name])
        for model_name in MODELS
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "sellers_tors_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    get_mae: dict[str, float],
    get_rmse: dict[str, float],
    get_barrier_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.
    get_rmse
        Root mean squared errors for all models.
    get_barrier_mae
        Torsional barrier mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
        "RMSE": get_rmse,
        "Barrier MAE": get_barrier_mae,
    }


def test_sellers_tors(
    metrics: dict[str, dict],
    fragment_scatter_figures: None,
    torsion_curve_figures: None,
    torsion_trajectories: None,
) -> None:
    """
    Run Sellers torsion scan test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    fragment_scatter_figures
        Per-model fragment MAE/RMSE/barrier error scatter figures (side-effect only).
    torsion_curve_figures
        Per-fragment torsion curve figures (side-effect only).
    torsion_trajectories
        Per-fragment torsion scan trajectories (side-effect only).
    """
    return
