"""Analyse low-dimensional (2D/1D) crystal relaxation benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.io import read as ase_read
from ase.io import write as ase_write
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    build_density_inputs,
    load_metrics_config,
    mae,
    write_density_trajectories,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "low_dimensional_relaxation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "low_dimensional_relaxation"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Energy outlier thresholds (eV/atom)
ENERGY_OUTLIER_MIN = -40
ENERGY_OUTLIER_MAX = 40

# Dimensionality-specific configuration
DIM_CONFIGS = {
    "2D": {
        "geom_col": "area_per_atom",
        "geom_label": "Area",
        "geom_unit": "Å²/atom",
        "traj_geom_dirname": "density_traj_area_2d",
        "traj_energy_dirname": "density_traj_energy_2d",
        "geom_plot_filename": "figure_area_2d.json",
        "energy_plot_filename": "figure_energy_2d.json",
    },
    "1D": {
        "geom_col": "length_per_atom",
        "geom_label": "Length",
        "geom_unit": "Å/atom",
        "traj_geom_dirname": "density_traj_length_1d",
        "traj_energy_dirname": "density_traj_energy_1d",
        "geom_plot_filename": "figure_length_1d.json",
        "energy_plot_filename": "figure_energy_1d.json",
    },
}


def load_results(model_name: str, dimensionality: str = "2D") -> pd.DataFrame:
    """
    Load results for a specific model and dimensionality.

    Parameters
    ----------
    model_name
        Name of the model.
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    pd.DataFrame
        Results dataframe or empty dataframe if not found.
    """
    csv_path = CALC_PATH / model_name / f"results_{dimensionality}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def get_converged_data(model_name: str, dimensionality: str = "2D") -> dict[str, list]:
    """
    Get converged geometric and energy data for a model by reading xyz files.

    Reads per-structure xyz files written by the calc step and copies them to
    the app data directory. Filters predicted-energy outliers (treated as
    non-converged).

    Parameters
    ----------
    model_name
        Name of the model.
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    dict[str, list]
        Labels (mat_ids) and lists of ref/pred values for geometric metric
        and energy.
    """
    xyz_dir = CALC_PATH / model_name / dimensionality
    if not xyz_dir.exists():
        return {
            "labels": [],
            "ref_geom": [],
            "pred_geom": [],
            "ref_energy": [],
            "pred_energy": [],
        }

    geom_col = DIM_CONFIGS[dimensionality]["geom_col"]
    ref_key = f"ref_{geom_col}"
    pred_key = f"pred_{geom_col}"

    app_dir = OUT_PATH / model_name / dimensionality
    app_dir.mkdir(parents=True, exist_ok=True)

    labels, ref_geom, pred_geom, ref_energy, pred_energy = [], [], [], [], []
    for xyz_file in sorted(xyz_dir.glob("*.xyz")):
        atoms = ase_read(xyz_file)
        pred_e = atoms.info.get("pred_energy_per_atom")
        pred_g = atoms.info.get(pred_key)
        if (
            pred_e is None
            or pred_g is None
            or pred_e <= ENERGY_OUTLIER_MIN
            or pred_e >= ENERGY_OUTLIER_MAX
        ):
            continue
        labels.append(xyz_file.stem)
        ref_geom.append(atoms.info[ref_key])
        pred_geom.append(pred_g)
        ref_energy.append(atoms.info["ref_energy_per_atom"])
        pred_energy.append(pred_e)
        ase_write(app_dir / xyz_file.name, atoms)

    return {
        "labels": labels,
        "ref_geom": ref_geom,
        "pred_geom": pred_geom,
        "ref_energy": ref_energy,
        "pred_energy": pred_energy,
    }


def get_convergence_rate(model_name: str, dimensionality: str = "2D") -> float | None:
    """
    Get convergence rate for a model.

    Converged structures whose predicted energy falls outside
    ``[ENERGY_OUTLIER_MIN, ENERGY_OUTLIER_MAX]`` are counted as unconverged.

    Parameters
    ----------
    model_name
        Name of the model.
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    float | None
        Convergence rate (%) or None if no data.
    """
    df = load_results(model_name, dimensionality)
    if df.empty:
        return None

    n_total = len(df)
    n_converged = int(df["converged"].sum())

    xyz_dir = CALC_PATH / model_name / dimensionality
    if xyz_dir.is_dir():
        for xyz_file in xyz_dir.glob("*.xyz"):
            atoms = ase_read(xyz_file)
            pred_e = atoms.info.get("pred_energy_per_atom")
            if pred_e is not None and (
                pred_e <= ENERGY_OUTLIER_MIN or pred_e >= ENERGY_OUTLIER_MAX
            ):
                n_converged -= 1

    return (n_converged / n_total) * 100


def _build_stats(dimensionality: str) -> dict[str, dict[str, Any]]:
    """
    Aggregate converged ref/pred data per model for a given dimensionality.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model dicts with "labels", "geom" and "energy" ref/pred lists.
    """
    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        data = get_converged_data(model_name, dimensionality)
        stats[model_name] = {
            "labels": data["labels"],
            "geom": {"ref": data["ref_geom"], "pred": data["pred_geom"]},
            "energy": {"ref": data["ref_energy"], "pred": data["pred_energy"]},
        }
    return stats


def _compute_mae(dimensionality: str, data_key: str) -> dict[str, float]:
    """
    Compute MAE across all models for a given dimensionality and data key.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".
    data_key
        Either "geom" or "energy".

    Returns
    -------
    dict[str, float]
        {model_name: mae_value}.
    """
    results = {}
    ref_key = f"ref_{data_key}"
    pred_key = f"pred_{data_key}"
    for model_name in MODELS:
        data = get_converged_data(model_name, dimensionality)
        if data[ref_key] and data[pred_key]:
            results[model_name] = mae(data[ref_key], data[pred_key])
    return results


def _compute_convergence(dimensionality: str) -> dict[str, float]:
    """
    Compute convergence rates across all models for a given dimensionality.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    dict[str, float]
        {model_name: convergence_rate}.
    """
    results = {}
    for model_name in MODELS:
        conv_rate = get_convergence_rate(model_name, dimensionality)
        if conv_rate is not None:
            results[model_name] = conv_rate
    return results


def _build_density_plot_for_condition(
    stats_per_model: dict[str, dict[str, Any]],
    *,
    quantity: str,
    dimensionality: str,
    filename: Path,
    title: str,
    x_label: str,
    y_label: str,
    traj_dirname: str,
) -> None:
    """
    Write a density-scatter plot and structure trajectories for one condition.

    Parameters
    ----------
    stats_per_model
        Mapping of model name to that model's stats for this dimensionality.
    quantity
        Either ``"geom"`` or ``"energy"``.
    dimensionality
        Either ``"2D"`` or ``"1D"``.
    filename
        Output JSON file for the density plot.
    title
        Plot title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    traj_dirname
        Per-model subdirectory name for sampled structure trajectories.
    """

    @plot_density_scatter(
        filename=filename,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )
    def _build(stats: dict[str, dict[str, Any]] = stats_per_model) -> dict[str, dict]:
        """
        Write sampled trajectories and return density-scatter inputs.

        Parameters
        ----------
        stats
            Mapping of model name to that model's stats.

        Returns
        -------
        dict[str, dict]
            Density-scatter inputs per model.
        """
        for model_name in MODELS:
            model_stats = stats.get(model_name)
            if model_stats is None or not model_stats["labels"]:
                continue
            write_density_trajectories(
                labels_list=model_stats["labels"],
                ref_vals=model_stats[quantity]["ref"],
                pred_vals=model_stats[quantity]["pred"],
                struct_dir=OUT_PATH / model_name / dimensionality,
                traj_dir=OUT_PATH / model_name / traj_dirname,
                struct_filename_builder=lambda label: f"{label}.xyz",
            )
        return build_density_inputs(MODELS, stats, quantity, metric_fn=mae)

    _build()


@pytest.fixture
def area_density_2d() -> None:
    """Write the 2D area-per-atom density plot."""
    cfg = DIM_CONFIGS["2D"]
    _build_density_plot_for_condition(
        _build_stats("2D"),
        quantity="geom",
        dimensionality="2D",
        filename=OUT_PATH / cfg["geom_plot_filename"],
        title="Area per atom (2D)",
        x_label="Reference area / Å²/atom",
        y_label="Predicted area / Å²/atom",
        traj_dirname=cfg["traj_geom_dirname"],
    )


@pytest.fixture
def energy_density_2d() -> None:
    """Write the 2D energy-per-atom density plot."""
    cfg = DIM_CONFIGS["2D"]
    _build_density_plot_for_condition(
        _build_stats("2D"),
        quantity="energy",
        dimensionality="2D",
        filename=OUT_PATH / cfg["energy_plot_filename"],
        title="Energy per atom (2D)",
        x_label="Reference energy / eV/atom",
        y_label="Predicted energy / eV/atom",
        traj_dirname=cfg["traj_energy_dirname"],
    )


@pytest.fixture
def length_density_1d() -> None:
    """Write the 1D length-per-atom density plot."""
    cfg = DIM_CONFIGS["1D"]
    _build_density_plot_for_condition(
        _build_stats("1D"),
        quantity="geom",
        dimensionality="1D",
        filename=OUT_PATH / cfg["geom_plot_filename"],
        title="Length per atom (1D)",
        x_label="Reference length / Å/atom",
        y_label="Predicted length / Å/atom",
        traj_dirname=cfg["traj_geom_dirname"],
    )


@pytest.fixture
def energy_density_1d() -> None:
    """Write the 1D energy-per-atom density plot."""
    cfg = DIM_CONFIGS["1D"]
    _build_density_plot_for_condition(
        _build_stats("1D"),
        quantity="energy",
        dimensionality="1D",
        filename=OUT_PATH / cfg["energy_plot_filename"],
        title="Energy per atom (1D)",
        x_label="Reference energy / eV/atom",
        y_label="Predicted energy / eV/atom",
        traj_dirname=cfg["traj_energy_dirname"],
    )


@pytest.fixture
@build_table(
    filename=OUT_PATH / "low_dimensional_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics() -> dict[str, dict]:
    """
    Compute all low-dimensional relaxation metrics.

    Returns
    -------
    dict[str, dict]
        All metrics for all models.
    """
    result = {}
    for dim, cfg in DIM_CONFIGS.items():
        geom_label = cfg["geom_label"]
        result[f"{geom_label} MAE ({dim})"] = _compute_mae(dim, "geom")
        result[f"Energy MAE ({dim})"] = _compute_mae(dim, "energy")
        result[f"Convergence ({dim})"] = _compute_convergence(dim)
    return result


def test_low_dimensional_relaxation(
    metrics: dict[str, dict],
    area_density_2d: None,
    energy_density_2d: None,
    length_density_1d: None,
    energy_density_1d: None,
) -> None:
    """
    Run low-dimensional relaxation analysis test.

    Parameters
    ----------
    metrics
        All low-dimensional relaxation metrics.
    area_density_2d
        Triggers 2D area density plot and trajectory generation.
    energy_density_2d
        Triggers 2D energy density plot and trajectory generation.
    length_density_1d
        Triggers 1D length density plot and trajectory generation.
    energy_density_1d
        Triggers 1D energy density plot and trajectory generation.
    """
    return
