"""Analyse low-dimensional (2D/1D) crystal relaxation benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import build_density_inputs, load_metrics_config, mae
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
    "2D": {"geom_col": "area_per_atom", "geom_label": "Area", "geom_unit": "Å²/atom"},
    "1D": {
        "geom_col": "length_per_atom",
        "geom_label": "Length",
        "geom_unit": "Å/atom",
    },
}


def _filter_energy_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark structures with extreme predicted energies as unconverged.

    Structures with predicted energy per atom outside [-40, 40] eV/atom
    are considered outliers and marked as unconverged.

    Parameters
    ----------
    df
        Results dataframe with 'pred_energy_per_atom' and 'converged' columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with outliers marked as unconverged.
    """
    df.loc[
        (df["pred_energy_per_atom"] <= ENERGY_OUTLIER_MIN)
        | (df["pred_energy_per_atom"] >= ENERGY_OUTLIER_MAX),
        "converged",
    ] = False
    return df


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
    Get converged geometric and energy data for a model.

    Parameters
    ----------
    model_name
        Name of the model.
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    dict[str, list]
        Dictionary with ref/pred values for geometric metric and energy.
    """
    df = load_results(model_name, dimensionality)
    if df.empty:
        return {"ref_geom": [], "pred_geom": [], "ref_energy": [], "pred_energy": []}

    df = _filter_energy_outliers(df)
    df_conv = df[df["converged"]].copy()

    geom_col = DIM_CONFIGS[dimensionality]["geom_col"]
    df_conv = df_conv.dropna(subset=[f"pred_{geom_col}", "pred_energy_per_atom"])

    return {
        "ref_geom": df_conv[f"ref_{geom_col}"].tolist(),
        "pred_geom": df_conv[f"pred_{geom_col}"].tolist(),
        "ref_energy": df_conv["ref_energy_per_atom"].tolist(),
        "pred_energy": df_conv["pred_energy_per_atom"].tolist(),
    }


def get_convergence_rate(model_name: str, dimensionality: str = "2D") -> float | None:
    """
    Get convergence rate for a model.

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
    df = _filter_energy_outliers(df)
    return (df["converged"].sum() / len(df)) * 100


def _build_stats(dimensionality: str) -> dict[str, dict]:
    """
    Aggregate converged ref/pred data per model for a given dimensionality.

    Parameters
    ----------
    dimensionality
        Either "2D" or "1D".

    Returns
    -------
    dict[str, dict]
        Per-model dicts with "geom" and "energy" ref/pred lists.
    """
    stats = {}
    for model_name in MODELS:
        data = get_converged_data(model_name, dimensionality)
        stats[model_name] = {
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


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_area_2d.json",
    title="Area per atom (2D)",
    x_label="Reference area / Å²/atom",
    y_label="Predicted area / Å²/atom",
)
def area_density_2d() -> dict[str, dict]:
    """
    Density scatter inputs for 2D area per atom.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    return build_density_inputs(MODELS, _build_stats("2D"), "geom", metric_fn=mae)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_energy_2d.json",
    title="Energy per atom (2D)",
    x_label="Reference energy / eV/atom",
    y_label="Predicted energy / eV/atom",
)
def energy_density_2d() -> dict[str, dict]:
    """
    Density scatter inputs for 2D energy per atom.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    return build_density_inputs(MODELS, _build_stats("2D"), "energy", metric_fn=mae)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_length_1d.json",
    title="Length per atom (1D)",
    x_label="Reference length / Å/atom",
    y_label="Predicted length / Å/atom",
)
def length_density_1d() -> dict[str, dict]:
    """
    Density scatter inputs for 1D length per atom.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    return build_density_inputs(MODELS, _build_stats("1D"), "geom", metric_fn=mae)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_energy_1d.json",
    title="Energy per atom (1D)",
    x_label="Reference energy / eV/atom",
    y_label="Predicted energy / eV/atom",
)
def energy_density_1d() -> dict[str, dict]:
    """
    Density scatter inputs for 1D energy per atom.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    return build_density_inputs(MODELS, _build_stats("1D"), "energy", metric_fn=mae)


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
    area_density_2d: dict[str, dict],
    energy_density_2d: dict[str, dict],
    length_density_1d: dict[str, dict],
    energy_density_1d: dict[str, dict],
) -> None:
    """
    Run low-dimensional relaxation analysis test.

    Parameters
    ----------
    metrics
        All low-dimensional relaxation metrics.
    area_density_2d
        Triggers 2D area density plot generation.
    energy_density_2d
        Triggers 2D energy density plot generation.
    length_density_1d
        Triggers 1D length density plot generation.
    energy_density_1d
        Triggers 1D energy density plot generation.
    """
    return
