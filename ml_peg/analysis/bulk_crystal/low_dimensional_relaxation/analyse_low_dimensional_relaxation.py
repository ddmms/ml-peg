"""Analyse low-dimensional (2D/1D) crystal relaxation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config, mae
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


def create_parity_plot(
    data_type: str,
    title: str,
    x_label: str,
    y_label: str,
    filename: Path,
    dimensionality: str = "2D",
) -> None:
    """
    Create a parity plot for low-dimensional structures.

    Parameters
    ----------
    data_type
        Either "geom" (area for 2D, length for 1D) or "energy".
    title
        Plot title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    filename
        Path to save the plot JSON.
    dimensionality
        Either "2D" or "1D".
    """
    fig = go.Figure()

    ref_values = []
    pred_values = []

    for model_name in MODELS:
        data = get_converged_data(model_name, dimensionality)
        if data_type == "geom":
            ref_values.extend(data["ref_geom"])
            pred_values.extend(data["pred_geom"])
        else:
            ref_values.extend(data["ref_energy"])
            pred_values.extend(data["pred_energy"])

    if ref_values and pred_values:
        fig.add_trace(
            go.Scatter(
                x=pred_values,
                y=ref_values,
                name=dimensionality,
                mode="markers",
                marker={"size": 6, "opacity": 0.7},
                hovertemplate=(
                    f"<b>{dimensionality}</b><br>"
                    "<b>Pred: </b>%{x:.4f}<br>"
                    "<b>Ref: </b>%{y:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    all_values = []
    for trace in fig.data:
        all_values.extend(trace.x)
        all_values.extend(trace.y)

    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="y=x",
                line={"color": "gray", "dash": "dash"},
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="closest",
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(fig.to_dict(), f)


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


def _generate_all_plots() -> None:
    """Generate all parity plots for each dimensionality."""
    for dim, cfg in DIM_CONFIGS.items():
        geom_label = cfg["geom_label"]
        geom_unit = cfg["geom_unit"]
        dim_lower = dim.lower()

        create_parity_plot(
            data_type="geom",
            title=f"{geom_label} per atom ({dim})",
            x_label=f"Predicted {geom_label.lower()} / {geom_unit}",
            y_label=f"Reference {geom_label.lower()} / {geom_unit}",
            filename=OUT_PATH / f"figure_{geom_label.lower()}_{dim_lower}.json",
            dimensionality=dim,
        )
        create_parity_plot(
            data_type="energy",
            title=f"Energy per atom ({dim})",
            x_label="Predicted energy / eV/atom",
            y_label="Reference energy / eV/atom",
            filename=OUT_PATH / f"figure_energy_{dim_lower}.json",
            dimensionality=dim,
        )


@pytest.fixture
def parity_plots() -> None:
    """Generate all parity plots for 2D and 1D structures."""
    _generate_all_plots()


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
    parity_plots: None,
) -> None:
    """
    Run low-dimensional relaxation analysis test.

    Parameters
    ----------
    metrics
        All low-dimensional relaxation metrics.
    parity_plots
        Triggers parity plot generation for all dimensionalities.
    """
    return
