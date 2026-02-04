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
ENERGY_OUTLIER_MIN = -25
ENERGY_OUTLIER_MAX = 25


def _filter_energy_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark structures with extreme predicted energies as unconverged.

    Structures with predicted energy per atom outside [-25, 25] eV/atom
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

    # Filter converged structures and remove NaN values
    df = _filter_energy_outliers(df)
    df_conv = df[df["converged"]].copy()

    # Determine geometric column based on dimensionality
    if dimensionality == "2D":
        geom_col = "area_per_atom"
    else:  # 1D
        geom_col = "length_per_atom"

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

    # Collect data from all models
    for model_name in MODELS:
        data = get_converged_data(model_name, dimensionality)
        if data_type == "geom":
            ref_values.extend(data["ref_geom"])
            pred_values.extend(data["pred_geom"])
        else:  # energy
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

    # Add diagonal line (y=x)
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

    # Save to JSON
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(fig.to_dict(), f)


@pytest.fixture
def area_predictions_2d() -> None:
    """Create area parity plot for 2D structures."""
    create_parity_plot(
        data_type="geom",
        title="Area per atom (2D)",
        x_label="Predicted area / Å²/atom",
        y_label="Reference area / Å²/atom",
        filename=OUT_PATH / "figure_area_2d.json",
        dimensionality="2D",
    )


@pytest.fixture
def energy_predictions_2d() -> None:
    """Create energy parity plot for 2D structures."""
    create_parity_plot(
        data_type="energy",
        title="Energy per atom (2D)",
        x_label="Predicted energy / eV/atom",
        y_label="Reference energy / eV/atom",
        filename=OUT_PATH / "figure_energy_2d.json",
        dimensionality="2D",
    )


@pytest.fixture
def length_predictions_1d() -> None:
    """Create length parity plot for 1D structures."""
    create_parity_plot(
        data_type="geom",
        title="Length per atom (1D)",
        x_label="Predicted length / Å/atom",
        y_label="Reference length / Å/atom",
        filename=OUT_PATH / "figure_length_1d.json",
        dimensionality="1D",
    )


@pytest.fixture
def energy_predictions_1d() -> None:
    """Create energy parity plot for 1D structures."""
    create_parity_plot(
        data_type="energy",
        title="Energy per atom (1D)",
        x_label="Predicted energy / eV/atom",
        y_label="Reference energy / eV/atom",
        filename=OUT_PATH / "figure_energy_1d.json",
        dimensionality="1D",
    )


@pytest.fixture
def area_mae_2d() -> dict[str, float]:
    """
    Calculate MAE for area predictions (2D).

    Returns
    -------
    dict[str, float]
        {model_name: mae_value}.
    """
    results = {}
    for model_name in MODELS:
        data = get_converged_data(model_name, "2D")
        if data["ref_geom"] and data["pred_geom"]:
            results[model_name] = mae(data["ref_geom"], data["pred_geom"])
    return results


@pytest.fixture
def energy_mae_2d() -> dict[str, float]:
    """
    Calculate MAE for energy predictions (2D).

    Returns
    -------
    dict[str, float]
        {model_name: mae_value}.
    """
    results = {}
    for model_name in MODELS:
        data = get_converged_data(model_name, "2D")
        if data["ref_energy"] and data["pred_energy"]:
            results[model_name] = mae(data["ref_energy"], data["pred_energy"])
    return results


@pytest.fixture
def convergence_2d() -> dict[str, float]:
    """
    Calculate convergence rate for 2D structures.

    Returns
    -------
    dict[str, float]
        {model_name: convergence_rate}.
    """
    results = {}
    for model_name in MODELS:
        conv_rate = get_convergence_rate(model_name, "2D")
        if conv_rate is not None:
            results[model_name] = conv_rate
    return results


@pytest.fixture
def length_mae_1d() -> dict[str, float]:
    """
    Calculate MAE for length predictions (1D).

    Returns
    -------
    dict[str, float]
        {model_name: mae_value}.
    """
    results = {}
    for model_name in MODELS:
        data = get_converged_data(model_name, "1D")
        if data["ref_geom"] and data["pred_geom"]:
            results[model_name] = mae(data["ref_geom"], data["pred_geom"])
    return results


@pytest.fixture
def energy_mae_1d() -> dict[str, float]:
    """
    Calculate MAE for energy predictions (1D).

    Returns
    -------
    dict[str, float]
        {model_name: mae_value}.
    """
    results = {}
    for model_name in MODELS:
        data = get_converged_data(model_name, "1D")
        if data["ref_energy"] and data["pred_energy"]:
            results[model_name] = mae(data["ref_energy"], data["pred_energy"])
    return results


@pytest.fixture
def convergence_1d() -> dict[str, float]:
    """
    Calculate convergence rate for 1D structures.

    Returns
    -------
    dict[str, float]
        {model_name: convergence_rate}.
    """
    results = {}
    for model_name in MODELS:
        conv_rate = get_convergence_rate(model_name, "1D")
        if conv_rate is not None:
            results[model_name] = conv_rate
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "low_dimensional_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    area_mae_2d: dict[str, float],
    energy_mae_2d: dict[str, float],
    convergence_2d: dict[str, float],
    length_mae_1d: dict[str, float],
    energy_mae_1d: dict[str, float],
    convergence_1d: dict[str, float],
) -> dict[str, dict]:
    """
    Get all low-dimensional relaxation metrics.

    Parameters
    ----------
    area_mae_2d
        Area MAE for 2D structures.
    energy_mae_2d
        Energy MAE for 2D structures.
    convergence_2d
        Convergence rate for 2D structures.
    length_mae_1d
        Length MAE for 1D structures.
    energy_mae_1d
        Energy MAE for 1D structures.
    convergence_1d
        Convergence rate for 1D structures.

    Returns
    -------
    dict[str, dict]
        All metrics for all models.
    """
    return {
        "Area MAE (2D)": area_mae_2d,
        "Energy MAE (2D)": energy_mae_2d,
        "Convergence (2D)": convergence_2d,
        "Length MAE (1D)": length_mae_1d,
        "Energy MAE (1D)": energy_mae_1d,
        "Convergence (1D)": convergence_1d,
    }


def test_low_dimensional_relaxation(
    metrics: dict[str, dict],
    area_predictions_2d: None,
    energy_predictions_2d: None,
    length_predictions_1d: None,
    energy_predictions_1d: None,
) -> None:
    """
    Run low-dimensional relaxation analysis test.

    Parameters
    ----------
    metrics
        All low-dimensional relaxation metrics.
    area_predictions_2d
        Triggers 2D area plot generation.
    energy_predictions_2d
        Triggers 2D energy plot generation.
    length_predictions_1d
        Triggers 1D length plot generation.
    energy_predictions_1d
        Triggers 1D energy plot generation.
    """
    return
