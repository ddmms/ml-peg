"""Analyse high-pressure crystal relaxation benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from sklearn.metrics import mean_absolute_error

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "high_pressure_relaxation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "high_pressure_relaxation"

# Pressure conditions
PRESSURES = [0, 25, 50, 75, 100, 125, 150]
PRESSURE_LABELS = ["P000", "P025", "P050", "P075", "P100", "P125", "P150"]

# Generate colors using viridis colorscale
PRESSURE_COLORS = px.colors.sample_colorscale("viridis", len(PRESSURES))

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def mae(ref: list, pred: list) -> float:
    """
    Calculate Mean Absolute Error.

    Parameters
    ----------
    ref
        Reference values.
    pred
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return mean_absolute_error(ref, pred)


def load_results_for_pressure(model_name: str, pressure_label: str) -> pd.DataFrame:
    """
    Load results for a specific model and pressure.

    Parameters
    ----------
    model_name
        Name of the model.
    pressure_label
        Pressure label (e.g., "P000").

    Returns
    -------
    pd.DataFrame
        Results dataframe or empty dataframe if not found.
    """
    csv_path = CALC_PATH / model_name / f"results_{pressure_label}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def get_converged_data_for_pressure(
    model_name: str, pressure_label: str
) -> tuple[list, list, list, list]:
    """
    Get converged volume and energy data for a model at a specific pressure.

    Parameters
    ----------
    model_name
        Name of the model.
    pressure_label
        Pressure label (e.g., "P000").

    Returns
    -------
    tuple[list, list, list, list]
        Ref volumes, pred volumes, ref energies, pred energies for converged structures.
    """
    df = load_results_for_pressure(model_name, pressure_label)
    if df.empty:
        return [], [], [], []

    # Filter converged structures and remove NaN values
    df_conv = df[df["converged"]].copy()
    df_conv = df_conv.dropna(subset=["pred_volume_per_atom", "pred_energy_per_atom"])

    return (
        df_conv["ref_volume_per_atom"].tolist(),
        df_conv["pred_volume_per_atom"].tolist(),
        df_conv["ref_energy_per_atom"].tolist(),
        df_conv["pred_energy_per_atom"].tolist(),
    )


def get_convergence_rate_for_pressure(
    model_name: str, pressure_label: str
) -> float | None:
    """
    Get convergence rate for a model at a specific pressure.

    Parameters
    ----------
    model_name
        Name of the model.
    pressure_label
        Pressure label (e.g., "P000").

    Returns
    -------
    float | None
        Convergence rate (%) or None if no data.
    """
    df = load_results_for_pressure(model_name, pressure_label)
    if df.empty:
        return None
    return (df["converged"].sum() / len(df)) * 100


def create_pressure_colored_parity_plot(
    data_getter: str,
    title: str,
    x_label: str,
    y_label: str,
    filename: Path,
) -> None:
    """
    Create a parity plot with different colors for each pressure.

    Parameters
    ----------
    data_getter
        Either "volume" or "energy" to select which data to plot.
    title
        Plot title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    filename
        Path to save the plot JSON.
    """
    fig = go.Figure()

    # Add a trace for each pressure
    for pressure, pressure_label, color in zip(
        PRESSURES, PRESSURE_LABELS, PRESSURE_COLORS, strict=False
    ):
        ref_values = []
        pred_values = []

        # Collect data from all models for this pressure
        for model_name in MODELS:
            if data_getter == "volume":
                ref_vol, pred_vol, _, _ = get_converged_data_for_pressure(
                    model_name, pressure_label
                )
                ref_values.extend(ref_vol)
                pred_values.extend(pred_vol)
            else:  # energy
                _, _, ref_energy, pred_energy = get_converged_data_for_pressure(
                    model_name, pressure_label
                )
                ref_values.extend(ref_energy)
                pred_values.extend(pred_energy)

        if ref_values and pred_values:
            fig.add_trace(
                go.Scatter(
                    x=pred_values,
                    y=ref_values,
                    name=f"{pressure} GPa",
                    mode="markers",
                    marker={"color": color, "size": 6, "opacity": 0.7},
                    hovertemplate=(
                        f"<b>{pressure} GPa</b><br>"
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
        legend_title="Pressure",
        hovermode="closest",
    )

    # Save to JSON
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(fig.to_dict(), f)


@pytest.fixture
def volume_predictions() -> None:
    """Create volume parity plot with different colors for each pressure."""
    create_pressure_colored_parity_plot(
        data_getter="volume",
        title="Volume per atom",
        x_label="Predicted volume / Å³/atom",
        y_label="Reference volume / Å³/atom",
        filename=OUT_PATH / "figure_volume.json",
    )


@pytest.fixture
def energy_predictions() -> None:
    """Create energy parity plot with different colors for each pressure."""
    create_pressure_colored_parity_plot(
        data_getter="energy",
        title="Energy per atom",
        x_label="Predicted energy / eV/atom",
        y_label="Reference energy / eV/atom",
        filename=OUT_PATH / "figure_energy.json",
    )


@pytest.fixture
def volume_mae_per_pressure() -> dict[str, dict[str, float]]:
    """
    Calculate MAE for volume predictions at each pressure.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {pressure_gpa: {model_name: mae_value}}.
    """
    results = {}
    for pressure, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        pressure_key = f"Volume MAE ({pressure} GPa)"
        results[pressure_key] = {}
        for model_name in MODELS:
            ref_vol, pred_vol, _, _ = get_converged_data_for_pressure(
                model_name, pressure_label
            )
            if ref_vol and pred_vol:
                results[pressure_key][model_name] = mae(ref_vol, pred_vol)
    return results


@pytest.fixture
def energy_mae_per_pressure() -> dict[str, dict[str, float]]:
    """
    Calculate MAE for energy predictions at each pressure.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {pressure_gpa: {model_name: mae_value}}.
    """
    results = {}
    for pressure, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        pressure_key = f"Energy MAE ({pressure} GPa)"
        results[pressure_key] = {}
        for model_name in MODELS:
            _, _, ref_energy, pred_energy = get_converged_data_for_pressure(
                model_name, pressure_label
            )
            if ref_energy and pred_energy:
                results[pressure_key][model_name] = mae(ref_energy, pred_energy)
    return results


@pytest.fixture
def convergence_per_pressure() -> dict[str, dict[str, float]]:
    """
    Calculate convergence rate at each pressure.

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {pressure_gpa: {model_name: convergence_rate}}.
    """
    results = {}
    for pressure, pressure_label in zip(PRESSURES, PRESSURE_LABELS, strict=False):
        pressure_key = f"Convergence ({pressure} GPa)"
        results[pressure_key] = {}
        for model_name in MODELS:
            conv_rate = get_convergence_rate_for_pressure(model_name, pressure_label)
            if conv_rate is not None:
                results[pressure_key][model_name] = conv_rate
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "high_pressure_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    volume_mae_per_pressure: dict[str, dict[str, float]],
    energy_mae_per_pressure: dict[str, dict[str, float]],
    convergence_per_pressure: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Get all high-pressure relaxation metrics separated by pressure.

    Parameters
    ----------
    volume_mae_per_pressure
        Volume MAE for all models at each pressure.
    energy_mae_per_pressure
        Energy MAE for all models at each pressure.
    convergence_per_pressure
        Convergence rate for all models at each pressure.

    Returns
    -------
    dict[str, dict]
        All metrics for all models.
    """
    all_metrics = {}
    all_metrics.update(volume_mae_per_pressure)
    all_metrics.update(energy_mae_per_pressure)
    all_metrics.update(convergence_per_pressure)
    return all_metrics


def test_high_pressure_relaxation(
    metrics: dict[str, dict],
    volume_predictions: None,
    energy_predictions: None,
) -> None:
    """
    Run high-pressure relaxation analysis test.

    Parameters
    ----------
    metrics
        All high-pressure relaxation metrics.
    volume_predictions
        Triggers volume plot generation.
    energy_predictions
        Triggers energy plot generation.
    """
    return
