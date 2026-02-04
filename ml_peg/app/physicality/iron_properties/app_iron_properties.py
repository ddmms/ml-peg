"""Run iron properties app."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, callback, dcc
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.iron_utils import load_dft_curve
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Iron Properties"
DATA_PATH = APP_ROOT / "data" / "physicality" / "iron_properties"
CALC_PATH = CALCS_ROOT / "physicality" / "iron_properties" / "outputs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#iron-properties"

# Path to DFT reference data
DFT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "iron_properties"


# Curve configuration: file name, x column, y column, title, x label, y label
CURVE_CONFIG = {
    "eos": {
        "file": "eos_curve.csv",
        "x": "volume",
        "y": "energy",
        "title": "Equation of State",
        "x_label": "Volume (Å³/atom)",
        "y_label": "Energy (eV/atom)",
    },
    "bain": {
        "file": "bain_path.csv",
        "x": "ca_ratio",
        "y": "energy_meV",
        "title": "Bain Path",
        "x_label": "c/a ratio",
        "y_label": "Energy (meV/atom)",
    },
    "sfe_110": {
        "file": "sfe_110_curve.csv",
        "x": "displacement",
        "y": "sfe_J_per_m2",
        "title": "Stacking Fault Energy {110}<111>",
        "x_label": "Displacement (Å)",
        "y_label": "SFE (J/m²)",
    },
    "sfe_112": {
        "file": "sfe_112_curve.csv",
        "x": "displacement",
        "y": "sfe_J_per_m2",
        "title": "Stacking Fault Energy {112}<111>",
        "x_label": "Displacement (Å)",
        "y_label": "SFE (J/m²)",
    },
    "ts_100": {
        "file": "ts_100_curve.csv",
        "x": "separation",
        "y": "traction",
        "title": "Traction-Separation Curve (100)",
        "x_label": "Separation (Å)",
        "y_label": "Traction (GPa)",
    },
    "ts_110": {
        "file": "ts_110_curve.csv",
        "x": "separation",
        "y": "traction",
        "title": "Traction-Separation Curve (110)",
        "x_label": "Separation (Å)",
        "y_label": "Traction (GPa)",
    },
}

# DFT reference curve configuration
DFT_CURVE_CONFIG = {
    "bain": {
        "file": "BainPath_DFT.csv",
        "sep": ",",
        "decimal": ".",
        "header": None,
        "x_col": 0,
        "y_col": 1,
        "normalize_energy_mev": True,
    },
    "sfe_110": {
        "file": "sfe_110_dft.csv",
        "sep": ",",
        "decimal": ".",
        "header": None,
        "x_col": 0,
        "y_col": 1,
        "x_scale": 2.831 * 1.7320508 / 2,  # Burgers vector: a * sqrt(3) / 2
    },
    "sfe_112": {
        "file": "sfe_112_dft.csv",
        "sep": ",",
        "decimal": ".",
        "header": None,
        "x_col": 0,
        "y_col": 1,
        "x_scale": 2.831 * 1.7320508 / 2,  # Burgers vector: a * sqrt(3) / 2
    },
    "ts_100": {
        "file": "ts_100_dft.csv",
        "sep": r"\s+",
        "decimal": ".",
        "header": None,
        "x_col": 0,
        "y_col": 1,
    },
    "ts_110": {
        "file": "ts_110_dft.csv",
        "sep": r"\s+",
        "decimal": ".",
        "header": None,
        "x_col": 0,
        "y_col": 1,
    },
}


def _load_curve_data(model_name: str, curve_type: str) -> pd.DataFrame | None:
    """
    Load curve data for a model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    curve_type : str
        Type of curve to load (e.g., 'eos', 'bain', 'sfe_110').

    Returns
    -------
    pd.DataFrame or None
        DataFrame with curve data, or None if not found.
    """
    model_dir = CALC_PATH / model_name
    if not model_dir.exists():
        return None

    config = CURVE_CONFIG.get(curve_type)
    if not config:
        return None

    csv_path = model_dir / config["file"]
    if not csv_path.exists():
        return None

    return pd.read_csv(csv_path)


def _create_figure(df: pd.DataFrame, curve_type: str, model_name: str) -> go.Figure:
    """
    Create plotly figure for the given curve type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the curve data.
    curve_type : str
        Type of curve to plot (e.g., 'eos', 'bain', 'sfe_110').
    model_name : str
        Name of the model for the title.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    config = CURVE_CONFIG.get(curve_type, {})

    fig = go.Figure()

    # Get model data x-range for limiting DFT reference
    model_x_max = df[config["x"]].max() if config.get("x") else None

    # Add DFT reference curve if available
    dft_data = load_dft_curve(curve_type, DFT_DATA_PATH, DFT_CURVE_CONFIG)
    if dft_data is not None:
        x_dft, y_dft = dft_data

        # For SFE curves, limit DFT data to model x-range (data is periodic)
        if curve_type.startswith("sfe_") and model_x_max is not None:
            mask = x_dft <= model_x_max
            x_dft = np.array(x_dft)[mask]
            y_dft = np.array(y_dft)[mask]

        fig.add_trace(
            go.Scatter(
                x=x_dft,
                y=y_dft,
                mode="lines",
                name="DFT Reference",
                line={"width": 2, "dash": "dash", "color": "gray"},
            )
        )

    # Add model curve
    fig.add_trace(
        go.Scatter(
            x=df[config["x"]],
            y=df[config["y"]],
            mode="lines+markers",
            name=model_name,
            line={"width": 2},
            marker={"size": 6},
        )
    )

    # Special handling for Bain path (add BCC/FCC reference lines)
    if curve_type == "bain":
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="BCC")
        fig.add_vline(
            x=1.414, line_dash="dash", line_color="gray", annotation_text="FCC"
        )

    fig.update_layout(
        title=f"{config['title']} - {model_name}",
        xaxis_title=config["x_label"],
        yaxis_title=config["y_label"],
        template="plotly_white",
        showlegend=True,
        height=500,
    )

    return fig


class IronPropertiesApp(BaseApp):
    """Iron properties benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks for curve visualization."""
        model_dropdown_id = f"{BENCHMARK_NAME}-model-dropdown"
        curve_dropdown_id = f"{BENCHMARK_NAME}-curve-dropdown"
        figure_id = f"{BENCHMARK_NAME}-figure"

        @callback(
            Output(figure_id, "figure"),
            Input(model_dropdown_id, "value"),
            Input(curve_dropdown_id, "value"),
        )
        def update_figure(model_name: str, curve_type: str) -> go.Figure:
            """
            Update figure based on model and curve selection.

            Parameters
            ----------
            model_name : str
                Name of the selected model.
            curve_type : str
                Type of curve to display.

            Returns
            -------
            go.Figure
                Updated plotly figure.
            """
            if not model_name or not curve_type:
                raise PreventUpdate

            df = _load_curve_data(model_name, curve_type)
            if df is None or df.empty:
                # Return empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No data available for {model_name} - {curve_type}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font={"size": 16},
                )
                fig.update_layout(
                    template="plotly_white",
                    height=500,
                )
                return fig

            return _create_figure(df, curve_type, model_name)


def get_app() -> IronPropertiesApp:
    """
    Get iron properties benchmark app layout and callback registration.

    Returns
    -------
    IronPropertiesApp
        Benchmark layout and callback registration.
    """
    model_options = [{"label": model, "value": model} for model in MODELS]
    default_model = model_options[0]["value"] if model_options else None

    extra_components = [
        Div(
            [
                Label("Select model for curve visualization:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-model-dropdown",
                    options=model_options,
                    value=default_model,
                    clearable=False,
                    style={"width": "300px", "marginBottom": "20px"},
                ),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-curve-dropdown",
                    options=[
                        {"label": "EOS Curve", "value": "eos"},
                        {"label": "Bain Path", "value": "bain"},
                        {"label": "SFE {110}<111>", "value": "sfe_110"},
                        {"label": "SFE {112}<111>", "value": "sfe_112"},
                        {"label": "T-S Curve (100)", "value": "ts_100"},
                        {"label": "T-S Curve (110)", "value": "ts_110"},
                    ],
                    value="eos",
                    clearable=False,
                    style={"width": "300px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        Loading(
            dcc.Graph(
                id=f"{BENCHMARK_NAME}-figure",
                style={"height": "500px", "width": "100%", "marginTop": "20px"},
            ),
            type="circle",
        ),
    ]

    return IronPropertiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Comprehensive BCC iron properties benchmark. "
            "Includes equation of state (lattice parameter, bulk modulus), "
            "elastic constants (C11, C12, C44), Bain path (BCC-FCC transformation), "
            "vacancy formation energy, surface energies (100, 110, 111, 112), "
            "generalized stacking fault energy curves for {110}<111> and "
            "{112}<111> slip systems, and traction-separation curves for (100) "
            "and (110) cleavage planes. "
            "This benchmark is computationally expensive and marked with "
            "@pytest.mark.slow."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "iron_properties_metrics_table.json",
        extra_components=extra_components,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    iron_properties_app = get_app()
    dash_app.layout = iron_properties_app.layout
    iron_properties_app.register_callbacks()
    dash_app.run(port=8060, debug=True)
