"""Run iron properties app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback, dcc
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label
import pandas as pd
import plotly.graph_objects as go

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Iron Properties"
DATA_PATH = APP_ROOT / "data" / "physicality" / "iron_properties"
CALC_PATH = CALCS_ROOT / "physicality" / "iron_properties" / "outputs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#iron-properties"


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

    file_map = {
        "eos": "eos_curve.csv",
        "bain": "bain_path.csv",
        "sfe_110": "sfe_110_curve.csv",
        "sfe_112": "sfe_112_curve.csv",
        "crack_1": "crack_1_KE.csv",
        "crack_2": "crack_2_KE.csv",
        "crack_3": "crack_3_KE.csv",
        "crack_4": "crack_4_KE.csv",
    }

    filename = file_map.get(curve_type)
    if not filename:
        return None

    csv_path = model_dir / filename
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
    fig = go.Figure()

    if curve_type == "eos":
        fig.add_trace(
            go.Scatter(
                x=df["volume"],
                y=df["energy"],
                mode="lines+markers",
                name=model_name,
                line={"width": 2},
                marker={"size": 6},
            )
        )
        fig.update_layout(
            title=f"Equation of State - {model_name}",
            xaxis_title="Volume (Å³/atom)",
            yaxis_title="Energy (eV/atom)",
        )

    elif curve_type == "bain":
        fig.add_trace(
            go.Scatter(
                x=df["ca_ratio"],
                y=df["energy_meV"],
                mode="lines+markers",
                name=model_name,
                line={"width": 2},
                marker={"size": 6},
            )
        )
        # Add vertical lines for BCC and FCC
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="BCC")
        fig.add_vline(
            x=1.414, line_dash="dash", line_color="gray", annotation_text="FCC"
        )
        fig.update_layout(
            title=f"Bain Path - {model_name}",
            xaxis_title="c/a ratio",
            yaxis_title="Energy (meV/atom)",
        )

    elif curve_type == "sfe_110":
        fig.add_trace(
            go.Scatter(
                x=df["displacement"],
                y=df["sfe_J_per_m2"],
                mode="lines+markers",
                name=model_name,
                line={"width": 2},
                marker={"size": 6},
            )
        )
        fig.update_layout(
            title=f"Stacking Fault Energy {{110}}<111> - {model_name}",
            xaxis_title="Displacement (Å)",
            yaxis_title="SFE (J/m²)",
        )

    elif curve_type == "sfe_112":
        fig.add_trace(
            go.Scatter(
                x=df["displacement"],
                y=df["sfe_J_per_m2"],
                mode="lines+markers",
                name=model_name,
                line={"width": 2},
                marker={"size": 6},
            )
        )
        fig.update_layout(
            title=f"Stacking Fault Energy {{112}}<111> - {model_name}",
            xaxis_title="Displacement (Å)",
            yaxis_title="SFE (J/m²)",
        )

    elif curve_type.startswith("crack_"):
        crack_names = {
            "crack_1": "(100)[010]",
            "crack_2": "(100)[001]",
            "crack_3": "(110)[001]",
            "crack_4": "(110)[1-10]",
        }
        crack_name = crack_names.get(curve_type, curve_type)
        fig.add_trace(
            go.Scatter(
                x=df["K"],
                y=df["energy"],
                mode="lines+markers",
                name=model_name,
                line={"width": 2},
                marker={"size": 6},
            )
        )
        fig.update_layout(
            title=f"Crack K-E Curve {crack_name} - {model_name}",
            xaxis_title="K (MPa√m)",
            yaxis_title="Energy (eV)",
        )

    fig.update_layout(
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
                        {"label": "(100)[010] K-E Curve", "value": "crack_1"},
                        {"label": "(100)[001] K-E Curve", "value": "crack_2"},
                        {"label": "(110)[001] K-E Curve", "value": "crack_3"},
                        {"label": "(110)[1-10] K-E Curve", "value": "crack_4"},
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
            "{112}<111> slip systems, "
            "dislocation core energies for 5 dislocation types (edge, mixed, screw), "
            "and crack K-tests for 4 crack systems. "
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
