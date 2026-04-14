"""Run iron properties app."""

from __future__ import annotations

import json

from dash import Dash, Input, Output, callback, dcc
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label
import plotly.graph_objects as go

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.plot_helpers import figure_from_dict
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Iron Properties"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "iron_properties"
FIGURES_PATH = DATA_PATH / "figures"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#iron-properties"


def _load_figure(model_name: str, curve_type: str) -> go.Figure | None:
    """
    Load a pre-created figure from JSON.

    Parameters
    ----------
    model_name : str
        Name of the model.
    curve_type : str
        Type of curve (e.g., 'eos', 'bain', 'sfe_110').

    Returns
    -------
    go.Figure or None
        Plotly figure, or None if not found.
    """
    fig_path = FIGURES_PATH / f"{model_name}_{curve_type}.json"
    if not fig_path.exists():
        return None
    return figure_from_dict(json.loads(fig_path.read_text()))


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
        def update_figure(model_name: str, curve_type: str) -> go.Figure:  # noqa: F811
            # Invoked by Dash's callback system, not called directly.
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

            fig = _load_figure(model_name, curve_type)
            if fig is None:
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
