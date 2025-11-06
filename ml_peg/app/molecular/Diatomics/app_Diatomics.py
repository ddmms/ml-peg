"""Run Diatomics app."""

from __future__ import annotations

import json

from dash import Dash, Input, Output, callback, dcc
from dash.exceptions import PreventUpdate
from dash.html import Div
import plotly.graph_objects as go
from plotly.io import read_json as plotly_read_json
from plotly.subplots import make_subplots

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Diatomics"
DATA_PATH = APP_ROOT / "data" / "molecular" / "Diatomics"
PERIODIC_TABLE_PATH = DATA_PATH / "periodic_tables"
CURVE_PATH = DATA_PATH / "curves"


class DiatomicsApp(BaseApp):
    """Diatomics benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks for model selection and curve rendering."""
        model_pairs = {
            model: sorted(
                path.stem
                for path in (CURVE_PATH / model).glob("*.json")
                if path.is_file()
            )
            for model in MODELS
            if (CURVE_PATH / model).exists()
        }

        @callback(
            Output(f"{BENCHMARK_NAME}-pair-dropdown", "options"),
            Output(f"{BENCHMARK_NAME}-pair-dropdown", "value"),
            Input(f"{BENCHMARK_NAME}-model-dropdown", "value"),
        )
        def _update_pair_options(model_name: str):
            """
            Return available diatomic pairs for the selected model.

            Parameters
            ----------
            model_name
                Name of the selected model.

            Returns
            -------
            tuple[list[dict[str, str]], str | None]
                Dropdown options and default pair value.
            """
            pairs = model_pairs.get(model_name, [])
            if not pairs:
                return [], None
            options = [{"label": pair, "value": pair} for pair in pairs]
            return options, pairs[0]

        @callback(
            Output(f"{BENCHMARK_NAME}-ptable-graph", "figure"),
            Input(f"{BENCHMARK_NAME}-model-dropdown", "value"),
        )
        def _update_periodic_table(model_name: str):
            """
            Render periodic-table heatmap for a model.

            Parameters
            ----------
            model_name
                Name of the selected model.

            Returns
            -------
            plotly.graph_objs.Figure
                Periodic-table figure.
            """
            fig_path = PERIODIC_TABLE_PATH / f"{model_name}.json"
            if not fig_path.exists():
                raise PreventUpdate
            return plotly_read_json(fig_path)

        @callback(
            Output(f"{BENCHMARK_NAME}-curve-graph", "figure"),
            Input(f"{BENCHMARK_NAME}-model-dropdown", "value"),
            Input(f"{BENCHMARK_NAME}-pair-dropdown", "value"),
        )
        def _update_curve_plot(model_name: str, pair: str | None):
            """
            Render energy/force curve for the selected diatomic pair.

            Parameters
            ----------
            model_name
                Name of the selected model.
            pair
                Identifier of the selected diatomic pair.

            Returns
            -------
            plotly.graph_objs.Figure
                Energy and force curves plotted against distance.
            """
            if not pair:
                raise PreventUpdate
            curve_file = CURVE_PATH / model_name / f"{pair}.json"
            if not curve_file.exists():
                raise PreventUpdate

            with curve_file.open("r", encoding="utf8") as fh:
                payload = json.load(fh)

            distances = payload.get("distance", [])
            energies = payload.get("energy", [])
            forces = payload.get("force_parallel", [])

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=distances,
                    y=energies,
                    mode="lines",
                    name="Energy (eV)",
                ),
                secondary_y=False,
            )
            if forces:
                fig.add_trace(
                    go.Scatter(
                        x=distances,
                        y=forces,
                        mode="lines",
                        name="Force ∥ (eV/Å)",
                        line={"dash": "dot"},
                    ),
                    secondary_y=True,
                )

            fig.update_layout(
                title=f"{model_name}: {pair} curve",
                legend={"orientation": "h"},
            )
            fig.update_xaxes(title="Distance / Å")
            fig.update_yaxes(title="Energy / eV", secondary_y=False)
            fig.update_yaxes(title="Force ∥ / eV·Å⁻¹", secondary_y=True)
            return fig


def get_app() -> DiatomicsApp:
    """
    Get Diatomics benchmark app layout and callback registration.

    Returns
    -------
    DiatomicsApp
        Benchmark layout and callback registration.
    """
    model_options = [
        {"label": model, "value": model}
        for model in MODELS
        if (CURVE_PATH / model).exists()
    ]
    default_model = model_options[0]["value"] if model_options else None

    extra_components = [
        Div(
            [
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-model-dropdown",
                    options=model_options,
                    value=default_model,
                    clearable=False,
                    style={"width": "280px"},
                ),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-pair-dropdown",
                    options=[],
                    value=None,
                    clearable=False,
                    style={"width": "280px", "marginTop": "12px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        dcc.Graph(id=f"{BENCHMARK_NAME}-ptable-graph"),
        Div(style={"height": "20px"}),
        dcc.Graph(id=f"{BENCHMARK_NAME}-curve-graph"),
    ]

    return DiatomicsApp(
        name=BENCHMARK_NAME,
        description=(
            "Physical diagnostics for homo- and heteronuclear diatomic interaction "
            "curves, including well-depth periodic tables and detailed energy-force "
            "profiles."
        ),
        docs_url=None,
        table_path=DATA_PATH / "diatomics_metrics_table.json",
        extra_components=extra_components,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    diatomics_app = get_app()
    dash_app.layout = diatomics_app.layout
    diatomics_app.register_callbacks()
    dash_app.run(port=8055, debug=True)
