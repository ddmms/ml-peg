"""Run equation of state benchmark app."""

from __future__ import annotations

from pathlib import Path

from dash import ALL, Dash, Input, Output, callback, callback_context
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div
import pandas as pd
from plotly.colors import qualitative
import plotly.graph_objects as go
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Equation of State (metals)"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/"
    "benchmarks/bulk_crystal.html#equation-of-state"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "equation_of_state"
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "equation_of_state" / "outputs"
INPUT_PATH = Path(__file__).parents[4] / "inputs" / "bulk_crystal" / "equation_of_state"
_PT_TYPE = "eos-periodic-table"
_EOS_CURVE_ID = f"{BENCHMARK_NAME}-eos-curve"
_METRICS = [
    ("\u0394", "delta_periodic_table"),
    ("Phase energy", "phase_energy_periodic_table"),
    ("Phase stability", "phase_stability_periodic_table"),
]


def _make_eos_figure(model: str, element: str) -> go.Figure | None:
    """
    Create an equation of state figure for a given model and element.

    Parameters
    ----------
    model : str
        The model name.
    element : str
        The element name.

    Returns
    -------
    go.Figure | None
        The equation of state figure or None if the data is not available.
    """
    model_csv = CALC_PATH / model / f"{element}_eos_results.csv"
    dft_csv = INPUT_PATH / f"{element}_eos_DFT.csv"
    if not model_csv.exists() or not dft_csv.exists():
        return None
    model_data = pd.read_csv(model_csv)
    dft_data = pd.read_csv(dft_csv, comment="#")
    phases = [
        col.split("_")[1]
        for col in dft_data.columns
        if col.startswith("Delta_") and col.endswith("_E")
    ]
    colours = qualitative.D3
    fig = go.Figure()
    for i, phase in enumerate(phases):
        colour = colours[i % len(colours)]
        dft_v = dft_data[f"V/atom_{phase}"].dropna()
        dft_e = dft_data[f"Delta_{phase}_E"].loc[dft_v.index]
        fig.add_trace(
            go.Scatter(
                x=dft_v,
                y=dft_e,
                mode="markers",
                name=f"DFT {phase}",
                marker={"symbol": "x", "color": colour, "size": 8},
            )
        )
        model_v = model_data["V/atom"]
        model_delta_e = model_data[f"{phase}_E"] - model_data[f"{phases[0]}_E"].min()
        fig.add_trace(
            go.Scatter(
                x=model_v,
                y=model_delta_e,
                mode="lines",
                name=f"{model} {phase}",
                line={"color": colour},
            )
        )
    fig.update_layout(
        title=f"EOS - {element} ({model})",
        xaxis_title="Volume per atom (\u00c5\u00b3)",
        yaxis_title="Energy per atom (eV)",
    )
    return fig


class EquationOfStateApp(BaseApp):
    """Equation of State benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        for model in MODELS:
            plots = {}
            for column_id, file_suffix in _METRICS:
                path = DATA_PATH / model / f"{file_suffix}.json"
                if not path.exists():
                    continue
                plots[column_id] = Graph(
                    id={
                        "type": _PT_TYPE,
                        "model": model,
                        "metric": file_suffix,
                    },
                    figure=read_json(path),
                )
            if plots:
                cell_to_plot[model] = plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )

        @callback(
            Output(_EOS_CURVE_ID, "children"),
            Input(
                {"type": _PT_TYPE, "model": ALL, "metric": ALL},
                "clickData",
            ),
            prevent_initial_call=True,
        )
        def show_eos_curve(_):
            """
            Show the equation of state curve for the clicked element and model.

            Parameters
            ----------
            _ : Any
                The click data from the periodic table graph.
                The actual value is not used, but the callback
                context is used to determine which cell was clicked.

            Returns
            -------
            Div
                The div containing the equation of state figure.
            """
            ctx = callback_context
            triggered_id = ctx.triggered_id
            if not isinstance(triggered_id, dict):
                raise PreventUpdate
            click_data = ctx.triggered[0]["value"]
            if not click_data:
                raise PreventUpdate
            points = click_data.get("points", [])
            if not points:
                raise PreventUpdate
            text = points[0].get("text", "")
            element = text.split("<br>")[0].strip()
            if not element or len(element) > 3:
                raise PreventUpdate
            model = triggered_id["model"]
            fig = _make_eos_figure(model, element)
            if fig is None:
                return Div(f"No data for {element} / {model}.")
            return Div(Graph(figure=fig))


def get_app() -> EquationOfStateApp:
    """
    Get equation of state benchmark app layout and callback registration.

    Returns
    -------
    EquationOfStateApp
        Benchmark layout and callback registration.
    """
    return EquationOfStateApp(
        name=BENCHMARK_NAME,
        description=(
            "Equation of state curves and phase stability for BCC metals "
            "(W, Mo, Nb), benchmarked against PBE reference data."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "eos_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=_EOS_CURVE_ID),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    equation_of_state_app = get_app()
    full_app.layout = equation_of_state_app.layout
    equation_of_state_app.register_callbacks()
    full_app.run(port=8054, debug=True)
