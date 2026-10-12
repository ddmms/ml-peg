"""Run ACWF equation-of-state benchmark app."""

from __future__ import annotations

from dash import ALL, Input, Output, callback, callback_context
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Equation of state (ACWF)"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/"
    "benchmarks/bulk_crystal.html#equation-of-state-acwf"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "equation_of_state_acwf"
PT_TYPE = "eos-acwf-periodic-table"
EOS_CURVE_ID = f"{BENCHMARK_NAME}-eos-curve"
METRICS = [
    ("ε", "epsilon_periodic_table"),
    ("ν", "nu_periodic_table"),
    ("Unphysical E(V)", "unphysical_periodic_table"),
    ("Δ", "delta_periodic_table"),
]
INFO_PATH = DATA_PATH / "info.json"


class EquationOfStateACWFApp(BaseApp):
    """ACWF equation-of-state benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        for model in MODELS:
            plots = {}
            for column_id, file_stem in METRICS:
                path = DATA_PATH / model / f"{file_stem}.json"
                if not path.exists():
                    continue
                plots[column_id] = Graph(
                    id={"type": PT_TYPE, "model": model, "metric": file_stem},
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
            Output(EOS_CURVE_ID, "children"),
            Input({"type": PT_TYPE, "model": ALL, "metric": ALL}, "clickData"),
            prevent_initial_call=True,
        )
        def show_eos_curves(_):
            """
            Show the E(V) curves of the clicked element for the selected model.

            Parameters
            ----------
            _ : Any
                Click data from the periodic table. The callback context is used to
                find which element and model were clicked.

            Returns
            -------
            Div
                The E(V) figure of the clicked element.
            """
            triggered_id = callback_context.triggered_id
            if not isinstance(triggered_id, dict):
                raise PreventUpdate
            click_data = callback_context.triggered[0]["value"]
            if not click_data or not click_data.get("points"):
                raise PreventUpdate
            element = click_data["points"][0].get("text", "").split("<br>")[0].strip()
            if not element or len(element) > 3:
                raise PreventUpdate
            model = triggered_id["model"]
            fig_path = DATA_PATH / model / f"{element}_eos_figure.json"
            if not fig_path.exists():
                return Div(f"No data for {element} / {model}.")
            return Div(Graph(figure=read_json(str(fig_path))))


def get_app() -> EquationOfStateACWFApp:
    """
    Get ACWF equation-of-state benchmark app layout and callback registration.

    Returns
    -------
    EquationOfStateACWFApp
        Benchmark layout and callback registration.
    """
    return EquationOfStateACWFApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy-volume curves of 960 unary and oxide crystals (Z = 1-96), "
            "compared with all-electron PBE reference data using the ε and ν "
            "measures of the ACWF verification study. Click a metric to show a "
            "periodic table, then an element to show its E(V) curves."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "eos_acwf_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=EOS_CURVE_ID),
        ],
        info_path=INFO_PATH,
    )
