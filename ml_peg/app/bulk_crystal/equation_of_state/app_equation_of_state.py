"""Run equation of state benchmark app."""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, callback, callback_context
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Equation of State (metals)"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/"
    "benchmarks/bulk_crystal.html#equation-of-state"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "equation_of_state"
PT_TYPE = "eos-periodic-table"
EOS_CURVE_ID = f"{BENCHMARK_NAME}-eos-curve"
METRICS = [
    ("\u0394", "delta_periodic_table"),
    ("Phase energy", "phase_energy_periodic_table"),
    ("Phase stability", "phase_stability_periodic_table"),
]


class EquationOfStateApp(BaseApp):
    """Equation of State benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        for model in MODELS:
            plots = {}
            for column_id, file_suffix in METRICS:
                path = DATA_PATH / model / f"{file_suffix}.json"
                if not path.exists():
                    continue
                plots[column_id] = Graph(
                    id={
                        "type": PT_TYPE,
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
            Output(EOS_CURVE_ID, "children"),
            Input(
                {"type": PT_TYPE, "model": ALL, "metric": ALL},
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
            fig_path = DATA_PATH / model / f"{element}_eos_figure.json"
            if not fig_path.exists():
                return Div(f"No data for {element} / {model}.")
            return Div(Graph(figure=read_json(str(fig_path))))


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
            "(W, Nb, Mo, Ta, Ti, Zr, Cr, Fe),"
            " benchmarked against PBE reference data."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "eos_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=EOS_CURVE_ID),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    equation_of_state_app = get_app()
    full_app.layout = equation_of_state_app.layout
    equation_of_state_app.register_callbacks()
    full_app.run(port=8054, debug=True)
