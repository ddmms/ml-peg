"""Run elemental slab oxygen adsorption app."""

from __future__ import annotations

import re

from dash import ALL, Input, Output, callback, callback_context
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Elemental Slab Oxygen Adsorption"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#elemental-slab-oxygen-adsorption"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "elemental_slab_oxygen_adsorption"
INFO_PATH = DATA_PATH / "info.json"
PT_TYPE = "oxygen-adsorption-periodic-table"
METRIC_COLUMNS = ("MAE", "U MAE", "non-U MAE")
STRUCT_ID = f"{BENCHMARK_NAME}-struct-placeholder"


class ElementalSlabOxygenAdsorptionApp(BaseApp):
    """Elemental slab oxygen adsorption benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        for model in MODELS:
            path = DATA_PATH / model / "adsorption_error_periodic_table.json"
            if not path.exists():
                continue
            graph = Graph(
                id={"type": PT_TYPE, "model": model},
                figure=read_json(path),
            )
            cell_to_plot[model] = dict.fromkeys(METRIC_COLUMNS, graph)

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )

        # Map each element to its structure file, e.g. "Fe" -> "Fe54-O.xyz"
        element_to_struct = {}
        for struct_file in sorted((DATA_PATH / MODELS[0]).glob("*.xyz")):
            match = re.match(r"([A-Z][a-z]?)\d", struct_file.stem)
            if match:
                element_to_struct[match.group(1)] = struct_file.name

        @callback(
            Output(STRUCT_ID, "children"),
            Input({"type": PT_TYPE, "model": ALL}, "clickData"),
            prevent_initial_call=True,
        )
        def show_struct(_):
            """
            Show the slab and adsorbed oxygen for the clicked element and model.

            Parameters
            ----------
            _
                Click data from the periodic table graph. The callback context is
                used instead, to determine which model's table was clicked.

            Returns
            -------
            Div
                Visualised structure on element click.
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

            element = points[0].get("text", "").split("<br>")[0].strip()
            struct_file = element_to_struct.get(element)
            model = triggered_id["model"]
            if not struct_file:
                return Div(f"No structure for {element}.")

            struct = (
                "/assets/surfaces/elemental_slab_oxygen_adsorption/"
                f"{model}/{struct_file}"
            )
            return Div(
                Iframe(
                    srcDoc=generate_weas_html(struct),
                    style={
                        "height": "550px",
                        "width": "100%",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                    },
                )
            )


def get_app() -> ElementalSlabOxygenAdsorptionApp:
    """
    Get elemental slab oxygen adsorption benchmark app layout and callback registration.

    Returns
    -------
    ElementalSlabOxygenAdsorptionApp
        Benchmark layout and callback registration.
    """
    return ElementalSlabOxygenAdsorptionApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting adsorption energies of oxygen "
            "on elemental slabs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "elemental_slab_oxygen_adsorption_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=STRUCT_ID),
        ],
        info_path=INFO_PATH,
    )
