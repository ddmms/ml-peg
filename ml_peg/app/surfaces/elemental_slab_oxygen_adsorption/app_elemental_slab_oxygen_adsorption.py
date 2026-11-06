"""Run elemental slab oxygen adsorption app."""

from __future__ import annotations

import re

from dash import Input, Output, callback
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    clear_from_table_cell,
    plot_from_table_cell,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Elemental Slab Oxygen Adsorption"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#elemental-slab-oxygen-adsorption"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "elemental_slab_oxygen_adsorption"
INFO_PATH = DATA_PATH / "info.json"
METRIC_COLUMNS = ("MAE", "U MAE", "non-U MAE")
# Structures are only written out for the mock calculation
STRUCTS_DIR = "mock"
STRUCT_ID = f"{BENCHMARK_NAME}-struct-placeholder"


def _element_to_struct() -> dict[str, str]:
    """
    Map each element to its structure file, e.g. "Fe" -> "Fe54-O.xyz".

    Returns
    -------
    dict[str, str]
        Structure filename, keyed by element symbol.
    """
    element_to_struct = {}
    for struct_file in sorted((DATA_PATH / STRUCTS_DIR).glob("*.xyz")):
        match = re.match(r"([A-Z][a-z]?)\d", struct_file.stem)
        if match:
            element_to_struct[match.group(1)] = struct_file.name
    return element_to_struct


# Energies are single-point, so all models share the mock calculation's structures
ELEMENT_TO_STRUCT = _element_to_struct()


def _struct_from_periodic_table(graph_id: str) -> None:
    """
    Attach callback to show a structure when an element is clicked.

    Parameters
    ----------
    graph_id
        ID for the Dash periodic table being clicked.
    """

    @callback(
        Output(STRUCT_ID, "children", allow_duplicate=True),
        Input(graph_id, "clickData"),
        prevent_initial_call=True,
    )
    def show_struct(click_data):
        """
        Show the slab and adsorbed oxygen for the clicked element.

        Parameters
        ----------
        click_data
            Clicked element in the periodic table.

        Returns
        -------
        Div
            Visualised structure on element click.
        """
        if not click_data:
            raise PreventUpdate
        points = click_data.get("points", [])
        if not points:
            raise PreventUpdate

        # Blank cells of the periodic table have no element to show
        element = points[0].get("text", "").split("<br>")[0].strip()
        if not element:
            raise PreventUpdate

        struct_file = ELEMENT_TO_STRUCT.get(element)
        if not struct_file:
            return Div(f"No structure for {element}.")

        struct = (
            f"/assets/surfaces/elemental_slab_oxygen_adsorption/"
            f"{STRUCTS_DIR}/{struct_file}"
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


class ElementalSlabOxygenAdsorptionApp(BaseApp):
    """Elemental slab oxygen adsorption benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        graph_ids = {}
        for model in MODELS:
            path = DATA_PATH / model / "adsorption_error_periodic_table.json"
            if not path.exists():
                continue
            graph_ids[model] = f"{BENCHMARK_NAME}-{model}-periodic-table"
            cell_to_plot[model] = dict.fromkeys(
                METRIC_COLUMNS, read_plot(path, id=graph_ids[model])
            )

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )

        # Switching model or metric clears the previous model's structure
        clear_from_table_cell(table_id=self.table_id, plot_ids=[STRUCT_ID])

        for graph_id in graph_ids.values():
            _struct_from_periodic_table(graph_id)


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
