"""Run Alexandria convex-hull benchmark app."""

from __future__ import annotations

from functools import lru_cache
import json

from dash import ALL, Input, Output, State, callback, clientside_callback, ctx
from dash.dcc import Dropdown, Graph, Store
from dash.exceptions import PreventUpdate
from dash.html import H3, Div, P
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    plot_with_download_controls,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Convex hull (Alexandria)"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/"
    "benchmarks/bulk_crystal.html#convex-hull-alexandria"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "convex_hull_alexandria"
INFO_PATH = DATA_PATH / "info.json"

HULL_MODEL_ID = f"{BENCHMARK_NAME}-hull-model"
HULL_SYSTEM_ID = f"{BENCHMARK_NAME}-hull-system"
HULL_FIGURE_ID = f"{BENCHMARK_NAME}-hull-figure"
HULL_GRAPH_ID = f"{BENCHMARK_NAME}-hull-graph"
HULL_STRUCT_ID = f"{BENCHMARK_NAME}-hull-struct"
# Element clicked on a periodic table, and the model of that table
HULL_ELEMENT_ID = f"{BENCHMARK_NAME}-hull-element"
PERIODIC_TABLE_TYPE = f"{BENCHMARK_NAME}-periodic-table"
# Structure of each point of the hulls, named by its ``mat_id``
STRUCT_TEMPLATE = "/assets/bulk_crystal/convex_hull_alexandria/structures/{id}.xyz"

# Metric column: density-scatter, violin and periodic-table files (None if not shown)
PLOTS = {
    "Energy above hull MAE": (
        "figure_e_hull_density",
        "figure_e_hull_violin",
        "e_hull_periodic_table",
    ),
    "Stability F1": (None, "figure_e_hull_violin", None),
    "Failed systems": (None, None, "failed_systems_periodic_table"),
    "Formation energy MAE": (
        "figure_e_form_density",
        "figure_e_form_violin",
        "e_form_periodic_table",
    ),
    "Formation energy RMSE": (None, "figure_e_form_violin", None),
    "Formation energy R²": (None, "figure_e_form_violin", None),
}


@lru_cache(maxsize=8)
def system_options(model: str) -> list[dict[str, str]]:
    """
    Get the systems of a model, sorted by decreasing energy above hull MAE.

    Parameters
    ----------
    model
        Name of the model.

    Returns
    -------
    list[dict[str, str]]
        Dropdown options, labelled with the MAE of each system, and searchable with
        the elements in either order, such as ``O-Si`` or ``Si-O``.
    """
    with open(DATA_PATH / model / "hull_systems.json", encoding="utf8") as file:
        maes = json.load(file)
    options = []
    for system, mae in maes.items():
        first, second = system.split("-")
        options.append(
            {
                "label": f"{system} (E_hull MAE {mae:.0f} meV/atom)",
                "value": system,
                "search": f"{system} {second}-{first}",
            }
        )
    return options


def clicked_element(click_data: dict | None) -> str | None:
    """
    Get the element clicked on a periodic table.

    Parameters
    ----------
    click_data
        Clicked data of the periodic table.

    Returns
    -------
    str | None
        Symbol of the element, or ``None`` if an empty cell was clicked.
    """
    if not click_data:
        return None
    # Each cell's text starts with the symbol of its element, such as ``H<br>...``
    element = click_data["points"][0].get("text", "").split("<br>")[0]
    return element or None


def element_system(model: str, element: str) -> str | None:
    """
    Get the system with the largest MAE that contains an element.

    Parameters
    ----------
    model
        Name of the model.
    element
        Chemical symbol.

    Returns
    -------
    str | None
        System, or ``None`` if no successful system of the model contains it.
    """
    for option in system_options(model):
        if element in option["value"].split("-"):
            return option["value"]
    return None


def hull_view(model: str, system: str | None) -> Div | P:
    """
    Load the PBE and model hulls of a system, as written by the analysis.

    Parameters
    ----------
    model
        Name of the model.
    system
        System, such as ``Cu-Zn``, or ``None`` if the model has no successful system.

    Returns
    -------
    Div | P
        Side-by-side hull figure with download controls, or a message naming the
        model, and the system if its figure is missing.
    """
    if not system:
        return P(f"No successful systems for {model}.")
    path = DATA_PATH / model / "hulls" / f"{system}.json"
    if not path.exists():
        return P(f"No hull figure for {system} with {model}.")
    return plot_with_download_controls(Graph(id=HULL_GRAPH_ID, figure=read_json(path)))


class ConvexHullAlexandriaApp(BaseApp):
    """Alexandria convex-hull benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        # Each violin figure holds all the models, so load it once for all cells
        violins = {}
        for _, violin_stem, _ in PLOTS.values():
            if violin_stem is None or violin_stem in violins:
                continue
            violin_path = DATA_PATH / f"{violin_stem}.json"
            if violin_path.exists():
                violins[violin_stem] = Graph(figure=read_json(violin_path))
        for model in MODELS:
            plots = {}
            for column_id, (density_stem, violin_stem, table_stem) in PLOTS.items():
                components = []
                if density_stem is not None:
                    density_path = DATA_PATH / f"{density_stem}.json"
                    if density_path.exists():
                        density = read_density_plot_for_model(
                            filename=density_path,
                            model=model,
                            id=f"{BENCHMARK_NAME}-{model}-{density_stem}",
                        )
                        if density is not None:
                            components.append(density)
                if violin_stem in violins:
                    components.append(violins[violin_stem])
                if table_stem is not None:
                    table_path = DATA_PATH / model / f"{table_stem}.json"
                    if table_path.exists():
                        # Clicking an element shows the hulls of its worst system
                        table = Graph(
                            id={
                                "type": PERIODIC_TABLE_TYPE,
                                "model": model,
                                "table": table_stem,
                            },
                            figure=read_json(table_path),
                        )
                        components.append(table)
                if components:
                    plots[column_id] = Div(components)
            if plots:
                cell_to_plot[model] = plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )

        @callback(
            Output(HULL_MODEL_ID, "value"),
            Output(HULL_ELEMENT_ID, "data"),
            Input(
                {"type": PERIODIC_TABLE_TYPE, "model": ALL, "table": ALL}, "clickData"
            ),
            prevent_initial_call=True,
        )
        def select_element(click_data):
            """
            Select the model of the periodic table on which an element was clicked.

            Parameters
            ----------
            click_data : list[dict | None]
                Clicked data of each periodic table shown.

            Returns
            -------
            tuple[str, str]
                Model of the clicked table, and the clicked element.
            """
            element = clicked_element(ctx.triggered[0]["value"])
            if not element or not isinstance(ctx.triggered_id, dict):
                raise PreventUpdate
            return ctx.triggered_id["model"], element

        @callback(
            Output(HULL_SYSTEM_ID, "options"),
            Output(HULL_SYSTEM_ID, "value"),
            Input(HULL_MODEL_ID, "value"),
            Input(HULL_ELEMENT_ID, "data"),
        )
        def update_systems(model, element):
            """
            List the systems of the selected model, worst first.

            Parameters
            ----------
            model : str | None
                Selected model.
            element : str | None
                Element last clicked on a periodic table.

            Returns
            -------
            tuple[list[dict[str, str]], str]
                Dropdown options, and the system with the largest MAE, containing the
                element if one was just clicked.
            """
            if not model:
                raise PreventUpdate
            options = system_options(model)
            system = options[0]["value"] if options else None
            # Only when an element was just clicked, not on a later change of model
            if element and HULL_ELEMENT_ID in ctx.triggered_prop_ids.values():
                system = element_system(model, element) or system
            return options, system

        # Triggered by the system only, which is set again whenever the model changes,
        # so that the model and system always match
        @callback(
            Output(HULL_FIGURE_ID, "children"),
            Input(HULL_SYSTEM_ID, "value"),
            State(HULL_MODEL_ID, "value"),
        )
        def show_hulls(system, model):
            """
            Show the PBE and model hulls of the selected system.

            Parameters
            ----------
            system : str | None
                Selected system.
            model : str | None
                Selected model.

            Returns
            -------
            Div | P
                Side-by-side hull figure, or a message if there is none.
            """
            if not model:
                raise PreventUpdate
            return hull_view(model, system)

        # A structure belongs to the hulls it was clicked on
        clientside_callback(
            "() => null",
            Output(HULL_STRUCT_ID, "children", allow_duplicate=True),
            Input(HULL_FIGURE_ID, "children"),
            prevent_initial_call=True,
        )
        struct_from_scatter(
            HULL_GRAPH_ID, HULL_STRUCT_ID, struct_template=STRUCT_TEMPLATE
        )


def get_app() -> ConvexHullAlexandriaApp:
    """
    Get Alexandria convex-hull benchmark app layout and callback registration.

    Returns
    -------
    ConvexHullAlexandriaApp
        Benchmark layout and callback registration.
    """
    models = [m for m in MODELS if (DATA_PATH / m / "hull_systems.json").exists()]
    return ConvexHullAlexandriaApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy above the convex hull of 41,048 binary crystals from the "
            "Alexandria PBE database (single points on the PBE structures), with "
            "the hull of each system built from the model energies. Click a metric "
            "to show the parity plot, error distributions and per-element errors. "
            "Below, compare the PBE and model hulls of any system (or click an "
            "element on a periodic table to show its worst system), and click a "
            "point to show its structure."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "convex_hull_alexandria_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            H3("PBE and model convex hulls"),
            Div(
                [
                    Dropdown(
                        id=HULL_MODEL_ID,
                        options=models,
                        value=models[0] if models else None,
                        clearable=False,
                        maxHeight=500,
                        style={"width": "300px"},
                    ),
                    Dropdown(
                        id=HULL_SYSTEM_ID,
                        clearable=False,
                        searchable=True,
                        maxHeight=500,
                        placeholder="Type a system, such as Si-O",
                        style={"width": "400px"},
                    ),
                ],
                style={"display": "flex", "gap": "10px"},
            ),
            Store(id=HULL_ELEMENT_ID),
            Div(id=HULL_FIGURE_ID),
            Div(id=HULL_STRUCT_ID),
        ],
        info_path=INFO_PATH,
    )
