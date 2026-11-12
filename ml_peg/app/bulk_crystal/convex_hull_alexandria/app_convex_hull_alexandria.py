"""Run Alexandria convex-hull benchmark app."""

from __future__ import annotations

from functools import lru_cache
import json

from dash import Input, Output, callback
from dash.dcc import Dropdown, Graph
from dash.exceptions import PreventUpdate
from dash.html import H3, Div, P
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
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

# Metric column: density-scatter, violin and periodic-table files (None if not shown)
PLOTS = {
    "Energy above hull MAE": (
        "figure_e_hull_density",
        "figure_e_hull_violin",
        "e_hull_periodic_table",
    ),
    "Stability F1": (None, "figure_e_hull_violin", None),
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


class ConvexHullAlexandriaApp(BaseApp):
    """Alexandria convex-hull benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        # Each violin figure holds all the models, so load it once for all cells
        violins = {}
        for _, violin_stem, _ in PLOTS.values():
            violin_path = DATA_PATH / f"{violin_stem}.json"
            if violin_stem not in violins and violin_path.exists():
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
                        components.append(Graph(figure=read_json(table_path)))
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
            Output(HULL_SYSTEM_ID, "options"),
            Output(HULL_SYSTEM_ID, "value"),
            Input(HULL_MODEL_ID, "value"),
        )
        def update_systems(model):
            """
            List the systems of the selected model, worst first.

            Parameters
            ----------
            model : str | None
                Selected model.

            Returns
            -------
            tuple[list[dict[str, str]], str]
                Dropdown options, and the system with the largest MAE.
            """
            if not model:
                raise PreventUpdate
            options = system_options(model)
            return options, options[0]["value"] if options else None

        @callback(
            Output(HULL_FIGURE_ID, "children"),
            Input(HULL_MODEL_ID, "value"),
            Input(HULL_SYSTEM_ID, "value"),
        )
        def show_hulls(model, system):
            """
            Draw the PBE and model hulls of the selected system.

            Parameters
            ----------
            model : str | None
                Selected model.
            system : str | None
                Selected system.

            Returns
            -------
            Graph | P
                Side-by-side hull figure, or a message if the model has no
                successful system.
            """
            if not model:
                raise PreventUpdate
            if not system:
                return P(f"No successful systems for {model}.")
            path = DATA_PATH / model / "hulls" / f"{system}.json"
            if not path.exists():
                # System of the previous model, before the system list is updated
                raise PreventUpdate
            return Graph(figure=read_json(path))


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
            "Below, compare the PBE and model hulls of any system."
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
            Div(id=HULL_FIGURE_ID),
        ],
        info_path=INFO_PATH,
    )
