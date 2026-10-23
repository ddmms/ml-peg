"""Run Alexandria convex-hull benchmark app."""

from __future__ import annotations

from functools import lru_cache
import json

from dash import Input, Output, callback
from dash.dcc import Dropdown, Graph
from dash.exceptions import PreventUpdate
from dash.html import H3, Div
import numpy as np
from plotly.io import read_json
from plotly.subplots import make_subplots

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

# Colour scale limit for the energy above the hull of unstable phases (meV/atom)
E_HULL_COLOUR_MAX = 200


@lru_cache(maxsize=8)
def load_hull_data(model: str) -> dict[str, dict[str, list]]:
    """
    Load the data to draw the reference and model hulls of each system.

    Parameters
    ----------
    model
        Name of the model.

    Returns
    -------
    dict[str, dict[str, list]]
        For each system, one list per field, as written by the analysis.
    """
    with open(DATA_PATH / model / "hull_data.json", encoding="utf8") as file:
        return json.load(file)


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
    errors = {
        system: np.mean(np.abs(np.subtract(data["e_hull_pred"], data["e_hull_ref"])))
        for system, data in load_hull_data(model).items()
    }
    options = []
    for system, mae in sorted(errors.items(), key=lambda item: -item[1]):
        first, second = system.split("-")
        options.append(
            {
                "label": f"{system} (E_hull MAE {mae:.0f} meV/atom)",
                "value": system,
                "search": f"{system} {second}-{first}",
            }
        )
    return options


def hull_figure(data: dict[str, list], system: str, model: str):
    """
    Draw the reference and model hulls of a system side by side.

    Parameters
    ----------
    data
        Data of the system, one list per field.
    system
        System, such as ``Cu-Zn``.
    model
        Name of the model.

    Returns
    -------
    plotly.graph_objects.Figure
        Formation energy against composition, with the hull, for PBE and the model.
    """
    first, second = system.split("-")
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=("PBE (Alexandria)", model),
    )
    x = np.array(data["x"])
    hover = [
        f"{formula} ({mat_id})"
        for formula, mat_id in zip(data["formula"], data["mat_id"], strict=True)
    ]
    for col, source in ((1, "ref"), (2, "pred")):
        e_form = np.array(data[f"e_form_{source}"])
        e_hull = np.array(data[f"e_hull_{source}"])
        stable = np.array(data[f"stable_{source}"])
        order = np.argsort(x[stable])
        fig.add_scatter(
            x=x[stable][order],
            y=e_form[stable][order],
            mode="lines",
            line={"color": "black"},
            hoverinfo="skip",
            showlegend=False,
            row=1,
            col=col,
        )
        texts = [
            f"{text}<br>E_form = {ef:.0f} meV/atom<br>E_hull = {eh:.0f} meV/atom"
            for text, ef, eh in zip(hover, e_form, e_hull, strict=True)
        ]
        fig.add_scatter(
            x=x[~stable],
            y=e_form[~stable],
            mode="markers",
            marker={
                "color": e_hull[~stable],
                "colorscale": "Viridis",
                "cmin": 0,
                "cmax": E_HULL_COLOUR_MAX,
                "size": 8,
                "colorbar": {"title": "E_hull (meV/atom)"} if col == 1 else None,
                "showscale": col == 1,
            },
            text=np.array(texts)[~stable],
            hoverinfo="text",
            name="Unstable",
            showlegend=col == 1,
            row=1,
            col=col,
        )
        fig.add_scatter(
            x=x[stable],
            y=e_form[stable],
            mode="markers",
            marker={"color": "black", "size": 10},
            text=np.array(texts)[stable],
            hoverinfo="text",
            name="Stable",
            showlegend=col == 1,
            row=1,
            col=col,
        )
    # Show the phases that are stable in PBE on the model hull too
    stable_ref = np.array(data["stable_ref"])
    fig.add_scatter(
        x=x[stable_ref],
        y=np.array(data["e_form_pred"])[stable_ref],
        mode="markers",
        marker={
            "color": "rgba(0,0,0,0)",
            "size": 16,
            "line": {"color": "#d62728", "width": 2},
        },
        hoverinfo="skip",
        name="Stable in PBE",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"x in {first}(1-x){second}x", range=[-0.03, 1.03])
    fig.update_yaxes(title_text="Formation energy (meV/atom)", col=1)
    fig.update_layout(
        title=f"{system}: PBE and {model} convex hulls",
        height=500,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


class ConvexHullAlexandriaApp(BaseApp):
    """Alexandria convex-hull benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
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
                violin_path = DATA_PATH / f"{violin_stem}.json"
                if violin_path.exists():
                    components.append(Graph(figure=read_json(violin_path)))
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
            Graph
                Side-by-side hull figure.
            """
            if not model or not system:
                raise PreventUpdate
            data = load_hull_data(model).get(system)
            if data is None:
                raise PreventUpdate
            return Graph(figure=hull_figure(data, system, model))


def get_app() -> ConvexHullAlexandriaApp:
    """
    Get Alexandria convex-hull benchmark app layout and callback registration.

    Returns
    -------
    ConvexHullAlexandriaApp
        Benchmark layout and callback registration.
    """
    models = [m for m in MODELS if (DATA_PATH / m / "hull_data.json").exists()]
    return ConvexHullAlexandriaApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy above the convex hull of 24,726 binary crystals from the "
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
                        style={"width": "300px"},
                    ),
                    Dropdown(
                        id=HULL_SYSTEM_ID,
                        clearable=False,
                        searchable=True,
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
